"""Canonical RAG runtime shared by Web serving and evaluation.

Composes validated research modules without forking inference semantics.
Legacy ``RagService`` (dense Top-5 + extractive) is not used here.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict
from typing import Any

from sqlalchemy.orm import Session

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.db.models import Document, DocumentVersion
from rag_workbench.experiments.atomic_requirement_contract_v1 import (
    FrozenQuestionPlan,
    assemble_frozen_plan,
    decompose_question,
    deterministic_support_complete,
    extract_direct_support_mappings,
    validate_verifier_result,
)
from rag_workbench.experiments.atomic_requirement_contract_v1.contract import FrozenValidation
from rag_workbench.experiments.atomic_requirement_contract_v1.verifier import (
    FrozenEvidenceVerifier,
    requirement_scoped_evidence_packets,
)
from rag_workbench.experiments.combined_targeted_real_api_validation_v1.runtime import (
    deterministic_requirement_map,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.router import (
    RoutingFeatures,
    SelectiveRiskRouter,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.versioning import (
    DeterministicVersionResolver,
    VersionCandidate,
    extract_resolved_token_support,
)
from rag_workbench.experiments.universal_requirement_completeness_v1 import (
    UniversalEvidenceChunk,
)
from rag_workbench.runtime.config import ProductionRagConfig
from rag_workbench.runtime.guards import RequestGuardError, RequestModelGuard
from rag_workbench.runtime.retrieval import HybridEvidencePool, retrieve_evidence_pool
from rag_workbench.runtime.types import (
    CanonicalQueryResult,
    CitationView,
    RequestTrace,
    RequirementView,
    RouteName,
)
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.security.permissions import Principal


def _semantic_document_ids(question: str, top15: list[dict[str, Any]]) -> set[str]:
    def tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.casefold()))

    question_tokens = tokenize(question)
    by_document: dict[str, list[dict[str, Any]]] = {}
    for item in top15:
        by_document.setdefault(item["document_id"], []).append(item)
    scored: list[tuple[int, str]] = []
    for document_id, rows in by_document.items():
        tokens = tokenize(document_id) - {"policy"}
        for row in rows:
            tokens |= tokenize(str(row.get("title") or ""))
            tokens |= tokenize(str(row.get("section") or ""))
            tokens |= tokenize(str(row.get("text") or ""))
        score = len(tokens & question_tokens)
        if score:
            scored.append((score, document_id))
    if not scored:
        return set(by_document)
    maximum = max(score for score, _ in scored)
    return {document_id for score, document_id in scored if score == maximum}


def _candidates_from_top15(
    session: Session,
    question: str,
    top15: list[dict[str, Any]],
    *,
    tenant_id: str,
) -> tuple[VersionCandidate, ...]:
    document_ids = _semantic_document_ids(question, top15)
    candidates: list[VersionCandidate] = []
    seen: set[str] = set()
    for item in top15:
        if item["document_id"] not in document_ids or item["document_version_id"] in seen:
            continue
        version = session.get(DocumentVersion, item["document_version_id"])
        document = session.get(Document, version.document_fk) if version else None
        if not version or not document or document.tenant_id != tenant_id:
            continue
        seen.add(version.id)
        candidates.append(
            VersionCandidate(
                item["chunk_id"],
                document.document_id,
                version.id,
                version.version,
                item["text"],
                document.tenant_id,
                (document.metadata_ or {}).get("region"),
                version.is_active,
                version.effective_at,
                authorized=True,
            )
        )
    return tuple(candidates)


def _gate_evidence(rows: list[dict[str, Any]]) -> tuple[GateEvidence, ...]:
    return tuple(
        GateEvidence(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            document_version_id=item.get("document_version_id", ""),
            version=item.get("version", ""),
            text=item["text"],
        )
        for item in rows
    )


def _universal_chunks(rows: list[dict[str, Any]]) -> tuple[UniversalEvidenceChunk, ...]:
    return tuple(
        UniversalEvidenceChunk(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            text=item["text"],
            version_id=item.get("document_version_id"),
        )
        for item in rows
    )


def _route_name(raw: str) -> RouteName:
    mapping = {
        "DETERMINISTIC": "deterministic",
        "LUNA": "luna",
        "SOL": "sol",
        "SAFE_ABSTAIN": "abstain",
    }
    return mapping.get(raw, "abstain")  # type: ignore[return-value]


def _citations_from_rows(
    chunk_ids: tuple[str, ...], rows: list[dict[str, Any]]
) -> tuple[CitationView, ...]:
    by_id = {row["chunk_id"]: row for row in rows}
    citations: list[CitationView] = []
    for chunk_id in chunk_ids:
        row = by_id.get(chunk_id)
        if row is None:
            continue
        citations.append(
            CitationView(
                document_id=row["document_id"],
                chunk_id=chunk_id,
                source=str(row.get("source") or ""),
                title=str(row.get("title") or ""),
                version=str(row.get("version") or ""),
                page=row.get("page"),
                section=row.get("section"),
            )
        )
    return tuple(citations)


class CanonicalRagRuntime:
    """Single inference implementation for Web serving and evaluation parity."""

    def __init__(
        self,
        session: Session,
        *,
        dense: Any,
        bm25: Any,
        reranker: Any,
        config: ProductionRagConfig,
        luna_verifier: FrozenEvidenceVerifier | None = None,
        sol_verifier: FrozenEvidenceVerifier | None = None,
    ) -> None:
        self.session = session
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self.config = config
        self.luna_verifier = luna_verifier
        self.sol_verifier = sol_verifier
        self.config.validate_for_serving()

    def query(
        self,
        question: str,
        principal: Principal,
        *,
        include_debug: bool = False,
        region: str | None = None,
    ) -> CanonicalQueryResult:
        del region  # Region is enforced via document metadata on candidates.
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        guard = RequestModelGuard(
            max_luna_calls=self.config.max_luna_calls_per_request,
            max_sol_calls=self.config.max_sol_calls_per_request,
        )
        trace = RequestTrace(request_id=request_id)

        if not question.strip():
            return self._finish(
                request_id,
                status="abstain",
                answer=None,
                citations=(),
                requirements=(),
                route="abstain",
                trace=trace,
                started=started,
                error_class="EMPTY_QUERY",
            )

        if is_question_injection_v2(question):
            plan = decompose_question(question)
            return self._finish(
                request_id,
                status="abstain",
                answer=None,
                citations=(),
                requirements=tuple(
                    RequirementView(item.requirement_id, item.requirement_text, "abstained")
                    for item in plan.requirements
                ),
                route="abstain",
                trace=trace,
                started=started,
                error_class="PROMPT_INJECTION",
                question_plan=plan.as_dict(),
            )

        try:
            pool = retrieve_evidence_pool(
                dense=self.dense,
                bm25=self.bm25,
                reranker=self.reranker,
                question=question,
                principal=principal,
                config=self.config,
            )
        except Exception as exc:
            return self._finish(
                request_id,
                status="unavailable",
                answer=None,
                citations=(),
                requirements=(),
                route="abstain",
                trace=trace,
                started=started,
                error_class=f"RETRIEVAL_UNAVAILABLE:{type(exc).__name__}",
            )

        trace.dense_candidate_count = len(pool.dense)
        trace.bm25_candidate_count = len(pool.bm25)
        trace.rrf_candidate_count = len(pool.fused)
        trace.ce_retained_count = len(pool.top15)
        trace.embedding_cache_hit = pool.embedding_cache_hit

        plan = decompose_question(question)
        trace.requirements_count = len(plan.requirements)
        trace.question_plan_hash = plan.question_plan_hash

        analysis = self._analyze(question, principal, plan, pool)
        trace.version_resolution_status = (analysis.get("version_resolution") or {}).get("status")
        route_raw = analysis["initial_route"]["route"]
        route = _route_name(route_raw)
        evidence_rows: list[dict[str, Any]] = analysis["evidence_rows"]
        evidence = _gate_evidence(evidence_rows)
        authorized_ids = frozenset(item.chunk_id for item in evidence)
        selected = {
            key: frozenset(value)
            for key, value in analysis["selected_versions_by_document"].items()
        }
        mappings = analysis["mapping_objects"]
        trace.validated_mappings_count = len(mappings)

        answer: str | None = None
        citations: tuple[CitationView, ...] = ()
        requirements: tuple[RequirementView, ...] = ()
        error_class: str | None = None
        validation = FrozenValidation(False, "NO_VALIDATION", plan.question_plan_hash)

        if route == "abstain":
            error_class = analysis["initial_route"].get("reason")
        elif route == "deterministic":
            support = analysis["support_decision"]
            if support.complete and support.validation.valid:
                validation = support.validation
                assembly = assemble_frozen_plan(
                    plan,
                    validation,
                    _universal_chunks(evidence_rows),
                    authorized_chunk_ids=authorized_ids,
                    selected_versions_by_document=selected,
                )
                if assembly.status == "answered" and assembly.answer:
                    answer = assembly.answer
                    citations = _citations_from_rows(assembly.citations, evidence_rows)
                    requirements = tuple(
                        RequirementView(
                            item.requirement_id,
                            item.requirement_text,
                            "supported",
                            (item.chunk_id,),
                        )
                        for item in validation.requirements
                    )
                else:
                    route = "abstain"
                    error_class = assembly.failure_code or "ASSEMBLER_REJECTED"
            else:
                route = "abstain"
                error_class = support.failure_code or "DETERMINISTIC_INCOMPLETE"
        else:
            verifier = self.luna_verifier if route == "luna" else self.sol_verifier
            model_kind = "LUNA" if route == "luna" else "SOL"
            if verifier is None:
                route = "abstain"
                error_class = f"{model_kind}_VERIFIER_UNAVAILABLE"
            else:
                try:
                    guard.authorize(model_kind)  # type: ignore[arg-type]
                    if model_kind == "LUNA":
                        trace.luna_calls += 1
                    else:
                        trace.sol_calls += 1
                    packets = requirement_scoped_evidence_packets(
                        plan,
                        evidence,
                        validated_mappings=mappings,
                        authorized_chunk_ids=authorized_ids,
                        selected_versions_by_document=selected,
                    )
                    packet_evidence = tuple(
                        item for packet in packets.values() for item in packet
                    ) or evidence
                    result = verifier.evaluate(
                        plan,
                        packet_evidence,
                        query_id=request_id,
                        arm="canonical_runtime",
                        routing_reason=analysis["initial_route"]["reason"],
                        validated_mappings=mappings,
                        authorized_chunk_ids=authorized_ids,
                        selected_versions_by_document=selected,
                    )
                    validation = validate_verifier_result(
                        plan,
                        result,
                        evidence,
                        authorized_chunk_ids=authorized_ids,
                        selected_versions_by_document=selected,
                    )
                    if validation.valid and validation.canonical_decision == "GO":
                        assembly = assemble_frozen_plan(
                            plan,
                            validation,
                            _universal_chunks(evidence_rows),
                            authorized_chunk_ids=authorized_ids,
                            selected_versions_by_document=selected,
                        )
                        if assembly.status == "answered" and assembly.answer:
                            answer = assembly.answer
                            citations = _citations_from_rows(assembly.citations, evidence_rows)
                            requirements = tuple(
                                RequirementView(
                                    item.requirement_id,
                                    item.requirement_text,
                                    "supported",
                                    (item.chunk_id,),
                                )
                                for item in validation.requirements
                            )
                        else:
                            route = "abstain"
                            error_class = assembly.failure_code or "ASSEMBLER_REJECTED"
                    elif (
                        route == "luna"
                        and self.sol_verifier is not None
                        and validation.canonical_decision in {None, "UNCERTAIN", "ABSTAIN"}
                        and analysis.get("suspicious_luna_abstention")
                    ):
                        # Selective Sol escalation for suspicious Luna abstention.
                        try:
                            guard.authorize("SOL")
                            trace.sol_calls += 1
                            route = "sol"
                            sol_result = self.sol_verifier.evaluate(
                                plan,
                                evidence,
                                query_id=request_id,
                                arm="canonical_runtime",
                                routing_reason="suspicious_luna_abstention",
                                validated_mappings=mappings,
                                authorized_chunk_ids=authorized_ids,
                                selected_versions_by_document=selected,
                            )
                            validation = validate_verifier_result(
                                plan,
                                sol_result,
                                evidence,
                                authorized_chunk_ids=authorized_ids,
                                selected_versions_by_document=selected,
                            )
                            if validation.valid and validation.canonical_decision == "GO":
                                assembly = assemble_frozen_plan(
                                    plan,
                                    validation,
                                    _universal_chunks(evidence_rows),
                                    authorized_chunk_ids=authorized_ids,
                                    selected_versions_by_document=selected,
                                )
                                if assembly.status == "answered" and assembly.answer:
                                    answer = assembly.answer
                                    citations = _citations_from_rows(
                                        assembly.citations, evidence_rows
                                    )
                                    requirements = tuple(
                                        RequirementView(
                                            item.requirement_id,
                                            item.requirement_text,
                                            "supported",
                                            (item.chunk_id,),
                                        )
                                        for item in validation.requirements
                                    )
                                else:
                                    route = "abstain"
                                    error_class = assembly.failure_code or "ASSEMBLER_REJECTED"
                            else:
                                route = "abstain"
                                error_class = validation.failure_code or "SOL_NOT_GO"
                        except RequestGuardError as exc:
                            route = "abstain"
                            error_class = exc.code
                    else:
                        route = "abstain"
                        error_class = validation.failure_code or "VERIFIER_NOT_GO"
                except RequestGuardError as exc:
                    route = "abstain"
                    error_class = exc.code
                except Exception as exc:
                    route = "abstain"
                    error_class = f"VERIFIER_ERROR:{type(exc).__name__}"

        status = "answer" if answer else "abstain"
        if not requirements and plan.requirements:
            requirements = tuple(
                RequirementView(
                    item.requirement_id,
                    item.requirement_text,
                    "supported" if answer else "abstained",
                )
                for item in plan.requirements
            )

        return self._finish(
            request_id,
            status=status,  # type: ignore[arg-type]
            answer=answer,
            citations=citations,
            requirements=requirements,
            route=route,
            trace=trace,
            started=started,
            error_class=error_class,
            question_plan=plan.as_dict(),
            retrieval_results=tuple(pool.top15) if include_debug else (),
            debug_extra={
                "initial_route": analysis["initial_route"],
                "version_resolution": analysis.get("version_resolution"),
                "deterministic_support": analysis.get("deterministic_support"),
                "dense_ids": [item.chunk_id for item in pool.dense],
                "bm25_ids": [item.chunk_id for item in pool.bm25],
                "rrf_ranks": pool.rrf_rank,
            }
            if include_debug
            else None,
        )

    def _analyze(
        self,
        question: str,
        principal: Principal,
        plan: FrozenQuestionPlan,
        pool: HybridEvidencePool,
    ) -> dict[str, Any]:
        top15 = list(pool.top15)
        temporal_scope = pool.temporal_scope
        candidates = _candidates_from_top15(
            self.session, question, top15, tenant_id=principal.tenant_id
        )
        resolution = (
            DeterministicVersionResolver().resolve(
                question,
                candidates,
                tenant_id=principal.tenant_id,
                temporal_scope=temporal_scope,
            )
            if candidates
            else None
        )
        selected_by_document = (
            resolution.selected_version_ids_by_document if resolution else {}
        )
        evidence_rows = [
            row
            for row in top15
            if row["document_id"] not in selected_by_document
            or row.get("document_version_id") in selected_by_document[row["document_id"]]
        ]
        version_support = (
            extract_resolved_token_support(question, resolution) if resolution else ()
        )
        deterministic = version_support or deterministic_requirement_map(
            question, _universal_chunks(evidence_rows)
        )
        evidence = _gate_evidence(evidence_rows)
        direct = extract_direct_support_mappings(plan, evidence)
        covered_ids = {item.requirement_id for item in direct}
        legacy_fill = []
        for item in deterministic:
            requirement_id = getattr(item, "requirement_id", None)
            if requirement_id is None or requirement_id in covered_ids:
                continue
            legacy_fill.append(item)
            covered_ids.add(requirement_id)
        combined = tuple(direct) + tuple(legacy_fill)
        unresolved = bool(
            temporal_scope.temporal_mode != "UNSPECIFIED_CURRENT_DEFAULT"
            and (
                resolution is None
                or resolution.status not in {"VERSION_RESOLVED", "VERSION_SET_RESOLVED"}
            )
        )
        support = deterministic_support_complete(
            plan,
            combined,
            evidence,
            authorized_chunk_ids=frozenset(item.chunk_id for item in evidence),
            acl_valid_chunk_ids=frozenset(item.chunk_id for item in evidence),
            tenant_valid_chunk_ids=frozenset(item.chunk_id for item in evidence),
            region_valid_chunk_ids=frozenset(item.chunk_id for item in evidence),
            selected_versions_by_document=selected_by_document,
            unresolved_version_conflict=unresolved,
            security_precheck_failed=False,
        )
        deterministic_complete = support.complete
        version_sensitive = temporal_scope.temporal_mode != "UNSPECIFIED_CURRENT_DEFAULT"
        version_resolved = bool(
            not version_sensitive
            or (
                resolution
                and resolution.status in {"VERSION_RESOLVED", "VERSION_SET_RESOLVED"}
            )
        )
        route = SelectiveRiskRouter().initial(
            RoutingFeatures(
                security_precheck_requires_abstention=False,
                deterministic_support_complete=deterministic_complete,
                conflicting_evidence=False,
                version_sensitive=version_sensitive,
                version_resolved=version_resolved,
                multiple_active_versions=False,
                version_metadata_complete=all(
                    item.is_active is not None for item in candidates
                ),
                top15_evidence_complete=deterministic_complete,
                required_fact_count=len(plan.requirements),
            )
        )
        suspicious = bool(
            deterministic_complete
            or version_resolved
            or bool(evidence_rows)
        )
        return {
            "initial_route": asdict(route),
            "evidence_rows": evidence_rows,
            "selected_versions_by_document": {
                document_id: sorted(ids)
                for document_id, ids in selected_by_document.items()
            },
            "version_resolution": None
            if resolution is None
            else {
                "status": resolution.status,
                "reason": resolution.reason,
                "selected_version_ids": list(resolution.selected_version_ids),
                "selections_by_document": {
                    key: list(value) for key, value in resolution.selections_by_document.items()
                },
            },
            "deterministic_support": {
                "complete": support.complete,
                "failure_code": support.failure_code,
            },
            "support_decision": support,
            "mapping_objects": combined if support.complete else combined,
            "suspicious_luna_abstention": suspicious,
        }

    def _finish(
        self,
        request_id: str,
        *,
        status: str,
        answer: str | None,
        citations: tuple[CitationView, ...],
        requirements: tuple[RequirementView, ...],
        route: RouteName,
        trace: RequestTrace,
        started: float,
        error_class: str | None = None,
        question_plan: dict[str, Any] | None = None,
        retrieval_results: tuple[dict[str, Any], ...] = (),
        debug_extra: dict[str, Any] | None = None,
    ) -> CanonicalQueryResult:
        trace.route = route
        trace.final_status = status
        trace.error_class = error_class
        trace.latency_ms = (time.perf_counter() - started) * 1000
        if debug_extra:
            trace.extra.update(debug_extra)
        return CanonicalQueryResult(
            request_id=request_id,
            status=status,  # type: ignore[arg-type]
            answer=answer,
            citations=citations,
            requirements=requirements,
            route=route,
            retrieval_results=retrieval_results,
            question_plan=question_plan,
            trace=trace.as_dict(),
            run_id=request_id,
            error_class=error_class,
        )


def build_canonical_runtime(
    session: Session,
    *,
    config: ProductionRagConfig,
    embedding_provider: Any,
    reranker: Any | None = None,
    luna_verifier: FrozenEvidenceVerifier | None = None,
    sol_verifier: FrozenEvidenceVerifier | None = None,
    index_identity: str | None = None,
) -> CanonicalRagRuntime:
    """Factory used by the API and evaluation harnesses."""
    from rag_workbench.reranking.cross_encoder import CrossEncoderReranker
    from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
    from rag_workbench.retrieval.retriever import Retriever

    identity = index_identity or config.index_identity
    dense = Retriever(session, embedding_provider, identity)
    bm25 = BM25Retriever(
        session,
        index_identity=dense.index_identity,
        embedding_provider=embedding_provider.provider_name,
        embedding_model=embedding_provider.model_name,
        embedding_version=embedding_provider.version,
        embedding_dimension=embedding_provider.dimension,
        config=BM25Config(),
    )
    if reranker is None:
        if config.allow_identity_reranker:
            from rag_workbench.runtime.identity_reranker import IdentityReranker

            reranker = IdentityReranker()
        elif config.require_cross_encoder:
            reranker = CrossEncoderReranker(device="cpu")
        else:
            raise RuntimeError("CROSS_ENCODER_REQUIRED")
    return CanonicalRagRuntime(
        session,
        dense=dense,
        bm25=bm25,
        reranker=reranker,
        config=config,
        luna_verifier=luna_verifier,
        sol_verifier=sol_verifier,
    )
