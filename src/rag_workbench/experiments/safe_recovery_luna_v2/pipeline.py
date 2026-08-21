# ruff: noqa: E501
"""Shared retrieval, routing, and local extractive generation for Luna V2."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import (
    AnswerabilityReason,
    AnswerabilityResult,
    GateEvidence,
)
from rag_workbench.answerability.cache import CachedAnswerabilityGate
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    OpenAICompatibleAnswerabilityGate,
)
from rag_workbench.answerability.planning import ExternalJudgeCallLimitGate
from rag_workbench.answerability.validation import validate_gate_result_with_error
from rag_workbench.config import Settings
from rag_workbench.db.models import Chunk, Document, DocumentPermission, DocumentVersion
from rag_workbench.evaluation.citation_authorization import (
    CitationEvidenceIdentity,
    authorize_citation_evidence,
)
from rag_workbench.evaluation.final_e2e_scorer_v2 import ScorerInput, score_case
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    DENSE_DEPTH,
    FINAL_TOP_K,
    RRF_K,
    UNION_LIMIT,
)
from rag_workbench.experiments.reranker_e2e_benchmark import FixedResultRetriever, _result
from rag_workbench.experiments.safe_recovery_luna_v2.identities import (
    GENERATOR_MODEL,
    INDEX_IDENTITY,
    LUNA_MODEL,
    PIPELINE_VERSION,
    RESULT_CACHE_DIR,
    SOL_MODEL,
    sha256_text,
)
from rag_workbench.experiments.safe_recovery_luna_v2.pricing import estimate_cost_usd
from rag_workbench.experiments.safe_recovery_luna_v2.validation import (
    deterministic_prechecks,
    validate_luna_go,
)
from rag_workbench.experiments.safe_recovery_luna_v2.verifier import (
    LUNA_SYSTEM_PROMPT,
    LunaCallUsage,
    LunaEvidenceVerifier,
    LunaVerifierResult,
    luna_schema,
    result_cache_key,
)
from rag_workbench.experiments.v2_document_diversity import ranking_candidate
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.generation.generator import RagService
from rag_workbench.providers.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from rag_workbench.providers.llm.extractive import EXTRACTIVE_V2_MODEL, ExtractiveGenerationProvider
from rag_workbench.reranking import CrossEncoderReranker
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.filters import RetrievalFilters
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.retriever import RetrievalTiming, Retriever
from rag_workbench.retrieval.temporal import (
    TemporalScopePlan,
    plan_temporal_scope,
)
from rag_workbench.safety.answerability_constraint_guard_v2 import (
    should_abstain_due_to_answerability_constraint as constraint_guard_v2,
)
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.security.permissions import Principal


class _FixedGate:
    provider_name = "openai"
    model_name = SOL_MODEL
    gate_version = "1"
    prompt_version = EVIDENCE_GATE_PROMPT_VERSION

    def __init__(self, result: AnswerabilityResult, timing: Any) -> None:
        self.result = result
        self.last_timing = timing

    def evaluate(self, question: str, chunks: tuple[GateEvidence, ...]) -> AnswerabilityResult:
        del question, chunks
        return self.result


@dataclass
class ArmConfig:
    name: str
    architecture: str
    luna_first: bool
    sol_recovery_after_no: bool
    prompt_cache: bool
    minimal_input: bool
    result_cache: bool
    adaptive_k: bool
    allow_result_cache_for_authoritative: bool = False


@dataclass
class PipelineRuntime:
    session: Session
    settings: Settings
    dense: Retriever
    bm25: BM25Retriever
    reranker: CrossEncoderReranker
    embedding_provider: OpenAICompatibleEmbeddingProvider
    luna: LunaEvidenceVerifier
    sol: CachedAnswerabilityGate
    ledger: list[dict[str, Any]] = field(default_factory=list)
    result_cache_hits: int = 0
    result_cache_misses: int = 0
    api_calls_avoided: int = 0
    dollars_avoided: float = 0.0
    cases_resolved_without_llm: int = 0
    final_generator_openai_calls: int = 0
    luna_calls: int = 0
    sol_calls: int = 0
    sol_escalations: int = 0
    retry_cost_usd: float = 0.0

    def write_ledger_row(self, usage: LunaCallUsage | None, extra: dict[str, Any] | None = None) -> None:
        if usage is None:
            return
        row = usage.as_dict()
        if extra:
            row.update(extra)
        self.ledger.append(row)


def api_key_from_settings(settings: Settings) -> str:
    return settings.effective_judge_api_key or settings.openai_api_key or ""


def build_runtime(session: Session, settings: Settings, *, prompt_cache: bool) -> PipelineRuntime:
    key = api_key_from_settings(settings)
    if not key:
        raise RuntimeError("OPENAI_API_KEY_OR_JUDGE_API_KEY_REQUIRED")
    embedding_provider = OpenAICompatibleEmbeddingProvider(
        api_key=settings.embedding_api_key or key,
        model="text-embedding-3-small",
        dimension=64,
        base_url=settings.embedding_base_url,
        version="1",
        provider_name="openai-compatible",
    )
    dense = Retriever(session, embedding_provider, INDEX_IDENTITY)
    bm25 = BM25Retriever(
        session,
        index_identity=INDEX_IDENTITY,
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_version="1",
        embedding_dimension=64,
        config=BM25Config(),
    )
    luna = LunaEvidenceVerifier(
        api_key=key,
        base_url=settings.judge_base_url or settings.openai_base_url,
        prompt_cache=prompt_cache,
    )
    sol_delegate = OpenAICompatibleAnswerabilityGate(
        api_key=key,
        model=SOL_MODEL,
        base_url=settings.judge_base_url or settings.openai_base_url,
        gate_version="1",
        prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
        provider_name="openai",
    )
    sol = CachedAnswerabilityGate(session, ExternalJudgeCallLimitGate(sol_delegate, 200000))
    return PipelineRuntime(
        session=session,
        settings=settings,
        dense=dense,
        bm25=bm25,
        reranker=CrossEncoderReranker(device="cpu"),
        embedding_provider=embedding_provider,
        luna=luna,
        sol=sol,
    )


def _embed_with_retry(dense: Retriever, question: str, attempts: int = 3) -> Any:
    last_error: Exception | None = None
    for i in range(attempts):
        try:
            return dense.query_embedding_cache.get_or_embed(question)
        except Exception as exc:
            last_error = exc
            if i + 1 < attempts:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError("EMBEDDING_PROVIDER_BLOCKED") from last_error


def retrieve_trace(
    runtime: PipelineRuntime, question: str, principal: Principal
) -> dict[str, Any]:
    temporal_scope = plan_temporal_scope(question)
    filters = RetrievalFilters(temporal_scope=temporal_scope)
    emb = _embed_with_retry(runtime.dense, question)
    dense = runtime.dense.retrieve_with_embedding(
        emb, top_k=DENSE_DEPTH, score_threshold=0.28, filters=filters, principal=principal
    )
    bm25 = runtime.bm25.retrieve(
        question, top_k=BM25_DEPTH, filters=filters, principal=principal
    )
    union = reciprocal_rank_fusion(dense, bm25, top_k=10_000, rrf_k=RRF_K)[:UNION_LIMIT]
    reranked = runtime.reranker.rerank(question, union)
    ranked = [
        ranking_candidate(x) | {"rank": x.reranked_rank, "score": x.reranker_score}
        for x in reranked
    ]
    dense_rank = {item.chunk_id: i + 1 for i, item in enumerate(dense)}
    bm25_rank = {item.chunk_id: i + 1 for i, item in enumerate(bm25)}
    rrf_rank = {item.chunk_id: i + 1 for i, item in enumerate(union)}
    return {
        "top20": ranked[:20],
        "top10": ranked[:10],
        "top5": ranked[:FINAL_TOP_K],
        "union": [
            {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "dense_rank": dense_rank.get(item.chunk_id),
                "bm25_rank": bm25_rank.get(item.chunk_id),
                "rrf_rank": rrf_rank.get(item.chunk_id),
            }
            for item in union
        ],
        "dense_ids": [item.chunk_id for item in dense],
        "bm25_ids": [item.chunk_id for item in bm25],
        "embedding_cache_hit": emb.cache_hit,
        "embedding_external_calls": emb.external_calls,
        "temporal_scope": temporal_scope.as_dict(),
    }


def gate_evidence(topn: list[dict[str, Any]]) -> tuple[GateEvidence, ...]:
    return tuple(
        GateEvidence(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            document_version_id=item.get("document_version_id", ""),
            version=item.get("version", ""),
            text=item.get("text", ""),
            index_identity=INDEX_IDENTITY,
        )
        for item in topn
    )


def luna_to_answerability(result: LunaVerifierResult) -> AnswerabilityResult:
    ids = tuple(dict.fromkeys(r.chunk_id for r in result.requirements if r.chunk_id))
    return AnswerabilityResult(
        answerable=True,
        supporting_chunk_ids=ids,
        confidence=1.0,
        reason_code=AnswerabilityReason.SUFFICIENT_EVIDENCE,
    )


def local_extractive_answer(
    runtime: PipelineRuntime,
    *,
    question: str,
    principal: Principal,
    top5: list[dict[str, Any]],
    gate_result: AnswerabilityResult,
) -> tuple[str, str | None, list[str], int]:
    retrieved = [_result(x) for x in top5]
    rt = RetrievalTiming(
        query_embedding_latency_ms=0.0,
        embedding_cache_lookup_latency_ms=0.0,
        vector_search_latency_ms=0.0,
        acl_filter_latency_ms=0.0,
        query_embedding_cache_hit=True,
        external_embedding_calls=0,
    )
    rag = RagService(
        runtime.session,
        FixedResultRetriever(runtime.embedding_provider, retrieved, rt),
        ContextBuilder(1200),
        ExtractiveGenerationProvider(revision=EXTRACTIVE_V2_MODEL),
        _FixedGate(gate_result, runtime.sol.last_timing),
        supporting_context_only=True,
    )
    out = rag.query(question, principal, top_k=5, score_threshold=0.28)
    return out.status, out.answer, [c.chunk_id for c in out.citations], 0


def _load_result_cache(key: str) -> dict[str, Any] | None:
    path = RESULT_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _store_result_cache(key: str, payload: dict[str, Any]) -> None:
    RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_CACHE_DIR / f"{key}.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )


def _call_luna(
    runtime: PipelineRuntime,
    *,
    question: str,
    chunks: tuple[GateEvidence, ...],
    query_id: str,
    arm: ArmConfig,
    principal: Principal,
    k: int,
    escalation_reason: str | None = None,
) -> LunaVerifierResult:
    prompt_hash = sha256_text(LUNA_SYSTEM_PROMPT)
    schema_hash = sha256_text(json.dumps(luna_schema(), sort_keys=True))
    key = result_cache_key(
        model=LUNA_MODEL,
        question=question,
        chunks=chunks,
        prompt_hash=prompt_hash,
        schema_hash=schema_hash,
        principal_tenant=principal.tenant_id,
        pipeline_version=PIPELINE_VERSION,
        extra={"minimal": arm.minimal_input, "k": k, "prompt_cache": arm.prompt_cache},
    )
    if arm.result_cache:
        cached = _load_result_cache(key)
        if cached is not None:
            runtime.result_cache_hits += 1
            runtime.api_calls_avoided += 1
            runtime.dollars_avoided += float(cached.get("estimated_cost_usd") or 0.0)
            usage = LunaCallUsage(
                timestamp=cached.get("timestamp", ""),
                query_id=query_id,
                arm=arm.name,
                stage="luna_verifier",
                model=LUNA_MODEL,
                cache_hit=True,
                estimated_cost_usd=0.0,
                input_tokens=int(cached.get("input_tokens") or 0),
                output_tokens=int(cached.get("output_tokens") or 0),
                cached_input_tokens=int(cached.get("cached_input_tokens") or 0),
                chunk_count=len(chunks),
            )
            runtime.write_ledger_row(usage, {"result_cache_hit": True})
            return LunaVerifierResult.model_validate(cached["result"])
        runtime.result_cache_misses += 1
    result = runtime.luna.evaluate(
        question,
        chunks,
        query_id=query_id,
        arm=arm.name,
        minimal=arm.minimal_input,
        escalation_reason=escalation_reason,
    )
    runtime.luna_calls += 1
    usage = runtime.luna.last_usage
    if usage and usage.retry_count:
        runtime.retry_cost_usd += usage.estimated_cost_usd
    runtime.write_ledger_row(usage)
    if arm.result_cache and usage is not None:
        _store_result_cache(
            key,
            {
                "result": result.model_dump(),
                "estimated_cost_usd": usage.estimated_cost_usd,
                "timestamp": usage.timestamp,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
            },
        )
    return result


def _call_sol(
    runtime: PipelineRuntime,
    *,
    question: str,
    chunks: tuple[GateEvidence, ...],
    query_id: str,
    arm: ArmConfig,
    principal: Principal,
    escalation_reason: str,
) -> AnswerabilityResult:
    proposed = runtime.sol.evaluate(question, chunks)
    validated = validate_gate_result_with_error(
        proposed,
        chunks,
        session=runtime.session,
        principal=principal,
        temporal_scope=plan_temporal_scope(question),
    )
    timing = runtime.sol.last_timing
    external = int(timing.external_calls or 0)
    runtime.sol_calls += 1
    if escalation_reason != "reference_sol_always":
        runtime.sol_escalations += 1
    usage = LunaCallUsage(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        query_id=query_id,
        arm=arm.name,
        stage="sol_judge",
        model=SOL_MODEL,
        input_tokens=int(timing.prompt_tokens or 0),
        cached_input_tokens=int(timing.cached_prompt_tokens or 0),
        output_tokens=int(timing.completion_tokens or 0),
        total_tokens=int((timing.prompt_tokens or 0) + (timing.completion_tokens or 0)),
        latency_ms=float(timing.judge_latency_ms or 0.0),
        retry_count=int(timing.retry_count or 0),
        estimated_cost_usd=(
            0.0
            if timing.cache_hit
            else estimate_cost_usd(
                model=SOL_MODEL,
                input_tokens=int(timing.prompt_tokens or 0),
                output_tokens=int(timing.completion_tokens or 0),
                cached_input_tokens=int(timing.cached_prompt_tokens or 0),
            )
        ),
        escalation_reason=escalation_reason,
        cache_hit=bool(timing.cache_hit),
        chunk_count=len(chunks),
    )
    runtime.write_ledger_row(usage, {"sol_external_calls": external})
    return validated.result


def verify_and_answer(
    runtime: PipelineRuntime,
    *,
    case: Any,
    principal: Principal,
    trace: dict[str, Any],
    arm: ArmConfig,
) -> dict[str, Any]:
    started = time.perf_counter()
    top5 = trace["top5"]
    inj = is_question_injection_v2(case.question)
    if inj:
        return _pack(
            status="abstained",
            answer=None,
            citations=[],
            decision="ABSTAIN",
            reason="question_injection_guard_v2",
            started=started,
            extra={"question_injection_guard_triggered": True, "cases_resolved_without_llm": True},
        )

    pre = deterministic_prechecks(
        topk=top5,
        principal=principal,
        session=runtime.session,
        temporal_scope=plan_temporal_scope(case.question),
    )
    if pre.resolved_without_llm:
        runtime.cases_resolved_without_llm += 1
        return _pack(
            status="abstained",
            answer=None,
            citations=[],
            decision="ABSTAIN",
            reason=pre.code,
            started=started,
            extra={"deterministic_precheck": pre.code, "cases_resolved_without_llm": True},
        )

    def answer_from_go(gate: AnswerabilityResult, source: str) -> dict[str, Any]:
        ev_texts = [x["text"] for x in top5 if x["chunk_id"] in set(gate.supporting_chunk_ids)]
        if constraint_guard_v2(question=case.question, evidence_texts=ev_texts):
            return _pack(
                status="abstained",
                answer=None,
                citations=[],
                decision="ABSTAIN",
                reason="constraint_guard_v2",
                started=started,
                extra={"generation_source": source},
            )
        status, answer, citations, openai_calls = local_extractive_answer(
            runtime, question=case.question, principal=principal, top5=top5, gate_result=gate
        )
        runtime.final_generator_openai_calls += openai_calls
        if openai_calls != 0:
            raise RuntimeError("final_generator_openai_calls_nonzero")
        return _pack(
            status=status,
            answer=answer,
            citations=citations,
            decision="GO",
            reason=source,
            started=started,
            extra={
                "generation_source": source,
                "generator_model": GENERATOR_MODEL,
                "final_generator_openai_calls": openai_calls,
            },
        )

    def luna_on(chunks: tuple[GateEvidence, ...], k: int, reason: str | None = None) -> tuple[LunaVerifierResult, str | None]:
        try:
            luna_result = _call_luna(
                runtime,
                question=case.question,
                chunks=chunks,
                query_id=case.query_id,
                arm=arm,
                principal=principal,
                k=k,
                escalation_reason=reason,
            )
        except Exception:
            fail_closed = LunaVerifierResult(
                decision="ABSTAIN",
                requirements=[],
                all_supported=False,
                conflict=False,
                version_valid=True,
                region_valid=True,
            )
            return fail_closed, "LUNA_REQUEST_ERROR"
        if luna_result.decision != "GO":
            return luna_result, None
        fail = validate_luna_go(
            luna_result,
            chunks,
            session=runtime.session,
            principal=principal,
            temporal_scope=plan_temporal_scope(case.question),
        )
        return luna_result, fail

    if arm.architecture == "R":
        sol = _call_sol(
            runtime,
            question=case.question,
            chunks=gate_evidence(top5),
            query_id=case.query_id,
            arm=arm,
            principal=principal,
            escalation_reason="reference_sol_always",
        )
        if sol.answerable:
            return answer_from_go(sol, "sol_judge")
        return _pack(
            status="abstained",
            answer=None,
            citations=[],
            decision="ABSTAIN",
            reason="sol_no",
            started=started,
            extra={"judge_answerable": False},
        )

    if arm.architecture == "A":
        sol = _call_sol(
            runtime,
            question=case.question,
            chunks=gate_evidence(top5),
            query_id=case.query_id,
            arm=arm,
            principal=principal,
            escalation_reason="candidate_a_sol_front_gate",
        )
        if sol.answerable:
            return answer_from_go(sol, "sol_judge")
        luna_result, fail = luna_on(gate_evidence(top5), 5, "sol_no_recovery")
        if luna_result.decision == "GO" and fail is None:
            return answer_from_go(luna_to_answerability(luna_result), "luna_recovery")
        return _pack(
            status="abstained",
            answer=None,
            citations=[],
            decision="ABSTAIN" if luna_result.decision != "GO" else "SAFE_ABSTENTION",
            reason=fail or luna_result.decision,
            started=started,
            extra={"luna_decision": luna_result.decision, "deterministic_failure": fail},
        )

    # Luna-first (B / C / B1-B5)
    k_first = 3 if arm.adaptive_k else 5
    first_chunks = gate_evidence(top5[:k_first])
    luna_result, fail = luna_on(first_chunks, k_first)
    if arm.adaptive_k and luna_result.decision != "GO":
        luna_result, fail = luna_on(gate_evidence(top5), 5, "adaptive_expand_top5")
    if luna_result.decision == "GO":
        if fail is not None:
            return _pack(
                status="abstained",
                answer=None,
                citations=[],
                decision="SAFE_ABSTENTION",
                reason=fail,
                started=started,
                extra={"luna_decision": "GO", "deterministic_failure": fail},
            )
        return answer_from_go(luna_to_answerability(luna_result), "luna_go")
    if luna_result.decision == "ABSTAIN":
        return _pack(
            status="abstained",
            answer=None,
            citations=[],
            decision="ABSTAIN",
            reason="luna_abstain",
            started=started,
            extra={"luna_decision": "ABSTAIN", "sol_escalation": False},
        )
    sol = _call_sol(
        runtime,
        question=case.question,
        chunks=gate_evidence(top5),
        query_id=case.query_id,
        arm=arm,
        principal=principal,
        escalation_reason="luna_uncertain",
    )
    if sol.answerable:
        return answer_from_go(sol, "sol_escalation")
    return _pack(
        status="abstained",
        answer=None,
        citations=[],
        decision="ABSTAIN",
        reason="sol_escalation_no",
        started=started,
        extra={"luna_decision": "UNCERTAIN", "sol_escalation": True},
    )


def _pack(
    *,
    status: str,
    answer: str | None,
    citations: list[str],
    decision: str,
    reason: str | None,
    started: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "status": status,
        "answer": answer,
        "citations": citations,
        "decision": decision,
        "reason": reason,
        "latency_ms": (time.perf_counter() - started) * 1000,
    }
    if extra:
        row.update(extra)
    return row


def support_meta(session: Session, chunk_ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    if not chunk_ids:
        return {}, {}
    rows = session.execute(
        select(
            Chunk.id,
            Chunk.document_fk,
            Document.document_id,
            Document.tenant_id,
            Document.visibility,
            Document.metadata_.label("document_metadata"),
            DocumentVersion.id.label("document_version_id"),
            DocumentVersion.is_active,
            DocumentVersion.version,
            DocumentVersion.metadata_.label("version_metadata"),
            Chunk.metadata_.label("chunk_metadata"),
        )
        .join(Document, Document.id == Chunk.document_fk)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .where(Chunk.id.in_(chunk_ids))
    ).all()
    meta = {
        r.id: {
            "document_fk": r.document_fk,
            "chunk_id": r.id,
            "document_id": r.document_id,
            "document_version_id": r.document_version_id,
            "tenant_id": r.tenant_id,
            "visibility": r.visibility,
            "is_active": bool(r.is_active),
            "version": r.version,
            "region": (r.document_metadata or {}).get("region")
            or (r.version_metadata or {}).get("region")
            or (r.chunk_metadata or {}).get("region"),
        }
        for r in rows
    }
    perms_rows = (
        session.execute(
            select(DocumentPermission.document_fk, DocumentPermission.permission_group).where(
                DocumentPermission.document_fk.in_({v["document_fk"] for v in meta.values()})
            )
        ).all()
        if meta
        else []
    )
    perms: dict[str, set[str]] = {}
    for r in perms_rows:
        perms.setdefault(r.document_fk, set()).add(r.permission_group)
    return meta, perms


def is_authorized(
    meta: dict[str, Any],
    principal: Principal,
    groups: set[str],
    temporal_scope: Any | None = None,
    requested_region: str | None = None,
) -> bool:
    required = {
        "chunk_id", "document_id", "document_version_id", "version", "is_active",
        "tenant_id", "visibility", "region",
    }
    if not required <= set(meta):
        return False
    decision = authorize_citation_evidence(
        CitationEvidenceIdentity(
            document_id=meta["document_id"],
            document_version_id=meta["document_version_id"],
            version=meta["version"],
            is_active=bool(meta["is_active"]),
            chunk_id=meta["chunk_id"],
            tenant=meta["tenant_id"],
            region=meta["region"],
            visibility=meta["visibility"],
            permission_groups=tuple(sorted(groups)),
        ),
        principal=principal,
        temporal_scope=temporal_scope or TemporalScopePlan("UNSPECIFIED_CURRENT_DEFAULT"),
        requested_region=requested_region,
    )
    return decision.authorized


def score_output(
    *,
    arm: str,
    case: Any,
    out: dict[str, Any],
    top5: list[dict[str, Any]],
    session: Session,
    principal: Principal,
) -> dict[str, Any]:
    citation_ids = tuple(out.get("citations") or [])
    cited_doc_ids = tuple(x["document_id"] for x in top5 if x["chunk_id"] in set(citation_ids))
    text_map = {x["chunk_id"]: x.get("text", "") for x in top5}
    cited_texts = {cid: text_map.get(cid, "") for cid in citation_ids}
    meta, perms = support_meta(session, set(citation_ids))
    authorized = tuple(
        cid
        for cid in citation_ids
        if cid in meta
        and is_authorized(
            meta[cid],
            principal,
            perms.get(meta[cid]["document_fk"], set()),
            plan_temporal_scope(case.question),
        )
    )
    scored = score_case(
        ScorerInput(
            arm=arm,
            query_id=case.query_id,
            question=case.question,
            category=case.category,
            should_abstain=case.should_abstain,
            expected_answerable=case.expected_answerable,
            required_facts=tuple(case.required_facts),
            required_document_ids=tuple(case.required_document_ids),
            final_answer=out.get("answer"),
            final_answer_present=bool(out.get("answer")),
            citation_ids=citation_ids,
            cited_document_ids=cited_doc_ids,
            cited_chunk_texts=cited_texts,
            retrieved_top_k_ids=tuple(x["chunk_id"] for x in top5),
            authorized_citation_ids=authorized,
        )
    )
    payload = {
        "query_id": case.query_id,
        "category": case.category,
        "should_abstain": case.should_abstain,
        "expected_answerable": case.expected_answerable,
        "behavior": scored.behavior,
        "status": out.get("status"),
        "answer": out.get("answer"),
        "citations": list(citation_ids),
        "decision": out.get("decision"),
        "reason": out.get("reason"),
        "latency_ms": out.get("latency_ms"),
        "fact_completeness_pass": scored.fact_completeness_pass,
        "citation_validity_pass": scored.citation_validity_pass,
        "citation_correctness_pass": scored.citation_correctness_pass,
        "citation_completeness_pass": scored.citation_completeness_pass,
        "citation_validity_rate": scored.citation_validity_rate,
        "citation_correctness_rate": scored.citation_correctness_rate,
        "citation_completeness_rate": scored.citation_completeness_rate,
        "question_injection_guard_triggered": bool(out.get("question_injection_guard_triggered")),
        "final_answer_present": bool(out.get("answer")),
        "top5_complete_evidence": 1.0 if set(case.required_document_ids) <= {x["document_id"] for x in top5} or not case.required_document_ids else 0.0,
        "top5_ids": [x["chunk_id"] for x in top5],
        "top5_docs": [x["document_id"] for x in top5],
    }
    payload.update({k: v for k, v in out.items() if k not in payload})
    return payload
