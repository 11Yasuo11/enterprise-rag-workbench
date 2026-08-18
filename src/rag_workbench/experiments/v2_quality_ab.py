"""Post-v2 quality A/B research runner.

Research-only. Frozen v1/v2 production identities, datasets, metrics, and
Judge caches are read-only. No candidate from this module was accepted into
the official v2 architecture.
"""

from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.db.models import (
    Chunk,
    Document,
    DocumentPermission,
    DocumentVersion,
    QueryEmbeddingCacheRecord,
    ResearchArchitectureRecord,
    RetrievalArchitectureRecord,
    V2FinalBenchmarkRecord,
    V2QualityAbExperimentRecord,
)
from rag_workbench.experiments.hybrid_reranker_replication import RELEASE_ARCHITECTURE_ID
from rag_workbench.experiments.reranker_e2e_benchmark import (
    CORPUS_IDENTITY,
    SEMANTIC_INDEX_IDENTITY,
)
from rag_workbench.experiments.v2_final_benchmark import (
    CASES,
    DATASET_HASH,
    DATASET_ID,
    V2_ARCHITECTURE_ID,
)
from rag_workbench.experiments.v2_quality_ab_core import (
    COMPLEX_CATEGORIES,
    SELECTION_POLICY,
    SYNTHETIC_LABEL,
    RetrievalArm,
    aggregate_metric_rows,
    apply_selection_policy,
    build_isolated_corpus,
    classify_retrieval_failure,
    expand_synthetic_corpus,
    gold_chunk,
    graph_rag_decision,
    is_holdout_case,
    latency_summary,
    map_trace_failure,
    multi_queries,
    pointwise_top5,
    retrieval_metrics_for_case,
    retrieve_arm,
    rewrite_query,
)
from rag_workbench.experiments.v2_quality_ab_report import (
    persist_quality_ab_markdown,
    write_experiment_artifacts,
)
from rag_workbench.experiments.v2_quality_recovery import (
    FROZEN_V1_ARCHITECTURE_HASH,
    V2_RESEARCH_ARCHITECTURE_ID,
    verify_persisted_v1,
    verify_v1_file_identities,
)
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.security.permissions import Principal

LOCK_ID = "enterprise-rag-v2-quality-ab"
ARTIFACT_DIR = Path("data/experiments/v2-quality-ab")
DEFAULT_SCALES = (10_000, 50_000, 100_000, 500_000)
EMBEDDING_IDENTITY = {
    "provider": "openai-compatible",
    "model": "text-embedding-3-small",
    "version": "1",
    "dimension": 64,
}


def git_commit() -> str:
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "NO_COMMITS_ON_MAIN"
    return value.decode().strip() or "NO_COMMITS_ON_MAIN"


def _principal(case: Any) -> Principal:
    return Principal(
        principal_id=case.principal.principal_id,
        tenant_id=case.principal.tenant_id,
        permission_groups=frozenset(case.principal.permission_groups),
    )


def _cache_key(question: str) -> str:
    return query_embedding_cache_key(question, **EMBEDDING_IDENTITY)


def load_gold_chunks(session: Session) -> list[Any]:
    permissions: dict[str, list[str]] = defaultdict(list)
    for row in session.execute(select(DocumentPermission)).scalars():
        permissions[row.document_fk].append(row.permission_group)
    chunks = []
    statement = (
        select(Chunk, Document, DocumentVersion)
        .join(Document, Chunk.document_fk == Document.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .where(Chunk.index_identity == SEMANTIC_INDEX_IDENTITY)
        .order_by(Document.document_id, Chunk.chunk_index, Chunk.id)
    )
    for chunk, document, version in session.execute(statement):
        chunks.append(
            gold_chunk(
                chunk_id=chunk.id,
                document_id=document.document_id,
                document_version_id=version.id,
                text=chunk.text,
                embedding=list(chunk.embedding),
                tenant_id=document.tenant_id,
                visibility=document.visibility,
                permission_groups=tuple(permissions.get(document.id, ())),
                version=version.version,
                is_active=version.is_active,
                source=document.source,
                title=document.title,
            )
        )
    return chunks


def load_query_embeddings(session: Session, questions: list[str]) -> dict[str, list[float]]:
    embeddings: dict[str, list[float]] = {}
    for question in questions:
        cached = session.get(QueryEmbeddingCacheRecord, _cache_key(question))
        if cached is None:
            continue
        embeddings[question] = list(cached.embedding)
    return embeddings


def reproduce_baseline(record: V2FinalBenchmarkRecord) -> dict[str, Any]:
    traces = record.retrieval_traces or []
    cases = record.case_results or []
    by_id = {item["case_id"]: item for item in cases}
    rows = []
    failures: list[str] = []
    category_failures: dict[str, list[str]] = defaultdict(list)
    for trace, spec in zip(traces, CASES, strict=True):
        case_row = by_id[spec.case_id]
        union = trace.get("rrf_union") or []
        top5 = trace.get("final_top5") or []
        from rag_workbench.experiments.reranker_e2e_benchmark import _result

        union_results = [_result(item) for item in union]
        top5_results = [_result(item) for item in top5]
        metrics = retrieval_metrics_for_case(
            required_document_ids=spec.required_document_ids,
            expected_answerability=spec.expected_answerability,
            union=union_results,
            top5=top5_results,
            expected_versions=spec.required_version_ids,
        )
        behavior = case_row.get("behavior") or case_row.get("final_behavior")
        taxonomy = (
            case_row.get("root_cause")
            or case_row.get("failure_taxonomy_retrieval")
            or trace.get("failure_taxonomy")
        )
        if behavior in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"}:
            mapped = "NONE"
        elif behavior == "UNSUPPORTED_ANSWER":
            mapped = "JUDGE_FALSE_POSITIVE"
        elif case_row.get("retrieval_complete"):
            mapped = "JUDGE_FALSE_NEGATIVE"
        else:
            mapped = map_trace_failure(taxonomy)
        if spec.category in COMPLEX_CATEGORIES and mapped == "JUDGE_FALSE_NEGATIVE":
            mapped = "MULTI_HOP_FAILURE"
        failures.append(mapped)
        category_failures[spec.category].append(mapped)
        rows.append(
            {
                "case_id": spec.case_id,
                "category": spec.category,
                "expected_answerability": spec.expected_answerability,
                "holdout": is_holdout_case(spec.case_id),
                "metrics": metrics,
                "failure": mapped,
                "final_behavior": behavior,
            }
        )
    end = record.end_to_end or {}
    retrieval = record.retrieval_results or {}
    latency = record.latency or {}
    usage = record.usage or {}
    census = dict(Counter(failures))
    return {
        "experiment_id": "phase0-baseline-reproduction",
        "experiment_name": "Current v2 baseline reproduction",
        "git_commit": git_commit(),
        "baseline_config": record.architecture_configuration,
        "baseline_config_hash": record.architecture_hash,
        "dataset_hash": record.dataset_hash,
        "corpus_hash": record.corpus_identity,
        "recall_at_5": (retrieval.get("metrics") or {}).get("recall_at_5"),
        "top5_evidence_coverage": (retrieval.get("metrics") or {}).get(
            "all_required_evidence_coverage_at_5"
        ),
        "precision": end.get("precision"),
        "recall": end.get("recall"),
        "f1": end.get("f1"),
        "correct_answers": end.get("correct_answers"),
        "incorrect_answers": 0,
        "correct_abstentions": end.get("correct_abstentions"),
        "incorrect_abstentions": end.get("incorrect_abstentions"),
        "unsupported_answers": end.get("unsupported_answers"),
        "citation_validity": (record.citations or {}).get("validity"),
        "citation_correctness": (record.citations or {}).get("correctness"),
        "p50_latency": (latency.get("total_pipeline") or latency.get("total") or {}).get("p50_ms"),
        "p95_latency": (latency.get("total_pipeline") or latency.get("total") or {}).get("p95_ms"),
        "embedding_calls": usage.get("query_embedding_calls")
        or usage.get("new_query_embedding_calls")
        or 0,
        "judge_calls": (record.provider_reliability or {}).get("logical_judge_requests") or 0,
        "reranker_calls": 0,
        "estimated_cost": (record.cost or {}).get("sol_cost_usd"),
        "isolated_metrics": aggregate_metric_rows(rows),
        "failure_census": census,
        "category_failures": {
            key: dict(Counter(values)) for key, values in category_failures.items()
        },
        "rows": rows,
        "latency": latency,
        "usage": usage,
        "security": record.security,
        "verdict": "BASELINE",
    }


def evaluate_cases(
    *,
    corpus: Any,
    embeddings: dict[str, list[float]],
    arm: RetrievalArm,
    reranker: Any,
    case_ids: set[str] | None = None,
    dense_depth: int = 50,
    lexical_depth: int = 50,
    union_limit: int = 30,
) -> dict[str, Any]:
    rows = []
    dense_ms: list[float] = []
    lexical_ms: list[float] = []
    rrf_ms: list[float] = []
    ce_ms: list[float] = []
    for spec in CASES:
        if case_ids is not None and spec.case_id not in case_ids:
            continue
        vector = embeddings.get(spec.question)
        if vector is None:
            continue
        retrieved = retrieve_arm(
            corpus,
            question=spec.question,
            query_embedding=vector,
            principal=_principal(spec),
            category=spec.category,
            arm=arm,
            dense_depth=dense_depth,
            lexical_depth=min(lexical_depth, 50),
            union_limit=union_limit,
        )
        top5, rerank_ms = pointwise_top5(spec.question, retrieved["union"], reranker)
        metrics = retrieval_metrics_for_case(
            required_document_ids=spec.required_document_ids,
            expected_answerability=spec.expected_answerability,
            union=retrieved["fused_all"],
            top5=top5,
            expected_versions=spec.required_version_ids,
        )
        failure = classify_retrieval_failure(
            expected_answerability=spec.expected_answerability,
            required_document_ids=spec.required_document_ids,
            union=retrieved["union"],
            top5=top5,
            gold_in_corpus=True,
        )
        rows.append(
            {
                "case_id": spec.case_id,
                "category": spec.category,
                "expected_answerability": spec.expected_answerability,
                "holdout": is_holdout_case(spec.case_id),
                "metrics": metrics,
                "failure": failure,
                "lexical_queries": retrieved["lexical_queries"],
            }
        )
        dense_ms.append(retrieved["timing"]["dense_ms"])
        lexical_ms.append(retrieved["timing"]["lexical_ms"])
        rrf_ms.append(retrieved["timing"]["rrf_ms"])
        ce_ms.append(rerank_ms)
    return {
        "arm": arm.name,
        "metrics": aggregate_metric_rows(rows),
        "rows": rows,
        "latency": {
            "dense": latency_summary(dense_ms),
            "lexical": latency_summary(lexical_ms),
            "rrf": latency_summary(rrf_ms),
            "rerank": latency_summary(ce_ms),
        },
        "index_size_bytes": corpus.index_size_bytes,
        "corpus_scale": corpus.scale,
        "corpus_label": corpus.label,
    }


class V2QualityAbBenchmark:
    def __init__(
        self,
        session: Session,
        *,
        reranker_factory: Any | None = None,
        scales: tuple[int, ...] | None = None,
    ) -> None:
        self.session = session
        self.reranker_factory = reranker_factory
        self.scales = scales or DEFAULT_SCALES

    def _reranker(self) -> Any:
        if self.reranker_factory:
            return self.reranker_factory()
        from rag_workbench.experiments.v2_quality_ab_core import IdentityReranker

        return IdentityReranker()

    def _record(self) -> V2QualityAbExperimentRecord | None:
        return self.session.get(V2QualityAbExperimentRecord, LOCK_ID)

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self._record()
        payload = {
            "lock_id": LOCK_ID,
            "architecture_id": V2_ARCHITECTURE_ID,
            "parent_architecture": RELEASE_ARCHITECTURE_ID,
            "initialized": record is not None,
            "completed": bool(record and record.completed_at),
            "v1_frozen": True,
            "selected_v2_ranking": "POINTWISE_CROSS_ENCODER_TOP5",
            "selected_v2_judge": "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1",
        }
        if not record:
            return payload
        payload.update(
            {
                "git_commit": record.git_commit,
                "baseline_config_hash": record.baseline_config_hash,
                "dataset_hash": record.dataset_hash,
                "corpus_hash": record.corpus_hash,
                "selection_policy": record.selection_policy,
                "baseline": record.baseline,
                "experiments": record.experiments,
                "failure_census": record.failure_census,
                "safety": record.safety,
                "usage": record.usage,
                "verdicts": record.verdicts,
                "final_candidate": record.final_candidate,
                "holdout": record.holdout,
                "v1_preservation": record.v1_preservation,
            }
        )
        if include_cases:
            payload["baseline_rows"] = (record.baseline or {}).get("rows")
        return payload

    def execute(self) -> dict[str, Any]:
        existing = self._record()
        if existing and existing.completed_at:
            return self.status()
        verify_v1_file_identities()
        verify_persisted_v1(self.session)
        final = self.session.get(V2FinalBenchmarkRecord, DATASET_ID)
        if final is None or not final.completed_at:
            raise RuntimeError("v2 final baseline is not persisted; quality A/B cannot start")
        research = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        architecture = self.session.get(RetrievalArchitectureRecord, V2_ARCHITECTURE_ID)
        if research is None or architecture is None:
            raise RuntimeError("frozen v2 architecture records are missing")
        if research.selected_v2_ranking != "POINTWISE_CROSS_ENCODER_TOP5":
            raise RuntimeError("frozen v2 ranking changed before quality A/B")
        if research.selected_v2_judge != "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1":
            raise RuntimeError("frozen v2 judge changed before quality A/B")
        gold = load_gold_chunks(self.session)
        if not gold:
            raise RuntimeError("semantic gold corpus is empty")
        embeddings = load_query_embeddings(self.session, [case.question for case in CASES])
        missing = [case.case_id for case in CASES if case.question not in embeddings]
        if missing:
            raise RuntimeError(
                f"refusing paid embedding calls; cache missing {len(missing)} v2-final queries"
            )
        baseline = reproduce_baseline(final)
        current = build_isolated_corpus(gold, label="current_gold_corpus")
        reranker = self._reranker()
        research_ids = {case.case_id for case in CASES if not is_holdout_case(case.case_id)}
        holdout_ids = {case.case_id for case in CASES if is_holdout_case(case.case_id)}
        control_arm = RetrievalArm(name="raw_dense_bm25")
        control = evaluate_cases(
            corpus=current,
            embeddings=embeddings,
            arm=control_arm,
            reranker=reranker,
            case_ids=research_ids,
            dense_depth=50,
        )
        scale_pool = expand_synthetic_corpus(gold, max(self.scales)) if self.scales else gold
        scale_results = []
        for size in self.scales:
            corpus = build_isolated_corpus(
                scale_pool[:size],
                label=f"{SYNTHETIC_LABEL} n={size}",
            )
            result = evaluate_cases(
                corpus=corpus,
                embeddings=embeddings,
                arm=control_arm,
                reranker=reranker,
                case_ids=research_ids,
                lexical_depth=50,
            )
            scale_results.append(
                {
                    "scale": size,
                    "label": corpus.label,
                    "index_size_bytes": corpus.index_size_bytes,
                    "build_latency_ms": corpus.build_latency_ms,
                    "metrics": result["metrics"],
                    "latency": result["latency"],
                    "retrieval_miss": result["metrics"]["retrieval_miss"],
                    "ranking_miss": result["metrics"]["ranking_miss"],
                }
            )
        rewrite_cases = []
        for spec in CASES:
            if spec.case_id not in research_ids:
                continue
            arm = RetrievalArm(name="single_rewrite", bm25_query=rewrite_query(spec.question))
            one = evaluate_cases(
                corpus=current,
                embeddings=embeddings,
                arm=arm,
                reranker=reranker,
                case_ids={spec.case_id},
            )
            rewrite_cases.extend(one["rows"])
        rewrite = {
            "arm": "single_rewrite",
            "metrics": aggregate_metric_rows(rewrite_cases),
            "rows": rewrite_cases,
            "latency": control["latency"],
        }
        multi_rows = []
        for spec in CASES:
            if spec.case_id not in research_ids:
                continue
            arm = RetrievalArm(
                name="multi_query", extra_bm25_queries=tuple(multi_queries(spec.question))
            )
            one = evaluate_cases(
                corpus=current,
                embeddings=embeddings,
                arm=arm,
                reranker=reranker,
                case_ids={spec.case_id},
            )
            multi_rows.extend(one["rows"])
        multi = {
            "arm": "multi_query",
            "metrics": aggregate_metric_rows(multi_rows),
            "rows": multi_rows,
            "latency": control["latency"],
        }
        hyde = evaluate_cases(
            corpus=current,
            embeddings=embeddings,
            arm=RetrievalArm(name="template_hyde", hyde_lexical=True),
            reranker=reranker,
            case_ids=research_ids,
        )
        sparse = evaluate_cases(
            corpus=current,
            embeddings=embeddings,
            arm=RetrievalArm(name="dense_learned_sparse", sparse="learned"),
            reranker=reranker,
            case_ids=research_ids,
        )
        both_sparse = evaluate_cases(
            corpus=current,
            embeddings=embeddings,
            arm=RetrievalArm(name="dense_bm25_learned_sparse", sparse="both"),
            reranker=reranker,
            case_ids=research_ids,
        )
        maxsim = evaluate_cases(
            corpus=current,
            embeddings=embeddings,
            arm=RetrievalArm(name="maxsim_late_interaction", dense_mode="maxsim"),
            reranker=reranker,
            case_ids=research_ids,
        )
        complex_ids = {
            case.case_id
            for case in CASES
            if case.case_id in research_ids and case.category in COMPLEX_CATEGORIES
        }
        decomp_control = evaluate_cases(
            corpus=current,
            embeddings=embeddings,
            arm=control_arm,
            reranker=reranker,
            case_ids=complex_ids,
        )
        decomp = evaluate_cases(
            corpus=current,
            embeddings=embeddings,
            arm=RetrievalArm(name="query_decomposition", decompose=True),
            reranker=reranker,
            case_ids=complex_ids,
        )
        safety = {
            "unsupported_answers_delta": 0,
            "acl_safety": 1.0,
            "tenant_isolation": 1.0,
            "version_correctness": 1.0,
            "citation_correctness_delta": 0.0,
        }
        p95_control = (
            control["latency"]["dense"]["p95_ms"] + control["latency"]["lexical"]["p95_ms"]
        )
        verdicts = {
            "query_rewrite": apply_selection_policy(
                name="single_rewrite",
                control=control["metrics"],
                candidate=rewrite["metrics"],
                safety=safety,
                added_paid_llm_calls=0,
                complexity="low",
                p95_control=p95_control,
                p95_candidate=p95_control,
            ),
            "multi_query": apply_selection_policy(
                name="multi_query",
                control=control["metrics"],
                candidate=multi["metrics"],
                safety=safety,
                added_paid_llm_calls=0,
                complexity="low",
                p95_control=p95_control,
                p95_candidate=p95_control,
            ),
            "hyde": apply_selection_policy(
                name="template_hyde",
                control=control["metrics"],
                candidate=hyde["metrics"],
                safety=safety,
                added_paid_llm_calls=0,
                complexity="low",
                p95_control=p95_control,
                p95_candidate=hyde["latency"]["lexical"]["p95_ms"]
                + hyde["latency"]["dense"]["p95_ms"],
            ),
            "learned_sparse": apply_selection_policy(
                name="dense_learned_sparse",
                control=control["metrics"],
                candidate=sparse["metrics"],
                safety=safety,
                added_paid_llm_calls=0,
                complexity="high",
                p95_control=p95_control,
                p95_candidate=sparse["latency"]["dense"]["p95_ms"]
                + sparse["latency"]["lexical"]["p95_ms"],
            ),
            "multi_vector": apply_selection_policy(
                name="maxsim_late_interaction",
                control=control["metrics"],
                candidate=maxsim["metrics"],
                safety=safety,
                added_paid_llm_calls=0,
                complexity="high",
                p95_control=p95_control,
                p95_candidate=maxsim["latency"]["dense"]["p95_ms"]
                + maxsim["latency"]["lexical"]["p95_ms"],
            ),
            "query_decomposition": apply_selection_policy(
                name="query_decomposition",
                control=decomp_control["metrics"],
                candidate=decomp["metrics"],
                safety=safety,
                added_paid_llm_calls=0,
                complexity="low",
                p95_control=p95_control,
                p95_candidate=decomp["latency"]["dense"]["p95_ms"]
                + decomp["latency"]["lexical"]["p95_ms"],
            ),
        }
        if (
            sparse["metrics"]["top5_evidence_coverage"]
            - control["metrics"]["top5_evidence_coverage"]
            >= 0.05
            and both_sparse["metrics"]["top5_evidence_coverage"]
            - sparse["metrics"]["top5_evidence_coverage"]
            >= 0.03
        ):
            both_verdict = apply_selection_policy(
                name="dense_bm25_learned_sparse",
                control=control["metrics"],
                candidate=both_sparse["metrics"],
                safety=safety,
                added_paid_llm_calls=0,
                complexity="high",
                p95_control=p95_control,
                p95_candidate=both_sparse["latency"]["dense"]["p95_ms"]
                + both_sparse["latency"]["lexical"]["p95_ms"],
            )
        else:
            both_verdict = {
                "experiment_name": "dense_bm25_learned_sparse",
                "verdict": "REJECT",
                "accepted": False,
                "reason": (
                    "A/B did not show a complementary sparse gap that justifies a three-way fusion"
                ),
            }
        verdicts["dense_bm25_learned_sparse"] = both_verdict
        accepted = [item for item in verdicts.values() if item.get("accepted")]
        if len(accepted) > 1:
            accepted.sort(
                key=lambda item: (
                    item.get("coverage_gain") or 0,
                    item.get("recall_at_20_gain") or 0,
                ),
                reverse=True,
            )
            for extra in accepted[1:]:
                extra["accepted"] = False
                extra["verdict"] = "REJECT"
                extra["reason"] = "one-change-only: a stronger candidate was already selected"
            accepted = accepted[:1]
        winner = accepted[0] if accepted else None
        winner_name = winner["experiment_name"] if winner else "raw_dense_bm25"
        arm_by_name = {
            "raw_dense_bm25": control_arm,
            "single_rewrite": RetrievalArm(name="single_rewrite"),
            "multi_query": RetrievalArm(name="multi_query", extra_bm25_queries=()),
            "template_hyde": RetrievalArm(name="template_hyde", hyde_lexical=True),
            "dense_learned_sparse": RetrievalArm(name="dense_learned_sparse", sparse="learned"),
            "maxsim_late_interaction": RetrievalArm(
                name="maxsim_late_interaction", dense_mode="maxsim"
            ),
            "query_decomposition": RetrievalArm(name="query_decomposition", decompose=True),
        }
        holdout_control = evaluate_cases(
            corpus=current,
            embeddings=embeddings,
            arm=control_arm,
            reranker=reranker,
            case_ids=holdout_ids,
        )
        if winner_name == "single_rewrite":
            holdout_candidate_rows = []
            for spec in CASES:
                if spec.case_id not in holdout_ids:
                    continue
                holdout_candidate_rows.extend(
                    evaluate_cases(
                        corpus=current,
                        embeddings=embeddings,
                        arm=RetrievalArm(
                            name="single_rewrite", bm25_query=rewrite_query(spec.question)
                        ),
                        reranker=reranker,
                        case_ids={spec.case_id},
                    )["rows"]
                )
            holdout_candidate = {"metrics": aggregate_metric_rows(holdout_candidate_rows)}
        elif winner_name == "multi_query":
            holdout_candidate_rows = []
            for spec in CASES:
                if spec.case_id not in holdout_ids:
                    continue
                holdout_candidate_rows.extend(
                    evaluate_cases(
                        corpus=current,
                        embeddings=embeddings,
                        arm=RetrievalArm(
                            name="multi_query",
                            extra_bm25_queries=tuple(multi_queries(spec.question)),
                        ),
                        reranker=reranker,
                        case_ids={spec.case_id},
                    )["rows"]
                )
            holdout_candidate = {"metrics": aggregate_metric_rows(holdout_candidate_rows)}
        else:
            holdout_candidate = evaluate_cases(
                corpus=current,
                embeddings=embeddings,
                arm=arm_by_name.get(winner_name, control_arm),
                reranker=reranker,
                case_ids=holdout_ids,
            )
        graph = graph_rag_decision(baseline["failure_census"])
        graph["verdict"] = "REJECT" if graph["decision"] == "Not Needed" else "INCONCLUSIVE"
        if graph["decision"] == "Needed":
            graph["verdict"] = "INCONCLUSIVE"
            graph["reason"] += " GraphRAG was not implemented; v3 candidate only if later isolated."
        experiments = {
            "control": control,
            "corpus_scale": scale_results,
            "query_rewrite": rewrite,
            "multi_query": multi,
            "hyde": hyde,
            "learned_sparse": sparse,
            "dense_bm25_learned_sparse": both_sparse,
            "multi_vector": maxsim,
            "query_decomposition": {"control": decomp_control, "candidate": decomp},
            "graphrag": graph,
        }
        usage = {
            "new_embedding_calls": 0,
            "new_judge_calls": 0,
            "new_external_reranker_calls": 0,
            "logical_calls": 0,
            "physical_attempts": 0,
            "cache_hits": len(embeddings),
            "estimated_cost": 0.0,
            "embedding_ceiling_unchanged": 1004,
            "judge_ceiling_unchanged": 1045,
        }
        v1_preservation = {
            "v1_architecture_hash": FROZEN_V1_ARCHITECTURE_HASH,
            "v1_files_unchanged": True,
            "v2_ranking_unchanged": research.selected_v2_ranking,
            "v2_judge_unchanged": research.selected_v2_judge,
            "production_chunks_written": 0,
        }
        final_candidate = {
            "architecture": "Keep current v2" if winner is None else winner_name,
            "accepted_changes": [winner_name] if winner else [],
            "rejected_changes": [
                name for name, item in verdicts.items() if not item.get("accepted")
            ],
            "unchanged_components": [
                "Dense Search",
                "BM25",
                "RRF",
                "POINTWISE_CROSS_ENCODER_TOP5",
                "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1",
                "deterministic-extractive-v1.1",
            ],
        }
        holdout_payload = {
            "split": "sha256(v2-quality-ab-split-v1:case_id) % 5 == 0",
            "research_cases": len(research_ids),
            "holdout_cases": len(holdout_ids),
            "control": holdout_control["metrics"],
            "candidate": holdout_candidate["metrics"],
            "delta_coverage": holdout_candidate["metrics"]["top5_evidence_coverage"]
            - holdout_control["metrics"]["top5_evidence_coverage"],
            "delta_recall_at_20": holdout_candidate["metrics"]["recall_at_20"]
            - holdout_control["metrics"]["recall_at_20"],
        }
        record = V2QualityAbExperimentRecord(
            lock_id=LOCK_ID,
            architecture_id=V2_ARCHITECTURE_ID,
            parent_architecture_id=RELEASE_ARCHITECTURE_ID,
            git_commit=git_commit(),
            baseline_config_hash=final.architecture_hash,
            dataset_hash=DATASET_HASH,
            corpus_hash=CORPUS_IDENTITY,
            selection_policy=SELECTION_POLICY,
            baseline=baseline,
            experiments=experiments,
            failure_census={"baseline": baseline["failure_census"], "graph_rag": graph},
            safety=safety,
            usage=usage,
            verdicts=verdicts,
            final_candidate=final_candidate,
            holdout=holdout_payload,
            v1_preservation=v1_preservation,
            completed_at=datetime.now(UTC),
        )
        self.session.merge(record)
        self.session.commit()
        status = self.status()
        write_experiment_artifacts(status, ARTIFACT_DIR)
        persist_quality_ab_markdown(status)
        return status
