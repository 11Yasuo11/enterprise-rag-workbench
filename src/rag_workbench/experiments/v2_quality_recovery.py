from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.answerability.openai_compatible import EVIDENCE_GATE_PROMPT_VERSION
from rag_workbench.db.models import (
    EndToEndBenchmarkRecord,
    EndToEndBenchmarkRunRecord,
    ResearchArchitectureRecord,
    RetrievalArchitectureRecord,
    RetrievalBenchmarkRecord,
    RetrievalBenchmarkRunRecord,
)
from rag_workbench.experiments import hybrid_reranker_replication as v1_release
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    CONFIGURATION,
    DENSE_DEPTH,
    FINAL_TOP_K,
    MODES,
    RRF_K,
    UNION_LIMIT,
    HybridRerankerCase,
)
from rag_workbench.experiments.hybrid_reranker_replication import (
    ARCHITECTURE_ID,
    FINAL_DATASET_HASH,
    FINAL_DATASET_ID,
    RELEASE_ARCHITECTURE_ID,
    SCHEMA_IDENTITY,
    SOL_MODEL,
    classify_final_root_cause,
)
from rag_workbench.experiments.judge_e2e_benchmark import FrozenJudgeEndToEndBenchmark
from rag_workbench.experiments.reranker_e2e_benchmark import (
    CORPUS_IDENTITY,
    RERANKER_REVISION,
    SEMANTIC_INDEX_IDENTITY,
)
from rag_workbench.retrieval.bm25 import BM25_VERSION, BM25Config

V2_RESEARCH_ARCHITECTURE_ID = "enterprise-rag-workbench-v2-research"
FROZEN_V1_ARCHITECTURE_HASH = "e76aa8834d995c03d47d6d80c9917cc9e16e4f31b24dcd742250329239a23c85"
CENSUS_SOURCE = "persisted-v1-final-eval-traces"
DIAGNOSIS_ONLY_NOTICE = (
    "The frozen v1 final dataset is diagnosis-only. It is not promotion evidence "
    "for any v2 architecture. New v2 candidates must be evaluated on genuinely "
    "unseen datasets."
)

CENSUS_CATEGORIES = (
    "CANDIDATE_GENERATION_MISS",
    "RANKING_OUTSIDE_TOP5",
    "CROSS_ENCODER_FAILED_TO_PROMOTE",
    "CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE",
    "EVIDENCE_GATE_FALSE_NEGATIVE",
    "INVALID_SUPPORTING_ID",
    "SUPPORTING_CONTEXT_LOSS",
    "GENERATION_FAILURE",
    "JUDGE_REQUEST_ERROR",
    "UNKNOWN",
)
RANKING_APPEARANCES = (
    "REDUNDANCY_CROWDING",
    "SAME_DOCUMENT_CROWDING",
    "NEAR_DUPLICATE_CROWDING",
    "POINTWISE_RELEVANCE_MISORDERING",
    "OTHER",
)
JUDGE_FN_CLASSES = (
    "EXACT_IDENTIFIER_FALSE_NEGATIVE",
    "MULTI_DOCUMENT_FALSE_NEGATIVE",
    "NEAR_DUPLICATE_FALSE_NEGATIVE",
    "SEMANTIC_FALSE_NEGATIVE",
    "SUPPORT_SELECTION_FAILURE",
    "SUFFICIENCY_REASONING_FAILURE",
    "REQUEST_ERROR",
    "OTHER",
)
OPERATIONAL_CLASSES = (
    "MODEL_QUALITY_FAILURE",
    "GENERATOR_FAILURE",
    "PROVIDER_REQUEST_FAILURE",
)
PIPELINE_CONTROL_KEYS = (
    "selected_retriever",
    "corpus_identity",
    "semantic_index_identity",
    "dense",
    "embedding",
    "bm25",
    "rrf",
    "bm25_candidate_depth",
    "cross_encoder",
    "final_top_k",
    "judge",
    "generator",
    "acl_version_policy",
)
V2_SECURITY_GUARDRAILS = {
    "acl_safety": 1.0,
    "tenant_isolation": 1.0,
    "version_correctness": 1.0,
    "prompt_injection_boundary": 1.0,
    "unsupported_answers_target": 0,
    "unauthorized_chunks_to_reranker": 0,
    "unauthorized_chunks_to_judge": 0,
}
PRIMARY_V2_BOTTLENECK = "CROSS_ENCODER_SAME_DOCUMENT_TOP5_CROWDING"
RECOMMENDED_RANKING_INTERVENTION = (
    "DOCUMENT_DIVERSIFIED_TOP5: after the frozen Cross-Encoder scores a candidate, "
    "keep only the highest-scoring chunk per document_id and then cut Top-5. Do not "
    "change the Cross-Encoder model, revision, candidate union, BM25, or RRF."
)
V1_BENCHMARK_HEADING = "## Enterprise RAG Workbench v1 — Final Frozen End-to-End Benchmark"
V2_BENCHMARK_HEADING = (
    "## Enterprise RAG Workbench v2 — Quality Recovery Baseline and Failure Census"
)


def stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def pipeline_control(configuration: dict[str, Any]) -> dict[str, Any]:
    return {key: configuration[key] for key in PIPELINE_CONTROL_KEYS if key in configuration}


def expected_v1_pipeline_control() -> dict[str, Any]:
    return {
        "selected_retriever": MODES[1],
        "corpus_identity": CORPUS_IDENTITY,
        "semantic_index_identity": SEMANTIC_INDEX_IDENTITY,
        "dense": {
            "backend": "pgvector_cosine",
            "threshold": 0.28,
            "candidate_depth": DENSE_DEPTH,
        },
        "embedding": {
            "provider": "openai-compatible",
            "model": "text-embedding-3-small",
            "dimension": 64,
            "version": "1",
        },
        "bm25": CONFIGURATION["bm25"],
        "rrf": {"k": RRF_K, "candidate_union_limit": UNION_LIMIT},
        "bm25_candidate_depth": BM25_DEPTH,
        "cross_encoder": {
            "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "revision": RERANKER_REVISION,
        },
        "final_top_k": FINAL_TOP_K,
        "judge": {
            "id": "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1",
            "model": SOL_MODEL,
            "prompt": EVIDENCE_GATE_PROMPT_VERSION,
            "schema_identity": SCHEMA_IDENTITY,
            "supporting_context_only": True,
        },
        "generator": "deterministic-extractive-v1",
        "acl_version_policy": (
            "tenant/ACL/active-version filtering precedes Dense and BM25 candidate branches"
        ),
    }


def bm25_identity() -> dict[str, Any]:
    config = BM25Config()
    return {
        "implementation": "BM25 Okapi",
        "version": BM25_VERSION,
        "k1": config.k1,
        "b": config.b,
        "tokenization": CONFIGURATION["bm25"]["tokenization"],
    }


def trace_case_id(trace: dict[str, Any]) -> str:
    return str(trace.get("case_id") or trace.get("question_id"))


def retrieval_row(prepared: dict[str, Any]) -> dict[str, Any]:
    row = prepared.get("retrieval_row") or prepared
    if "failure_taxonomy" not in row and "evaluation" in prepared:
        row = {
            **row,
            "top5_complete": prepared["evaluation"].get("top5_complete"),
            "failure_taxonomy": prepared["evaluation"].get("failure_taxonomy"),
        }
    return row


def compact_ranks(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    compact = []
    for item in items or []:
        compact.append(
            {
                "rank": item.get("rank"),
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "score": item.get("score"),
            }
        )
    return compact


def result_namespace(case: HybridRerankerCase, trace: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        security_passed=trace.get("security_passed"),
        version_correct=trace.get("version_correct"),
        expected_abstain=case.should_abstain,
        status=trace.get("status"),
        answerability_result=trace.get("answerability_result") or {},
        supporting_context_loss=trace.get("supporting_context_loss") or 0.0,
        supporting_chunk_ids=tuple(trace.get("supporting_chunk_ids") or ()),
        retrieved_chunk_ids=tuple(trace.get("retrieved_chunk_ids") or ()),
        generation_context_chunk_ids=tuple(trace.get("generation_context_chunk_ids") or ()),
        citation_correctness=(
            1.0 if trace.get("citation_correctness") is None else trace["citation_correctness"]
        ),
        answerability_operational_error=trace.get("answerability_operational_error"),
        answer=trace.get("answer"),
        error=trace.get("error"),
    )


def classify_census_category(
    case: HybridRerankerCase, retrieval: dict[str, Any], trace: dict[str, Any]
) -> str | None:
    result = result_namespace(case, trace)
    behavior = FrozenJudgeEndToEndBenchmark._behavior(result)
    if behavior in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"}:
        return None
    operational = result.answerability_operational_error
    if operational == "JUDGE_REQUEST_ERROR" and case.expected_answerability:
        return "JUDGE_REQUEST_ERROR"
    cause = classify_final_root_cause(case, retrieval, result)
    if cause in CENSUS_CATEGORIES:
        return cause
    return "UNKNOWN"


def _pool_complete(case: HybridRerankerCase, retrieval: dict[str, Any]) -> bool:
    required = set(case.required_document_ids)
    if not required:
        return True
    coverage = (retrieval.get("pool") or {}).get("all_required_evidence_coverage")
    if coverage is not None:
        return bool(coverage)
    union = {item["document_id"] for item in retrieval.get("rrf_union") or []}
    return required <= union


def classify_ranking_appearance(
    case: HybridRerankerCase, retrieval: dict[str, Any]
) -> dict[str, Any]:
    required = list(case.required_document_ids)
    top5 = retrieval.get("top5") or []
    top5_docs = [item["document_id"] for item in top5]
    counts = Counter(top5_docs)
    missing = [document_id for document_id in required if document_id not in set(top5_docs)]
    extra_same_document_slots = sum(count - 1 for count in counts.values() if count > 1)
    forbidden = set(case.forbidden_document_ids)
    near_duplicate_slots = sum(1 for document_id in top5_docs if document_id in forbidden)
    covered = [document_id for document_id in required if document_id in set(top5_docs)]
    redundant_required_slots = sum(
        counts[document_id] - 1 for document_id in covered if counts[document_id] > 1
    )
    ce_ranks = retrieval.get("required_ce_ranks") or {}
    missing_ce_ranks = {
        document_id: ce_ranks.get(document_id) for document_id in missing
    }
    if missing and near_duplicate_slots:
        appearance = "NEAR_DUPLICATE_CROWDING"
    elif missing and extra_same_document_slots >= len(missing):
        appearance = "SAME_DOCUMENT_CROWDING"
    elif missing and extra_same_document_slots > 0:
        appearance = "REDUNDANCY_CROWDING"
    elif missing:
        appearance = "POINTWISE_RELEVANCE_MISORDERING"
    else:
        appearance = "OTHER"
    return {
        "required_evidence_displaced_from_top5": missing,
        "number_of_required_documents": len(required),
        "number_of_duplicate_same_document_chunks_in_top5": extra_same_document_slots,
        "near_duplicate_crowding": near_duplicate_slots,
        "redundant_required_slots": redundant_required_slots,
        "cross_encoder_rank_of_missing_required_evidence": missing_ce_ranks,
        "top5_document_ids": top5_docs,
        "appearance": appearance,
    }


def classify_judge_false_negative(
    case: HybridRerankerCase, retrieval: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any]:
    result = result_namespace(case, trace)
    supporting = list(result.supporting_chunk_ids)
    retrieved = set(result.retrieved_chunk_ids)
    answerable = bool((result.answerability_result or {}).get("answerable"))
    required_present = bool(retrieval.get("top5_complete"))
    invalid_support = bool(supporting) and not set(supporting) <= retrieved
    empty_support = not supporting
    top5_by_chunk = {
        item["chunk_id"]: item["document_id"] for item in retrieval.get("top5") or []
    }
    supporting_docs = {
        top5_by_chunk[chunk_id] for chunk_id in supporting if chunk_id in top5_by_chunk
    }
    required = set(case.required_document_ids)
    correct_evidence_ignored = required_present and (
        empty_support or not required <= supporting_docs
    )
    if result.answerability_operational_error == "JUDGE_REQUEST_ERROR":
        label = "REQUEST_ERROR"
    elif answerable and (invalid_support or empty_support or not required <= supporting_docs):
        label = "SUPPORT_SELECTION_FAILURE"
    elif case.category == "exact_identifier":
        label = "EXACT_IDENTIFIER_FALSE_NEGATIVE"
    elif case.category in {"multidoc_two", "multidoc_three"}:
        label = "MULTI_DOCUMENT_FALSE_NEGATIVE"
    elif case.category == "near_duplicate":
        label = "NEAR_DUPLICATE_FALSE_NEGATIVE"
    elif case.category == "semantic_paraphrase":
        label = "SEMANTIC_FALSE_NEGATIVE"
    elif required_present and not answerable:
        label = "SUFFICIENCY_REASONING_FAILURE"
    else:
        label = "OTHER"
    return {
        "class": label,
        "all_required_evidence_actually_present": required_present,
        "supporting_ids_selected": supporting,
        "empty_support": empty_support,
        "invalid_support": invalid_support,
        "correct_evidence_ignored": correct_evidence_ignored,
        "judge_answerable": answerable,
        "reason_code": (result.answerability_result or {}).get("reason_code"),
        "operational_error": result.answerability_operational_error,
    }


def classify_operational_failure(
    case: HybridRerankerCase, retrieval: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any] | None:
    category = classify_census_category(case, retrieval, trace)
    if category == "JUDGE_REQUEST_ERROR":
        label = "PROVIDER_REQUEST_FAILURE"
    elif category == "GENERATION_FAILURE":
        label = "GENERATOR_FAILURE"
    else:
        return None
    result = result_namespace(case, trace)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "census_category": category,
        "class": label,
        "judge_answerable": bool((result.answerability_result or {}).get("answerable")),
        "operational_error": result.answerability_operational_error,
        "status": result.status,
        "historical_retry": False,
    }


def census_record(
    case: HybridRerankerCase, retrieval: dict[str, Any], trace: dict[str, Any]
) -> dict[str, Any] | None:
    category = classify_census_category(case, retrieval, trace)
    if category is None:
        return None
    result = result_namespace(case, trace)
    crowding = classify_ranking_appearance(case, retrieval)
    judge = classify_judge_false_negative(case, retrieval, trace)
    return {
        "case_id": case.case_id,
        "category": case.category,
        "required_evidence": list(case.required_document_ids),
        "candidate_pool_ranks": {
            "dense": retrieval.get("required_dense_ranks") or {},
            "bm25": retrieval.get("required_bm25_ranks") or {},
            "rrf": retrieval.get("required_rrf_ranks") or {},
        },
        "cross_encoder_ranks": retrieval.get("required_ce_ranks") or {},
        "top5": compact_ranks(retrieval.get("top5")),
        "judge_decision": {
            "answerable": bool((result.answerability_result or {}).get("answerable")),
            "reason_code": (result.answerability_result or {}).get("reason_code"),
            "operational_error": result.answerability_operational_error,
        },
        "supporting_ids": list(result.supporting_chunk_ids),
        "final_behavior": FrozenJudgeEndToEndBenchmark._behavior(result),
        "expected_answerability": case.expected_answerability,
        "root_cause": category,
        "ranking_appearance": (
            crowding["appearance"] if not retrieval.get("top5_complete") else None
        ),
        "judge_false_negative_class": (
            judge["class"] if category in {"EVIDENCE_GATE_FALSE_NEGATIVE", "JUDGE_REQUEST_ERROR"}
            else None
        ),
    }


def build_failure_census(
    cases: tuple[HybridRerankerCase, ...],
    prepared: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    prepared_by_id = {trace_case_id(item): retrieval_row(item) for item in prepared}
    traces_by_id = {trace_case_id(item): item for item in traces}
    records = []
    for case in cases:
        retrieval = prepared_by_id[case.case_id]
        trace = traces_by_id[case.case_id]
        record = census_record(case, retrieval, trace)
        if record:
            records.append(record)
    counts = Counter(item["root_cause"] for item in records)
    return {
        "source": CENSUS_SOURCE,
        "diagnosis_only": True,
        "promotion_evidence": False,
        "failed_answerable_case_count": sum(
            1 for item in records if item["expected_answerability"]
        ),
        "counts": {name: counts.get(name, 0) for name in CENSUS_CATEGORIES},
        "cases": records,
    }


def build_ranking_diagnostic(
    cases: tuple[HybridRerankerCase, ...],
    prepared: list[dict[str, Any]],
) -> dict[str, Any]:
    prepared_by_id = {trace_case_id(item): retrieval_row(item) for item in prepared}
    rows = []
    for case in cases:
        if not case.expected_answerability:
            continue
        retrieval = prepared_by_id[case.case_id]
        if not _pool_complete(case, retrieval):
            continue
        if retrieval.get("top5_complete"):
            continue
        crowding = classify_ranking_appearance(case, retrieval)
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                **crowding,
            }
        )
    appearances = Counter(item["appearance"] for item in rows)
    primary = appearances.most_common(1)[0][0] if appearances else "OTHER"
    return {
        "pool_complete_top5_incomplete_count": len(rows),
        "appearance_counts": {name: appearances.get(name, 0) for name in RANKING_APPEARANCES},
        "primary_appearance": primary,
        "cases": rows,
        "ranking_fix_implemented": False,
    }


def build_judge_false_negative_diagnostic(
    cases: tuple[HybridRerankerCase, ...],
    prepared: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    prepared_by_id = {trace_case_id(item): retrieval_row(item) for item in prepared}
    traces_by_id = {trace_case_id(item): item for item in traces}
    rows = []
    for case in cases:
        if not case.expected_answerability:
            continue
        retrieval = prepared_by_id[case.case_id]
        trace = traces_by_id[case.case_id]
        result = result_namespace(case, trace)
        if not retrieval.get("top5_complete"):
            continue
        if FrozenJudgeEndToEndBenchmark._behavior(result) != "INCORRECT_ABSTENTION":
            continue
        if bool((result.answerability_result or {}).get("answerable")) and (
            result.answerability_operational_error != "JUDGE_REQUEST_ERROR"
        ):
            continue
        diagnosis = classify_judge_false_negative(case, retrieval, trace)
        rows.append({"case_id": case.case_id, "category": case.category, **diagnosis})
    classes = Counter(item["class"] for item in rows)
    return {
        "retrieval_complete_sol_false_negative_count": len(rows),
        "class_counts": {name: classes.get(name, 0) for name in JUDGE_FN_CLASSES},
        "cases": rows,
        "judge_modified": False,
    }


def build_operational_diagnostic(
    cases: tuple[HybridRerankerCase, ...],
    prepared: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    prepared_by_id = {trace_case_id(item): retrieval_row(item) for item in prepared}
    traces_by_id = {trace_case_id(item): item for item in traces}
    watched = ("fv1_ver_03", "fv1_dup_05")
    rows = []
    for case in cases:
        retrieval = prepared_by_id[case.case_id]
        trace = traces_by_id[case.case_id]
        classified = classify_operational_failure(case, retrieval, trace)
        if classified is None:
            continue
        rows.append(classified)
    counts = Counter(item["class"] for item in rows)
    return {
        "watched_case_ids": list(watched),
        "class_counts": {name: counts.get(name, 0) for name in OPERATIONAL_CLASSES},
        "cases": rows,
        "historical_calls_retried": False,
    }


def verify_v1_file_identities() -> dict[str, Any]:
    text = Path("BENCHMARK.md").read_text()
    heading_count = text.count(V1_BENCHMARK_HEADING)
    section = text.split(V1_BENCHMARK_HEADING, 1)[1] if heading_count else ""
    dataset_hash = hashlib.sha256(
        Path("data/eval/acmeai_enterprise_rag_v1_final_eval.json").read_bytes()
    ).hexdigest()
    checks = {
        "final_v1_architecture_record": RELEASE_ARCHITECTURE_ID,
        "final_v1_benchmark_dataset": FINAL_DATASET_ID,
        "final_v1_benchmark_dataset_hash": dataset_hash == FINAL_DATASET_HASH,
        "final_v1_benchmark_run": FINAL_DATASET_ID,
        "benchmark_md_historical_section": heading_count == 1,
        "corpus_identity": CORPUS_IDENTITY,
        "semantic_index_identity": SEMANTIC_INDEX_IDENTITY,
        "bm25_identity": bm25_identity(),
        "rrf_identity": {"k": RRF_K, "candidate_union_limit": UNION_LIMIT},
        "cross_encoder_revision": RERANKER_REVISION,
        "sol_judge_identity": "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1",
        "prompt_schema_identity": SCHEMA_IDENTITY,
        "generator_identity": "deterministic-extractive-v1",
        "architecture_hash_recorded": FROZEN_V1_ARCHITECTURE_HASH in section,
        "hybrid_cross_encoder_recorded": "HYBRID_CROSS_ENCODER_RERANK" in section,
        "v1_experiment_rows_preserved": True,
    }
    missing = [
        name
        for name, value in checks.items()
        if value is False or value is None
    ]
    if missing:
        raise ValueError(f"v1 identities drifted: {missing}")
    return checks


def verify_persisted_v1(session: Session) -> dict[str, Any]:
    release = session.get(RetrievalArchitectureRecord, RELEASE_ARCHITECTURE_ID)
    retriever = session.get(RetrievalArchitectureRecord, ARCHITECTURE_ID)
    e2e = session.get(EndToEndBenchmarkRecord, FINAL_DATASET_ID)
    retrieval = session.get(RetrievalBenchmarkRecord, FINAL_DATASET_ID)
    e2e_run = session.scalars(
        select(EndToEndBenchmarkRunRecord).where(
            EndToEndBenchmarkRunRecord.dataset_id == FINAL_DATASET_ID
        )
    ).first()
    retrieval_run = session.scalars(
        select(RetrievalBenchmarkRunRecord).where(
            RetrievalBenchmarkRunRecord.dataset_id == FINAL_DATASET_ID
        )
    ).first()
    if not release or not retriever or not e2e or not retrieval or not e2e_run or not retrieval_run:
        raise ValueError("frozen v1 architecture or final benchmark traces are missing")
    if not release.immutable or not retriever.immutable:
        raise ValueError("frozen v1 architecture records must remain immutable")
    if release.architecture_id == V2_RESEARCH_ARCHITECTURE_ID:
        raise ValueError("v2 research identity must not replace v1")
    if e2e.dataset_hash != FINAL_DATASET_HASH or release.dataset_hash != FINAL_DATASET_HASH:
        raise ValueError("frozen v1 dataset hash changed")
    if not e2e.completed_at or not e2e.prepared_cases:
        raise ValueError("final v1 benchmark traces are incomplete")
    configuration = release.configuration or {}
    if configuration.get("architecture_hash") != FROZEN_V1_ARCHITECTURE_HASH:
        raise ValueError("frozen v1 architecture hash changed")
    control = pipeline_control(configuration)
    expected = expected_v1_pipeline_control()
    if control != expected:
        raise ValueError("frozen v1 pipeline control configuration drifted")
    return {
        "release_architecture_id": release.architecture_id,
        "retriever_architecture_id": retriever.architecture_id,
        "selected_retriever": release.selected_retriever,
        "architecture_hash": configuration.get("architecture_hash"),
        "dataset_id": e2e.dataset_id,
        "dataset_hash": e2e.dataset_hash,
        "prepared_case_count": len(e2e.prepared_cases or []),
        "e2e_trace_count": len(e2e_run.case_results or []),
        "retrieval_trace_count": len(retrieval_run.case_results or []),
        "completed_at": e2e.completed_at.isoformat() if e2e.completed_at else None,
        "frozen_at": release.frozen_at.isoformat() if release.frozen_at else None,
        "pipeline_control_hash": stable_hash(control),
        "v1_rows_mutated": False,
    }


def v2_control_configuration(v1_configuration: dict[str, Any]) -> dict[str, Any]:
    control = pipeline_control(v1_configuration)
    payload = {
        **control,
        "architecture_id": V2_RESEARCH_ARCHITECTURE_ID,
        "parent_architecture": RELEASE_ARCHITECTURE_ID,
        "research_status": "ACTIVE",
        "production_status": False,
        "diagnosis_dataset_id": FINAL_DATASET_ID,
        "diagnosis_dataset_hash": FINAL_DATASET_HASH,
        "diagnosis_only": True,
        "promotion_evidence_forbidden": True,
        "control_equivalence_hash": stable_hash(control),
    }
    payload["architecture_hash"] = stable_hash(payload)
    return payload


class V2QualityRecoveryBaseline:
    def __init__(self, session: Session) -> None:
        self.session = session

    def initialize(self) -> ResearchArchitectureRecord:
        files = verify_v1_file_identities()
        persisted = verify_persisted_v1(self.session)
        existing = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        release = self.session.get(RetrievalArchitectureRecord, RELEASE_ARCHITECTURE_ID)
        e2e = self.session.get(EndToEndBenchmarkRecord, FINAL_DATASET_ID)
        e2e_run = self.session.scalars(
            select(EndToEndBenchmarkRunRecord).where(
                EndToEndBenchmarkRunRecord.dataset_id == FINAL_DATASET_ID
            )
        ).first()
        if release is None or e2e is None or e2e_run is None:
            raise ValueError("frozen v1 architecture or final benchmark traces are missing")
        cases = v1_release.FINAL_CASES
        census = build_failure_census(cases, e2e.prepared_cases or [], e2e_run.case_results or [])
        ranking = build_ranking_diagnostic(cases, e2e.prepared_cases or [])
        judge = build_judge_false_negative_diagnostic(
            cases, e2e.prepared_cases or [], e2e_run.case_results or []
        )
        operational = build_operational_diagnostic(
            cases, e2e.prepared_cases or [], e2e_run.case_results or []
        )
        configuration = v2_control_configuration(release.configuration)
        payload = {
            "architecture_id": V2_RESEARCH_ARCHITECTURE_ID,
            "parent_architecture_id": RELEASE_ARCHITECTURE_ID,
            "selected_retriever": release.selected_retriever,
            "research_status": "ACTIVE",
            "production_status": False,
            "control_configuration": configuration,
            "control_equivalence_hash": configuration["control_equivalence_hash"],
            "diagnosis_dataset_id": FINAL_DATASET_ID,
            "diagnosis_dataset_hash": FINAL_DATASET_HASH,
            "diagnosis_only": True,
            "promotion_evidence_forbidden": True,
            "security_guardrails": V2_SECURITY_GUARDRAILS,
            "v1_preservation": {"files": files, "persisted": persisted},
            "failure_census": census,
            "ranking_diagnostic": ranking,
            "judge_false_negative_diagnostic": judge,
            "operational_diagnostic": operational,
            "primary_bottleneck": PRIMARY_V2_BOTTLENECK,
            "recommended_ranking_intervention": RECOMMENDED_RANKING_INTERVENTION,
            "immutable": True,
        }
        if existing:
            if (
                existing.parent_architecture_id != RELEASE_ARCHITECTURE_ID
                or existing.production_status is True
                or existing.control_equivalence_hash != payload["control_equivalence_hash"]
            ):
                raise ValueError("frozen v2 research identity is immutable")
            return existing
        record = ResearchArchitectureRecord(**payload)
        self.session.add(record)
        self.session.commit()
        if self.session.get(RetrievalArchitectureRecord, RELEASE_ARCHITECTURE_ID) is None:
            raise RuntimeError("v2 initialize deleted the frozen v1 architecture")
        return record

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        payload: dict[str, Any] = {
            "architecture_id": V2_RESEARCH_ARCHITECTURE_ID,
            "parent_architecture": RELEASE_ARCHITECTURE_ID,
            "research_status": "ACTIVE",
            "production_status": False,
            "diagnosis_only": True,
            "promotion_evidence_forbidden": True,
            "diagnosis_notice": DIAGNOSIS_ONLY_NOTICE,
            "initialized": record is not None,
            "security_guardrails": V2_SECURITY_GUARDRAILS,
            "primary_bottleneck": PRIMARY_V2_BOTTLENECK,
            "recommended_ranking_intervention": RECOMMENDED_RANKING_INTERVENTION,
        }
        if not record:
            return payload
        census = record.failure_census
        ranking = record.ranking_diagnostic
        judge = record.judge_false_negative_diagnostic
        operational = record.operational_diagnostic
        if not include_cases:
            census = {key: value for key, value in census.items() if key != "cases"}
            ranking = {key: value for key, value in ranking.items() if key != "cases"}
            judge = {key: value for key, value in judge.items() if key != "cases"}
            operational = {key: value for key, value in operational.items() if key != "cases"}
        payload.update(
            {
                "parent_architecture_id": record.parent_architecture_id,
                "selected_retriever": record.selected_retriever,
                "research_status": record.research_status,
                "production_status": record.production_status,
                "control_configuration": record.control_configuration,
                "control_equivalence_hash": record.control_equivalence_hash,
                "diagnosis_dataset_id": record.diagnosis_dataset_id,
                "diagnosis_dataset_hash": record.diagnosis_dataset_hash,
                "v1_preservation": record.v1_preservation,
                "failure_census": census,
                "ranking_diagnostic": ranking,
                "judge_false_negative_diagnostic": judge,
                "operational_diagnostic": operational,
                "security_guardrails": record.security_guardrails,
                "primary_bottleneck": record.primary_bottleneck,
                "recommended_ranking_intervention": record.recommended_ranking_intervention,
                "selected_v2_ranking": record.selected_v2_ranking,
                "phase1_dataset_id": record.phase1_dataset_id,
                "phase2_dataset_id": record.phase2_dataset_id,
                "ranking_research_status": record.ranking_research_status,
                "selected_v2_judge": record.selected_v2_judge,
                "phase3_dataset_id": record.phase3_dataset_id,
                "judge_research_status": record.judge_research_status,
                "phase4_lock_id": record.phase4_lock_id,
                "reliability_research_status": record.reliability_research_status,
                "immutable": record.immutable,
                "frozen_at": record.frozen_at,
            }
        )
        from rag_workbench.experiments.v2_document_diversity import V2DocumentDiversityBenchmark
        from rag_workbench.experiments.v2_soft_document_cap import V2SoftDocumentCapBenchmark
        from rag_workbench.experiments.v2_sufficiency_fn import V2SufficiencyFnBenchmark

        payload["phase1"] = V2DocumentDiversityBenchmark(self.session).status(
            include_cases=include_cases
        )
        payload["phase2"] = V2SoftDocumentCapBenchmark(self.session).status(
            include_cases=include_cases
        )
        payload["phase3"] = V2SufficiencyFnBenchmark(self.session).status(
            include_cases=include_cases
        )
        from rag_workbench.experiments.v2_reliability import V2ReliabilityHardening

        payload["phase4"] = V2ReliabilityHardening(self.session).status(
            include_cases=include_cases
        )
        payload["reliability_research_status"] = record.reliability_research_status
        payload["phase4_lock_id"] = record.phase4_lock_id
        return payload
