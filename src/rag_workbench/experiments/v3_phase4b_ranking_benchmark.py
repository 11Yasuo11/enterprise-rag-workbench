# ruff: noqa: E501
"""V3 Phase 4B: Ranking validation benchmark.

Runs Control A (pointwise CE Top-5) vs Candidate B (listwise evidence-set Top-5)
over the 100-case ranking validation dataset using a shared frozen candidate pool.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from sqlalchemy.orm import Session

from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import QueryEmbeddingCacheRecord
from rag_workbench.evaluation.hybrid_metrics import (
    aggregate_retrieval_metrics,
    category_retrieval_metrics,
    retrieval_case_metrics,
)
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    DENSE_DEPTH,
    FINAL_TOP_K,
    RRF_K,
    UNION_LIMIT,
    _principal,
    pool_metrics,
)
from rag_workbench.experiments.reranker_e2e_benchmark import (
    RERANKER_REVISION,
    SEMANTIC_INDEX_IDENTITY,
    _result,
)
from rag_workbench.experiments.v2_final_benchmark import HIDDEN_GROUND_TRUTH_FIELDS
from rag_workbench.experiments.v2_quality_recovery import stable_hash
from rag_workbench.experiments.v2_sufficiency_fn import SufficiencyFnCase, retrieval_complete
from rag_workbench.experiments.v3_phase4b_ranking_cases import (
    DATASET_ID,
    DATASET_PATH,
    dataset_overlap_report,
    write_dataset,
)
from rag_workbench.providers.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from rag_workbench.reranking import CrossEncoderReranker, Reranker
from rag_workbench.reranking.listwise_evidence_set import (
    ALGORITHM_ID as LISTWISE_ALGORITHM_ID,
)
from rag_workbench.reranking.listwise_evidence_set import (
    ALGORITHM_VERSION as LISTWISE_ALGORITHM_VERSION,
)
from rag_workbench.reranking.listwise_evidence_set import (
    listwise_configuration,
    select_listwise_evidence_set_top5,
)
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import Retriever

EXPERIMENT_FAMILY = "v3-phase4b-final-ranking-research"
CONTROL_STRATEGY = "POINTWISE_CROSS_ENCODER_TOP5"
CANDIDATE_B_STRATEGY = "LISTWISE_EVIDENCE_SET_TOP5"
OUTPUT_DIR = Path("data/experiments/v3-phase4b-final-ranking-research")

PROMOTION_POLICY = {
    "all_required_evidence_coverage_at_5_absolute_improvement": 0.08,
    "three_document_coverage_at_5_absolute_improvement": 0.15,
    "rescue_margin": 5,
    "exact_id_recall_gate": ">=_control",
    "version_correctness_gate": ">=_control",
    "near_duplicate_preferred_source_gate": ">=_control",
    "acl_safety_gate": 1.0,
    "tenant_isolation_gate": 1.0,
    "unauthorized_evidence_gate": 0,
    "same_doc_multi_chunk_regression_gate": 0.05,
    "policy_version": "1.0",
    "frozen_before_validation": True,
}


class V2FinalCase(SufficiencyFnCase):
    preferred_source_id: str | None = None


def load_ranking_cases() -> tuple[V2FinalCase, ...]:
    if not DATASET_PATH.exists():
        write_dataset()
    payload = json.loads(DATASET_PATH.read_text())
    return tuple(V2FinalCase.model_validate(item) for item in payload["cases"])


def _ranking_result(candidate: Any, *, rank: int, retrieval_source: str) -> dict[str, Any]:
    if hasattr(candidate, "result"):
        r = candidate.result
        return {
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "document_version_id": r.document_version_id,
            "text": r.text,
            "rank": rank,
            "score": candidate.reranker_score,
            "source": r.source,
            "source_type": r.source_type,
            "title": r.title,
            "version": r.version,
            "page": r.page,
            "section": r.section,
            "metadata": r.metadata,
            "retrieval_source": retrieval_source,
            "dense_score": r.dense_score,
            "lexical_score": r.lexical_score,
            "fusion_score": r.fusion_score,
            "found_by_dense": r.found_by_dense,
            "found_by_bm25": r.found_by_bm25,
        }
    r = candidate if isinstance(candidate, dict) else candidate.__dict__
    return {
        **{k: v for k, v in r.items() if k in {
            "chunk_id", "document_id", "document_version_id", "text",
            "source", "source_type", "title", "version", "page", "section",
            "metadata", "dense_score", "lexical_score", "fusion_score",
            "found_by_dense", "found_by_bm25",
        }},
        "rank": rank,
        "score": r.get("score") or r.get("reranker_score") or 0.0,
        "retrieval_source": retrieval_source,
    }


def _set_quality_metrics(top5: list[dict[str, Any]]) -> dict[str, Any]:
    doc_ids = [item["document_id"] for item in top5]
    unique_docs = set(doc_ids)
    doc_counts = Counter(doc_ids)
    redundant = sum(v - 1 for v in doc_counts.values() if v > 1)
    same_doc_repeats = sum(1 for v in doc_counts.values() if v > 1)
    return {
        "unique_documents_in_top5": len(unique_docs),
        "redundant_chunk_count": redundant,
        "same_document_repeat_count": same_doc_repeats,
    }


def _rescue_regression(
    control_top5: list[dict[str, Any]],
    candidate_top5: list[dict[str, Any]],
    case_obj: V2FinalCase,
) -> dict[str, Any]:
    control_complete = retrieval_complete(case_obj, control_top5)
    candidate_complete = retrieval_complete(case_obj, candidate_top5)
    if not case_obj.expected_answerability:
        return {"rescue": False, "regression": False, "category": None}
    rescue = (not control_complete) and bool(candidate_complete)
    regression = bool(control_complete) and (not candidate_complete)
    regression_category = None
    if regression:
        c_docs = {item["document_id"] for item in candidate_top5}
        required = set(case_obj.required_document_ids)
        missing_docs = required - c_docs
        if missing_docs and case_obj.category == "same_doc_multi_chunk":
            regression_category = "same-document required evidence lost"
        elif case_obj.category == "exact_identifier":
            regression_category = "exact-ID displaced"
        elif case_obj.category == "near_duplicate" or case_obj.category == "version_region":
            regression_category = "version/source displaced"
        elif case_obj.category == "single_document":
            regression_category = "single-document relevance loss"
        else:
            regression_category = "other"
    return {
        "rescue": rescue,
        "regression": regression,
        "regression_category": regression_category,
    }


def apply_promotion_policy(
    control_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    rescues: int,
    regressions: int,
    *,
    same_doc_control: float,
    same_doc_candidate: float,
) -> dict[str, Any]:
    coverage_improvement = (
        candidate_metrics["all_required_evidence_coverage_at_5"]
        - control_metrics["all_required_evidence_coverage_at_5"]
    )
    three_doc_improvement = (
        candidate_metrics["three_document_coverage_at_5"]
        - control_metrics["three_document_coverage_at_5"]
    )
    coverage_pass = coverage_improvement >= PROMOTION_POLICY["all_required_evidence_coverage_at_5_absolute_improvement"]
    three_doc_pass = three_doc_improvement >= PROMOTION_POLICY["three_document_coverage_at_5_absolute_improvement"]
    primary_pass = coverage_pass or three_doc_pass
    rescue_pass = rescues >= regressions + PROMOTION_POLICY["rescue_margin"]
    exact_pass = candidate_metrics["exact_identifier_recall_at_5"] >= control_metrics["exact_identifier_recall_at_5"]
    version_pass = candidate_metrics["version_correctness"] >= control_metrics["version_correctness"]
    dup_pass = candidate_metrics["near_duplicate_preferred_source_success"] >= control_metrics["near_duplicate_preferred_source_success"]
    acl_pass = candidate_metrics["acl_safety"] >= PROMOTION_POLICY["acl_safety_gate"]
    unauthorized_pass = candidate_metrics["unauthorized_result_exposure"] <= PROMOTION_POLICY["unauthorized_evidence_gate"]
    same_doc_pass = (same_doc_control - same_doc_candidate) <= PROMOTION_POLICY["same_doc_multi_chunk_regression_gate"]
    qualified = primary_pass and rescue_pass and exact_pass and version_pass and dup_pass and acl_pass and unauthorized_pass and same_doc_pass
    return {
        "qualified": qualified,
        "coverage_improvement": round(coverage_improvement, 4),
        "three_document_improvement": round(three_doc_improvement, 4),
        "primary_pass": primary_pass,
        "rescue_pass": rescue_pass,
        "exact_id_pass": exact_pass,
        "version_pass": version_pass,
        "near_duplicate_pass": dup_pass,
        "acl_pass": acl_pass,
        "unauthorized_pass": unauthorized_pass,
        "same_doc_multi_chunk_pass": same_doc_pass,
        "rescues": rescues,
        "regressions": regressions,
        "net_rescues": rescues - regressions,
    }


class V3Phase4bRankingBenchmark:
    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        *,
        embedding_provider_factory: Any | None = None,
        reranker_factory: Any | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self._embedding_factory = embedding_provider_factory
        self._reranker_factory = reranker_factory

    def _build_embedding_provider(self) -> OpenAICompatibleEmbeddingProvider:
        if self._embedding_factory:
            return self._embedding_factory()
        return OpenAICompatibleEmbeddingProvider(
            api_key=self.settings.embedding_api_key or "",
            model="text-embedding-3-small",
            dimension=64,
            base_url=self.settings.embedding_base_url,
            version="1",
            provider_name="openai-compatible",
        )

    def _build_retriever(self, provider: OpenAICompatibleEmbeddingProvider) -> Retriever:
        return Retriever(self.session, provider, SEMANTIC_INDEX_IDENTITY)

    def _build_bm25(self) -> BM25Retriever:
        return BM25Retriever(
            self.session,
            index_identity=SEMANTIC_INDEX_IDENTITY,
            embedding_provider="openai-compatible",
            embedding_model="text-embedding-3-small",
            embedding_version="1",
            embedding_dimension=64,
            config=BM25Config(),
        )

    def _build_reranker(self) -> Reranker:
        if self._reranker_factory:
            return self._reranker_factory()
        return CrossEncoderReranker(device="cpu", resolved_revision=RERANKER_REVISION)

    def embedding_preflight(self) -> dict[str, Any]:
        cases = load_ranking_cases()
        keys = {
            query_embedding_cache_key(
                item.question,
                provider="openai-compatible",
                model="text-embedding-3-small",
                version="1",
                dimension=64,
            )
            for item in cases
        }
        matches = sum(self.session.get(QueryEmbeddingCacheRecord, key) is not None for key in keys)
        missing = len(keys) - matches
        return {
            "total_questions": len(cases),
            "cached": matches,
            "missing": missing,
            "document_embedding_calls": 0,
        }

    def run(self) -> dict[str, Any]:
        cases = load_ranking_cases()
        provider = self._build_embedding_provider()
        retriever = self._build_retriever(provider)
        reranker = self._build_reranker()
        bm25 = self._build_bm25()

        dataset_hash = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
        overlap = dataset_overlap_report()
        policy_hash = stable_hash(PROMOTION_POLICY)
        freeze_ts = datetime.now(UTC).isoformat()

        traces: list[dict[str, Any]] = []
        usage = {
            "embedding_calls": 0,
            "cross_encoder_pairs": 0,
            "hosted_judge_calls": 0,
            "hosted_draft_calls": 0,
            "hosted_verifier_calls": 0,
        }
        latencies: dict[str, list[float]] = {
            "embedding_ms": [],
            "dense_ms": [],
            "bm25_ms": [],
            "rrf_ms": [],
            "cross_encoder_ms": [],
            "listwise_ms": [],
        }

        for case_obj in cases:
            embedding = retriever.query_embedding_cache.get_or_embed(case_obj.question)
            self.session.commit()
            usage["embedding_calls"] += embedding.external_calls
            latencies["embedding_ms"].append(embedding.embedding_latency_ms)

            principal = _principal(case_obj)
            dense_started = time.perf_counter()
            dense_candidates = retriever.retrieve_with_embedding(
                embedding,
                top_k=DENSE_DEPTH,
                score_threshold=0.28,
                principal=principal,
            )
            dense_ms = (time.perf_counter() - dense_started) * 1000
            latencies["dense_ms"].append(dense_ms)

            bm25_started = time.perf_counter()
            bm25_candidates = bm25.retrieve(
                case_obj.question,
                top_k=BM25_DEPTH,
                principal=principal,
            )
            bm25_ms = (time.perf_counter() - bm25_started) * 1000
            latencies["bm25_ms"].append(bm25_ms)

            fusion_started = time.perf_counter()
            union_all = reciprocal_rank_fusion(dense_candidates, bm25_candidates, top_k=10_000, rrf_k=RRF_K)
            union = union_all[:UNION_LIMIT]
            rrf_ms = (time.perf_counter() - fusion_started) * 1000
            latencies["rrf_ms"].append(rrf_ms)

            forbidden = set(case_obj.forbidden_document_ids)
            if case_obj.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                leaked = [item for item in union if item.document_id in forbidden]
                if leaked:
                    raise RuntimeError("unauthorized content in candidate pool")

            # --- Cross-Encoder scoring (shared) ---
            ce_started = time.perf_counter()
            reranked = reranker.rerank(case_obj.question, union)
            ce_ms = (time.perf_counter() - ce_started) * 1000
            usage["cross_encoder_pairs"] += len(union)
            latencies["cross_encoder_ms"].append(ce_ms)

            # --- Control A: pointwise CE Top-5 ---
            control_top5 = [
                _ranking_result(item, rank=item.reranked_rank, retrieval_source="pointwise_cross_encoder_top5")
                for item in reranked[:FINAL_TOP_K]
            ]

            # --- Candidate B: listwise evidence-set Top-5 ---
            lw_started = time.perf_counter()
            listwise_selected = select_listwise_evidence_set_top5(
                reranked,
                case_obj.question,
                top_k=FINAL_TOP_K,
            )
            lw_ms = (time.perf_counter() - lw_started) * 1000
            latencies["listwise_ms"].append(lw_ms)
            candidate_top5 = [
                _ranking_result(item, rank=i + 1, retrieval_source="listwise_evidence_set_top5")
                for i, item in enumerate(listwise_selected)
            ]

            leaked_fields = HIDDEN_GROUND_TRUTH_FIELDS.intersection(control_top5[0] if control_top5 else {})
            if leaked_fields:
                raise RuntimeError(f"evaluator labels leaked: {sorted(leaked_fields)}")

            pool_complete = (
                set(case_obj.required_document_ids) <= {item.document_id for item in union}
                if case_obj.expected_answerability else None
            )

            control_metrics = retrieval_case_metrics(case_obj.as_retrieval(), [_result(item) for item in control_top5])
            candidate_metrics = retrieval_case_metrics(case_obj.as_retrieval(), [_result(item) for item in candidate_top5])
            rr = _rescue_regression(control_top5, candidate_top5, case_obj)

            traces.append({
                "case_id": case_obj.case_id,
                "category": case_obj.category,
                "expected_answerability": case_obj.expected_answerability,
                "required_document_ids": list(case_obj.required_document_ids),
                "forbidden_document_ids": list(case_obj.forbidden_document_ids),
                "shared_rrf_union_size": len(union),
                "pool_complete": pool_complete,
                "control_top5": control_top5,
                "candidate_top5": candidate_top5,
                "control_complete": retrieval_complete(case_obj, control_top5),
                "candidate_complete": retrieval_complete(case_obj, candidate_top5),
                "control_metrics": control_metrics,
                "candidate_metrics": candidate_metrics,
                "control_set_quality": _set_quality_metrics(control_top5),
                "candidate_set_quality": _set_quality_metrics(candidate_top5),
                "rescue_regression": rr,
                "pool_metrics": pool_metrics(case_obj.as_retrieval(), union),
                "timing": {
                    "embedding_ms": embedding.embedding_latency_ms,
                    "dense_ms": dense_ms,
                    "bm25_ms": bm25_ms,
                    "rrf_ms": rrf_ms,
                    "cross_encoder_ms": ce_ms,
                    "listwise_ms": lw_ms,
                },
            })

        answerable_traces = [t for t in traces if t["expected_answerability"]]
        control_rows = [{"case_id": t["case_id"], "category": t["category"], "expected_answerability": t["expected_answerability"], "metrics": t["control_metrics"]} for t in traces]
        candidate_rows = [{"case_id": t["case_id"], "category": t["category"], "expected_answerability": t["expected_answerability"], "metrics": t["candidate_metrics"]} for t in traces]

        control_agg = aggregate_retrieval_metrics(control_rows)
        candidate_agg = aggregate_retrieval_metrics(candidate_rows)

        rescues = [t for t in traces if t["rescue_regression"]["rescue"]]
        regressions = [t for t in traces if t["rescue_regression"]["regression"]]

        same_doc_cases = [t for t in answerable_traces if t["category"] == "same_doc_multi_chunk"]
        same_doc_control_coverage = mean([t["control_metrics"]["all_required_evidence_coverage_at_5"] or 0.0 for t in same_doc_cases]) if same_doc_cases else 0.0
        same_doc_candidate_coverage = mean([t["candidate_metrics"]["all_required_evidence_coverage_at_5"] or 0.0 for t in same_doc_cases]) if same_doc_cases else 0.0

        policy_result = apply_promotion_policy(
            control_agg, candidate_agg,
            len(rescues), len(regressions),
            same_doc_control=same_doc_control_coverage,
            same_doc_candidate=same_doc_candidate_coverage,
        )

        pool_all_coverage = mean([
            float(t["pool_complete"]) for t in answerable_traces if t["pool_complete"] is not None
        ]) if answerable_traces else 0.0

        control_set_quality = {
            "mean_unique_documents_in_top5": mean([t["control_set_quality"]["unique_documents_in_top5"] for t in answerable_traces]),
            "mean_redundant_chunk_count": mean([t["control_set_quality"]["redundant_chunk_count"] for t in answerable_traces]),
            "mean_same_document_repeat_count": mean([t["control_set_quality"]["same_document_repeat_count"] for t in answerable_traces]),
        }
        candidate_set_quality = {
            "mean_unique_documents_in_top5": mean([t["candidate_set_quality"]["unique_documents_in_top5"] for t in answerable_traces]),
            "mean_redundant_chunk_count": mean([t["candidate_set_quality"]["redundant_chunk_count"] for t in answerable_traces]),
            "mean_same_document_repeat_count": mean([t["candidate_set_quality"]["same_document_repeat_count"] for t in answerable_traces]),
        }

        regression_categories = Counter(
            t["rescue_regression"]["regression_category"]
            for t in traces if t["rescue_regression"]["regression"]
        )

        result = {
            "experiment_family": EXPERIMENT_FAMILY,
            "experiment_id": f"{EXPERIMENT_FAMILY}-candidate-b",
            "dataset_id": DATASET_ID,
            "dataset_hash": dataset_hash,
            "dataset_case_count": len(cases),
            "dataset_distribution": dict(sorted(Counter(str(c.category) for c in cases).items())),
            "dataset_overlap": overlap,
            "freeze_timestamp": freeze_ts,
            "promotion_policy": PROMOTION_POLICY,
            "promotion_policy_hash": policy_hash,
            "candidate_pool": {
                "embedding_model": "text-embedding-3-small",
                "dense_depth": DENSE_DEPTH,
                "bm25_depth": BM25_DEPTH,
                "rrf_k": RRF_K,
                "union_limit": UNION_LIMIT,
                "pool_all_required_evidence_coverage": pool_all_coverage,
            },
            "control": {
                "strategy": CONTROL_STRATEGY,
                "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
                "revision": RERANKER_REVISION,
                "metrics": control_agg,
                "category_metrics": category_retrieval_metrics(control_rows),
                "set_quality": control_set_quality,
            },
            "candidate_b": {
                "strategy": CANDIDATE_B_STRATEGY,
                "algorithm_id": LISTWISE_ALGORITHM_ID,
                "algorithm_version": LISTWISE_ALGORITHM_VERSION,
                "configuration": listwise_configuration(),
                "configuration_hash": stable_hash(listwise_configuration()),
                "metrics": candidate_agg,
                "category_metrics": category_retrieval_metrics(candidate_rows),
                "set_quality": candidate_set_quality,
            },
            "promotion": policy_result,
            "rescues": {
                "count": len(rescues),
                "case_ids": [t["case_id"] for t in rescues],
            },
            "regressions": {
                "count": len(regressions),
                "case_ids": [t["case_id"] for t in regressions],
                "categories": dict(regression_categories),
            },
            "same_doc_multi_chunk": {
                "control_coverage": round(same_doc_control_coverage, 4),
                "candidate_coverage": round(same_doc_candidate_coverage, 4),
            },
            "usage": usage,
            "latency": {k: {"mean_ms": round(mean(v), 2) if v else 0.0} for k, v in latencies.items()},
            "cost": {
                "embedding_cost": "see embedding preflight",
                "reranking_cost": "$0 (local cross-encoder)",
                "listwise_cost": "$0 (local algorithm)",
                "hosted_llm_cost": "$0",
            },
            "traces": traces,
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = OUTPUT_DIR / "summary.json"
        summary_payload = {k: v for k, v in result.items() if k != "traces"}
        summary_path.write_text(json.dumps(summary_payload, indent=2, default=str) + "\n")
        traces_path = OUTPUT_DIR / "traces.json"
        traces_path.write_text(json.dumps(traces, indent=2, default=str) + "\n")

        return result
