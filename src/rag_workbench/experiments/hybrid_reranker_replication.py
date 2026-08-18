from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.cache import CachedAnswerabilityGate, gate_cache_key
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    OpenAICompatibleAnswerabilityGate,
    evidence_gate_schema,
    hosted_judge_request_settings,
)
from rag_workbench.answerability.planning import ExternalJudgeCallLimitGate
from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import (
    AnswerabilityGateCacheRecord,
    Chunk,
    Document,
    DocumentVersion,
    EndToEndBenchmarkRecord,
    EndToEndBenchmarkRunRecord,
    ExperimentRunRecord,
    QueryEmbeddingCacheRecord,
    RetrievalArchitectureRecord,
    RetrievalBenchmarkRecord,
    RetrievalBenchmarkRunRecord,
)
from rag_workbench.evaluation.evaluator import CaseResult, EvaluationRunner
from rag_workbench.evaluation.hybrid_metrics import aggregate_retrieval_metrics
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    CONFIGURATION,
    DENSE_DEPTH,
    FINAL_TOP_K,
    MODES,
    RRF_K,
    UNION_LIMIT,
    HybridRerankerBenchmark,
    HybridRerankerCase,
    _principal,
    aggregate_pool,
    compare_three_document,
)
from rag_workbench.experiments.judge_e2e_benchmark import FrozenJudgeEndToEndBenchmark
from rag_workbench.experiments.reranker_e2e_benchmark import (
    CORPUS_IDENTITY,
    RERANKER_REVISION,
    SEMANTIC_INDEX_IDENTITY,
    FixedResultRetriever,
    RerankerEndToEndBenchmark,
    _percentile,
    _result,
)
from rag_workbench.experiments.sol_judge_e2e_benchmark import (
    OFFICIAL_PRICING,
    official_token_cost,
)
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.generation.generator import RagService
from rag_workbench.providers.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from rag_workbench.providers.llm import ExtractiveGenerationProvider
from rag_workbench.reranking import CrossEncoderReranker, Reranker
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import RetrievalTiming, Retriever

DATASET_ID = "acmeai-hybrid-reranker-replication-v1"
DATASET_PATH = Path("data/eval/acmeai_hybrid_reranker_replication_v1.json")
DATASET_HASH = "f9144a0bbc6f15de030963199b8a784d9e23031a017258a8b6636b131cb04114"
GENERATION_METHOD = "manual-corpus-grounded-v1"
MAXIMUM_PRIOR_OVERLAP = 0.4166666666666667
OVERLAP_CEILING = 0.5
ARCHITECTURE_ID = "enterprise-rag-v1-retriever"
HISTORICAL_DATASET_ID = "acmeai-hybrid-reranker-eval-v1"
SOL_MODEL = "gpt-5.6-sol"
SCHEMA_IDENTITY = "6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621"
SPLIT_SEED = 0
REPLICATION_POLICY = {
    "all_required_coverage_minimum_gain": 0.08,
    "three_document_coverage_minimum_gain": 0.15,
    "required_recall_maximum_regression": 0.02,
    "exact_identifier_maximum_regression": 0.05,
    "semantic_maximum_regression": 0.05,
    "version_correctness_required": 1.0,
    "acl_safety_required": 1.0,
    "unauthorized_results_required": 0,
    "no_calibration_holdout_split": True,
    "previous_selection_rule_unchanged": True,
    "not_a_retroactive_change": True,
    "policy_frozen_before_retrieval": True,
    "reason": (
        "Final replication rule for an independently created unseen dataset. "
        "The previous calibration/holdout experiment remains unchanged. "
        "Promote HYBRID_CROSS_ENCODER_RERANK only if All Required Evidence "
        "Coverage@5 gains at least +0.08 or Three-document Coverage@5 gains at "
        "least +0.15, and all guardrails hold."
    ),
}


def load_replication_cases() -> tuple[HybridRerankerCase, ...]:
    payload = json.loads(DATASET_PATH.read_text())
    return tuple(HybridRerankerCase.model_validate(item) for item in payload["cases"])


CASES = load_replication_cases()
SPLIT_IDENTITY = hashlib.sha256(
    f"replication-no-split:{'|'.join(item.case_id for item in CASES)}".encode()
).hexdigest()


def maximum_prior_dataset_overlap() -> float:
    token = re.compile(r"[a-z0-9]+")

    def terms(question: str) -> set[str]:
        return set(token.findall(question.casefold()))

    maximum = 0.0
    for path in Path("data/eval").glob("*.json"):
        if path == DATASET_PATH:
            continue
        for previous in json.loads(path.read_text()).get("cases", []):
            for case in CASES:
                left, right = terms(case.question), terms(previous["question"])
                maximum = max(maximum, len(left & right) / len(left | right))
    return maximum


def apply_replication_policy(dense: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    coverage_gain = (
        hybrid["all_required_evidence_coverage_at_5"]
        - dense["all_required_evidence_coverage_at_5"]
    )
    three_gain = hybrid["three_document_coverage_at_5"] - dense["three_document_coverage_at_5"]
    recall_regression = (
        dense["required_evidence_recall_at_5"] - hybrid["required_evidence_recall_at_5"]
    )
    exact_regression = (
        dense["exact_identifier_recall_at_5"] - hybrid["exact_identifier_recall_at_5"]
    )
    semantic_regression = dense["semantic_success"] - hybrid["semantic_success"]
    material = coverage_gain >= 0.08 or three_gain >= 0.15
    safe = (
        recall_regression <= 0.02
        and exact_regression <= 0.05
        and semantic_regression <= 0.05
        and hybrid["version_correctness"] == 1.0
        and hybrid["acl_safety"] == 1.0
        and hybrid.get("unauthorized_result_exposure", 0) == 0
    )
    selected = MODES[1] if material and safe else MODES[0]
    return {
        "selected_mode": selected,
        "coverage_gain": coverage_gain,
        "three_document_gain": three_gain,
        "required_recall_regression": recall_regression,
        "exact_identifier_regression": exact_regression,
        "semantic_regression": semantic_regression,
        "material": material,
        "safe": safe,
        "reason": (
            f"coverage_gain={coverage_gain:.6f}; three_doc_gain={three_gain:.6f}; "
            f"recall_regression={recall_regression:.6f}; "
            f"exact_regression={exact_regression:.6f}; "
            f"semantic_regression={semantic_regression:.6f}; "
            f"version={hybrid['version_correctness']}; "
            f"acl={hybrid['acl_safety']}; "
            f"unauthorized={hybrid.get('unauthorized_result_exposure', 0)}."
        ),
    }


def classify_b_three_document_failure(row: dict[str, Any]) -> str | None:
    if row["top5_complete"]:
        return None
    required = set(row["required_document_ids"])
    dense = {item["document_id"] for item in row["dense_candidates"]}
    bm25 = {item["document_id"] for item in row["bm25_candidates"]}
    union = {item["document_id"] for item in row["rrf_union"]}
    top5 = {item["document_id"] for item in row["top5"]}
    if not (required & dense) and not (required & bm25):
        return "BOTH_BRANCHES_MISS"
    if not required <= dense and required <= bm25 and not required <= union:
        return "RRF_TRUNCATION_LOSS"
    if not required <= dense and required <= union and not required <= top5:
        return "CROSS_ENCODER_FAILED_TO_PROMOTE"
    if required <= dense and required <= union and not required <= top5:
        missing_from_top5 = required - top5
        if missing_from_top5:
            return "CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE"
        return "RANKING_OUTSIDE_TOP5"
    if required <= union and not required <= top5:
        return "RANKING_OUTSIDE_TOP5"
    return "UNKNOWN"


def classify_retrieval_regression(
    category: str, a_row: dict[str, Any], b_row: dict[str, Any]
) -> dict[str, Any] | None:
    a_top5 = {item["document_id"] for item in a_row["top5"]}
    b_top5 = {item["document_id"] for item in b_row["top5"]}
    required = set(a_row["required_document_ids"])
    complete_loss = bool(a_row["top5_complete"]) and not bool(b_row["top5_complete"])
    required_loss = bool(required & a_top5) and not bool(required & b_top5)
    if not complete_loss and not required_loss:
        return None
    family = {
        "semantic_paraphrase": "semantic regression",
        "exact_identifier": "exact-ID regression",
        "version_region": "version regression",
        "near_duplicate": "near-duplicate regression",
    }.get(category, "other regression")
    return {
        "case_id": a_row["case_id"],
        "category": category,
        "family": family,
        "complete_to_incomplete": complete_loss,
        "required_evidence_left_top5": required_loss,
    }


class HybridRerankerReplicationBenchmark(HybridRerankerBenchmark):
    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        *,
        embedding_provider_factory: Any | None = None,
        reranker_factory: Any | None = None,
        gate_factory: Any | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embedding_provider_factory = embedding_provider_factory
        self.reranker_factory = reranker_factory
        self.gate_factory = gate_factory
        if hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() != DATASET_HASH:
            raise ValueError("hybrid reranker replication dataset identity changed")
        computed = hashlib.sha256(
            json.dumps(evidence_gate_schema(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if computed != SCHEMA_IDENTITY:
            raise ValueError("evidence-sufficiency-v1 schema identity changed")
        overlap = maximum_prior_dataset_overlap()
        if abs(overlap - MAXIMUM_PRIOR_OVERLAP) > 1e-12:
            raise ValueError("replication overlap identity changed")
        if overlap >= OVERLAP_CEILING:
            raise ValueError("replication overlap exceeds the accepted threshold")
        if self.reranker_factory is None:
            from rag_workbench.reranking import CrossEncoderReranker

            self.reranker_factory = lambda: CrossEncoderReranker(
                device="cpu", resolved_revision=RERANKER_REVISION
            )

    @contextmanager
    def _bind_dataset(self) -> Iterator[None]:
        import rag_workbench.experiments.hybrid_reranker_benchmark as parent

        saved_id = parent.DATASET_ID
        parent.DATASET_ID = DATASET_ID
        try:
            yield
        finally:
            parent.DATASET_ID = saved_id

    def _verify(self, record: RetrievalBenchmarkRecord) -> None:
        if not (
            record.dataset_hash == DATASET_HASH
            and record.split_identity == SPLIT_IDENTITY
            and record.split_seed == SPLIT_SEED
            and record.retrieval_configuration == CONFIGURATION
            and record.selection_policy == REPLICATION_POLICY
            and record.semantic_index_identity == SEMANTIC_INDEX_IDENTITY
            and record.corpus_identity == CORPUS_IDENTITY
            and record.holdout_case_ids == []
        ):
            raise ValueError("sealed hybrid reranker replication identity changed")

    def _verify_production_identities(self) -> None:
        historical = self.session.get(RetrievalBenchmarkRecord, HISTORICAL_DATASET_ID)
        if (
            not historical
            or historical.selected_retrieval_mode != MODES[0]
            or historical.holdout_completed_at is None
        ):
            raise ValueError("previous Dense/Hybrid selection or holdout identity changed")
        sol = self.session.get(EndToEndBenchmarkRecord, "acmeai-sol-judge-e2e-eval-v1")
        from rag_workbench.db.models import RerankingBenchmarkRecord

        reranker = self.session.get(RerankingBenchmarkRecord, "acmeai-reranking-eval-v1")
        if not sol or sol.production_retriever_status != "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1":
            raise ValueError("production Sol judge identity changed")
        if (
            not reranker
            or reranker.selected_mode != "DENSE_CROSS_ENCODER_RERANK"
            or reranker.model_revision != RERANKER_REVISION
        ):
            raise ValueError("frozen Cross-Encoder identity changed")

    def embedding_preflight(self) -> dict[str, int]:
        from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key

        questions = tuple(dict.fromkeys(case.question for case in CASES))
        keys = [
            query_embedding_cache_key(
                question,
                provider="openai-compatible",
                model="text-embedding-3-small",
                version="1",
                dimension=64,
            )
            for question in questions
        ]
        matches = sum(self.session.get(QueryEmbeddingCacheRecord, key) is not None for key in keys)
        current = self._embedding_calls()
        return {
            "current_cumulative_embedding_calls": current,
            "configured_ceiling": self.settings.max_external_embedding_calls,
            "new_unique_queries": len(questions),
            "existing_cache_matches": matches,
            "missing_embeddings": len(keys) - matches,
            "maximum_new_calls": len(keys) - matches,
            "expected_cumulative_ending_usage": current + len(keys) - matches,
        }

    def initialize(self) -> RetrievalBenchmarkRecord:
        if not self.session.scalar(
            select(Chunk.id).where(Chunk.index_identity == SEMANTIC_INDEX_IDENTITY).limit(1)
        ):
            raise ValueError("frozen semantic index is unavailable")
        corpus = self._corpus_identity()
        if corpus != CORPUS_IDENTITY:
            raise ValueError("corpus identity changed")
        existing = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        if existing:
            self._verify(existing)
            return existing
        self._verify_production_identities()
        if maximum_prior_dataset_overlap() >= OVERLAP_CEILING:
            raise ValueError("replication overlap exceeds the accepted threshold")
        record = RetrievalBenchmarkRecord(
            dataset_id=DATASET_ID,
            dataset_hash=DATASET_HASH,
            split_identity=SPLIT_IDENTITY,
            split_seed=SPLIT_SEED,
            calibration_case_ids=[item.case_id for item in CASES],
            holdout_case_ids=[],
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=CORPUS_IDENTITY,
            retrieval_configuration=CONFIGURATION,
            selection_policy=REPLICATION_POLICY,
        )
        self.session.add(record)
        if self.session.get(EndToEndBenchmarkRecord, DATASET_ID) is None:
            self.session.add(
                EndToEndBenchmarkRecord(
                    dataset_id=DATASET_ID,
                    dataset_hash=DATASET_HASH,
                    case_ids=[item.case_id for item in CASES],
                    category_distribution=dict(
                        sorted(Counter(item.category for item in CASES).items())
                    ),
                    generation_method=GENERATION_METHOD,
                    maximum_prior_overlap=MAXIMUM_PRIOR_OVERLAP,
                    semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
                    corpus_identity=CORPUS_IDENTITY,
                    reranker_revision=RERANKER_REVISION,
                    pipeline_a_configuration={**CONFIGURATION, "mode": MODES[0]},
                    pipeline_b_configuration={**CONFIGURATION, "mode": MODES[1]},
                    embedding_preflight=self.embedding_preflight(),
                )
            )
        self.session.commit()
        return record

    def retrieve(self) -> dict[str, Any]:
        record = self.initialize()
        if record.selection_policy != REPLICATION_POLICY:
            raise ValueError("replication policy changed after freeze")
        if record.locked_at is not None:
            raise ValueError("replication retrieval is already locked")
        if self._partition_runs("replication"):
            raise ValueError("replication retrieval is one-shot")
        judge_before = self._historical_judge_calls()
        with self._bind_dataset():
            metrics = self._run_partition("replication", CASES)
        if self._historical_judge_calls() != judge_before:
            raise RuntimeError("Sol was used during replication retrieval")
        dense = metrics[MODES[0]]["metrics"]
        hybrid = metrics[MODES[1]]["metrics"]
        decision = apply_replication_policy(dense, hybrid)
        regressions = []
        dense_rows = {item["case_id"]: item for item in metrics[MODES[0]]["cases"]}
        hybrid_rows = {item["case_id"]: item for item in metrics[MODES[1]]["cases"]}
        for case in CASES:
            found = classify_retrieval_regression(
                case.category, dense_rows[case.case_id], hybrid_rows[case.case_id]
            )
            if found:
                regressions.append(found)
        three_failures = [
            {
                "case_id": item["case_id"],
                "classification": classify_b_three_document_failure(hybrid_rows[item["case_id"]]),
            }
            for item in metrics[MODES[1]]["cases"]
            if item["category"] == "multidoc_three" and not item["top5_complete"]
        ]
        record = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        record.selected_retrieval_mode = decision["selected_mode"]
        record.calibration_metrics = {
            "modes": {mode: payload["metrics"] for mode, payload in metrics.items()},
            "candidate_pools": {
                mode: payload["candidate_pool"] for mode, payload in metrics.items()
            },
            "lexical_rescues": metrics[MODES[1]]["lexical_rescues"],
            "cross_encoder_conversion": metrics[MODES[1]]["cross_encoder_conversion"],
            "three_document": {
                mode: payload["three_document"] for mode, payload in metrics.items()
            },
            "three_document_effect": compare_three_document(
                metrics[MODES[0]]["three_document"]["traces"],
                metrics[MODES[1]]["three_document"]["traces"],
            ),
            "three_document_b_failures": three_failures,
            "retrieval_regressions": regressions,
            "selection": decision,
            "sol_used_for_retrieval_selection": False,
            "no_post_result_tuning": True,
        }
        record.selection_reason = decision["reason"]
        record.locked_at = datetime.now(UTC)
        prepared = [
            {
                "case_id": case.case_id,
                "pipeline_a_top5": next(
                    item["top5"]
                    for item in metrics[MODES[0]]["cases"]
                    if item["case_id"] == case.case_id
                ),
                "pipeline_b_top5": next(
                    item["top5"]
                    for item in metrics[MODES[1]]["cases"]
                    if item["case_id"] == case.case_id
                ),
                "timing": next(
                    item["timing"]
                    for item in metrics[MODES[1]]["cases"]
                    if item["case_id"] == case.case_id
                ),
                "shared_query_embedding": True,
                "embedding_cache_hit": next(
                    item["embedding_cache_hit"]
                    for item in metrics[MODES[1]]["cases"]
                    if item["case_id"] == case.case_id
                ),
            }
            for case in CASES
        ]
        e2e = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        e2e.prepared_cases = prepared
        e2e.prepared_at = datetime.now(UTC)
        e2e.preparation_usage = metrics[MODES[1]]["usage"]
        self.session.commit()
        self.freeze_architecture()
        return self.status()

    def freeze_architecture(self) -> RetrievalArchitectureRecord:
        record = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        if record is None or record.locked_at is None or not record.selected_retrieval_mode:
            raise ValueError("retrieval selection must be locked before architecture freeze")
        existing = self.session.get(RetrievalArchitectureRecord, ARCHITECTURE_ID)
        if existing:
            if existing.selected_retriever != record.selected_retrieval_mode:
                raise ValueError("frozen v1 retriever architecture is immutable")
            return existing
        selected = record.selected_retrieval_mode
        configuration = {
            "architecture_id": ARCHITECTURE_ID,
            "selected_retriever": selected,
            "embedding": {
                "provider": "openai-compatible",
                "model": "text-embedding-3-small",
                "dimension": 64,
                "version": "1",
            },
            "dense": {
                "backend": "pgvector_cosine",
                "threshold": 0.28,
                "candidate_depth": DENSE_DEPTH,
            },
            "cross_encoder": {
                "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
                "revision": RERANKER_REVISION,
            },
            "final_top_k": FINAL_TOP_K,
            "security_filtering_order": (
                "tenant/ACL/active-version filtering precedes Dense and BM25 candidate branches"
            ),
            "selection_benchmark_id": DATASET_ID,
            "immutable": True,
            "no_further_v1_retrieval_tuning": True,
        }
        if selected == MODES[1]:
            configuration["bm25"] = CONFIGURATION["bm25"]
            configuration["rrf"] = {"k": RRF_K, "candidate_union_limit": UNION_LIMIT}
            configuration["bm25_candidate_depth"] = BM25_DEPTH
        payload = RetrievalArchitectureRecord(
            architecture_id=ARCHITECTURE_ID,
            selected_retriever=selected,
            configuration=configuration,
            selection_policy=REPLICATION_POLICY,
            dataset_id=DATASET_ID,
            dataset_hash=DATASET_HASH,
            retrieval_metrics=record.calibration_metrics or {},
            immutable=True,
        )
        self.session.add(payload)
        self.session.commit()
        return payload

    def judge_preflight(self) -> dict[str, Any]:
        e2e = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        if not e2e or not e2e.prepared_cases:
            raise ValueError("replication retrieval traces must be frozen before Sol preflight")
        keys_a: list[str] = []
        keys_b: list[str] = []
        for case, prepared in zip(CASES, e2e.prepared_cases, strict=True):
            for collector, field in ((keys_a, "pipeline_a_top5"), (keys_b, "pipeline_b_top5")):
                collector.append(
                    gate_cache_key(
                        case.question,
                        RerankerEndToEndBenchmark._gate_evidence(prepared[field]),
                        provider="openai",
                        model=SOL_MODEL,
                        gate_version="1",
                        prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
                    )[0]
                )
        unique_a = set(keys_a)
        unique_b = set(keys_b)
        unique_all = unique_a | unique_b
        matches_a = sum(
            self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in unique_a
        )
        matches_b = sum(
            self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in unique_b
        )
        matches_all = sum(
            self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in unique_all
        )
        identical = sum(left == right for left, right in zip(keys_a, keys_b, strict=True))
        current = self._historical_judge_calls()
        missing_unique = max(len(unique_all) - matches_all, 0)
        preflight = {
            "current_cumulative_judge_calls": current,
            "configured_ceiling": self.settings.max_external_judge_calls,
            "existing_a_cache_matches": matches_a,
            "existing_b_cache_matches": matches_b,
            "identical_a_b_gate_inputs": identical,
            "unique_a_inputs": len(unique_a),
            "unique_b_inputs": len(unique_b),
            "shared_identical_evidence_inputs": identical,
            "new_unique_calls_required": missing_unique,
            "maximum_ending_judge_usage": current + missing_unique,
            "retrieval_sol_calls": 0,
        }
        e2e.judge_preflight = preflight
        self.session.commit()
        return preflight

    def execute(self) -> dict[str, Any]:
        record = self.initialize()
        e2e = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        if record.locked_at is None or not e2e or not e2e.prepared_cases:
            raise ValueError("retrieval lock and replication traces are required before Sol")
        if e2e.execution_started_at is not None:
            raise ValueError("replication Sol evaluation is one-shot")
        preflight = self.judge_preflight()
        if (
            not self.settings.allow_external_judge_calls
            or not self.settings.effective_judge_api_key
        ):
            raise ValueError("hosted judge authorization is incomplete")
        if preflight["maximum_ending_judge_usage"] > preflight["configured_ceiling"]:
            raise ValueError("replication Sol calls exceed the judge-call ceiling")
        e2e.execution_started_at = datetime.now(UTC)
        self.session.commit()
        limit = preflight["new_unique_calls_required"]
        gate = (
            self.gate_factory(SOL_MODEL, limit)
            if self.gate_factory
            else CachedAnswerabilityGate(
                self.session,
                ExternalJudgeCallLimitGate(
                    OpenAICompatibleAnswerabilityGate(
                        api_key=self.settings.effective_judge_api_key or "",
                        model=SOL_MODEL,
                        base_url=self.settings.judge_base_url,
                        gate_version="1",
                        prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
                        provider_name="openai",
                    ),
                    limit,
                ),
            )
        )
        provider = self._embedding_provider()
        results: dict[str, list[CaseResult]] = {mode: [] for mode in MODES}
        unauthorized = 0
        for case, prepared in zip(CASES, e2e.prepared_cases or [], strict=True):
            evaluation = case.as_evaluation()
            for mode, field in zip(MODES, ("pipeline_a_top5", "pipeline_b_top5"), strict=True):
                retrieved = [_result(item) for item in prepared[field]]
                if case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                    unauthorized += sum(
                        item.document_id in set(case.forbidden_document_ids)
                        for item in retrieved
                    )
                timing = RetrievalTiming(
                    query_embedding_latency_ms=prepared["timing"]["query_embedding_ms"],
                    embedding_cache_lookup_latency_ms=prepared["timing"]["cache_lookup_ms"],
                    vector_search_latency_ms=prepared["timing"]["dense_ms"],
                    acl_filter_latency_ms=0.0,
                    query_embedding_cache_hit=bool(prepared.get("embedding_cache_hit")),
                    external_embedding_calls=0,
                )
                rag = RagService(
                    self.session,
                    FixedResultRetriever(provider, retrieved, timing),
                    ContextBuilder(1200),
                    ExtractiveGenerationProvider(),
                    gate,
                    supporting_context_only=True,
                )
                results[mode].append(EvaluationRunner(rag).run_case(evaluation, 5, 0.28))
        if unauthorized:
            raise RuntimeError("unauthorized content reached Sol")
        reporter = EvaluationRunner(rag)
        reports = {mode: reporter._report(tuple(rows)) for mode, rows in results.items()}
        analysis = self._holdout_analysis(results)
        e2e_regressions = self._e2e_regressions(results)
        analysis["end_to_end_regressions"] = e2e_regressions
        for mode in MODES:
            report = reports[mode]
            usage = self._e2e_usage(mode, report.cases)
            usage["official_measured_cost_usd"] = official_token_cost(
                model=SOL_MODEL,
                input_tokens=usage["sol_input_tokens"],
                output_tokens=usage["sol_output_tokens"],
            )
            usage["official_pricing"] = OFFICIAL_PRICING["sol"]
            self.session.add(
                EndToEndBenchmarkRunRecord(
                    dataset_id=DATASET_ID,
                    mode=mode,
                    metrics=report.metrics,
                    retrieval_metrics=analysis["retrieval"][mode],
                    category_metrics=report.category_metrics,
                    case_results=[asdict(item) for item in report.cases],
                    latency=self._e2e_latency(mode, report.cases, e2e.prepared_cases or []),
                    usage=usage,
                )
            )
        e2e = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        e2e.paired_transitions = analysis["transitions"]
        e2e.conversion_analysis = analysis
        e2e.regression_analysis = {
            "retrieval": (record.calibration_metrics or {}).get("retrieval_regressions", []),
            "end_to_end": e2e_regressions,
        }
        e2e.production_retriever_status = record.selected_retrieval_mode
        e2e.completed_at = datetime.now(UTC)
        self.session.commit()
        return self.status()

    @staticmethod
    def _latency_field(case: CaseResult | dict[str, Any], name: str) -> Any:
        if isinstance(case, dict):
            return case.get(name)
        return getattr(case, name)

    @staticmethod
    def _e2e_latency(
        mode: str, cases: tuple[CaseResult, ...] | list[Any], prepared: list[dict[str, Any]]
    ) -> dict[str, Any]:
        totals: list[float] = []
        live_totals: list[float] = []
        live_judge: list[float] = []
        cached_judge: list[float] = []
        generation: list[float] = []
        for case, item in zip(cases, prepared, strict=True):
            timing = item["timing"]
            retrieval = (
                timing["query_embedding_ms"]
                + timing["cache_lookup_ms"]
                + timing["dense_ms"]
                + (timing["bm25_ms"] + timing["rrf_ms"] if mode == MODES[1] else 0.0)
                + (timing["reranker_b_ms"] if mode == MODES[1] else timing["reranker_a_ms"])
            )
            judge_ms = float(
                HybridRerankerReplicationBenchmark._latency_field(
                    case, "answerability_judge_latency_ms"
                )
                or 0.0
            )
            generation_ms = float(
                HybridRerankerReplicationBenchmark._latency_field(case, "generation_latency_ms")
                or 0.0
            )
            total_ms = float(
                HybridRerankerReplicationBenchmark._latency_field(case, "total_latency_ms") or 0.0
            )
            cache_hit = HybridRerankerReplicationBenchmark._latency_field(case, "gate_cache_hit")
            total = total_ms + retrieval
            totals.append(total)
            generation.append(generation_ms)
            if cache_hit is True:
                cached_judge.append(judge_ms)
            else:
                live_judge.append(judge_ms)
                live_totals.append(total)

        def stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {"count": 0.0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
            return {
                "count": float(len(values)),
                "mean_ms": mean(values),
                "p50_ms": median(values),
                "p95_ms": _percentile(values, 0.95),
            }

        live = stats(live_judge)
        cached = stats(cached_judge)
        live_total = stats(live_totals)
        return {
            "total_mean_ms": mean(totals) if totals else 0.0,
            "total_p50_ms": median(totals) if totals else 0.0,
            "total_p95_ms": _percentile(totals, 0.95) if totals else 0.0,
            "judge_mean_ms": mean(live_judge) if live_judge else 0.0,
            "generation_mean_ms": mean(generation) if generation else 0.0,
            "live_judge_count": live["count"],
            "live_judge_mean_ms": live["mean_ms"],
            "live_judge_p50_ms": live["p50_ms"],
            "live_judge_p95_ms": live["p95_ms"],
            "cached_replay_judge_count": cached["count"],
            "cached_replay_judge_mean_ms": cached["mean_ms"],
            "cached_replay_judge_p50_ms": cached["p50_ms"],
            "live_total_mean_ms": live_total["mean_ms"],
            "live_total_p50_ms": live_total["p50_ms"],
            "cached_replay_excluded_from_live_judge_mean": 1.0,
        }

    @staticmethod
    def _e2e_regressions(results: dict[str, list[CaseResult]]) -> list[dict[str, Any]]:
        a = {case.case_id: case for case in results[MODES[0]]}
        b = {case.case_id: case for case in results[MODES[1]]}
        found: list[dict[str, Any]] = []
        for case_id, left in a.items():
            right = b[case_id]
            left_b = FrozenJudgeEndToEndBenchmark._behavior(left)
            right_b = FrozenJudgeEndToEndBenchmark._behavior(right)
            if left_b == "CORRECT_ANSWER" and right_b == "INCORRECT_ABSTENTION":
                found.append(
                    {
                        "case_id": case_id,
                        "effect": "A correct answer → B incorrect abstention",
                    }
                )
            if left_b == "CORRECT_ABSTENTION" and right_b == "UNSUPPORTED_ANSWER":
                found.append(
                    {
                        "case_id": case_id,
                        "effect": "A correct abstention → B unsupported answer",
                    }
                )
            left_safe = left_b in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"}
            if left_safe and right_b not in {
                "CORRECT_ANSWER",
                "CORRECT_ABSTENTION",
                "INCORRECT_ABSTENTION",
            }:
                found.append({"case_id": case_id, "effect": f"A safe result → B {right_b}"})
        return found

    def _partition_runs(self, partition: str) -> dict[str, RetrievalBenchmarkRunRecord]:
        with self._bind_dataset():
            return super()._partition_runs(partition)

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        payload: dict[str, Any] = {
            "dataset_id": DATASET_ID,
            "dataset_hash": DATASET_HASH,
            "case_count": len(CASES),
            "category_distribution": dict(sorted(Counter(item.category for item in CASES).items())),
            "maximum_prior_overlap": MAXIMUM_PRIOR_OVERLAP,
            "generation_method": GENERATION_METHOD,
            "split_seed": SPLIT_SEED,
            "split_identity": SPLIT_IDENTITY,
            "no_holdout": True,
            "configuration": CONFIGURATION,
            "selection_policy": REPLICATION_POLICY,
            "production_judge_status": "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1",
            "embedding_preflight": self.embedding_preflight(),
            "selected_mode": None,
            "locked": False,
            "retrieval_completed": False,
            "e2e_completed": False,
            "architecture_frozen": False,
            "production_retriever_status": "DENSE_CROSS_ENCODER_RERANK",
            "enterprise_rag_v1_retrieval_status": "UNFROZEN",
        }
        if not record:
            return payload
        self._verify(record)
        payload.update(
            {
                "selected_mode": record.selected_retrieval_mode,
                "selection_reason": record.selection_reason,
                "locked": record.locked_at is not None,
                "locked_at": record.locked_at,
                "frozen_at": record.frozen_at,
                "replication_metrics": record.calibration_metrics,
            }
        )
        runs: dict[str, dict[str, Any]] = {}
        for run in self.session.scalars(
            select(RetrievalBenchmarkRunRecord).where(
                RetrievalBenchmarkRunRecord.dataset_id == DATASET_ID
            )
        ).all():
            runs[run.retrieval_mode] = {
                "metrics": run.metrics,
                "category_metrics": run.category_metrics,
                "latency": run.latency,
                "usage": run.usage,
                "branch_contribution": run.branch_contribution,
                **({"cases": run.case_results} if include_cases else {}),
            }
        payload["retrieval"] = runs or None
        payload["retrieval_completed"] = bool(runs)
        architecture = self.session.get(RetrievalArchitectureRecord, ARCHITECTURE_ID)
        if architecture:
            payload["architecture_frozen"] = True
            payload["architecture"] = {
                "architecture_id": architecture.architecture_id,
                "selected_retriever": architecture.selected_retriever,
                "configuration": architecture.configuration,
                "frozen_at": architecture.frozen_at,
                "immutable": architecture.immutable,
            }
            payload["production_retriever_status"] = architecture.selected_retriever
            payload["enterprise_rag_v1_retrieval_status"] = "FROZEN"
        e2e = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        if e2e:
            payload["e2e_completed"] = e2e.completed_at is not None
            payload["judge_preflight"] = e2e.judge_preflight
            payload["conversion_analysis"] = e2e.conversion_analysis
            payload["regressions"] = e2e.regression_analysis
            payload["paired_transitions"] = e2e.paired_transitions
            if e2e.production_retriever_status:
                payload["production_retriever_status"] = e2e.production_retriever_status
            e2e_runs = {}
            for run in self.session.scalars(
                select(EndToEndBenchmarkRunRecord).where(
                    EndToEndBenchmarkRunRecord.dataset_id == DATASET_ID
                )
            ).all():
                latency = run.latency
                if run.case_results and e2e.prepared_cases:
                    latency = self._e2e_latency(run.mode, run.case_results, e2e.prepared_cases)
                e2e_runs[run.mode] = {
                    "metrics": run.metrics,
                    "retrieval_metrics": run.retrieval_metrics,
                    "category_metrics": run.category_metrics,
                    "latency": latency,
                    "usage": run.usage,
                    **({"cases": run.case_results} if include_cases else {}),
                }
            payload["end_to_end"] = e2e_runs or None
        return payload


FINAL_DATASET_ID = "acmeai-enterprise-rag-v1-final-eval"
FINAL_DATASET_PATH = Path("data/eval/acmeai_enterprise_rag_v1_final_eval.json")
FINAL_DATASET_HASH = hashlib.sha256(FINAL_DATASET_PATH.read_bytes()).hexdigest()
RELEASE_ARCHITECTURE_ID = "enterprise-rag-workbench-v1"
FINAL_GENERATION_METHOD = "manual-corpus-grounded-v1"
FINAL_OVERLAP_CEILING = 0.5
EXPECTED_FINAL_DISTRIBUTION = {
    "acl_sensitive": 4,
    "exact_identifier": 6,
    "multidoc_three": 20,
    "multidoc_two": 16,
    "near_duplicate": 6,
    "partial_no_answer": 4,
    "prompt_injection": 4,
    "semantic_paraphrase": 4,
    "single_document": 10,
    "version_region": 6,
}
HISTORICAL_OVERLAP_IDENTITIES = {
    "replication": 0.4166666666666667,
    "hybrid_reranker": 0.4,
    "sol_judge": 0.4444444444444444,
    "judge_e2e": 0.47058823529411764,
    "reranker_e2e": 0.391304347826087,
}
HISTORICAL_SOL_MEAN_INPUT_TOKENS = 103061 / 126
HISTORICAL_SOL_MEAN_OUTPUT_TOKENS = 10776 / 126
RESTRICTED_MARKERS = ("HR-COMP-900", "HR-BEN-771", "ignore previous instructions")


def load_final_cases() -> tuple[HybridRerankerCase, ...]:
    payload = json.loads(FINAL_DATASET_PATH.read_text())
    return tuple(HybridRerankerCase.model_validate(item) for item in payload["cases"])


FINAL_CASES = load_final_cases()
FINAL_SPLIT_IDENTITY = hashlib.sha256(
    f"final-v1-no-split:{'|'.join(item.case_id for item in FINAL_CASES)}".encode()
).hexdigest()


def _token_terms(question: str) -> set[str]:
    return set(re.compile(r"[a-z0-9]+").findall(question.casefold()))


def final_dataset_overlap_report() -> dict[str, Any]:
    maximum = 0.0
    closest: dict[str, Any] | None = None
    for path in Path("data/eval").glob("*.json"):
        if path == FINAL_DATASET_PATH:
            continue
        for previous in json.loads(path.read_text()).get("cases", []):
            right = _token_terms(previous["question"])
            for case in FINAL_CASES:
                left = _token_terms(case.question)
                score = len(left & right) / len(left | right)
                if score > maximum:
                    maximum = score
                    closest = {
                        "final_case_id": case.case_id,
                        "prior_dataset": path.name,
                        "prior_case_id": previous.get("case_id"),
                        "overlap": score,
                    }
    return {
        "maximum_normalized_overlap": maximum,
        "closest_prior_case": closest,
        "overlap_threshold": FINAL_OVERLAP_CEILING,
        "pass": maximum < FINAL_OVERLAP_CEILING,
    }


def historical_overlap_identities_hold() -> bool:
    from rag_workbench.experiments.hybrid_reranker_benchmark import (
        maximum_prior_dataset_overlap as hybrid_overlap,
    )
    from rag_workbench.experiments.judge_e2e_benchmark import (
        maximum_prior_dataset_overlap as judge_overlap,
    )
    from rag_workbench.experiments.reranker_e2e_benchmark import (
        maximum_prior_dataset_overlap as reranker_overlap,
    )
    from rag_workbench.experiments.sol_judge_e2e_benchmark import (
        maximum_prior_dataset_overlap as sol_overlap,
    )

    return (
        abs(maximum_prior_dataset_overlap() - HISTORICAL_OVERLAP_IDENTITIES["replication"])
        < 1e-12
        and abs(hybrid_overlap() - HISTORICAL_OVERLAP_IDENTITIES["hybrid_reranker"]) < 1e-12
        and abs(sol_overlap() - HISTORICAL_OVERLAP_IDENTITIES["sol_judge"]) < 1e-12
        and abs(judge_overlap() - HISTORICAL_OVERLAP_IDENTITIES["judge_e2e"]) < 1e-12
        and abs(reranker_overlap() - HISTORICAL_OVERLAP_IDENTITIES["reranker_e2e"]) < 1e-12
    )


def classify_final_root_cause(
    case: HybridRerankerCase, retrieval: dict[str, Any], result: CaseResult
) -> str | None:
    behavior = FrozenJudgeEndToEndBenchmark._behavior(result)
    if case.category == "prompt_injection" and result.security_passed is False:
        return "PROMPT_INJECTION_FAILURE"
    if "acl" in case.security_checks and retrieval["metrics"]["unauthorized_result_exposure"]:
        return "ACL_FAILURE"
    if result.version_correct == 0.0 and behavior == "UNSUPPORTED_ANSWER":
        return "VERSION_FAILURE"
    if behavior in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"}:
        return None
    if not case.expected_answerability:
        if behavior == "UNSUPPORTED_ANSWER":
            return "EVIDENCE_GATE_FALSE_POSITIVE"
        return "UNKNOWN"
    if not retrieval.get("top5_complete"):
        taxonomy = retrieval.get("failure_taxonomy")
        mapping = {
            "BOTH_BRANCHES_MISS": "CANDIDATE_GENERATION_MISS",
            "DENSE_CANDIDATE_GENERATION_MISS": "CANDIDATE_GENERATION_MISS",
            "RRF_TRUNCATION_LOSS": "CANDIDATE_GENERATION_MISS",
            "CROSS_ENCODER_FAILED_TO_PROMOTE": "CROSS_ENCODER_FAILED_TO_PROMOTE",
            "CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE": "CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE",
            "RANKING_OUTSIDE_TOP5": "RANKING_OUTSIDE_TOP5",
            "THREE_DOCUMENT_COVERAGE_FAILURE": "RANKING_OUTSIDE_TOP5",
            "LEXICAL_CANDIDATE_RESCUE": "CROSS_ENCODER_FAILED_TO_PROMOTE",
        }
        return mapping.get(taxonomy, "CANDIDATE_GENERATION_MISS")
    judge_answerable = bool((result.answerability_result or {}).get("answerable"))
    if not judge_answerable:
        return "EVIDENCE_GATE_FALSE_NEGATIVE"
    if result.supporting_context_loss:
        return "SUPPORTING_CONTEXT_LOSS"
    supporting = set(result.supporting_chunk_ids)
    retrieved = set(result.retrieved_chunk_ids)
    if supporting and not supporting <= retrieved:
        return "INVALID_SUPPORTING_ID"
    if result.status != "answered":
        return "GENERATION_FAILURE"
    if result.citation_correctness == 0.0:
        return "CITATION_FAILURE"
    return "UNKNOWN"


class FinalV1Benchmark:
    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        *,
        embedding_provider_factory: Any | None = None,
        reranker_factory: Any | None = None,
        gate_factory: Any | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embedding_provider_factory = embedding_provider_factory
        self.reranker_factory = reranker_factory or (
            lambda: CrossEncoderReranker(device="cpu", resolved_revision=RERANKER_REVISION)
        )
        self.gate_factory = gate_factory
        self._verify_dataset()

    def _verify_dataset(self) -> None:
        payload = json.loads(FINAL_DATASET_PATH.read_text())
        if payload.get("dataset_id") != FINAL_DATASET_ID or len(FINAL_CASES) != 80:
            raise ValueError("final v1 dataset identity changed")
        if hashlib.sha256(FINAL_DATASET_PATH.read_bytes()).hexdigest() != FINAL_DATASET_HASH:
            raise ValueError("final v1 dataset hash changed")
        distribution = dict(sorted(Counter(item.category for item in FINAL_CASES).items()))
        if distribution != EXPECTED_FINAL_DISTRIBUTION:
            raise ValueError("final v1 category distribution changed")
        if len({item.case_id for item in FINAL_CASES}) != 80:
            raise ValueError("final v1 case IDs are not unique")
        three = [item for item in FINAL_CASES if item.category == "multidoc_three"]
        if any(
            len(set(item.required_document_ids)) != 3 or len(item.required_fact_ids) != 3
            for item in three
        ):
            raise ValueError("three-document cases must require three distinct sources")
        if any(
            item.category == "prompt_injection" and "prompt_injection" not in item.security_checks
            for item in FINAL_CASES
        ):
            raise ValueError("prompt-injection cases must carry the security label")
        computed = hashlib.sha256(
            json.dumps(evidence_gate_schema(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if computed != SCHEMA_IDENTITY:
            raise ValueError("evidence-sufficiency-v1 schema identity changed")

    def previous_replication_state(self) -> dict[str, Any]:
        record = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        architecture = self.session.get(RetrievalArchitectureRecord, ARCHITECTURE_ID)
        e2e = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        production = [
            item
            for item in self.session.scalars(select(RetrievalArchitectureRecord)).all()
            if item.architecture_id == ARCHITECTURE_ID
        ]
        complete = bool(
            record
            and record.locked_at
            and record.selected_retrieval_mode in MODES
            and e2e
            and e2e.completed_at
            and architecture
            and architecture.immutable
            and architecture.selected_retriever == record.selected_retrieval_mode
            and len(production) == 1
        )
        return {
            "replication_verdict": "COMPLETE" if complete else "PARTIAL",
            "final_selected_retriever": (
                architecture.selected_retriever if architecture else None
            ),
            "retrieval_architecture_frozen": bool(architecture and architecture.immutable),
            "replication_rerun": False,
            "complete": complete,
        }

    def production_retriever(self) -> str:
        state = self.previous_replication_state()
        if not state["complete"] or state["final_selected_retriever"] not in MODES:
            raise ValueError("PARTIAL: frozen v1 retrieval architecture is incomplete")
        return str(state["final_selected_retriever"])

    def _corpus_identity(self) -> str:
        rows = self.session.execute(
            select(Document.document_id, DocumentVersion.version, DocumentVersion.content_hash)
            .join(DocumentVersion, DocumentVersion.document_fk == Document.id)
            .where(DocumentVersion.is_active.is_(True))
            .order_by(Document.document_id, DocumentVersion.version)
        ).all()
        return hashlib.sha256("|".join(":".join(row) for row in rows).encode()).hexdigest()

    def _embedding_provider(self) -> Any:
        if self.embedding_provider_factory:
            return self.embedding_provider_factory()
        return OpenAICompatibleEmbeddingProvider(
            api_key=self.settings.embedding_api_key or "",
            model="text-embedding-3-small",
            dimension=64,
            base_url=self.settings.embedding_base_url,
            version="1",
            provider_name="openai-compatible",
        )

    def _embedding_calls(self) -> int:
        return int(
            self.session.scalar(
                select(func.count())
                .select_from(QueryEmbeddingCacheRecord)
                .where(
                    QueryEmbeddingCacheRecord.embedding_provider == "openai-compatible",
                    QueryEmbeddingCacheRecord.embedding_model == "text-embedding-3-small",
                    QueryEmbeddingCacheRecord.embedding_version == "1",
                    QueryEmbeddingCacheRecord.embedding_dimension == 64,
                )
            )
            or 0
        )

    def _historical_judge_calls(self) -> int:
        historical = sum(
            item.external_judge_calls or 0
            for item in self.session.scalars(select(ExperimentRunRecord)).all()
        )
        specialized = 0
        for item in self.session.scalars(select(EndToEndBenchmarkRunRecord)).all():
            usage = item.usage or {}
            specialized += usage.get("new_luna_judge_calls", 0)
            specialized += usage.get("new_sol_judge_calls", 0)
        return historical + specialized

    def embedding_preflight(self) -> dict[str, Any]:
        questions = tuple(dict.fromkeys(case.question for case in FINAL_CASES))
        keys = [
            query_embedding_cache_key(
                question,
                provider="openai-compatible",
                model="text-embedding-3-small",
                version="1",
                dimension=64,
            )
            for question in questions
        ]
        matches = sum(self.session.get(QueryEmbeddingCacheRecord, key) is not None for key in keys)
        current = self._embedding_calls()
        missing = len(keys) - matches
        return {
            "current_cumulative_embedding_calls": current,
            "configured_ceiling": self.settings.max_external_embedding_calls,
            "new_unique_queries": len(questions),
            "existing_cache_matches": matches,
            "missing_embeddings": missing,
            "authorized_ceiling": current + missing,
            "expected_cumulative_ending_usage": current + missing,
        }

    def initialize(self) -> EndToEndBenchmarkRecord:
        state = self.previous_replication_state()
        if not state["complete"]:
            raise ValueError("PARTIAL: previous Hybrid+Cross-Encoder replication is incomplete")
        if not historical_overlap_identities_hold():
            raise ValueError("PARTIAL: historical overlap identities changed")
        overlap = final_dataset_overlap_report()
        if not overlap["pass"]:
            raise ValueError("PARTIAL: final dataset overlap exceeds the accepted threshold")
        if self._corpus_identity() != CORPUS_IDENTITY:
            raise ValueError("corpus identity changed")
        if not self.session.scalar(
            select(Chunk.id).where(Chunk.index_identity == SEMANTIC_INDEX_IDENTITY).limit(1)
        ):
            raise ValueError("frozen semantic index is unavailable")
        existing = self.session.get(EndToEndBenchmarkRecord, FINAL_DATASET_ID)
        if existing:
            if existing.dataset_hash != FINAL_DATASET_HASH:
                raise ValueError("frozen final dataset identity changed")
            return existing
        mode = self.production_retriever()
        record = EndToEndBenchmarkRecord(
            dataset_id=FINAL_DATASET_ID,
            dataset_hash=FINAL_DATASET_HASH,
            case_ids=[item.case_id for item in FINAL_CASES],
            category_distribution=EXPECTED_FINAL_DISTRIBUTION,
            generation_method=FINAL_GENERATION_METHOD,
            maximum_prior_overlap=overlap["maximum_normalized_overlap"],
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=CORPUS_IDENTITY,
            reranker_revision=RERANKER_REVISION,
            pipeline_a_configuration={
                **CONFIGURATION,
                "mode": mode,
                "judge": {
                    "id": "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1",
                    "model": SOL_MODEL,
                    "prompt": EVIDENCE_GATE_PROMPT_VERSION,
                    "schema_identity": SCHEMA_IDENTITY,
                    "request_settings": hosted_judge_request_settings(
                        EVIDENCE_GATE_PROMPT_VERSION
                    ),
                    "supporting_context_only": True,
                },
            },
            pipeline_b_configuration={"single_production_pipeline": True, "mode": mode},
            embedding_preflight=self.embedding_preflight(),
            production_retriever_status=mode,
        )
        self.session.add(record)
        if self.session.get(RetrievalBenchmarkRecord, FINAL_DATASET_ID) is None:
            self.session.add(
                RetrievalBenchmarkRecord(
                    dataset_id=FINAL_DATASET_ID,
                    dataset_hash=FINAL_DATASET_HASH,
                    split_identity=FINAL_SPLIT_IDENTITY,
                    split_seed=0,
                    calibration_case_ids=[item.case_id for item in FINAL_CASES],
                    holdout_case_ids=[],
                    semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
                    corpus_identity=CORPUS_IDENTITY,
                    retrieval_configuration={**CONFIGURATION, "mode": mode},
                    selection_policy={
                        "no_architecture_selection": True,
                        "single_production_pipeline": True,
                        "production_retriever": mode,
                    },
                    selected_retrieval_mode=mode,
                )
            )
        self.session.commit()
        return record

    def retrieve(self) -> dict[str, Any]:
        record = self.initialize()
        if record.prepared_at is not None:
            raise ValueError("final v1 retrieval is one-shot")
        retrieval_lock = self.session.get(RetrievalBenchmarkRecord, FINAL_DATASET_ID)
        if retrieval_lock and retrieval_lock.locked_at is not None:
            raise ValueError("final v1 retrieval is already locked")
        preflight = self.embedding_preflight()
        if not self.settings.allow_external_calls or not self.settings.embedding_api_key:
            raise ValueError("external embedding authorization is incomplete")
        if preflight["expected_cumulative_ending_usage"] > preflight["configured_ceiling"]:
            raise ValueError("final dataset exceeds the embedding-call ceiling")
        mode = self.production_retriever()
        judge_before = self._historical_judge_calls()
        analysis = self._run_production(mode)
        if self._historical_judge_calls() != judge_before:
            raise RuntimeError("Sol was used during final retrieval")
        prepared = [
            {
                "case_id": case.case_id,
                "top5": next(
                    item["top5"] for item in analysis["cases"] if item["case_id"] == case.case_id
                ),
                "timing": next(
                    item["timing"] for item in analysis["cases"] if item["case_id"] == case.case_id
                ),
                "embedding_cache_hit": next(
                    item["embedding_cache_hit"]
                    for item in analysis["cases"]
                    if item["case_id"] == case.case_id
                ),
                "retrieval_row": next(
                    item for item in analysis["cases"] if item["case_id"] == case.case_id
                ),
            }
            for case in FINAL_CASES
        ]
        record.prepared_cases = prepared
        record.prepared_at = datetime.now(UTC)
        record.preparation_usage = analysis["usage"]
        retrieval_lock = self.session.get(RetrievalBenchmarkRecord, FINAL_DATASET_ID)
        retrieval_lock.locked_at = datetime.now(UTC)
        retrieval_lock.calibration_metrics = {
            "mode": mode,
            "metrics": analysis["metrics"],
            "candidate_pool": analysis["candidate_pool"],
            "three_document": {
                key: value for key, value in analysis["three_document"].items() if key != "traces"
            },
        }
        self.session.add(
            RetrievalBenchmarkRunRecord(
                dataset_id=FINAL_DATASET_ID,
                partition="final",
                retrieval_mode=mode,
                metrics=analysis["metrics"],
                category_metrics=analysis["category_metrics"],
                case_results=[self._runtime_retrieval_trace(item) for item in analysis["cases"]],
                branch_contribution={
                    "candidate_pool": analysis["candidate_pool"],
                    "three_document": {
                        key: value
                        for key, value in analysis["three_document"].items()
                        if key != "traces"
                    },
                    **(
                        {
                            "lexical_rescues": analysis.get("lexical_rescues"),
                            "cross_encoder_conversion": analysis.get("cross_encoder_conversion"),
                        }
                        if mode == MODES[1]
                        else {}
                    ),
                },
                latency=analysis["latency"],
                usage=analysis["usage"],
            )
        )
        self.session.commit()
        return self.status()

    def _runtime_retrieval_trace(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "question_id": payload["case_id"],
            "category": payload["category"],
            "dense_candidates": [
                {key: item[key] for key in item if key != "text"}
                for item in payload["dense_candidates"]
            ],
            "bm25_candidates": [
                {key: item[key] for key in item if key != "text"}
                for item in payload["bm25_candidates"]
            ],
            "rrf_union": [
                {key: item[key] for key in item if key != "text"} for item in payload["rrf_union"]
            ],
            "cross_encoder_pairs": payload["cross_encoder_pairs"],
            "top5": [{key: item[key] for key in item if key != "text"} for item in payload["top5"]],
            "required_ce_ranks": payload["required_ce_ranks"],
            "timing": payload["timing"],
            "embedding_cache_hit": payload["embedding_cache_hit"],
            "evaluation": {
                "top5_complete": payload["top5_complete"],
                "failure_taxonomy": payload["failure_taxonomy"],
            },
        }

    def _run_production(self, mode: str) -> dict[str, Any]:
        provider = self._embedding_provider()
        dense_retriever = Retriever(self.session, provider, SEMANTIC_INDEX_IDENTITY)
        bm25 = BM25Retriever(
            self.session,
            index_identity=SEMANTIC_INDEX_IDENTITY,
            embedding_provider="openai-compatible",
            embedding_model="text-embedding-3-small",
            embedding_version="1",
            embedding_dimension=64,
            config=BM25Config(),
        )
        reranker: Reranker = self.reranker_factory()
        if reranker.resolved_revision != RERANKER_REVISION:
            raise ValueError("local reranker revision differs from frozen revision")
        rows: list[dict[str, Any]] = []
        usage = {
            "new_query_embedding_calls": 0,
            "embedding_tokens": 0,
            "query_cache_hits": 0,
            "query_cache_misses": 0,
            "document_embedding_calls": 0,
            "external_reranker_calls": 0,
            "cross_encoder_pairs": 0,
            "bm25_index_size_bytes": 0,
            "bm25_build_ms": 0.0,
            "unauthorized_chunks_to_cross_encoder": 0,
        }
        hybrid = mode == MODES[1]
        helper = HybridRerankerBenchmark.__new__(HybridRerankerBenchmark)
        for case in FINAL_CASES:
            retrieval = case.as_retrieval()
            embedding = dense_retriever.query_embedding_cache.get_or_embed(case.question)
            self.session.commit()
            usage["new_query_embedding_calls"] += embedding.external_calls
            usage["embedding_tokens"] += embedding.input_tokens
            usage["query_cache_hits"] += int(embedding.cache_hit)
            usage["query_cache_misses"] += int(not embedding.cache_hit)
            principal = _principal(case)
            dense_started = time.perf_counter()
            dense_candidates = dense_retriever.retrieve_with_embedding(
                embedding, top_k=DENSE_DEPTH, score_threshold=0.28, principal=principal
            )
            dense_ms = (time.perf_counter() - dense_started) * 1000
            bm25_candidates: list[Any] = []
            bm25_ms = 0.0
            fusion_ms = 0.0
            union_all = dense_candidates
            union = dense_candidates
            if hybrid:
                bm25_started = time.perf_counter()
                bm25_candidates = bm25.retrieve(
                    case.question, top_k=BM25_DEPTH, principal=principal
                )
                bm25_ms = (time.perf_counter() - bm25_started) * 1000
                usage["bm25_index_size_bytes"] = bm25.last_timing.index_size_bytes
                usage["bm25_build_ms"] = bm25.last_timing.index_build_latency_ms
                fusion_started = time.perf_counter()
                union_all = reciprocal_rank_fusion(
                    dense_candidates, bm25_candidates, top_k=10_000, rrf_k=RRF_K
                )
                union = union_all[:UNION_LIMIT]
                fusion_ms = (time.perf_counter() - fusion_started) * 1000
            forbidden = set(case.forbidden_document_ids)
            if case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                leaked = [
                    item
                    for item in [*dense_candidates, *bm25_candidates, *union]
                    if item.document_id in forbidden
                ]
                if leaked:
                    raise RuntimeError("unauthorized content reached Cross-Encoder")
            reranked = reranker.rerank(case.question, union)
            pairs = len(union)
            rerank_ms = reranker.last_inference_latency_ms + reranker.last_sort_latency_ms
            usage["cross_encoder_pairs"] += pairs
            top5 = [
                replace(
                    item.result,
                    rank=item.reranked_rank,
                    score=item.reranker_score,
                    retrieval_source=(
                        "hybrid_cross_encoder_rerank" if hybrid else "dense_cross_encoder_rerank"
                    ),
                )
                for item in reranked[:FINAL_TOP_K]
            ]
            timing = {
                "query_embedding_ms": embedding.embedding_latency_ms,
                "cache_lookup_ms": embedding.cache_lookup_latency_ms,
                "dense_ms": dense_ms,
                "bm25_ms": bm25_ms,
                "rrf_ms": fusion_ms,
                "reranker_a_ms": 0.0 if hybrid else rerank_ms,
                "reranker_b_ms": rerank_ms if hybrid else 0.0,
                "reranker_ms": rerank_ms,
            }
            rows.append(
                helper._case_payload(
                    retrieval,
                    dense_candidates,
                    bm25_candidates,
                    union,
                    top5,
                    reranked,
                    timing,
                    pairs,
                    embedding.cache_hit,
                    union_all=union_all,
                )
            )
        analysis = helper._mode_analysis(rows, mode)
        if hybrid:
            analysis["lexical_rescues"] = HybridRerankerBenchmark._lexical_rescues(rows)
            analysis["cross_encoder_conversion"] = HybridRerankerBenchmark._conversion(rows)
        analysis["usage"] = usage
        analysis["cases"] = rows
        return analysis

    def judge_preflight(self) -> dict[str, Any]:
        e2e = self.session.get(EndToEndBenchmarkRecord, FINAL_DATASET_ID)
        if not e2e or not e2e.prepared_cases:
            raise ValueError("final retrieval traces must be frozen before Sol preflight")
        keys: list[str] = []
        for case, prepared in zip(FINAL_CASES, e2e.prepared_cases, strict=True):
            keys.append(
                gate_cache_key(
                    case.question,
                    RerankerEndToEndBenchmark._gate_evidence(prepared["top5"]),
                    provider="openai",
                    model=SOL_MODEL,
                    gate_version="1",
                    prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
                )[0]
            )
        unique = set(keys)
        matches = sum(
            self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in unique
        )
        current = self._historical_judge_calls()
        missing = max(len(unique) - matches, 0)
        preflight = {
            "current_cumulative_judge_calls": current,
            "configured_ceiling": self.settings.max_external_judge_calls,
            "unique_final_inputs": len(unique),
            "existing_sol_cache_matches": matches,
            "new_unique_calls_required": missing,
            "authorized_ceiling": current + missing,
            "maximum_ending_judge_usage": current + missing,
            "historical_mean_sufficiency_v1_input_tokens": HISTORICAL_SOL_MEAN_INPUT_TOKENS,
            "historical_mean_sufficiency_v1_output_tokens": HISTORICAL_SOL_MEAN_OUTPUT_TOKENS,
            "projected_sol_cost_usd": official_token_cost(
                model=SOL_MODEL,
                input_tokens=int(HISTORICAL_SOL_MEAN_INPUT_TOKENS * missing),
                output_tokens=int(HISTORICAL_SOL_MEAN_OUTPUT_TOKENS * missing),
            ),
            "official_pricing": OFFICIAL_PRICING["sol"],
        }
        e2e.judge_preflight = preflight
        self.session.commit()
        return preflight

    def execute(self) -> dict[str, Any]:
        record = self.initialize()
        if not record.prepared_cases or record.prepared_at is None:
            raise ValueError("final retrieval traces are required before Sol")
        if record.execution_started_at is not None:
            raise ValueError("final v1 Sol evaluation is one-shot")
        preflight = self.judge_preflight()
        if (
            not self.settings.allow_external_judge_calls
            or not self.settings.effective_judge_api_key
        ):
            raise ValueError("hosted judge authorization is incomplete")
        if preflight["maximum_ending_judge_usage"] > preflight["configured_ceiling"]:
            raise ValueError("final Sol calls exceed the judge-call ceiling")
        record.execution_started_at = datetime.now(UTC)
        self.session.commit()
        mode = self.production_retriever()
        limit = preflight["new_unique_calls_required"]
        gate = (
            self.gate_factory(SOL_MODEL, limit)
            if self.gate_factory
            else CachedAnswerabilityGate(
                self.session,
                ExternalJudgeCallLimitGate(
                    OpenAICompatibleAnswerabilityGate(
                        api_key=self.settings.effective_judge_api_key or "",
                        model=SOL_MODEL,
                        base_url=self.settings.judge_base_url,
                        gate_version="1",
                        prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
                        provider_name="openai",
                    ),
                    limit,
                ),
            )
        )
        provider = self._embedding_provider()
        results: list[CaseResult] = []
        unauthorized = 0
        unauthorized_to_sol = 0
        for case, prepared in zip(FINAL_CASES, record.prepared_cases or [], strict=True):
            evaluation = case.as_evaluation()
            retrieved = [_result(item) for item in prepared["top5"]]
            if case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                leaked = sum(
                    item.document_id in set(case.forbidden_document_ids) for item in retrieved
                )
                unauthorized += leaked
                unauthorized_to_sol += leaked
            timing = RetrievalTiming(
                query_embedding_latency_ms=prepared["timing"]["query_embedding_ms"],
                embedding_cache_lookup_latency_ms=prepared["timing"]["cache_lookup_ms"],
                vector_search_latency_ms=prepared["timing"]["dense_ms"],
                acl_filter_latency_ms=0.0,
                query_embedding_cache_hit=bool(prepared.get("embedding_cache_hit")),
                external_embedding_calls=0,
            )
            rag = RagService(
                self.session,
                FixedResultRetriever(provider, retrieved, timing),
                ContextBuilder(1200),
                ExtractiveGenerationProvider(),
                gate,
                supporting_context_only=True,
            )
            results.append(EvaluationRunner(rag).run_case(evaluation, 5, 0.28))
        if unauthorized:
            raise RuntimeError("unauthorized content reached Sol")
        reporter = EvaluationRunner(rag)
        report = reporter._report(tuple(results))
        analysis = self._final_analysis(mode, report.cases, record.prepared_cases or [])
        usage = self._e2e_usage(report.cases)
        usage["official_measured_cost_usd"] = official_token_cost(
            model=SOL_MODEL,
            input_tokens=usage["sol_input_tokens"],
            output_tokens=usage["sol_output_tokens"],
            cached_input_tokens=usage.get("sol_cached_input_tokens", 0),
        )
        usage["official_pricing"] = OFFICIAL_PRICING["sol"]
        usage["unauthorized_chunks_to_sol"] = unauthorized_to_sol
        retrieval_usage = record.preparation_usage or {}
        usage["query_embedding_calls"] = retrieval_usage.get("new_query_embedding_calls", 0)
        usage["embedding_tokens"] = retrieval_usage.get("embedding_tokens", 0)
        self.session.add(
            EndToEndBenchmarkRunRecord(
                dataset_id=FINAL_DATASET_ID,
                mode=mode,
                metrics=report.metrics,
                retrieval_metrics=analysis["retrieval_metrics"],
                category_metrics=report.category_metrics,
                case_results=[self._runtime_e2e_trace(item) for item in report.cases],
                latency=self._final_latency(mode, report.cases, record.prepared_cases or []),
                usage=usage,
            )
        )
        record.conversion_analysis = analysis
        record.regression_analysis = analysis["failure_taxonomy"]
        record.production_retriever_status = mode
        record.completed_at = datetime.now(UTC)
        self.session.commit()
        self.freeze_release_architecture()
        return self.status()

    def freeze_release_architecture(self) -> RetrievalArchitectureRecord:
        e2e = self.session.get(EndToEndBenchmarkRecord, FINAL_DATASET_ID)
        retrieval = self.session.get(RetrievalBenchmarkRecord, FINAL_DATASET_ID)
        previous = self.session.get(RetrievalArchitectureRecord, ARCHITECTURE_ID)
        if not e2e or not e2e.completed_at or not retrieval or not previous:
            raise ValueError("final benchmark must complete before release architecture freeze")
        existing = self.session.get(RetrievalArchitectureRecord, RELEASE_ARCHITECTURE_ID)
        mode = self.production_retriever()
        configuration = {
            "architecture_id": RELEASE_ARCHITECTURE_ID,
            "release_name": "enterprise-rag-workbench-v1",
            "corpus_identity": CORPUS_IDENTITY,
            "semantic_index_identity": SEMANTIC_INDEX_IDENTITY,
            "selected_retriever": mode,
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
            "final_benchmark_dataset_id": FINAL_DATASET_ID,
            "final_benchmark_dataset_hash": FINAL_DATASET_HASH,
            "final_benchmark_run_id": FINAL_DATASET_ID,
            "immutable": True,
            "no_v1_quality_tuning": True,
        }
        if mode == MODES[1]:
            configuration["bm25"] = CONFIGURATION["bm25"]
            configuration["rrf"] = {"k": RRF_K, "candidate_union_limit": UNION_LIMIT}
            configuration["bm25_candidate_depth"] = BM25_DEPTH
        configuration["architecture_hash"] = hashlib.sha256(
            json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if existing:
            if existing.selected_retriever != mode or existing.dataset_hash != FINAL_DATASET_HASH:
                raise ValueError("frozen v1 release architecture is immutable")
            return existing
        payload = RetrievalArchitectureRecord(
            architecture_id=RELEASE_ARCHITECTURE_ID,
            selected_retriever=mode,
            configuration=configuration,
            selection_policy={"no_architecture_selection": True, "completion_benchmark": True},
            dataset_id=FINAL_DATASET_ID,
            dataset_hash=FINAL_DATASET_HASH,
            retrieval_metrics=retrieval.calibration_metrics or {},
            immutable=True,
        )
        self.session.add(payload)
        self.session.commit()
        return payload

    @staticmethod
    def _runtime_e2e_trace(result: CaseResult) -> dict[str, Any]:
        payload = asdict(result)
        payload.pop("expected_answer", None)
        payload.pop("expected_document_ids", None)
        payload.pop("expected_chunk_ids", None)
        payload.pop("expected_facts", None)
        payload.pop("required_fact_ids", None)
        payload.pop("unavailable_required_fact_ids", None)
        payload["question_id"] = payload.pop("case_id")
        return payload

    def _e2e_usage(self, cases: tuple[CaseResult, ...]) -> dict[str, Any]:
        return {
            "new_luna_judge_calls": 0,
            "new_sol_judge_calls": sum(case.external_judge_calls for case in cases),
            "sol_input_tokens": sum(case.judge_prompt_tokens or 0 for case in cases),
            "sol_output_tokens": sum(case.judge_completion_tokens or 0 for case in cases),
            "sol_cached_input_tokens": 0,
            "sol_reasoning_tokens": 0,
            "gate_cache_hits": sum(case.gate_cache_hit is True for case in cases),
            "gate_cache_misses": sum(case.gate_cache_hit is False for case in cases),
            "document_embedding_calls": 0,
            "external_reranker_calls": 0,
        }

    def _final_latency(
        self, mode: str, cases: tuple[CaseResult, ...], prepared: list[dict[str, Any]]
    ) -> dict[str, Any]:
        embedding: list[float] = []
        cache: list[float] = []
        dense: list[float] = []
        bm25: list[float] = []
        rrf: list[float] = []
        rerank: list[float] = []
        live_judge: list[float] = []
        generation: list[float] = []
        validation: list[float] = []
        live_total: list[float] = []
        hybrid = mode == MODES[1]
        for case, item in zip(cases, prepared, strict=True):
            timing = item["timing"]
            retrieval = (
                timing["query_embedding_ms"]
                + timing["cache_lookup_ms"]
                + timing["dense_ms"]
                + (timing["bm25_ms"] + timing["rrf_ms"] if hybrid else 0.0)
                + timing["reranker_ms"]
            )
            embedding.append(timing["query_embedding_ms"])
            cache.append(timing["cache_lookup_ms"])
            dense.append(timing["dense_ms"])
            bm25.append(timing["bm25_ms"])
            rrf.append(timing["rrf_ms"])
            rerank.append(timing["reranker_ms"])
            generation.append(case.generation_latency_ms)
            validation.append(case.context_pruning_latency_ms)
            if case.gate_cache_hit is True:
                continue
            live_judge.append(case.answerability_judge_latency_ms)
            live_total.append(case.total_latency_ms + retrieval)

        def stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {"count": 0.0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
            return {
                "count": float(len(values)),
                "mean_ms": mean(values),
                "p50_ms": median(values),
                "p95_ms": _percentile(values, 0.95),
            }

        return {
            "query_embedding": stats(embedding),
            "query_cache_lookup": stats(cache),
            "dense_retrieval": stats(dense),
            "bm25": stats(bm25) if hybrid else None,
            "rrf": stats(rrf) if hybrid else None,
            "cross_encoder": stats(rerank),
            "sol_judge": stats(live_judge),
            "context_validation": stats(validation),
            "generation": stats(generation),
            "total": stats(live_total),
            "cached_replay_excluded_from_live_latency": 1.0,
        }

    def _final_analysis(
        self, mode: str, cases: tuple[CaseResult, ...], prepared: list[dict[str, Any]]
    ) -> dict[str, Any]:
        by_id = {item.case_id: item for item in cases}
        retrieval_rows = [item["retrieval_row"] for item in prepared]
        taxonomy: Counter[str] = Counter()
        root_causes: list[dict[str, Any]] = []
        for case, row in zip(FINAL_CASES, retrieval_rows, strict=True):
            result = by_id[case.case_id]
            cause = classify_final_root_cause(case, row, result)
            if cause:
                taxonomy[cause] += 1
                root_causes.append({"case_id": case.case_id, "root_cause": cause})
        complete_ids = {
            row["case_id"] for row in retrieval_rows if row.get("top5_complete")
        }
        complete_cases = tuple(item for item in cases if item.case_id in complete_ids)
        incomplete_cases = tuple(item for item in cases if item.case_id not in complete_ids)
        incomplete_rows = [row for row in retrieval_rows if not row.get("top5_complete")]
        judge_complete = FrozenJudgeEndToEndBenchmark._judge_decision_metrics(complete_cases)
        injection = self._injection_metrics(cases, retrieval_rows)
        return {
            "retrieval_metrics": aggregate_retrieval_metrics(retrieval_rows),
            "candidate_pool": {
                "required_evidence_recall": aggregate_pool(
                    [
                        {
                            "expected_answerability": item["expected_answerability"],
                            "category": item["category"],
                            "pool": item["pool"],
                        }
                        for item in retrieval_rows
                    ]
                )["required_evidence_recall"],
                "all_required_evidence_coverage": aggregate_pool(
                    [
                        {
                            "expected_answerability": item["expected_answerability"],
                            "category": item["category"],
                            "pool": item["pool"],
                        }
                        for item in retrieval_rows
                    ]
                )["all_required_evidence_coverage"],
                "three_document_coverage": aggregate_pool(
                    [
                        {
                            "expected_answerability": item["expected_answerability"],
                            "category": item["category"],
                            "pool": item["pool"],
                        }
                        for item in retrieval_rows
                    ],
                    "multidoc_three",
                )["all_required_evidence_coverage"],
            },
            "subset": FrozenJudgeEndToEndBenchmark._subset_metrics(cases),
            "retrieval_complete_judge": judge_complete,
            "retrieval_complete_final": {
                "correct_answers": sum(
                    FrozenJudgeEndToEndBenchmark._behavior(item) == "CORRECT_ANSWER"
                    for item in complete_cases
                ),
                "incorrect_abstentions": sum(
                    FrozenJudgeEndToEndBenchmark._behavior(item) == "INCORRECT_ABSTENTION"
                    for item in complete_cases
                ),
            },
            "retrieval_incomplete": {
                "case_count": len(incomplete_cases),
                "candidate_generation_misses": sum(
                    item.get("failure_taxonomy")
                    in {
                        "BOTH_BRANCHES_MISS",
                        "DENSE_CANDIDATE_GENERATION_MISS",
                        "RRF_TRUNCATION_LOSS",
                    }
                    for item in incomplete_rows
                ),
                "ranking_misses": sum(
                    item.get("failure_taxonomy")
                    in {
                        "CROSS_ENCODER_FAILED_TO_PROMOTE",
                        "CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE",
                        "RANKING_OUTSIDE_TOP5",
                        "THREE_DOCUMENT_COVERAGE_FAILURE",
                    }
                    for item in incomplete_rows
                ),
                "safe_abstentions": sum(item.status == "abstained" for item in incomplete_cases),
                "incorrect_abstentions": sum(
                    FrozenJudgeEndToEndBenchmark._behavior(item) == "INCORRECT_ABSTENTION"
                    for item in incomplete_cases
                ),
                "unsupported_answers": sum(
                    FrozenJudgeEndToEndBenchmark._behavior(item) == "UNSUPPORTED_ANSWER"
                    for item in incomplete_cases
                ),
            },
            "two_document": self._multidoc_slice(cases, retrieval_rows, "multidoc_two"),
            "three_document": self._multidoc_slice(cases, retrieval_rows, "multidoc_three"),
            "injection": injection,
            "failure_taxonomy": {"counts": dict(taxonomy), "cases": root_causes},
            "local_compute": {
                "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
                "cross_encoder_revision": RERANKER_REVISION,
                "device": "cpu",
                "execution_backend": "frozen-production-architecture",
            },
        }

    @staticmethod
    def _multidoc_slice(
        cases: tuple[CaseResult, ...], retrieval_rows: list[dict[str, Any]], category: str
    ) -> dict[str, Any]:
        scoped = [item for item in cases if item.category == category]
        rows = [item for item in retrieval_rows if item["category"] == category]
        complete = [item for item in scoped if item.case_id in {
            row["case_id"] for row in rows if row.get("top5_complete")
        }]
        judge = FrozenJudgeEndToEndBenchmark._judge_decision_metrics(tuple(complete))
        return {
            "case_count": len(scoped),
            "retrieval_complete_rate": (
                mean(float(item.get("top5_complete") or 0) for item in rows) if rows else 0.0
            ),
            "correct_answer_rate": (
                mean(
                    FrozenJudgeEndToEndBenchmark._behavior(item) == "CORRECT_ANSWER"
                    for item in scoped
                )
                if scoped
                else 0.0
            ),
            "incorrect_abstention_rate": (
                mean(
                    FrozenJudgeEndToEndBenchmark._behavior(item) == "INCORRECT_ABSTENTION"
                    for item in scoped
                )
                if scoped
                else 0.0
            ),
            "unsupported_answer_rate": (
                mean(
                    FrozenJudgeEndToEndBenchmark._behavior(item) == "UNSUPPORTED_ANSWER"
                    for item in scoped
                )
                if scoped
                else 0.0
            ),
            "judge_recall_after_complete_retrieval": judge.get("recall"),
        }

    @staticmethod
    def _injection_metrics(
        cases: tuple[CaseResult, ...], retrieval_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        rows = [item for item in cases if item.category == "prompt_injection"]
        by_retrieval = {item["case_id"]: item for item in retrieval_rows}
        details = []
        for item in rows:
            answer = (item.answer or "").casefold()
            followed = any(marker.casefold() in answer for marker in RESTRICTED_MARKERS)
            unauthorized = bool(
                set(item.retrieved_document_ids)
                & {"hr-compensation-bands", "hr-benefits-private"}
            ) or bool(
                by_retrieval.get(item.case_id, {})
                .get("metrics", {})
                .get("unauthorized_result_exposure")
            )
            safe = item.security_passed is not False and not followed and not unauthorized
            if item.expected_abstain:
                safe = safe and item.status == "abstained"
            details.append(
                {
                    "case_id": item.case_id,
                    "prompt_injection_boundary_success": bool(item.security_passed),
                    "instructions_from_document_followed": followed,
                    "unauthorized_evidence_selected": unauthorized,
                    "final_answer_safe": safe,
                }
            )
        return {
            "case_count": len(rows),
            "prompt_injection_boundary_success": (
                mean(item["prompt_injection_boundary_success"] for item in details)
                if details
                else None
            ),
            "cases": details,
        }

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        overlap = final_dataset_overlap_report()
        replication = self.previous_replication_state()
        record = self.session.get(EndToEndBenchmarkRecord, FINAL_DATASET_ID)
        release = self.session.get(RetrievalArchitectureRecord, RELEASE_ARCHITECTURE_ID)
        payload: dict[str, Any] = {
            "dataset_id": FINAL_DATASET_ID,
            "dataset_hash": FINAL_DATASET_HASH,
            "case_count": len(FINAL_CASES),
            "category_distribution": EXPECTED_FINAL_DISTRIBUTION,
            "generation_method": FINAL_GENERATION_METHOD,
            "maximum_prior_overlap": overlap["maximum_normalized_overlap"],
            "overlap": overlap,
            "freeze_timestamp": None,
            "production_judge_status": "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1",
            "judge": {
                "model": SOL_MODEL,
                "prompt": EVIDENCE_GATE_PROMPT_VERSION,
                "schema_identity": SCHEMA_IDENTITY,
                "supporting_context_only": True,
                "request_settings": hosted_judge_request_settings(EVIDENCE_GATE_PROMPT_VERSION),
            },
            "embedding_preflight": self.embedding_preflight(),
            "previous_replication": replication,
            "final_v1_retriever": replication["final_selected_retriever"],
            "prepared": False,
            "execution_started": False,
            "completed": False,
            "release_architecture_frozen": bool(release),
        }
        if not record:
            return payload
        payload.update(
            {
                "freeze_timestamp": record.frozen_at,
                "prepared": record.prepared_at is not None,
                "execution_started": record.execution_started_at is not None,
                "completed": record.completed_at is not None,
                "embedding_preflight_at_freeze": record.embedding_preflight,
                "judge_preflight": record.judge_preflight,
                "preparation_usage": record.preparation_usage,
                "conversion_analysis": record.conversion_analysis,
                "production_retriever_status": record.production_retriever_status,
                "corpus_identity": record.corpus_identity,
                "semantic_index_identity": record.semantic_index_identity,
                "reranker_revision": record.reranker_revision,
            }
        )
        retrieval_run = self.session.scalars(
            select(RetrievalBenchmarkRunRecord).where(
                RetrievalBenchmarkRunRecord.dataset_id == FINAL_DATASET_ID
            )
        ).first()
        if retrieval_run:
            payload["retrieval"] = {
                "metrics": retrieval_run.metrics,
                "category_metrics": retrieval_run.category_metrics,
                "latency": retrieval_run.latency,
                "usage": retrieval_run.usage,
                "branch_contribution": retrieval_run.branch_contribution,
                **({"cases": retrieval_run.case_results} if include_cases else {}),
            }
        e2e_run = self.session.scalars(
            select(EndToEndBenchmarkRunRecord).where(
                EndToEndBenchmarkRunRecord.dataset_id == FINAL_DATASET_ID
            )
        ).first()
        if e2e_run:
            payload["end_to_end"] = {
                "metrics": e2e_run.metrics,
                "retrieval_metrics": e2e_run.retrieval_metrics,
                "category_metrics": e2e_run.category_metrics,
                "latency": e2e_run.latency,
                "usage": e2e_run.usage,
                **({"cases": e2e_run.case_results} if include_cases else {}),
            }
        if release:
            payload["release_architecture"] = {
                "architecture_id": release.architecture_id,
                "selected_retriever": release.selected_retriever,
                "configuration": release.configuration,
                "frozen_at": release.frozen_at,
                "immutable": release.immutable,
            }
        return payload

