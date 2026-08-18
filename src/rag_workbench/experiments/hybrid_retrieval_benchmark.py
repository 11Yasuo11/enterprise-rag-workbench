from __future__ import annotations

import hashlib
import random
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    ExperimentRunRecord,
    QueryEmbeddingCacheRecord,
    RetrievalBenchmarkRecord,
    RetrievalBenchmarkRunRecord,
)
from rag_workbench.evaluation.hybrid_metrics import (
    aggregate_retrieval_metrics,
    category_retrieval_metrics,
    retrieval_case_metrics,
)
from rag_workbench.evaluation.retrieval_dataset import (
    RetrievalGroundTruthCase,
    load_retrieval_ground_truth,
)
from rag_workbench.providers.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from rag_workbench.retrieval.base import RetrievalMode
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal

DATASET_ID = "acmeai-hybrid-retrieval-eval-v1"
DATASET_PATH = Path("data/eval/acmeai_hybrid_retrieval_eval_v1.json")
DATASET_HASH = "b935a5bcad77ea03128f280cf6dacb06cacad2c62b4c4ec0da477f37c7b43f3b"
SPLIT_SEED = 2026081703
SEMANTIC_INDEX_IDENTITY = (
    "e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466"
)
HOLDOUT_ALLOCATION = {
    "multidoc_two": 4,
    "multidoc_three": 3,
    "exact_identifier": 3,
    "version_region": 3,
    "near_duplicate": 3,
    "semantic_paraphrase": 2,
    "acl_sensitive": 1,
    "partial_no_answer": 1,
}
RETRIEVAL_CONFIGURATION = {
    "embedding_provider": "openai-compatible",
    "embedding_model": "text-embedding-3-small",
    "embedding_version": "1",
    "embedding_dimension": 64,
    "dense_distance": "pgvector_cosine",
    "dense_threshold": 0.28,
    "dense_candidate_depth": 20,
    "bm25_candidate_depth": 20,
    "final_top_k": 5,
    "bm25": {
        "implementation": "BM25 Okapi",
        "version": "bm25-okapi-v1",
        "k1": 1.2,
        "b": 0.75,
        "tokenization": "NFKC + casefold + words + preserved identifier and components",
    },
    "rrf_k": 60,
    "dense_threshold_policy": "0.28 applies to dense branch only; never to RRF scores",
    "deduplication": "canonical chunk_id",
    "tie_breaking": "fusion score desc, best branch rank asc, chunk_id asc",
}
SELECTION_POLICY = {
    "primary_metric": "all_required_evidence_coverage_at_5",
    "minimum_absolute_gain": 0.10,
    "required_evidence_recall_must_not_regress": True,
    "semantic_material_regression_threshold": 0.10,
    "acl_safety_required": 1.0,
    "eligible_candidates": ["DENSE", "HYBRID_RRF"],
}


def deterministic_retrieval_split(
    cases: tuple[RetrievalGroundTruthCase, ...],
    *,
    seed: int = SPLIT_SEED,
) -> tuple[tuple[RetrievalGroundTruthCase, ...], tuple[RetrievalGroundTruthCase, ...], str]:
    grouped: dict[str, list[RetrievalGroundTruthCase]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)
    if set(grouped) != set(HOLDOUT_ALLOCATION):
        raise ValueError("retrieval dataset category composition changed")
    holdout_ids: set[str] = set()
    for category, allocation in sorted(HOLDOUT_ALLOCATION.items()):
        candidates = sorted(grouped[category], key=lambda item: item.case_id)
        rng = random.Random(f"{seed}:{category}")
        rng.shuffle(candidates)
        holdout_ids.update(item.case_id for item in candidates[:allocation])
    calibration = tuple(case for case in cases if case.case_id not in holdout_ids)
    holdout = tuple(case for case in cases if case.case_id in holdout_ids)
    payload = "|".join(item.case_id for item in calibration)
    payload += "::" + "|".join(item.case_id for item in holdout)
    identity = hashlib.sha256(f"{seed}:{payload}".encode()).hexdigest()
    if len(calibration) != 40 or len(holdout) != 20:
        raise ValueError("retrieval split must be exactly 40 calibration / 20 holdout")
    return calibration, holdout, identity


DATASET = load_retrieval_ground_truth(DATASET_PATH)
CALIBRATION_CASES, HOLDOUT_CASES, SPLIT_IDENTITY = deterministic_retrieval_split(
    DATASET.cases
)


def _principal(case: RetrievalGroundTruthCase) -> Principal:
    return Principal(
        principal_id=case.principal.principal_id,
        tenant_id=case.principal.tenant_id,
        permission_groups=frozenset(case.principal.permission_groups),
    )


def _trace(results: list[RetrievalResult]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item.chunk_id,
            "document_id": item.document_id,
            "document_version_id": item.document_version_id,
            "version": item.version,
            "rank": item.rank,
            "retrieval_source": item.retrieval_source,
            "dense_score": item.dense_score,
            "lexical_score": item.lexical_score,
            "fusion_score": item.fusion_score,
            "found_by_dense": item.found_by_dense,
            "found_by_bm25": item.found_by_bm25,
        }
        for item in results
    ]


class HybridRetrievalBenchmark:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        actual_hash = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
        if DATASET.dataset_id != DATASET_ID or actual_hash != DATASET_HASH:
            raise ValueError("hybrid retrieval dataset identity changed; execution is blocked")

    def _corpus_identity(self) -> str:
        rows = self.session.execute(
            select(Document.document_id, DocumentVersion.version, DocumentVersion.content_hash)
            .join(DocumentVersion, DocumentVersion.document_fk == Document.id)
            .where(DocumentVersion.is_active.is_(True))
            .order_by(Document.document_id, DocumentVersion.version)
        ).all()
        payload = "|".join(":".join(row) for row in rows)
        return hashlib.sha256(payload.encode()).hexdigest()

    def initialize(self) -> RetrievalBenchmarkRecord:
        if not self.session.scalar(
            select(Chunk.id)
            .where(Chunk.index_identity == SEMANTIC_INDEX_IDENTITY)
            .limit(1)
        ):
            raise ValueError("frozen semantic index is unavailable")
        corpus_identity = self._corpus_identity()
        existing = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        if existing is not None:
            self._verify_frozen(existing, corpus_identity)
            return existing
        record = RetrievalBenchmarkRecord(
            dataset_id=DATASET_ID,
            dataset_hash=DATASET_HASH,
            split_identity=SPLIT_IDENTITY,
            split_seed=SPLIT_SEED,
            calibration_case_ids=[item.case_id for item in CALIBRATION_CASES],
            holdout_case_ids=[item.case_id for item in HOLDOUT_CASES],
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=corpus_identity,
            retrieval_configuration=RETRIEVAL_CONFIGURATION,
            selection_policy=SELECTION_POLICY,
        )
        self.session.add(record)
        self.session.commit()
        return record

    def _verify_frozen(
        self, record: RetrievalBenchmarkRecord, corpus_identity: str | None = None
    ) -> None:
        expected = (
            record.dataset_hash == DATASET_HASH
            and record.split_identity == SPLIT_IDENTITY
            and record.split_seed == SPLIT_SEED
            and record.semantic_index_identity == SEMANTIC_INDEX_IDENTITY
            and record.retrieval_configuration == RETRIEVAL_CONFIGURATION
            and record.selection_policy == SELECTION_POLICY
            and record.calibration_case_ids == [item.case_id for item in CALIBRATION_CASES]
            and record.holdout_case_ids == [item.case_id for item in HOLDOUT_CASES]
        )
        if corpus_identity is not None:
            expected = expected and record.corpus_identity == corpus_identity
        if not expected:
            raise ValueError("frozen retrieval benchmark identity changed")

    def preflight(self) -> dict[str, Any]:
        unique_queries = tuple(dict.fromkeys(case.question for case in DATASET.cases))
        keys = [
            query_embedding_cache_key(
                query,
                provider="openai-compatible",
                model="text-embedding-3-small",
                version="1",
                dimension=64,
            )
            for query in unique_queries
        ]
        cache_matches = sum(
            self.session.get(QueryEmbeddingCacheRecord, key) is not None for key in keys
        )
        cumulative = self._cumulative_embedding_calls()
        missing = len(keys) - cache_matches
        return {
            "current_cumulative_embedding_calls": cumulative,
            "configured_ceiling": self.settings.max_external_embedding_calls,
            "new_unique_queries": len(unique_queries),
            "existing_cache_matches": cache_matches,
            "missing_embeddings": missing,
            "maximum_required_new_calls": missing,
            "document_embedding_calls": 0,
            "remaining_capacity": max(
                0, self.settings.max_external_embedding_calls - cumulative
            ),
        }

    def _cumulative_embedding_calls(self) -> int:
        experiment_calls = sum(
            item.external_embedding_calls or 0
            for item in self.session.scalars(select(ExperimentRunRecord)).all()
        )
        benchmark_calls = sum(
            int((item.usage or {}).get("new_query_embedding_calls", 0))
            for item in self.session.scalars(select(RetrievalBenchmarkRunRecord)).all()
            if item.retrieval_mode == RetrievalMode.DENSE
        )
        cached = self.session.scalar(
            select(func.count())
            .select_from(QueryEmbeddingCacheRecord)
            .where(
                QueryEmbeddingCacheRecord.embedding_provider == "openai-compatible",
                QueryEmbeddingCacheRecord.embedding_model == "text-embedding-3-small",
                QueryEmbeddingCacheRecord.embedding_version == "1",
                QueryEmbeddingCacheRecord.embedding_dimension == 64,
            )
        ) or 0
        return max(experiment_calls + benchmark_calls, int(cached))

    def _authorize_embeddings(self, cases: tuple[RetrievalGroundTruthCase, ...]) -> None:
        if not self.settings.allow_external_calls:
            raise ValueError("ALLOW_EXTERNAL_CALLS=true is required")
        if not self.settings.embedding_api_key:
            raise ValueError("EMBEDDING_API_KEY is required")
        preflight = self.preflight()
        case_keys = {
            query_embedding_cache_key(
                case.question,
                provider="openai-compatible",
                model="text-embedding-3-small",
                version="1",
                dimension=64,
            )
            for case in cases
        }
        missing = sum(
            self.session.get(QueryEmbeddingCacheRecord, key) is None for key in case_keys
        )
        if missing > preflight["remaining_capacity"]:
            raise ValueError(
                f"retrieval benchmark has {preflight['current_cumulative_embedding_calls']} "
                f"persisted calls and needs {missing} more, exceeding ceiling "
                f"{preflight['configured_ceiling']}"
            )

    def _embedding_provider(self) -> OpenAICompatibleEmbeddingProvider:
        return OpenAICompatibleEmbeddingProvider(
            api_key=self.settings.embedding_api_key or "",
            model="text-embedding-3-small",
            dimension=64,
            base_url=self.settings.embedding_base_url,
            version="1",
            provider_name="openai-compatible",
        )

    def run_calibration(self) -> dict[str, Any]:
        record = self.initialize()
        if record.locked_at is not None:
            raise ValueError("retrieval calibration is already locked and immutable")
        if self._partition_runs("calibration"):
            raise ValueError("retrieval calibration was already executed")
        metrics = self._run_partition("calibration", CALIBRATION_CASES)
        dense = metrics[RetrievalMode.DENSE]
        hybrid = metrics[RetrievalMode.HYBRID_RRF]
        gain = (
            hybrid["all_required_evidence_coverage_at_5"]
            - dense["all_required_evidence_coverage_at_5"]
        )
        required_recall_ok = (
            hybrid["required_evidence_recall_at_5"]
            >= dense["required_evidence_recall_at_5"]
        )
        runs = self._partition_runs("calibration")
        semantic_dense = runs[RetrievalMode.DENSE].category_metrics[
            "semantic_paraphrase"
        ]["all_required_evidence_coverage_at_5"]
        semantic_hybrid = runs[RetrievalMode.HYBRID_RRF].category_metrics[
            "semantic_paraphrase"
        ]["all_required_evidence_coverage_at_5"]
        semantic_ok = semantic_hybrid >= semantic_dense - 0.10
        acl_ok = hybrid["acl_safety"] == 1.0
        selected = (
            RetrievalMode.HYBRID_RRF
            if gain >= 0.10 and required_recall_ok and semantic_ok and acl_ok
            else RetrievalMode.DENSE
        )
        reason = (
            f"Hybrid coverage gain={gain:.6f}; required recall non-regression="
            f"{required_recall_ok}; semantic non-regression={semantic_ok}; ACL safety={acl_ok}."
        )
        record = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        record.selected_retrieval_mode = selected
        record.calibration_metrics = {mode: values for mode, values in metrics.items()}
        record.selection_reason = reason
        record.locked_at = datetime.now(UTC)
        self.session.commit()
        return self.status(include_cases=True)

    def run_holdout(self) -> dict[str, Any]:
        record = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        if record is None or record.locked_at is None or not record.selected_retrieval_mode:
            raise ValueError("calibration retrieval configuration must be locked first")
        self._verify_frozen(record, self._corpus_identity())
        if record.holdout_started_at is not None:
            raise ValueError("retrieval holdout is one-shot and already started")
        record.holdout_started_at = datetime.now(UTC)
        self.session.commit()
        self._run_partition("holdout", HOLDOUT_CASES)
        record = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        record.holdout_completed_at = datetime.now(UTC)
        self.session.commit()
        return self.status(include_cases=True)

    def _run_partition(
        self,
        partition: str,
        cases: tuple[RetrievalGroundTruthCase, ...],
    ) -> dict[RetrievalMode, dict[str, Any]]:
        self._authorize_embeddings(cases)
        judge_calls_before = sum(
            item.external_judge_calls or 0
            for item in self.session.scalars(select(ExperimentRunRecord)).all()
        )
        provider = self._embedding_provider()
        dense = Retriever(self.session, provider, SEMANTIC_INDEX_IDENTITY)
        bm25 = BM25Retriever(
            self.session,
            index_identity=SEMANTIC_INDEX_IDENTITY,
            embedding_provider="openai-compatible",
            embedding_model="text-embedding-3-small",
            embedding_version="1",
            embedding_dimension=64,
            config=BM25Config(),
        )
        by_mode: dict[RetrievalMode, list[dict[str, Any]]] = {
            mode: [] for mode in RetrievalMode
        }
        latencies: dict[str, list[float]] = defaultdict(list)
        cache_hits = 0
        cache_misses = 0
        input_tokens = 0
        external_calls = 0
        bm25_index_identities: set[str] = set()
        bm25_index_sizes: list[int] = []
        for case in cases:
            embedding = dense.query_embedding_cache.get_or_embed(case.question)
            self.session.commit()  # Persist each paid call before any later benchmark work.
            cache_hits += int(embedding.cache_hit)
            cache_misses += int(not embedding.cache_hit)
            input_tokens += embedding.input_tokens
            external_calls += embedding.external_calls
            principal = _principal(case)

            dense_started = time.perf_counter()
            dense_candidates = dense.retrieve_with_embedding(
                embedding,
                top_k=20,
                score_threshold=0.28,
                principal=principal,
            )
            dense_ms = (time.perf_counter() - dense_started) * 1000
            bm25_started = time.perf_counter()
            bm25_candidates = bm25.retrieve(case.question, top_k=20, principal=principal)
            bm25_ms = (time.perf_counter() - bm25_started) * 1000
            fusion_started = time.perf_counter()
            hybrid_results = reciprocal_rank_fusion(
                dense_candidates, bm25_candidates, top_k=5, rrf_k=60
            )
            fusion_ms = (time.perf_counter() - fusion_started) * 1000
            views = {
                RetrievalMode.DENSE: dense_candidates[:5],
                RetrievalMode.LEXICAL_BM25: bm25_candidates[:5],
                RetrievalMode.HYBRID_RRF: hybrid_results,
            }
            latencies["query_embedding"].append(embedding.embedding_latency_ms)
            latencies["cache_lookup"].append(embedding.cache_lookup_latency_ms)
            latencies["dense"].append(dense_ms)
            latencies["bm25"].append(bm25_ms)
            latencies["bm25_acl_filter"].append(bm25.last_timing.acl_filter_latency_ms)
            latencies["bm25_index_build"].append(
                bm25.last_timing.index_build_latency_ms
            )
            latencies["bm25_search"].append(bm25.last_timing.lexical_search_latency_ms)
            latencies["fusion"].append(fusion_ms)
            latencies["dense_total"].append(
                embedding.embedding_latency_ms + embedding.cache_lookup_latency_ms + dense_ms
            )
            latencies["bm25_total"].append(bm25_ms)
            latencies["hybrid_total"].append(
                embedding.embedding_latency_ms
                + embedding.cache_lookup_latency_ms
                + dense_ms
                + bm25_ms
                + fusion_ms
            )
            bm25_index_identities.add(bm25.last_index_identity)
            bm25_index_sizes.append(bm25.last_timing.index_size_bytes)
            for mode, results in views.items():
                metrics = retrieval_case_metrics(case, results)
                item_result = {
                    "case_id": case.case_id,
                    "question": case.question,
                    "category": case.category,
                    "expected_answerability": case.expected_answerability,
                    "expected_access_behavior": case.expected_access_behavior,
                    "required_document_ids": list(case.required_document_ids),
                    "required_chunk_ids": list(case.required_chunk_ids),
                    "required_version_ids": case.required_version_ids,
                    "forbidden_document_ids": list(case.forbidden_document_ids),
                    "retrieval_trace": _trace(results),
                    "metrics": metrics,
                    "failure_types": self._failures(
                        case,
                        mode,
                        results,
                        dense_candidates,
                        bm25_candidates,
                    ),
                }
                if mode == RetrievalMode.HYBRID_RRF:
                    item_result.update(
                        {
                            "dense_top5_document_ids": [
                                item.document_id for item in dense_candidates[:5]
                            ],
                            "bm25_top5_document_ids": [
                                item.document_id for item in bm25_candidates[:5]
                            ],
                        }
                    )
                by_mode[mode].append(item_result)

        judge_calls_after = sum(
            item.external_judge_calls or 0
            for item in self.session.scalars(select(ExperimentRunRecord)).all()
        )
        if judge_calls_after != judge_calls_before:
            raise RuntimeError("retrieval-only benchmark changed hosted judge usage")
        common_latency = {
            name + "_ms": mean(values) if values else 0.0
            for name, values in latencies.items()
        }
        common_latency.update(
            {
                "bm25_index_size_bytes": max(bm25_index_sizes, default=0),
            }
        )
        usage = {
            "new_query_embedding_calls": external_calls,
            "embedding_tokens": input_tokens,
            "query_cache_hits": cache_hits,
            "query_cache_misses": cache_misses,
            "document_embedding_calls": 0,
            "judge_calls": 0,
        }
        output: dict[RetrievalMode, dict[str, Any]] = {}
        for mode, results in by_mode.items():
            aggregate = aggregate_retrieval_metrics(results)
            branch = self._branch_contribution(results) if mode == RetrievalMode.HYBRID_RRF else {}
            record = RetrievalBenchmarkRunRecord(
                dataset_id=DATASET_ID,
                partition=partition,
                retrieval_mode=mode,
                metrics=aggregate,
                category_metrics=category_retrieval_metrics(results),
                case_results=results,
                branch_contribution=branch,
                latency=common_latency,
                usage=(
                    usage
                    if mode == RetrievalMode.DENSE
                    else {**usage, "new_query_embedding_calls": 0}
                ),
            )
            if mode == RetrievalMode.LEXICAL_BM25:
                record.branch_contribution = {
                    "bm25_index_identities": sorted(bm25_index_identities),
                    "bm25_index_size_bytes": max(bm25_index_sizes, default=0),
                }
            self.session.add(record)
            output[mode] = aggregate
        self.session.commit()
        return output

    @staticmethod
    def _failures(
        case: RetrievalGroundTruthCase,
        mode: RetrievalMode,
        results: list[RetrievalResult],
        dense_candidates: list[RetrievalResult],
        bm25_candidates: list[RetrievalResult],
    ) -> list[str]:
        if not case.expected_answerability:
            if case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                return ["ACL_EXPECTED_EXCLUSION"]
            return []
        required = set(case.required_document_ids)
        retrieved = {item.document_id for item in results}
        if required <= retrieved:
            return []
        failures: list[str] = []
        candidate_documents = {
            item.document_id
            for item in (
                dense_candidates if mode == RetrievalMode.DENSE else bm25_candidates
            )
        }
        if mode == RetrievalMode.HYBRID_RRF:
            candidate_documents = {
                item.document_id for item in dense_candidates + bm25_candidates
            }
        if required <= candidate_documents:
            failures.append("RANKING_OUTSIDE_TOP5")
        if case.category == "exact_identifier":
            failures.append("EXACT_IDENTIFIER_MISS")
        if case.category in {"version_region", "near_duplicate"}:
            failures.append(
                "NEAR_DUPLICATE_CONFUSION"
                if case.category == "near_duplicate"
                else "VERSION_EVIDENCE_MISS"
            )
        failures.append(
            {
                RetrievalMode.DENSE: "DENSE_MISS",
                RetrievalMode.LEXICAL_BM25: "LEXICAL_MISS",
                RetrievalMode.HYBRID_RRF: "HYBRID_MISS",
            }[mode]
        )
        return list(dict.fromkeys(failures))

    @staticmethod
    def _branch_contribution(results: list[dict[str, Any]]) -> dict[str, Any]:
        dense_only = 0
        bm25_only = 0
        both = 0
        bm25_rescued = 0
        dense_rescued = 0
        semantic_regressions: list[str] = []
        for case in results:
            required = set(case["required_document_ids"])
            hybrid_documents = {
                item["document_id"] for item in case["retrieval_trace"]
            }
            dense_top5 = set(case.get("dense_top5_document_ids", []))
            bm25_top5 = set(case.get("bm25_top5_document_ids", []))
            bm25_rescued += len((required & hybrid_documents & bm25_top5) - dense_top5)
            dense_rescued += len((required & hybrid_documents & dense_top5) - bm25_top5)
            if (
                case["category"] == "semantic_paraphrase"
                and required <= dense_top5
                and not required <= hybrid_documents
            ):
                semantic_regressions.append(case["case_id"])
            for item in case["retrieval_trace"]:
                if item["document_id"] not in required:
                    continue
                if item["found_by_dense"] and item["found_by_bm25"]:
                    both += 1
                elif item["found_by_dense"]:
                    dense_only += 1
                elif item["found_by_bm25"]:
                    bm25_only += 1
        return {
            "dense_only_required_evidence": dense_only,
            "bm25_only_required_evidence": bm25_only,
            "found_by_both_required_evidence": both,
            "bm25_rescued_required_evidence_missed_by_dense": bm25_rescued,
            "dense_rescued_required_evidence_missed_by_bm25": dense_rescued,
            "semantic_regression_case_ids": semantic_regressions,
        }

    def _partition_runs(self, partition: str) -> dict[str, RetrievalBenchmarkRunRecord]:
        rows = self.session.scalars(
            select(RetrievalBenchmarkRunRecord).where(
                RetrievalBenchmarkRunRecord.dataset_id == DATASET_ID,
                RetrievalBenchmarkRunRecord.partition == partition,
            )
        ).all()
        return {item.retrieval_mode: item for item in rows}

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        base: dict[str, Any] = {
            "dataset_id": DATASET_ID,
            "dataset_hash": DATASET_HASH,
            "case_count": len(DATASET.cases),
            "category_distribution": dict(
                sorted(Counter(item.category for item in DATASET.cases).items())
            ),
            "split_seed": SPLIT_SEED,
            "split_identity": SPLIT_IDENTITY,
            "calibration_count": len(CALIBRATION_CASES),
            "holdout_count": len(HOLDOUT_CASES),
            "retrieval_configuration": RETRIEVAL_CONFIGURATION,
            "selection_policy": SELECTION_POLICY,
            "selected_retrieval_mode": None,
            "holdout_started": False,
            "holdout_completed": False,
            "preflight": self.preflight(),
        }
        if record is None:
            return base
        self._verify_frozen(record)
        base.update(
            {
                "corpus_identity": record.corpus_identity,
                "semantic_index_identity": record.semantic_index_identity,
                "selected_retrieval_mode": record.selected_retrieval_mode,
                "selection_reason": record.selection_reason,
                "locked_at": record.locked_at,
                "holdout_started": record.holdout_started_at is not None,
                "holdout_completed": record.holdout_completed_at is not None,
            }
        )
        for partition in ("calibration", "holdout"):
            runs = self._partition_runs(partition)
            if runs:
                base[partition] = {
                    mode: {
                        "id": run.id,
                        "metrics": run.metrics,
                        "category_metrics": run.category_metrics,
                        "branch_contribution": run.branch_contribution,
                        "latency": run.latency,
                        "usage": run.usage,
                        **({"cases": run.case_results} if include_cases else {}),
                    }
                    for mode, run in runs.items()
                }
        return base
