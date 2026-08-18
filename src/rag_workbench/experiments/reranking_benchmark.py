from __future__ import annotations

import hashlib
import os
import random
import resource
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
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
    RerankingBenchmarkRecord,
    RerankingBenchmarkRunRecord,
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
from rag_workbench.reranking import CrossEncoderReranker, Reranker
from rag_workbench.reranking.cross_encoder import MODEL_ID
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal

DATASET_ID = "acmeai-reranking-eval-v1"
DATASET_PATH = Path("data/eval/acmeai_reranking_eval_v1.json")
DATASET_HASH = "36ad031f08f4dbbc0a1c67ab292eda549cf7291e91807e2cd0ba4f89b051b12f"
SPLIT_SEED = 2026081711
SEMANTIC_INDEX_IDENTITY = "e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466"
MODES = ("DENSE", "DENSE_CROSS_ENCODER_RERANK")
HOLDOUT_ALLOCATION = {
    "multidoc_two": 3,
    "multidoc_three": 5,
    "near_duplicate": 3,
    "exact_identifier": 2,
    "version_region": 2,
    "semantic_paraphrase": 2,
    "acl_sensitive": 1,
    "partial_no_answer": 2,
}
CONFIGURATION = {
    "embedding_model": "text-embedding-3-small",
    "embedding_dimension": 64,
    "dense_index_identity": SEMANTIC_INDEX_IDENTITY,
    "dense_distance": "pgvector_cosine",
    "dense_threshold": 0.28,
    "candidate_depth": 20,
    "final_top_k": 5,
    "reranker_model": MODEL_ID,
    "reranker_backend": "sentence-transformers/CrossEncoder (PyTorch)",
    "device": "cpu",
    "reranker_threshold": None,
}
SELECTION_POLICY = {
    "all_required_coverage_minimum_gain": 0.10,
    "three_document_coverage_minimum_gain": 0.20,
    "required_recall_maximum_regression": 0.02,
    "exact_identifier_maximum_regression": 0.05,
    "semantic_maximum_regression": 0.05,
    "acl_safety_required": 1.0,
    "version_correctness_required": 1.0,
}


def deterministic_reranking_split(
    cases: tuple[RetrievalGroundTruthCase, ...], *, seed: int = SPLIT_SEED
) -> tuple[tuple[RetrievalGroundTruthCase, ...], tuple[RetrievalGroundTruthCase, ...], str]:
    grouped: dict[str, list[RetrievalGroundTruthCase]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)
    if set(grouped) != set(HOLDOUT_ALLOCATION):
        raise ValueError("reranking dataset category composition changed")
    holdout_ids: set[str] = set()
    for category, allocation in sorted(HOLDOUT_ALLOCATION.items()):
        values = sorted(grouped[category], key=lambda item: item.case_id)
        random.Random(f"{seed}:{category}").shuffle(values)
        holdout_ids.update(item.case_id for item in values[:allocation])
    calibration = tuple(item for item in cases if item.case_id not in holdout_ids)
    holdout = tuple(item for item in cases if item.case_id in holdout_ids)
    payload = "|".join(item.case_id for item in calibration)
    payload += "::" + "|".join(item.case_id for item in holdout)
    identity = hashlib.sha256(f"{seed}:{payload}".encode()).hexdigest()
    if len(calibration) != 40 or len(holdout) != 20:
        raise ValueError("reranking split must be exactly 40/20")
    return calibration, holdout, identity


DATASET = load_retrieval_ground_truth(DATASET_PATH)
CALIBRATION_CASES, HOLDOUT_CASES, SPLIT_IDENTITY = deterministic_reranking_split(DATASET.cases)


def _principal(case: RetrievalGroundTruthCase) -> Principal:
    return Principal(
        case.principal.principal_id,
        case.principal.tenant_id,
        frozenset(case.principal.permission_groups),
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _first_rank(results: list[RetrievalResult], document_id: str) -> int | None:
    return next((item.rank for item in results if item.document_id == document_id), None)


def _candidate_pool_case_metrics(
    case: RetrievalGroundTruthCase, results: list[RetrievalResult]
) -> dict[str, Any]:
    """Measure candidate generation at its frozen depth, without a Top-5 truncation."""
    documents = {item.document_id for item in results[:20]}
    required = set(case.required_document_ids)
    retrieved = required & documents
    version_correct = all(
        any(item.document_id == document_id and item.version == version for item in results[:20])
        for document_id, version in case.required_version_ids.items()
    )
    unauthorized = (
        sum(item.document_id in set(case.forbidden_document_ids) for item in results[:20])
        if case.expected_access_behavior == "EXCLUDE_FORBIDDEN"
        else 0
    )
    return {
        "required_evidence_count": len(required),
        "required_evidence_retrieved_at_20": len(retrieved),
        "required_evidence_recall_at_20": (len(retrieved) / len(required) if required else None),
        "all_required_evidence_coverage_at_20": (
            float(required <= documents) if case.expected_answerability else None
        ),
        "version_correct_at_20": float(version_correct),
        "unauthorized_result_exposure_at_20": unauthorized,
    }


def _aggregate_candidate_pool_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in rows if item["expected_answerability"]]

    def average(name: str, values: list[dict[str, Any]] = answerable) -> float:
        observed = [item["metrics"][name] for item in values if item["metrics"][name] is not None]
        return mean(observed) if observed else 0.0

    two = [item for item in answerable if item["category"] == "multidoc_two"]
    three = [item for item in answerable if item["category"] == "multidoc_three"]
    version = [item for item in answerable if item["category"] == "version_region"]
    unauthorized = sum(item["metrics"]["unauthorized_result_exposure_at_20"] for item in rows)
    return {
        "required_evidence_recall_at_20": average("required_evidence_recall_at_20"),
        "all_required_evidence_coverage_at_20": average("all_required_evidence_coverage_at_20"),
        "two_document_coverage_at_20": average("all_required_evidence_coverage_at_20", two),
        "three_document_coverage_at_20": average("all_required_evidence_coverage_at_20", three),
        "version_correctness_at_20": average("version_correct_at_20", version),
        "acl_safety_at_20": float(unauthorized == 0),
        "unauthorized_result_exposure_at_20": unauthorized,
    }


class DenseCrossEncoderBenchmark:
    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        reranker_factory: Any | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.reranker_factory = reranker_factory or (lambda: CrossEncoderReranker(device="cpu"))
        if hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() != DATASET_HASH:
            raise ValueError("reranking dataset identity changed; execution is blocked")

    def _corpus_identity(self) -> str:
        rows = self.session.execute(
            select(Document.document_id, DocumentVersion.version, DocumentVersion.content_hash)
            .join(DocumentVersion, DocumentVersion.document_fk == Document.id)
            .where(DocumentVersion.is_active.is_(True))
            .order_by(Document.document_id, DocumentVersion.version)
        ).all()
        return hashlib.sha256("|".join(":".join(row) for row in rows).encode()).hexdigest()

    def initialize(self) -> RerankingBenchmarkRecord:
        if not self.session.scalar(
            select(Chunk.id).where(Chunk.index_identity == SEMANTIC_INDEX_IDENTITY).limit(1)
        ):
            raise ValueError("frozen semantic index is unavailable")
        corpus_identity = self._corpus_identity()
        existing = self.session.get(RerankingBenchmarkRecord, DATASET_ID)
        if existing:
            self._verify(existing, corpus_identity)
            return existing
        record = RerankingBenchmarkRecord(
            dataset_id=DATASET_ID,
            dataset_hash=DATASET_HASH,
            split_identity=SPLIT_IDENTITY,
            split_seed=SPLIT_SEED,
            calibration_case_ids=[item.case_id for item in CALIBRATION_CASES],
            holdout_case_ids=[item.case_id for item in HOLDOUT_CASES],
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=corpus_identity,
            configuration=CONFIGURATION,
            selection_policy=SELECTION_POLICY,
        )
        self.session.add(record)
        self.session.commit()
        return record

    def _verify(self, record: RerankingBenchmarkRecord, corpus_identity: str | None = None) -> None:
        valid = (
            record.dataset_hash == DATASET_HASH
            and record.split_identity == SPLIT_IDENTITY
            and record.split_seed == SPLIT_SEED
            and record.semantic_index_identity == SEMANTIC_INDEX_IDENTITY
            and record.configuration == CONFIGURATION
            and record.selection_policy == SELECTION_POLICY
            and record.calibration_case_ids == [item.case_id for item in CALIBRATION_CASES]
            and record.holdout_case_ids == [item.case_id for item in HOLDOUT_CASES]
        )
        if corpus_identity is not None:
            valid = valid and record.corpus_identity == corpus_identity
        if not valid:
            raise ValueError("frozen reranking benchmark identity changed")

    def _cumulative_calls(self) -> int:
        cached = (
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
        return int(cached)

    def preflight(self) -> dict[str, int]:
        queries = tuple(dict.fromkeys(item.question for item in DATASET.cases))
        keys = [
            query_embedding_cache_key(
                query,
                provider="openai-compatible",
                model="text-embedding-3-small",
                version="1",
                dimension=64,
            )
            for query in queries
        ]
        matches = sum(self.session.get(QueryEmbeddingCacheRecord, key) is not None for key in keys)
        current = self._cumulative_calls()
        return {
            "current_cumulative_embedding_calls": current,
            "configured_ceiling": self.settings.max_external_embedding_calls,
            "new_unique_queries": len(queries),
            "query_cache_matches": matches,
            "missing_query_embeddings": len(keys) - matches,
            "maximum_new_calls": len(keys) - matches,
            "remaining_capacity": max(0, self.settings.max_external_embedding_calls - current),
        }

    def _authorize(self, cases: tuple[RetrievalGroundTruthCase, ...]) -> None:
        if not self.settings.allow_external_calls or not self.settings.embedding_api_key:
            raise ValueError("external query embedding authorization is incomplete")
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
        missing = sum(self.session.get(QueryEmbeddingCacheRecord, key) is None for key in keys)
        if missing > self.preflight()["remaining_capacity"]:
            raise ValueError("reranking benchmark would exceed embedding call ceiling")

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
            raise ValueError("reranking calibration is already locked and immutable")
        if self._runs("calibration"):
            raise ValueError("reranking calibration was already executed")
        metrics = self._run_partition("calibration", CALIBRATION_CASES)
        dense, candidate = metrics[MODES[0]], metrics[MODES[1]]
        coverage_gain = (
            candidate["all_required_evidence_coverage_at_5"]
            - dense["all_required_evidence_coverage_at_5"]
        )
        three_gain = (
            candidate["three_document_coverage_at_5"] - dense["three_document_coverage_at_5"]
        )
        recall_ok = (
            candidate["required_evidence_recall_at_5"]
            >= dense["required_evidence_recall_at_5"] - 0.02
        )
        exact_ok = (
            candidate["exact_identifier_recall_at_5"]
            >= dense["exact_identifier_recall_at_5"] - 0.05
        )
        runs = self._runs("calibration")
        semantic_dense = runs[MODES[0]].category_metrics["semantic_paraphrase"][
            "all_required_evidence_coverage_at_5"
        ]
        semantic_candidate = runs[MODES[1]].category_metrics["semantic_paraphrase"][
            "all_required_evidence_coverage_at_5"
        ]
        semantic_ok = semantic_candidate >= semantic_dense - 0.05
        security_ok = candidate["acl_safety"] == 1.0 and candidate["version_correctness"] == 1.0
        material = coverage_gain >= 0.10 or three_gain >= 0.20
        selected = (
            MODES[1]
            if material and recall_ok and exact_ok and semantic_ok and security_ok
            else MODES[0]
        )
        record = self.session.get(RerankingBenchmarkRecord, DATASET_ID)
        record.selected_mode = selected
        record.calibration_metrics = metrics
        record.selection_reason = (
            f"coverage gain={coverage_gain:.6f}; three-document gain={three_gain:.6f}; "
            f"recall={recall_ok}; exact-ID={exact_ok}; semantic={semantic_ok}; "
            f"security={security_ok}."
        )
        record.locked_at = datetime.now(UTC)
        self.session.commit()
        return self.status(include_cases=True)

    def run_holdout(self) -> dict[str, Any]:
        record = self.session.get(RerankingBenchmarkRecord, DATASET_ID)
        if not record or not record.locked_at or not record.selected_mode:
            raise ValueError("reranking selection must be locked before holdout")
        self._verify(record, self._corpus_identity())
        if record.holdout_started_at is not None:
            raise ValueError("reranking holdout is one-shot and already started")
        record.holdout_started_at = datetime.now(UTC)
        self.session.commit()
        self._run_partition("holdout", HOLDOUT_CASES)
        record = self.session.get(RerankingBenchmarkRecord, DATASET_ID)
        record.holdout_completed_at = datetime.now(UTC)
        self.session.commit()
        return self.status(include_cases=True)

    def _run_partition(
        self, partition: str, cases: tuple[RetrievalGroundTruthCase, ...]
    ) -> dict[str, dict[str, Any]]:
        self._authorize(cases)
        judge_before = sum(
            item.external_judge_calls or 0
            for item in self.session.scalars(select(ExperimentRunRecord)).all()
        )
        provider = self._embedding_provider()
        dense = Retriever(self.session, provider, SEMANTIC_INDEX_IDENTITY)
        cache_path = (
            Path.home() / ".cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2"
        )
        download_occurred = not cache_path.exists()
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        reranker: Reranker = self.reranker_factory()
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record = self.session.get(RerankingBenchmarkRecord, DATASET_ID)
        record.model_revision = reranker.resolved_revision
        raw_model_path = getattr(reranker, "model_path", None)
        model_path = Path(raw_model_path) if raw_model_path else None
        model_size = (
            sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file())
            if model_path is not None and model_path.is_dir()
            else 0
        )
        previous_download = bool((record.model_metadata or {}).get("download_occurred"))
        record.model_metadata = {
            "model_id": reranker.model_id,
            "resolved_revision": reranker.resolved_revision,
            "backend": reranker.backend,
            "device": reranker.device,
            "download_occurred": previous_download or download_occurred,
            "model_size_bytes": model_size,
            "process_max_rss_before": rss_before,
            "process_max_rss_after": rss_after,
            "platform": os.uname().machine,
        }
        self.session.commit()

        rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
        pool_rows: list[dict[str, Any]] = []
        rerank_latencies: list[float] = []
        sort_latencies: list[float] = []
        dense_latencies: list[float] = []
        embedding_latencies: list[float] = []
        lookup_latencies: list[float] = []
        pair_counts: list[int] = []
        cache_hits = cache_misses = embedding_calls = embedding_tokens = 0
        unauthorized_to_reranker = 0
        for case in cases:
            embedding = dense.query_embedding_cache.get_or_embed(case.question)
            self.session.commit()
            cache_hits += int(embedding.cache_hit)
            cache_misses += int(not embedding.cache_hit)
            embedding_calls += embedding.external_calls
            embedding_tokens += embedding.input_tokens
            dense_started = time.perf_counter()
            candidates = dense.retrieve_with_embedding(
                embedding, top_k=20, score_threshold=0.28, principal=_principal(case)
            )
            dense_ms = (time.perf_counter() - dense_started) * 1000
            if case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                unauthorized_to_reranker += sum(
                    item.document_id in set(case.forbidden_document_ids) for item in candidates
                )
            reranked = reranker.rerank(case.question, candidates)
            reranked_results = [
                replace(
                    item.result,
                    rank=item.reranked_rank,
                    score=item.reranker_score,
                    retrieval_source="dense_cross_encoder_rerank",
                )
                for item in reranked
            ]
            dense_top5 = candidates[:5]
            reranked_top5 = reranked_results[:5]
            pool_metrics = _candidate_pool_case_metrics(case, candidates)
            pool_metrics["candidate_pool_size"] = len(candidates)
            pool_rows.append(
                {
                    "category": case.category,
                    "expected_answerability": case.expected_answerability,
                    "metrics": pool_metrics,
                }
            )
            movement = self._movement(case, candidates, reranked_results)
            dense_case_metrics = retrieval_case_metrics(case, dense_top5)
            reranked_case_metrics = retrieval_case_metrics(case, reranked_top5)
            rerank_trace = [
                {
                    "chunk_id": item.result.chunk_id,
                    "document_id": item.result.document_id,
                    "version": item.result.version,
                    "original_dense_rank": item.original_dense_rank,
                    "dense_score": item.dense_score,
                    "reranker_score": item.reranker_score,
                    "reranked_rank": item.reranked_rank,
                }
                for item in reranked
            ]
            for mode, results in ((MODES[0], dense_top5), (MODES[1], reranked_top5)):
                metrics = dense_case_metrics if mode == MODES[0] else reranked_case_metrics
                rows[mode].append(
                    {
                        "case_id": case.case_id,
                        "question": case.question,
                        "category": case.category,
                        "expected_answerability": case.expected_answerability,
                        "required_document_ids": list(case.required_document_ids),
                        "required_version_ids": case.required_version_ids,
                        "forbidden_document_ids": list(case.forbidden_document_ids),
                        "candidate_pool_size": len(candidates),
                        "candidate_trace": [self._dense_trace(item) for item in candidates],
                        "reranking_trace": rerank_trace,
                        "retrieval_trace": [self._dense_trace(item) for item in results],
                        "metrics": metrics,
                        "candidate_pool_metrics": pool_metrics,
                        "movement": movement,
                        "dense_top5_coverage": dense_case_metrics[
                            "all_required_evidence_coverage_at_5"
                        ],
                        "reranked_top5_coverage": reranked_case_metrics[
                            "all_required_evidence_coverage_at_5"
                        ],
                        "rerankable": bool(
                            set(case.required_document_ids)
                            <= {item.document_id for item in candidates}
                        )
                        if case.expected_answerability
                        else False,
                        "failure_types": self._failures(case, mode, results, candidates, movement),
                    }
                )
            embedding_latencies.append(embedding.embedding_latency_ms)
            lookup_latencies.append(embedding.cache_lookup_latency_ms)
            dense_latencies.append(dense_ms)
            rerank_latencies.append(getattr(reranker, "last_inference_latency_ms", 0.0))
            sort_latencies.append(getattr(reranker, "last_sort_latency_ms", 0.0))
            pair_counts.append(len(candidates))

        if unauthorized_to_reranker:
            raise RuntimeError("unauthorized content reached the Cross-Encoder")
        judge_after = sum(
            item.external_judge_calls or 0
            for item in self.session.scalars(select(ExperimentRunRecord)).all()
        )
        if judge_after != judge_before:
            raise RuntimeError("retrieval-only reranking benchmark changed judge usage")
        pool_aggregate = _aggregate_candidate_pool_metrics(pool_rows)
        pool_aggregate["mean_candidate_pool_size"] = mean(pair_counts)
        latency = {
            "query_embedding_ms": mean(embedding_latencies),
            "cache_lookup_ms": mean(lookup_latencies),
            "dense_top20_ms": mean(dense_latencies),
            "reranker_inference_mean_ms": mean(rerank_latencies),
            "reranker_inference_p50_ms": median(rerank_latencies),
            "reranker_inference_p95_ms": _percentile(rerank_latencies, 0.95),
            "final_sort_mean_ms": mean(sort_latencies),
            "dense_total_ms": mean(
                [
                    a + b + c
                    for a, b, c in zip(
                        embedding_latencies, lookup_latencies, dense_latencies, strict=True
                    )
                ]
            ),
            "reranked_total_ms": mean(
                [
                    a + b + c + d + e
                    for a, b, c, d, e in zip(
                        embedding_latencies,
                        lookup_latencies,
                        dense_latencies,
                        rerank_latencies,
                        sort_latencies,
                        strict=True,
                    )
                ]
            ),
        }
        output: dict[str, dict[str, Any]] = {}
        for mode in MODES:
            aggregate = aggregate_retrieval_metrics(rows[mode])
            subset = [item for item in rows[mode] if item["rerankable"]]
            subset_metrics = aggregate_retrieval_metrics(subset)
            movement_metrics = self._aggregate_movement(rows[mode])
            usage = {
                "new_query_embedding_calls": embedding_calls if mode == MODES[0] else 0,
                "embedding_tokens": embedding_tokens,
                "query_cache_hits": cache_hits,
                "query_cache_misses": cache_misses,
                "document_embedding_calls": 0,
                "judge_calls": 0,
                "external_reranker_calls": 0,
                "local_reranking_requests": len(cases) if mode == MODES[1] else 0,
                "local_pairs_scored": sum(pair_counts) if mode == MODES[1] else 0,
                "mean_pairs_per_query": mean(pair_counts) if mode == MODES[1] else 0,
                "unauthorized_chunks_sent_to_reranker": 0,
            }
            self.session.add(
                RerankingBenchmarkRunRecord(
                    dataset_id=DATASET_ID,
                    partition=partition,
                    mode=mode,
                    metrics=aggregate,
                    category_metrics=category_retrieval_metrics(rows[mode]),
                    candidate_pool_metrics=pool_aggregate,
                    rerankable_subset_metrics=subset_metrics,
                    movement_metrics=movement_metrics,
                    case_results=rows[mode],
                    latency=latency,
                    usage=usage,
                )
            )
            output[mode] = aggregate
        self.session.commit()
        return output

    @staticmethod
    def _dense_trace(item: RetrievalResult) -> dict[str, Any]:
        return {
            "chunk_id": item.chunk_id,
            "document_id": item.document_id,
            "document_version_id": item.document_version_id,
            "version": item.version,
            "rank": item.rank,
            "dense_score": item.dense_score,
        }

    @staticmethod
    def _movement(
        case: RetrievalGroundTruthCase,
        dense: list[RetrievalResult],
        reranked: list[RetrievalResult],
    ) -> dict[str, Any]:
        required = set(case.required_document_ids)
        movements = []
        for document_id in required:
            original = _first_rank(dense, document_id)
            revised = _first_rank(reranked, document_id)
            movements.append(
                {
                    "document_id": document_id,
                    "original_dense_rank": original,
                    "reranked_rank": revised,
                    "rank_delta": original - revised if original and revised else None,
                    "promoted_into_top5": bool(
                        original and original > 5 and revised and revised <= 5
                    ),
                    "demoted_out_of_top5": bool(
                        original and original <= 5 and (not revised or revised > 5)
                    ),
                }
            )
        dense_top = {item.chunk_id for item in dense[:5]}
        reranked_top = {item.chunk_id for item in reranked[:5]}
        irrelevant_removed = sum(
            item.chunk_id in dense_top - reranked_top and item.document_id not in required
            for item in dense[:5]
        )
        return {"required_evidence": movements, "irrelevant_chunks_removed": irrelevant_removed}

    @staticmethod
    def _aggregate_movement(rows: list[dict[str, Any]]) -> dict[str, Any]:
        movements = [m for row in rows for m in row["movement"]["required_evidence"]]
        deltas = [item["rank_delta"] for item in movements if item["rank_delta"] is not None]
        three = [row for row in rows if row["category"] == "multidoc_three"]
        return {
            "required_evidence_promoted_into_top5": sum(
                item["promoted_into_top5"] for item in movements
            ),
            "required_evidence_demoted_out_of_top5": sum(
                item["demoted_out_of_top5"] for item in movements
            ),
            "irrelevant_chunks_removed_from_top5": sum(
                row["movement"]["irrelevant_chunks_removed"] for row in rows
            ),
            "mean_required_evidence_rank_delta": mean(deltas) if deltas else 0.0,
            "three_document_fixed": sum(
                row["dense_top5_coverage"] == 0 and row["reranked_top5_coverage"] == 1
                for row in three
            ),
            "three_document_worsened": sum(
                row["dense_top5_coverage"] == 1 and row["reranked_top5_coverage"] == 0
                for row in three
            ),
            "three_document_still_failed": sum(row["reranked_top5_coverage"] == 0 for row in three),
            "three_document_outside_top20": sum(not row["rerankable"] for row in three),
        }

    @staticmethod
    def _failures(
        case: RetrievalGroundTruthCase,
        mode: str,
        results: list[RetrievalResult],
        candidates: list[RetrievalResult],
        movement: dict[str, Any],
    ) -> list[str]:
        if not case.expected_answerability:
            return []
        required = set(case.required_document_ids)
        candidate_docs = {item.document_id for item in candidates}
        result_docs = {item.document_id for item in results}
        failures = []
        if not required <= candidate_docs:
            failures.extend(["CANDIDATE_GENERATION_MISS", "REQUIRED_EVIDENCE_OUTSIDE_TOP20"])
        elif not required <= result_docs:
            failures.append("RANKING_OUTSIDE_TOP5")
            if mode == MODES[1]:
                failures.append("RERANKER_FAILED_TO_PROMOTE")
        if mode == MODES[1] and any(
            item["demoted_out_of_top5"] for item in movement["required_evidence"]
        ):
            failures.append("RERANKER_DEMOTED_REQUIRED_EVIDENCE")
            if case.category == "exact_identifier":
                failures.append("EXACT_IDENTIFIER_RERANK_REGRESSION")
            if case.category == "semantic_paraphrase":
                failures.append("SEMANTIC_RERANK_REGRESSION")
            if case.category == "near_duplicate":
                failures.append("NEAR_DUPLICATE_RERANK_FAILURE")
        if case.category == "multidoc_three" and not required <= result_docs:
            failures.append("THREE_DOCUMENT_COVERAGE_FAILURE")
        return list(dict.fromkeys(failures))

    def _runs(self, partition: str) -> dict[str, RerankingBenchmarkRunRecord]:
        values = self.session.scalars(
            select(RerankingBenchmarkRunRecord).where(
                RerankingBenchmarkRunRecord.dataset_id == DATASET_ID,
                RerankingBenchmarkRunRecord.partition == partition,
            )
        ).all()
        return {item.mode: item for item in values}

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(RerankingBenchmarkRecord, DATASET_ID)
        payload: dict[str, Any] = {
            "dataset_id": DATASET_ID,
            "dataset_hash": DATASET_HASH,
            "case_count": len(DATASET.cases),
            "category_distribution": dict(
                sorted(Counter(item.category for item in DATASET.cases).items())
            ),
            "split_seed": SPLIT_SEED,
            "split_identity": SPLIT_IDENTITY,
            "calibration_count": 40,
            "holdout_count": 20,
            "configuration": CONFIGURATION,
            "selection_policy": SELECTION_POLICY,
            "preflight": self.preflight(),
            "selected_mode": None,
            "holdout_started": False,
            "holdout_completed": False,
        }
        if not record:
            return payload
        self._verify(record)
        payload.update(
            {
                "corpus_identity": record.corpus_identity,
                "selected_mode": record.selected_mode,
                "selection_reason": record.selection_reason,
                "locked_at": record.locked_at,
                "holdout_started": record.holdout_started_at is not None,
                "holdout_completed": record.holdout_completed_at is not None,
                "model_revision": record.model_revision,
                "model_metadata": record.model_metadata,
            }
        )
        for partition in ("calibration", "holdout"):
            runs = self._runs(partition)
            if runs:
                payload[partition] = {
                    mode: {
                        "id": run.id,
                        "metrics": run.metrics,
                        "category_metrics": run.category_metrics,
                        "candidate_pool_metrics": run.candidate_pool_metrics,
                        "rerankable_subset_metrics": run.rerankable_subset_metrics,
                        "movement_metrics": run.movement_metrics,
                        "latency": run.latency,
                        "usage": run.usage,
                        **({"cases": run.case_results} if include_cases else {}),
                    }
                    for mode, run in runs.items()
                }
        return payload
