"""Deterministic identity reranker for zero-API / hashing local tests only."""

from __future__ import annotations

from rag_workbench.reranking.base import RerankedResult
from rag_workbench.retrieval.vector_search import RetrievalResult


class IdentityReranker:
    """Preserves RRF order. Not for production semantic quality."""

    model_id = "identity-reranker-test"
    backend = "identity"
    external_calls = 0
    resolved_revision = "test"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.last_inference_latency_ms = 0.0
        self.last_sort_latency_ms = 0.0

    def rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RerankedResult]:
        del query
        return [
            RerankedResult(
                result=candidate,
                original_dense_rank=candidate.rank,
                dense_score=candidate.dense_score or candidate.score,
                reranker_score=float(len(candidates) - index),
                reranked_rank=index + 1,
            )
            for index, candidate in enumerate(candidates)
        ]
