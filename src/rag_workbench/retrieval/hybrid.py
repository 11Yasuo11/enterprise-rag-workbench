from __future__ import annotations

import time
from dataclasses import dataclass, replace

from rag_workbench.retrieval.base import RetrievalMode
from rag_workbench.retrieval.bm25 import BM25Retriever
from rag_workbench.retrieval.filters import RetrievalFilters
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.retrieval.temporal import plan_temporal_scope
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal


@dataclass(frozen=True)
class HybridTiming:
    query_embedding_latency_ms: float = 0.0
    embedding_cache_lookup_latency_ms: float = 0.0
    dense_retrieval_latency_ms: float = 0.0
    bm25_retrieval_latency_ms: float = 0.0
    rrf_fusion_latency_ms: float = 0.0
    total_retrieval_latency_ms: float = 0.0
    query_embedding_cache_hit: bool = False
    external_embedding_calls: int = 0


def reciprocal_rank_fusion(
    dense: list[RetrievalResult],
    lexical: list[RetrievalResult],
    *,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    by_chunk: dict[str, dict[str, object]] = {}
    for branch, results in (("dense", dense), ("bm25", lexical)):
        for item in results:
            state = by_chunk.setdefault(
                item.chunk_id,
                {"item": item, "score": 0.0, "dense": None, "bm25": None},
            )
            state["score"] = float(state["score"]) + 1.0 / (rrf_k + item.rank)
            state[branch] = item
    ordered = sorted(
        by_chunk.values(),
        key=lambda state: (
            -float(state["score"]),
            min(
                item.rank
                for item in (state["dense"], state["bm25"])
                if isinstance(item, RetrievalResult)
            ),
            state["item"].chunk_id,
        ),
    )
    fused: list[RetrievalResult] = []
    for rank, state in enumerate(ordered[:top_k], start=1):
        dense_item = state["dense"]
        lexical_item = state["bm25"]
        base = dense_item if isinstance(dense_item, RetrievalResult) else lexical_item
        assert isinstance(base, RetrievalResult)
        fusion_score = round(float(state["score"]), 9)
        fused.append(
            replace(
                base,
                rank=rank,
                score=fusion_score,
                retrieval_source="hybrid_rrf",
                dense_score=(
                    dense_item.dense_score
                    if isinstance(dense_item, RetrievalResult)
                    else None
                ),
                lexical_score=(
                    lexical_item.lexical_score
                    if isinstance(lexical_item, RetrievalResult)
                    else None
                ),
                fusion_score=fusion_score,
                found_by_dense=isinstance(dense_item, RetrievalResult),
                found_by_bm25=isinstance(lexical_item, RetrievalResult),
            )
        )
    return fused


class HybridRetriever:
    mode = RetrievalMode.HYBRID_RRF

    def __init__(
        self,
        dense: Retriever,
        bm25: BM25Retriever,
        *,
        dense_candidate_depth: int = 20,
        bm25_candidate_depth: int = 20,
        final_top_k: int = 5,
        dense_threshold: float = 0.28,
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.dense_candidate_depth = dense_candidate_depth
        self.bm25_candidate_depth = bm25_candidate_depth
        self.final_top_k = final_top_k
        self.dense_threshold = dense_threshold
        self.rrf_k = rrf_k
        self.last_timing = HybridTiming()
        self.last_dense_candidates: list[RetrievalResult] = []
        self.last_bm25_candidates: list[RetrievalResult] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: RetrievalFilters | None = None,
        principal: Principal | None = None,
    ) -> list[RetrievalResult]:
        del score_threshold  # The frozen threshold is branch-local, never an RRF cutoff.
        if top_k != self.final_top_k:
            raise ValueError(f"hybrid final top_k is frozen at {self.final_top_k}")
        if principal is None:
            raise ValueError("A principal is required; retrieval is never authorization-free")
        if filters is None:
            filters = RetrievalFilters(temporal_scope=plan_temporal_scope(query))
        elif filters.temporal_scope is None:
            filters = replace(filters, temporal_scope=plan_temporal_scope(query))
        total_started = time.perf_counter()
        embedding = self.dense.query_embedding_cache.get_or_embed(query)
        dense_started = time.perf_counter()
        dense = self.dense.retrieve_with_embedding(
            embedding,
            top_k=self.dense_candidate_depth,
            score_threshold=self.dense_threshold,
            filters=filters,
            principal=principal,
        )
        dense_latency = (time.perf_counter() - dense_started) * 1000
        bm25_started = time.perf_counter()
        lexical = self.bm25.retrieve(
            query,
            top_k=self.bm25_candidate_depth,
            filters=filters,
            principal=principal,
        )
        bm25_latency = (time.perf_counter() - bm25_started) * 1000
        fusion_started = time.perf_counter()
        fused = reciprocal_rank_fusion(dense, lexical, top_k=top_k, rrf_k=self.rrf_k)
        fusion_latency = (time.perf_counter() - fusion_started) * 1000
        self.last_dense_candidates = dense
        self.last_bm25_candidates = lexical
        self.last_timing = HybridTiming(
            query_embedding_latency_ms=embedding.embedding_latency_ms,
            embedding_cache_lookup_latency_ms=embedding.cache_lookup_latency_ms,
            dense_retrieval_latency_ms=dense_latency,
            bm25_retrieval_latency_ms=bm25_latency,
            rrf_fusion_latency_ms=fusion_latency,
            total_retrieval_latency_ms=(time.perf_counter() - total_started) * 1000,
            query_embedding_cache_hit=embedding.cache_hit,
            external_embedding_calls=embedding.external_calls,
        )
        return fused
