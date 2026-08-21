"""Canonical hybrid retrieval + Cross-Encoder pool shared by serving and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_workbench.retrieval.filters import RetrievalFilters
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.temporal import TemporalScopePlan, plan_temporal_scope
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.runtime.config import ProductionRagConfig
from rag_workbench.security.permissions import Principal


def _ranking_candidate(item: Any) -> dict[str, Any]:
    result = item.result if hasattr(item, "result") else item
    return {
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "document_version_id": result.document_version_id,
        "text": result.text,
        "rank": getattr(item, "reranked_rank", result.rank),
        "score": getattr(item, "reranker_score", result.score),
        "source": result.source,
        "source_type": result.source_type,
        "title": result.title,
        "version": result.version,
        "page": result.page,
        "section": result.section,
        "metadata": result.metadata,
        "retrieval_source": result.retrieval_source,
        "dense_score": result.dense_score,
        "lexical_score": result.lexical_score,
        "fusion_score": result.fusion_score,
        "found_by_dense": result.found_by_dense,
        "found_by_bm25": result.found_by_bm25,
    }


@dataclass(frozen=True)
class HybridEvidencePool:
    temporal_scope: TemporalScopePlan
    dense: tuple[RetrievalResult, ...]
    bm25: tuple[RetrievalResult, ...]
    fused: tuple[RetrievalResult, ...]
    top15: tuple[dict[str, Any], ...]
    top20: tuple[dict[str, Any], ...]
    embedding_cache_hit: bool
    embedding_external_calls: int
    dense_rank: dict[str, int]
    bm25_rank: dict[str, int]
    rrf_rank: dict[str, int]


def retrieve_evidence_pool(
    *,
    dense: Any,
    bm25: Any,
    reranker: Any,
    question: str,
    principal: Principal,
    config: ProductionRagConfig,
    temporal_scope: TemporalScopePlan | None = None,
) -> HybridEvidencePool:
    """Dense+BM25→RRF→cap→CE→Top-15 using the validated production depths."""
    scope = temporal_scope or plan_temporal_scope(question)
    filters = RetrievalFilters(temporal_scope=scope)
    embedding = dense.query_embedding_cache.get_or_embed(question)
    dense_hits = dense.retrieve_with_embedding(
        embedding,
        top_k=config.dense_top_k,
        score_threshold=config.dense_score_threshold,
        filters=filters,
        principal=principal,
    )
    bm25_hits = bm25.retrieve(
        question,
        top_k=config.bm25_top_k,
        filters=filters,
        principal=principal,
    )
    fused = reciprocal_rank_fusion(
        dense_hits,
        bm25_hits,
        top_k=10_000,
        rrf_k=config.rrf_k,
    )[: config.fused_candidate_cap]
    reranked = reranker.rerank(question, fused)
    ranked = [
        _ranking_candidate(item) | {"rank": item.reranked_rank, "score": item.reranker_score}
        for item in reranked
    ]
    top20 = ranked[:20]
    top15 = ranked[: config.cross_encoder_top_k]
    return HybridEvidencePool(
        temporal_scope=scope,
        dense=tuple(dense_hits),
        bm25=tuple(bm25_hits),
        fused=tuple(fused),
        top15=tuple(top15),
        top20=tuple(top20),
        embedding_cache_hit=bool(embedding.cache_hit),
        embedding_external_calls=int(embedding.external_calls),
        dense_rank={item.chunk_id: index + 1 for index, item in enumerate(dense_hits)},
        bm25_rank={item.chunk_id: index + 1 for index, item in enumerate(bm25_hits)},
        rrf_rank={item.chunk_id: index + 1 for index, item in enumerate(fused)},
    )
