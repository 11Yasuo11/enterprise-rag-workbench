import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.config import get_settings
from rag_workbench.db.models import Chunk, Document, DocumentVersion
from rag_workbench.experiments.configs import IngestionConfig, index_identity_for
from rag_workbench.providers.embeddings import EmbeddingProvider
from rag_workbench.retrieval.base import RetrievalMode
from rag_workbench.retrieval.filters import RetrievalFilters, apply_temporal_version_filter
from rag_workbench.retrieval.query_embedding_cache import (
    QueryEmbeddingCache,
    QueryEmbeddingResult,
)
from rag_workbench.retrieval.temporal import plan_temporal_scope
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal, apply_document_acl


def meets_score_threshold(similarity: float, threshold: float | None) -> bool:
    """Cosine similarity is `1 - cosine_distance`; larger values are better."""
    return threshold is None or similarity >= threshold


@dataclass(frozen=True)
class RetrievalTiming:
    query_embedding_latency_ms: float = 0.0
    embedding_cache_lookup_latency_ms: float = 0.0
    vector_search_latency_ms: float = 0.0
    acl_filter_latency_ms: float = 0.0
    query_embedding_cache_hit: bool = False
    external_embedding_calls: int = 0


class Retriever:
    mode = RetrievalMode.DENSE

    def __init__(
        self,
        session: Session,
        embedding_provider: EmbeddingProvider,
        index_identity: str | None = None,
    ) -> None:
        self.session = session
        self.embedding_provider = embedding_provider
        self.query_embedding_cache = QueryEmbeddingCache(session, embedding_provider)
        self.last_timing = RetrievalTiming()
        settings = get_settings()
        self.index_identity = index_identity or index_identity_for(
            settings.corpus_version,
            IngestionConfig(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                embedding_provider=embedding_provider.provider_name,
                embedding_model=embedding_provider.model_name,
                embedding_dimension=embedding_provider.dimension,
                embedding_version=embedding_provider.version,
            ),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = 0.2,
        filters: RetrievalFilters | None = None,
        principal: Principal | None = None,
    ) -> list[RetrievalResult]:
        if principal is None:
            raise ValueError("A principal is required; retrieval is never authorization-free")
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        embedding = self.query_embedding_cache.get_or_embed(query)
        if filters is None:
            filters = RetrievalFilters(temporal_scope=plan_temporal_scope(query))
        return self.retrieve_with_embedding(
            embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
            principal=principal,
        )

    def retrieve_with_embedding(
        self,
        embedding: QueryEmbeddingResult,
        *,
        top_k: int = 5,
        score_threshold: float | None = 0.2,
        filters: RetrievalFilters | None = None,
        principal: Principal | None = None,
    ) -> list[RetrievalResult]:
        if principal is None:
            raise ValueError("A principal is required; retrieval is never authorization-free")
        query_vector = embedding.vector
        distance = Chunk.embedding.cosine_distance(query_vector)
        provider_names = {self.embedding_provider.provider_name}
        if self.embedding_provider.provider_name in {"openai", "openai-compatible"}:
            provider_names = {"openai", "openai-compatible"}
        statement = (
            select(Chunk, Document, DocumentVersion, distance.label("distance"))
            .join(Document, Chunk.document_fk == Document.id)
            .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
            .where(
                Chunk.index_identity == self.index_identity,
                Chunk.embedding_provider.in_(tuple(sorted(provider_names))),
                Chunk.embedding_model == self.embedding_provider.model_name,
                Chunk.embedding_version == self.embedding_provider.version,
                Chunk.embedding_dimension == self.embedding_provider.dimension,
            )
        )
        acl_started = time.perf_counter()
        statement = apply_document_acl(statement, principal)
        statement = apply_temporal_version_filter(
            statement, filters.temporal_scope if filters else None
        )
        acl_filter_latency_ms = (time.perf_counter() - acl_started) * 1000
        if filters:
            if filters.document_ids:
                statement = statement.where(Document.document_id.in_(filters.document_ids))
            if filters.source_types:
                statement = statement.where(Document.source_type.in_(filters.source_types))
        if score_threshold is not None:
            statement = statement.where(distance <= 1 - score_threshold)
        statement = statement.order_by(distance, Chunk.id).limit(top_k)
        search_started = time.perf_counter()
        rows = self.session.execute(statement).all()
        vector_search_latency_ms = (time.perf_counter() - search_started) * 1000
        self.last_timing = RetrievalTiming(
            query_embedding_latency_ms=embedding.embedding_latency_ms,
            embedding_cache_lookup_latency_ms=embedding.cache_lookup_latency_ms,
            vector_search_latency_ms=vector_search_latency_ms,
            acl_filter_latency_ms=acl_filter_latency_ms,
            query_embedding_cache_hit=embedding.cache_hit,
            external_embedding_calls=embedding.external_calls,
        )
        return [
            RetrievalResult(
                chunk_id=chunk.id,
                document_id=document.document_id,
                document_version_id=version.id,
                text=chunk.text,
                rank=rank,
                score=round(1.0 - float(row_distance), 6),
                source=document.source,
                source_type=document.source_type,
                title=document.title,
                version=version.version,
                page=chunk.page,
                section=chunk.section,
                metadata=chunk.metadata_,
                retrieval_source="dense",
                dense_score=round(1.0 - float(row_distance), 6),
                found_by_dense=True,
            )
            for rank, (chunk, document, version, row_distance) in enumerate(rows, start=1)
        ]
