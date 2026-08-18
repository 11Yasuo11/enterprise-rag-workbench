import hashlib
import json
import time
import unicodedata
from dataclasses import dataclass

from sqlalchemy.orm import Session

from rag_workbench.db.models import STORED_EMBEDDING_DIMENSION, QueryEmbeddingCacheRecord
from rag_workbench.providers.embeddings import EmbeddingProvider


def normalize_query_text(query: str) -> str:
    """Normalize representation without changing query case or semantics."""
    return " ".join(unicodedata.normalize("NFKC", query).split())


def query_embedding_cache_key(
    query: str,
    *,
    provider: str,
    model: str,
    version: str,
    dimension: int,
) -> str:
    payload = {
        "dimension": dimension,
        "model": model,
        "provider": provider,
        "query": normalize_query_text(query),
        "version": version,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QueryEmbeddingResult:
    vector: list[float]
    cache_hit: bool
    cache_lookup_latency_ms: float
    embedding_latency_ms: float
    input_tokens: int
    external_calls: int


class QueryEmbeddingCache:
    def __init__(self, session: Session, provider: EmbeddingProvider) -> None:
        self.session = session
        self.provider = provider

    def key_for(self, query: str) -> str:
        return query_embedding_cache_key(
            query,
            provider=self.provider.provider_name,
            model=self.provider.model_name,
            version=self.provider.version,
            dimension=self.provider.dimension,
        )

    def get_or_embed(self, query: str) -> QueryEmbeddingResult:
        normalized = normalize_query_text(query)
        if not normalized:
            raise ValueError("query cannot be empty")
        if self.provider.dimension != STORED_EMBEDDING_DIMENSION:
            raise ValueError("query embedding cache dimension is incompatible with this deployment")
        key = self.key_for(query)
        lookup_started = time.perf_counter()
        cached = self.session.get(QueryEmbeddingCacheRecord, key)
        lookup_latency = (time.perf_counter() - lookup_started) * 1000
        if cached is not None:
            vector = list(cached.embedding)
            if len(vector) != self.provider.dimension:
                raise ValueError("cached query embedding has an incompatible vector dimension")
            return QueryEmbeddingResult(vector, True, lookup_latency, 0.0, 0, 0)

        tokens_before = self.provider.usage.input_tokens
        embedding_started = time.perf_counter()
        vector = self.provider.embed_query(normalized)
        embedding_latency = (time.perf_counter() - embedding_started) * 1000
        if len(vector) != self.provider.dimension:
            raise ValueError("query embedding has an incompatible vector dimension")
        input_tokens = self.provider.usage.input_tokens - tokens_before
        self.session.add(
            QueryEmbeddingCacheRecord(
                cache_key=key,
                normalized_query=normalized,
                embedding_provider=self.provider.provider_name,
                embedding_model=self.provider.model_name,
                embedding_version=self.provider.version,
                embedding_dimension=self.provider.dimension,
                embedding=vector,
                input_tokens=input_tokens,
            )
        )
        self.session.flush()
        return QueryEmbeddingResult(
            vector,
            False,
            lookup_latency,
            embedding_latency,
            input_tokens,
            int(self.provider.is_external),
        )
