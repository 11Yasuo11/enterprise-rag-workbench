from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.db.models import Chunk, Document, DocumentVersion
from rag_workbench.retrieval.base import RetrievalMode
from rag_workbench.retrieval.filters import RetrievalFilters
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal, apply_document_acl

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-_.:/][^\W_]+)+|[^\W_]+", re.UNICODE)
BM25_VERSION = "bm25-okapi-v1"


def tokenize_bm25(text: str) -> tuple[str, ...]:
    """NFKC/case normalized words with exact enterprise identifiers preserved."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalized):
        token = match.group(0)
        tokens.append(token)
        if any(separator in token for separator in "-_.:/"):
            tokens.extend(part for part in re.split(r"[-_.:/]+", token) if part)
    return tuple(tokens)


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.2
    b: float = 0.75
    version: str = BM25_VERSION


@dataclass(frozen=True)
class BM25Timing:
    acl_filter_latency_ms: float = 0.0
    index_build_latency_ms: float = 0.0
    lexical_search_latency_ms: float = 0.0
    index_size_bytes: int = 0
    candidate_count: int = 0


@dataclass(frozen=True)
class _Candidate:
    chunk: Chunk
    document: Document
    version: DocumentVersion


class BM25Retriever:
    mode = RetrievalMode.LEXICAL_BM25

    def __init__(
        self,
        session: Session,
        *,
        index_identity: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_version: str,
        embedding_dimension: int,
        config: BM25Config | None = None,
    ) -> None:
        self.session = session
        self.index_identity = index_identity
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_version = embedding_version
        self.embedding_dimension = embedding_dimension
        self.config = config or BM25Config()
        self.last_timing = BM25Timing()
        self.last_index_identity = ""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: RetrievalFilters | None = None,
        principal: Principal | None = None,
    ) -> list[RetrievalResult]:
        del score_threshold  # Cosine thresholds are never meaningful for BM25.
        if principal is None:
            raise ValueError("A principal is required; retrieval is never authorization-free")
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")

        statement = (
            select(Chunk, Document, DocumentVersion)
            .join(Document, Chunk.document_fk == Document.id)
            .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
            .where(
                DocumentVersion.is_active.is_(True),
                Chunk.index_identity == self.index_identity,
                Chunk.embedding_provider == self.embedding_provider,
                Chunk.embedding_model == self.embedding_model,
                Chunk.embedding_version == self.embedding_version,
                Chunk.embedding_dimension == self.embedding_dimension,
            )
        )
        # Tenant, ACL, and active-version predicates are part of the SQL candidate set.
        statement = apply_document_acl(statement, principal)
        if filters:
            if filters.document_ids:
                statement = statement.where(Document.document_id.in_(filters.document_ids))
            if filters.source_types:
                statement = statement.where(Document.source_type.in_(filters.source_types))
        statement = statement.order_by(Chunk.id)
        acl_started = time.perf_counter()
        candidates = [_Candidate(*row) for row in self.session.execute(statement).all()]
        acl_latency = (time.perf_counter() - acl_started) * 1000

        build_started = time.perf_counter()
        tokenized = [tokenize_bm25(item.chunk.text) for item in candidates]
        term_frequencies = [Counter(tokens) for tokens in tokenized]
        document_frequencies = Counter(
            token for frequencies in term_frequencies for token in frequencies
        )
        lengths = [len(tokens) for tokens in tokenized]
        average_length = sum(lengths) / len(lengths) if lengths else 0.0
        serialized_index = json.dumps(
            [
                [item.chunk.id, sorted(frequencies.items())]
                for item, frequencies in zip(candidates, term_frequencies, strict=True)
            ],
            separators=(",", ":"),
        ).encode()
        self.last_index_identity = hashlib.sha256(
            self.config.version.encode() + b":" + serialized_index
        ).hexdigest()
        build_latency = (time.perf_counter() - build_started) * 1000

        search_started = time.perf_counter()
        query_terms = Counter(tokenize_bm25(query))
        scored: list[tuple[float, _Candidate]] = []
        count = len(candidates)
        for item, frequencies, length in zip(
            candidates, term_frequencies, lengths, strict=True
        ):
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = document_frequencies[term]
                inverse_document_frequency = math.log(
                    1.0 + (count - df + 0.5) / (df + 0.5)
                )
                normalization = self.config.k1 * (
                    1.0
                    - self.config.b
                    + self.config.b * length / average_length
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * (self.config.k1 + 1.0)
                    / (frequency + normalization)
                    * query_frequency
                )
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], pair[1].chunk.id))
        search_latency = (time.perf_counter() - search_started) * 1000
        self.last_timing = BM25Timing(
            acl_filter_latency_ms=acl_latency,
            index_build_latency_ms=build_latency,
            lexical_search_latency_ms=search_latency,
            index_size_bytes=len(serialized_index),
            candidate_count=count,
        )
        return [
            RetrievalResult(
                chunk_id=item.chunk.id,
                document_id=item.document.document_id,
                document_version_id=item.version.id,
                text=item.chunk.text,
                rank=rank,
                score=round(score, 6),
                source=item.document.source,
                source_type=item.document.source_type,
                title=item.document.title,
                version=item.version.version,
                page=item.chunk.page,
                section=item.chunk.section,
                metadata=item.chunk.metadata_,
                retrieval_source="bm25",
                lexical_score=round(score, 6),
                found_by_bm25=True,
            )
            for rank, (score, item) in enumerate(scored[:top_k], start=1)
        ]
