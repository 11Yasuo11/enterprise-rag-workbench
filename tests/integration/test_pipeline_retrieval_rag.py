from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.db.models import Chunk, Document, DocumentVersion
from rag_workbench.domain.documents import CanonicalDocument
from rag_workbench.experiments.configs import IngestionConfig, index_identity_for
from rag_workbench.generation import RagService
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
from rag_workbench.ingestion.pipeline import IngestionPipeline
from rag_workbench.providers.embeddings import HashingEmbeddingProvider
from rag_workbench.providers.embeddings.base import EmbeddingUsage
from rag_workbench.providers.llm import ExtractiveGenerationProvider
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.security.permissions import Principal

pytestmark = pytest.mark.integration


def document(
    content: str,
    *,
    document_id: str = "incident-policy",
    version: str = "1",
    effective_at: datetime | None = None,
    visibility: str = "public",
    groups: tuple[str, ...] = (),
) -> CanonicalDocument:
    return CanonicalDocument(
        document_id=document_id,
        title="Incident Policy",
        content=content,
        source=f"{document_id}-{version}.md",
        source_type="markdown",
        version=version,
        tenant_id="test-acme",
        effective_at=effective_at,
        visibility=visibility,
        permission_groups=groups,
    )


def services(session: Session):
    embedding = HashingEmbeddingProvider(64)
    pipeline = IngestionPipeline(session, FixedTokenChunker(FixedTokenConfig(40, 5)), embedding)
    retriever = Retriever(session, embedding, pipeline.index_identity)
    rag = RagService(session, retriever, ContextBuilder(120), ExtractiveGenerationProvider())
    return pipeline, retriever, rag


def test_deduplication_and_version_activation(db_session: Session) -> None:
    pipeline, _, _ = services(db_session)
    old = document(
        "Incidents are reported within sixty minutes.",
        version="2025",
        effective_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    current = document(
        "Current incidents are reported within fifteen minutes.",
        version="2026",
        effective_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    first = pipeline.ingest(old)
    duplicate = pipeline.ingest(old)
    second = pipeline.ingest(current)
    assert first.status == "ingested"
    assert duplicate.status == "duplicate"
    assert duplicate.chunks_created == 0
    assert second.is_active
    versions = db_session.scalars(
        select(DocumentVersion).join(Document).where(Document.document_id == "incident-policy")
    ).all()
    assert {item.version: item.is_active for item in versions} == {"2025": False, "2026": True}
    assert db_session.scalar(select(func.count()).select_from(Chunk)) == 2


def test_permission_filtering_happens_before_retrieval(db_session: Session) -> None:
    pipeline, retriever, _ = services(db_session)
    pipeline.ingest(
        document(
            "The confidential compensation code is HR-COMP-900.",
            document_id="compensation",
            visibility="restricted",
            groups=("hr-leadership",),
        )
    )
    unauthorized = Principal("employee", "test-acme", frozenset({"employees"}))
    authorized = Principal("hr", "test-acme", frozenset({"hr-leadership"}))
    assert retriever.retrieve("confidential compensation HR-COMP-900", principal=unauthorized) == []
    allowed = retriever.retrieve("confidential compensation HR-COMP-900", principal=authorized)
    assert allowed[0].document_id == "compensation"


def test_rag_citations_and_abstention(db_session: Session) -> None:
    pipeline, _, rag = services(db_session)
    pipeline.ingest(document("The incident response identifier is SEC-77."))
    principal = Principal("employee", "test-acme", frozenset({"employees"}))
    answered = rag.query(
        "What is the incident response identifier?", principal, score_threshold=0.2
    )
    assert answered.status == "answered"
    assert "SEC-77" in (answered.answer or "")
    assert answered.citations[0].chunk_id in {item.chunk_id for item in answered.retrieval_results}
    abstained = rag.query(
        "Where is the lunar archaeology laboratory?", principal, score_threshold=0.75
    )
    assert abstained.status == "abstained"
    assert abstained.answer is None
    assert abstained.citations == ()


class CountingEmbeddingProvider:
    provider_name = "semantic-test"
    model_name = "semantic-test-v1"
    version = "1"
    dimension = 64
    is_external = False

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0
        self._hashing = HashingEmbeddingProvider(64)

    @property
    def usage(self) -> EmbeddingUsage:
        return EmbeddingUsage()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return self._hashing.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._hashing.embed_query(text)


def test_index_identity_isolates_embedding_spaces_and_reuses_document_cache(
    db_session: Session,
) -> None:
    canonical = document("The incident response identifier is SEC-77.")
    hashing = HashingEmbeddingProvider(64)
    semantic = CountingEmbeddingProvider()
    chunker = FixedTokenChunker(FixedTokenConfig(40, 5))
    hashing_identity = index_identity_for(
        "test-v1",
        IngestionConfig(
            chunk_size=40,
            chunk_overlap=5,
            embedding_provider=hashing.provider_name,
            embedding_model=hashing.model_name,
        ),
    )
    semantic_identity = index_identity_for(
        "test-v1",
        IngestionConfig(
            chunk_size=40,
            chunk_overlap=5,
            embedding_provider=semantic.provider_name,
            embedding_model=semantic.model_name,
        ),
    )
    hashing_pipeline = IngestionPipeline(db_session, chunker, hashing, hashing_identity)
    semantic_pipeline = IngestionPipeline(db_session, chunker, semantic, semantic_identity)
    hashing_pipeline.ingest(canonical)
    first = semantic_pipeline.ingest(canonical)
    second = semantic_pipeline.ingest(canonical)
    assert first.chunks_created == 1
    assert second.chunks_created == 0
    assert semantic.document_calls == 1

    principal = Principal("employee", "test-acme", frozenset({"employees"}))
    results = Retriever(db_session, semantic, semantic_identity).retrieve(
        "incident identifier", top_k=3, score_threshold=None, principal=principal
    )
    assert results
    assert semantic.query_calls == 1
    returned_identities = set(
        db_session.scalars(
            select(Chunk.index_identity).where(Chunk.id.in_([item.chunk_id for item in results]))
        ).all()
    )
    assert returned_identities == {semantic_identity}


def test_query_embedding_cache_reports_miss_then_normalized_hit(db_session: Session) -> None:
    canonical = document("The incident response identifier is SEC-77.")
    provider = CountingEmbeddingProvider()
    chunker = FixedTokenChunker(FixedTokenConfig(40, 5))
    identity = index_identity_for(
        "test-v1",
        IngestionConfig(
            chunk_size=40,
            chunk_overlap=5,
            embedding_provider=provider.provider_name,
            embedding_model=provider.model_name,
        ),
    )
    IngestionPipeline(db_session, chunker, provider, identity).ingest(canonical)
    principal = Principal("employee", "test-acme", frozenset({"employees"}))
    retriever = Retriever(db_session, provider, identity)

    first = retriever.retrieve("incident   identifier", principal=principal)
    first_timing = retriever.last_timing
    second = retriever.retrieve("incident identifier", principal=principal)
    second_timing = retriever.last_timing

    assert first and second
    assert not first_timing.query_embedding_cache_hit
    assert second_timing.query_embedding_cache_hit
    assert provider.query_calls == 1
    assert first_timing.embedding_cache_lookup_latency_ms >= 0
    assert second_timing.query_embedding_latency_ms == 0


class WrongDimensionProvider(CountingEmbeddingProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 63 for _ in texts]


def test_ingestion_rejects_provider_dimension_mismatch(db_session: Session) -> None:
    provider = WrongDimensionProvider()
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(40, 5)),
        provider,
        "wrong-dimension-index",
    )
    with pytest.raises(ValueError, match="incompatible vector dimension"):
        pipeline.ingest(document("Dimension mismatch must fail explicitly."))
