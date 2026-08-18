from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from rag_workbench.config import get_settings
from rag_workbench.db.models import Chunk, Document, DocumentPermission, DocumentVersion
from rag_workbench.domain.documents import CanonicalDocument
from rag_workbench.experiments.configs import IngestionConfig, index_identity_for
from rag_workbench.ingestion.chunkers import FixedTokenChunker
from rag_workbench.ingestion.loaders import load_document
from rag_workbench.providers.embeddings import EmbeddingProvider


class IngestionConflictError(ValueError):
    pass


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    document_version_id: str
    version: str
    chunks_created: int
    status: str
    content_hash: str
    is_active: bool


class IngestionPipeline:
    def __init__(
        self,
        session: Session,
        chunker: FixedTokenChunker,
        embedding_provider: EmbeddingProvider,
        index_identity: str | None = None,
    ) -> None:
        self.session = session
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.index_identity = index_identity or index_identity_for(
            get_settings().corpus_version,
            IngestionConfig(
                chunk_strategy=chunker.config.strategy,
                chunk_size=chunker.config.chunk_size,
                chunk_overlap=chunker.config.chunk_overlap,
                embedding_provider=embedding_provider.provider_name,
                embedding_model=embedding_provider.model_name,
                embedding_dimension=embedding_provider.dimension,
                embedding_version=embedding_provider.version,
            ),
        )

    def ingest_path(self, path: Path, tenant_id: str = "acmeai") -> IngestionResult:
        return self.ingest(load_document(path, tenant_id))

    def ingest(self, canonical: CanonicalDocument) -> IngestionResult:
        duplicate = self.session.execute(
            select(DocumentVersion, Document)
            .join(Document, DocumentVersion.document_fk == Document.id)
            .where(
                Document.tenant_id == canonical.tenant_id,
                DocumentVersion.content_hash == canonical.content_hash,
            )
        ).first()
        if duplicate:
            version, document = duplicate
            chunks_created = self._index_version(document, version, canonical)
            if chunks_created:
                self.session.commit()
            return IngestionResult(
                document_id=document.document_id,
                document_version_id=version.id,
                version=version.version,
                chunks_created=chunks_created,
                status="indexed" if chunks_created else "duplicate",
                content_hash=version.content_hash,
                is_active=version.is_active,
            )

        document = self.session.scalar(
            select(Document).where(
                Document.tenant_id == canonical.tenant_id,
                Document.document_id == canonical.document_id,
            )
        )
        if document is None:
            document = Document(
                document_id=canonical.document_id,
                tenant_id=canonical.tenant_id,
                title=canonical.title,
                source=canonical.source,
                source_type=canonical.source_type,
                source_url=canonical.source_url,
                visibility=canonical.visibility,
                metadata_=canonical.metadata,
            )
            self.session.add(document)
            self.session.flush()
        else:
            same_version = self.session.scalar(
                select(DocumentVersion).where(
                    DocumentVersion.document_fk == document.id,
                    DocumentVersion.version == canonical.version,
                )
            )
            if same_version:
                raise IngestionConflictError(
                    f"Version {canonical.version!r} already exists with different content for "
                    f"{canonical.document_id!r}"
                )
            document.title = canonical.title
            document.source = canonical.source
            document.source_type = canonical.source_type
            document.source_url = canonical.source_url
            document.visibility = canonical.visibility
            document.metadata_ = canonical.metadata

        active_version = self.session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.document_fk == document.id,
                DocumentVersion.is_active.is_(True),
            )
        )
        should_activate = canonical.is_active and (
            active_version is None
            or canonical.effective_at is None
            or active_version.effective_at is None
            or canonical.effective_at >= active_version.effective_at
        )
        if should_activate:
            self.session.execute(
                update(DocumentVersion)
                .where(DocumentVersion.document_fk == document.id)
                .values(is_active=False)
            )

        version = DocumentVersion(
            document_fk=document.id,
            version=canonical.version,
            content=canonical.content,
            content_hash=canonical.content_hash,
            effective_at=canonical.effective_at,
            is_active=should_activate,
            metadata_={
                **canonical.metadata,
                "source_created_at": canonical.created_at.isoformat(),
                "source_updated_at": canonical.updated_at.isoformat(),
            },
        )
        self.session.add(version)
        self.session.flush()

        chunks_created = self._index_version(document, version, canonical)

        self.session.execute(
            delete(DocumentPermission).where(DocumentPermission.document_fk == document.id)
        )
        for group in canonical.permission_groups:
            self.session.add(DocumentPermission(document_fk=document.id, permission_group=group))
        self.session.commit()
        return IngestionResult(
            document_id=document.document_id,
            document_version_id=version.id,
            version=version.version,
            chunks_created=chunks_created,
            status="ingested",
            content_hash=canonical.content_hash,
            is_active=version.is_active,
        )

    def _index_version(
        self, document: Document, version: DocumentVersion, canonical: CanonicalDocument
    ) -> int:
        existing = self.session.scalar(
            select(Chunk.id)
            .where(
                Chunk.document_version_id == version.id,
                Chunk.index_identity == self.index_identity,
            )
            .limit(1)
        )
        if existing is not None:
            return 0
        drafts = self.chunker.chunk(canonical)
        embeddings = self.embedding_provider.embed_documents([draft.text for draft in drafts])
        if len(embeddings) != len(drafts):
            raise ValueError("embedding provider returned an unexpected vector count")
        if any(len(embedding) != self.embedding_provider.dimension for embedding in embeddings):
            raise ValueError("embedding provider returned an incompatible vector dimension")
        for draft, embedding in zip(drafts, embeddings, strict=True):
            self.session.add(
                Chunk(
                    document_fk=document.id,
                    document_version_id=version.id,
                    chunk_index=draft.chunk_index,
                    text=draft.text,
                    token_count=draft.token_count,
                    page=draft.page,
                    section=draft.section,
                    embedding=embedding,
                    embedding_provider=self.embedding_provider.provider_name,
                    embedding_model=self.embedding_provider.model_name,
                    embedding_version=self.embedding_provider.version,
                    embedding_dimension=self.embedding_provider.dimension,
                    index_identity=self.index_identity,
                    metadata_=draft.metadata,
                )
            )
        return len(drafts)
