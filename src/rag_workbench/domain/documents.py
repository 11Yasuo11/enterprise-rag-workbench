import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class DocumentSegment:
    text: str
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalDocument:
    document_id: str
    title: str
    content: str
    source: str
    source_type: str
    version: str
    tenant_id: str = "acmeai"
    source_url: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    effective_at: datetime | None = None
    is_active: bool = True
    visibility: str = "public"
    permission_groups: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    segments: tuple[DocumentSegment, ...] = ()

    @property
    def content_hash(self) -> str:
        normalized = " ".join(self.content.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChunkDraft:
    chunk_index: int
    text: str
    token_count: int
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
