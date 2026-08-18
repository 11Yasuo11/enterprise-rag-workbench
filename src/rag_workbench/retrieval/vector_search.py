from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    document_id: str
    document_version_id: str
    text: str
    rank: int
    score: float
    source: str
    source_type: str
    title: str
    version: str
    page: int | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_source: str = "dense"
    dense_score: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    found_by_dense: bool = False
    found_by_bm25: bool = False
