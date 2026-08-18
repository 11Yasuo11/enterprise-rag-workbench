import re
from dataclasses import dataclass

from rag_workbench.domain.documents import CanonicalDocument, ChunkDraft, DocumentSegment
from rag_workbench.ingestion.cleaners import clean_text

TOKEN = re.compile(r"[\w]+(?:[-_][\w]+)*|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class FixedTokenConfig:
    chunk_size: int = 180
    chunk_overlap: int = 30
    strategy: str = "fixed_token"
    version: str = "1"

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and less than chunk_size")


class FixedTokenChunker:
    def __init__(self, config: FixedTokenConfig) -> None:
        self.config = config

    def chunk(self, document: CanonicalDocument) -> list[ChunkDraft]:
        segments = document.segments or (DocumentSegment(text=document.content),)
        chunks: list[ChunkDraft] = []
        step = self.config.chunk_size - self.config.chunk_overlap
        for segment in segments:
            cleaned = clean_text(segment.text)
            tokens = TOKEN.findall(cleaned)
            for start in range(0, len(tokens), step):
                window = tokens[start : start + self.config.chunk_size]
                if not window:
                    continue
                chunks.append(
                    ChunkDraft(
                        chunk_index=len(chunks),
                        text=" ".join(window),
                        token_count=len(window),
                        page=segment.page,
                        section=segment.section,
                        metadata={
                            **segment.metadata,
                            "chunk_strategy": self.config.strategy,
                            "chunker_version": self.config.version,
                            "chunk_size": self.config.chunk_size,
                            "chunk_overlap": self.config.chunk_overlap,
                        },
                    )
                )
                if start + self.config.chunk_size >= len(tokens):
                    break
        return chunks
