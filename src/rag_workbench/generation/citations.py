from dataclasses import asdict, dataclass

from rag_workbench.generation.context_builder import ContextBundle


@dataclass(frozen=True)
class Citation:
    document_id: str
    chunk_id: str
    source: str
    title: str
    version: str
    page: int | None
    section: str | None

    def to_dict(self) -> dict[str, str | int | None]:
        return asdict(self)


def build_citations(context: ContextBundle, used_chunk_ids: tuple[str, ...]) -> list[Citation]:
    allowed = set(used_chunk_ids)
    return [
        Citation(
            document_id=item.result.document_id,
            chunk_id=item.result.chunk_id,
            source=item.result.source,
            title=item.result.title,
            version=item.result.version,
            page=item.result.page,
            section=item.result.section,
        )
        for item in context.items
        if item.result.chunk_id in allowed
    ]
