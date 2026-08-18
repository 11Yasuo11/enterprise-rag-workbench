from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalFilters:
    document_ids: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()
