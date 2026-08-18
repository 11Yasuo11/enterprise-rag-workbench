import hashlib
from dataclasses import dataclass

from rag_workbench.retrieval.vector_search import RetrievalResult


@dataclass(frozen=True)
class ContextItem:
    citation_label: str
    result: RetrievalResult


@dataclass(frozen=True)
class ContextBundle:
    items: tuple[ContextItem, ...]
    text: str
    token_count: int
    dropped_chunk_ids: tuple[str, ...]


class ContextBuilder:
    def __init__(self, token_budget: int = 1200) -> None:
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        self.token_budget = token_budget

    def build(self, results: list[RetrievalResult]) -> ContextBundle:
        items: list[ContextItem] = []
        blocks: list[str] = []
        seen: set[str] = set()
        dropped: list[str] = []
        used_tokens = 0
        for result in sorted(results, key=lambda item: item.rank):
            fingerprint = hashlib.sha256(" ".join(result.text.split()).encode()).hexdigest()
            if fingerprint in seen:
                dropped.append(result.chunk_id)
                continue
            token_count = len(result.text.split())
            if used_tokens + token_count > self.token_budget:
                dropped.append(result.chunk_id)
                continue
            seen.add(fingerprint)
            label = f"C{len(items) + 1}"
            item = ContextItem(citation_label=label, result=result)
            items.append(item)
            used_tokens += token_count
            location = (
                f"page={result.page}" if result.page else f"section={result.section or 'n/a'}"
            )
            blocks.append(
                f"[{label}] chunk_id={result.chunk_id} document_id={result.document_id} "
                f"version={result.version} {location}\n{result.text}"
            )
        return ContextBundle(tuple(items), "\n\n".join(blocks), used_tokens, tuple(dropped))
