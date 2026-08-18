from dataclasses import dataclass
from typing import Protocol

from rag_workbench.retrieval.vector_search import RetrievalResult


@dataclass(frozen=True)
class RerankedResult:
    result: RetrievalResult
    original_dense_rank: int
    dense_score: float
    reranker_score: float
    reranked_rank: int


class Reranker(Protocol):
    model_id: str
    resolved_revision: str
    backend: str
    device: str
    external_calls: int

    def rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RerankedResult]: ...
