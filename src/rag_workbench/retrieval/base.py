from enum import StrEnum
from typing import Protocol

from rag_workbench.retrieval.filters import RetrievalFilters
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal


class RetrievalMode(StrEnum):
    DENSE = "DENSE"
    LEXICAL_BM25 = "LEXICAL_BM25"
    HYBRID_RRF = "HYBRID_RRF"


class RetrievalProvider(Protocol):
    mode: RetrievalMode

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: RetrievalFilters | None = None,
        principal: Principal | None = None,
    ) -> list[RetrievalResult]: ...
