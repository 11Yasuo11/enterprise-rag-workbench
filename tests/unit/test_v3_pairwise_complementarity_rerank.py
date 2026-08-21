import inspect

import pytest

from rag_workbench.reranking.base import RerankedResult
from rag_workbench.reranking.document_diversity import EVALUATION_LABEL_FIELDS
from rag_workbench.reranking.pairwise_complementarity import (
    select_pairwise_complementarity_top5,
)
from rag_workbench.retrieval.vector_search import RetrievalResult


def _make_reranked(
    chunk_id: str,
    document_id: str,
    text: str,
    reranker_score: float,
    reranked_rank: int,
) -> RerankedResult:
    return RerankedResult(
        result=RetrievalResult(
            chunk_id=chunk_id,
            document_id=document_id,
            document_version_id=f"{document_id}-v1",
            text=text,
            rank=reranked_rank,
            score=reranker_score,
            source="test",
            source_type="document",
            title=document_id,
            version="1.0",
            retrieval_source="dense",
        ),
        original_dense_rank=reranked_rank,
        dense_score=0.5,
        reranker_score=reranker_score,
        reranked_rank=reranked_rank,
    )


def test_pairwise_complementarity_top5_is_deterministic() -> None:
    candidates = [
        _make_reranked(f"c{i}", f"doc-{i % 3}", f"text about topic {i}", 0.9 - i * 0.05, i + 1)
        for i in range(10)
    ]
    query = "what are the required topics across documents"
    selected_a = select_pairwise_complementarity_top5(candidates, query, top_k=5)
    selected_b = select_pairwise_complementarity_top5(candidates, query, top_k=5)

    assert [item.result.chunk_id for item in selected_a] == [
        item.result.chunk_id for item in selected_b
    ]
    assert all(item in candidates for item in selected_a)
    assert len(selected_a) == 5


def test_pairwise_complementarity_does_not_leak_evaluation_labels() -> None:
    src = inspect.getsource(select_pairwise_complementarity_top5)
    for field in EVALUATION_LABEL_FIELDS:
        assert field not in src


def test_pairwise_complementarity_rejects_label_leakage() -> None:
    ranked = [
        {
            "chunk_id": "a1",
            "document_id": "doc-a",
            "rank": 1,
            "score": 0.99,
            "text": "secret",
            "expected_answer": "leak",
        },
        {"chunk_id": "a2", "document_id": "doc-b", "rank": 2, "score": 0.98, "text": "ok"},
    ]
    with pytest.raises(ValueError, match="evaluation labels leaked"):
        select_pairwise_complementarity_top5(ranked, "test query", top_k=5)
