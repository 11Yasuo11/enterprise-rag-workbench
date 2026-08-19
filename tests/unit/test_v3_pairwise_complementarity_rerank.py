import inspect

import pytest

from rag_workbench.reranking.document_diversity import EVALUATION_LABEL_FIELDS
from rag_workbench.reranking.pairwise_complementarity import (
    pairwise_complementarity_rerank_top5,
)


def test_pairwise_complementarity_top5_is_deterministic_and_caps_per_doc() -> None:
    ranked = [
        {"chunk_id": "a1", "document_id": "doc-a", "rank": 1, "score": 0.99},
        {"chunk_id": "a2", "document_id": "doc-a", "rank": 2, "score": 0.98},
        {"chunk_id": "a3", "document_id": "doc-a", "rank": 3, "score": 0.97},  # dropped
        {"chunk_id": "b1", "document_id": "doc-b", "rank": 4, "score": 0.96},
        {"chunk_id": "b2", "document_id": "doc-b", "rank": 5, "score": 0.95},
        {"chunk_id": "b3", "document_id": "doc-b", "rank": 6, "score": 0.94},  # dropped
        {"chunk_id": "c1", "document_id": "doc-c", "rank": 7, "score": 0.93},  # makes top-5
    ]
    selected = pairwise_complementarity_rerank_top5(ranked, top_k=5)

    assert [item["chunk_id"] for item in selected] == ["a1", "a2", "b1", "b2", "c1"]
    assert all(
        item["retrieval_source"] == "pairwise_complementarity_rerank_top5" for item in selected
    )


def test_pairwise_complementarity_does_not_leak_evaluation_labels() -> None:
    src = inspect.getsource(pairwise_complementarity_rerank_top5)
    for field in EVALUATION_LABEL_FIELDS:
        assert field not in src


def test_pairwise_complementarity_rejects_label_leakage() -> None:
    ranked = [
        {
            "chunk_id": "a1",
            "document_id": "doc-a",
            "rank": 1,
            "score": 0.99,
            "expected_answer": "leak",
        },
        {"chunk_id": "a2", "document_id": "doc-b", "rank": 2, "score": 0.98},
    ]
    with pytest.raises(ValueError, match="evaluation labels leaked"):
        pairwise_complementarity_rerank_top5(ranked, top_k=5)

