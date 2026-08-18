import pytest

from rag_workbench.evaluation.generation_metrics import (
    abstention_classification,
    abstention_correct,
    deterministic_citation_correctness,
)
from rag_workbench.evaluation.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_retrieval_metrics() -> None:
    retrieved = ["noise", "relevant-a", "relevant-b"]
    relevant = {"relevant-a", "relevant-b"}
    assert hit_at_k(retrieved, relevant, 2) == 1.0
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(0.6934, rel=1e-3)


def test_empty_ground_truth_rewards_empty_retrieval() -> None:
    assert recall_at_k([], set(), 5) == 1.0
    assert ndcg_at_k(["noise"], set(), 5) == 0.0
    assert abstention_correct("abstained", True) == 1.0


def test_ndcg_does_not_count_duplicate_relevant_identity_twice() -> None:
    assert ndcg_at_k(["relevant", "relevant"], {"relevant"}, 2) == 1.0


def test_abstention_precision_recall_and_undefined_denominators() -> None:
    measured = abstention_classification([True, True, False], [True, False, True])
    assert measured["abstention_precision"] == 0.5
    assert measured["abstention_recall"] == 0.5
    assert measured["abstention_f1"] == 0.5
    undefined = abstention_classification([False], [False])
    assert undefined["abstention_precision"] is None
    assert undefined["abstention_recall"] is None
    assert deterministic_citation_correctness(["c1"], ["c1", "c2"]) == 1.0
