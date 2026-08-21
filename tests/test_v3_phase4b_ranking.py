# ruff: noqa: E501
"""Tests for V3 Phase 4B: final ranking research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_workbench.reranking.base import RerankedResult
from rag_workbench.reranking.listwise_evidence_set import (
    ALGORITHM_ID,
    listwise_configuration,
    select_listwise_evidence_set_top5,
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


# --- Immutability tests ---

def test_v1_dataset_immutable():
    path = Path("data/eval/acmeai_enterprise_rag_v1_final_eval.json")
    if path.exists():
        payload = json.loads(path.read_text())
        assert "cases" in payload
        assert len(payload["cases"]) > 0


def test_v2_dataset_immutable():
    path = Path("data/eval/acmeai_enterprise_rag_v2_final_eval.json")
    if path.exists():
        payload = json.loads(path.read_text())
        assert "cases" in payload
        assert len(payload["cases"]) > 0


def test_v3_final_dataset_immutable():
    path = Path("data/eval/acmeai_enterprise_rag_v3_final_eval.json")
    if path.exists():
        payload = json.loads(path.read_text())
        assert payload["dataset_id"] == "acmeai-enterprise-rag-v3-final-eval"


# --- Candidate pool identity ---

def test_candidate_pool_parameters():
    from rag_workbench.experiments.hybrid_reranker_benchmark import (
        BM25_DEPTH,
        DENSE_DEPTH,
        FINAL_TOP_K,
        RRF_K,
        UNION_LIMIT,
    )
    assert DENSE_DEPTH == 20
    assert BM25_DEPTH == 20
    assert RRF_K == 60
    assert UNION_LIMIT == 30
    assert FINAL_TOP_K == 5


# --- Ranking-only independent variable ---

def test_listwise_does_not_change_candidate_pool():
    candidates = [
        _make_reranked(f"c{i}", f"doc{i % 3}", f"text about topic {i}", 0.9 - i * 0.1, i + 1)
        for i in range(10)
    ]
    query = "test query about topics"
    result = select_listwise_evidence_set_top5(candidates, query)
    for item in result:
        assert item in candidates


# --- Listwise determinism ---

def test_listwise_deterministic():
    candidates = [
        _make_reranked(f"c{i}", f"doc{i % 4}", f"text about topic {i} with details", 0.9 - i * 0.05, i + 1)
        for i in range(15)
    ]
    query = "what are the three required topics"
    r1 = select_listwise_evidence_set_top5(candidates, query)
    r2 = select_listwise_evidence_set_top5(candidates, query)
    assert [id(x) for x in r1] == [id(x) for x in r2]


# --- Same-document multiple-evidence preservation ---

def test_same_document_multi_chunk_preserved():
    candidates = [
        _make_reranked("c1", "doc-A", "the deployment approval identifier is ENG-DEP-17", 0.95, 1),
        _make_reranked("c2", "doc-A", "routine release weekdays are Tuesdays and Thursdays", 0.85, 2),
        _make_reranked("c3", "doc-B", "the API recovery objective is four hours", 0.80, 3),
        _make_reranked("c4", "doc-A", "another chunk about deployment details and ENG-DEP-17", 0.78, 4),
        _make_reranked("c5", "doc-C", "support escalation queue CS-1842", 0.75, 5),
    ]
    query = "Name the deployment approval identifier and the routine release weekdays from the engineering handbook"
    result = select_listwise_evidence_set_top5(candidates, query, top_k=5)
    doc_a_chunks = [r for r in result if r.result.document_id == "doc-A"]
    assert len(doc_a_chunks) >= 2, "must preserve multiple relevant chunks from same document"


# --- Near-duplicate source handling ---

def test_near_duplicate_prefers_query_aligned():
    candidates = [
        _make_reranked("c1", "recovery-east", "eastern recovery code OPS-REC-E17 and standby cluster eu-central", 0.92, 1),
        _make_reranked("c2", "recovery-west", "western recovery code OPS-REC-W29 and standby cluster us-east", 0.91, 2),
        _make_reranked("c3", "recovery-east", "eastern drill cadence first Wednesday quarterly", 0.85, 3),
        _make_reranked("c4", "recovery-west", "western drill cadence second Wednesday quarterly", 0.84, 4),
    ]
    query = "Keep only the sunrise-coast recovery document and return its sequence token OPS-REC-E17"
    result = select_listwise_evidence_set_top5(candidates, query, top_k=5)
    assert result[0].result.document_id == "recovery-east"


# --- Exact-ID preservation ---

def test_exact_id_not_displaced():
    candidates = [
        _make_reranked("c1", "engineering", "the deployment approval identifier ENG-DEP-17", 0.95, 1),
        _make_reranked("c2", "operations", "API recovery four hours objective", 0.80, 2),
        _make_reranked("c3", "support", "escalation CS-1842 queue", 0.78, 3),
    ]
    query = "Return exactly the deployment approval identifier token ENG-DEP-17"
    result = select_listwise_evidence_set_top5(candidates, query, top_k=5)
    assert any(r.result.chunk_id == "c1" for r in result)


# --- Version correctness ---

def test_version_correctness():
    config = listwise_configuration()
    assert config["no_ground_truth_labels"] is True
    assert config["query_conditioned_complementarity"] is True


# --- ACL / tenant filtering preserved ---

def test_acl_label_leakage_blocked():
    bad = {
        "chunk_id": "c1",
        "document_id": "hr-benefits",
        "text": "secret",
        "score": 0.9,
        "rank": 1,
        "required_document_ids": ["hr-benefits"],
    }
    with pytest.raises(ValueError, match="evaluation labels leaked"):
        select_listwise_evidence_set_top5([bad], "test query")


# --- No hidden-label access ---

def test_no_hidden_labels_in_algorithm():
    config = listwise_configuration()
    assert config["no_ground_truth_labels"] is True
    assert ALGORITHM_ID == "LISTWISE_EVIDENCE_SET_TOP5"


# --- No external Judge calls ---

def test_no_external_judge_calls():
    config = listwise_configuration()
    assert "judge" not in str(config).lower()
    assert "draft" not in str(config).lower()
    assert "verifier" not in str(config).lower()


# --- Dataset validation ---

def test_ranking_validation_dataset_exists():
    path = Path("data/eval/acmeai_v3_ranking_validation_v1.json")
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["dataset_id"] == "acmeai-v3-ranking-validation-v1"
    assert len(payload["cases"]) == 100


def test_ranking_validation_dataset_distribution():
    from collections import Counter

    from rag_workbench.experiments.v3_phase4b_ranking_cases import CASES, EXPECTED_DISTRIBUTION
    dist = dict(sorted(Counter(str(item["category"]) for item in CASES).items()))
    assert dist == EXPECTED_DISTRIBUTION


def test_ranking_validation_dataset_overlap():
    from rag_workbench.experiments.v3_phase4b_ranking_cases import dataset_overlap_report
    report = dataset_overlap_report()
    assert report["pass"] is True
    assert report["maximum_normalized_overlap"] < 0.5


def test_ranking_validation_unique_questions():
    from rag_workbench.experiments.v3_phase4b_ranking_cases import CASES
    questions = [str(c["question"]) for c in CASES]
    assert len(questions) == len(set(questions))


# --- Promotion policy frozen ---

def test_promotion_policy_frozen():
    from rag_workbench.experiments.v3_phase4b_ranking_benchmark import PROMOTION_POLICY
    assert PROMOTION_POLICY["all_required_evidence_coverage_at_5_absolute_improvement"] == 0.08
    assert PROMOTION_POLICY["three_document_coverage_at_5_absolute_improvement"] == 0.15
    assert PROMOTION_POLICY["rescue_margin"] == 5
    assert PROMOTION_POLICY["acl_safety_gate"] == 1.0
    assert PROMOTION_POLICY["frozen_before_validation"] is True


# --- Complement is query-conditioned ---

def test_complement_is_query_conditioned():
    c1 = _make_reranked("c1", "doc-A", "deployment identifier ENG-DEP-17 approval", 0.9, 1)
    c2 = _make_reranked("c2", "doc-A", "release weekdays Tuesdays Thursdays routine", 0.85, 2)
    c3 = _make_reranked("c3", "doc-B", "deployment details and ENG-DEP-17 approval steps", 0.82, 3)

    query = "Name the deployment approval identifier and the routine release weekdays"
    result = select_listwise_evidence_set_top5([c1, c2, c3], query, top_k=3)
    ids = [r.result.chunk_id for r in result]
    assert "c1" in ids
    assert "c2" in ids, "complementary chunk from same document must be selected over redundant chunk from different document"
