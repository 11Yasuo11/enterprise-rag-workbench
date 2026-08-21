import json
from copy import deepcopy

import pytest

from rag_workbench.evaluation.eval_loop import (
    PrimaryError,
    aggregate_metrics,
    apply_gate,
    classify_failure,
    compare_results,
    load_frozen_cases,
    normalize_trace,
    slice_analysis,
)
from rag_workbench.evaluation.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def _case(**overrides):
    case = {
        "case_id": "c1",
        "question": "q",
        "gold_answer": None,
        "gold_document_ids": ["d1"],
        "gold_chunk_ids": ["g1"],
        "answerable": True,
        "category": "single_document",
        "difficulty": "unknown",
        "requires_acl": None,
        "requires_tenant_filter": None,
        "requires_version_filter": None,
        "requires_multi_hop": None,
        "required_facts": [],
        "raw_metadata": {},
    }
    case.update(overrides)
    return case


def _trace(case=None, **overrides):
    raw = {
        "retrieved_candidate_ids": ["g1", "x"],
        "reranked_top_ids": ["g1"],
        "final_answer": "answer",
        "abstained": False,
        "answer_correct": True,
        "citation_ids": ["g1"],
    }
    raw.update(overrides)
    return normalize_trace(case or _case(), raw)


def test_hit_at_k() -> None:
    assert hit_at_k(["x", "g"], {"g"}, 1) == 0.0
    assert hit_at_k(["x", "g"], {"g"}, 2) == 1.0


def test_recall_at_k_multiple_gold_and_empty_retrieval() -> None:
    assert recall_at_k(["g1"], {"g1", "g2"}, 5) == 0.5
    assert recall_at_k([], {"g1"}, 5) == 0.0


def test_mrr() -> None:
    assert reciprocal_rank(["x", "g"], {"g"}) == 0.5


def test_ndcg() -> None:
    assert ndcg_at_k(["x", "g1", "g2"], {"g1", "g2"}, 3) == pytest.approx(0.6934, rel=1e-3)


def test_retrieval_failure() -> None:
    trace = _trace(retrieved_candidate_ids=["x"], reranked_top_ids=["x"], answer_correct=False)
    assert classify_failure(_case(), trace) == PrimaryError.RETRIEVAL_FAILURE


def test_ranking_failure() -> None:
    trace = _trace(retrieved_candidate_ids=["g1"], reranked_top_ids=["x"], answer_correct=False)
    assert classify_failure(_case(), trace) == PrimaryError.RANKING_FAILURE


def test_judge_false_negative() -> None:
    trace = _trace(
        final_answer="",
        abstained=True,
        answer_correct=False,
        evidence_sufficient=True,
        judge_answerable=False,
        behavior="INCORRECT_ABSTENTION",
    )
    assert classify_failure(_case(), trace) == PrimaryError.JUDGE_FALSE_NEGATIVE


def test_judge_false_positive() -> None:
    case = _case(answerable=False, gold_document_ids=[], gold_chunk_ids=[])
    trace = _trace(case, answer_correct=False, citation_ids=[])
    assert classify_failure(case, trace) == PrimaryError.JUDGE_FALSE_POSITIVE


def test_citation_failure() -> None:
    trace = _trace(answer_correct=False, citation_ids=["not-retrieved"])
    assert classify_failure(_case(), trace) == PrimaryError.CITATION_FAILURE


def test_missing_gold_remains_unknown_not_failure_guess() -> None:
    case = _case(gold_document_ids=[], gold_chunk_ids=[], answerable=None)
    trace = _trace(case, answer_correct=None, citation_ids=[])
    assert trace["retrieval_hit"] is None
    assert trace["primary_error"] == PrimaryError.UNRESOLVED
    assert "INSUFFICIENT_GOLD_LABEL" in trace["secondary_tags"]


def test_frozen_loader_keeps_missing_metadata_unknown(tmp_path) -> None:
    dataset = tmp_path / "frozen.jsonl"
    dataset.write_text(
        json.dumps({"query_id": "c1", "question": "q", "category": "unknown"}) + "\n",
        encoding="utf-8",
    )
    cases, dataset_hash = load_frozen_cases(dataset)
    assert dataset_hash
    assert cases[0]["answerable"] is None
    assert cases[0]["difficulty"] == "unknown"
    assert cases[0]["gold_chunk_ids"] == []


def test_empty_retrieval_is_deterministic_miss() -> None:
    trace = _trace(
        retrieved_candidate_ids=[],
        retrieved_candidate_document_ids=["noise"],
        reranked_top_ids=[],
        reranked_top_document_ids=[],
        answer_correct=False,
        citation_ids=[],
    )
    assert trace["retrieval_hit"] is False
    assert trace["primary_error"] == PrimaryError.RETRIEVAL_FAILURE


def test_slice_analysis_marks_small_samples() -> None:
    case = _case()
    trace = _trace(case)
    slices = slice_analysis([case], [trace], min_size=5)
    assert slices["slices"]["single_document"]["small_sample"] is True
    assert slices["slices"]["answerable"]["f1"] == 1.0


def _result(case, trace, metrics=None):
    measured = aggregate_metrics([case], [trace], (1, 3, 5, 10))
    measured.update(metrics or {})
    return {"dataset_hash": "same", "metrics": measured, "traces": [trace]}


def test_regression_detection() -> None:
    case = _case()
    passing = _trace(case)
    failing = _trace(case, answer_correct=False, citation_ids=["bad"])
    gate = {"promotion_gate": {"precision": {"min_delta": 0}}}
    compared = compare_results(_result(case, passing), _result(case, failing), gate).payload
    assert compared["decision"] == "REJECT"
    assert compared["case_comparison"]["regressed_cases"] == ["c1"]


def test_promotion_gate() -> None:
    comparisons = {
        "precision": {"baseline": 1.0, "candidate": 0.99, "delta": -0.01},
        "unsupported_answers": {"baseline": 0, "candidate": 1, "delta": 1},
    }
    gate = {
        "promotion_gate": {
            "precision": {"min": 1.0},
            "unsupported_answers": {"max_regression": 0},
        }
    }
    reasons = apply_gate(comparisons, gate)
    assert len(reasons) == 2


def test_dataset_mismatch_requires_manual_review() -> None:
    case = _case()
    result = _result(case, _trace(case))
    candidate = deepcopy(result)
    candidate["dataset_hash"] = "different"
    compared = compare_results(result, candidate, {}).payload
    assert compared["decision"] == "MANUAL_REVIEW"


def test_missing_gate_metric_requires_manual_review() -> None:
    case = _case()
    result = _result(case, _trace(case))
    gate = {"promotion_gate": {"not_measured": {"min": 1.0}}}
    compared = compare_results(result, result, gate).payload
    assert compared["decision"] == "MANUAL_REVIEW"
