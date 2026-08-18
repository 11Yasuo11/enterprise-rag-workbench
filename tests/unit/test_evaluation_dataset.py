from collections import Counter
from pathlib import Path

import pytest
from fastapi import HTTPException

from rag_workbench.api.app import _safe_corpus_path
from rag_workbench.evaluation.datasets import (
    EvaluationCase,
    EvaluationDataset,
    load_evaluation_dataset,
    validate_evaluation_dataset,
)


def test_eval_v1_is_valid_and_has_expected_distribution() -> None:
    dataset = load_evaluation_dataset(Path("data/eval/eval_v1.json"))
    result = validate_evaluation_dataset(dataset, Path("data/synthetic_company"))
    assert result.valid
    assert result.case_count == 100
    assert Counter(case.category for case in dataset) == {
        "single_document": 20,
        "multi_document": 20,
        "exact_identifier": 10,
        "versioning": 10,
        "access_control": 10,
        "abstention": 15,
        "prompt_injection": 10,
        "duplicate": 5,
    }


def test_duplicate_case_ids_fail_validation() -> None:
    item = EvaluationCase(
        case_id="duplicate",
        question="Question?",
        expected_answer=None,
        should_abstain=True,
        category="abstention",
    )
    dataset = EvaluationDataset(dataset_version="v", cases=(item, item))
    result = validate_evaluation_dataset(dataset, Path("data/synthetic_company"))
    assert not result.valid
    assert "duplicate evaluation IDs" in result.errors[0]


def test_eval_ground_truth_cannot_be_ingested() -> None:
    with pytest.raises(HTTPException):
        _safe_corpus_path("data/eval/eval_v1.json")
