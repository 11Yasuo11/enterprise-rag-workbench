from pathlib import Path

import pytest

from rag_workbench.evaluation.datasets import load_evaluation_dataset
from rag_workbench.evaluation.splits import (
    EVIDENCE_GATE_SPLIT_SEED,
    EvaluationPartition,
    HoldoutEvaluationGuard,
    deterministic_stratified_split,
)


def test_holdout_split_is_reproducible_stratified_and_complete() -> None:
    dataset = load_evaluation_dataset(Path("data/eval/eval_v1.json"))
    first = deterministic_stratified_split(dataset)
    second = deterministic_stratified_split(dataset)
    assert first == second
    assert first.seed == EVIDENCE_GATE_SPLIT_SEED
    assert len(first.calibration) == 70
    assert len(first.holdout) == 30
    assert set(case.category for case in first.holdout) == set(
        case.category for case in dataset
    )
    assert {case.case_id for case in first.calibration}.isdisjoint(
        case.case_id for case in first.holdout
    )


def test_holdout_cannot_select_or_retune_configuration() -> None:
    guard = HoldoutEvaluationGuard()
    with pytest.raises(ValueError, match="calibration"):
        guard.lock_from_calibration("candidate", partition=EvaluationPartition.HOLDOUT)
    guard.lock_from_calibration("candidate", partition=EvaluationPartition.CALIBRATION)
    with pytest.raises(ValueError, match="exact locked"):
        guard.authorize_holdout("changed")
    guard.authorize_holdout("candidate")
    with pytest.raises(ValueError, match="already been consumed"):
        guard.authorize_holdout("candidate")
