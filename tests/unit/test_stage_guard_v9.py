from __future__ import annotations

import pytest

from rag_workbench.evaluation.stage_guard import (
    ExperimentResourceLedger,
    StageGuardConfig,
    StageGuardError,
)


def config(stage: str, cases: int, cap: float) -> StageGuardConfig:
    return StageGuardConfig(stage, cases, 1, 1, cap)


def consume(guard, case_id: str, model: str, cost: float = 0.01) -> None:
    guard.authorize_call(case_id=case_id, model=model, projected_cost_usd=cost)
    guard.record_call(case_id=case_id, model=model, actual_cost_usd=cost)


def test_targeted_four_with_zero_sol_calls() -> None:
    ledger = ExperimentResourceLedger()
    guard = ledger.start_stage(
        config("TARGETED_FOUR", 4, 0.10),
        expected_stage="TARGETED_FOUR",
        expected_case_count=4,
    )
    assert guard.close()["sol_calls"] == 0


def test_targeted_four_allows_legitimate_bounded_escalation() -> None:
    ledger = ExperimentResourceLedger()
    guard = ledger.start_stage(
        config("TARGETED_FOUR", 4, 0.10),
        expected_stage="TARGETED_FOUR",
        expected_case_count=4,
    )
    consume(guard, "case-1", "SOL")
    assert guard.snapshot()["sol_calls"] == 1


@pytest.mark.parametrize("sol_calls", [6, 8])
def test_regression_allows_more_than_five_distinct_case_escalations(sol_calls: int) -> None:
    ledger = ExperimentResourceLedger()
    guard = ledger.start_stage(
        config("FRESH_V2_REGRESSION", 60, 0.30),
        expected_stage="FRESH_V2_REGRESSION",
        expected_case_count=60,
    )
    for index in range(sol_calls):
        consume(guard, f"case-{index}", "SOL", 0.01)
    assert guard.snapshot()["sol_calls"] == sol_calls


def test_second_sol_escalation_for_same_case_is_rejected() -> None:
    ledger = ExperimentResourceLedger()
    guard = ledger.start_stage(
        config("FRESH_V2_REGRESSION", 60, 0.30),
        expected_stage="FRESH_V2_REGRESSION",
        expected_case_count=60,
    )
    consume(guard, "case-1", "SOL")
    with pytest.raises(StageGuardError) as raised:
        guard.authorize_call(case_id="case-1", model="SOL", projected_cost_usd=0.01)
    assert raised.value.code == "PER_CASE_ESCALATION_LIMIT_TRIGGERED"


def test_stage_budget_rejects_before_provider_call() -> None:
    ledger = ExperimentResourceLedger()
    guard = ledger.start_stage(
        config("FRESH_V2_REGRESSION", 60, 0.03),
        expected_stage="FRESH_V2_REGRESSION",
        expected_case_count=60,
    )
    consume(guard, "case-1", "SOL", 0.02)
    with pytest.raises(StageGuardError) as raised:
        guard.authorize_call(case_id="case-2", model="SOL", projected_cost_usd=0.02)
    assert raised.value.code == "STAGE_BUDGET_GUARD_TRIGGERED"
    assert guard.snapshot()["sol_calls"] == 1


def test_new_stage_resets_counts_but_lifetime_cost_accumulates() -> None:
    ledger = ExperimentResourceLedger(historical_spend_usd=0.05)
    targeted = ledger.start_stage(
        config("TARGETED_FOUR", 4, 0.10),
        expected_stage="TARGETED_FOUR",
        expected_case_count=4,
    )
    consume(targeted, "case-1", "LUNA", 0.01)
    consume(targeted, "case-1", "SOL", 0.02)
    targeted.close()
    regression = ledger.start_stage(
        config("FRESH_V2_REGRESSION", 60, 0.30),
        expected_stage="FRESH_V2_REGRESSION",
        expected_case_count=60,
    )
    assert regression.snapshot()["luna_calls"] == 0
    assert regression.snapshot()["sol_calls"] == 0
    assert regression.spend_usd == 0
    assert ledger.lifetime_spend_usd == pytest.approx(0.08)


def test_wrong_stage_configuration_fails_before_execution() -> None:
    ledger = ExperimentResourceLedger()
    with pytest.raises(StageGuardError) as raised:
        ledger.start_stage(
            config("TARGETED_FOUR", 4, 0.10),
            expected_stage="FRESH_V2_REGRESSION",
            expected_case_count=60,
        )
    assert raised.value.code == "WRONG_STAGE_GUARD_CONFIGURATION"


def test_v2_sixth_sol_never_emits_targeted_four_error() -> None:
    ledger = ExperimentResourceLedger()
    guard = ledger.start_stage(
        config("FRESH_V2_REGRESSION", 60, 0.30),
        expected_stage="FRESH_V2_REGRESSION",
        expected_case_count=60,
    )
    for index in range(5):
        consume(guard, f"case-{index}", "SOL", 0.02)
    consume(guard, "fresh_v2_052", "SOL", 0.02)
    assert guard.snapshot()["sol_calls"] == 6

