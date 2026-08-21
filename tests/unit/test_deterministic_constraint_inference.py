from rag_workbench.experiments.deterministic_constraint_inference import infer_constraint


def test_eleventh_business_day_does_not_satisfy_within_ten() -> None:
    result = infer_constraint(
        "Would filing on the eleventh business day satisfy the deadline?",
        "Expense reports are due within ten business days.",
    )
    assert result is not None
    assert not result.conclusion


def test_forty_five_minute_cadence_does_not_match_thirty() -> None:
    result = infer_constraint(
        "If updates are sent every thirty minutes, would a forty-five-minute cadence match?",
        "Updates are published every thirty minutes until mitigation.",
    )
    assert result is not None
    assert not result.conclusion


def test_at_most_at_least_and_not_equal() -> None:
    assert infer_constraint("Would 9 days comply?", "Complete at most 10 days.").conclusion
    assert not infer_constraint("Would 4 items comply?", "Provide at least 5 items.").conclusion
    assert infer_constraint("Would 6 hours comply?", "Use a value not equal to 5 hours.").conclusion


def test_incompatible_units_do_not_guess() -> None:
    assert infer_constraint("Would 10 hours comply?", "Complete within 10 days.") is None


def test_ungrounded_operands_do_not_guess() -> None:
    assert infer_constraint("Would that comply?", "Complete within 10 days.") is None
