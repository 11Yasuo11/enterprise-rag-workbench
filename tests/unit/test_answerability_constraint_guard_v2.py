import pytest

from rag_workbench.safety.answerability_constraint_guard_v2 import (
    should_abstain_due_to_answerability_constraint,
)


def assert_abstain(*, question: str, evidence: list[str]) -> None:
    assert (
        should_abstain_due_to_answerability_constraint(
            question=question, evidence_texts=evidence
        )
        is True
    )


def assert_allow(*, question: str, evidence: list[str]) -> None:
    assert (
        should_abstain_due_to_answerability_constraint(
            question=question, evidence_texts=evidence
        )
        is False
    )


@pytest.mark.parametrize(
    "question,evidence",
    [
        ("What amount is at least 25 euros?", ["Expenses are greater than 25 euros."]),
        ("What amount is at most 25 euros?", ["Expenses are less than 25 euros."]),
        ("What amount is exactly 25 euros?", ["Expenses are greater than 25 euros."]),
        ("What amount is exactly 25 euros?", ["Expenses are less than 25 euros."]),
        ("What amount is exactly 25 euros?", ["Expenses are at least 25 euros."]),
        ("What amount is exactly 25 euros?", ["Expenses are at most 25 euros."]),
        ("Which date is on 2026?", ["Date is before 2026."]),
        ("Which date is on 2026?", ["Date is after 2026."]),
    ],
)
def test_constraint_non_entailment_matrix(question: str, evidence: list[str]) -> None:
    assert_abstain(question=question, evidence=evidence)


@pytest.mark.parametrize(
    "question,evidence",
    [
        ("What amount is exactly 25 euros?", ["The amount is 25 euros."]),
        ("What amount is exactly 25 euros?", ["Expenses are exactly 25 euros."]),
        ("What amount is at least 25 euros?", ["Expenses are at least 25 euros."]),
        ("What amount is above 25 euros?", ["Expenses are above 25 euros."]),
        ("What amount is at most 25 euros?", ["Expenses are at most 25 euros."]),
        ("What amount is below 25 euros?", ["Expenses are below 25 euros."]),
        ("What date is before 2026?", ["Date is before 2026."]),
        ("What date is after 2026?", ["Date is after 2026."]),
        ("Which date is on 2026?", ["Date is on 2026."]),
    ],
)
def test_constraint_positive_controls(question: str, evidence: list[str]) -> None:
    assert_allow(question=question, evidence=evidence)


def test_boundary_operators_other_values() -> None:
    assert_abstain(
        question="What amount is at least 10 euros?",
        evidence=["The policy applies to expenses greater than 10 euros."],
    )
    assert_allow(
        question="What amount is at least 10 euros?",
        evidence=["The policy applies to expenses at least 10 euros."],
    )
    assert_allow(
        question="What amount is at most 50 euros?",
        evidence=["The policy applies to expenses at most 50 euros."],
    )
    assert_abstain(
        question="What amount is at most 50 euros?",
        evidence=["The policy applies to expenses less than 50 euros."],
    )


def test_date_semantics_iso_tokens() -> None:
    q = "Which date is on 2026-07-15?"
    assert_allow(
        question=q,
        evidence=["The contract starts on 2026-07-15."],
    )
    assert_abstain(
        question=q,
        evidence=["The contract starts before 2026-07-15."],
    )


def test_date_semantics_corpus_on_month_day_year() -> None:
    q = "Which date is on 2026?"
    assert_allow(
        question=q,
        evidence=["Project Atlas launched on April 12, 2026."],
    )

