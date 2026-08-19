from rag_workbench.safety.answerability_constraint_guard_v1 import (
    should_abstain_due_to_answerability_constraint,
)


def test_exactly_vs_above_abstains() -> None:
    question = "What amount is exactly 25 euros?"
    evidence = ["Receipts are required for expenses above 25 euros."]
    assert should_abstain_due_to_answerability_constraint(
        question=question, evidence_texts=evidence
    ) is True


def test_exactly_vs_below_abstains() -> None:
    question = "What amount is exactly 25 euros?"
    evidence = ["Receipts are required for expenses below 25 euros."]
    assert should_abstain_due_to_answerability_constraint(
        question=question, evidence_texts=evidence
    ) is True


def test_at_least_vs_greater_than_entails() -> None:
    # Evidence "greater than" is stronger than required "at least".
    question = "What amount is at least 10 euros?"
    evidence = ["The amount is greater than 10 euros."]
    assert should_abstain_due_to_answerability_constraint(
        question=question, evidence_texts=evidence
    ) is False


def test_at_most_vs_less_than_entails() -> None:
    question = "What amount is at most 10 euros?"
    evidence = ["The amount is less than 10 euros."]
    assert should_abstain_due_to_answerability_constraint(
        question=question, evidence_texts=evidence
    ) is False


def test_before_vs_on_abstains() -> None:
    question = "What did the policy do before 2026?"
    evidence = ["The policy applies in 2026."]
    assert should_abstain_due_to_answerability_constraint(
        question=question, evidence_texts=evidence
    ) is True


def test_after_vs_on_abstains() -> None:
    question = "What did the policy do after 2026?"
    evidence = ["The policy applies in 2026."]
    assert should_abstain_due_to_answerability_constraint(
        question=question, evidence_texts=evidence
    ) is True


def test_exact_entailment_control() -> None:
    question = "What amount is exactly 25 euros?"
    evidence = ["The amount is 25 euros."]
    assert should_abstain_due_to_answerability_constraint(
        question=question, evidence_texts=evidence
    ) is False


def test_historical_clean_none_003_triggers() -> None:
    question = "What receipt requirement applies to an expense of exactly 25 euros?"
    evidence = ["Receipts are required for expenses above 25 euros."]
    assert should_abstain_due_to_answerability_constraint(
        question=question, evidence_texts=evidence
    ) is True

