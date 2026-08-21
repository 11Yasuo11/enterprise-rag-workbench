# ruff: noqa: E501
"""
ANSWERABILITY_CONSTRAINT_GUARD_V1

Deterministic constraint checker used for Phase 5D.

Goal:
Prevent a Judge-positive ("answerable") decision from becoming a user-visible
answer when the retrieved evidence does not satisfy an explicit semantic
constraint in the question.

This guard is generic for relational semantics. It is intentionally
deterministic and conservative:
- If we detect a constraint but cannot find an entailment-supporting relation
  in the evidence, we SAFE-ABSTAIN.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def _norm(text: str) -> str:
    return text.translate(_ZERO_WIDTH).replace("\xa0", " ").strip().casefold()


@dataclass(frozen=True)
class NumericConstraint:
    # One of: EXACT, ABOVE, BELOW, AT_LEAST, AT_MOST
    kind: str
    value: str


@dataclass(frozen=True)
class YearConstraint:
    # One of: BEFORE, AFTER, ON
    kind: str
    year: str


_NUM = r"(?P<val>\d+(?:\.\d+)?)"

_Q_EXACT = re.compile(rf"(?is)\b(?:exactly|equal to)\s*{_NUM}\b")
_Q_ABOVE = re.compile(rf"(?is)\b(?:above|greater than|more than)\s*{_NUM}\b")
_Q_BELOW = re.compile(rf"(?is)\b(?:below|less than)\s*{_NUM}\b")
_Q_AT_LEAST = re.compile(rf"(?is)\b(?:at least|no less than)\s*{_NUM}\b")
_Q_AT_MOST = re.compile(rf"(?is)\b(?:at most|no more than)\s*{_NUM}\b")

_Q_BEFORE_YEAR = re.compile(r"(?is)\bbefore\s+(?P<year>\d{4})\b")
_Q_AFTER_YEAR = re.compile(r"(?is)\bafter\s+(?P<year>\d{4})\b")
_Q_ON_YEAR = re.compile(r"(?is)\b(?:on|in)\s+(?P<year>\d{4})\b")


def _extract_numeric_constraints(question: str) -> list[NumericConstraint]:
    q = question
    found: list[NumericConstraint] = []

    for rx, kind in (
        (_Q_EXACT, "EXACT"),
        (_Q_ABOVE, "ABOVE"),
        (_Q_BELOW, "BELOW"),
        (_Q_AT_LEAST, "AT_LEAST"),
        (_Q_AT_MOST, "AT_MOST"),
    ):
        for m in rx.finditer(q):
            found.append(NumericConstraint(kind=kind, value=m.group("val")))
    # Deduplicate identical constraints
    uniq = {}
    for c in found:
        uniq[(c.kind, c.value)] = c
    return list(uniq.values())


def _extract_year_constraints(question: str) -> list[YearConstraint]:
    found: list[YearConstraint] = []
    for rx, kind in (
        (_Q_BEFORE_YEAR, "BEFORE"),
        (_Q_AFTER_YEAR, "AFTER"),
        (_Q_ON_YEAR, "ON"),
    ):
        for m in rx.finditer(question):
            found.append(YearConstraint(kind=kind, year=m.group("year")))
    uniq = {}
    for c in found:
        uniq[(c.kind, c.year)] = c
    return list(uniq.values())


def _evidence_contains_operator(evidence_texts: Iterable[str], *, value: str, op: str) -> bool:
    """
    op in {EXACT, ABOVE, BELOW, AT_LEAST, AT_MOST}
    """
    joined = "\n".join(evidence_texts)
    e = _norm(joined)
    n = str(value).casefold()

    # Equality patterns: either "exactly N" / "equal to N" or "is N" / "equals N"
    if op == "EXACT":
        exact_rx = re.compile(
            rf"(?is)\b(?:exactly|equal to)\s*{re.escape(n)}\b"
            rf"|\b(?:is|equals|equal to)\s*{re.escape(n)}\b"
        )
        return bool(exact_rx.search(e))

    if op == "ABOVE":
        rx = re.compile(rf"(?is)\b(?:above|greater than|more than)\s*{re.escape(n)}\b")
        return bool(rx.search(e))

    if op == "BELOW":
        rx = re.compile(rf"(?is)\b(?:below|less than)\s*{re.escape(n)}\b")
        return bool(rx.search(e))

    if op == "AT_LEAST":
        rx = re.compile(rf"(?is)\b(?:at least|no less than)\s*{re.escape(n)}\b")
        return bool(rx.search(e))

    if op == "AT_MOST":
        rx = re.compile(rf"(?is)\b(?:at most|no more than)\s*{re.escape(n)}\b")
        return bool(rx.search(e))

    return False


def _evidence_contains_year_op(evidence_texts: Iterable[str], *, year: str, op: str) -> bool:
    joined = "\n".join(evidence_texts)
    e = _norm(joined)
    y = str(year)

    if op == "BEFORE":
        rx = re.compile(rf"(?is)\bbefore\s+{re.escape(y)}\b")
        return bool(rx.search(e))
    if op == "AFTER":
        rx = re.compile(rf"(?is)\bafter\s+{re.escape(y)}\b")
        return bool(rx.search(e))
    if op == "ON":
        # "in 2026" / "on 2026"
        rx = re.compile(rf"(?is)\b(?:on|in)\s+{re.escape(y)}\b")
        return bool(rx.search(e))
    return False


def _numeric_entails(required: NumericConstraint, evidence_texts: Iterable[str]) -> bool:
    v = required.value
    kind = required.kind

    # Detect evidence operator for that same value.
    evidence_exact = _evidence_contains_operator(evidence_texts, value=v, op="EXACT")
    evidence_above = _evidence_contains_operator(evidence_texts, value=v, op="ABOVE")
    evidence_below = _evidence_contains_operator(evidence_texts, value=v, op="BELOW")
    evidence_at_least = _evidence_contains_operator(evidence_texts, value=v, op="AT_LEAST")
    evidence_at_most = _evidence_contains_operator(evidence_texts, value=v, op="AT_MOST")

    if kind == "EXACT":
        return evidence_exact
    if kind == "ABOVE":
        # AT_LEAST does NOT entail strict ABOVE; EXACT does NOT entail ABOVE.
        return evidence_above
    if kind == "BELOW":
        return evidence_below
    if kind == "AT_LEAST":
        return evidence_at_least or evidence_above or evidence_exact
    if kind == "AT_MOST":
        return evidence_at_most or evidence_below or evidence_exact

    # Unknown kind => conservative abstain.
    return False


def _year_entails(required: YearConstraint, evidence_texts: Iterable[str]) -> bool:
    y = required.year
    kind = required.kind

    evidence_before = _evidence_contains_year_op(evidence_texts, year=y, op="BEFORE")
    evidence_after = _evidence_contains_year_op(evidence_texts, year=y, op="AFTER")
    evidence_on = _evidence_contains_year_op(evidence_texts, year=y, op="ON")

    if kind == "BEFORE":
        return evidence_before
    if kind == "AFTER":
        return evidence_after
    if kind == "ON":
        return evidence_on
    return False


def should_abstain_due_to_answerability_constraint(
    *,
    question: str,
    evidence_texts: Iterable[str],
) -> bool:
    """
    Returns True if the semantic constraint in the question is NOT entailed
    by the evidence (SAFE-ABSTAIN).
    """
    numeric = _extract_numeric_constraints(question)
    years = _extract_year_constraints(question)

    if not numeric and not years:
        return False  # No explicit constraint detected.

    evidence_texts = list(evidence_texts)

    # Fail-closed per detected constraint: all constraints must be entailed.
    for c in numeric:
        if not _numeric_entails(c, evidence_texts):
            return True
    return any(not _year_entails(c, evidence_texts) for c in years)

