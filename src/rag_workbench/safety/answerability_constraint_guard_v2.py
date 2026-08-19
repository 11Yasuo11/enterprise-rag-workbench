# ruff: noqa: E501
from __future__ import annotations

# ANSWERABILITY_CONSTRAINT_GUARD_V2
# Deterministic constraint semantics guard with explicit relational boundary
# modeling:
# - EQ / GT / GE / LT / LE for numeric thresholds
# - BEFORE / AFTER / ON for date thresholds
#
# Guard policy:
# - If a constraint/operator is detected in the question but we cannot find a
#   corresponding supporting evidence relation, SAFE-ABSTAIN (fail-closed).
# - For inequality operators, entailment is modeled as set inclusion over
#   open/closed boundaries.
# - For EQ and ON operators, entailment is stricter: only an explicit matching
#   evidence operator entails the corresponding EQ/ON requirement.
import re
from collections.abc import Iterable
from dataclasses import dataclass

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def _norm(text: str) -> str:
    return text.translate(_ZERO_WIDTH).replace("\xa0", " ").strip().casefold()


@dataclass(frozen=True)
class NumericConstraint:
    op: str  # EQ, GT, GE, LT, LE
    value: str


@dataclass(frozen=True)
class DateConstraint:
    op: str  # BEFORE, AFTER, ON
    value: str  # year or ISO date-ish token captured by regex


_NUM = r"(?P<val>\d+(?:\.\d+)?)"
_ISO_DATE = r"(?P<date>\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|\d{4})"
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

# --- Question extraction ---

_Q_EXACT = re.compile(rf"(?is)\b(?:exactly|equal to)\s*{_NUM}\b")
_Q_ABOVE = re.compile(rf"(?is)\b(?:above|greater than|more than)\s*{_NUM}\b")
_Q_AT_LEAST = re.compile(rf"(?is)\b(?:at least|no less than|min(?:imum)?)\s*{_NUM}\b")
_Q_BELOW = re.compile(rf"(?is)\b(?:below|less than)\s*{_NUM}\b")
_Q_AT_MOST = re.compile(rf"(?is)\b(?:at most|no more than|max(?:imum)?)\s*{_NUM}\b")

_Q_BEFORE = re.compile(rf"(?is)\bbefore\s+{_ISO_DATE}\b")
_Q_AFTER = re.compile(rf"(?is)\bafter\s+{_ISO_DATE}\b")
_Q_ON = re.compile(rf"(?is)\b(?:on|in)\s+{_ISO_DATE}\b")


def _extract_numeric_constraints(question: str) -> list[NumericConstraint]:
    found: list[NumericConstraint] = []
    for rx, op in (
        (_Q_EXACT, "EQ"),
        (_Q_ABOVE, "GT"),
        (_Q_AT_LEAST, "GE"),
        (_Q_BELOW, "LT"),
        (_Q_AT_MOST, "LE"),
    ):
        for m in rx.finditer(question):
            found.append(NumericConstraint(op=op, value=m.group("val")))
    uniq: dict[tuple[str, str], NumericConstraint] = {}
    for c in found:
        uniq[(c.op, c.value)] = c
    return list(uniq.values())


def _extract_date_constraints(question: str) -> list[DateConstraint]:
    found: list[DateConstraint] = []
    for rx, op in (
        (_Q_BEFORE, "BEFORE"),
        (_Q_AFTER, "AFTER"),
        (_Q_ON, "ON"),
    ):
        for m in rx.finditer(question):
            found.append(DateConstraint(op=op, value=m.group("date")))
    uniq: dict[tuple[str, str], DateConstraint] = {}
    for c in found:
        uniq[(c.op, c.value)] = c
    return list(uniq.values())


# --- Evidence extraction ---

def _evidence_join(evidence_texts: Iterable[str]) -> str:
    return "\n".join(evidence_texts)


def _evidence_find_numeric_relation(evidence_texts: Iterable[str]) -> dict[str, str]:
    """
    Returns mapping value -> operator among {EQ, GT, GE, LT, LE}.
    If multiple operators match for the same value, we keep the strictest
    operator detected (EQ strongest; then GE/LE; then GT/LT).
    """
    e = _norm(_evidence_join(evidence_texts))

    def _score(op: str) -> int:
        # Higher score = stronger specificity for matching.
        return {"EQ": 5, "GE": 4, "LE": 4, "GT": 3, "LT": 3}.get(op, 0)

    found: dict[str, str] = {}

    # Explicit threshold operators
    for rx, op in (
        (re.compile(rf"(?is)\b(?:exactly|equal to)\s*{_NUM}\b"), "EQ"),
        (re.compile(rf"(?is)\b(?:above|greater than|more than)\s*{_NUM}\b"), "GT"),
        (re.compile(rf"(?is)\b(?:at least|no less than)\s*{_NUM}\b"), "GE"),
        (re.compile(rf"(?is)\b(?:below|less than)\s*{_NUM}\b"), "LT"),
        (re.compile(rf"(?is)\b(?:at most|no more than)\s*{_NUM}\b"), "LE"),
    ):
        for m in rx.finditer(e):
            val = m.group("val")
            if val not in found or _score(op) > _score(found[val]):
                found[val] = op

    # Equality without "exactly": "is N", "amount is N", "equals N"
    # (We treat this as EQ.)
    eq_rx = re.compile(rf"(?is)\b(?:is|equals|equal to)\s*{_NUM}\b")
    for m in eq_rx.finditer(e):
        val = m.group("val")
        if val not in found or _score("EQ") > _score(found[val]):
            found[val] = "EQ"

    return found


def _evidence_find_date_relation(evidence_texts: Iterable[str]) -> dict[str, str]:
    """
    Returns mapping date_token -> operator among {BEFORE, AFTER, ON}.
    """
    e = _norm(_evidence_join(evidence_texts))
    found: dict[str, str] = {}

    for rx, op in (
        (re.compile(rf"(?is)\bbefore\s+{_ISO_DATE}\b"), "BEFORE"),
        (re.compile(rf"(?is)\bafter\s+{_ISO_DATE}\b"), "AFTER"),
        (re.compile(rf"(?is)\b(?:on|in)\s+{_ISO_DATE}\b"), "ON"),
        # ON evidence often appears as: "launched on April 12, 2026".
        (re.compile(rf"(?is)\bon\s+{_MONTH}\s+\d{{1,2}}\s*,?\s*(?P<date>\d{{4}})\b"), "ON"),
        # Metadata style evidence, e.g. effective_at: "2026-04-12T00:00:00Z"
        (re.compile(r"(?is)\beffective_at\s*:\s*\"?(?P<date>\d{4})-\d{2}-\d{2}"), "ON"),
    ):
        for m in rx.finditer(e):
            dt = m.group("date")
            found.setdefault(dt, op)
            # Prefer ON when ambiguous; otherwise keep first.
            if op == "ON":
                found[dt] = "ON"

    return found


# --- Entailment (relation semantics) ---

def _numeric_entails(required: NumericConstraint, evidence_op: str | None) -> bool:
    if evidence_op is None:
        return False
    if evidence_op == required.op:
        return True

    # EQ is strict: only EQ evidence entails EQ requirement.
    if required.op == "EQ":
        return False

    # ON is handled separately for dates.

    # Inequality entailment via interval containment over open/closed boundaries.
    #
    # Required sets:
    #   GT(x) -> (x, +inf)
    #   GE(x) -> [x, +inf)
    #   LT(x) -> (-inf, x)
    #   LE(x) -> (-inf, x]
    #
    # Evidence must be a superset of the required set.
    #
    # - GE(x) does NOT entail GT(x)? Actually evidence GE includes x, which is
    #   extra, but required GT excludes x. That is OK for policy scope:
    #   if evidence says >=x then it covers all >x values.
    #   Therefore GE entails GT.
    #
    # - GT(x) does NOT entail GE(x) because GT excludes x.

    required_to_entailing_evidence = {
        "GT": {"GT", "GE"},
        "GE": {"GE"},
        "LT": {"LT", "LE"},
        "LE": {"LE"},
    }
    return evidence_op in required_to_entailing_evidence.get(required.op, set())


def _date_entails(required: DateConstraint, evidence_op: str | None) -> bool:
    if evidence_op is None:
        return False
    if required.op == "ON":
        # Strict: only ON evidence entails ON requirement.
        return evidence_op == "ON"
    return evidence_op == required.op


def should_abstain_due_to_answerability_constraint(
    *,
    question: str,
    evidence_texts: Iterable[str],
) -> bool:
    """
    Returns True iff we should SAFE-ABSTAIN because evidence does NOT entail
    the explicit semantic constraint detected in the question.
    """
    numeric = _extract_numeric_constraints(question)
    dates = _extract_date_constraints(question)

    if not numeric and not dates:
        return False

    evidence_texts = list(evidence_texts)
    numeric_rel = _evidence_find_numeric_relation(evidence_texts)
    date_rel = _evidence_find_date_relation(evidence_texts)

    # Fail-closed: every detected constraint must be entailed.
    for c in numeric:
        evidence_op = numeric_rel.get(c.value)
        if not _numeric_entails(c, evidence_op):
            return True

    for c in dates:
        evidence_op = date_rel.get(c.value)
        if not _date_entails(c, evidence_op):
            return True

    return False

