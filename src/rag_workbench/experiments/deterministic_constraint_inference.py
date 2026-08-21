"""Deterministic comparisons for grounded numeric, deadline, and cadence questions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

Operator = Literal[">", ">=", "<", "<=", "=", "NOT_EQUAL", "within", "at most", "at least"]

_ONES = {
    "zero": 0, "one": 1, "first": 1, "two": 2, "second": 2, "three": 3,
    "third": 3, "four": 4, "fourth": 4, "five": 5, "fifth": 5, "six": 6,
    "sixth": 6, "seven": 7, "seventh": 7, "eight": 8, "eighth": 8,
    "nine": 9, "ninth": 9, "ten": 10, "tenth": 10, "eleven": 11,
    "eleventh": 11, "twelve": 12, "twelfth": 12,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_NUMBER = (
    r"(?:\d+(?:\.\d+)?|"
    r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?|"
    r"(?:zero|one|first|two|second|three|third|four|fourth|five|fifth|six|sixth|"
    r"seven|seventh|eight|eighth|nine|ninth|ten|tenth|eleven|eleventh|twelve|twelfth))"
)
_UNIT = r"(?:business\s+days?|minutes?|hours?|days?|weeks?|months?|years?|items?|times?)"


@dataclass(frozen=True)
class GroundedConstraint:
    operator: Operator
    value: Decimal
    unit: str
    raw_text: str


@dataclass(frozen=True)
class ConstraintInference:
    conclusion: bool
    policy: GroundedConstraint
    scenario: GroundedConstraint
    relation: str


def _number(text: str) -> Decimal | None:
    normalized = text.casefold().replace("-", " ")
    try:
        return Decimal(normalized)
    except Exception:
        pass
    parts = normalized.split()
    if len(parts) == 1 and parts[0] in _ONES:
        return Decimal(_ONES[parts[0]])
    if parts and parts[0] in _TENS:
        return Decimal(_TENS[parts[0]] + (_ONES.get(parts[1], 0) if len(parts) == 2 else 0))
    return None


def _unit(text: str) -> str:
    value = re.sub(r"\s+", " ", text.casefold()).rstrip("s")
    return value


def extract_grounded_constraints(
    text: str, *, scenario: bool = False
) -> tuple[GroundedConstraint, ...]:
    patterns: tuple[tuple[Operator, str], ...] = (
        ("within", rf"\bwithin\s+(?P<value>{_NUMBER})\s+(?P<unit>{_UNIT})\b"),
        ("at most", rf"\b(?:at most|no more than)\s+(?P<value>{_NUMBER})\s+(?P<unit>{_UNIT})\b"),
        ("at least", rf"\b(?:at least|no less than)\s+(?P<value>{_NUMBER})\s+(?P<unit>{_UNIT})\b"),
        (
            "NOT_EQUAL",
            rf"\b(?:not equal to|other than)\s+(?P<value>{_NUMBER})\s+(?P<unit>{_UNIT})\b",
        ),
        (">=", rf"(?:>=|greater than or equal to)\s*(?P<value>{_NUMBER})\s+(?P<unit>{_UNIT})\b"),
        ("<=", rf"(?:<=|less than or equal to)\s*(?P<value>{_NUMBER})\s+(?P<unit>{_UNIT})\b"),
        (">", rf"(?:>|greater than|more than)\s*(?P<value>{_NUMBER})\s+(?P<unit>{_UNIT})\b"),
        ("<", rf"(?:<|less than)\s*(?P<value>{_NUMBER})\s+(?P<unit>{_UNIT})\b"),
        ("=", rf"\b(?:exactly|equal to|every)\s+(?P<value>{_NUMBER})[- ](?P<unit>{_UNIT})\b"),
    )
    found: list[tuple[int, GroundedConstraint]] = []
    occupied: list[tuple[int, int]] = []
    for operator, pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if any(match.start() < end and start < match.end() for start, end in occupied):
                continue
            value = _number(match.group("value"))
            if value is None:
                continue
            found.append(
                (
                    match.start(),
                    GroundedConstraint(
                        operator, value, _unit(match.group("unit")), match.group(0)
                    ),
                )
            )
            occupied.append(match.span())
    if scenario:
        bare = re.compile(rf"\b(?P<value>{_NUMBER})(?:st|nd|rd|th)?[- ](?P<unit>{_UNIT})\b", re.I)
        for match in bare.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in occupied):
                continue
            value = _number(match.group("value"))
            if value is not None:
                found.append(
                    (
                        match.start(),
                        GroundedConstraint(
                            "=", value, _unit(match.group("unit")), match.group(0)
                        ),
                    )
                )
    return tuple(item for _, item in sorted(found, key=lambda pair: pair[0]))


def infer_constraint(question: str, supporting_span: str) -> ConstraintInference | None:
    policy_constraints = extract_grounded_constraints(supporting_span)
    scenario_constraints = extract_grounded_constraints(question, scenario=True)
    if len(policy_constraints) != 1 or not scenario_constraints:
        return None
    policy = policy_constraints[0]
    candidates = [item for item in scenario_constraints if item.unit == policy.unit]
    if len(candidates) > 1:
        candidates = [item for item in candidates if item.value != policy.value]
    if len(candidates) != 1:
        return None
    scenario_value = candidates[0].value
    comparisons = {
        "within": scenario_value <= policy.value,
        "at most": scenario_value <= policy.value,
        "<=": scenario_value <= policy.value,
        "at least": scenario_value >= policy.value,
        ">=": scenario_value >= policy.value,
        ">": scenario_value > policy.value,
        "<": scenario_value < policy.value,
        "=": scenario_value == policy.value,
        "NOT_EQUAL": scenario_value != policy.value,
    }
    conclusion = comparisons[policy.operator]
    relation = "matches" if policy.operator == "=" else "satisfies"
    return ConstraintInference(conclusion, policy, candidates[0], relation)
