"""Deterministic numeric and temporal constraint extraction and validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_workbench.evaluation.final_e2e_scorer_v2 import normalize_fact

MONTH = (
    r"(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)"
)
DATE = rf"{MONTH}\s+\d{{1,2}}\s*,?\s*\d{{4}}"
NUMBER = r"\d+(?:\.\d+)?"
UNIT = r"(?:%|percent|eur|usd|gbp|€|\$|£|minutes?|days?|hours?|weeks?|months?|years?|items?|times?)"


@dataclass(frozen=True)
class Constraint:
    constraint_type: str
    operator: str
    value: str
    unit: str | None
    upper_value: str | None
    raw_text: str

    @property
    def fingerprint(self) -> tuple[str, str, str, str | None, str | None]:
        return (
            self.constraint_type,
            self.operator,
            normalize_fact(self.value),
            normalize_fact(self.unit) if self.unit else None,
            normalize_fact(self.upper_value) if self.upper_value else None,
        )


_TEMPORAL_OPERATORS = (
    ("on_or_before", rf"on\s+or\s+before\s+(?P<value>{DATE})"),
    ("on_or_after", rf"on\s+or\s+after\s+(?P<value>{DATE})"),
    ("before", rf"\bbefore\s+(?P<value>{DATE})"),
    ("after", rf"\bafter\s+(?P<value>{DATE})"),
    ("on", rf"\bon\s+(?P<value>{DATE})"),
)
_NUMERIC_WORD_OPERATORS = (
    (">=", r"(?:at\s+least|no\s+less\s+than)"),
    ("<=", r"(?:at\s+most|no\s+more\s+than)"),
    (">", r"(?:above|greater\s+than|more\s+than)"),
    ("<", r"(?:below|less\s+than)"),
    ("=", r"(?:exactly|equal\s+to)"),
)


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    return any(span[0] < other[1] and other[0] < span[1] for other in occupied)


def extract_constraints(text: str) -> tuple[Constraint, ...]:
    """Extract supported constraints without collapsing boundary semantics."""
    lower = text.casefold()
    found: list[tuple[int, Constraint, tuple[int, int]]] = []
    occupied: list[tuple[int, int]] = []

    between_date = re.compile(
        rf"\bbetween\s+(?P<value>{DATE})\s+and\s+(?P<upper>{DATE})", re.IGNORECASE
    )
    for match in between_date.finditer(text):
        constraint = Constraint(
            "date", "between", match.group("value"), None, match.group("upper"), match.group(0)
        )
        found.append((match.start(), constraint, match.span()))
        occupied.append(match.span())

    for operator, pattern in _TEMPORAL_OPERATORS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if _overlaps(match.span(), occupied):
                continue
            constraint = Constraint(
                "date", operator, match.group("value"), None, None, match.group(0)
            )
            found.append((match.start(), constraint, match.span()))
            occupied.append(match.span())

    for match in re.finditer(DATE, text, re.IGNORECASE):
        if _overlaps(match.span(), occupied):
            continue
        constraint = Constraint("date", "=", match.group(0), None, None, match.group(0))
        found.append((match.start(), constraint, match.span()))
        occupied.append(match.span())

    duration_pattern = re.compile(
        rf"\b(?P<operator>within|after|before)\s+(?P<value>{NUMBER})\s+"
        rf"(?P<unit>minutes?|days?|hours?|weeks?|months?|years?)\b",
        re.IGNORECASE,
    )
    for match in duration_pattern.finditer(text):
        if _overlaps(match.span(), occupied):
            continue
        constraint = Constraint(
            "duration",
            match.group("operator").casefold(),
            match.group("value"),
            match.group("unit"),
            None,
            match.group(0),
        )
        found.append((match.start(), constraint, match.span()))
        occupied.append(match.span())

    range_pattern = re.compile(
        rf"\bbetween\s+(?P<value>{NUMBER})\s+and\s+(?P<upper>{NUMBER})"
        rf"(?:\s*(?P<unit>{UNIT}))?\b",
        re.IGNORECASE,
    )
    for match in range_pattern.finditer(text):
        if _overlaps(match.span(), occupied):
            continue
        constraint = Constraint(
            "numeric",
            "range",
            match.group("value"),
            match.group("unit"),
            match.group("upper"),
            match.group(0),
        )
        found.append((match.start(), constraint, match.span()))
        occupied.append(match.span())

    symbol_pattern = re.compile(
        rf"(?P<operator>>=|<=|>|<|=)\s*(?P<currency>€|\$|£)?\s*"
        rf"(?P<value>{NUMBER})(?:\s*(?P<unit>{UNIT}))?",
        re.IGNORECASE,
    )
    for match in symbol_pattern.finditer(text):
        if _overlaps(match.span(), occupied):
            continue
        unit = match.group("currency") or match.group("unit")
        constraint = Constraint(
            "numeric", match.group("operator"), match.group("value"), unit, None, match.group(0)
        )
        found.append((match.start(), constraint, match.span()))
        occupied.append(match.span())

    for operator, words in _NUMERIC_WORD_OPERATORS:
        pattern = re.compile(
            rf"\b{words}\s+(?P<currency>€|\$|£)?\s*(?P<value>{NUMBER})"
            rf"(?:\s*(?P<unit>{UNIT}))?",
            re.IGNORECASE,
        )
        for match in pattern.finditer(lower):
            if _overlaps(match.span(), occupied):
                continue
            unit = match.group("currency") or match.group("unit")
            constraint = Constraint(
                "numeric", operator, match.group("value"), unit, None, match.group(0)
            )
            found.append((match.start(), constraint, match.span()))
            occupied.append(match.span())

    return tuple(item[1] for item in sorted(found, key=lambda item: item[0]))


@dataclass(frozen=True)
class ConstraintValidation:
    passed: bool
    expected_count: int
    output_count: int
    failure_code: str | None


class DeterministicConstraintValidator:
    def validate(
        self, expected: tuple[Constraint, ...], supporting_span: str, output_text: str
    ) -> ConstraintValidation:
        span_constraints = extract_constraints(supporting_span)
        output_constraints = extract_constraints(output_text)
        expected_fingerprints = tuple(item.fingerprint for item in expected)
        if tuple(item.fingerprint for item in span_constraints) != expected_fingerprints:
            return ConstraintValidation(
                False, len(expected), len(output_constraints), "EVIDENCE_CONSTRAINT_MISMATCH"
            )
        if tuple(item.fingerprint for item in output_constraints) != expected_fingerprints:
            return ConstraintValidation(
                False, len(expected), len(output_constraints), "OUTPUT_CONSTRAINT_MISMATCH"
            )
        return ConstraintValidation(True, len(expected), len(output_constraints), None)
