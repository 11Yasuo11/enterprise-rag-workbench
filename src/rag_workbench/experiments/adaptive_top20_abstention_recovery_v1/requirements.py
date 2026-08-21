"""Deterministic, label-blind decomposition of benchmark questions."""

from __future__ import annotations

import re

_AUDIT_PREFIX = re.compile(r"^.*?tundra\.\s*", re.IGNORECASE)


def decompose_requirements(question: str) -> tuple[str, ...]:
    """Split the user-visible request into atomic information requirements.

    This intentionally does not use evaluator-only required facts/documents.
    The decomposition is sent to Luna, while labels remain scorer-only.
    """
    request = _AUDIT_PREFIX.sub("", question).strip().rstrip("?.")
    request = re.sub(r"^(provide|state|report)\s+", "", request, flags=re.IGNORECASE)
    request = re.sub(r"^both\s+", "", request, flags=re.IGNORECASE)
    if "," in request:
        parts = [p.strip() for p in re.split(r",\s*(?:and\s+)?|\s+and\s+", request)]
    elif re.search(r"\s+and\s+", request, flags=re.IGNORECASE):
        parts = re.split(r"\s+and\s+", request, flags=re.IGNORECASE)
    else:
        parts = [request]
    cleaned = tuple(p.strip(" .") for p in parts if p.strip(" ."))
    return cleaned or (question.strip(),)
