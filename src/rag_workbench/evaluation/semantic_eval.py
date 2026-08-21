"""Explicit boundary for non-deterministic semantic evaluation.

Evidence sufficiency, semantic correctness, completeness, and claim-level citation support
belong here.  The deterministic runner never imports a provider and defaults this layer off.
"""

from __future__ import annotations

from typing import Any, Protocol

SEMANTIC_DIMENSIONS = (
    "evidence_sufficiency",
    "semantic_answer_correctness",
    "answer_completeness",
    "citation_support",
)


class SemanticJudge(Protocol):
    def evaluate(self, case: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]: ...


def evaluate_with_adapter(
    unresolved: list[tuple[dict[str, Any], dict[str, Any]]], judge: SemanticJudge
) -> list[dict[str, Any]]:
    """Run only through an explicitly injected, reviewed adapter."""
    return [judge.evaluate(case, trace) for case, trace in unresolved]
