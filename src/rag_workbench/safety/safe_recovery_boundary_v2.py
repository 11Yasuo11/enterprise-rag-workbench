from __future__ import annotations

from collections.abc import Iterable

from rag_workbench.safety.answerability_constraint_guard_v1 import (
    should_abstain_due_to_answerability_constraint,
)
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2


def should_abstain_due_to_safe_recovery_boundary_v2(
    *,
    question: str,
    evidence_texts: Iterable[str] = (),
) -> bool:
    # Returns True iff Phase 5D S3 must SAFE-ABSTAIN:
    # - Injection-style questions: abstain.
    # - Semantic constraint mismatches: abstain.
    return is_question_injection_v2(question) or should_abstain_due_to_answerability_constraint(
        question=question, evidence_texts=evidence_texts
    )

