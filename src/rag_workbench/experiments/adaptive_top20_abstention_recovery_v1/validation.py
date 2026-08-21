"""Fail-closed deterministic validation of a Luna Top-20 GO."""

from __future__ import annotations

from sqlalchemy.orm import Session

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.experiments.adaptive_top20_abstention_recovery_v1.verifier import (
    RecoveryDecision,
)
from rag_workbench.experiments.safe_recovery_luna_v2.validation import deterministic_prechecks
from rag_workbench.security.permissions import Principal


def validate_go(
    result: RecoveryDecision,
    requirements: tuple[str, ...],
    chunks: tuple[GateEvidence, ...],
    *,
    session: Session,
    principal: Principal,
) -> str | None:
    if result.decision != "GO":
        return "NOT_GO"
    if (
        not all(
            (
                result.all_supported,
                result.tenant_valid,
                result.acl_valid,
                result.version_valid,
                result.region_valid,
            )
        )
        or result.conflict
    ):
        return "GO_INVARIANT_FAILED"
    if result.required_count != len(requirements) or len(result.requirements) != len(requirements):
        return "REQUIREMENT_COUNT_MISMATCH"
    if result.supported_count != len(requirements):
        return "INCOMPLETE_REQUIREMENTS"
    expected_ids = [f"R{i}" for i in range(1, len(requirements) + 1)]
    if [r.requirement_id for r in result.requirements] != expected_ids:
        return "REQUIREMENT_ID_MISMATCH"
    by_id = {c.chunk_id: (i, c) for i, c in enumerate(chunks, 1)}
    used_documents: set[str] = set()
    for index, support in enumerate(result.requirements):
        if support.requirement != requirements[index]:
            return "REQUIREMENT_TEXT_MISMATCH"
        if (
            not support.supported
            or not support.chunk_id
            or not support.document_id
            or not support.supporting_span
            or support.cross_encoder_rank is None
        ):
            return "MISSING_EXPLICIT_SUPPORT"
        packed = by_id.get(support.chunk_id)
        if packed is None:
            return "CHUNK_NOT_IN_TOP20"
        actual_rank, chunk = packed
        if support.cross_encoder_rank != actual_rank or support.document_id != chunk.document_id:
            return "RANK_OR_DOCUMENT_MISMATCH"
        if support.supporting_span not in chunk.text:
            return "SPAN_NOT_LITERAL"
        used_documents.add(chunk.document_id)
    if result.distinct_documents_used != len(used_documents):
        return "DISTINCT_DOCUMENT_COUNT_MISMATCH"
    pre = deterministic_prechecks(
        topk=[{"chunk_id": c.chunk_id, "text": c.text} for c in chunks],
        principal=principal,
        session=session,
    )
    if pre.resolved_without_llm:
        return pre.code or "AUTHORIZATION_PRECHECK_FAILED"
    return None
