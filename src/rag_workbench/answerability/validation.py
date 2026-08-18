from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import (
    AnswerabilityReason,
    AnswerabilityResult,
    GateEvidence,
    GateOperationalError,
    ValidatedAnswerabilityResult,
)
from rag_workbench.db.models import Chunk, Document, DocumentVersion
from rag_workbench.security.permissions import Principal, apply_document_acl


def validate_gate_result_with_error(
    result: AnswerabilityResult,
    retrieved_chunks: tuple[GateEvidence, ...],
    *,
    session: Session,
    principal: Principal,
) -> ValidatedAnswerabilityResult:
    """Fail closed and identify the operational reason without exposing evidence labels."""
    requirement_support = tuple(
        chunk_id
        for requirement in result.requirements
        for chunk_id in requirement.supporting_chunk_ids
    )
    all_selected = tuple(
        dict.fromkeys((*result.supporting_chunk_ids, *requirement_support))
    )
    supporting = tuple(dict.fromkeys(result.supporting_chunk_ids))
    if result.answerable and not supporting:
        return _failed(GateOperationalError.MISSING_SUPPORTING_ID)
    evidence_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
    if not set(all_selected) <= set(evidence_by_id):
        return _failed(GateOperationalError.INVALID_SUPPORTING_ID)

    if not all_selected:
        return ValidatedAnswerabilityResult(result)

    rows = session.execute(
        select(Chunk, DocumentVersion)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .where(Chunk.id.in_(all_selected))
    ).all()
    if len(rows) != len(all_selected):
        return _failed(GateOperationalError.INVALID_SUPPORTING_ID)
    for chunk, version in rows:
        evidence = evidence_by_id[chunk.id]
        if (
            chunk.document_version_id != evidence.document_version_id
            or version.id != evidence.document_version_id
            or version.version != evidence.version
        ):
            return _failed(GateOperationalError.VERSION_MISMATCH)
        if chunk.text != evidence.text:
            return _failed(GateOperationalError.INVALID_SUPPORTING_ID)
        if not version.is_active:
            return _failed(GateOperationalError.INACTIVE_VERSION)
        if evidence.index_identity and chunk.index_identity != evidence.index_identity:
            return _failed(GateOperationalError.EXPERIMENT_MISMATCH)

    authorized_statement = (
        select(Chunk.id)
        .join(Document, Chunk.document_fk == Document.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .where(Chunk.id.in_(all_selected), DocumentVersion.is_active.is_(True))
    )
    authorized = set(
        session.scalars(apply_document_acl(authorized_statement, principal)).all()
    )
    if authorized != set(all_selected):
        return _failed(
            GateOperationalError.UNAUTHORIZED_SUPPORTING_ID,
            AnswerabilityReason.ACCESS_RESTRICTED_EVIDENCE,
        )
    if not result.answerable:
        return ValidatedAnswerabilityResult(
            result.model_copy(update={"supporting_chunk_ids": ()})
        )
    return ValidatedAnswerabilityResult(
        result.model_copy(update={"supporting_chunk_ids": supporting})
    )


def validate_gate_result(
    result: AnswerabilityResult,
    retrieved_chunks: tuple[GateEvidence, ...],
    *,
    session: Session,
    principal: Principal,
) -> AnswerabilityResult:
    """Backward-compatible result-only validation API."""
    return validate_gate_result_with_error(
        result, retrieved_chunks, session=session, principal=principal
    ).result


def _failed(
    error: GateOperationalError,
    reason: AnswerabilityReason = AnswerabilityReason.UNKNOWN,
) -> ValidatedAnswerabilityResult:
    return ValidatedAnswerabilityResult(
        AnswerabilityResult.fail_closed(reason), operational_error=error
    )
