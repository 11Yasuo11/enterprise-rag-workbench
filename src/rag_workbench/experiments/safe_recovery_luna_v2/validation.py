# ruff: noqa: E501
"""Deterministic pre-LLM checks and post-Luna GO validation."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.db.models import Chunk, Document, DocumentVersion
from rag_workbench.experiments.safe_recovery_luna_v2.verifier import LunaVerifierResult
from rag_workbench.retrieval.filters import apply_temporal_version_filter
from rag_workbench.retrieval.temporal import TemporalScopePlan, version_is_eligible
from rag_workbench.security.permissions import Principal, apply_document_acl


@dataclass(frozen=True)
class PrecheckResult:
    resolved_without_llm: bool
    decision: str | None
    code: str | None
    evidence: str


def deterministic_prechecks(
    *,
    topk: list[dict],
    principal: Principal,
    session: Session,
    temporal_scope: TemporalScopePlan | None = None,
) -> PrecheckResult:
    if not topk:
        return PrecheckResult(True, "ABSTAIN", "EMPTY_TOPK", "no authorized retrieved evidence")
    for item in topk:
        if not item.get("chunk_id") or item.get("text") is None:
            return PrecheckResult(
                True, "ABSTAIN", "MALFORMED_CANDIDATE", "malformed retrieved candidate"
            )
    ids = [item["chunk_id"] for item in topk]
    rows = session.execute(
        select(Chunk, Document, DocumentVersion)
        .join(Document, Chunk.document_fk == Document.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .where(Chunk.id.in_(ids))
    ).all()
    by_id = {chunk.id: (chunk, document, version) for chunk, document, version in rows}
    for item in topk:
        packed = by_id.get(item["chunk_id"])
        if packed is None:
            return PrecheckResult(True, "ABSTAIN", "UNKNOWN_CHUNK", "chunk not in index")
        _chunk, document, version = packed
        if document.tenant_id != principal.tenant_id:
            return PrecheckResult(True, "ABSTAIN", "TENANT_MISMATCH", "tenant mismatch")
        if not version_is_eligible(
            temporal_scope or TemporalScopePlan("UNSPECIFIED_CURRENT_DEFAULT"),
            version=version.version,
            is_active=version.is_active,
        ):
            return PrecheckResult(True, "ABSTAIN", "INACTIVE_VERSION", "inactive version")
    authorized_statement = (
        select(Chunk.id)
        .join(Document, Chunk.document_fk == Document.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .where(Chunk.id.in_(ids))
    )
    authorized_statement = apply_temporal_version_filter(
        authorized_statement, temporal_scope
    )
    authorized = set(session.scalars(apply_document_acl(authorized_statement, principal)).all())
    if authorized != set(ids):
        return PrecheckResult(True, "ABSTAIN", "ACL_REJECTION", "unauthorized chunk in top-k")
    return PrecheckResult(False, None, None, "semantic verification required")


def validate_luna_go(
    result: LunaVerifierResult,
    chunks: tuple[GateEvidence, ...],
    *,
    session: Session,
    principal: Principal,
    temporal_scope: TemporalScopePlan | None = None,
) -> str | None:
    """Return a deterministic failure code, or None if GO is locally valid."""
    if result.decision != "GO":
        return "NOT_GO"
    if result.conflict or not result.all_supported or not result.version_valid or not result.region_valid:
        return "GO_INVARIANT_FAILED"
    if not result.requirements:
        return "MISSING_REQUIREMENTS"
    passed_ids = {c.chunk_id for c in chunks}
    evidence_by_id = {c.chunk_id: c for c in chunks}
    for req in result.requirements:
        if not req.supported or not req.chunk_id or not req.supporting_span:
            return "MISSING_SUPPORTING_SPAN"
        if req.chunk_id not in passed_ids:
            return "FABRICATED_CHUNK_ID"
        if req.supporting_span not in evidence_by_id[req.chunk_id].text:
            return "SPAN_NOT_IN_CHUNK"
    selected = tuple(dict.fromkeys(req.chunk_id for req in result.requirements if req.chunk_id))
    rows = session.execute(
        select(Chunk, Document, DocumentVersion)
        .join(Document, Chunk.document_fk == Document.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .where(Chunk.id.in_(selected))
    ).all()
    if len(rows) != len(selected):
        return "CHUNK_NOT_IN_STORE"
    for chunk, document, version in rows:
        ev = evidence_by_id[chunk.id]
        if document.tenant_id != principal.tenant_id:
            return "TENANT_MISMATCH"
        if not version_is_eligible(
            temporal_scope or TemporalScopePlan("UNSPECIFIED_CURRENT_DEFAULT"),
            version=version.version,
            is_active=version.is_active,
        ):
            return "INACTIVE_VERSION"
        if chunk.text != ev.text:
            return "TEXT_MISMATCH"
    authorized_statement = (
        select(Chunk.id)
        .join(Document, Chunk.document_fk == Document.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .where(Chunk.id.in_(selected))
    )
    authorized_statement = apply_temporal_version_filter(
        authorized_statement, temporal_scope
    )
    authorized = set(session.scalars(apply_document_acl(authorized_statement, principal)).all())
    if authorized != set(selected):
        return "UNAUTHORIZED_CHUNK"
    return None
