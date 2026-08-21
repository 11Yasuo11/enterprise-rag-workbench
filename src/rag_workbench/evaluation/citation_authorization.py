"""Citation authorization with frozen temporal and security identities.

The scorer must authorize the exact evidence identity selected at runtime.  It
must not infer version eligibility from a chunk id or from partial citation
metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.db.models import Chunk, Document, DocumentPermission, DocumentVersion
from rag_workbench.retrieval.temporal import TemporalScopePlan, version_is_eligible
from rag_workbench.security.permissions import Principal


@dataclass(frozen=True)
class CitationEvidenceIdentity:
    """Complete identity required to authorize one cited evidence chunk."""

    document_id: str
    document_version_id: str
    version: str
    is_active: bool
    chunk_id: str
    tenant: str
    region: str | None
    visibility: str
    permission_groups: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CitationAuthorizationDecision:
    chunk_id: str
    authorized: bool
    acl_allowed: bool
    tenant_allowed: bool
    region_allowed: bool
    temporal_allowed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_citation_evidence_identities(
    session: Session, chunk_ids: tuple[str, ...]
) -> tuple[CitationEvidenceIdentity, ...]:
    """Load complete, database-backed identities without applying eligibility."""
    if not chunk_ids:
        return ()
    rows = session.execute(
        select(Chunk, Document, DocumentVersion)
        .join(Document, Chunk.document_fk == Document.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .where(Chunk.id.in_(chunk_ids))
    ).all()
    document_fks = {document.id for _, document, _ in rows}
    permissions: dict[str, list[str]] = {item: [] for item in document_fks}
    if document_fks:
        for document_fk, group in session.execute(
            select(DocumentPermission.document_fk, DocumentPermission.permission_group).where(
                DocumentPermission.document_fk.in_(document_fks)
            )
        ):
            permissions[document_fk].append(group)
    by_chunk = {
        chunk.id: CitationEvidenceIdentity(
            document_id=document.document_id,
            document_version_id=version.id,
            version=version.version,
            is_active=bool(version.is_active),
            chunk_id=chunk.id,
            tenant=document.tenant_id,
            region=(document.metadata_ or {}).get("region")
            or (version.metadata_ or {}).get("region")
            or (chunk.metadata_ or {}).get("region"),
            visibility=document.visibility,
            permission_groups=tuple(sorted(permissions.get(document.id, ()))),
        )
        for chunk, document, version in rows
    }
    # Preserve citation order and omit unresolved ids so callers can fail closed.
    return tuple(by_chunk[item] for item in chunk_ids if item in by_chunk)


def _temporal_allowed(
    evidence: CitationEvidenceIdentity, temporal_scope: TemporalScopePlan
) -> bool:
    if temporal_scope.document_scope:
        document_versions = {
            item.document_id: set(item.versions) for item in temporal_scope.document_scope
        }
        allowed_versions = document_versions.get(evidence.document_id)
        if allowed_versions is None or evidence.version not in allowed_versions:
            return False
    return version_is_eligible(
        temporal_scope, version=evidence.version, is_active=evidence.is_active
    )


def authorize_citation_evidence(
    evidence: CitationEvidenceIdentity,
    *,
    principal: Principal,
    temporal_scope: TemporalScopePlan,
    requested_region: str | None = None,
) -> CitationAuthorizationDecision:
    """Authorize security boundaries and frozen temporal scope independently."""
    tenant_allowed = evidence.tenant == principal.tenant_id
    acl_allowed = evidence.visibility == "public" or bool(
        set(evidence.permission_groups) & set(principal.permission_groups)
    )
    region_allowed = requested_region is None or evidence.region in {None, requested_region}
    temporal_allowed = _temporal_allowed(evidence, temporal_scope)
    checks = {
        "TENANT_DENIED": tenant_allowed,
        "ACL_DENIED": acl_allowed,
        "REGION_DENIED": region_allowed,
        "TEMPORAL_SCOPE_DENIED": temporal_allowed,
    }
    reasons = tuple(reason for reason, passed in checks.items() if not passed)
    return CitationAuthorizationDecision(
        chunk_id=evidence.chunk_id,
        authorized=not reasons,
        acl_allowed=acl_allowed,
        tenant_allowed=tenant_allowed,
        region_allowed=region_allowed,
        temporal_allowed=temporal_allowed,
        reasons=reasons,
    )


def authorize_citation_set(
    evidence: tuple[CitationEvidenceIdentity, ...],
    *,
    principal: Principal,
    temporal_scope: TemporalScopePlan,
    requested_region: str | None = None,
) -> tuple[CitationAuthorizationDecision, ...]:
    return tuple(
        authorize_citation_evidence(
            item,
            principal=principal,
            temporal_scope=temporal_scope,
            requested_region=requested_region,
        )
        for item in evidence
    )
