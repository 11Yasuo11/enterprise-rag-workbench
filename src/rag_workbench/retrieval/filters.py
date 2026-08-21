from dataclasses import dataclass

from sqlalchemy import Select, and_, or_

from rag_workbench.db.models import Document, DocumentVersion
from rag_workbench.retrieval.temporal import TemporalScopePlan


@dataclass(frozen=True)
class RetrievalFilters:
    document_ids: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()
    temporal_scope: TemporalScopePlan | None = None


def apply_temporal_version_filter(
    statement: Select, temporal_scope: TemporalScopePlan | None
) -> Select:
    """Apply temporal eligibility after the caller has constrained authorization."""
    if temporal_scope is None or temporal_scope.temporal_mode in {
        "CURRENT_ONLY",
        "UNSPECIFIED_CURRENT_DEFAULT",
    }:
        statement = statement.where(DocumentVersion.is_active.is_(True))
        if temporal_scope and temporal_scope.requested_versions:
            statement = statement.where(
                DocumentVersion.version.in_(temporal_scope.requested_versions)
            )
        return statement
    if temporal_scope.document_scope:
        clauses = [
            and_(
                Document.document_id == item.document_id,
                DocumentVersion.version.in_(item.versions),
            )
            for item in temporal_scope.document_scope
        ]
        return statement.where(or_(*clauses))
    if temporal_scope.requested_versions:
        return statement.where(DocumentVersion.version.in_(temporal_scope.requested_versions))
    # An unnumbered historical request is still a meaningful temporal scope. The
    # caller has already constrained tenant/region/ACL authorization; within that
    # authorized set, only superseded versions are eligible.
    return statement.where(DocumentVersion.is_active.is_(False))
