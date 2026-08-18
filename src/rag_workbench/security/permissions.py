from dataclasses import dataclass

from sqlalchemy import Select, exists, or_, select

from rag_workbench.db.models import Document, DocumentPermission


@dataclass(frozen=True)
class Principal:
    principal_id: str
    tenant_id: str
    permission_groups: frozenset[str] = frozenset()


def apply_document_acl(statement: Select, principal: Principal) -> Select:
    group_access = exists(
        select(DocumentPermission.id).where(
            DocumentPermission.document_fk == Document.id,
            DocumentPermission.permission_group.in_(principal.permission_groups or {"__none__"}),
        )
    )
    return statement.where(
        Document.tenant_id == principal.tenant_id,
        or_(Document.visibility == "public", group_access),
    )
