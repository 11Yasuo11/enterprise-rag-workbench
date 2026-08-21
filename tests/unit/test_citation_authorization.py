from rag_workbench.evaluation.citation_authorization import (
    CitationEvidenceIdentity,
    authorize_citation_evidence,
)
from rag_workbench.retrieval.temporal import DocumentVersionScope, TemporalScopePlan
from rag_workbench.security.permissions import Principal

PRINCIPAL = Principal("user", "tenant-a", frozenset({"employees"}))


def evidence(**overrides: object) -> CitationEvidenceIdentity:
    values = {
        "document_id": "policy",
        "document_version_id": "policy-v2",
        "version": "2",
        "is_active": True,
        "chunk_id": "chunk-2",
        "tenant": "tenant-a",
        "region": "east",
        "visibility": "restricted",
        "permission_groups": ("employees",),
    }
    values.update(overrides)
    return CitationEvidenceIdentity(**values)  # type: ignore[arg-type]


def decision(item: CitationEvidenceIdentity, scope: TemporalScopePlan, region: str | None = None):
    return authorize_citation_evidence(
        item, principal=PRINCIPAL, temporal_scope=scope, requested_region=region
    )


def test_current_active_version_is_authorized() -> None:
    assert decision(evidence(), TemporalScopePlan("CURRENT_ONLY", ("2",))).authorized


def test_current_superseded_version_is_unauthorized() -> None:
    result = decision(
        evidence(version="1", is_active=False), TemporalScopePlan("CURRENT_ONLY", ("1",))
    )
    assert not result.authorized
    assert result.reasons == ("TEMPORAL_SCOPE_DENIED",)


def test_requested_historical_version_is_authorized() -> None:
    assert decision(
        evidence(version="1", is_active=False), TemporalScopePlan("HISTORICAL_ONLY", ("1",))
    ).authorized


def test_both_cross_version_citations_are_authorized() -> None:
    scope = TemporalScopePlan("CROSS_VERSION", ("1", "2"))
    assert decision(evidence(version="1", is_active=False), scope).authorized
    assert decision(evidence(version="2", is_active=True), scope).authorized


def test_acl_denied_historical_version_is_unauthorized() -> None:
    result = decision(
        evidence(version="1", is_active=False, permission_groups=("security",)),
        TemporalScopePlan("HISTORICAL_ONLY", ("1",)),
    )
    assert not result.authorized
    assert "ACL_DENIED" in result.reasons


def test_wrong_tenant_historical_version_is_unauthorized() -> None:
    result = decision(
        evidence(version="1", is_active=False, tenant="tenant-b"),
        TemporalScopePlan("HISTORICAL_ONLY", ("1",)),
    )
    assert not result.authorized
    assert "TENANT_DENIED" in result.reasons


def test_wrong_region_historical_version_is_unauthorized() -> None:
    result = decision(
        evidence(version="1", is_active=False, region="west"),
        TemporalScopePlan("HISTORICAL_ONLY", ("1",)),
        "east",
    )
    assert not result.authorized
    assert "REGION_DENIED" in result.reasons


def test_unrequested_historical_version_is_unauthorized() -> None:
    result = decision(
        evidence(version="0", is_active=False), TemporalScopePlan("CROSS_VERSION", ("1", "2"))
    )
    assert not result.authorized
    assert "TEMPORAL_SCOPE_DENIED" in result.reasons


def test_document_scope_prevents_cross_document_version_leakage() -> None:
    scope = TemporalScopePlan(
        "CROSS_VERSION",
        ("1", "2"),
        (DocumentVersionScope("policy", ("1",)), DocumentVersionScope("other", ("2",))),
    )
    assert decision(evidence(version="1", is_active=False), scope).authorized
    assert not decision(evidence(version="2", is_active=True), scope).authorized


def test_unspecified_default_is_current_only() -> None:
    scope = TemporalScopePlan("UNSPECIFIED_CURRENT_DEFAULT")
    assert decision(evidence(), scope).authorized
    assert not decision(evidence(version="1", is_active=False), scope).authorized
