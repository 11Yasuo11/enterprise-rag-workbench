"""V12 cross-version resolution contract: explicit multi-version is not ambiguity."""

from __future__ import annotations

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.experiments.atomic_requirement_contract_v1 import (
    assemble_frozen_plan,
    decompose_question,
    deterministic_support_complete,
    extract_direct_support_mappings,
    validate_verifier_result,
)
from rag_workbench.experiments.atomic_requirement_contract_v1.verifier import (
    FrozenRequirementResult,
    FrozenVerifierResult,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.router import (
    RoutingFeatures,
    SelectiveRiskRouter,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.versioning import (
    DeterministicVersionResolver,
    VersionCandidate,
)
from rag_workbench.experiments.universal_requirement_completeness_v1 import (
    UniversalEvidenceChunk,
)
from rag_workbench.retrieval.temporal import plan_temporal_scope
from scripts.run_p1_version_temporal_v1 import semantic_document_ids


def _candidate(
    document: str,
    version: str,
    *,
    active: bool,
    text: str | None = None,
    tenant: str = "acmeai",
    authorized: bool = True,
) -> VersionCandidate:
    return VersionCandidate(
        f"c-{document}-{version}",
        document,
        f"dv-{document}-{version}",
        version,
        text or f"{document} {version}",
        tenant,
        None,
        active,
        authorized=authorized,
    )


def _reporting_top15() -> list[dict[str, object]]:
    return [
        {
            "document_id": "security-incident-policy",
            "document_version_id": "dv-sec-2026",
            "chunk_id": "c-sec-2026",
            "version": "2026",
            "title": "Security Incident Response Policy",
            "section": "Initial notification",
            "text": (
                "Under the current 2026 policy , suspected severity-one incidents must "
                "be reported to the security duty officer within 15 minutes of discovery ."
            ),
        },
        {
            "document_id": "security-incident-policy",
            "document_version_id": "dv-sec-2025",
            "chunk_id": "c-sec-2025",
            "version": "2025",
            "title": "Security Incident Response Policy",
            "section": "Initial notification",
            "text": (
                "Under the 2025 policy , suspected severity-one incidents had to be "
                "reported to the security duty officer within 60 minutes of discovery ."
            ),
        },
        {
            "document_id": "remote-work-policy",
            "document_version_id": "dv-remote-2025",
            "chunk_id": "c-remote-2025",
            "version": "2025",
            "title": "Remote Work Policy",
            "section": "Weekly allowance",
            "text": "Under the 2025 policy , employees could work remotely three days per week .",
        },
        {
            "document_id": "remote-work-policy",
            "document_version_id": "dv-remote-2026",
            "chunk_id": "c-remote-2026",
            "version": "2026",
            "title": "Remote Work Policy",
            "section": "Weekly allowance",
            "text": (
                "Under the current 2026 policy , employees may work remotely "
                "two days per week ."
            ),
        },
    ]


def test_explicit_version_a_and_b_comparison_is_cross_version() -> None:
    scope = plan_temporal_scope(
        "Set the 2025 and 2026 severity-one reporting windows side by side."
    )
    assert scope.temporal_mode == "CROSS_VERSION"
    assert scope.requested_versions == ("2025", "2026")


def test_historical_plus_current_requested_together() -> None:
    scope = plan_temporal_scope(
        "Compare the historical 2025 remote allowance with the current 2026 allowance."
    )
    assert scope.temporal_mode == "CROSS_VERSION"
    assert scope.requested_versions == ("2025", "2026")


def test_two_requested_versions_with_different_values_are_not_conflict() -> None:
    candidates = (
        _candidate(
            "security-incident-policy",
            "2025",
            active=False,
            text="severity-one reported within 60 minutes",
        ),
        _candidate(
            "security-incident-policy",
            "2026",
            active=True,
            text="severity-one reported within 15 minutes",
        ),
    )
    resolution = DeterministicVersionResolver().resolve(
        "Set the 2025 and 2026 severity-one reporting windows side by side.",
        candidates,
        tenant_id="acmeai",
    )
    assert resolution.status == "VERSION_SET_RESOLVED"
    assert resolution.selections_by_document == {
        "security-incident-policy": ("2025", "2026")
    }


def test_same_requirement_across_two_versions_preserves_both_identities() -> None:
    plan = decompose_question(
        "How did the remote-work allowance change from 2025 to 2026?"
    )
    evidence = (
        GateEvidence(
            "c25",
            "remote-work-policy",
            "dv25",
            "2025",
            "Under the 2025 policy , employees could work remotely three days per week .",
        ),
        GateEvidence(
            "c26",
            "remote-work-policy",
            "dv26",
            "2026",
            "Under the current 2026 policy , employees may work remotely two days per week .",
        ),
    )
    mappings = extract_direct_support_mappings(plan, evidence)
    versions = sorted({item.version for item in mappings})
    assert "2025" in versions and "2026" in versions
    support = deterministic_support_complete(
        plan,
        mappings,
        evidence,
        authorized_chunk_ids=frozenset(item.chunk_id for item in evidence),
        selected_versions_by_document={"remote-work-policy": frozenset({"dv25", "dv26"})},
    )
    assert support.complete is True


def test_separate_requirements_each_tied_to_a_version() -> None:
    candidates = (
        _candidate("remote-work-policy", "2025", active=False),
        _candidate("remote-work-policy", "2026", active=True),
    )
    resolution = DeterministicVersionResolver().resolve(
        "Contrast the weekly remote allowance in the 2025 and 2026 policy editions.",
        candidates,
        tenant_id="acmeai",
    )
    assert resolution.status == "VERSION_SET_RESOLVED"
    assert len(resolution.selected_version_set) == 2


def test_version_set_resolved_accepted_by_router() -> None:
    route = SelectiveRiskRouter().initial(
        RoutingFeatures(
            security_precheck_requires_abstention=False,
            deterministic_support_complete=False,
            conflicting_evidence=False,
            version_sensitive=True,
            version_resolved=True,
            multiple_active_versions=False,
            version_metadata_complete=True,
            top15_evidence_complete=False,
            required_fact_count=2,
        )
    )
    assert route.reason != "unresolved_version"
    assert route.route == "LUNA"


def test_both_version_scoped_outputs_assembled_and_cited() -> None:
    plan = decompose_question(
        "How did the remote-work allowance change from 2025 to 2026?"
    )
    evidence = (
        GateEvidence(
            "c25",
            "remote-work-policy",
            "dv25",
            "2025",
            "Under the 2025 policy , employees could work remotely three days per week .",
        ),
        GateEvidence(
            "c26",
            "remote-work-policy",
            "dv26",
            "2026",
            "Under the current 2026 policy , employees may work remotely two days per week .",
        ),
    )
    mappings = extract_direct_support_mappings(plan, evidence)
    support = deterministic_support_complete(
        plan,
        mappings,
        evidence,
        authorized_chunk_ids=frozenset(item.chunk_id for item in evidence),
        selected_versions_by_document={"remote-work-policy": frozenset({"dv25", "dv26"})},
    )
    assert support.complete is True
    assembly = assemble_frozen_plan(
        plan,
        support.validation,
        tuple(
            UniversalEvidenceChunk(
                item.chunk_id, item.document_id, item.text, item.document_version_id
            )
            for item in evidence
        ),
        authorized_chunk_ids=frozenset(item.chunk_id for item in evidence),
        selected_versions_by_document={"remote-work-policy": frozenset({"dv25", "dv26"})},
    )
    assert assembly.status == "answered"
    assert "three days per week" in (assembly.answer or "")
    assert "two days per week" in (assembly.answer or "")
    assert len(assembly.citations) >= 2


def test_cross_version_deterministic_support_when_both_mappings_complete() -> None:
    plan = decompose_question(
        "Create a two-year change card: remote allowance and incident reporting "
        "deadline for both 2025 and 2026."
    )
    evidence = (
        GateEvidence(
            "a25",
            "remote-work-policy",
            "dv25",
            "2025",
            "Under the 2025 policy , employees could work remotely three days per week .",
        ),
        GateEvidence(
            "a26",
            "remote-work-policy",
            "dv26",
            "2026",
            "Under the current 2026 policy , employees may work remotely two days per week .",
        ),
        GateEvidence(
            "d25",
            "security-incident-policy",
            "dvi25",
            "2025",
            "Under the 2025 policy , suspected severity-one incidents had to be reported "
            "to the security duty officer within 60 minutes of discovery .",
        ),
        GateEvidence(
            "d26",
            "security-incident-policy",
            "dvi26",
            "2026",
            "Under the current 2026 policy , suspected severity-one incidents must be "
            "reported to the security duty officer within 15 minutes of discovery .",
        ),
    )
    mappings = extract_direct_support_mappings(plan, evidence)
    support = deterministic_support_complete(
        plan,
        mappings,
        evidence,
        authorized_chunk_ids=frozenset(item.chunk_id for item in evidence),
        selected_versions_by_document={
            "remote-work-policy": frozenset({"dv25", "dv26"}),
            "security-incident-policy": frozenset({"dvi25", "dvi26"}),
        },
    )
    assert support.complete is True


def test_semantic_document_ids_uses_evidence_tokens_not_only_document_id() -> None:
    question = "Set the 2025 and 2026 severity-one reporting windows side by side."
    selected = semantic_document_ids(question, _reporting_top15())
    assert selected == {"security-incident-policy"}


def test_semantic_document_ids_falls_back_to_top15_when_no_overlap() -> None:
    top15 = [
        {
            "document_id": "alpha-handbook",
            "title": "Alpha",
            "section": "Misc",
            "text": "Completely unrelated wording about cafeteria hours.",
        },
        {
            "document_id": "beta-handbook",
            "title": "Beta",
            "section": "Misc",
            "text": "Other unrelated cafeteria seating rules.",
        },
    ]
    selected = semantic_document_ids("Explain the zeta protocol.", top15)
    assert selected == {"alpha-handbook", "beta-handbook"}


def test_fail_closed_when_only_one_of_two_requested_versions_available() -> None:
    candidates = (_candidate("security-incident-policy", "2026", active=True),)
    resolution = DeterministicVersionResolver().resolve(
        "Set the 2025 and 2026 severity-one reporting windows side by side.",
        candidates,
        tenant_id="acmeai",
    )
    assert resolution.status == "VERSION_AMBIGUOUS"
    assert resolution.selected_version_set == ()


def test_fail_closed_drops_partial_doc_keeps_complete_doc() -> None:
    candidates = (
        _candidate("security-incident-policy", "2025", active=False),
        _candidate("security-incident-policy", "2026", active=True),
        _candidate("remote-work-policy", "2026", active=True),
    )
    resolution = DeterministicVersionResolver().resolve(
        "Set the 2025 and 2026 severity-one reporting windows side by side.",
        candidates,
        tenant_id="acmeai",
    )
    assert resolution.status == "VERSION_SET_RESOLVED"
    assert resolution.selections_by_document == {
        "security-incident-policy": ("2025", "2026")
    }


def test_fail_closed_unauthorized_version_candidate() -> None:
    candidates = (
        _candidate("security-incident-policy", "2025", active=False, authorized=False),
        _candidate("security-incident-policy", "2026", active=True, authorized=False),
    )
    resolution = DeterministicVersionResolver().resolve(
        "Set the 2025 and 2026 severity-one reporting windows side by side.",
        candidates,
        tenant_id="acmeai",
    )
    assert resolution.status == "METADATA_INSUFFICIENT"
    assert resolution.selected_version_set == ()


def test_fail_closed_wrong_tenant() -> None:
    candidates = (
        _candidate("security-incident-policy", "2025", active=False, tenant="other"),
        _candidate("security-incident-policy", "2026", active=True, tenant="other"),
    )
    resolution = DeterministicVersionResolver().resolve(
        "Set the 2025 and 2026 severity-one reporting windows side by side.",
        candidates,
        tenant_id="acmeai",
    )
    assert resolution.status == "METADATA_INSUFFICIENT"


def test_fail_closed_historical_not_requested_stays_current_only() -> None:
    candidates = (
        _candidate("remote-work-policy", "2025", active=False),
        _candidate("remote-work-policy", "2026", active=True),
    )
    resolution = DeterministicVersionResolver().resolve(
        "What is the current remote allowance?",
        candidates,
        tenant_id="acmeai",
    )
    assert resolution.status == "VERSION_RESOLVED"
    assert resolution.candidate is not None
    assert resolution.candidate.version == "2026"


def test_fail_closed_unresolved_temporal_wording_without_versions() -> None:
    scope = plan_temporal_scope("What changed in the prior edition?")
    assert scope.requested_versions == ()
    assert scope.temporal_mode in {"HISTORICAL_ONLY", "UNSPECIFIED_CURRENT_DEFAULT"}


def test_fail_closed_missing_supporting_span_rejects_support() -> None:
    plan = decompose_question(
        "How did the remote-work allowance change from 2025 to 2026?"
    )
    evidence = (
        GateEvidence(
            "a26",
            "remote-work-policy",
            "dv26",
            "2026",
            "Under the current 2026 policy , employees may work remotely two days per week .",
        ),
    )
    mappings = extract_direct_support_mappings(plan, evidence)
    support = deterministic_support_complete(
        plan,
        mappings,
        evidence,
        authorized_chunk_ids=frozenset({"a26"}),
        selected_versions_by_document={"remote-work-policy": frozenset({"dv26"})},
    )
    assert support.complete is False


def test_router_unresolved_when_version_not_resolved() -> None:
    route = SelectiveRiskRouter().initial(
        RoutingFeatures(
            security_precheck_requires_abstention=False,
            deterministic_support_complete=False,
            conflicting_evidence=False,
            version_sensitive=True,
            version_resolved=False,
            multiple_active_versions=False,
            version_metadata_complete=True,
            top15_evidence_complete=False,
            required_fact_count=2,
        )
    )
    assert route.route == "SOL"
    assert route.reason == "unresolved_version"


def test_validator_accepts_version_set_go_with_both_years() -> None:
    plan = decompose_question(
        "How did the remote-work allowance change from 2025 to 2026?"
    )
    evidence = (
        GateEvidence(
            "c25",
            "remote-work-policy",
            "dv25",
            "2025",
            "Under the 2025 policy , employees could work remotely three days per week .",
        ),
        GateEvidence(
            "c26",
            "remote-work-policy",
            "dv26",
            "2026",
            "Under the current 2026 policy , employees may work remotely two days per week .",
        ),
    )
    result = FrozenVerifierResult(
        decision="GO",
        question_plan_hash=plan.question_plan_hash,
        requirements=[
            FrozenRequirementResult(
                requirement_id=item.requirement_id,
                status="SUPPORTED",
                chunk_ids=["c25", "c26"],
                document_ids=["remote-work-policy", "remote-work-policy"],
                supporting_spans=[evidence[0].text, evidence[1].text],
                document_version_ids=["dv25", "dv26"],
                versions=["2025", "2026"],
            )
            for item in plan.requirements
        ],
    )
    validation = validate_verifier_result(
        plan,
        result,
        evidence,
        authorized_chunk_ids=frozenset(item.chunk_id for item in evidence),
    )
    assert validation.valid is True
