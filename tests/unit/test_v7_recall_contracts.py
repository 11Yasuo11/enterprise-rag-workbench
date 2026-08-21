from __future__ import annotations

from sqlalchemy import select

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.db.models import DocumentVersion
from rag_workbench.experiments.atomic_requirement_contract_v1 import (
    decompose_question,
    validate_verifier_result,
)
from rag_workbench.experiments.atomic_requirement_contract_v1.verifier import (
    FrozenRequirementResult,
    FrozenVerifierResult,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.router import (
    PostLunaFeatures,
    SelectiveRiskRouter,
)
from rag_workbench.retrieval.bm25 import tokenize_bm25
from rag_workbench.retrieval.filters import apply_temporal_version_filter
from rag_workbench.retrieval.temporal import plan_temporal_scope, version_is_eligible


def test_bm25_adds_conservative_singular_forms() -> None:
    assert "receipt" in tokenize_bm25("Receipts are required")
    assert "policy" in tokenize_bm25("policies")
    assert "day" in tokenize_bm25("days")


def test_next_to_construction_creates_two_atomic_requirements() -> None:
    plan = decompose_question("Put the reviewer next to the team paged after an outage.")
    assert [item.requirement_text for item in plan.requirements] == [
        "the reviewer",
        "the team paged after an outage",
    ]


def test_effective_revision_prefix_is_a_qualifier_not_requirement() -> None:
    plan = decompose_question(
        "From the effective incident revision, identify the command role for coordination."
    )
    assert [item.requirement_text for item in plan.requirements] == [
        "the command role for coordination"
    ]
    assert [(item.qualifier_type, item.text) for item in plan.context_qualifiers] == [
        ("version", "From the effective incident revision")
    ]


def test_archived_revision_prefix_is_a_qualifier_not_requirement() -> None:
    plan = decompose_question(
        "In the archived remote-work revision, what was the schedule-review rhythm?"
    )
    assert len(plan.requirements) == 1
    assert plan.context_qualifiers[0].qualifier_type == "version"


def test_trailing_two_year_scope_does_not_become_a_bare_year_requirement() -> None:
    plan = decompose_question(
        "Create a change card: allowance and reporting deadline for both 2025 and 2026."
    )
    assert len(plan.requirements) == 2
    assert all(item.requirement_text != "2026" for item in plan.requirements)
    assert [item.text for item in plan.context_qualifiers] == ["2025", "2026"]


def test_multiple_explicit_versions_override_current_wording() -> None:
    scope = plan_temporal_scope("Give the 2025 value and the active 2026 value.")
    assert scope.temporal_mode == "CROSS_VERSION"
    assert scope.requested_versions == ("2025", "2026")


def test_archived_without_explicit_version_selects_only_inactive_versions() -> None:
    scope = plan_temporal_scope("In the archived policy, what was the cadence?")
    assert scope.temporal_mode == "HISTORICAL_ONLY"
    assert version_is_eligible(scope, version="old", is_active=False)
    assert not version_is_eligible(scope, version="current", is_active=True)


def test_archived_sql_scope_filters_to_inactive_authorized_versions() -> None:
    statement = apply_temporal_version_filter(
        select(DocumentVersion),
        plan_temporal_scope("In the archived policy, what was the cadence?"),
    )
    assert "document_versions.is_active IS false" in str(statement)


def test_single_document_identity_broadcast_is_validated_without_loss() -> None:
    plan = decompose_question("Compare the 2025 allowance and the 2026 allowance.")
    evidence = (
        GateEvidence(
            "c25",
            "remote",
            "dv25",
            "2025",
            "Under the 2025 policy , allowance was three days.",
        ),
        GateEvidence(
            "c26",
            "remote",
            "dv26",
            "2026",
            "Under the current 2026 policy , allowance is two days.",
        ),
    )
    result = FrozenVerifierResult(
        decision="GO",
        question_plan_hash=plan.question_plan_hash,
        requirements=[
            FrozenRequirementResult(
                requirement_id="R1",
                status="SUPPORTED",
                chunk_ids=["c25"],
                document_ids=["remote"],
                supporting_spans=["Under the 2025 policy , allowance was three days."],
                document_version_ids=["dv25"],
                versions=["2025"],
            ),
            FrozenRequirementResult(
                requirement_id="R2",
                status="SUPPORTED",
                chunk_ids=["c25", "c26"],
                document_ids=["remote", "remote"],
                supporting_spans=[
                    "Under the 2025 policy , allowance was three days.",
                    "Under the current 2026 policy , allowance is two days.",
                ],
                document_version_ids=["dv25", "dv26"],
                versions=["2025", "2026"],
            ),
        ],
    )
    validation = validate_verifier_result(
        plan,
        result,
        evidence,
        authorized_chunk_ids=frozenset({"c25", "c26"}),
        selected_versions_by_document={"remote": frozenset({"dv25", "dv26"})},
    )
    assert validation.valid
    assert len(validation.requirements[1].mappings) == 2


def test_complete_authorized_packets_make_luna_abstain_suspicious() -> None:
    route = SelectiveRiskRouter().after_luna(
        PostLunaFeatures(
            decision="ABSTAIN",
            genuine_required_evidence_absence=False,
            authorized_candidate_evidence_complete=True,
        )
    )
    assert route.route == "SOL"
    assert route.reason == "suspicious_luna_abstention"


def test_empty_required_packet_remains_safe_abstain() -> None:
    route = SelectiveRiskRouter().after_luna(
        PostLunaFeatures(
            decision="ABSTAIN",
            genuine_required_evidence_absence=True,
            authorized_candidate_evidence_complete=False,
        )
    )
    assert route.route == "SAFE_ABSTAIN"
    assert route.reason == "required_evidence_absent"
