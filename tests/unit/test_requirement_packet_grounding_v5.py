from types import SimpleNamespace

import pytest

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.experiments.atomic_requirement_contract_v1 import decompose_question
from rag_workbench.experiments.atomic_requirement_contract_v1.verifier import (
    grounded_requirement_evidence,
    normalized_packet_tokens,
    requirement_scoped_evidence_packets,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.router import (
    PostLunaFeatures,
    SelectiveRiskRouter,
)


def mapping(
    *,
    requirement_id: str = "R1",
    chunk_id: str = "c1",
    document_id: str = "doc",
    span: str,
    version_id: str = "dv1",
):
    return SimpleNamespace(
        requirement_id=requirement_id,
        chunk_id=chunk_id,
        document_id=document_id,
        supporting_span=span,
        document_version_id=version_id,
    )


@pytest.mark.parametrize(
    ("requirement", "span"),
    [
        ("receipt threshold", "Receipts are required for expenses above 25 euros."),
        ("filing deadline", "Reports must be submitted within 10 business days."),
        ("status cadence", "Customer updates are published every 30 minutes."),
        ("plan owner", "Operations owns the continuity plan."),
        ("API maintainer", "Platform Interfaces maintains the API."),
    ],
)
def test_validated_mapping_is_first_packet_source_without_abstract_term(
    requirement: str, span: str
) -> None:
    plan = decompose_question(f"State the {requirement}.")
    evidence = (GateEvidence("c1", "doc", "dv1", "1", span),)
    packets = requirement_scoped_evidence_packets(
        plan,
        evidence,
        validated_mappings=(mapping(span=span),),
        authorized_chunk_ids=frozenset({"c1"}),
        selected_versions_by_document={"doc": frozenset({"dv1"})},
    )
    assert packets["R1"] == evidence


@pytest.mark.parametrize(
    ("singular", "plural"),
    [
        ("receipt", "receipts"),
        ("policy", "policies"),
        ("day", "days"),
        ("requirement", "requirements"),
    ],
)
def test_fallback_normalizes_simple_english_plural_morphology(
    singular: str, plural: str
) -> None:
    assert singular in normalized_packet_tokens(plural)


def test_fallback_normalizes_unicode_case_punctuation_and_whitespace() -> None:
    assert normalized_packet_tokens("  POLICIES—Days\t") == frozenset({"policy", "day"})


def test_wrong_document_version_mapping_is_rejected() -> None:
    plan = decompose_question("State the owner.")
    evidence = (GateEvidence("c1", "doc", "dv2", "2", "Operations owns the plan."),)
    grounded = grounded_requirement_evidence(
        plan,
        evidence,
        (mapping(span="Operations owns the plan.", version_id="dv1"),),
        authorized_chunk_ids=frozenset({"c1"}),
        selected_versions_by_document={"doc": frozenset({"dv2"})},
    )
    assert grounded["R1"] == ()


@pytest.mark.parametrize("denial", ["acl", "tenant", "region"])
def test_authorization_denied_mapping_is_rejected(denial: str) -> None:
    plan = decompose_question("State the owner.")
    evidence = (GateEvidence("c1", f"doc-{denial}", "dv1", "1", "Operations owns the plan."),)
    grounded = grounded_requirement_evidence(
        plan,
        evidence,
        (mapping(document_id=f"doc-{denial}", span="Operations owns the plan."),),
        authorized_chunk_ids=frozenset(),
    )
    assert grounded["R1"] == ()


def test_wrong_document_or_nonliteral_span_is_rejected() -> None:
    plan = decompose_question("State the owner.")
    evidence = (GateEvidence("c1", "doc", "dv1", "1", "Operations owns the plan."),)
    grounded = grounded_requirement_evidence(
        plan,
        evidence,
        (
            mapping(document_id="other", span="Operations owns the plan."),
            mapping(span="Finance owns the plan."),
        ),
        authorized_chunk_ids=frozenset({"c1"}),
    )
    assert grounded["R1"] == ()


def test_weak_lexical_fallback_is_not_promoted_to_grounded_evidence() -> None:
    plan = decompose_question("State the owner.")
    evidence = (GateEvidence("c1", "doc", "dv1", "1", "The owner attended training."),)
    packets = requirement_scoped_evidence_packets(plan, evidence)
    grounded = grounded_requirement_evidence(plan, evidence)
    assert packets["R1"] == evidence
    assert grounded["R1"] == ()


def test_grounded_unsupported_requirement_prevents_false_absence_and_escalates() -> None:
    plan = decompose_question("State the receipt threshold and remote-work allowance.")
    evidence = (
        GateEvidence("c1", "finance", "fv1", "1", "Receipts are required above 25 euros."),
        GateEvidence("c2", "remote", "rv2", "2", "Employees may work remotely two days."),
    )
    mappings = (
        mapping(
            document_id="finance",
            version_id="fv1",
            span="Receipts are required above 25 euros.",
        ),
        mapping(
            requirement_id="R2",
            chunk_id="c2",
            document_id="remote",
            version_id="rv2",
            span="Employees may work remotely two days.",
        ),
    )
    grounded = grounded_requirement_evidence(
        plan, evidence, mappings, authorized_chunk_ids=frozenset({"c1", "c2"})
    )
    complete = all(grounded[item.requirement_id] for item in plan.requirements)
    genuine_absence = not bool(grounded["R1"])
    route = SelectiveRiskRouter().after_luna(
        PostLunaFeatures(
            "ABSTAIN",
            genuine_required_evidence_absence=genuine_absence,
            authorized_candidate_evidence_complete=complete,
        )
    )
    assert route.route == "SOL"
    assert route.reason == "suspicious_luna_abstention"
