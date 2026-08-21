from __future__ import annotations

from dataclasses import replace

import pytest

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.experiments.atomic_requirement_contract_v1 import (
    DirectSupportMapping,
    assemble_frozen_plan,
    decompose_question,
    deterministic_support_complete,
    extract_direct_support_mappings,
)
from rag_workbench.experiments.universal_requirement_completeness_v1 import (
    UniversalEvidenceChunk,
)


def evidence(
    chunk_id: str,
    document_id: str,
    text: str,
    *,
    version_id: str | None = None,
    version: str = "1",
) -> GateEvidence:
    return GateEvidence(chunk_id, document_id, version_id or f"dv-{chunk_id}", version, text)


def mapping(
    requirement_id: str,
    item: GateEvidence,
    *spans: str,
    citations: tuple[str, ...] | None = None,
) -> DirectSupportMapping:
    return DirectSupportMapping(
        requirement_id,
        item.document_id,
        item.document_version_id,
        item.version,
        item.chunk_id,
        spans,
        citations or (item.chunk_id,),
        "test-direct-fact",
    )


def decide(
    question: str,
    mappings: tuple[DirectSupportMapping, ...],
    rows: tuple[GateEvidence, ...],
    **kwargs,
):
    ids = frozenset(item.chunk_id for item in rows)
    return deterministic_support_complete(
        decompose_question(question),
        mappings,
        rows,
        authorized_chunk_ids=kwargs.pop("authorized_chunk_ids", ids),
        acl_valid_chunk_ids=kwargs.pop("acl_valid_chunk_ids", ids),
        tenant_valid_chunk_ids=kwargs.pop("tenant_valid_chunk_ids", ids),
        region_valid_chunk_ids=kwargs.pop("region_valid_chunk_ids", ids),
        selected_versions_by_document=kwargs.pop("selected_versions_by_document", {}),
        **kwargs,
    )


def test_complete_single_document_support_assembles_without_model() -> None:
    row = evidence("c1", "service-guide", "The service is maintained by Interfaces.")
    decision = decide(
        "Name the service maintainer.",
        (mapping("R1", row, row.text),),
        (row,),
    )
    assert decision.complete
    assembly = assemble_frozen_plan(
        decompose_question("Name the service maintainer."),
        decision.validation,
        (UniversalEvidenceChunk(row.chunk_id, row.document_id, row.text, row.document_version_id),),
        authorized_chunk_ids=frozenset({row.chunk_id}),
        selected_versions_by_document={},
    )
    assert assembly.status == "answered"
    assert assembly.citations == ("c1",)


def test_complete_multi_document_support() -> None:
    first = evidence("c1", "alpha", "Alpha is owned by Operations.")
    second = evidence("c2", "beta", "Beta is maintained by Platform.")
    question = "Pair the Alpha owner with the Beta maintainer."
    decision = decide(
        question,
        (mapping("R1", first, first.text), mapping("R2", second, second.text)),
        (first, second),
    )
    assert decision.complete


def test_complete_cross_version_support() -> None:
    old = evidence(
        "old", "policy", "The 2025 allowance was three days.", version_id="dv25", version="2025"
    )
    new = evidence(
        "new", "policy", "The 2026 allowance is two days.", version_id="dv26", version="2026"
    )
    decision = decide(
        "Compare the 2025 allowance and the 2026 allowance.",
        (mapping("R1", old, old.text), mapping("R2", new, new.text)),
        (old, new),
        selected_versions_by_document={"policy": frozenset({"dv25", "dv26"})},
    )
    assert decision.complete


def test_complete_multiple_spans_and_citations_are_preserved() -> None:
    row = evidence("c1", "policy", "First literal. Second literal.")
    supporting = mapping(
        "R1",
        row,
        "First literal.",
        "Second literal.",
        citations=("c1",),
    )
    decision = decide("State the policy facts.", (supporting,), (row,))
    assert decision.complete
    assert decision.mappings[0].supporting_spans == ("First literal.", "Second literal.")
    assert decision.mappings[0].citations == ("c1",)
    assert len(decision.validation.requirements[0].mappings) == 2


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        ({"authorized_chunk_ids": frozenset()}, "INVALID_EVIDENCE_MAPPING_SCHEMA"),
        ({"acl_valid_chunk_ids": frozenset()}, "ACL_DENIED"),
        ({"tenant_valid_chunk_ids": frozenset()}, "TENANT_DENIED"),
        ({"region_valid_chunk_ids": frozenset()}, "REGION_DENIED"),
        ({"unresolved_version_conflict": True}, "UNRESOLVED_VERSION_CONFLICT"),
        ({"requires_unsupported_inference": True}, "UNSUPPORTED_SEMANTIC_INFERENCE_REQUIRED"),
        ({"security_precheck_failed": True}, "SECURITY_PRECHECK_FAILED"),
        ({"conflicting_requirement_ids": frozenset({"R1"})}, "CONTRADICTORY_MAPPING"),
    ],
)
def test_support_fails_closed_for_security_and_semantic_invariants(change, failure) -> None:
    row = evidence("c1", "policy", "The owner is Operations.")
    decision = decide("Name the owner.", (mapping("R1", row, row.text),), (row,), **change)
    assert not decision.complete
    assert decision.failure_code == failure


def test_missing_requirement_mapping_fails_closed() -> None:
    first = evidence("c1", "alpha", "Alpha is owned by Operations.")
    decision = decide(
        "Pair the Alpha owner with the Beta owner.",
        (mapping("R1", first, first.text),),
        (first,),
    )
    assert not decision.complete
    assert decision.failure_code == "REQUIREMENT_COVERAGE_INCOMPLETE"


def test_wrong_version_fails_closed() -> None:
    row = evidence("c1", "policy", "The allowance is two days.", version_id="current")
    decision = decide(
        "State the allowance.",
        (mapping("R1", row, row.text),),
        (row,),
        selected_versions_by_document={"policy": frozenset({"historical"})},
    )
    assert not decision.complete
    assert decision.failure_code == "INVALID_EVIDENCE_MAPPING_SCHEMA"


def test_nonliteral_span_and_invalid_citation_fail_closed() -> None:
    row = evidence("c1", "policy", "The owner is Operations.")
    nonliteral = replace(mapping("R1", row, row.text), supporting_spans=("The owner is Finance.",))
    invalid_citation = replace(mapping("R1", row, row.text), citations=("missing",))
    assert not decide("Name the owner.", (nonliteral,), (row,)).complete
    assert not decide("Name the owner.", (invalid_citation,), (row,)).complete


def test_direct_extractor_maps_maintenance_without_rank_guessing() -> None:
    rows = (
        evidence("right", "atlas-interface", "The Atlas API is maintained by Interfaces."),
        evidence("wrong", "atlas-launch", "The launch owner is Product Operations."),
    )
    plan = decompose_question("Name the group that keeps the Atlas interface operational.")
    assert [item.chunk_id for item in extract_direct_support_mappings(plan, rows)] == ["right"]


def test_direct_extractor_maps_unique_schedule_anchors() -> None:
    rows = (
        evidence(
            "west", "recovery-runbook-west", "The recovery drill runs on the second Wednesday."
        ),
        evidence(
            "east", "recovery-runbook-east", "The recovery drill runs on the first Wednesday."
        ),
        evidence("remote", "remote-policy", "Managers review schedules every month."),
    )
    plan = decompose_question("State the western rehearsal position.")
    assert [item.chunk_id for item in extract_direct_support_mappings(plan, rows)] == ["west"]


def test_direct_extractor_maps_coordination_allowance_and_deadline() -> None:
    rows = (
        evidence("command", "incident-policy", "The incident commander coordinates Engineering."),
        evidence("remote", "remote-policy", "Employees may work remotely two days per week."),
        evidence(
            "deadline",
            "incident-policy",
            "Severity-one incidents must be reported within 15 minutes.",
        ),
    )
    command = decompose_question(
        "In the effective incident revision, identify the command role for coordination."
    )
    pair = decompose_question(
        "Pair the remote weekly allowance with the severity-one reporting window."
    )
    assert [item.chunk_id for item in extract_direct_support_mappings(command, rows)] == ["command"]
    assert [item.chunk_id for item in extract_direct_support_mappings(pair, rows)] == [
        "remote",
        "deadline",
    ]


def test_direct_extractor_maps_region_scoped_destination() -> None:
    rows = (
        evidence(
            "east",
            "recovery-runbook-east",
            "Failover targets the eu-central standby cluster . "
            "The recovery drill runs on the first Wednesday.",
        ),
        evidence(
            "west",
            "recovery-runbook-west",
            "Failover targets the us-east standby cluster . "
            "The recovery drill runs on the second Wednesday.",
        ),
        evidence(
            "deploy",
            "engineering-deployment-handbook",
            "Routine production releases occur on Tuesdays and Thursdays .",
        ),
    )
    plan = decompose_question(
        "Give the eastern failover destination alongside the ordinary production-release weekdays."
    )
    mapped = extract_direct_support_mappings(plan, rows)
    assert [item.requirement_id for item in mapped] == ["R1", "R2"]
    assert [item.chunk_id for item in mapped] == ["east", "deploy"]
    assert "eu-central" in mapped[0].supporting_spans[0]


def test_equally_grounded_direct_facts_are_treated_as_conflict() -> None:
    rows = (
        evidence("a", "atlas-interface", "The Atlas API is maintained by Team A."),
        evidence("b", "atlas-interface", "The Atlas API is maintained by Team B."),
    )
    plan = decompose_question("Name the group that keeps the Atlas interface operational.")
    assert extract_direct_support_mappings(plan, rows) == ()
