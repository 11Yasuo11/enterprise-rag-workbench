"""Cross-version output cardinality and packet preservation contracts for V11."""

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
    normalized_packet_tokens,
    requirement_packet_terms,
    requirement_scoped_evidence_packets,
)
from rag_workbench.experiments.universal_requirement_completeness_v1 import (
    UniversalEvidenceChunk,
)


def _evidence() -> tuple[GateEvidence, ...]:
    return (
        GateEvidence(
            "allow-25",
            "remote-work-policy",
            "dv-25",
            "2025",
            "Under the 2025 policy , employees could work remotely three days per week .",
        ),
        GateEvidence(
            "allow-26",
            "remote-work-policy",
            "dv-26",
            "2026",
            "Under the current 2026 policy , employees may work remotely two days per week .",
        ),
        GateEvidence(
            "sched-25",
            "remote-work-policy",
            "dv-25",
            "2025",
            "Managers reviewed remote-work schedules every quarter .",
        ),
        GateEvidence(
            "dead-25",
            "security-incident-policy",
            "dv-i25",
            "2025",
            "Under the 2025 policy , suspected severity-one incidents had to be reported "
            "to the security duty officer within 60 minutes of discovery .",
        ),
        GateEvidence(
            "dead-26",
            "security-incident-policy",
            "dv-i26",
            "2026",
            "Under the current 2026 policy , suspected severity-one incidents must be "
            "reported to the security duty officer within 15 minutes of discovery .",
        ),
    )


def test_one_requirement_one_version_one_fact() -> None:
    plan = decompose_question("Under the current policy, report the weekly off-site allowance.")
    rows = (
        GateEvidence(
            "c1",
            "remote-work-policy",
            "dv-26",
            "2026",
            "Under the current 2026 policy , employees may work remotely two days per week .",
        ),
    )
    mappings = extract_direct_support_mappings(plan, rows)
    decision = deterministic_support_complete(
        plan,
        mappings,
        rows,
        authorized_chunk_ids=frozenset({"c1"}),
        selected_versions_by_document={"remote-work-policy": frozenset({"dv-26"})},
    )
    assert decision.complete
    assert len(decision.validation.requirements[0].mappings) == 1


def test_two_requirements_two_versions_preserve_all_facts() -> None:
    plan = decompose_question(
        "Create a two-year change card: remote allowance and incident reporting "
        "deadline for both 2025 and 2026."
    )
    rows = _evidence()
    mappings = extract_direct_support_mappings(plan, rows)
    assert {(m.requirement_id, m.version) for m in mappings} == {
        ("R1", "2025"),
        ("R1", "2026"),
        ("R2", "2025"),
        ("R2", "2026"),
    }
    decision = deterministic_support_complete(
        plan,
        mappings,
        rows,
        authorized_chunk_ids=frozenset(item.chunk_id for item in rows),
        selected_versions_by_document={
            "remote-work-policy": frozenset({"dv-25", "dv-26"}),
            "security-incident-policy": frozenset({"dv-i25", "dv-i26"}),
        },
    )
    assert decision.complete
    assembly = assemble_frozen_plan(
        plan,
        decision.validation,
        tuple(
            UniversalEvidenceChunk(
                item.chunk_id, item.document_id, item.text, item.document_version_id
            )
            for item in rows
        ),
        authorized_chunk_ids=frozenset(item.chunk_id for item in rows),
        selected_versions_by_document={
            "remote-work-policy": frozenset({"dv-25", "dv-26"}),
            "security-incident-policy": frozenset({"dv-i25", "dv-i26"}),
        },
    )
    assert assembly.status == "answered"
    assert "three days per week" in (assembly.answer or "")
    assert "two days per week" in (assembly.answer or "")
    assert "within 60 minutes" in (assembly.answer or "")
    assert "within 15 minutes" in (assembly.answer or "")
    assert len(assembly.citations) == 4


def test_packet_includes_both_year_allowance_chunks() -> None:
    plan = decompose_question(
        "Create a two-year change card: remote allowance and incident reporting "
        "deadline for both 2025 and 2026."
    )
    rows = _evidence()
    packets = requirement_scoped_evidence_packets(
        plan,
        rows,
        authorized_chunk_ids=frozenset(item.chunk_id for item in rows),
        selected_versions_by_document={
            "remote-work-policy": frozenset({"dv-25", "dv-26"}),
            "security-incident-policy": frozenset({"dv-i25", "dv-i26"}),
        },
    )
    r1_ids = {item.chunk_id for item in packets["R1"]}
    assert "allow-25" in r1_ids
    assert "allow-26" in r1_ids


def test_remotely_normalizes_to_remote_and_two_year_does_not_prefer_two() -> None:
    assert "remote" in normalized_packet_tokens("work remotely three days")
    terms = requirement_packet_terms("Create a two-year change card: remote allowance")
    assert "two" not in terms
    assert "remote" in terms
    assert "allowance" in terms


def test_completeness_rejects_omitted_year_scoped_fact() -> None:
    plan = decompose_question(
        "Create a two-year change card: remote allowance and incident reporting "
        "deadline for both 2025 and 2026."
    )
    rows = _evidence()
    result = FrozenVerifierResult(
        decision="GO",
        question_plan_hash=plan.question_plan_hash,
        requirements=[
            FrozenRequirementResult(
                requirement_id="R1",
                status="SUPPORTED",
                chunk_ids=["allow-26", "sched-25"],
                document_ids=["remote-work-policy", "remote-work-policy"],
                document_version_ids=["dv-26", "dv-25"],
                versions=["2026", "2025"],
                supporting_spans=[
                    "Under the current 2026 policy , employees may work remotely "
                    "two days per week .",
                    "Managers reviewed remote-work schedules every quarter .",
                ],
            ),
            FrozenRequirementResult(
                requirement_id="R2",
                status="SUPPORTED",
                chunk_ids=["dead-26", "dead-25"],
                document_ids=["security-incident-policy", "security-incident-policy"],
                document_version_ids=["dv-i26", "dv-i25"],
                versions=["2026", "2025"],
                supporting_spans=[
                    "Under the current 2026 policy , suspected severity-one incidents must be "
                    "reported to the security duty officer within 15 minutes of discovery .",
                    "Under the 2025 policy , suspected severity-one incidents had to be reported "
                    "to the security duty officer within 60 minutes of discovery .",
                ],
            ),
        ],
    )
    validation = validate_verifier_result(
        plan,
        result,
        rows,
        authorized_chunk_ids=frozenset(item.chunk_id for item in rows),
        selected_versions_by_document={
            "remote-work-policy": frozenset({"dv-25", "dv-26"}),
            "security-incident-policy": frozenset({"dv-i25", "dv-i26"}),
        },
    )
    assert validation.valid is False
    assert validation.failure_code == "CROSS_VERSION_OUTPUT_INCOMPLETE"


def test_wrong_version_still_fail_closed() -> None:
    plan = decompose_question(
        "Create a two-year change card: remote allowance and incident reporting "
        "deadline for both 2025 and 2026."
    )
    rows = _evidence()
    result = FrozenVerifierResult(
        decision="GO",
        question_plan_hash=plan.question_plan_hash,
        requirements=[
            FrozenRequirementResult(
                requirement_id="R1",
                status="SUPPORTED",
                chunk_ids=["allow-26"],
                document_ids=["remote-work-policy"],
                document_version_ids=["dv-26"],
                versions=["2026"],
                supporting_spans=[
                    "Under the current 2026 policy , employees may work remotely "
                    "two days per week ."
                ],
            ),
            FrozenRequirementResult(
                requirement_id="R2",
                status="SUPPORTED",
                chunk_ids=["dead-26"],
                document_ids=["security-incident-policy"],
                document_version_ids=["dv-i26"],
                versions=["2026"],
                supporting_spans=[
                    "Under the current 2026 policy , suspected severity-one incidents must be "
                    "reported to the security duty officer within 15 minutes of discovery ."
                ],
            ),
        ],
    )
    validation = validate_verifier_result(
        plan,
        result,
        rows,
        authorized_chunk_ids=frozenset(item.chunk_id for item in rows),
        selected_versions_by_document={
            "remote-work-policy": frozenset({"dv-25", "dv-26"}),
            "security-incident-policy": frozenset({"dv-i25", "dv-i26"}),
        },
    )
    assert validation.failure_code == "CROSS_VERSION_OUTPUT_INCOMPLETE"


def test_unauthorized_chunk_still_rejected() -> None:
    plan = decompose_question("Under the current policy, report the weekly off-site allowance.")
    rows = (
        GateEvidence(
            "c1",
            "remote-work-policy",
            "dv-26",
            "2026",
            "Under the current 2026 policy , employees may work remotely two days per week .",
        ),
    )
    result = FrozenVerifierResult(
        decision="GO",
        question_plan_hash=plan.question_plan_hash,
        requirements=[
            FrozenRequirementResult(
                requirement_id="R1",
                status="SUPPORTED",
                chunk_ids=["c1"],
                document_ids=["remote-work-policy"],
                document_version_ids=["dv-26"],
                versions=["2026"],
                supporting_spans=[
                    "Under the current 2026 policy , employees may work remotely "
                    "two days per week ."
                ],
            )
        ],
    )
    validation = validate_verifier_result(
        plan,
        result,
        rows,
        authorized_chunk_ids=frozenset(),
        selected_versions_by_document={"remote-work-policy": frozenset({"dv-26"})},
    )
    assert validation.failure_code == "UNAUTHORIZED_CHUNK"
