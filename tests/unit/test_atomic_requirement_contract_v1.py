from __future__ import annotations

import inspect

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.experiments.atomic_requirement_contract_v1 import (
    assemble_frozen_plan,
    decompose_question,
    validate_verifier_result,
)
from rag_workbench.experiments.atomic_requirement_contract_v1.contract import (
    FrozenValidation,
    ValidatedMapping,
    ValidatedRequirement,
    semantic_cardinality_audit,
)
from rag_workbench.experiments.atomic_requirement_contract_v1.verifier import (
    FrozenRequirementResult,
    FrozenVerifierResult,
    frozen_schema,
    requirement_scoped_evidence_packets,
)
from rag_workbench.experiments.universal_requirement_completeness_v1 import UniversalEvidenceChunk


def texts(question: str) -> list[str]:
    return [item.requirement_text for item in decompose_question(question).requirements]


def test_simple_one_fact_question() -> None:
    assert texts("Who owns the Atlas API?") == ["Who owns the Atlas API"]


def test_two_explicit_requested_facts() -> None:
    assert texts("Provide remote-work policy and approval deadline.") == [
        "remote-work policy",
        "approval deadline",
    ]


def test_three_explicit_requested_facts() -> None:
    assert texts("Provide owner, deadline, and review interval.") == [
        "owner",
        "deadline",
        "review interval",
    ]


def test_audit_prefix_plus_two_requested_facts() -> None:
    plan = decompose_question("For audit purposes, provide allowance and review interval.")
    assert len(plan.requirements) == 2
    assert [(q.qualifier_type, q.text) for q in plan.context_qualifiers] == [
        ("audit", "For audit purposes")
    ]


def test_region_qualifier_plus_requested_fact() -> None:
    plan = decompose_question("For the EU region, what is the retention period?")
    assert len(plan.requirements) == 1
    assert plan.context_qualifiers[0].qualifier_type == "region"


def test_version_qualifier_plus_requested_fact() -> None:
    plan = decompose_question("Under the current policy, what is the allowance?")
    assert len(plan.requirements) == 1
    assert plan.context_qualifiers[0].qualifier_type == "version"


def test_date_qualifier_plus_requested_facts() -> None:
    plan = decompose_question("As of April 2026, provide allowance and review interval.")
    assert len(plan.requirements) == 2
    assert plan.context_qualifiers[0].qualifier_type == "time"


def test_discourse_prefix() -> None:
    plan = decompose_question("In plain language, who maintains the Atlas API?")
    assert texts(plan.question) == ["who maintains the Atlas API"]
    assert plan.context_qualifiers[0].qualifier_type == "other"


def test_comma_separated_clauses_and_conjunctions() -> None:
    assert len(decompose_question("Give owner, approval ID, and deadline.").requirements) == 3
    assert len(decompose_question("Give owner and deadline.").requirements) == 2


def test_qualifier_not_promoted_into_output_requirement() -> None:
    plan = decompose_question(
        "Use the currently enforceable text, not its superseded edition: "
        "give allowance and review interval."
    )
    assert texts(plan.question) == ["allowance", "review interval"]
    assert len(plan.context_qualifiers) == 1


def test_explicit_requested_qualifier_related_fact_is_preserved() -> None:
    assert texts("What region applies?") == ["What region applies"]


def test_stable_ids_and_hash() -> None:
    first = decompose_question("Provide owner, deadline, and interval.")
    second = decompose_question("Provide owner, deadline, and interval.")
    assert [x.requirement_id for x in first.requirements] == ["R1", "R2", "R3"]
    assert first.question_plan_hash == second.question_plan_hash


def test_dynamic_schema_prohibits_r4_for_luna_and_sol() -> None:
    plan = decompose_question("Give owner and deadline.")
    requirement_id_schema = frozen_schema(plan)["properties"]["requirements"]["items"][
        "properties"
    ]["requirement_id"]
    assert requirement_id_schema["enum"] == ["R1", "R2"]
    assert "R4" not in requirement_id_schema["enum"]
    # Both Luna and Sol use this same plan-derived verifier schema.


def test_validator_rejects_invented_id_without_redecomposing() -> None:
    plan = decompose_question("Who owns Atlas?")
    result = FrozenVerifierResult(
        decision="GO",
        question_plan_hash=plan.question_plan_hash,
        requirements=[
            FrozenRequirementResult(
                requirement_id="R4",
                status="SUPPORTED",
                chunk_ids=["c1"],
                document_ids=["d1"],
                supporting_spans=["Alice owns Atlas."],
            )
        ],
    )
    validation = validate_verifier_result(
        plan,
        result,
        (GateEvidence("c1", "d1", "v1", "1", "Alice owns Atlas."),),
        authorized_chunk_ids=frozenset({"c1"}),
    )
    assert validation.failure_code == "REQUIREMENT_ID_MISMATCH"
    assert "decompose_question" not in inspect.getsource(validate_verifier_result)


def test_assembler_uses_same_plan_hash_and_zero_model_calls() -> None:
    plan = decompose_question("Who owns Atlas?")
    validation = FrozenValidation(True, None, "wrong-hash", ())
    failed = assemble_frozen_plan(
        plan,
        validation,
        (UniversalEvidenceChunk("c1", "d1", "Alice owns Atlas.", "v1"),),
        authorized_chunk_ids=frozenset({"c1"}),
    )
    assert failed.failure_code == "QUESTION_PLAN_HASH_MISMATCH"
    assert failed.final_generator_openai_calls == 0


def test_no_expected_answer_leakage_or_historical_case_routing() -> None:
    signature = inspect.signature(decompose_question)
    assert list(signature.parameters) == ["question"]
    synthetic = "Audit marker never-seen-xyz. Provide owner and deadline."
    assert [r.requirement_id for r in decompose_question(synthetic).requirements] == ["R1", "R2"]
    assert "judge_near_05" not in inspect.getsource(decompose_question)
    assert "p5k_semantic_paraphrase_054" not in inspect.getsource(decompose_question)


def test_semantic_multi_output_predicates_are_atomic() -> None:
    cases = {
        "Who owns X and who maintains Y?": 2,
        "Give X, Y, and Z.": 3,
        "How many X and how often Y?": 2,
        "State the code and the weekday.": 2,
        "Pair X with Y.": 2,
        "Combine X with Y.": 2,
    }
    for question, expected in cases.items():
        plan = decompose_question(question)
        assert len(plan.requirements) == expected
        assert len(plan.detected_output_units) == expected
        assert plan.cardinality_match


def test_comparison_outputs_retain_temporal_qualifiers() -> None:
    plan = decompose_question("Compare the 2025 value and the 2026 value.")
    assert texts(plan.question) == ["the 2025 value", "the 2026 value"]
    assert [(item.qualifier_type, item.text) for item in plan.context_qualifiers] == [
        ("time", "2025"),
        ("time", "2026"),
    ]


def test_negative_cases_do_not_over_split() -> None:
    cases = {
        "The Atlas API is maintained by which team?": (1, 0),
        "What is the recovery time objective for the customer API?": (1, 0),
        "Under the current 2026 policy, how many remote days are allowed?": (1, 1),
        "During an outage, who coordinates Engineering and Customer Support?": (1, 1),
    }
    for question, (requirements, qualifiers) in cases.items():
        plan = decompose_question(question)
        assert len(plan.requirements) == requirements
        assert len(plan.context_qualifiers) == qualifiers


def test_explicit_temporal_fact_is_not_demoted_to_qualifier() -> None:
    plan = decompose_question(
        "What policy year applies and how many remote days are allowed?"
    )
    assert len(plan.requirements) == 2
    assert len(plan.context_qualifiers) == 0


def test_version_validation_is_scoped_per_document() -> None:
    plan = decompose_question("Who owns Atlas?")
    evidence = (
        GateEvidence("c1", "doc-a", "doc-a-v2", "2", "Alice owns Atlas."),
        GateEvidence("c2", "doc-b", "doc-b-v7", "7", "Bob maintains Atlas."),
    )
    result = FrozenVerifierResult(
        decision="GO",
        question_plan_hash=plan.question_plan_hash,
        requirements=[
            FrozenRequirementResult(
                requirement_id="R1",
                status="SUPPORTED",
                chunk_ids=["c1", "c2"],
                document_ids=["doc-a", "doc-b"],
                document_version_ids=["doc-a-v2", "doc-b-v7"],
                versions=["2", "7"],
                supporting_spans=["Alice owns Atlas.", "Bob maintains Atlas."],
            )
        ],
    )
    validation = validate_verifier_result(
        plan,
        result,
        evidence,
        authorized_chunk_ids=frozenset({"c1", "c2"}),
        selected_versions_by_document={
            "doc-a": frozenset({"doc-a-v2"}),
            "doc-b": frozenset({"doc-b-v7"}),
        },
    )
    assert validation.valid


def test_version_validation_rejects_wrong_version_for_its_document() -> None:
    plan = decompose_question("Who owns Atlas?")
    evidence = (GateEvidence("c1", "doc-a", "doc-a-v1", "1", "Alice owns Atlas."),)
    result = FrozenVerifierResult(
        decision="GO",
        question_plan_hash=plan.question_plan_hash,
        requirements=[
            FrozenRequirementResult(
                requirement_id="R1",
                status="SUPPORTED",
                chunk_ids=["c1"],
                document_ids=["doc-a"],
                document_version_ids=["doc-a-v1"],
                versions=["1"],
                supporting_spans=["Alice owns Atlas."],
            )
        ],
    )
    validation = validate_verifier_result(
        plan,
        result,
        evidence,
        authorized_chunk_ids=frozenset({"c1"}),
        selected_versions_by_document={"doc-a": frozenset({"doc-a-v2"})},
    )
    assert validation.failure_code == "WRONG_VERSION"


def test_all_supported_statuses_canonicalize_raw_abstain_to_go() -> None:
    plan = decompose_question("Who owns Atlas?")
    result = FrozenVerifierResult(
        decision="ABSTAIN",
        question_plan_hash=plan.question_plan_hash,
        requirements=[
            FrozenRequirementResult(
                requirement_id="R1",
                status="SUPPORTED",
                chunk_ids=["c1"],
                document_ids=["doc-a"],
                document_version_ids=["doc-a-v1"],
                versions=["1"],
                supporting_spans=["Alice owns Atlas."],
            )
        ],
    )
    validation = validate_verifier_result(
        plan,
        result,
        (GateEvidence("c1", "doc-a", "doc-a-v1", "1", "Alice owns Atlas."),),
        authorized_chunk_ids=frozenset({"c1"}),
    )
    assert validation.valid
    assert validation.raw_decision == "ABSTAIN"
    assert validation.canonical_decision == "GO"


def test_requirement_scoped_packets_preserve_candidates_per_requirement() -> None:
    plan = decompose_question(
        "Give the east and west recovery codes and the maximum customer API recovery window."
    )
    evidence = (
        GateEvidence("east", "east-doc", "east-v1", "1", "The east recovery code is E-17."),
        GateEvidence("west", "west-doc", "west-v1", "1", "The west recovery code is W-29."),
        GateEvidence(
            "rto", "plan", "plan-v1", "1", "The customer API recovery objective is four hours."
        ),
    )
    packets = requirement_scoped_evidence_packets(plan, evidence)
    assert set(packets) == {"R1", "R2", "R3"}
    assert all(packets.values())
    assert packets["R3"][0].chunk_id == "rto"


def test_general_leading_context_is_qualifier_not_output() -> None:
    cases = (
        "After a priority-one outage queue item is opened, which team is paged?",
        "When a severity-one incident occurs, what is the deadline?",
        "In the event of a failover, which cluster is used?",
        "As part of launch review, who owns the API?",
        "According to the current policy, how many days are allowed?",
        "For Atlas operations, what is the API version?",
        "As an HR leadership user, what is the exception identifier?",
        "Using only the presently enforceable remote-work revision, report the weekly allowance.",
        "Read the current remote-work text and state its manager review interval.",
    )
    for question in cases:
        plan = decompose_question(question)
        assert len(plan.requirements) == 1
        assert len(plan.context_qualifiers) == 1


def test_alongside_and_similar_connectors_split_independent_outputs() -> None:
    assert texts(
        "Give the eastern failover destination alongside the ordinary production-release weekdays."
    ) == [
        "the eastern failover destination",
        "the ordinary production-release weekdays",
    ]
    assert texts("Give the training reference together with the receipt rule.") == [
        "the training reference",
        "the receipt rule",
    ]
    assert texts("State the east code as well as the west code.") == [
        "the east code",
        "the west code",
    ]


def test_evidence_access_preamble_is_version_qualifier() -> None:
    plan = decompose_question(
        "Consult the archived edition and identify the receipt threshold."
    )
    assert texts(plan.question) == ["the receipt threshold"]
    assert plan.context_qualifiers[0].qualifier_type == "version"
    assert "archived edition" in plan.context_qualifiers[0].text.casefold()


def test_coordinated_shared_head_outputs_are_expanded() -> None:
    assert texts(
        "Give the east and west recovery codes and the maximum customer API recovery window."
    ) == [
        "the east recovery codes",
        "west recovery codes",
        "the maximum customer API recovery window",
    ]


def test_comma_list_preserves_conjunction_inside_one_output() -> None:
    plan = decompose_question(
        "For a severity-one outage, state the reporting deadline, "
        "the role coordinating Engineering and Customer Support, and "
        "how often customers receive status updates until mitigation."
    )
    assert texts(plan.question) == [
        "the reporting deadline",
        "the role coordinating Engineering and Customer Support",
        "how often customers receive status updates until mitigation",
    ]
    assert len(plan.context_qualifiers) == 1


def test_cardinality_audit_is_persistable_and_matches_plan() -> None:
    plan = decompose_question("Who owns X and who maintains Y?")
    audit = semantic_cardinality_audit(plan)
    assert audit["detected_output_units"] == ["Who owns X", "who maintains Y"]
    assert audit["requested_output_count"] == 2
    assert audit["atomic_requirement_count"] == 2
    assert audit["cardinality_match"] is True


def test_validator_and_assembler_preserve_every_mapping_and_citation() -> None:
    plan = decompose_question("Who owns X?")
    validation = FrozenValidation(
        True,
        None,
        plan.question_plan_hash,
        (
            ValidatedRequirement(
                "R1",
                plan.requirements[0].requirement_text,
                "c1",
                "d1",
                "Primary owner is Product Operations.",
                "v1",
                (
                    ValidatedMapping(
                        "c2", "d2", "The ownership register confirms Product Operations.", "v2"
                    ),
                ),
            ),
        ),
    )
    result = assemble_frozen_plan(
        plan,
        validation,
        (
            UniversalEvidenceChunk(
                "c1", "d1", "Primary owner is Product Operations.", "v1"
            ),
            UniversalEvidenceChunk(
                "c2", "d2", "The ownership register confirms Product Operations.", "v2"
            ),
        ),
        authorized_chunk_ids=frozenset({"c1", "c2"}),
    )
    assert result.status == "answered"
    assert result.citations == ("c1", "c2")
    assert result.validated_mapping_keys == result.assembled_mapping_keys
    assert "Primary owner is Product Operations." in (result.answer or "")
    assert "The ownership register confirms Product Operations." in (result.answer or "")


def test_assembly_fails_closed_on_semantic_cardinality_tamper() -> None:
    plan = decompose_question("Who owns X and who maintains Y?")
    tampered = type(plan)(
        plan.question,
        plan.requirements[:1],
        plan.context_qualifiers,
        plan.question_plan_hash,
        plan.detected_output_units,
        False,
    )
    result = assemble_frozen_plan(
        tampered,
        FrozenValidation(True, None, tampered.question_plan_hash, ()),
        (),
        authorized_chunk_ids=frozenset(),
    )
    assert result.status == "abstained"
    assert result.failure_code == "SEMANTIC_CARDINALITY_MISMATCH"
