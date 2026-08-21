from rag_workbench.evaluation.final_e2e_scorer_v2 import fact_in_text
from rag_workbench.experiments.universal_requirement_completeness_v1 import (
    DeterministicConstraintValidator,
    UniversalEvidenceChunk,
    UniversalRequirement,
    UniversalRequirementAssembler,
    extract_constraints,
    map_launch_date_requirement,
)


def chunk(
    chunk_id: str = "c1",
    document_id: str = "d1",
    text: str = "Receipts are required for expenses above €25.",
    version_id: str | None = "v1",
) -> UniversalEvidenceChunk:
    return UniversalEvidenceChunk(chunk_id, document_id, text, version_id)


def requirement(
    index: int,
    evidence: UniversalEvidenceChunk,
    *,
    span: str | None = None,
    expected_version_id: str | None = None,
) -> UniversalRequirement:
    literal = span or evidence.text
    return UniversalRequirement(
        f"R{index}",
        f"requirement {index}",
        "constraint" if extract_constraints(literal) else "fact",
        evidence.chunk_id,
        evidence.document_id,
        literal,
        None,
        extract_constraints(literal),
        expected_version_id,
    )


def assemble(
    requirements: tuple[UniversalRequirement, ...],
    chunks: tuple[UniversalEvidenceChunk, ...],
    *,
    authorized: frozenset[str] | None = None,
    selected_versions: frozenset[str] | None = None,
):
    result = UniversalRequirementAssembler().assemble(
        requirements,
        chunks,
        authorized_chunk_ids=authorized or frozenset(item.chunk_id for item in chunks),
        selected_version_ids=selected_versions,
    )
    assert result.final_generator_openai_calls == 0
    return result


def test_all_required_facts_are_preserved_across_documents() -> None:
    chunks = (chunk("c1", "d1", "Owner is Alice."), chunk("c2", "d2", "Deadline is Friday."))
    result = assemble(tuple(requirement(i, item) for i, item in enumerate(chunks, 1)), chunks)
    assert result.status == "answered"
    assert result.required_requirement_count == result.verified_requirement_count == 2
    assert result.output_requirement_count == 2
    assert "Owner is Alice." in result.answer and "Deadline is Friday." in result.answer


def test_missing_requirement_rejects_answer() -> None:
    chunks = (chunk("c1", text="A."), chunk("c2", text="B."))
    result = assemble((requirement(1, chunks[0]), requirement(3, chunks[1])), chunks)
    assert result.failure_code == "INVALID_REQUIREMENT_IDS"


def test_greater_than_preserved_exactly() -> None:
    constraints = extract_constraints("Receipts are required above €25.")
    assert constraints[0].operator == ">"
    assert (
        DeterministicConstraintValidator()
        .validate(
            constraints,
            "Receipts are required above €25.",
            "R1: Receipts are required above €25. [C1]",
        )
        .passed
    )


def test_greater_than_or_equal_preserved_exactly() -> None:
    constraints = extract_constraints("Receipts are required for >= €25.")
    assert constraints[0].operator == ">="
    assert (
        DeterministicConstraintValidator()
        .validate(
            constraints,
            "Receipts are required for >= €25.",
            "R1: Receipts are required for >= €25. [C1]",
        )
        .passed
    )


def test_greater_than_is_not_greater_than_or_equal() -> None:
    expected = extract_constraints("Value > 25")
    result = DeterministicConstraintValidator().validate(expected, "Value > 25", "R1: Value >= 25")
    assert not result.passed


def test_before_is_not_on_or_before() -> None:
    expected = extract_constraints("Submit before April 12, 2026.")
    result = DeterministicConstraintValidator().validate(
        expected, "Submit before April 12, 2026.", "R1: Submit on or before April 12, 2026."
    )
    assert expected[0].operator == "before"
    assert not result.passed


def test_after_is_not_on_or_after() -> None:
    expected = extract_constraints("Submit after April 12, 2026.")
    result = DeterministicConstraintValidator().validate(
        expected, "Submit after April 12, 2026.", "R1: Submit on or after April 12, 2026."
    )
    assert expected[0].operator == "after"
    assert not result.passed


def test_deadline_is_preserved() -> None:
    evidence = chunk(text="The deadline is on or before April 12, 2026.")
    result = assemble((requirement(1, evidence),), (evidence,))
    assert result.status == "answered"
    assert "on or before April 12, 2026" in result.answer
    assert result.required_constraint_count == result.validated_output_constraint_count == 1


def test_duration_is_preserved() -> None:
    evidence = chunk(text="Submit within 10 days.")
    result = assemble((requirement(1, evidence),), (evidence,))
    assert result.status == "answered"
    assert extract_constraints(result.answer)[0].operator == "within"


def test_multi_requirement_single_document() -> None:
    evidence = chunk(text="Owner is Alice. Deadline is Friday.")
    requirements = (
        requirement(1, evidence, span="Owner is Alice."),
        requirement(2, evidence, span="Deadline is Friday."),
    )
    result = assemble(requirements, (evidence,))
    assert result.output_requirement_count == 2
    assert result.citations == ("c1",)
    assert result.answer.count("[C1]") == 2


def test_duplicate_supporting_spans_are_not_collapsed() -> None:
    evidence = chunk(text="Manager approval is required.")
    result = assemble((requirement(1, evidence), requirement(2, evidence)), (evidence,))
    assert result.output_requirement_count == 2
    assert result.answer.count("Manager approval is required.") == 2


def test_invalid_supporting_span_rejected() -> None:
    evidence = chunk(text="Owner is Alice.")
    result = assemble((requirement(1, evidence, span="Owner is Bob."),), (evidence,))
    assert result.failure_code == "SPAN_NOT_LITERAL"


def test_unauthorized_chunk_rejected() -> None:
    evidence = chunk(text="Owner is Alice.")
    result = assemble((requirement(1, evidence),), (evidence,), authorized=frozenset({"other"}))
    assert result.failure_code == "UNAUTHORIZED_CHUNK"


def test_wrong_version_rejected() -> None:
    evidence = chunk(text="The active value is 7.", version_id="v2")
    result = assemble(
        (requirement(1, evidence, expected_version_id="v1"),),
        (evidence,),
        selected_versions=frozenset({"v1"}),
    )
    assert result.failure_code == "WRONG_VERSION"


def test_launch_date_mapping_is_runtime_observable_and_literal() -> None:
    evidence = chunk(
        "atlas",
        "project-atlas-launch",
        "Project Atlas launched on April 12 , 2026 . The owner is Product Operations .",
    )
    mapped = map_launch_date_requirement(
        "What launch date is stated for Project Atlas?", (evidence,)
    )
    assert len(mapped) == 1
    assert mapped[0].supporting_span == "Project Atlas launched on April 12 , 2026 ."
    assert mapped[0].normalized_value == "april 12, 2026"


def test_scorer_normalizes_tokenized_date_punctuation() -> None:
    assert fact_in_text("April 12, 2026", "Project Atlas launched on April 12 , 2026 .")


def test_scorer_does_not_collapse_numeric_operators() -> None:
    assert not fact_in_text("> 25", ">= 25")
