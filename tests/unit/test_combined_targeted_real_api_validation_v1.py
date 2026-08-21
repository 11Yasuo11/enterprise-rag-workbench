from rag_workbench.answerability.base import GateEvidence
from rag_workbench.experiments.combined_targeted_real_api_validation_v1 import (
    deterministic_requirement_map,
    estimate_experiment_budget,
    validate_luna_mapping,
)
from rag_workbench.experiments.safe_recovery_luna_v2.verifier import (
    LunaRequirement,
    LunaVerifierResult,
)
from rag_workbench.experiments.universal_requirement_completeness_v1 import (
    UniversalEvidenceChunk,
)


def test_deterministic_map_preserves_greater_than() -> None:
    chunks = (
        UniversalEvidenceChunk(
            "c1", "finance", "Receipts are required for expenses above 25 euros .", "v1"
        ),
    )
    mapped = deterministic_requirement_map(
        "What expense threshold relation is stated for requiring receipts?", chunks
    )
    assert mapped[0].constraints[0].operator == ">"


def test_deterministic_map_normalizes_date_punctuation() -> None:
    chunks = (
        UniversalEvidenceChunk("c1", "atlas", "Project Atlas launched on April 12 , 2026 .", "v1"),
    )
    mapped = deterministic_requirement_map("What launch date is stated for Project Atlas?", chunks)
    assert mapped[0].normalized_value == "april 12, 2026"


def test_luna_mapping_rejects_nonliteral_span() -> None:
    evidence = (GateEvidence("c1", "d1", "v1", "1", "Owner is Alice.", "idx"),)
    result = LunaVerifierResult(
        decision="GO",
        requirements=[
            LunaRequirement(
                requirement="owner", supported=True, chunk_id="c1", supporting_span="Owner is Bob."
            )
        ],
        all_supported=True,
        conflict=False,
        version_valid=True,
        region_valid=True,
    )
    _, failure = validate_luna_mapping(result, evidence, expected_requirement_count=1)
    assert failure == "LUNA_SPAN_NOT_LITERAL"


def test_budget_includes_four_sol_call_reserve() -> None:
    rows = [
        {
            "whether_luna_expected": True,
            "estimated_tokens": {"input": 2500, "output": 220},
        }
    ] * 6
    budget = estimate_experiment_budget(rows)
    assert budget["reserved_sol_calls"] == 4
    assert budget["projected_total_cost_usd"] < 0.25


def test_minute_duration_is_preserved() -> None:
    chunks = (
        UniversalEvidenceChunk(
            "c1",
            "incident",
            "Suspected severity-one incidents must be reported within 15 minutes.",
            "v1",
        ),
    )
    mapped = deterministic_requirement_map(
        "How quickly must a severity-one incident reach the duty officer?", chunks
    )
    assert mapped[0].constraints[0].fingerprint == (
        "duration",
        "within",
        "15",
        "minutes",
        None,
    )
