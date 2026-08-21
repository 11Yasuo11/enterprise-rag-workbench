"""Label-blind runtime helpers for COMBINED_TARGETED_REAL_API_VALIDATION_V1."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.experiments.adaptive_top20_abstention_recovery_v1.requirements import (
    decompose_requirements,
)
from rag_workbench.experiments.safe_recovery_luna_v2.pricing import estimate_cost_usd
from rag_workbench.experiments.safe_recovery_luna_v2.verifier import LunaVerifierResult
from rag_workbench.experiments.universal_requirement_completeness_v1 import (
    UniversalEvidenceChunk,
    UniversalRequirement,
    extract_constraints,
    map_launch_date_requirement,
)


@dataclass(frozen=True)
class TargetCase:
    query_id: str
    group: str
    category: str
    question: str
    principal_id: str
    tenant_id: str
    permission_groups: tuple[str, ...]


def atomic_requirements(question: str) -> tuple[str, ...]:
    """Drop scope clauses that constrain evidence but are not requested answer fields."""
    requirements = decompose_requirements(question)
    if len(requirements) > 1 and requirements[0].casefold().startswith(("when ", "ignoring ")):
        requirements = requirements[1:]
    return requirements


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip())


def _unique_sentence(
    chunks: tuple[UniversalEvidenceChunk, ...], required_terms: tuple[str, ...]
) -> tuple[UniversalEvidenceChunk, str] | None:
    hits: list[tuple[UniversalEvidenceChunk, str]] = []
    for chunk in chunks:
        for sentence in _sentences(chunk.text):
            lower = sentence.casefold()
            if all(term in lower for term in required_terms):
                hits.append((chunk, sentence))
    return hits[0] if len(hits) == 1 else None


def _requirement(
    hit: tuple[UniversalEvidenceChunk, str], description: str
) -> tuple[UniversalRequirement, ...]:
    chunk, span = hit
    constraints = extract_constraints(span)
    return (
        UniversalRequirement(
            "R1",
            description,
            "constraint" if constraints else "fact",
            chunk.chunk_id,
            chunk.document_id,
            span,
            None,
            constraints,
            chunk.version_id,
        ),
    )


def deterministic_requirement_map(
    question: str, chunks: tuple[UniversalEvidenceChunk, ...]
) -> tuple[UniversalRequirement, ...]:
    """Map only narrow observable structures; never consumes benchmark labels or IDs."""
    launch = map_launch_date_requirement(question, chunks)
    if launch:
        return launch
    lower = question.casefold()
    rules: list[tuple[tuple[str, ...], tuple[str, ...], str]] = [
        (("receipt", "threshold"), ("receipts", "required", "above"), "receipt threshold"),
        (
            ("severity-one", "duty officer"),
            ("severity-one", "reported", "within"),
            "reporting deadline",
        ),
        (
            ("deployment", "approval", "identifier"),
            ("deployment", "approval", "identifier"),
            "deployment approval identifier",
        ),
    ]
    for question_terms, evidence_terms, description in rules:
        if all(term in lower for term in question_terms):
            hit = _unique_sentence(chunks, evidence_terms)
            if hit:
                return _requirement(hit, description)
    return ()


def validate_luna_mapping(
    result: LunaVerifierResult,
    evidence: tuple[GateEvidence, ...],
    *,
    expected_requirement_count: int,
) -> tuple[tuple[UniversalRequirement, ...], str | None]:
    """Turn verifier output into literal, authorized requirements or fail closed."""
    if result.decision != "GO" or not result.all_supported or result.conflict:
        return (), "LUNA_NOT_GO"
    if not result.version_valid or not result.region_valid:
        return (), "LUNA_SCOPE_INVALID"
    if len(result.requirements) != expected_requirement_count:
        return (), "LUNA_REQUIREMENT_COUNT_MISMATCH"
    by_id = {item.chunk_id: item for item in evidence}
    mapped: list[UniversalRequirement] = []
    for index, item in enumerate(result.requirements, 1):
        if not item.supported or not item.chunk_id or not item.supporting_span:
            return (), "LUNA_REQUIREMENT_UNSUPPORTED"
        chunk = by_id.get(item.chunk_id)
        if chunk is None:
            return (), "LUNA_UNKNOWN_CHUNK"
        if item.supporting_span not in chunk.text:
            return (), "LUNA_SPAN_NOT_LITERAL"
        constraints = extract_constraints(item.supporting_span)
        mapped.append(
            UniversalRequirement(
                f"R{index}",
                item.requirement,
                "constraint" if constraints else "fact",
                chunk.chunk_id,
                chunk.document_id,
                item.supporting_span,
                None,
                constraints,
                chunk.document_version_id or None,
            )
        )
    return tuple(mapped), None


def estimate_experiment_budget(
    dry_rows: list[dict[str, Any]], *, sol_reserve_calls: int = 4
) -> dict[str, Any]:
    luna_input = sum(
        int(row["estimated_tokens"]["input"]) for row in dry_rows if row["whether_luna_expected"]
    )
    luna_output = sum(
        int(row["estimated_tokens"]["output"]) for row in dry_rows if row["whether_luna_expected"]
    )
    luna_calls = sum(bool(row["whether_luna_expected"]) for row in dry_rows)
    luna_cost = estimate_cost_usd(
        model="gpt-5.6-luna", input_tokens=luna_input, output_tokens=luna_output
    )
    # Conservative reserve uses 2,500 input and the frozen 700-token structured-output cap.
    sol_cost = estimate_cost_usd(
        model="gpt-5.6-sol",
        input_tokens=sol_reserve_calls * 2500,
        output_tokens=sol_reserve_calls * 700,
    )
    return {
        "expected_luna_calls": luna_calls,
        "reserved_sol_calls": sol_reserve_calls,
        "projected_luna_cost_usd": luna_cost,
        "projected_sol_reserve_cost_usd": sol_cost,
        "projected_total_cost_usd": luna_cost + sol_cost,
        "budget_cap_usd": 0.25,
        "passes": luna_cost + sol_cost <= 0.25,
    }
