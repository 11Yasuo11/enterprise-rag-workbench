"""Universal mapping and fail-closed completeness assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_workbench.evaluation.final_e2e_scorer_v2 import normalize_fact
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.assembler import (
    DeterministicRequirementAssemblerV3,
    EvidenceChunk,
    VerifiedRequirement,
)

from .constraints import Constraint, DeterministicConstraintValidator, extract_constraints


@dataclass(frozen=True)
class UniversalEvidenceChunk:
    chunk_id: str
    document_id: str
    text: str
    version_id: str | None = None


@dataclass(frozen=True)
class UniversalRequirement:
    requirement_id: str
    requirement: str
    requirement_type: str
    chunk_id: str
    document_id: str
    supporting_span: str
    normalized_value: str | None = None
    constraints: tuple[Constraint, ...] = ()
    expected_version_id: str | None = None


@dataclass(frozen=True)
class UniversalAssemblyResult:
    status: str
    answer: str | None
    citations: tuple[str, ...]
    failure_code: str | None
    required_requirement_count: int
    verified_requirement_count: int
    output_requirement_count: int
    required_constraint_count: int
    validated_output_constraint_count: int
    final_generator_openai_calls: int = 0


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip())


def map_launch_date_requirement(
    question: str,
    chunks: tuple[UniversalEvidenceChunk, ...],
) -> tuple[UniversalRequirement, ...]:
    """Map an observable launch-date question to one uniquely supported literal span."""
    request = question.rsplit(". ", 1)[-1]
    match = re.fullmatch(r"what launch date is stated for (?P<subject>.+?)\?", request, re.I)
    if not match:
        return ()
    subject_tokens = tuple(match.group("subject").casefold().split())
    candidates: list[tuple[UniversalEvidenceChunk, str, Constraint]] = []
    for chunk in chunks:
        for sentence in _sentences(chunk.text):
            lower = sentence.casefold()
            dates = tuple(
                item for item in extract_constraints(sentence) if item.constraint_type == "date"
            )
            if (
                all(token in lower for token in subject_tokens)
                and "launch" in lower
                and len(dates) == 1
            ):
                candidates.append((chunk, sentence, dates[0]))
    if len(candidates) != 1:
        return ()
    chunk, sentence, constraint = candidates[0]
    return (
        UniversalRequirement(
            "R1",
            f"launch date for {match.group('subject')}",
            "date_constraint",
            chunk.chunk_id,
            chunk.document_id,
            sentence,
            normalize_fact(constraint.value),
            (constraint,),
            chunk.version_id,
        ),
    )


class UniversalRequirementAssembler:
    """Canonical deterministic path for any verified requirement mapping."""

    model_name = "deterministic-requirement-assembler-v3+universal-completeness-v1"

    def __init__(self) -> None:
        self._assembler = DeterministicRequirementAssemblerV3()
        self._validator = DeterministicConstraintValidator()

    def assemble(
        self,
        requirements: tuple[UniversalRequirement, ...],
        chunks: tuple[UniversalEvidenceChunk, ...],
        *,
        authorized_chunk_ids: frozenset[str],
        selected_version_ids: frozenset[str] | None = None,
    ) -> UniversalAssemblyResult:
        required_count = len(requirements)
        if not requirements:
            return self._reject("MISSING_REQUIREMENTS", required_count)
        expected_ids = tuple(f"R{index}" for index in range(1, required_count + 1))
        if tuple(item.requirement_id for item in requirements) != expected_ids:
            return self._reject("INVALID_REQUIREMENT_IDS", required_count)
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        verified: list[VerifiedRequirement] = []
        for requirement in requirements:
            chunk = by_id.get(requirement.chunk_id)
            if chunk is None:
                return self._reject("INVALID_CHUNK", required_count)
            if requirement.chunk_id not in authorized_chunk_ids:
                return self._reject("UNAUTHORIZED_CHUNK", required_count)
            if chunk.document_id != requirement.document_id:
                return self._reject("DOCUMENT_MISMATCH", required_count)
            if requirement.supporting_span not in chunk.text:
                return self._reject("SPAN_NOT_LITERAL", required_count)
            if (
                requirement.expected_version_id
                and chunk.version_id != requirement.expected_version_id
            ):
                return self._reject("WRONG_VERSION", required_count)
            if selected_version_ids is not None and chunk.version_id not in selected_version_ids:
                return self._reject("WRONG_VERSION", required_count)
            if requirement.normalized_value and requirement.normalized_value not in normalize_fact(
                requirement.supporting_span
            ):
                return self._reject("NORMALIZED_VALUE_NOT_SUPPORTED", required_count)
            if tuple(extract_constraints(requirement.supporting_span)) != requirement.constraints:
                return self._reject("EVIDENCE_CONSTRAINT_MISMATCH", required_count)
            verified.append(
                VerifiedRequirement(
                    requirement.requirement_id,
                    requirement.requirement,
                    requirement.chunk_id,
                    requirement.document_id,
                    (requirement.supporting_span,),
                )
            )
        base = self._assembler.assemble(
            tuple(verified),
            tuple(EvidenceChunk(item.chunk_id, item.document_id, item.text) for item in chunks),
            authorized_chunk_ids=authorized_chunk_ids,
        )
        if base.status != "answered" or not base.answer:
            return self._reject(base.failure_code or "ASSEMBLY_REJECTED", required_count)
        if len(base.requirements) != required_count:
            return self._reject("REQUIREMENT_COUNT_MISMATCH", required_count)
        validated_constraints = 0
        for requirement, output in zip(requirements, base.requirements, strict=True):
            line = next(
                (
                    line
                    for line in base.answer.splitlines()
                    if line.startswith(f"{requirement.requirement_id}:")
                ),
                "",
            )
            validation = self._validator.validate(
                requirement.constraints, requirement.supporting_span, line
            )
            if not validation.passed:
                return self._reject(
                    validation.failure_code or "CONSTRAINT_MISMATCH", required_count
                )
            validated_constraints += validation.output_count
            if f"[{output.citation_label}]" not in line:
                return self._reject("OUTPUT_CITATION_MISSING", required_count)
        required_constraints = sum(len(item.constraints) for item in requirements)
        if validated_constraints != required_constraints:
            return self._reject("CONSTRAINT_COUNT_MISMATCH", required_count)
        return UniversalAssemblyResult(
            "answered",
            base.answer,
            base.citations,
            None,
            required_count,
            len(verified),
            len(base.requirements),
            required_constraints,
            validated_constraints,
            0,
        )

    @staticmethod
    def _reject(code: str, required_count: int) -> UniversalAssemblyResult:
        return UniversalAssemblyResult("abstained", None, (), code, required_count, 0, 0, 0, 0, 0)
