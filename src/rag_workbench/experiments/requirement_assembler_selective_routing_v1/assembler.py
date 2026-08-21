"""Requirement-centric deterministic answer assembly with fail-closed validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    document_id: str
    text: str


@dataclass(frozen=True)
class VerifiedRequirement:
    requirement_id: str
    requirement: str
    chunk_id: str
    document_id: str
    supporting_spans: tuple[str, ...]


@dataclass(frozen=True)
class AssembledRequirement:
    requirement_id: str
    chunk_id: str
    document_id: str
    supporting_spans: tuple[str, ...]
    citation_label: str


@dataclass(frozen=True)
class AssemblyResult:
    status: str
    answer: str | None
    citations: tuple[str, ...]
    requirements: tuple[AssembledRequirement, ...]
    failure_code: str | None
    final_generator_openai_calls: int = 0


class DeterministicRequirementAssemblerV3:
    model_name = "deterministic-requirement-assembler-v3"
    provider_name = "deterministic"

    def assemble(
        self,
        requirements: tuple[VerifiedRequirement, ...],
        chunks: tuple[EvidenceChunk, ...],
        *,
        authorized_chunk_ids: frozenset[str],
    ) -> AssemblyResult:
        if not requirements:
            return self._reject("MISSING_REQUIREMENTS")
        expected_ids = tuple(f"R{i}" for i in range(1, len(requirements) + 1))
        actual_ids = tuple(item.requirement_id for item in requirements)
        if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
            return self._reject("INVALID_REQUIREMENT_IDS")
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        citation_labels: dict[str, str] = {}
        output: list[AssembledRequirement] = []
        lines: list[str] = []
        citations: list[str] = []
        for requirement in requirements:
            chunk = chunks_by_id.get(requirement.chunk_id)
            if chunk is None:
                return self._reject("INVALID_CHUNK")
            if requirement.chunk_id not in authorized_chunk_ids:
                return self._reject("UNAUTHORIZED_CHUNK")
            if requirement.document_id != chunk.document_id:
                return self._reject("DOCUMENT_MISMATCH")
            if not requirement.supporting_spans:
                return self._reject("MISSING_SUPPORTING_SPAN")
            if any(not span or span not in chunk.text for span in requirement.supporting_spans):
                return self._reject("SPAN_NOT_LITERAL")
            label = citation_labels.setdefault(requirement.chunk_id, f"C{len(citation_labels) + 1}")
            rendered_spans = " ".join(requirement.supporting_spans)
            lines.append(f"{requirement.requirement_id}: {rendered_spans} [{label}]")
            output.append(
                AssembledRequirement(
                    requirement.requirement_id,
                    requirement.chunk_id,
                    requirement.document_id,
                    requirement.supporting_spans,
                    label,
                )
            )
            if requirement.chunk_id not in citations:
                citations.append(requirement.chunk_id)
        if len(output) != len(requirements):
            return self._reject("REQUIREMENT_DROPPED")
        answer = "\n".join(lines)
        if any(span not in answer for item in output for span in item.supporting_spans):
            return self._reject("OUTPUT_SPAN_MISSING")
        if any(f"[{item.citation_label}]" not in answer for item in output):
            return self._reject("OUTPUT_CITATION_MISSING")
        return AssemblyResult("answered", answer, tuple(citations), tuple(output), None, 0)

    @staticmethod
    def _reject(code: str) -> AssemblyResult:
        return AssemblyResult("abstained", None, (), (), code, 0)
