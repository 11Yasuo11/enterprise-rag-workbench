"""Canonical, fail-closed evidence-mapping normalization.

This module is the sole boundary between the repository's legacy singular
``supporting_span`` shapes and current plural ``supporting_spans`` shapes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any

from rag_workbench.answerability.base import GateEvidence

SCHEMA_ERROR_CODE = "INVALID_EVIDENCE_MAPPING_SCHEMA"


class EvidenceMappingSchemaError(ValueError):
    """Structured fail-closed error for malformed or ineligible mappings."""

    def __init__(
        self,
        *,
        source_component: str,
        reason: str,
        mapping_shape: dict[str, str],
        requirement_id: str | None,
        missing_field: str | None = None,
    ) -> None:
        self.code = SCHEMA_ERROR_CODE
        self.source_component = source_component
        self.reason = reason
        self.mapping_shape = mapping_shape
        self.requirement_id = requirement_id
        self.missing_field = missing_field
        super().__init__(f"{self.code}:{source_component}:{reason}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "source_component": self.source_component,
            "reason": self.reason,
            "missing_field": self.missing_field,
            "requirement_id": self.requirement_id,
            "mapping_shape": self.mapping_shape,
        }


@dataclass(frozen=True)
class CanonicalEvidenceMapping:
    """One authorized document/chunk mapping with every validated span retained."""

    requirement_id: str
    document_id: str
    document_version_id: str
    version: str
    chunk_id: str
    supporting_spans: tuple[str, ...]
    citations: tuple[str, ...]
    authorized: bool
    tenant: str | None = None
    region: str | None = None
    permission_groups: tuple[str, ...] = ()

    @property
    def supporting_span(self) -> str:
        """Legacy accessor, valid only when singular cardinality is proven."""
        if len(self.supporting_spans) != 1:
            raise EvidenceMappingSchemaError(
                source_component="CanonicalEvidenceMapping.supporting_span",
                reason="SINGULAR_SPAN_CARDINALITY_REQUIRED",
                mapping_shape={"supporting_spans": f"tuple[{len(self.supporting_spans)}]"},
                requirement_id=self.requirement_id,
            )
        return self.supporting_spans[0]


def mapping_shape(value: Any) -> dict[str, str]:
    """Return a non-sensitive field/type inventory for diagnostics."""
    if isinstance(value, Mapping):
        items = value.items()
    elif hasattr(value, "__dict__"):
        items = vars(value).items()
    else:
        fields = getattr(value, "__dataclass_fields__", {})
        items = ((name, getattr(value, name, None)) for name in fields)
    return {str(key): type(item).__name__ for key, item in items}


def _read(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _span_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        candidate = value.get("text") or value.get("supporting_text")
    else:
        candidate = getattr(value, "text", None) or getattr(value, "supporting_text", None)
    return candidate if isinstance(candidate, str) else None


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _required_text(value: Any, field: str, source: str, requirement_id: str | None) -> str:
    candidate = _read(value, field)
    if not isinstance(candidate, str) or not candidate:
        raise EvidenceMappingSchemaError(
            source_component=source,
            reason="MISSING_REQUIRED_IDENTITY",
            missing_field=field,
            mapping_shape=mapping_shape(value),
            requirement_id=requirement_id,
        )
    return candidate


def normalize_evidence_mapping(
    value: Any,
    *,
    evidence_by_chunk: Mapping[str, GateEvidence],
    authorized_chunk_ids: Set[str],
    selected_versions_by_document: Mapping[str, Set[str]] | None = None,
    source_component: str,
) -> CanonicalEvidenceMapping:
    """Normalize legacy/current mappings and validate real security/identity invariants."""
    requirement_id = _read(value, "requirement_id")
    requirement_id = requirement_id if isinstance(requirement_id, str) else None
    requirement_id = _required_text(value, "requirement_id", source_component, requirement_id)
    chunk_id = _required_text(value, "chunk_id", source_component, requirement_id)
    document_id = _required_text(value, "document_id", source_component, requirement_id)
    chunk = evidence_by_chunk.get(chunk_id)
    if chunk is None:
        raise EvidenceMappingSchemaError(
            source_component=source_component,
            reason="UNKNOWN_CHUNK_IDENTITY",
            missing_field="chunk_id",
            mapping_shape=mapping_shape(value),
            requirement_id=requirement_id,
        )
    if chunk_id not in authorized_chunk_ids:
        raise EvidenceMappingSchemaError(
            source_component=source_component,
            reason="UNAUTHORIZED_MAPPING",
            mapping_shape=mapping_shape(value),
            requirement_id=requirement_id,
        )
    if document_id != chunk.document_id:
        raise EvidenceMappingSchemaError(
            source_component=source_component,
            reason="DOCUMENT_IDENTITY_MISMATCH",
            mapping_shape=mapping_shape(value),
            requirement_id=requirement_id,
        )

    supplied_version_id = _read(value, "document_version_id") or _read(
        value, "expected_version_id"
    )
    if supplied_version_id and str(supplied_version_id) != chunk.document_version_id:
        raise EvidenceMappingSchemaError(
            source_component=source_component,
            reason="DOCUMENT_VERSION_IDENTITY_MISMATCH",
            mapping_shape=mapping_shape(value),
            requirement_id=requirement_id,
        )
    supplied_version = _read(value, "version")
    if supplied_version and str(supplied_version) != chunk.version:
        raise EvidenceMappingSchemaError(
            source_component=source_component,
            reason="VERSION_IDENTITY_MISMATCH",
            mapping_shape=mapping_shape(value),
            requirement_id=requirement_id,
        )
    selected = (
        selected_versions_by_document.get(document_id)
        if selected_versions_by_document is not None
        else None
    )
    if selected is not None and chunk.document_version_id not in selected:
        raise EvidenceMappingSchemaError(
            source_component=source_component,
            reason="VERSION_OUTSIDE_SELECTED_DOCUMENT_SCOPE",
            mapping_shape=mapping_shape(value),
            requirement_id=requirement_id,
        )

    plural = _read(value, "supporting_spans")
    singular = _read(value, "supporting_span")
    raw_spans = _sequence(plural) if plural is not None else _sequence(singular)
    spans = tuple(filter(None, (_span_text(item) for item in raw_spans)))
    if not spans:
        raise EvidenceMappingSchemaError(
            source_component=source_component,
            reason="MISSING_VALID_SUPPORTING_SPAN",
            missing_field="supporting_spans",
            mapping_shape=mapping_shape(value),
            requirement_id=requirement_id,
        )
    invalid_spans = tuple(span for span in spans if span not in chunk.text)
    if invalid_spans:
        raise EvidenceMappingSchemaError(
            source_component=source_component,
            reason="SUPPORTING_SPAN_NOT_LITERAL",
            mapping_shape=mapping_shape(value),
            requirement_id=requirement_id,
        )

    raw_citations = _read(value, "citations")
    if raw_citations is None:
        raw_citations = _read(value, "citation")
    citations = tuple(str(item) for item in _sequence(raw_citations) if item)
    if not citations:
        citations = (chunk_id,)
    if any(
        citation not in authorized_chunk_ids or citation not in evidence_by_chunk
        for citation in citations
    ):
        raise EvidenceMappingSchemaError(
            source_component=source_component,
            reason="UNAUTHORIZED_OR_UNKNOWN_CITATION_IDENTITY",
            mapping_shape=mapping_shape(value),
            requirement_id=requirement_id,
        )

    groups = tuple(str(item) for item in _sequence(_read(value, "permission_groups")) if item)
    return CanonicalEvidenceMapping(
        requirement_id=requirement_id,
        document_id=document_id,
        document_version_id=chunk.document_version_id,
        version=chunk.version,
        chunk_id=chunk_id,
        supporting_spans=spans,
        citations=citations,
        authorized=True,
        tenant=_read(value, "tenant"),
        region=_read(value, "region"),
        permission_groups=groups,
    )


def canonical_mappings_to_validation(
    plan: Any,
    mappings: tuple[Any, ...],
    evidence: tuple[GateEvidence, ...],
    *,
    authorized_chunk_ids: frozenset[str],
    selected_versions_by_document: Mapping[str, Set[str]],
) -> Any:
    """Convert normalized mappings to the frozen contract without losing spans."""
    from collections import defaultdict

    from .contract import (
        FrozenValidation,
        ValidatedMapping,
        ValidatedRequirement,
        _mapping_years,
        _required_years_for_requirement,
    )

    evidence_by_chunk = {item.chunk_id: item for item in evidence}
    grouped: dict[str, list[Any]] = defaultdict(list)
    try:
        for value in mappings:
            mapping = normalize_evidence_mapping(
                value,
                evidence_by_chunk=evidence_by_chunk,
                authorized_chunk_ids=authorized_chunk_ids,
                selected_versions_by_document=selected_versions_by_document,
                source_component="deterministic_validation",
            )
            grouped[mapping.requirement_id].append(mapping)
    except EvidenceMappingSchemaError as exc:
        return FrozenValidation(
            False,
            exc.code,
            plan.question_plan_hash,
            schema_error=exc.as_dict(),
        )

    required_ids = tuple(item.requirement_id for item in plan.requirements)
    if set(grouped) != set(required_ids) or any(not grouped[item] for item in required_ids):
        return FrozenValidation(
            False,
            "DETERMINISTIC_REQUIREMENT_COUNT_MISMATCH",
            plan.question_plan_hash,
        )

    validated: list[ValidatedRequirement] = []
    try:
        for planned in plan.requirements:
            span_mappings: list[ValidatedMapping] = []
            for mapping in grouped[planned.requirement_id]:
                if mapping.requirement_id != planned.requirement_id:
                    raise EvidenceMappingSchemaError(
                        source_component="deterministic_validation",
                        reason="REQUIREMENT_IDENTITY_MISMATCH",
                        mapping_shape={"requirement_id": "str"},
                        requirement_id=mapping.requirement_id,
                    )
                for span in mapping.supporting_spans:
                    candidate = ValidatedMapping(
                        mapping.chunk_id,
                        mapping.document_id,
                        span,
                        mapping.document_version_id,
                    )
                    if candidate not in span_mappings:
                        span_mappings.append(candidate)
            if not span_mappings:
                return FrozenValidation(
                    False,
                    "DETERMINISTIC_REQUIREMENT_COUNT_MISMATCH",
                    plan.question_plan_hash,
                )
            first, *additional = span_mappings
            validated.append(
                ValidatedRequirement(
                    planned.requirement_id,
                    planned.requirement_text,
                    first.chunk_id,
                    first.document_id,
                    first.supporting_span,
                    first.document_version_id,
                    tuple(additional),
                )
            )
    except EvidenceMappingSchemaError as exc:
        return FrozenValidation(
            False,
            exc.code,
            plan.question_plan_hash,
            schema_error=exc.as_dict(),
        )

    for requirement in validated:
        needed = _required_years_for_requirement(plan, requirement.requirement_text)
        if not needed:
            continue
        covered = _mapping_years(requirement, evidence_by_chunk)
        if not set(needed) <= covered:
            return FrozenValidation(
                False,
                "CROSS_VERSION_OUTPUT_INCOMPLETE",
                plan.question_plan_hash,
            )
    return FrozenValidation(True, None, plan.question_plan_hash, tuple(validated), "GO", "GO")
