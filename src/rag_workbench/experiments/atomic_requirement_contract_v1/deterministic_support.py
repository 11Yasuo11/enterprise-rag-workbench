"""Fail-closed direct-extractive support for frozen atomic requirements."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Set
from dataclasses import dataclass
from typing import Any

from rag_workbench.answerability.base import GateEvidence

from .contract import (
    FrozenQuestionPlan,
    FrozenValidation,
    ValidatedMapping,
    ValidatedRequirement,
    _mapping_years,
    _plan_year_qualifiers,
    _required_years_for_requirement,
)
from .evidence_mapping import (
    CanonicalEvidenceMapping,
    EvidenceMappingSchemaError,
    normalize_evidence_mapping,
)


@dataclass(frozen=True)
class DirectSupportMapping:
    requirement_id: str
    document_id: str
    document_version_id: str
    version: str
    chunk_id: str
    supporting_spans: tuple[str, ...]
    citations: tuple[str, ...]
    relation_family: str


@dataclass(frozen=True)
class DeterministicSupportDecision:
    complete: bool
    failure_code: str | None
    validation: FrozenValidation
    mappings: tuple[CanonicalEvidenceMapping, ...] = ()


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "compose",
        "create",
        "for",
        "from",
        "give",
        "identify",
        "line",
        "name",
        "of",
        "pair",
        "provide",
        "the",
        "to",
        "use",
        "what",
        "which",
        "with",
    }
)

_ALIASES = {
    "allowance": "allowance",
    "cadence": "cadence",
    "commander": "command",
    "coordination": "coordinate",
    "coordinates": "coordinate",
    "coordinating": "coordinate",
    "deadline": "deadline",
    "destination": "destination",
    "drills": "drill",
    "eastern": "east",
    "geography": "geography",
    "guide": "interface",
    "interval": "cadence",
    "keeps": "maintain",
    "maintained": "maintain",
    "maintainer": "maintain",
    "maintains": "maintain",
    "operational": "maintain",
    "position": "cadence",
    "rehearsal": "drill",
    "reporting": "report",
    "reported": "report",
    "remotely": "remote",
    "rhythm": "cadence",
    "schedules": "schedule",
    "standby": "standby",
    "submitted": "submit",
    "targets": "target",
    "updates": "update",
    "weekday": "cadence",
    "weekdays": "cadence",
    "weekly": "week",
    "western": "west",
    "window": "deadline",
}

_RELATION_TERMS = frozenset(
    {
        "allowance",
        "cadence",
        "command",
        "coordinate",
        "deadline",
        "destination",
        "geography",
        "group",
        "maintain",
        "role",
        "schedule",
        "standby",
        "target",
    }
)


def _singular(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "shes", "ches", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _tokens(text: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    values = []
    for raw in re.findall(r"[a-z0-9]+", normalized):
        token = _singular(raw)
        values.append(_ALIASES.get(token, token))
    return frozenset(values)


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip())


def _relation_family(requirement_text: str) -> str | None:
    terms = _tokens(requirement_text)
    if "maintain" in terms:
        return "maintenance"
    if "coordinate" in terms or "command" in terms:
        return "coordination"
    if "allowance" in terms:
        return "allowance"
    if "deadline" in terms or ("report" in terms and "within" in terms):
        return "deadline"
    if "destination" in terms or "geography" in terms or (
        "failover" in terms and ("target" in terms or "standby" in terms)
    ):
        return "destination"
    # "drill matrix containing the X token/code" asks for an identifier, not a cadence.
    if {"token", "code", "identifier"} & terms:
        return None
    if "cadence" in terms or "drill" in terms:
        return "cadence"
    return None


def _sentence_has_relation(family: str, sentence: str) -> bool:
    lower = sentence.casefold()
    terms = _tokens(sentence)
    if family == "maintenance":
        return bool(re.search(r"\bmaintain(?:ed|s|ing)?\s+by\b", lower))
    if family == "coordination":
        return "coordinate" in terms and ("command" in terms or "commander" in lower)
    if family == "allowance":
        return (
            "remote" in terms
            and "week" in terms
            and bool(re.search(r"\b(?:may|could)\s+work\b", lower))
        )
    if family == "deadline":
        return "within" in terms and bool({"report", "submit", "due"} & terms)
    if family == "destination":
        return bool(
            re.search(r"\bfailover\s+targets?\b", lower)
            or re.search(r"\btargets?\s+the\b", lower)
            or ("standby" in terms and bool({"cluster", "geography", "region"} & terms))
        )
    if family == "cadence":
        return bool(
            re.search(
                r"\b(?:every|daily|weekly|monthly|quarterly|mondays?|tuesdays?|"
                r"wednesdays?|thursdays?|fridays?|saturdays?|sundays?)\b",
                lower,
            )
            or re.search(r"\b(?:first|second|third|fourth|last)\s+\w+day\b", lower)
        )
    return False


def _anchors(plan: FrozenQuestionPlan, requirement_text: str) -> frozenset[str]:
    qualifier_terms = {
        token
        for qualifier in plan.context_qualifiers
        for token in _tokens(qualifier.text)
        if token not in {"active", "archived", "current", "effective", "revision", "version"}
    }
    return frozenset(
        token
        for token in (_tokens(requirement_text) | qualifier_terms)
        if len(token) > 2
        and token not in _STOPWORDS
        and token not in _RELATION_TERMS
        and not token.isdigit()
    )


def _chunk_matches_year(chunk: GateEvidence, year: str) -> bool:
    if chunk.version == year:
        return True
    return bool(re.search(rf"\b{re.escape(year)}\b", chunk.text))


def extract_direct_support_mappings(
    plan: FrozenQuestionPlan,
    evidence: tuple[GateEvidence, ...],
) -> tuple[DirectSupportMapping, ...]:
    """Extract only unique, literal relation facts with matching subject anchors."""
    mappings: list[DirectSupportMapping] = []
    for requirement in plan.requirements:
        family = _relation_family(requirement.requirement_text)
        if family is None:
            continue
        anchors = _anchors(plan, requirement.requirement_text)
        years_for_requirement = _required_years_for_requirement(
            plan, requirement.requirement_text
        )
        if len(years_for_requirement) >= 2:
            year_mappings: list[DirectSupportMapping] = []
            for year in years_for_requirement:
                candidates: list[tuple[int, int, GateEvidence, str]] = []
                for evidence_index, chunk in enumerate(evidence):
                    if not _chunk_matches_year(chunk, year):
                        continue
                    if not _region_anchor_satisfied(anchors, chunk):
                        continue
                    context_terms = _tokens(f"{chunk.document_id} {chunk.text}")
                    overlap = len(anchors & context_terms)
                    if overlap == 0:
                        continue
                    for sentence in _sentences(chunk.text):
                        if _sentence_has_relation(family, sentence):
                            doc_bonus = len(anchors & _tokens(chunk.document_id))
                            candidates.append(
                                (overlap + doc_bonus, -evidence_index, chunk, sentence)
                            )
                if not candidates:
                    year_mappings = []
                    break
                best_score = max(item[0] for item in candidates)
                best = [item for item in candidates if item[0] == best_score]
                identities = {(item[2].chunk_id, item[3]) for item in best}
                if len(identities) != 1:
                    year_mappings = []
                    break
                _, _, chunk, sentence = best[0]
                year_mappings.append(
                    DirectSupportMapping(
                        requirement.requirement_id,
                        chunk.document_id,
                        chunk.document_version_id,
                        chunk.version,
                        chunk.chunk_id,
                        (sentence,),
                        (chunk.chunk_id,),
                        family,
                    )
                )
            if len(year_mappings) == len(years_for_requirement):
                mappings.extend(year_mappings)
            continue

        candidates: list[tuple[int, int, GateEvidence, str]] = []
        for evidence_index, chunk in enumerate(evidence):
            if years_for_requirement and not any(
                _chunk_matches_year(chunk, year) for year in years_for_requirement
            ):
                continue
            context_terms = _tokens(f"{chunk.document_id} {chunk.text}")
            overlap = len(anchors & context_terms)
            if overlap == 0:
                continue
            if not _region_anchor_satisfied(anchors, chunk):
                continue
            for sentence in _sentences(chunk.text):
                if _sentence_has_relation(family, sentence):
                    # Prefer document-id/title region matches over incidental tokens
                    # such as "us-east" inside a western runbook.
                    doc_bonus = len(anchors & _tokens(chunk.document_id))
                    candidates.append((overlap + doc_bonus, -evidence_index, chunk, sentence))
        if not candidates:
            continue
        best_score = max(item[0] for item in candidates)
        best = [item for item in candidates if item[0] == best_score]
        identities = {(item[2].chunk_id, item[3]) for item in best}
        if len(identities) != 1:
            # Equally grounded facts may conflict. Ambiguity is never resolved by rank.
            continue
        _, _, chunk, sentence = best[0]
        mappings.append(
            DirectSupportMapping(
                requirement.requirement_id,
                chunk.document_id,
                chunk.document_version_id,
                chunk.version,
                chunk.chunk_id,
                (sentence,),
                (chunk.chunk_id,),
                family,
            )
        )
    return tuple(mappings)


_REGION_ANCHORS = frozenset({"east", "west", "north", "south", "eu", "us"})


def _region_anchor_satisfied(anchors: frozenset[str], chunk: GateEvidence) -> bool:
    """Require explicit regional subject match when the requirement is region-scoped."""
    regions = anchors & _REGION_ANCHORS
    if not regions:
        return True
    document_terms = _tokens(chunk.document_id)
    if regions & document_terms:
        return True
    lower = chunk.text.casefold()
    for region in regions:
        if re.search(rf"\b{re.escape(region)}[- ]region\b", lower):
            return True
        if re.search(rf"\b{re.escape(region)}ern?\b", lower) and "region" in lower:
            return True
    return False


def deterministic_support_complete(
    plan: FrozenQuestionPlan,
    mappings: tuple[Any, ...],
    evidence: tuple[GateEvidence, ...],
    *,
    authorized_chunk_ids: Set[str],
    acl_valid_chunk_ids: Set[str] | None = None,
    tenant_valid_chunk_ids: Set[str] | None = None,
    region_valid_chunk_ids: Set[str] | None = None,
    selected_versions_by_document: Mapping[str, Set[str]] | None = None,
    conflicting_requirement_ids: Set[str] = frozenset(),
    unresolved_version_conflict: bool = False,
    requires_unsupported_inference: bool = False,
    security_precheck_failed: bool = False,
) -> DeterministicSupportDecision:
    """Prove complete direct support or return a structured fail-closed decision."""

    def reject(code: str) -> DeterministicSupportDecision:
        return DeterministicSupportDecision(
            False,
            code,
            FrozenValidation(False, code, plan.question_plan_hash),
        )

    if security_precheck_failed:
        return reject("SECURITY_PRECHECK_FAILED")
    if unresolved_version_conflict:
        return reject("UNRESOLVED_VERSION_CONFLICT")
    if requires_unsupported_inference:
        return reject("UNSUPPORTED_SEMANTIC_INFERENCE_REQUIRED")
    if conflicting_requirement_ids:
        return reject("CONTRADICTORY_MAPPING")
    if not plan.requirements or not mappings:
        return reject("MISSING_REQUIREMENT_MAPPING")

    evidence_by_chunk = {item.chunk_id: item for item in evidence}
    canonical: list[CanonicalEvidenceMapping] = []
    try:
        for mapping in mappings:
            normalized = normalize_evidence_mapping(
                mapping,
                evidence_by_chunk=evidence_by_chunk,
                authorized_chunk_ids=authorized_chunk_ids,
                selected_versions_by_document=selected_versions_by_document,
                source_component="deterministic_support_complete",
            )
            for valid_ids, code in (
                (acl_valid_chunk_ids, "ACL_DENIED"),
                (tenant_valid_chunk_ids, "TENANT_DENIED"),
                (region_valid_chunk_ids, "REGION_DENIED"),
            ):
                if valid_ids is not None and normalized.chunk_id not in valid_ids:
                    return reject(code)
            canonical.append(normalized)
    except EvidenceMappingSchemaError as exc:
        return DeterministicSupportDecision(
            False,
            exc.code,
            FrozenValidation(
                False,
                exc.code,
                plan.question_plan_hash,
                schema_error=exc.as_dict(),
            ),
        )

    required_ids = tuple(item.requirement_id for item in plan.requirements)
    grouped: dict[str, list[CanonicalEvidenceMapping]] = defaultdict(list)
    for mapping in canonical:
        grouped[mapping.requirement_id].append(mapping)
    if set(grouped) != set(required_ids) or any(not grouped[item] for item in required_ids):
        return reject("REQUIREMENT_COVERAGE_INCOMPLETE")

    validated: list[ValidatedRequirement] = []
    for requirement in plan.requirements:
        span_mappings: list[ValidatedMapping] = []
        for mapping in grouped[requirement.requirement_id]:
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
            return reject("REQUIREMENT_COVERAGE_INCOMPLETE")
        first, *additional = span_mappings
        validated.append(
            ValidatedRequirement(
                requirement.requirement_id,
                requirement.requirement_text,
                first.chunk_id,
                first.document_id,
                first.supporting_span,
                first.document_version_id,
                tuple(additional),
            )
        )

    validation = FrozenValidation(
        True,
        None,
        plan.question_plan_hash,
        tuple(validated),
        "GO",
        "GO",
    )
    required_years = _plan_year_qualifiers(plan)
    if len(required_years) >= 2:
        for requirement in validated:
            needed = _required_years_for_requirement(plan, requirement.requirement_text)
            if not needed:
                continue
            covered = _mapping_years(requirement, evidence_by_chunk)
            if not set(needed) <= covered:
                return reject("CROSS_VERSION_OUTPUT_INCOMPLETE")
    return DeterministicSupportDecision(True, None, validation, tuple(canonical))
