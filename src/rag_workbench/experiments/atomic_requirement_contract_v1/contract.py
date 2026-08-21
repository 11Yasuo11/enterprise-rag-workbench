"""Canonical label-blind question plan, validation, and deterministic assembly.

The decomposer accepts only production-visible question text. Benchmark labels,
gold answers, expected documents, historical outcomes, and case IDs are neither
arguments nor inputs to any routing decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Set
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.experiments.universal_requirement_completeness_v1 import (
    UniversalEvidenceChunk,
)

if TYPE_CHECKING:
    from .verifier import FrozenVerifierResult


@dataclass(frozen=True)
class AtomicRequirement:
    requirement_id: str
    requirement_text: str
    requirement_type: str = "fact"


@dataclass(frozen=True)
class ContextQualifier:
    qualifier_id: str
    qualifier_type: str
    text: str


@dataclass(frozen=True)
class FrozenQuestionPlan:
    question: str
    requirements: tuple[AtomicRequirement, ...]
    context_qualifiers: tuple[ContextQualifier, ...]
    question_plan_hash: str
    detected_output_units: tuple[str, ...] = ()
    cardinality_match: bool = True

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "requirements": [asdict(item) for item in self.requirements],
            "context_qualifiers": [asdict(item) for item in self.context_qualifiers],
            "detected_output_units": list(self.detected_output_units),
            "cardinality_match": self.cardinality_match,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self.canonical_payload(), "question_plan_hash": self.question_plan_hash}


@dataclass(frozen=True)
class ValidatedMapping:
    chunk_id: str
    document_id: str
    supporting_span: str
    document_version_id: str | None


@dataclass(frozen=True)
class ValidatedRequirement:
    requirement_id: str
    requirement_text: str
    chunk_id: str
    document_id: str
    supporting_span: str
    document_version_id: str | None
    additional_mappings: tuple[ValidatedMapping, ...] = ()

    @property
    def mappings(self) -> tuple[ValidatedMapping, ...]:
        return (
            ValidatedMapping(
                self.chunk_id,
                self.document_id,
                self.supporting_span,
                self.document_version_id,
            ),
            *self.additional_mappings,
        )


@dataclass(frozen=True)
class FrozenValidation:
    valid: bool
    failure_code: str | None
    question_plan_hash: str
    requirements: tuple[ValidatedRequirement, ...] = ()
    raw_decision: str | None = None
    canonical_decision: str | None = None
    schema_error: dict[str, Any] | None = None


@dataclass(frozen=True)
class FrozenAssembly:
    status: str
    answer: str | None
    citations: tuple[str, ...]
    failure_code: str | None
    question_plan_hash: str
    required_requirement_ids: tuple[str, ...]
    verified_requirement_ids: tuple[str, ...]
    output_requirement_ids: tuple[str, ...]
    final_generator_openai_calls: int = 0
    detected_output_unit_count: int = 0
    validated_output_fact_count: int = 0
    assembled_output_fact_count: int = 0
    validated_mapping_keys: tuple[str, ...] = ()
    assembled_mapping_keys: tuple[str, ...] = ()


_AUDIT_TOKEN = re.compile(
    r"^(?P<prefix>.*?audit token\s+[^.]+\.)\s*(?P<request>.*)$", re.IGNORECASE
)
_EVIDENCE_ACCESS_VERB = (
    r"(?:read|consult|open|examine|inspect|review|refer(?:ring)? to|look(?:ing)? at)"
)
_OUTPUT_IMPERATIVE = (
    r"(?:state|report|give|provide|identify|list|name|return|produce|prepare|"
    r"create|pair|combine|put|place)"
)
_LEADING_QUALIFIERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^for audit purposes\s*,\s*", re.I), "audit"),
    (re.compile(r"^in plain language\s*,\s*", re.I), "other"),
    (re.compile(r"^for (?:the )?[^,]*\bregion\s*,\s*", re.I), "region"),
    (re.compile(r"^for (?:an? |the )?[^,]*(?:card|checklist|framing)\s*,\s*", re.I), "scope"),
    (re.compile(r"^for (?:an? |the )?[^,]*\bresilience\s*,\s*", re.I), "region"),
    (
        re.compile(
            r"^under (?:the )?(?:(?:current|active|effective)[^,]*|20\d{2}[^,]*)\s*,\s*",
            re.I,
        ),
        "version",
    ),
    (
        re.compile(
            r"^using (?:only )?(?:the )?(?:presently |currently )?"
            r"(?:enforceable|active|current|effective|archived|historical|superseded)"
            r"[^,]*\s*,\s*",
            re.I,
        ),
        "version",
    ),
    (re.compile(r"^as of [^,]+\s*,\s*", re.I), "time"),
    (
        re.compile(
            r"^(?:from|in) (?:the )?(?:active|archived|current|effective|"
            r"historical|superseded)[^,]*(?:edition|policy|revision|text)\s*,\s*",
            re.I,
        ),
        "version",
    ),
    (re.compile(r"^according to [^,]+\s*,\s*", re.I), "scope"),
    (re.compile(r"^in the event of [^,]+\s*,\s*", re.I), "scope"),
    (re.compile(r"^as part of [^,]+\s*,\s*", re.I), "scope"),
    (re.compile(r"^after [^,]+\s*,\s*", re.I), "scope"),
    (re.compile(r"^during [^,]+\s*,\s*", re.I), "time"),
    (re.compile(r"^when [^,]+\s*,\s*", re.I), "scope"),
    (re.compile(r"^under [^,]+\s*,\s*", re.I), "scope"),
    (re.compile(r"^for [^,]+\s*,\s*", re.I), "scope"),
    (re.compile(r"^as an? [^,]+\s*,\s*", re.I), "principal"),
    (re.compile(r"^if [^,]+\s*,\s*", re.I), "scope"),
    (re.compile(r"^(?:to be clear|for context|please note)\s*,\s*", re.I), "other"),
)
_EVIDENCE_ACCESS_AND_OUTPUT = re.compile(
    rf"^(?P<qual>{_EVIDENCE_ACCESS_VERB}\s+.+?)\s+and\s+"
    rf"(?={_OUTPUT_IMPERATIVE}\b)(?P<body>.+)$",
    re.I,
)


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _qualifier_type(text: str) -> str:
    lower = text.casefold()
    if "audit" in lower:
        return "audit"
    if "region" in lower or re.search(r"\b(?:eu|us|east|west)[- ]region\b", lower):
        return "region"
    if re.search(
        r"\b(?:current|active|effective|enforceable|archived|historical|superseded)\b"
        r".*\b(?:policy|revision|text|edition|version)\b",
        lower,
    ) or any(
        word in lower
        for word in (
            "active",
            "archived",
            "current policy",
            "edition",
            "effective",
            "enforceable",
            "revision",
            "superseded",
            "version",
        )
    ):
        return "version"
    if re.search(r"\b(?:as of|before|after|during)\b", lower) or re.search(r"\b20\d{2}\b", lower):
        return "time"
    if any(word in lower for word in ("scope", "within", "only")):
        return "scope"
    return "other"


def _strip_qualifiers(question: str) -> tuple[str, list[tuple[str, str]]]:
    request = question.strip()
    qualifiers: list[tuple[str, str]] = []
    audit = _AUDIT_TOKEN.match(request)
    if audit:
        qualifiers.append(("audit", audit.group("prefix").strip()))
        request = audit.group("request").strip()

    trailing_versions = re.match(
        r"^(?P<body>.+?)\s+for both (?P<first>20\d{2}) and (?P<second>20\d{2})[.?]?$",
        request,
        re.I,
    )
    if trailing_versions:
        request = trailing_versions.group("body").strip()
        qualifiers.extend(
            (
                ("time", trailing_versions.group("first")),
                ("time", trailing_versions.group("second")),
            )
        )

    # An imperative scope clause before a colon constrains evidence selection.
    if ":" in request:
        prefix, suffix = request.split(":", 1)
        if re.match(r"^(?:use|consider|under|within|for|as of|ignoring)\b", prefix.strip(), re.I):
            qualifiers.append((_qualifier_type(prefix), prefix.strip()))
            request = suffix.strip()

    # Evidence-access preamble before an output imperative is scope, not output.
    # Example: "Read the current remote-work text and state its review interval."
    evidence_access = _EVIDENCE_ACCESS_AND_OUTPUT.match(request)
    if evidence_access:
        qualifiers.append(
            (_qualifier_type(evidence_access.group("qual")), evidence_access.group("qual").strip())
        )
        request = evidence_access.group("body").strip()

    changed = True
    while changed:
        changed = False
        for pattern, qualifier_type in _LEADING_QUALIFIERS:
            match = pattern.match(request)
            if match:
                qualifiers.append((qualifier_type, request[: match.end()].strip(" ,")))
                request = request[match.end() :].strip()
                changed = True
                break
    return request, qualifiers


def _request_body(request: str) -> str:
    body = request.strip().rstrip("?. ")
    body = re.sub(
        r"^(?:please\s+)?(?:provide|give|state|report|list|identify|name)\s+",
        "",
        body,
        flags=re.I,
    )
    body = re.sub(
        r"^(?:could|can|would) you\s+(?:please\s+)?(?:provide|give|state|report|list)\s+",
        "",
        body,
        flags=re.I,
    )
    body = re.sub(r"^both\s+", "", body, flags=re.I)
    return body.strip()


def _split_output_requirements(body: str) -> tuple[str, ...]:
    """Split coordinated requested predicates, not arbitrary noun phrases.

    The rules model a small, deterministic linguistic structure: repeated
    interrogatives, imperative lists, and pair/combine constructions.  In
    particular, a coordinated object such as "Engineering and Support" remains
    inside one predicate unless a second requested predicate is present.
    """
    body = body.strip()

    changed = re.match(
        r"^how did (?P<subject>.+?) change (?:from (?:the )?|between )"
        r"(?P<first>20\d{2})(?: policy)? (?:to (?:the )?|and )"
        r"(?P<second>20\d{2})(?: policy)?$",
        body,
        re.I,
    )
    if changed:
        subject = changed.group("subject").strip()
        return (
            f"{changed.group('first')} {subject}",
            f"{changed.group('second')} {subject}",
        )

    scoped_comparison = re.match(
        r"^compare (?P<subject>.+?) (?:in|under) (?:the )?(?P<first>20\d{2})"
        r" and (?P<second>20\d{2})(?P<context>.*)$",
        body,
        re.I,
    )
    if scoped_comparison:
        subject = scoped_comparison.group("subject").strip()
        context = scoped_comparison.group("context").strip()
        suffix = f" {context}" if context else ""
        return (
            f"{scoped_comparison.group('first')} {subject}{suffix}".strip(),
            f"{scoped_comparison.group('second')} {subject}{suffix}".strip(),
        )

    comparison = re.match(r"^compare\s+(.+?)\s+and\s+(.+)$", body, re.I)
    if comparison:
        return tuple(item.strip(" .") for item in comparison.groups())

    pair = re.match(r"^(?:pair|combine)\s+(.+?)\s+with\s+(.+)$", body, re.I)
    if pair:
        return tuple(item.strip(" .") for item in pair.groups())

    adjacent = re.match(r"^(?:put|place)\s+(.+?)\s+next to\s+(.+)$", body, re.I)
    if adjacent:
        return tuple(item.strip(" .") for item in adjacent.groups())

    plus = re.match(r"^(.+?)\s+plus\s+(.+)$", body, re.I)
    if plus:
        return tuple(item.strip(" .") for item in plus.groups())

    for connector in (r"alongside", r"together with", r"as well as"):
        coordinated = re.match(rf"^(.+?)\s+{connector}\s+(.+)$", body, re.I)
        if coordinated:
            return tuple(item.strip(" .") for item in coordinated.groups())

    # Expand a shared nominal head across coordinated modifiers.  For example,
    # "the east and west recovery codes and the recovery window" requests the
    # east recovery codes, west recovery codes, and recovery window.  The
    # single-token modifier constraint keeps ordinary coordinated objects such
    # as "Engineering and Customer Support" intact.
    shared_head = re.match(
        r"^(?P<article>the\s+)?(?P<first>[\w-]+)\s+and\s+"
        r"(?P<second>[\w-]+)\s+(?P<head>[^,]+?)\s+and\s+(?P<rest>.+)$",
        body,
        re.I,
    )
    if shared_head:
        article = shared_head.group("article") or ""
        head = shared_head.group("head").strip()
        return (
            f"{article}{shared_head.group('first')} {head}".strip(),
            f"{shared_head.group('second')} {head}".strip(),
            shared_head.group("rest").strip(" ."),
        )

    # Coordinated independent questions repeat an interrogative predicate.
    # A comma-delimited repeated-wh list may omit the conjunction until the last item.
    interrogative = (
        r"(?:(?:on|at|in)\s+which|who|what|which|where|when|why|"
        r"how(?:\s+(?:long|quickly|often|many|much))?)"
    )
    if re.match(rf"^{interrogative}\b", body, re.I) and re.search(
        rf",\s*(?:and\s+)?(?={interrogative}\b)", body, re.I
    ):
        parts = re.split(rf"\s*,\s*(?:and\s+)?(?={interrogative}\b)", body, flags=re.I)
        if len(parts) > 1:
            return tuple(item.strip(" .") for item in parts)

    if "," not in body:
        repeated = re.split(
            rf"\s*,?\s+and\s+(?={interrogative}\b)",
            body,
            flags=re.I,
        )
        if len(repeated) > 1:
            return tuple(item.strip(" .") for item in repeated)

    # Wh-questions otherwise describe one requested predicate.  This protects
    # "who coordinates Engineering and Customer Support" from over-splitting.
    wh = re.match(r"^(what (?:is|are)|which (?:is|are))\s+(.+)$", body, re.I)
    prefix = ""
    complement = body
    if wh:
        prefix, complement = wh.group(1), wh.group(2)
    elif re.match(rf"^{interrogative}\b", body, re.I):
        return (body,)

    if "," in complement:
        # Commas delimit list members; an "and" inside a member may join the
        # object of one requested predicate ("coordinates Engineering and
        # Customer Support") and must not create another requirement.
        parts = re.split(r"\s*,\s*(?:and\s+)?", complement, flags=re.I)
    elif re.search(r"\s+and\s+", complement, re.I):
        parts = re.split(r"\s+and\s+", complement, flags=re.I)
    else:
        return (body,)
    cleaned = tuple(item.strip(" .") for item in parts if item.strip(" ."))
    if len(cleaned) < 2:
        return (body,)
    # Keep the grammatical prompt on a single uncoordinated wh-complement only.
    if prefix and len(cleaned) == 1:
        return (f"{prefix} {cleaned[0]}",)
    return cleaned


def semantic_cardinality_audit(plan: FrozenQuestionPlan) -> dict[str, Any]:
    """Return the pre-freeze semantic cardinality evidence in persisted form."""
    requirements = [asdict(item) for item in plan.requirements]
    qualifiers = [asdict(item) for item in plan.context_qualifiers]
    return {
        "question": plan.question,
        "detected_output_units": list(plan.detected_output_units),
        "requirements": requirements,
        "qualifiers": qualifiers,
        "requested_output_count": len(plan.detected_output_units),
        "atomic_requirement_count": len(plan.requirements),
        "cardinality_match": plan.cardinality_match,
    }


def decompose_question(question: str) -> FrozenQuestionPlan:
    """Create and freeze a plan using only the raw production question."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    request, raw_qualifiers = _strip_qualifiers(question)
    texts = _split_output_requirements(_request_body(request))
    if not texts or any(not text for text in texts):
        raise ValueError("question plan has no output requirements")
    requirements = tuple(
        AtomicRequirement(f"R{index}", text, "fact") for index, text in enumerate(texts, 1)
    )
    if re.search(r"\b(?:compare|change(?:d)?|between)\b", request, re.I):
        seen_years = {text for _, text in raw_qualifiers if re.fullmatch(r"20\d{2}", text)}
        for text in texts:
            for year in re.findall(r"\b20\d{2}\b", text):
                if year not in seen_years:
                    raw_qualifiers.append(("time", year))
                    seen_years.add(year)
    qualifiers = tuple(
        ContextQualifier(f"Q{index}", kind, text)
        for index, (kind, text) in enumerate(raw_qualifiers, 1)
    )
    cardinality_match = len(texts) == len(requirements)
    if not cardinality_match:
        raise ValueError("SEMANTIC_CARDINALITY_MISMATCH")
    payload = {
        "question": question,
        "requirements": [asdict(item) for item in requirements],
        "context_qualifiers": [asdict(item) for item in qualifiers],
        "detected_output_units": list(texts),
        "cardinality_match": cardinality_match,
    }
    return FrozenQuestionPlan(
        question,
        requirements,
        qualifiers,
        _hash_payload(payload),
        texts,
        cardinality_match,
    )


def _plan_year_qualifiers(plan: FrozenQuestionPlan) -> tuple[str, ...]:
    years = [
        qualifier.text
        for qualifier in plan.context_qualifiers
        if re.fullmatch(r"20\d{2}", qualifier.text)
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for year in years:
        if year not in seen:
            seen.add(year)
            ordered.append(year)
    return tuple(ordered)


def _required_years_for_requirement(
    plan: FrozenQuestionPlan, requirement_text: str
) -> tuple[str, ...]:
    """Return years a single requirement must literally evidence.

    Year-scoped predicates such as "2025 allowance" only require that year.
    Shared semantic predicates under multi-year qualifiers require every year.
    """
    years = _plan_year_qualifiers(plan)
    if len(years) < 2:
        return ()
    mentioned = tuple(
        year for year in years if re.search(rf"\b{re.escape(year)}\b", requirement_text)
    )
    if len(mentioned) == 1:
        return mentioned
    return years


def _mapping_years(
    requirement: ValidatedRequirement,
    evidence_by_id: Mapping[str, GateEvidence],
) -> set[str]:
    """Years evidenced by literal supporting spans (not metadata alone)."""
    del evidence_by_id  # identity remains available to callers; years come from spans
    years: set[str] = set()
    for mapping in requirement.mappings:
        years.update(re.findall(r"\b20\d{2}\b", mapping.supporting_span))
    return years


def validate_verifier_result(
    plan: FrozenQuestionPlan,
    result: FrozenVerifierResult,
    evidence: tuple[GateEvidence, ...],
    *,
    authorized_chunk_ids: frozenset[str],
    selected_version_ids: frozenset[str] | None = None,
    selected_versions_by_document: Mapping[str, Set[str]] | None = None,
) -> FrozenValidation:
    """Validate against the frozen IDs; never reparses the raw question."""
    if result.question_plan_hash != plan.question_plan_hash:
        return FrozenValidation(False, "QUESTION_PLAN_HASH_MISMATCH", plan.question_plan_hash)
    expected_ids = tuple(item.requirement_id for item in plan.requirements)
    actual_ids = tuple(item.requirement_id for item in result.requirements)
    if actual_ids != expected_ids:
        return FrozenValidation(False, "REQUIREMENT_ID_MISMATCH", plan.question_plan_hash)
    statuses = tuple(item.status for item in result.requirements)
    expected_decision = (
        "GO"
        if all(status == "SUPPORTED" for status in statuses)
        else "ABSTAIN"
        if any(status == "UNSUPPORTED" for status in statuses)
        else "UNCERTAIN"
    )
    if expected_decision != "GO":
        return FrozenValidation(
            True,
            None,
            plan.question_plan_hash,
            raw_decision=result.decision,
            canonical_decision=expected_decision,
        )
    by_id = {item.chunk_id: item for item in evidence}
    mapped: list[ValidatedRequirement] = []
    for planned, item in zip(plan.requirements, result.requirements, strict=True):
        if not item.chunk_ids or not item.document_ids or not item.supporting_spans:
            return FrozenValidation(False, "SUPPORTED_MAPPING_EMPTY", plan.question_plan_hash)
        document_ids = item.document_ids
        if len(document_ids) == 1 and len(item.chunk_ids) > 1:
            candidate_chunks = [by_id.get(chunk_id) for chunk_id in item.chunk_ids]
            if all(
                chunk is not None and chunk.document_id == document_ids[0]
                for chunk in candidate_chunks
            ):
                document_ids = document_ids * len(item.chunk_ids)
        if not (len(item.chunk_ids) == len(document_ids) == len(item.supporting_spans)):
            return FrozenValidation(False, "MAPPING_CARDINALITY_MISMATCH", plan.question_plan_hash)
        if not (
            len(item.document_version_ids) == len(item.chunk_ids)
            and len(item.versions) == len(item.chunk_ids)
        ):
            return FrozenValidation(False, "VERSION_METADATA_MISMATCH", plan.question_plan_hash)
        for chunk_id, document_id, span, document_version_id, version in zip(
            item.chunk_ids,
            document_ids,
            item.supporting_spans,
            item.document_version_ids,
            item.versions,
            strict=True,
        ):
            chunk = by_id.get(chunk_id)
            if chunk is None:
                return FrozenValidation(False, "UNKNOWN_CHUNK", plan.question_plan_hash)
            if chunk_id not in authorized_chunk_ids:
                return FrozenValidation(False, "UNAUTHORIZED_CHUNK", plan.question_plan_hash)
            if document_id != chunk.document_id:
                return FrozenValidation(False, "DOCUMENT_MISMATCH", plan.question_plan_hash)
            if not span or span not in chunk.text:
                return FrozenValidation(False, "SPAN_NOT_LITERAL", plan.question_plan_hash)
            if document_version_id != chunk.document_version_id or version != chunk.version:
                return FrozenValidation(False, "VERSION_METADATA_MISMATCH", plan.question_plan_hash)
            document_versions = (
                selected_versions_by_document.get(chunk.document_id)
                if selected_versions_by_document is not None
                else None
            )
            if (
                document_versions is not None
                and chunk.document_version_id not in document_versions
            ):
                return FrozenValidation(False, "WRONG_VERSION", plan.question_plan_hash)
            if selected_versions_by_document is None and (
                selected_version_ids is not None
                and chunk.document_version_id not in selected_version_ids
            ):
                return FrozenValidation(False, "WRONG_VERSION", plan.question_plan_hash)
        first = by_id[item.chunk_ids[0]]
        additional = tuple(
            ValidatedMapping(
                chunk_id,
                document_id,
                span,
                document_version_id or None,
            )
            for chunk_id, document_id, span, document_version_id in zip(
                item.chunk_ids[1:],
                document_ids[1:],
                item.supporting_spans[1:],
                item.document_version_ids[1:],
                strict=True,
            )
        )
        mapped.append(
            ValidatedRequirement(
                planned.requirement_id,
                planned.requirement_text,
                first.chunk_id,
                first.document_id,
                item.supporting_spans[0],
                first.document_version_id or None,
                additional,
            )
        )
    required_years = _plan_year_qualifiers(plan)
    if len(required_years) >= 2:
        for requirement in mapped:
            needed = _required_years_for_requirement(plan, requirement.requirement_text)
            if not needed:
                continue
            covered = _mapping_years(requirement, by_id)
            if not set(needed) <= covered:
                return FrozenValidation(
                    False,
                    "CROSS_VERSION_OUTPUT_INCOMPLETE",
                    plan.question_plan_hash,
                )
    return FrozenValidation(
        True,
        None,
        plan.question_plan_hash,
        tuple(mapped),
        result.decision,
        expected_decision,
    )


def assemble_frozen_plan(
    plan: FrozenQuestionPlan,
    validation: FrozenValidation,
    chunks: tuple[UniversalEvidenceChunk, ...],
    *,
    authorized_chunk_ids: frozenset[str],
    selected_version_ids: frozenset[str] | None = None,
    selected_versions_by_document: Mapping[str, Set[str]] | None = None,
) -> FrozenAssembly:
    required_ids = tuple(item.requirement_id for item in plan.requirements)
    detected_count = len(plan.detected_output_units)

    def reject(
        code: str,
        *,
        verified_ids: tuple[str, ...] = (),
        output_ids: tuple[str, ...] = (),
        validated_mapping_keys: tuple[str, ...] = (),
        assembled_mapping_keys: tuple[str, ...] = (),
    ) -> FrozenAssembly:
        return FrozenAssembly(
            "abstained",
            None,
            (),
            code,
            plan.question_plan_hash,
            required_ids,
            verified_ids,
            output_ids,
            0,
            detected_count,
            len(verified_ids),
            len(output_ids),
            validated_mapping_keys,
            assembled_mapping_keys,
        )

    if not plan.cardinality_match or detected_count != len(required_ids):
        return reject("SEMANTIC_CARDINALITY_MISMATCH")
    if validation.question_plan_hash != plan.question_plan_hash:
        return reject("QUESTION_PLAN_HASH_MISMATCH")
    verified_ids = tuple(item.requirement_id for item in validation.requirements)
    if not validation.valid or verified_ids != required_ids:
        return reject(
            validation.failure_code or "REQUIREMENT_ID_MISMATCH",
            verified_ids=verified_ids,
        )

    required_years = _plan_year_qualifiers(plan)
    if len(required_years) >= 2:
        for requirement in validation.requirements:
            needed = _required_years_for_requirement(plan, requirement.requirement_text)
            if not needed:
                continue
            covered = {
                year
                for mapping in requirement.mappings
                for year in re.findall(r"\b20\d{2}\b", mapping.supporting_span)
            }
            if not set(needed) <= covered:
                return reject(
                    "CROSS_VERSION_OUTPUT_INCOMPLETE",
                    verified_ids=verified_ids,
                )

    by_id = {item.chunk_id: item for item in chunks}
    citation_labels: dict[str, str] = {}
    citation_ids: list[str] = []
    lines: list[str] = []
    validated_keys: list[str] = []
    assembled_keys: list[str] = []
    for requirement in validation.requirements:
        rendered: list[str] = []
        if not requirement.mappings:
            return reject("SUPPORTED_MAPPING_EMPTY", verified_ids=verified_ids)
        for mapping in requirement.mappings:
            key = _mapping_key(requirement.requirement_id, mapping)
            validated_keys.append(key)
            chunk = by_id.get(mapping.chunk_id)
            if chunk is None:
                return reject(
                    "INVALID_CHUNK",
                    verified_ids=verified_ids,
                    validated_mapping_keys=tuple(validated_keys),
                )
            if mapping.chunk_id not in authorized_chunk_ids:
                return reject("UNAUTHORIZED_CHUNK", verified_ids=verified_ids)
            if mapping.document_id != chunk.document_id:
                return reject("DOCUMENT_MISMATCH", verified_ids=verified_ids)
            if not mapping.supporting_span or mapping.supporting_span not in chunk.text:
                return reject("SPAN_NOT_LITERAL", verified_ids=verified_ids)
            if mapping.document_version_id and mapping.document_version_id != chunk.version_id:
                return reject("VERSION_METADATA_MISMATCH", verified_ids=verified_ids)
            document_versions = (
                selected_versions_by_document.get(chunk.document_id)
                if selected_versions_by_document is not None
                else None
            )
            if document_versions is not None and chunk.version_id not in document_versions:
                return reject("WRONG_VERSION", verified_ids=verified_ids)
            if (
                selected_versions_by_document is None
                and selected_version_ids is not None
                and chunk.version_id not in selected_version_ids
            ):
                return reject("WRONG_VERSION", verified_ids=verified_ids)
            label = citation_labels.setdefault(
                mapping.chunk_id, f"C{len(citation_labels) + 1}"
            )
            if mapping.chunk_id not in citation_ids:
                citation_ids.append(mapping.chunk_id)
            rendered.append(f"{mapping.supporting_span} [{label}]")
            assembled_keys.append(key)
        lines.append(f"{requirement.requirement_id}: {' '.join(rendered)}")

    output_ids = tuple(
        line.split(":", 1)[0] for line in lines if ":" in line
    )
    if output_ids != required_ids:
        return reject(
            "OUTPUT_REQUIREMENT_MISMATCH",
            verified_ids=verified_ids,
            output_ids=output_ids,
            validated_mapping_keys=tuple(validated_keys),
            assembled_mapping_keys=tuple(assembled_keys),
        )
    if not set(validated_keys).issubset(assembled_keys):
        return reject(
            "VALIDATED_MAPPING_DROPPED",
            verified_ids=verified_ids,
            output_ids=output_ids,
            validated_mapping_keys=tuple(validated_keys),
            assembled_mapping_keys=tuple(assembled_keys),
        )
    answer = "\n".join(lines)
    if any(
        mapping.supporting_span not in answer
        for item in validation.requirements
        for mapping in item.mappings
    ):
        return reject("OUTPUT_SPAN_MISSING", verified_ids=verified_ids)
    return FrozenAssembly(
        "answered",
        answer,
        tuple(citation_ids),
        None,
        plan.question_plan_hash,
        required_ids,
        verified_ids,
        output_ids,
        0,
        detected_count,
        len(verified_ids),
        len(output_ids),
        tuple(validated_keys),
        tuple(assembled_keys),
    )


def _mapping_key(requirement_id: str, mapping: ValidatedMapping) -> str:
    return "|".join(
        (
            requirement_id,
            mapping.chunk_id,
            mapping.document_id,
            mapping.document_version_id or "",
            mapping.supporting_span,
        )
    )
