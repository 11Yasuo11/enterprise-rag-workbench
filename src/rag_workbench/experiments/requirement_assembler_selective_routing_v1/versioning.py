"""Deterministic tenant/region/version resolution and explicit support extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from rag_workbench.experiments.requirement_assembler_selective_routing_v1.assembler import (
    VerifiedRequirement,
)
from rag_workbench.retrieval.temporal import (
    TemporalScopePlan,
    bind_document_scope,
    plan_temporal_scope,
)

VersionResolutionStatus = Literal[
    "VERSION_RESOLVED",
    "VERSION_SET_RESOLVED",
    "VERSION_AMBIGUOUS",
    "METADATA_INSUFFICIENT",
]


@dataclass(frozen=True)
class VersionCandidate:
    chunk_id: str
    document_id: str
    document_version_id: str
    version: str | None
    text: str
    tenant_id: str | None
    region: str | None
    is_active: bool | None
    effective_from: datetime | None = None
    effective_to: datetime | None = None
    superseded_by: str | None = None
    authorized: bool = True


@dataclass(frozen=True)
class VersionResolution:
    status: VersionResolutionStatus
    candidate: VersionCandidate | None
    reason: str
    considered_count: int
    valid_count: int
    requested_region: str | None
    requested_version: str | None
    temporal_semantics: str | None
    selected_version_set: tuple[VersionCandidate, ...] = ()
    temporal_scope: TemporalScopePlan | None = None

    @property
    def selected_version_ids(self) -> tuple[str, ...]:
        return tuple(item.document_version_id for item in self.selected_version_set)

    @property
    def selections_by_document(self) -> dict[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = {}
        for item in self.selected_version_set:
            if item.version and item.version not in grouped.setdefault(item.document_id, []):
                grouped[item.document_id].append(item.version)
        return {document_id: tuple(versions) for document_id, versions in grouped.items()}

    @property
    def selected_version_ids_by_document(self) -> dict[str, frozenset[str]]:
        """Return selected version identities scoped to their owning document."""
        grouped: dict[str, set[str]] = {}
        for item in self.selected_version_set:
            grouped.setdefault(item.document_id, set()).add(item.document_version_id)
        return {document_id: frozenset(ids) for document_id, ids in grouped.items()}


class RequirementIdentity(Protocol):
    requirement_id: str
    requirement_text: str


@dataclass(frozen=True)
class RequirementVersionBinding:
    requirement_id: str
    document_id: str
    document_version_id: str
    version: str
    chunk_id: str


def bind_requirements_to_versions(
    requirements: tuple[RequirementIdentity, ...],
    resolution: VersionResolution,
) -> tuple[RequirementVersionBinding, ...]:
    """Bind atomic cross-version outputs to one explicit version identity each."""
    if resolution.status not in {"VERSION_RESOLVED", "VERSION_SET_RESOLVED"}:
        return ()
    bindings: list[RequirementVersionBinding] = []
    for requirement in requirements:
        requested = plan_temporal_scope(requirement.requirement_text).requested_versions
        if not requested and len(resolution.selected_version_set) == 1:
            requested = (resolution.selected_version_set[0].version or "",)
        candidates = [
            item for item in resolution.selected_version_set if item.version in requested
        ]
        identities = {(item.document_id, item.document_version_id) for item in candidates}
        if len(identities) != 1:
            return ()
        candidate = candidates[0]
        if candidate.version is None:
            return ()
        bindings.append(
            RequirementVersionBinding(
                requirement.requirement_id,
                candidate.document_id,
                candidate.document_version_id,
                candidate.version,
                candidate.chunk_id,
            )
        )
    return tuple(bindings)


def requested_region(question: str) -> str | None:
    request = question.rsplit(". ", 1)[-1].casefold()
    east = bool(re.search(r"\beast(?:-region|-failover)?\b", request))
    west = bool(re.search(r"\bwest(?:-region|-failover)?\b", request))
    if east == west:
        return None
    return "east" if east else "west"


def requested_explicit_version(question: str) -> str | None:
    versions = plan_temporal_scope(question).requested_versions
    return versions[0] if len(versions) == 1 else None


def temporal_semantics(question: str) -> str | None:
    mode = plan_temporal_scope(question).temporal_mode
    if mode == "CURRENT_ONLY":
        return "CURRENT"
    if mode == "HISTORICAL_ONLY":
        return "HISTORICAL"
    if mode == "CROSS_VERSION":
        return "CROSS_VERSION"
    return None


class DeterministicVersionResolver:
    def resolve(
        self,
        question: str,
        candidates: tuple[VersionCandidate, ...],
        *,
        tenant_id: str,
        query_time: datetime | None = None,
        temporal_scope: TemporalScopePlan | None = None,
    ) -> VersionResolution:
        region = requested_region(question)
        scope = temporal_scope or plan_temporal_scope(question)
        explicit = scope.requested_versions[0] if len(scope.requested_versions) == 1 else None
        temporal = temporal_semantics(question)
        considered = [c for c in candidates if c.authorized and c.tenant_id == tenant_id]
        explicit_document_scope = bool(scope.document_scope)
        if not scope.document_scope:
            scope = bind_document_scope(scope, considered)
        if not considered:
            return VersionResolution(
                "METADATA_INSUFFICIENT",
                None,
                "no authorized tenant candidate",
                0,
                0,
                region,
                explicit,
                temporal,
                (),
                scope,
            )
        valid = considered
        if region:
            valid = [c for c in valid if c.region == region]
        if scope.document_scope:
            requested_by_document = {
                item.document_id: set(item.versions) for item in scope.document_scope
            }
            valid = [
                item
                for item in valid
                if item.document_id in requested_by_document
                and item.version in requested_by_document[item.document_id]
            ]
        elif scope.requested_versions:
            requested = set(scope.requested_versions)
            valid = [
                c
                for c in valid
                if c.version in requested or (c.version or "").lstrip("v") in requested
            ]
        if scope.temporal_mode in {"CURRENT_ONLY", "UNSPECIFIED_CURRENT_DEFAULT"}:
            if any(c.is_active is None for c in valid):
                return VersionResolution(
                    "METADATA_INSUFFICIENT",
                    None,
                    "active status missing",
                    len(considered),
                    0,
                    region,
                    explicit,
                    temporal,
                    (),
                    scope,
                )
            valid = [c for c in valid if c.is_active and not c.superseded_by]
            if query_time is not None:
                valid = [
                    c
                    for c in valid
                    if (c.effective_from is None or c.effective_from <= query_time)
                    and (c.effective_to is None or query_time < c.effective_to)
                ]
        elif query_time is not None and scope.temporal_mode != "HISTORICAL_ONLY":
            valid = [
                c
                for c in valid
                if (c.effective_from is None or c.effective_from <= query_time)
                and (c.effective_to is None or query_time < c.effective_to)
            ]
        # Collapse chunk-level candidates to explicit document-version identities.
        unique: dict[tuple[str, str], VersionCandidate] = {}
        for item in valid:
            unique.setdefault((item.document_id, item.document_version_id), item)
        selected = tuple(unique.values())
        version_order = {
            version: index for index, version in enumerate(scope.requested_versions)
        }
        selected = tuple(
            sorted(
                selected,
                key=lambda item: (
                    item.document_id,
                    version_order.get(item.version or "", len(version_order)),
                    item.version or "",
                ),
            )
        )
        by_document: dict[str, list[VersionCandidate]] = {}
        for item in selected:
            by_document.setdefault(item.document_id, []).append(item)

        if scope.temporal_mode == "CROSS_VERSION":
            if explicit_document_scope:
                required_by_document = {
                    item.document_id: set(item.versions) for item in scope.document_scope
                }
                if (
                    not required_by_document
                    or set(by_document) != set(required_by_document)
                    or any(
                        {item.version for item in by_document[document_id]} != versions
                        for document_id, versions in required_by_document.items()
                    )
                ):
                    selected = ()
            elif scope.requested_versions:
                # Explicit multi-version requests require the full requested set on
                # at least one document. Partial coverage is unresolved, not a
                # single-version collapse; unrelated partial docs are dropped.
                requested = set(scope.requested_versions)
                complete_docs = {
                    document_id: items
                    for document_id, items in by_document.items()
                    if requested <= {item.version for item in items}
                }
                selected = (
                    tuple(
                        item
                        for document_id in sorted(complete_docs)
                        for item in complete_docs[document_id]
                    )
                    if complete_docs
                    else ()
                )
            else:
                required_by_document = {
                    item.document_id: set(item.versions) for item in scope.document_scope
                }
                if not required_by_document or set(by_document) != set(required_by_document) or any(
                    {item.version for item in by_document[document_id]} != versions
                    for document_id, versions in required_by_document.items()
                ):
                    selected = ()
        elif scope.temporal_mode in {"CURRENT_ONLY", "UNSPECIFIED_CURRENT_DEFAULT"} and any(
            len(items) != 1 for items in by_document.values()
        ):
            selected = ()

        if len(selected) == 1:
            return VersionResolution(
                "VERSION_RESOLVED",
                selected[0],
                "unique valid candidate",
                len(considered),
                1,
                region,
                explicit,
                temporal,
                selected,
                scope,
            )
        if len(selected) > 1:
            return VersionResolution(
                "VERSION_SET_RESOLVED",
                None,
                "explicit version set resolved per document",
                len(considered),
                len(selected),
                region,
                explicit,
                temporal,
                selected,
                scope,
            )
        reason = "no valid version" if not valid else "multiple valid versions"
        return VersionResolution(
            "VERSION_AMBIGUOUS",
            None,
            reason,
            len(considered),
            len(valid),
            region,
            explicit,
            temporal,
            (),
            scope,
        )


def extract_resolved_token_support(
    question: str, resolution: VersionResolution
) -> tuple[VerifiedRequirement, ...]:
    candidate = resolution.candidate
    if resolution.status != "VERSION_RESOLVED" or candidate is None:
        return ()
    request = question.rsplit(". ", 1)[-1]
    if not re.search(r"\b(token|code|identifier)\b", request, re.I):
        return ()
    sentences = re.split(r"(?<=[.!?])\s+", candidate.text)
    matches = [
        sentence.strip()
        for sentence in sentences
        if re.search(r"\b(?:code|identifier)\b", sentence, re.I)
        and re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b", sentence)
    ]
    if len(matches) != 1:
        return ()
    requirement = re.sub(r"^.*?,\s*", "", request).strip().rstrip(".")
    return (
        VerifiedRequirement(
            "R1", requirement, candidate.chunk_id, candidate.document_id, (matches[0],)
        ),
    )
