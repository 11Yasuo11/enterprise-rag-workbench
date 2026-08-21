"""Label-blind temporal scope planning for version-aware retrieval."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

TemporalMode = Literal[
    "CURRENT_ONLY",
    "HISTORICAL_ONLY",
    "CROSS_VERSION",
    "UNSPECIFIED_CURRENT_DEFAULT",
]


class VersionMetadata(Protocol):
    document_id: str
    version: str | None
    is_active: bool | None


@dataclass(frozen=True)
class DocumentVersionScope:
    document_id: str
    versions: tuple[str, ...]


@dataclass(frozen=True)
class TemporalScopePlan:
    temporal_mode: TemporalMode
    requested_versions: tuple[str, ...] = ()
    document_scope: tuple[DocumentVersionScope, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "temporal_mode": self.temporal_mode,
            "requested_versions": list(self.requested_versions),
            "document_scope": [asdict(item) for item in self.document_scope],
        }


def temporal_scope_from_dict(value: dict[str, object]) -> TemporalScopePlan:
    """Restore a frozen temporal plan without re-planning from question text."""
    mode = value.get("temporal_mode")
    if mode not in {
        "CURRENT_ONLY",
        "HISTORICAL_ONLY",
        "CROSS_VERSION",
        "UNSPECIFIED_CURRENT_DEFAULT",
    }:
        raise ValueError(f"invalid frozen temporal mode: {mode!r}")
    requested = tuple(str(item) for item in value.get("requested_versions", ()))
    raw_document_scope = value.get("document_scope", ())
    if not isinstance(raw_document_scope, (list, tuple)):
        raise ValueError("frozen document_scope must be a sequence")
    document_scope = tuple(
        DocumentVersionScope(
            str(item["document_id"]), tuple(str(version) for version in item["versions"])
        )
        for item in raw_document_scope
        if isinstance(item, dict)
    )
    return TemporalScopePlan(mode, requested, document_scope)


_CURRENT = re.compile(r"\b(?:current|currently|latest|active|effective|enforceable)\b", re.I)
_HISTORICAL = re.compile(r"\b(?:archived|historical|previous|former|superseded)\b", re.I)


def requested_versions(question: str) -> tuple[str, ...]:
    """Return explicit version labels in question order without duplicates."""
    matches: list[tuple[int, str]] = [
        (match.start(), match.group(0)) for match in re.finditer(r"\b20\d{2}\b", question)
    ]
    matches.extend(
        (match.start(1), match.group(1))
        for match in re.finditer(
            r"\b(?:version|revision|edition)\s+v?(\d+(?:\.\d+)*)\b",
            question,
            re.I,
        )
    )
    ordered: list[str] = []
    for _, version in sorted(matches):
        if version not in ordered:
            ordered.append(version)
    return tuple(ordered)


def plan_temporal_scope(question: str) -> TemporalScopePlan:
    """Infer temporal mode from production-visible question semantics only."""
    versions = requested_versions(question)
    current = bool(_CURRENT.search(question))
    historical = bool(_HISTORICAL.search(question))
    if len(versions) > 1:
        mode: TemporalMode = "CROSS_VERSION"
    elif current:
        mode = "CURRENT_ONLY"
    elif versions or historical:
        mode = "HISTORICAL_ONLY"
    else:
        mode = "UNSPECIFIED_CURRENT_DEFAULT"
    return TemporalScopePlan(mode, versions)


def bind_document_scope(
    plan: TemporalScopePlan, candidates: Iterable[VersionMetadata]
) -> TemporalScopePlan:
    """Bind requested versions to authorized production-visible document metadata."""
    grouped: dict[str, list[str]] = {}
    for candidate in candidates:
        version = candidate.version
        if not version:
            continue
        if plan.requested_versions:
            eligible = version in plan.requested_versions
        elif plan.temporal_mode == "HISTORICAL_ONLY":
            eligible = candidate.is_active is False
        else:
            eligible = bool(candidate.is_active)
        if eligible and version not in grouped.setdefault(candidate.document_id, []):
            grouped[candidate.document_id].append(version)
    order = {version: index for index, version in enumerate(plan.requested_versions)}
    for versions in grouped.values():
        versions.sort(key=lambda version: (order.get(version, len(order)), version))
    scope = tuple(
        DocumentVersionScope(document_id, tuple(versions))
        for document_id, versions in sorted(grouped.items())
        if versions
    )
    return TemporalScopePlan(plan.temporal_mode, plan.requested_versions, scope)


def version_is_eligible(
    plan: TemporalScopePlan, *, version: str | None, is_active: bool | None
) -> bool:
    if plan.temporal_mode in {"CURRENT_ONLY", "UNSPECIFIED_CURRENT_DEFAULT"}:
        return bool(is_active) and (
            not plan.requested_versions or version in plan.requested_versions
        )
    if plan.requested_versions:
        return bool(version and version in plan.requested_versions)
    return plan.temporal_mode == "HISTORICAL_ONLY" and is_active is False
