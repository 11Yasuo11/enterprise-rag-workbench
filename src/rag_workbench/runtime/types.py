"""Shared request/response and observability types for CanonicalRagRuntime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RouteName = Literal["deterministic", "luna", "sol", "abstain"]
FinalStatus = Literal["answer", "abstain", "unavailable"]


@dataclass(frozen=True)
class CitationView:
    document_id: str
    chunk_id: str
    source: str
    title: str
    version: str
    page: int | None = None
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequirementView:
    requirement_id: str
    requirement_text: str
    status: str
    chunk_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "requirement_text": self.requirement_text,
            "status": self.status,
            "chunk_ids": list(self.chunk_ids),
        }


@dataclass
class RequestTrace:
    request_id: str
    route: RouteName | str = "abstain"
    dense_candidate_count: int = 0
    bm25_candidate_count: int = 0
    rrf_candidate_count: int = 0
    ce_retained_count: int = 0
    requirements_count: int = 0
    validated_mappings_count: int = 0
    luna_calls: int = 0
    sol_calls: int = 0
    final_status: FinalStatus | str = "abstain"
    latency_ms: float = 0.0
    error_class: str | None = None
    question_plan_hash: str | None = None
    version_resolution_status: str | None = None
    embedding_cache_hit: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "request_id": self.request_id,
            "route": self.route,
            "dense_candidate_count": self.dense_candidate_count,
            "bm25_candidate_count": self.bm25_candidate_count,
            "rrf_candidate_count": self.rrf_candidate_count,
            "ce_retained_count": self.ce_retained_count,
            "requirements_count": self.requirements_count,
            "validated_mappings_count": self.validated_mappings_count,
            "luna_calls": self.luna_calls,
            "sol_calls": self.sol_calls,
            "final_status": self.final_status,
            "latency_ms": self.latency_ms,
            "error_class": self.error_class,
            "question_plan_hash": self.question_plan_hash,
            "version_resolution_status": self.version_resolution_status,
            "embedding_cache_hit": self.embedding_cache_hit,
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


@dataclass(frozen=True)
class CanonicalQueryResult:
    request_id: str
    status: FinalStatus
    answer: str | None
    citations: tuple[CitationView, ...]
    requirements: tuple[RequirementView, ...]
    route: RouteName
    retrieval_results: tuple[dict[str, Any], ...] = ()
    question_plan: dict[str, Any] | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    error_class: str | None = None

    def to_api_dict(self, *, include_debug: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "run_id": self.run_id or self.request_id,
            "status": self.status,
            # Legacy UI aliases
            "legacy_status": "answered" if self.status == "answer" else "abstained",
            "answer": self.answer,
            "citations": [item.to_dict() for item in self.citations],
            "requirements": [item.to_dict() for item in self.requirements],
            "route": self.route,
            "error_class": self.error_class,
        }
        if include_debug:
            payload["retrieval_results"] = list(self.retrieval_results)
            payload["question_plan"] = self.question_plan
            payload["trace"] = self.trace
        return payload
