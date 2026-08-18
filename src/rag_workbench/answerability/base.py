from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnswerabilityReason(StrEnum):
    SUFFICIENT_EVIDENCE = "SUFFICIENT_EVIDENCE"
    MISSING_REQUIRED_FACT = "MISSING_REQUIRED_FACT"
    PARTIAL_EVIDENCE = "PARTIAL_EVIDENCE"
    IRRELEVANT_EVIDENCE = "IRRELEVANT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    ACCESS_RESTRICTED_EVIDENCE = "ACCESS_RESTRICTED_EVIDENCE"
    AMBIGUOUS_EVIDENCE = "AMBIGUOUS_EVIDENCE"
    UNKNOWN = "UNKNOWN"


class RequirementStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    CONFLICTING = "CONFLICTING"


class EvidenceRequirement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    requirement_id: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=1, max_length=180)
    status: RequirementStatus
    supporting_chunk_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def supported_requires_evidence(self) -> "EvidenceRequirement":
        if self.status == RequirementStatus.SUPPORTED and not self.supporting_chunk_ids:
            raise ValueError("supported requirements need at least one supporting chunk")
        return self


class AnswerabilityResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    answerable: bool
    supporting_chunk_ids: tuple[str, ...] = ()
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason_code: AnswerabilityReason
    requirements: tuple[EvidenceRequirement, ...] = ()

    @model_validator(mode="after")
    def reason_matches_decision(self) -> "AnswerabilityResult":
        if self.answerable and self.reason_code != AnswerabilityReason.SUFFICIENT_EVIDENCE:
            raise ValueError("answerable results require SUFFICIENT_EVIDENCE")
        if not self.answerable and self.reason_code == AnswerabilityReason.SUFFICIENT_EVIDENCE:
            raise ValueError("insufficient results cannot use SUFFICIENT_EVIDENCE")
        if self.requirements:
            ids = [item.requirement_id for item in self.requirements]
            if len(ids) != len(set(ids)):
                raise ValueError("requirement IDs must be unique")
            if self.answerable:
                if any(item.status != RequirementStatus.SUPPORTED for item in self.requirements):
                    raise ValueError(
                        "answerable coverage results require every requirement supported"
                    )
                union = tuple(
                    dict.fromkeys(
                        chunk_id
                        for requirement in self.requirements
                        for chunk_id in requirement.supporting_chunk_ids
                    )
                )
                if set(union) != set(self.supporting_chunk_ids) or not union:
                    raise ValueError("supporting chunks must equal the requirement evidence union")
            elif self.supporting_chunk_ids:
                raise ValueError("insufficient coverage results cannot expose generation support")
        return self

    @classmethod
    def fail_closed(
        cls, reason: AnswerabilityReason = AnswerabilityReason.UNKNOWN
    ) -> "AnswerabilityResult":
        if reason == AnswerabilityReason.SUFFICIENT_EVIDENCE:
            reason = AnswerabilityReason.UNKNOWN
        return cls(answerable=False, supporting_chunk_ids=(), reason_code=reason)


@dataclass(frozen=True)
class GateEvidence:
    chunk_id: str
    document_id: str
    document_version_id: str
    version: str
    text: str
    index_identity: str | None = None


@dataclass(frozen=True)
class GateTiming:
    cache_lookup_latency_ms: float = 0.0
    judge_latency_ms: float = 0.0
    cache_hit: bool = False
    external_calls: int = 0
    local_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    resolved_model: str | None = None
    logical_request_id: str | None = None
    attempt_count: int = 0
    retry_count: int = 0
    failure_class: str | None = None
    retryable: bool | None = None
    retry_reason: str | None = None
    final_outcome: str | None = None
    quality_payload_sha256: str | None = None
    attempts: tuple[dict[str, Any], ...] = ()


class GateOperationalError(StrEnum):
    JUDGE_REQUEST_ERROR = "JUDGE_REQUEST_ERROR"
    JUDGE_FORMAT_ERROR = "JUDGE_FORMAT_ERROR"
    MISSING_SUPPORTING_ID = "MISSING_SUPPORTING_ID"
    INVALID_SUPPORTING_ID = "INVALID_SUPPORTING_ID"
    UNAUTHORIZED_SUPPORTING_ID = "UNAUTHORIZED_SUPPORTING_ID"
    INACTIVE_VERSION = "INACTIVE_VERSION"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    EXPERIMENT_MISMATCH = "EXPERIMENT_MISMATCH"


class AnswerabilityGateError(RuntimeError):
    def __init__(
        self,
        code: GateOperationalError,
        message: str,
        *,
        provider_failure_class: str | None = None,
        retryable: bool = False,
        attempt_number: int | None = None,
        logical_request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.provider_failure_class = provider_failure_class
        self.retryable = retryable
        self.attempt_number = attempt_number
        self.logical_request_id = logical_request_id


@dataclass(frozen=True)
class ValidatedAnswerabilityResult:
    result: AnswerabilityResult
    operational_error: GateOperationalError | None = None


class AnswerabilityGate(Protocol):
    provider_name: str
    model_name: str
    gate_version: str
    prompt_version: str
    last_timing: GateTiming

    def evaluate(
        self, question: str, retrieved_chunks: tuple[GateEvidence, ...]
    ) -> AnswerabilityResult: ...
