from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import (
    AnswerabilityGateError,
    GateEvidence,
    GateOperationalError,
    GateTiming,
)
from rag_workbench.answerability.transport import (
    DEFAULT_TRANSPORT_RETRY_POLICY,
    ProviderFailureClass,
    Sleeper,
    TransportRetryPolicy,
    attempt_record,
    classify_httpx_failure,
    is_retryable_failure,
    retry_after_seconds,
    utc_now_iso,
)
from rag_workbench.answerability.validation import validate_gate_result_with_error
from rag_workbench.db.models import RecoveryStageCacheRecord
from rag_workbench.recovery.contracts import (
    CANNOT_DRAFT,
    CLAIM_CONTRADICTED,
    CLAIM_NOT_SUPPORTED,
    CLAIM_SUPPORTED,
    CLAIM_VERIFIER_PROMPT_VERSION,
    COMPLETENESS_COMPLETE,
    RECOVERY_DRAFT_PROMPT_VERSION,
    RECOVERY_DRAFT_STAGE,
    RECOVERY_TIMEOUT_SECONDS,
    STAGE_CLAIM_VERIFIER,
    RecoveryDraft,
    RecoveryVerification,
    build_claim_verifier_messages,
    build_recovery_draft_messages,
    hosted_recovery_request_settings,
    recovery_draft_schema,
    recovery_draft_schema_identity,
    recovery_prompt_render_hash,
    recovery_verifier_schema,
    recovery_verifier_schema_identity,
)
from rag_workbench.retrieval.query_embedding_cache import normalize_query_text
from rag_workbench.security.permissions import Principal


def ordered_top5_identity(chunks: tuple[GateEvidence, ...]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_version_id": chunk.document_version_id,
            "version": chunk.version,
            "content_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
        }
        for chunk in chunks
    ]


def recovery_cache_key(
    question: str,
    chunks: tuple[GateEvidence, ...],
    *,
    stage: str,
    model: str,
    prompt_version: str,
    prompt_hash: str,
    schema_identity: str,
) -> tuple[str, dict[str, object]]:
    identity: dict[str, object] = {
        "stage": stage,
        "normalized_question": normalize_query_text(question),
        "ordered_top5": ordered_top5_identity(chunks),
        "model": model,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "schema_identity": schema_identity,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), identity


def _chat_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    stage: str,
) -> dict[str, object]:
    settings = hosted_recovery_request_settings(stage)
    schema = (
        recovery_draft_schema()
        if stage == RECOVERY_DRAFT_STAGE
        else recovery_verifier_schema()
    )
    name = settings["response_format"]["json_schema"]["name"]
    return {
        "model": model,
        "temperature": settings["temperature"],
        "reasoning_effort": settings["reasoning_effort"],
        "max_completion_tokens": settings["max_completion_tokens"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": settings["response_format"]["json_schema"]["strict"],
                "schema": schema,
            },
        },
        "messages": messages,
    }


class HostedStructuredRecoveryClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = RECOVERY_TIMEOUT_SECONDS,
        provider_name: str = "openai",
        retry_policy: TransportRetryPolicy = DEFAULT_TRANSPORT_RETRY_POLICY,
        sleeper: Sleeper | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("An API key is required for recovery hosted calls")
        self.api_key = api_key
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.provider_name = provider_name
        self.retry_policy = retry_policy
        self.sleeper: Sleeper = sleeper or time.sleep
        self.last_timing = GateTiming()

    def complete(self, payload: dict[str, object], *, logical_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        payload_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        attempts: list[dict[str, object]] = []
        last_error: AnswerabilityGateError | None = None
        usage: dict[str, object] = {}
        resolved_model: str | None = None
        prompt_identity = str(payload.get("model") or self.model_name)
        for attempt_number in range(1, self.retry_policy.max_total_attempts + 1):
            attempt_started_at = utc_now_iso()
            response: httpx.Response | None = None
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage", {}) if isinstance(body, dict) else {}
                resolved_model = body.get("model") if isinstance(body, dict) else None
                raw = body["choices"][0]["message"]["content"]
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("structured recovery payload was not an object")
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                failure_class = classify_httpx_failure(
                    exc, getattr(exc, "response", response)
                )
                schema_error = isinstance(exc, (ValueError, json.JSONDecodeError))
                if schema_error and not isinstance(exc, httpx.HTTPError):
                    failure_class = ProviderFailureClass.SCHEMA_VALIDATION_ERROR
                retryable = is_retryable_failure(
                    failure_class,
                    attempt_number=attempt_number,
                    policy=self.retry_policy,
                )
                attempts.append(
                    attempt_record(
                        logical_request_id=logical_id,
                        attempt_number=attempt_number,
                        provider=self.provider_name,
                        model=self.model_name,
                        prompt_identity=prompt_identity,
                        request_cache_identity=logical_id,
                        started_at=attempt_started_at,
                        completed_at=utc_now_iso(),
                        failure_class=failure_class,
                        retryable=retryable,
                        retry_reason=failure_class.value if retryable else "non_retryable",
                        outcome="retry" if retryable else "failure",
                    )
                )
                last_error = AnswerabilityGateError(
                    GateOperationalError.JUDGE_REQUEST_ERROR,
                    "hosted recovery request failed",
                    provider_failure_class=failure_class.value,
                    retryable=retryable,
                    attempt_number=attempt_number,
                    logical_request_id=logical_id,
                )
                if retryable:
                    delay = retry_after_seconds(
                        getattr(exc, "response", response), policy=self.retry_policy
                    )
                    self.sleeper(delay)
                    continue
                break
            else:
                attempts.append(
                    attempt_record(
                        logical_request_id=logical_id,
                        attempt_number=attempt_number,
                        provider=self.provider_name,
                        model=self.model_name,
                        prompt_identity=prompt_identity,
                        request_cache_identity=logical_id,
                        started_at=attempt_started_at,
                        completed_at=utc_now_iso(),
                        failure_class=None,
                        retryable=False,
                        retry_reason=None,
                        outcome="success",
                    )
                )
                completion_details = (
                    usage.get("completion_tokens_details") if isinstance(usage, dict) else None
                )
                prompt_details = (
                    usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
                )
                self.last_timing = GateTiming(
                    judge_latency_ms=(time.perf_counter() - started) * 1000,
                    external_calls=len(attempts),
                    prompt_tokens=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
                    completion_tokens=(
                        usage.get("completion_tokens") if isinstance(usage, dict) else None
                    ),
                    reasoning_tokens=(
                        completion_details.get("reasoning_tokens")
                        if isinstance(completion_details, dict)
                        else None
                    ),
                    cached_prompt_tokens=(
                        prompt_details.get("cached_tokens")
                        if isinstance(prompt_details, dict)
                        else None
                    ),
                    resolved_model=resolved_model,
                    logical_request_id=logical_id,
                    attempt_count=len(attempts),
                    retry_count=max(0, len(attempts) - 1),
                    failure_class=None,
                    retryable=False,
                    retry_reason=None,
                    final_outcome="success",
                    quality_payload_sha256=payload_hash,
                    attempts=tuple(attempts),
                )
                return parsed
        self.last_timing = GateTiming(
            judge_latency_ms=(time.perf_counter() - started) * 1000,
            external_calls=len(attempts),
            logical_request_id=logical_id,
            attempt_count=len(attempts),
            retry_count=max(0, len(attempts) - 1),
            failure_class=last_error.provider_failure_class if last_error else None,
            retryable=False,
            retry_reason=(
                last_error.provider_failure_class if last_error else "unknown_provider_error"
            ),
            final_outcome="JUDGE_REQUEST_ERROR",
            quality_payload_sha256=payload_hash,
            attempts=tuple(attempts),
        )
        raise last_error or AnswerabilityGateError(
            GateOperationalError.JUDGE_REQUEST_ERROR,
            "hosted recovery request failed",
            logical_request_id=logical_id,
        )


class CachedRecoveryStage:
    def __init__(
        self,
        session: Session,
        client: HostedStructuredRecoveryClient,
        *,
        maximum_calls: int,
    ) -> None:
        self.session = session
        self.client = client
        self.maximum_calls = maximum_calls
        self.calls = 0
        self.last_timing = GateTiming()

    def complete(
        self,
        question: str,
        chunks: tuple[GateEvidence, ...],
        *,
        stage: str,
        messages: list[dict[str, str]],
        prompt_version: str,
        schema_identity: str,
    ) -> dict[str, Any]:
        prompt_hash = recovery_prompt_render_hash(messages)
        key, identity = recovery_cache_key(
            question,
            chunks,
            stage=stage,
            model=self.client.model_name,
            prompt_version=prompt_version,
            prompt_hash=prompt_hash,
            schema_identity=schema_identity,
        )
        lookup_started = time.perf_counter()
        record = self.session.get(RecoveryStageCacheRecord, key)
        lookup_ms = (time.perf_counter() - lookup_started) * 1000
        if record is not None:
            self.last_timing = GateTiming(
                cache_lookup_latency_ms=lookup_ms,
                cache_hit=True,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                resolved_model=record.model,
                logical_request_id=key,
                attempt_count=0,
                retry_count=0,
                failure_class=record.operational_error,
                retryable=False,
                retry_reason=None,
                final_outcome="cache_hit",
            )
            if record.operational_error:
                raise AnswerabilityGateError(
                    GateOperationalError(record.operational_error),
                    "cached fail-closed recovery outcome",
                    logical_request_id=key,
                )
            return record.result
        if self.calls >= self.maximum_calls:
            raise ValueError("recovery hosted-call ceiling reached")
        self.calls += 1
        payload = _chat_payload(model=self.client.model_name, messages=messages, stage=stage)
        operational_error: str | None = None
        try:
            result = self.client.complete(payload, logical_id=key)
        except AnswerabilityGateError as exc:
            result = {}
            operational_error = exc.code.value
            self.last_timing = self.client.last_timing
            self.session.add(
                RecoveryStageCacheRecord(
                    cache_key=key,
                    stage=stage,
                    normalized_question=str(identity["normalized_question"]),
                    ordered_top5=list(identity["ordered_top5"]),
                    provider=self.client.provider_name,
                    model=self.client.model_name,
                    prompt_version=prompt_version,
                    prompt_hash=prompt_hash,
                    schema_identity=schema_identity,
                    result=result,
                    operational_error=operational_error,
                    prompt_tokens=self.client.last_timing.prompt_tokens,
                    completion_tokens=self.client.last_timing.completion_tokens,
                )
            )
            self.session.flush()
            raise
        timing = self.client.last_timing
        self.session.add(
            RecoveryStageCacheRecord(
                cache_key=key,
                stage=stage,
                normalized_question=str(identity["normalized_question"]),
                ordered_top5=list(identity["ordered_top5"]),
                provider=self.client.provider_name,
                model=self.client.model_name,
                prompt_version=prompt_version,
                prompt_hash=prompt_hash,
                schema_identity=schema_identity,
                result=result,
                operational_error=None,
                prompt_tokens=timing.prompt_tokens,
                completion_tokens=timing.completion_tokens,
            )
        )
        self.session.flush()
        self.last_timing = GateTiming(
            cache_lookup_latency_ms=lookup_ms,
            judge_latency_ms=timing.judge_latency_ms,
            cache_hit=False,
            external_calls=timing.external_calls,
            prompt_tokens=timing.prompt_tokens,
            completion_tokens=timing.completion_tokens,
            reasoning_tokens=timing.reasoning_tokens,
            cached_prompt_tokens=timing.cached_prompt_tokens,
            resolved_model=timing.resolved_model or self.client.model_name,
            logical_request_id=timing.logical_request_id or key,
            attempt_count=timing.attempt_count,
            retry_count=timing.retry_count,
            failure_class=None,
            retryable=False,
            retry_reason=None,
            final_outcome="success",
            quality_payload_sha256=timing.quality_payload_sha256,
            attempts=timing.attempts,
        )
        return result


def citation_ids_from_draft(draft: RecoveryDraft) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.chunk_id for item in draft.citations))


def support_ids_from_verification(verification: RecoveryVerification) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            chunk_id
            for item in verification.claim_results
            for chunk_id in item.supporting_chunk_ids
        )
    )


def draft_uses_only_authorized_ids(
    draft: RecoveryDraft, chunks: tuple[GateEvidence, ...]
) -> bool:
    allowed = {chunk.chunk_id for chunk in chunks}
    cited = set(citation_ids_from_draft(draft))
    claim_ids = {chunk_id for claim in draft.atomic_claims for chunk_id in claim.citation_ids}
    return cited <= allowed and claim_ids <= allowed


def verification_uses_only_authorized_ids(
    verification: RecoveryVerification, chunks: tuple[GateEvidence, ...]
) -> bool:
    allowed = {chunk.chunk_id for chunk in chunks}
    return set(support_ids_from_verification(verification)) <= allowed


def claims_all_supported(verification: RecoveryVerification) -> bool:
    if not verification.claim_results:
        return False
    return all(item.state == CLAIM_SUPPORTED for item in verification.claim_results)


def verification_passes(verification: RecoveryVerification) -> bool:
    contradicted = sum(item.state == CLAIM_CONTRADICTED for item in verification.claim_results)
    unsupported = sum(item.state == CLAIM_NOT_SUPPORTED for item in verification.claim_results)
    return (
        claims_all_supported(verification)
        and contradicted == 0
        and unsupported == 0
        and verification.completeness == COMPLETENESS_COMPLETE
    )


def validate_recovery_support(
    *,
    session: Session,
    principal: Principal,
    chunks: tuple[GateEvidence, ...],
    supporting_ids: tuple[str, ...],
) -> str | None:
    from rag_workbench.answerability.base import AnswerabilityReason, AnswerabilityResult

    proposed = AnswerabilityResult(
        answerable=True,
        supporting_chunk_ids=supporting_ids,
        reason_code=AnswerabilityReason.SUFFICIENT_EVIDENCE,
    )
    validated = validate_gate_result_with_error(
        proposed, chunks, session=session, principal=principal
    )
    if validated.operational_error:
        return validated.operational_error.value
    return None


@dataclass(frozen=True)
class RecoveryOutcome:
    triggered: bool
    answered: bool
    answer: str | None
    citations: tuple[str, ...]
    supporting_chunk_ids: tuple[str, ...]
    draft: RecoveryDraft | None
    verification: RecoveryVerification | None
    draft_logical_request_id: str | None
    verifier_logical_request_id: str | None
    draft_success: bool
    verification_pass: bool
    completeness: str | None
    claim_states: tuple[str, ...]
    validation_error: str | None
    typed_failure: str | None
    draft_timing: GateTiming
    verifier_timing: GateTiming


def evaluate_recovery(
    *,
    session: Session,
    principal: Principal,
    question: str,
    chunks: tuple[GateEvidence, ...],
    cache: CachedRecoveryStage,
    primary_answerable: bool,
    primary_schema_valid: bool,
) -> RecoveryOutcome:
    empty = GateTiming()
    if primary_answerable or not primary_schema_valid:
        return RecoveryOutcome(
            triggered=False,
            answered=False,
            answer=None,
            citations=(),
            supporting_chunk_ids=(),
            draft=None,
            verification=None,
            draft_logical_request_id=None,
            verifier_logical_request_id=None,
            draft_success=False,
            verification_pass=False,
            completeness=None,
            claim_states=(),
            validation_error=None,
            typed_failure=None if primary_answerable else "PRIMARY_NOT_SCHEMA_VALID_NEGATIVE",
            draft_timing=empty,
            verifier_timing=empty,
        )
    messages = build_recovery_draft_messages(question, chunks)
    try:
        raw_draft = cache.complete(
            question,
            chunks,
            stage=RECOVERY_DRAFT_STAGE,
            messages=messages,
            prompt_version=RECOVERY_DRAFT_PROMPT_VERSION,
            schema_identity=recovery_draft_schema_identity(),
        )
    except AnswerabilityGateError as exc:
        return RecoveryOutcome(
            triggered=True,
            answered=False,
            answer=None,
            citations=(),
            supporting_chunk_ids=(),
            draft=None,
            verification=None,
            draft_logical_request_id=exc.logical_request_id,
            verifier_logical_request_id=None,
            draft_success=False,
            verification_pass=False,
            completeness=None,
            claim_states=(),
            validation_error=exc.code.value,
            typed_failure="DRAFT_REQUEST_ERROR",
            draft_timing=cache.last_timing,
            verifier_timing=empty,
        )
    draft_timing = cache.last_timing
    try:
        draft = RecoveryDraft.model_validate(raw_draft)
    except ValidationError:
        return RecoveryOutcome(
            triggered=True,
            answered=False,
            answer=None,
            citations=(),
            supporting_chunk_ids=(),
            draft=None,
            verification=None,
            draft_logical_request_id=draft_timing.logical_request_id,
            verifier_logical_request_id=None,
            draft_success=False,
            verification_pass=False,
            completeness=None,
            claim_states=(),
            validation_error="JUDGE_FORMAT_ERROR",
            typed_failure="DRAFT_SCHEMA_INVALID",
            draft_timing=draft_timing,
            verifier_timing=empty,
        )
    if draft.status == CANNOT_DRAFT or not draft_uses_only_authorized_ids(draft, chunks):
        failure = (
            CANNOT_DRAFT
            if draft.status == CANNOT_DRAFT
            else "UNAUTHORIZED_OR_INVALID_CITATION"
        )
        return RecoveryOutcome(
            triggered=True,
            answered=False,
            answer=None,
            citations=(),
            supporting_chunk_ids=(),
            draft=draft,
            verification=None,
            draft_logical_request_id=draft_timing.logical_request_id,
            verifier_logical_request_id=None,
            draft_success=False,
            verification_pass=False,
            completeness=None,
            claim_states=(),
            validation_error=failure,
            typed_failure=failure,
            draft_timing=draft_timing,
            verifier_timing=empty,
        )
    verifier_messages = build_claim_verifier_messages(question, chunks, draft)
    try:
        raw_verification = cache.complete(
            question,
            chunks,
            stage=STAGE_CLAIM_VERIFIER,
            messages=verifier_messages,
            prompt_version=CLAIM_VERIFIER_PROMPT_VERSION,
            schema_identity=recovery_verifier_schema_identity(),
        )
    except AnswerabilityGateError as exc:
        return RecoveryOutcome(
            triggered=True,
            answered=False,
            answer=None,
            citations=(),
            supporting_chunk_ids=(),
            draft=draft,
            verification=None,
            draft_logical_request_id=draft_timing.logical_request_id,
            verifier_logical_request_id=exc.logical_request_id,
            draft_success=True,
            verification_pass=False,
            completeness=None,
            claim_states=(),
            validation_error=exc.code.value,
            typed_failure="VERIFIER_REQUEST_ERROR",
            draft_timing=draft_timing,
            verifier_timing=cache.last_timing,
        )
    verifier_timing = cache.last_timing
    try:
        verification = RecoveryVerification.model_validate(raw_verification)
    except ValidationError:
        return RecoveryOutcome(
            triggered=True,
            answered=False,
            answer=None,
            citations=(),
            supporting_chunk_ids=(),
            draft=draft,
            verification=None,
            draft_logical_request_id=draft_timing.logical_request_id,
            verifier_logical_request_id=verifier_timing.logical_request_id,
            draft_success=True,
            verification_pass=False,
            completeness=None,
            claim_states=(),
            validation_error="JUDGE_FORMAT_ERROR",
            typed_failure="VERIFIER_SCHEMA_INVALID",
            draft_timing=draft_timing,
            verifier_timing=verifier_timing,
        )
    if not verification_uses_only_authorized_ids(verification, chunks):
        return RecoveryOutcome(
            triggered=True,
            answered=False,
            answer=None,
            citations=(),
            supporting_chunk_ids=(),
            draft=draft,
            verification=verification,
            draft_logical_request_id=draft_timing.logical_request_id,
            verifier_logical_request_id=verifier_timing.logical_request_id,
            draft_success=True,
            verification_pass=False,
            completeness=verification.completeness,
            claim_states=tuple(item.state for item in verification.claim_results),
            validation_error="UNAUTHORIZED_SUPPORTING_ID",
            typed_failure="UNAUTHORIZED_SUPPORTING_ID",
            draft_timing=draft_timing,
            verifier_timing=verifier_timing,
        )
    expected_ids = {item.claim_id for item in draft.atomic_claims}
    verified_ids = {item.claim_id for item in verification.claim_results}
    if expected_ids != verified_ids:
        return RecoveryOutcome(
            triggered=True,
            answered=False,
            answer=None,
            citations=(),
            supporting_chunk_ids=(),
            draft=draft,
            verification=verification,
            draft_logical_request_id=draft_timing.logical_request_id,
            verifier_logical_request_id=verifier_timing.logical_request_id,
            draft_success=True,
            verification_pass=False,
            completeness=verification.completeness,
            claim_states=tuple(item.state for item in verification.claim_results),
            validation_error="CLAIM_ID_MISMATCH",
            typed_failure="CLAIM_ID_MISMATCH",
            draft_timing=draft_timing,
            verifier_timing=verifier_timing,
        )
    if not verification_passes(verification):
        incomplete = verification.completeness != COMPLETENESS_COMPLETE
        return RecoveryOutcome(
            triggered=True,
            answered=False,
            answer=None,
            citations=(),
            supporting_chunk_ids=(),
            draft=draft,
            verification=verification,
            draft_logical_request_id=draft_timing.logical_request_id,
            verifier_logical_request_id=verifier_timing.logical_request_id,
            draft_success=True,
            verification_pass=False,
            completeness=verification.completeness,
            claim_states=tuple(item.state for item in verification.claim_results),
            validation_error="COMPLETENESS_FAILURE" if incomplete else "CLAIM_NOT_ALL_SUPPORTED",
            typed_failure="COMPLETENESS_FAILURE" if incomplete else "CLAIM_NOT_ALL_SUPPORTED",
            draft_timing=draft_timing,
            verifier_timing=verifier_timing,
        )
    citations = citation_ids_from_draft(draft)
    supporting_ids = support_ids_from_verification(verification)
    citation_error = validate_recovery_support(
        session=session,
        principal=principal,
        chunks=chunks,
        supporting_ids=citations,
    )
    validation_error = citation_error or validate_recovery_support(
        session=session,
        principal=principal,
        chunks=chunks,
        supporting_ids=supporting_ids,
    )
    if validation_error:
        return RecoveryOutcome(
            triggered=True,
            answered=False,
            answer=None,
            citations=(),
            supporting_chunk_ids=(),
            draft=draft,
            verification=verification,
            draft_logical_request_id=draft_timing.logical_request_id,
            verifier_logical_request_id=verifier_timing.logical_request_id,
            draft_success=True,
            verification_pass=False,
            completeness=verification.completeness,
            claim_states=tuple(item.state for item in verification.claim_results),
            validation_error=validation_error,
            typed_failure=validation_error,
            draft_timing=draft_timing,
            verifier_timing=verifier_timing,
        )
    return RecoveryOutcome(
        triggered=True,
        answered=True,
        answer=draft.candidate_answer,
        citations=citations,
        supporting_chunk_ids=supporting_ids,
        draft=draft,
        verification=verification,
        draft_logical_request_id=draft_timing.logical_request_id,
        verifier_logical_request_id=verifier_timing.logical_request_id,
        draft_success=True,
        verification_pass=True,
        completeness=verification.completeness,
        claim_states=tuple(item.state for item in verification.claim_results),
        validation_error=None,
        typed_failure=None,
        draft_timing=draft_timing,
        verifier_timing=verifier_timing,
    )


class RecoveryPipeline:
    def __init__(self, cache: CachedRecoveryStage) -> None:
        self.cache = cache

    def recover(
        self,
        *,
        session: Session,
        principal: Principal,
        question: str,
        chunks: tuple[GateEvidence, ...],
        primary_answerable: bool,
        primary_schema_valid: bool,
    ) -> RecoveryOutcome:
        return evaluate_recovery(
            session=session,
            principal=principal,
            question=question,
            chunks=chunks,
            cache=self.cache,
            primary_answerable=primary_answerable,
            primary_schema_valid=primary_schema_valid,
        )
