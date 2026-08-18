import hashlib
import json
import time

from sqlalchemy.orm import Session

from rag_workbench.answerability.base import (
    AnswerabilityGate,
    AnswerabilityGateError,
    AnswerabilityResult,
    GateEvidence,
    GateOperationalError,
    GateTiming,
)
from rag_workbench.answerability.openai_compatible import (
    evidence_gate_prompt_hash,
    evidence_sufficiency_schema_identity,
)
from rag_workbench.db.models import AnswerabilityGateCacheRecord
from rag_workbench.retrieval.query_embedding_cache import normalize_query_text


def gate_cache_key(
    question: str,
    chunks: tuple[GateEvidence, ...],
    *,
    provider: str,
    model: str,
    gate_version: str,
    prompt_version: str,
) -> tuple[str, dict[str, object]]:
    identity: dict[str, object] = {
        "normalized_question": normalize_query_text(question),
        "retrieved_evidence": [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_version_id": chunk.document_version_id,
                "version": chunk.version,
                "index_identity": chunk.index_identity,
                "content_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
            }
            for chunk in chunks
        ],
        "judge_provider": provider,
        "judge_model": model,
        "judge_version": gate_version,
        "judge_prompt_version": prompt_version,
        "prompt_render_sha256": evidence_gate_prompt_hash(
            question, chunks, provider=provider, prompt_version=prompt_version
        ),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), identity


def judge_logical_request_identity(
    question: str,
    chunks: tuple[GateEvidence, ...],
    *,
    provider: str,
    model: str,
    gate_version: str,
    prompt_version: str,
) -> tuple[str, dict[str, object]]:
    key, identity = gate_cache_key(
        question,
        chunks,
        provider=provider,
        model=model,
        gate_version=gate_version,
        prompt_version=prompt_version,
    )
    payload: dict[str, object] = {
        "logical_request_id": key,
        "normalized_question": identity["normalized_question"],
        "ordered_top5": [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "document_version_id": chunk.document_version_id,
                "version": chunk.version,
                "content_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
            }
            for chunk in chunks
        ],
        "provider": provider,
        "model": model,
        "prompt_hash": identity["prompt_render_sha256"],
        "schema_identity": evidence_sufficiency_schema_identity(),
        "prompt_version": prompt_version,
        "gate_version": gate_version,
        "cache_identity": identity,
    }
    return key, payload


class CachedAnswerabilityGate:
    def __init__(self, session: Session, delegate: AnswerabilityGate) -> None:
        self.session = session
        self.delegate = delegate
        self.provider_name = delegate.provider_name
        self.model_name = delegate.model_name
        self.gate_version = delegate.gate_version
        self.prompt_version = delegate.prompt_version
        self.last_timing = GateTiming()

    def evaluate(
        self, question: str, retrieved_chunks: tuple[GateEvidence, ...]
    ) -> AnswerabilityResult:
        key, identity = gate_cache_key(
            question,
            retrieved_chunks,
            provider=self.provider_name,
            model=self.model_name,
            gate_version=self.gate_version,
            prompt_version=self.prompt_version,
        )
        lookup_started = time.perf_counter()
        record = self.session.get(AnswerabilityGateCacheRecord, key)
        lookup_ms = (time.perf_counter() - lookup_started) * 1000
        if record is not None:
            self.last_timing = GateTiming(
                cache_lookup_latency_ms=lookup_ms,
                cache_hit=True,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                resolved_model=record.judge_model,
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
                    "cached fail-closed judge outcome",
                )
            return AnswerabilityResult.model_validate_json(json.dumps(record.result))

        operational_error: str | None = None
        try:
            result = self.delegate.evaluate(question, retrieved_chunks)
        except AnswerabilityGateError as exc:
            result = AnswerabilityResult.fail_closed()
            operational_error = exc.code.value
        delegate_timing = self.delegate.last_timing
        self.session.add(
            AnswerabilityGateCacheRecord(
                cache_key=key,
                normalized_question=str(identity["normalized_question"]),
                retrieved_evidence=identity["retrieved_evidence"],
                judge_provider=self.provider_name,
                judge_model=self.model_name,
                judge_version=self.gate_version,
                judge_prompt_version=self.prompt_version,
                result=result.model_dump(mode="json"),
                operational_error=operational_error,
                prompt_render_sha256=str(identity["prompt_render_sha256"]),
                prompt_tokens=delegate_timing.prompt_tokens,
                completion_tokens=delegate_timing.completion_tokens,
            )
        )
        self.session.flush()
        self.last_timing = GateTiming(
            cache_lookup_latency_ms=lookup_ms,
            judge_latency_ms=delegate_timing.judge_latency_ms,
            cache_hit=False,
            external_calls=delegate_timing.external_calls,
            local_calls=delegate_timing.local_calls,
            prompt_tokens=delegate_timing.prompt_tokens,
            completion_tokens=delegate_timing.completion_tokens,
            reasoning_tokens=delegate_timing.reasoning_tokens,
            cached_prompt_tokens=delegate_timing.cached_prompt_tokens,
            resolved_model=delegate_timing.resolved_model or self.model_name,
            logical_request_id=delegate_timing.logical_request_id or key,
            attempt_count=delegate_timing.attempt_count,
            retry_count=delegate_timing.retry_count,
            failure_class=delegate_timing.failure_class or operational_error,
            retryable=delegate_timing.retryable,
            retry_reason=delegate_timing.retry_reason,
            final_outcome=delegate_timing.final_outcome
            or ("JUDGE_REQUEST_ERROR" if operational_error else "success"),
            quality_payload_sha256=delegate_timing.quality_payload_sha256,
            attempts=delegate_timing.attempts,
        )
        if operational_error:
            raise AnswerabilityGateError(
                GateOperationalError(operational_error),
                "cached fail-closed judge outcome",
            )
        return result
