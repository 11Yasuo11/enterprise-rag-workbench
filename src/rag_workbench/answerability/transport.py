from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import httpx

TRANSPORT_MAX_TOTAL_ATTEMPTS = 2
DEFAULT_TRANSPORT_BACKOFF_SECONDS = 0.05


class ProviderFailureClass(StrEnum):
    TIMEOUT = "TIMEOUT"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    PROVIDER_5XX = "PROVIDER_5XX"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    INVALID_PROVIDER_RESPONSE = "INVALID_PROVIDER_RESPONSE"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    UNKNOWN_PROVIDER_ERROR = "UNKNOWN_PROVIDER_ERROR"


RETRYABLE_PROVIDER_FAILURES = frozenset(
    {
        ProviderFailureClass.TIMEOUT,
        ProviderFailureClass.CONNECTION_ERROR,
        ProviderFailureClass.RATE_LIMIT,
        ProviderFailureClass.PROVIDER_5XX,
    }
)
NON_RETRYABLE_PROVIDER_FAILURES = frozenset(
    {
        ProviderFailureClass.AUTHENTICATION_ERROR,
        ProviderFailureClass.INVALID_PROVIDER_RESPONSE,
        ProviderFailureClass.SCHEMA_VALIDATION_ERROR,
        ProviderFailureClass.UNKNOWN_PROVIDER_ERROR,
    }
)


@dataclass(frozen=True)
class TransportRetryPolicy:
    max_total_attempts: int = TRANSPORT_MAX_TOTAL_ATTEMPTS
    retryable: frozenset[ProviderFailureClass] = RETRYABLE_PROVIDER_FAILURES
    backoff_seconds: float = DEFAULT_TRANSPORT_BACKOFF_SECONDS


DEFAULT_TRANSPORT_RETRY_POLICY = TransportRetryPolicy()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def classify_httpx_failure(
    exc: BaseException, response: httpx.Response | None = None
) -> ProviderFailureClass:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderFailureClass.TIMEOUT
    if isinstance(exc, (httpx.ConnectError, httpx.NetworkError)):
        return ProviderFailureClass.CONNECTION_ERROR
    status = None
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        response = exc.response
    elif response is not None:
        status = response.status_code
    if status == 429:
        return ProviderFailureClass.RATE_LIMIT
    if status in {401, 403}:
        return ProviderFailureClass.AUTHENTICATION_ERROR
    if status is not None and status >= 500:
        return ProviderFailureClass.PROVIDER_5XX
    if isinstance(exc, (ValueError, KeyError, IndexError, TypeError)):
        return ProviderFailureClass.INVALID_PROVIDER_RESPONSE
    return ProviderFailureClass.UNKNOWN_PROVIDER_ERROR


def is_retryable_failure(
    failure_class: ProviderFailureClass,
    *,
    attempt_number: int,
    policy: TransportRetryPolicy = DEFAULT_TRANSPORT_RETRY_POLICY,
) -> bool:
    return failure_class in policy.retryable and attempt_number < policy.max_total_attempts


def retry_after_seconds(
    response: httpx.Response | None,
    *,
    policy: TransportRetryPolicy = DEFAULT_TRANSPORT_RETRY_POLICY,
) -> float:
    if response is not None:
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
    return policy.backoff_seconds


def attempt_record(
    *,
    logical_request_id: str,
    attempt_number: int,
    provider: str,
    model: str,
    prompt_identity: str,
    request_cache_identity: str,
    started_at: str,
    completed_at: str,
    failure_class: ProviderFailureClass | None,
    retryable: bool,
    retry_reason: str | None,
    outcome: str,
) -> dict[str, object]:
    return {
        "logical_request_id": logical_request_id,
        "attempt_number": attempt_number,
        "provider": provider,
        "model": model,
        "prompt_identity": prompt_identity,
        "request_cache_identity": request_cache_identity,
        "started_at": started_at,
        "completed_at": completed_at,
        "failure_class": failure_class.value if failure_class else None,
        "retryable": retryable,
        "retry_reason": retry_reason,
        "final_outcome": outcome,
    }


def no_sleep(_seconds: float) -> None:
    return None


Sleeper = Callable[[float], None]
