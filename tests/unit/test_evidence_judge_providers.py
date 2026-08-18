import json

import httpx
import pytest

from rag_workbench.answerability import (
    AnswerabilityGateError,
    AnswerabilityReason,
    CachedAnswerabilityGate,
    GateEvidence,
    OpenAICompatibleAnswerabilityGate,
    QwenAnswerabilityGate,
)
from rag_workbench.answerability.openai_compatible import hosted_judge_chat_payload
from rag_workbench.answerability.planning import (
    ExternalJudgeCallLimitGate,
    LocalJudgeCallLimitGate,
)


def evidence() -> tuple[GateEvidence, ...]:
    return (
        GateEvidence(
            "chunk-1",
            "policy",
            "version-1",
            "2026",
            "Employees receive 20 days of leave.",
            "index-1",
        ),
    )


def response_payload(content: object) -> dict[str, object]:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 12},
    }


def qwen_response_payload(content: object) -> dict[str, object]:
    return {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 30,
        "eval_count": 12,
    }


class Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


def sufficient_json() -> str:
    return json.dumps(
        {
            "answerable": True,
            "supporting_chunk_ids": ["chunk-1"],
            "confidence": 0.9,
            "reason_code": "SUFFICIENT_EVIDENCE",
        }
    )


def test_qwen_adapter_strict_schema_and_deterministic_request(monkeypatch) -> None:
    captured = {}

    def post(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response(qwen_response_payload(sufficient_json()))

    monkeypatch.setattr(httpx, "post", post)
    gate = QwenAnswerabilityGate(model="qwen3:8b", base_url="http://localhost:11434/v1")
    result = gate.evaluate("How much leave?", evidence())
    assert result.reason_code == AnswerabilityReason.SUFFICIENT_EVIDENCE
    assert captured["options"]["temperature"] == 0
    assert captured["think"] is False
    assert captured["format"]["additionalProperties"] is False
    assert gate.last_timing.local_calls == 1
    assert gate.last_timing.external_calls == 0


def test_luna_and_sol_requests_match_except_model_and_persist_usage(monkeypatch) -> None:
    captured = {}

    def post(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response(
            {
                "model": "gpt-5.6-sol",
                "choices": [{"message": {"content": sufficient_json()}}],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 18,
                    "prompt_tokens_details": {"cached_tokens": 4},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )

    monkeypatch.setattr(httpx, "post", post)
    luna = hosted_judge_chat_payload(
        model="gpt-5.6-luna", question="How much leave?", chunks=evidence()
    )
    sol = hosted_judge_chat_payload(
        model="gpt-5.6-sol", question="How much leave?", chunks=evidence()
    )
    assert luna["model"] == "gpt-5.6-luna"
    assert sol["model"] == "gpt-5.6-sol"
    luna.pop("model")
    sol.pop("model")
    assert luna == sol
    gate = OpenAICompatibleAnswerabilityGate(
        api_key="not-a-real-key",
        model="gpt-5.6-sol",
        base_url="https://api.openai.com/v1",
    )
    assert gate.evaluate("How much leave?", evidence()).answerable
    assert captured["model"] == "gpt-5.6-sol"
    assert captured["reasoning_effort"] == "none"
    assert captured["max_completion_tokens"] == 160
    assert gate.last_timing.resolved_model == "gpt-5.6-sol"
    assert gate.last_timing.reasoning_tokens == 0
    assert gate.last_timing.cached_prompt_tokens == 4
    captured = {}

    def post(*args, **kwargs):
        captured.update(kwargs["json"])
        return Response(response_payload(sufficient_json()))

    monkeypatch.setattr(httpx, "post", post)
    gate = OpenAICompatibleAnswerabilityGate(
        api_key="not-a-real-key",
        model="gpt-5.6-luna",
        base_url="https://api.openai.com/v1",
    )
    assert gate.evaluate("How much leave?", evidence()).answerable
    assert captured["model"] == "gpt-5.6-luna"
    assert captured["reasoning_effort"] == "none"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert gate.last_timing.external_calls == 1


def test_malformed_qwen_json_fails_closed_at_adapter_boundary(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx, "post", lambda *args, **kwargs: Response(qwen_response_payload("not-json"))
    )
    gate = QwenAnswerabilityGate(model="qwen3:8b", base_url="http://localhost:11434/v1")
    with pytest.raises(AnswerabilityGateError) as error:
        gate.evaluate("How much leave?", evidence())
    assert error.value.code == "JUDGE_FORMAT_ERROR"


def test_malformed_qwen_outcome_is_cached_for_paired_candidates(
    db_session, monkeypatch
) -> None:
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Response(qwen_response_payload("not-json"))

    monkeypatch.setattr(httpx, "post", post)
    gate = CachedAnswerabilityGate(
        db_session,
        QwenAnswerabilityGate(model="qwen3:8b", base_url="http://localhost:11434/v1"),
    )
    for _context_policy in ("all", "supporting-only"):
        with pytest.raises(AnswerabilityGateError) as error:
            gate.evaluate("Malformed response case", evidence())
        assert error.value.code == "JUDGE_FORMAT_ERROR"
    assert calls == 1
    assert gate.last_timing.cache_hit


def test_provider_model_cache_isolation_and_same_provider_reuse(db_session, monkeypatch) -> None:
    qwen_calls = 0
    luna_calls = 0

    def post(url, *args, **kwargs):
        nonlocal qwen_calls, luna_calls
        if url.startswith("http://localhost"):
            qwen_calls += 1
            return Response(qwen_response_payload(sufficient_json()))
        else:
            luna_calls += 1
        return Response(response_payload(sufficient_json()))

    monkeypatch.setattr(httpx, "post", post)
    qwen = CachedAnswerabilityGate(
        db_session,
        QwenAnswerabilityGate(model="qwen3:8b", base_url="http://localhost:11434/v1"),
    )
    luna = CachedAnswerabilityGate(
        db_session,
        OpenAICompatibleAnswerabilityGate(
            api_key="not-a-real-key",
            model="gpt-5.6-luna",
            base_url="https://api.openai.com/v1",
        ),
    )
    for _context_policy in ("all", "supporting-only"):
        assert qwen.evaluate("How much leave?", evidence()).answerable
        assert luna.evaluate("How much leave?", evidence()).answerable
    assert qwen_calls == 1
    assert luna_calls == 1
    assert qwen.last_timing.cache_hit
    assert luna.last_timing.cache_hit


@pytest.mark.parametrize("wrapper", [ExternalJudgeCallLimitGate, LocalJudgeCallLimitGate])
def test_judge_call_ceilings_are_enforced(wrapper) -> None:
    class Gate:
        provider_name = "test"
        model_name = "test"
        gate_version = "1"
        prompt_version = "v1"
        last_timing = type("Timing", (), {})()

        def evaluate(self, question, chunks):
            return None

    gate = wrapper(Gate(), 1)
    gate.evaluate("first", evidence())
    with pytest.raises(ValueError, match="ceiling"):
        gate.evaluate("second", evidence())
