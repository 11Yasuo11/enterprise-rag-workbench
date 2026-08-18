import time

import httpx

from rag_workbench.answerability.base import (
    AnswerabilityGateError,
    AnswerabilityResult,
    GateEvidence,
    GateOperationalError,
    GateTiming,
)
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    build_provider_evidence_gate_messages,
    evidence_gate_schema,
    parse_answerability_content,
)


class QwenAnswerabilityGate:
    """Local Qwen adapter using Ollama native JSON Schema and thinking controls."""

    provider_name = "qwen"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "ollama",
        gate_version: str = "1",
        prompt_version: str = EVIDENCE_GATE_PROMPT_VERSION,
        timeout: float = 60.0,
    ) -> None:
        if not model:
            raise ValueError("An explicit local Qwen model is required")
        self.model_name = model
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.api_key = api_key
        self.gate_version = gate_version
        self.prompt_version = prompt_version
        self.timeout = timeout
        self.last_timing = GateTiming()

    def evaluate(
        self, question: str, retrieved_chunks: tuple[GateEvidence, ...]
    ) -> AnswerabilityResult:
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model_name,
                    "stream": False,
                    "think": False,
                    "format": evidence_gate_schema(),
                    "options": {"temperature": 0, "seed": 0, "num_predict": 160},
                    "messages": build_provider_evidence_gate_messages(
                        question,
                        retrieved_chunks,
                        provider=self.provider_name,
                        prompt_version=self.prompt_version,
                    ),
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            self.last_timing = GateTiming(
                judge_latency_ms=(time.perf_counter() - started) * 1000,
                local_calls=1,
            )
            raise AnswerabilityGateError(
                GateOperationalError.JUDGE_REQUEST_ERROR,
                "local Qwen judge request failed",
            ) from exc
        self.last_timing = GateTiming(
            judge_latency_ms=(time.perf_counter() - started) * 1000,
            local_calls=1,
            prompt_tokens=payload.get("prompt_eval_count"),
            completion_tokens=payload.get("eval_count"),
        )
        try:
            raw = payload["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise AnswerabilityGateError(
                GateOperationalError.JUDGE_FORMAT_ERROR,
                "local Qwen response omitted JSON content",
            ) from exc
        return parse_answerability_content(raw)
