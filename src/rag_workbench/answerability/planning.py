from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import (
    AnswerabilityGate,
    AnswerabilityResult,
    GateEvidence,
    GateTiming,
)
from rag_workbench.answerability.cache import gate_cache_key
from rag_workbench.db.models import AnswerabilityGateCacheRecord


@dataclass(frozen=True)
class GateInput:
    question: str
    chunks: tuple[GateEvidence, ...]


@dataclass(frozen=True)
class JudgeCallPlan:
    judge_provider: str
    judge_model: str
    unique_gate_inputs: int
    cached_gate_results: int
    missing_gate_results: int
    expected_maximum_calls: int
    maximum_possible_holdout_calls: int
    configured_ceiling: int
    recorded_provider_results: int

    def safe_summary(self) -> str:
        return "\n".join(
            (
                f"judge provider: {self.judge_provider}",
                f"judge model: {self.judge_model}",
                f"calibration unique inputs: {self.unique_gate_inputs}",
                f"cached inputs: {self.cached_gate_results}",
                f"missing inputs: {self.missing_gate_results}",
                f"maximum possible holdout calls: {self.maximum_possible_holdout_calls}",
                f"configured ceiling: {self.configured_ceiling}",
            )
        )


def plan_judge_calls(
    session: Session,
    inputs: tuple[GateInput, ...],
    *,
    provider: str,
    model: str,
    gate_version: str,
    prompt_version: str,
    configured_ceiling: int,
    maximum_possible_holdout_calls: int = 30,
) -> JudgeCallPlan:
    keys = {
        gate_cache_key(
            item.question,
            item.chunks,
            provider=provider,
            model=model,
            gate_version=gate_version,
            prompt_version=prompt_version,
        )[0]
        for item in inputs
    }
    cached = sum(session.get(AnswerabilityGateCacheRecord, key) is not None for key in keys)
    missing = len(keys) - cached
    recorded = session.scalar(
        select(func.count())
        .select_from(AnswerabilityGateCacheRecord)
        .where(
            AnswerabilityGateCacheRecord.judge_provider == provider,
            AnswerabilityGateCacheRecord.judge_model == model,
            AnswerabilityGateCacheRecord.judge_version == gate_version,
            AnswerabilityGateCacheRecord.judge_prompt_version == prompt_version,
        )
    ) or 0
    return JudgeCallPlan(
        judge_provider=provider,
        judge_model=model,
        unique_gate_inputs=len(keys),
        cached_gate_results=cached,
        missing_gate_results=missing,
        expected_maximum_calls=missing,
        maximum_possible_holdout_calls=maximum_possible_holdout_calls,
        configured_ceiling=configured_ceiling,
        recorded_provider_results=recorded,
    )


class ExternalJudgeCallLimitGate:
    def __init__(self, delegate: AnswerabilityGate, maximum_calls: int) -> None:
        self.delegate = delegate
        self.maximum_calls = maximum_calls
        self.calls = 0
        self.provider_name = delegate.provider_name
        self.model_name = delegate.model_name
        self.gate_version = delegate.gate_version
        self.prompt_version = delegate.prompt_version
        self.last_timing = GateTiming()

    def evaluate(
        self, question: str, retrieved_chunks: tuple[GateEvidence, ...]
    ) -> AnswerabilityResult:
        if self.calls >= self.maximum_calls:
            raise ValueError("answerability judge external-call ceiling reached")
        self.calls += 1
        try:
            result = self.delegate.evaluate(question, retrieved_chunks)
            return result
        finally:
            self.last_timing = self.delegate.last_timing


class LocalJudgeCallLimitGate(ExternalJudgeCallLimitGate):
    def evaluate(
        self, question: str, retrieved_chunks: tuple[GateEvidence, ...]
    ) -> AnswerabilityResult:
        if self.calls >= self.maximum_calls:
            raise ValueError("answerability judge local-call ceiling reached")
        self.calls += 1
        try:
            result = self.delegate.evaluate(question, retrieved_chunks)
            return result
        finally:
            self.last_timing = self.delegate.last_timing
