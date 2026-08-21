"""Explicit stage-scoped model-call and monetary guards for evaluations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Literal

ModelKind = Literal["LUNA", "SOL"]


class StageGuardError(RuntimeError):
    def __init__(self, code: str, *, stage: str, case_id: str | None = None) -> None:
        self.code = code
        self.stage = stage
        self.case_id = case_id
        super().__init__(f"{code}:stage={stage}:case_id={case_id or '-'}")

    def as_dict(self) -> dict[str, str | None]:
        return {"code": self.code, "stage": self.stage, "case_id": self.case_id}


@dataclass(frozen=True)
class StageGuardConfig:
    stage: str
    case_count: int
    max_luna_calls_per_case: int
    max_sol_calls_per_case: int
    stage_cost_cap_usd: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        if not self.stage or self.case_count <= 0:
            raise StageGuardError("WRONG_STAGE_GUARD_CONFIGURATION", stage=self.stage)
        if self.max_luna_calls_per_case < 0 or self.max_sol_calls_per_case < 0:
            raise StageGuardError("WRONG_STAGE_GUARD_CONFIGURATION", stage=self.stage)
        if self.stage_cost_cap_usd <= 0:
            raise StageGuardError("WRONG_STAGE_GUARD_CONFIGURATION", stage=self.stage)


class StageResourceGuard:
    """Tracks one active stage; counters and spend never cross stage boundaries."""

    def __init__(self, config: StageGuardConfig, ledger: ExperimentResourceLedger) -> None:
        config.validate()
        self.config = config
        self._ledger = ledger
        self._counts: dict[str, dict[ModelKind, int]] = defaultdict(
            lambda: {"LUNA": 0, "SOL": 0}
        )
        self._spend_usd = 0.0
        self._pending: tuple[str, ModelKind, float] | None = None
        self._closed = False

    @property
    def spend_usd(self) -> float:
        return self._spend_usd

    def count(self, model: ModelKind) -> int:
        return sum(values[model] for values in self._counts.values())

    def count_for_case(self, case_id: str, model: ModelKind) -> int:
        return self._counts[case_id][model]

    def authorize_call(
        self,
        *,
        case_id: str,
        model: ModelKind,
        projected_cost_usd: float,
    ) -> None:
        if self._closed or self._pending is not None or projected_cost_usd < 0:
            raise StageGuardError(
                "MODEL_CALL_GUARD_TRIGGERED", stage=self.config.stage, case_id=case_id
            )
        if case_id not in self._counts and len(self._counts) >= self.config.case_count:
            raise StageGuardError(
                "WRONG_STAGE_GUARD_CONFIGURATION", stage=self.config.stage, case_id=case_id
            )
        limit = (
            self.config.max_luna_calls_per_case
            if model == "LUNA"
            else self.config.max_sol_calls_per_case
        )
        if self._counts[case_id][model] >= limit:
            code = (
                "PER_CASE_ESCALATION_LIMIT_TRIGGERED"
                if model == "SOL"
                else "MODEL_CALL_GUARD_TRIGGERED"
            )
            raise StageGuardError(code, stage=self.config.stage, case_id=case_id)
        if self._spend_usd + projected_cost_usd > self.config.stage_cost_cap_usd:
            raise StageGuardError(
                "STAGE_BUDGET_GUARD_TRIGGERED", stage=self.config.stage, case_id=case_id
            )
        self._pending = (case_id, model, projected_cost_usd)

    def record_call(self, *, case_id: str, model: ModelKind, actual_cost_usd: float) -> None:
        if self._pending is None or self._pending[:2] != (case_id, model) or actual_cost_usd < 0:
            raise StageGuardError(
                "MODEL_CALL_GUARD_TRIGGERED", stage=self.config.stage, case_id=case_id
            )
        self._counts[case_id][model] += 1
        self._spend_usd += actual_cost_usd
        self._ledger._record(actual_cost_usd)
        self._pending = None

    def cancel_pending_call(self) -> None:
        """Clear authorization after a provider transport failure before an identical retry."""
        self._pending = None

    def close(self) -> dict[str, object]:
        if self._pending is not None:
            raise StageGuardError("MODEL_CALL_GUARD_TRIGGERED", stage=self.config.stage)
        self._closed = True
        snapshot = self.snapshot()
        self._ledger._close_stage(self)
        return snapshot

    def snapshot(self) -> dict[str, object]:
        return {
            "stage": self.config.stage,
            "case_count": self.config.case_count,
            "luna_calls": self.count("LUNA"),
            "sol_calls": self.count("SOL"),
            "stage_spend_usd": self._spend_usd,
            "stage_cost_cap_usd": self.config.stage_cost_cap_usd,
            "per_case_counts": {key: dict(value) for key, value in sorted(self._counts.items())},
        }


class ExperimentResourceLedger:
    """Owns cumulative accounting while creating isolated stage guards."""

    def __init__(self, *, historical_spend_usd: float = 0.0) -> None:
        self._lifetime_spend_usd = historical_spend_usd
        self._active: StageResourceGuard | None = None
        self._closed_stages: list[dict[str, object]] = []

    @property
    def lifetime_spend_usd(self) -> float:
        return self._lifetime_spend_usd

    @property
    def closed_stages(self) -> tuple[dict[str, object], ...]:
        return tuple(self._closed_stages)

    def start_stage(
        self,
        config: StageGuardConfig,
        *,
        expected_stage: str,
        expected_case_count: int,
    ) -> StageResourceGuard:
        if self._active is not None or (
            config.stage != expected_stage or config.case_count != expected_case_count
        ):
            raise StageGuardError("WRONG_STAGE_GUARD_CONFIGURATION", stage=config.stage)
        guard = StageResourceGuard(config, self)
        self._active = guard
        return guard

    def _record(self, cost_usd: float) -> None:
        self._lifetime_spend_usd += cost_usd

    def _close_stage(self, guard: StageResourceGuard) -> None:
        if guard is not self._active:
            raise StageGuardError("WRONG_STAGE_GUARD_CONFIGURATION", stage=guard.config.stage)
        self._closed_stages.append(guard.snapshot())
        self._active = None
