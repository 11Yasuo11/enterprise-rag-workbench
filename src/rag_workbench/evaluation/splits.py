import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from math import floor

from rag_workbench.evaluation.datasets import EvaluationCase, EvaluationDataset

EVIDENCE_GATE_SPLIT_SEED = 20260813


class EvaluationPartition(StrEnum):
    CALIBRATION = "calibration"
    HOLDOUT = "holdout"


@dataclass(frozen=True)
class EvaluationSplit:
    calibration: tuple[EvaluationCase, ...]
    holdout: tuple[EvaluationCase, ...]
    seed: int

    def distribution(self, partition: EvaluationPartition) -> dict[str, object]:
        cases = self.calibration if partition == EvaluationPartition.CALIBRATION else self.holdout
        categories: dict[str, int] = defaultdict(int)
        behavior = {"answerable": 0, "abstention": 0}
        for case in cases:
            categories[case.category] += 1
            behavior["abstention" if case.should_abstain else "answerable"] += 1
        return {
            "cases": len(cases),
            "categories": dict(sorted(categories.items())),
            "behavior": behavior,
        }

    @property
    def identity(self) -> str:
        payload = "|".join(case.case_id for case in self.calibration) + "::" + "|".join(
            case.case_id for case in self.holdout
        )
        return hashlib.sha256(f"{self.seed}:{payload}".encode()).hexdigest()


def deterministic_stratified_split(
    dataset: EvaluationDataset,
    *,
    seed: int = EVIDENCE_GATE_SPLIT_SEED,
    holdout_size: int = 30,
) -> EvaluationSplit:
    if not 0 < holdout_size < len(dataset):
        raise ValueError("holdout_size must leave non-empty calibration and holdout sets")
    strata: dict[tuple[str, bool], list[EvaluationCase]] = defaultdict(list)
    for case in dataset:
        strata[(case.category, case.should_abstain)].append(case)

    exact = {key: len(cases) * holdout_size / len(dataset) for key, cases in strata.items()}
    allocations = {key: floor(value) for key, value in exact.items()}
    remaining = holdout_size - sum(allocations.values())
    order = sorted(strata, key=lambda key: (-(exact[key] - allocations[key]), key))
    for key in order[:remaining]:
        allocations[key] += 1

    holdout_ids: set[str] = set()
    for key in sorted(strata):
        cases = sorted(strata[key], key=lambda case: case.case_id)
        rng = random.Random(f"{seed}:{key[0]}:{int(key[1])}")
        rng.shuffle(cases)
        holdout_ids.update(case.case_id for case in cases[: allocations[key]])

    calibration = tuple(case for case in dataset if case.case_id not in holdout_ids)
    holdout = tuple(case for case in dataset if case.case_id in holdout_ids)
    if len(holdout) != holdout_size:
        raise RuntimeError("deterministic split allocation did not reach requested size")
    return EvaluationSplit(calibration=calibration, holdout=holdout, seed=seed)


@dataclass
class HoldoutEvaluationGuard:
    """Makes selection calibration-only and a locked holdout evaluation one-shot."""

    locked_configuration_hash: str | None = None
    holdout_consumed: bool = False

    def lock_from_calibration(
        self, configuration_hash: str, *, partition: EvaluationPartition
    ) -> None:
        if partition != EvaluationPartition.CALIBRATION:
            raise ValueError("gate configuration may only be selected from calibration results")
        if self.locked_configuration_hash is not None:
            raise ValueError("gate configuration is already locked")
        self.locked_configuration_hash = configuration_hash

    def authorize_holdout(self, configuration_hash: str) -> None:
        if self.locked_configuration_hash != configuration_hash:
            raise ValueError("holdout requires the exact locked configuration")
        if self.holdout_consumed:
            raise ValueError("holdout evaluation has already been consumed")
        self.holdout_consumed = True
