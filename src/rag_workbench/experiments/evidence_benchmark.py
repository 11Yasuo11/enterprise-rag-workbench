from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import (
    AnswerabilityGateCacheRecord,
    EvidenceBenchmarkLockRecord,
    ExperimentRunRecord,
)
from rag_workbench.evaluation.datasets import EvaluationCase, load_evaluation_dataset
from rag_workbench.evaluation.splits import deterministic_stratified_split
from rag_workbench.experiments.configs import (
    AnswerabilityGateConfig,
    CorpusDatasetIdentity,
    ExperimentConfig,
    GenerationConfig,
    IngestionConfig,
    RetrievalConfig,
)
from rag_workbench.experiments.reporting import experiment_to_dict
from rag_workbench.experiments.runner import ExperimentExecutionOptions, ExperimentRunner

DATASET_PATH = Path("data/eval/eval_v1.json")
SPLIT_IDENTITY = "208f41caafcf5d05e8aa7c32f912a56324ee6a52d57fa073d49b6ddc5c35adf1"
ORIGINAL_NINE = frozenset(
    {
        "abstention_04",
        "abstention_11",
        "abstention_13",
        "abstention_14",
        "acl_benefit_denied_01",
        "acl_benefit_denied_02",
        "acl_comp_denied_01",
        "acl_comp_denied_02",
        "acl_comp_denied_03",
    }
)


@dataclass(frozen=True)
class Candidate:
    code: str
    config: ExperimentConfig


def benchmark_candidates(settings: Settings | None = None) -> tuple[Candidate, ...]:
    settings = settings or get_settings()
    common = {
        "identity": CorpusDatasetIdentity(
            corpus_version="acmeai-v1", evaluation_dataset_version="acmeai-eval-v1"
        ),
        "ingestion": IngestionConfig(
            chunk_size=180,
            chunk_overlap=30,
            embedding_provider="openai-compatible",
            embedding_model="text-embedding-3-small",
            embedding_dimension=64,
            embedding_version="1",
        ),
        "retrieval": RetrievalConfig(top_k=5, score_threshold=0.28),
        "generation": GenerationConfig(
            llm_provider="extractive",
            llm_model="deterministic-extractive-v1",
            prompt_version="baseline-v1",
            temperature=0,
            context_budget=1200,
        ),
    }

    def config(code: str, gate: AnswerabilityGateConfig | None) -> Candidate:
        return Candidate(
            code,
            ExperimentConfig(
                name=f"evidence-judge-calibration-{code}",
                answerability_gate=gate,
                **common,
            ),
        )

    qwen = {
        "judge_provider": "qwen",
        "judge_model": settings.local_judge_model,
        "judge_version": "4",
        "prompt_version": "evidence-sufficiency-v1",
    }
    luna = {
        "judge_provider": "openai",
        "judge_model": "gpt-5.6-luna",
        "judge_version": "1",
        "prompt_version": "evidence-sufficiency-v1",
    }
    return (
        config("A", None),
        config("B1", AnswerabilityGateConfig(**qwen, supporting_context_only=False)),
        config("B2", AnswerabilityGateConfig(**qwen, supporting_context_only=True)),
        config("C1", AnswerabilityGateConfig(**luna, supporting_context_only=False)),
        config("C2", AnswerabilityGateConfig(**luna, supporting_context_only=True)),
    )


class EvidenceJudgeBenchmark:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.dataset = load_evaluation_dataset(DATASET_PATH)
        self.split = deterministic_stratified_split(self.dataset)
        if self.split.identity != SPLIT_IDENTITY:
            raise ValueError("evaluation split identity changed; benchmark execution is blocked")
        self.candidates = {item.code: item for item in benchmark_candidates(self.settings)}

    def run_calibration(self) -> dict[str, Any]:
        if self.session.get(EvidenceBenchmarkLockRecord, self.split.identity) is not None:
            raise ValueError("benchmark configuration is already locked; calibration is immutable")
        runner = ExperimentRunner(self.session, settings=self.settings)
        run_ids: dict[str, str] = {}
        for code in ("A", "B1", "B2", "C1", "C2"):
            candidate = self.candidates[code]
            options = ExperimentExecutionOptions(
                offline_only=False,
                confirm_external_calls=True,
                confirm_external_judge_calls=code in {"C1", "C2"},
            )
            result = runner.run(
                candidate.config,
                DATASET_PATH,
                options,
                cases_override=self.split.calibration,
            )
            if result.run_id is None:
                raise RuntimeError(f"calibration run {code} was not persisted")
            run_ids[code] = result.run_id
        winner, reason = self._select(run_ids)
        selected = self.candidates[winner].config
        selected_run = self.session.get(ExperimentRunRecord, run_ids[winner])
        if selected_run is None:
            raise RuntimeError("selected calibration run disappeared")
        gate = selected.answerability_gate
        lock = EvidenceBenchmarkLockRecord(
            split_identity=self.split.identity,
            selected_candidate=winner,
            selected_configuration_hash=selected.experiment_config_hash,
            judge_provider=gate.judge_provider if gate else None,
            judge_model=gate.judge_model if gate else None,
            judge_version=gate.judge_version if gate else None,
            prompt_version=gate.prompt_version if gate else None,
            threshold=selected.retrieval.score_threshold or 0.0,
            supporting_context_only=gate.supporting_context_only if gate else False,
            calibration_metrics={
                "run_ids": run_ids,
                "candidates": {
                    code: self.session.get(ExperimentRunRecord, run_id).aggregate_metrics
                    for code, run_id in run_ids.items()
                },
            },
            selection_reason=reason,
        )
        self.session.add(lock)
        self.session.commit()
        return self.status(include_cases=True)

    def run_holdout(self) -> dict[str, Any]:
        lock = self.session.get(EvidenceBenchmarkLockRecord, self.split.identity)
        if lock is None:
            raise ValueError("calibration winner must be locked before holdout")
        if lock.holdout_started_at is not None:
            raise ValueError("holdout execution is one-shot and has already started")
        selected = self.candidates[lock.selected_candidate]
        if selected.config.experiment_config_hash != lock.selected_configuration_hash:
            raise ValueError("locked configuration hash no longer matches the candidate")

        lock.holdout_started_at = datetime.now(UTC)
        self.session.commit()
        runner = ExperimentRunner(self.session, settings=self.settings)
        baseline = self.candidates["A"].config.model_copy(
            update={"name": "evidence-judge-holdout-A"}
        )
        winner = selected.config.model_copy(
            update={"name": f"evidence-judge-holdout-{lock.selected_candidate}"}
        )
        baseline_result = runner.run(
            baseline,
            DATASET_PATH,
            ExperimentExecutionOptions(
                offline_only=False,
                confirm_external_calls=True,
            ),
            cases_override=self.split.holdout,
        )
        winner_result = runner.run(
            winner,
            DATASET_PATH,
            ExperimentExecutionOptions(
                offline_only=False,
                confirm_external_calls=True,
                confirm_external_judge_calls=lock.judge_provider == "openai",
            ),
            cases_override=self.split.holdout,
        )
        lock = self.session.get(EvidenceBenchmarkLockRecord, self.split.identity)
        lock.holdout_baseline_run_id = baseline_result.run_id
        lock.holdout_winner_run_id = winner_result.run_id
        lock.holdout_completed_at = datetime.now(UTC)
        self.session.commit()
        return self.status(include_cases=True)

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        lock = self.session.get(EvidenceBenchmarkLockRecord, self.split.identity)
        if lock is None:
            return {
                "split_identity": self.split.identity,
                "selected_candidate": None,
                "holdout_started": False,
                "holdout_completed": False,
            }
        calibration = {
            code: experiment_to_dict(
                self.session,
                self.session.get(ExperimentRunRecord, run_id),
                include_cases=include_cases,
            )
            for code, run_id in lock.calibration_metrics["run_ids"].items()
        }
        payload: dict[str, Any] = {
            "split_identity": lock.split_identity,
            "selected_candidate": lock.selected_candidate,
            "selected_configuration_hash": lock.selected_configuration_hash,
            "selection_reason": lock.selection_reason,
            "locked_at": lock.locked_at,
            "calibration": calibration,
            "holdout_started": lock.holdout_started_at is not None,
            "holdout_completed": lock.holdout_completed_at is not None,
            "judge_cache": {
                provider: self._judge_cache_usage(provider)
                for provider in ("qwen", "openai")
            },
        }
        if lock.holdout_completed_at is not None:
            payload["holdout"] = {
                "A": experiment_to_dict(
                    self.session,
                    self.session.get(ExperimentRunRecord, lock.holdout_baseline_run_id),
                    include_cases=include_cases,
                ),
                lock.selected_candidate: experiment_to_dict(
                    self.session,
                    self.session.get(ExperimentRunRecord, lock.holdout_winner_run_id),
                    include_cases=include_cases,
                ),
            }
        return payload

    def _judge_cache_usage(self, provider: str) -> dict[str, int]:
        model = self.settings.local_judge_model if provider == "qwen" else "gpt-5.6-luna"
        judge_version = "4" if provider == "qwen" else "1"
        count, prompt_tokens, completion_tokens, prompt_hashes = self.session.execute(
            select(
                func.count(),
                func.coalesce(func.sum(AnswerabilityGateCacheRecord.prompt_tokens), 0),
                func.coalesce(func.sum(AnswerabilityGateCacheRecord.completion_tokens), 0),
                func.count(func.distinct(AnswerabilityGateCacheRecord.prompt_render_sha256)),
            ).where(
                AnswerabilityGateCacheRecord.judge_provider == provider,
                AnswerabilityGateCacheRecord.judge_model == model,
                AnswerabilityGateCacheRecord.judge_version == judge_version,
            )
        ).one()
        return {
            "recorded_attempts": self._recorded_judge_attempts(
                provider, model, judge_version
            ),
            "cached_results": int(count),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "prompt_render_hashes": int(prompt_hashes),
        }

    def _recorded_judge_attempts(
        self, provider: str, model: str, judge_version: str
    ) -> int:
        runs = self.session.scalars(select(ExperimentRunRecord)).all()
        return sum(
            (
                (run.external_judge_calls or 0)
                if provider == "openai"
                else (run.local_judge_calls or 0)
            )
            for run in runs
            if run.config.gate_config
            and run.config.gate_config.get("judge_provider") == provider
            and run.config.gate_config.get("judge_model") == model
            and run.config.gate_config.get("judge_version") == judge_version
        )

    def _select(self, run_ids: dict[str, str]) -> tuple[str, str]:
        runs = {
            code: self.session.get(ExperimentRunRecord, run_id)
            for code, run_id in run_ids.items()
        }
        eligible = {
            code: run
            for code, run in runs.items()
            if run.status == "completed"
            and run.aggregate_metrics.get("acl_safety") == 1.0
            and run.aggregate_metrics.get("version_accuracy") == 1.0
            and run.aggregate_metrics.get("prompt_injection_boundary") == 1.0
            and run.aggregate_metrics.get("unauthorized_evidence_selection") == 0
        }
        if not eligible:
            raise ValueError("no candidate satisfies the benchmark hard constraints")

        def rank(item: tuple[str, ExperimentRunRecord]) -> tuple[float, ...]:
            code, run = item
            metrics = run.aggregate_metrics
            desired_loss = int(metrics.get("multi_document_answer_success") != 1.0) + int(
                metrics.get("exact_identifier_answer_success") != 1.0
            )
            provider_penalty = 1 if code.startswith("C") else 0
            pruning_penalty = 0 if code.endswith("2") else 1
            return (
                float(metrics.get("unsupported_answer_count") or 0),
                float(metrics.get("incorrect_abstention_count") or 0),
                float(desired_loss),
                -float(metrics.get("answerability_f1") or 0),
                -float(metrics.get("claim_support_rate") or 0),
                float(metrics.get("judge_error_count") or 0),
                provider_penalty,
                pruning_penalty,
                float(run.total_latency_ms or float("inf")),
            )

        winner, run = min(eligible.items(), key=rank)
        metrics = run.aggregate_metrics
        reason = (
            f"Calibration-only selection minimized unsupported answers to "
            f"{metrics['unsupported_answer_count']} with "
            f"{metrics['incorrect_abstention_count']} incorrect abstentions and "
            f"answerability F1 {metrics['answerability_f1']}; all hard constraints passed."
        )
        return winner, reason


def calibration_original_nine_cases(
    cases: tuple[EvaluationCase, ...],
) -> tuple[EvaluationCase, ...]:
    return tuple(case for case in cases if case.case_id in ORIGINAL_NINE)
