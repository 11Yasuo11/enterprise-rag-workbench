from __future__ import annotations

import hashlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_COVERAGE_PROMPT_VERSION,
    evidence_coverage_template_hash,
)
from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import (
    ExperimentRunRecord,
    MultiDocumentBenchmarkRecord,
)
from rag_workbench.evaluation.datasets import load_evaluation_dataset
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

DATASET_ID = "acmeai-multidoc-eval-v2"
DATASET_PATH = Path("data/eval/acmeai_multidoc_eval_v2.json")
DATASET_HASH = "a5d652ab76ee68f5adce4dd420a65bd7cc285f2b9fa4097aa070435ddb58e6b9"
SPLIT_SEED = 20260817
SPLIT_IDENTITY = "32b1bf1a29dc1fd612931b372810f8a186398c529c76054c18d58d4549e0bd7c"


def multidoc_candidates(settings: Settings | None = None) -> dict[str, ExperimentConfig]:
    settings = settings or get_settings()
    common = {
        "identity": CorpusDatasetIdentity(
            corpus_version="acmeai-v1", evaluation_dataset_version=DATASET_ID
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
    gate = {
        "judge_provider": "openai",
        "judge_model": "gpt-5.6-luna",
        "judge_version": "1",
        "supporting_context_only": True,
    }
    return {
        "C2": ExperimentConfig(
            name="multidoc-v2-calibration-C2",
            answerability_gate=AnswerabilityGateConfig(
                **gate, prompt_version="evidence-sufficiency-v1"
            ),
            **common,
        ),
        "D": ExperimentConfig(
            name="multidoc-v2-calibration-D",
            answerability_gate=AnswerabilityGateConfig(
                **gate, prompt_version=EVIDENCE_COVERAGE_PROMPT_VERSION
            ),
            **common,
        ),
    }


class MultiDocumentEvidenceBenchmark:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.dataset = load_evaluation_dataset(DATASET_PATH)
        actual_hash = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
        if self.dataset.dataset_version != DATASET_ID or actual_hash != DATASET_HASH:
            raise ValueError("multi-document dataset identity changed; execution is blocked")
        self.split = deterministic_stratified_split(
            self.dataset, seed=SPLIT_SEED, holdout_size=20
        )
        if self.split.identity != SPLIT_IDENTITY:
            raise ValueError("multi-document split identity changed; execution is blocked")
        self.candidates = multidoc_candidates(self.settings)

    def initialize(self) -> MultiDocumentBenchmarkRecord:
        existing = self.session.get(MultiDocumentBenchmarkRecord, DATASET_ID)
        if existing is not None:
            self._verify_frozen(existing)
            return existing
        record = MultiDocumentBenchmarkRecord(
            dataset_id=DATASET_ID,
            dataset_hash=DATASET_HASH,
            split_identity=SPLIT_IDENTITY,
            split_seed=SPLIT_SEED,
            calibration_case_ids=[case.case_id for case in self.split.calibration],
            holdout_case_ids=[case.case_id for case in self.split.holdout],
            control_configuration_hash=self.candidates["C2"].experiment_config_hash,
            candidate_configuration_hash=self.candidates["D"].experiment_config_hash,
            candidate_prompt_hash=evidence_coverage_template_hash(),
        )
        self.session.add(record)
        self.session.commit()
        return record

    def run_calibration(self) -> dict[str, Any]:
        record = self.initialize()
        if record.locked_at is not None:
            raise ValueError("multi-document calibration is already locked and immutable")
        runner = ExperimentRunner(self.session, settings=self.settings)
        run_ids: dict[str, str] = {}
        for code in ("C2", "D"):
            existing_id = (
                record.calibration_control_run_id
                if code == "C2"
                else record.calibration_candidate_run_id
            )
            if existing_id:
                existing = self.session.get(ExperimentRunRecord, existing_id)
                if existing is None or existing.status != "completed":
                    raise ValueError(f"persisted calibration run {code} is not complete")
                run_ids[code] = existing_id
                continue
            result = runner.run(
                self.candidates[code],
                DATASET_PATH,
                ExperimentExecutionOptions(
                    offline_only=False,
                    confirm_external_calls=True,
                    confirm_external_judge_calls=True,
                ),
                cases_override=self.split.calibration,
            )
            if result.run_id is None:
                raise RuntimeError(f"calibration run {code} was not persisted")
            record = self.session.get(MultiDocumentBenchmarkRecord, DATASET_ID)
            if code == "C2":
                record.calibration_control_run_id = result.run_id
            else:
                record.calibration_candidate_run_id = result.run_id
            self.session.commit()
            run_ids[code] = result.run_id

        winner, reason = self._select(run_ids)
        selected = self.candidates[winner]
        record = self.session.get(MultiDocumentBenchmarkRecord, DATASET_ID)
        record.selected_candidate = winner
        record.selected_configuration_hash = selected.experiment_config_hash
        record.calibration_metrics = {
            code: self.session.get(ExperimentRunRecord, run_id).aggregate_metrics
            for code, run_id in run_ids.items()
        }
        record.selection_reason = reason
        record.locked_at = datetime.now(UTC)
        self.session.commit()
        return self.status(include_cases=True)

    def run_holdout(self) -> dict[str, Any]:
        record = self.session.get(MultiDocumentBenchmarkRecord, DATASET_ID)
        if record is None or record.locked_at is None or record.selected_candidate is None:
            raise ValueError("calibration selection must be locked before holdout")
        self._verify_frozen(record)
        if record.holdout_started_at is not None:
            raise ValueError("new multi-document holdout is one-shot and already started")
        selected = self.candidates[record.selected_candidate]
        if selected.experiment_config_hash != record.selected_configuration_hash:
            raise ValueError("selected configuration no longer matches the persisted lock")

        record.holdout_started_at = datetime.now(UTC)
        self.session.commit()
        runner = ExperimentRunner(self.session, settings=self.settings)
        control = self.candidates["C2"].model_copy(
            update={"name": "multidoc-v2-holdout-C2"}
        )
        control_result = runner.run(
            control,
            DATASET_PATH,
            ExperimentExecutionOptions(
                offline_only=False,
                confirm_external_calls=True,
                confirm_external_judge_calls=True,
            ),
            cases_override=self.split.holdout,
        )
        if record.selected_candidate == "C2":
            selected_result = control_result
        else:
            winner = selected.model_copy(update={"name": "multidoc-v2-holdout-D"})
            selected_result = runner.run(
                winner,
                DATASET_PATH,
                ExperimentExecutionOptions(
                    offline_only=False,
                    confirm_external_calls=True,
                    confirm_external_judge_calls=True,
                ),
                cases_override=self.split.holdout,
            )
        record = self.session.get(MultiDocumentBenchmarkRecord, DATASET_ID)
        record.holdout_control_run_id = control_result.run_id
        record.holdout_selected_run_id = selected_result.run_id
        record.holdout_completed_at = datetime.now(UTC)
        self.session.commit()
        return self.status(include_cases=True)

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(MultiDocumentBenchmarkRecord, DATASET_ID)
        base: dict[str, Any] = {
            "dataset_id": DATASET_ID,
            "dataset_hash": DATASET_HASH,
            "case_count": len(self.dataset),
            "category_distribution": dict(
                sorted(Counter(case.category for case in self.dataset).items())
            ),
            "split_seed": SPLIT_SEED,
            "split_identity": SPLIT_IDENTITY,
            "calibration_count": len(self.split.calibration),
            "holdout_count": len(self.split.holdout),
            "candidate_prompt_hash": evidence_coverage_template_hash(),
            "selected_candidate": None,
            "holdout_started": False,
            "holdout_completed": False,
        }
        if record is None:
            return base
        base.update(
            {
                "frozen_at": record.frozen_at,
                "selected_candidate": record.selected_candidate,
                "selection_reason": record.selection_reason,
                "locked_at": record.locked_at,
                "holdout_started": record.holdout_started_at is not None,
                "holdout_completed": record.holdout_completed_at is not None,
            }
        )
        if record.calibration_control_run_id and record.calibration_candidate_run_id:
            base["calibration"] = {
                "C2": experiment_to_dict(
                    self.session,
                    self.session.get(
                        ExperimentRunRecord, record.calibration_control_run_id
                    ),
                    include_cases=include_cases,
                ),
                "D": experiment_to_dict(
                    self.session,
                    self.session.get(
                        ExperimentRunRecord, record.calibration_candidate_run_id
                    ),
                    include_cases=include_cases,
                ),
            }
        if record.holdout_completed_at is not None:
            control = experiment_to_dict(
                self.session,
                self.session.get(ExperimentRunRecord, record.holdout_control_run_id),
                include_cases=include_cases,
            )
            selected = experiment_to_dict(
                self.session,
                self.session.get(ExperimentRunRecord, record.holdout_selected_run_id),
                include_cases=include_cases,
            )
            base["holdout"] = {"C2": control}
            if record.selected_candidate == "D":
                base["holdout"]["D"] = selected
        return base

    def _verify_frozen(self, record: MultiDocumentBenchmarkRecord) -> None:
        expected = (
            record.dataset_hash == DATASET_HASH
            and record.split_identity == SPLIT_IDENTITY
            and record.split_seed == SPLIT_SEED
            and record.control_configuration_hash
            == self.candidates["C2"].experiment_config_hash
            and record.candidate_configuration_hash
            == self.candidates["D"].experiment_config_hash
            and record.candidate_prompt_hash == evidence_coverage_template_hash()
        )
        if not expected:
            raise ValueError("frozen multi-document benchmark identity changed")

    def _select(self, run_ids: dict[str, str]) -> tuple[str, str]:
        runs = {
            code: self.session.get(ExperimentRunRecord, run_id)
            for code, run_id in run_ids.items()
        }
        for code, run in runs.items():
            if run is None or run.status != "completed":
                raise ValueError(f"calibration run {code} is incomplete")
        control = runs["C2"].aggregate_metrics
        candidate = runs["D"].aggregate_metrics
        hard_constraints = (
            candidate.get("unsupported_answer_count") == 0
            and candidate.get("acl_safety") == 1.0
            and candidate.get("version_accuracy") == 1.0
            and candidate.get("prompt_injection_boundary") == 1.0
            and candidate.get("unauthorized_evidence_selection") == 0
            and candidate.get("judge_error_count") == 0
        )
        coverage_gain = float(
            candidate.get("all_required_evidence_coverage_rate") or 0
        ) - float(control.get("all_required_evidence_coverage_rate") or 0)
        abstention_gain = float(
            control.get("multi_document_false_abstention_rate") or 0
        ) - float(candidate.get("multi_document_false_abstention_rate") or 0)
        no_support_regression = float(candidate.get("required_evidence_precision") or 0) >= float(
            control.get("required_evidence_precision") or 0
        )
        if hard_constraints and no_support_regression and (
            coverage_gain >= 0.05 or abstention_gain > 0
        ):
            return (
                "D",
                "Calibration selected D: requirement-level coverage improved the primary "
                f"targets (coverage delta {coverage_gain:+.6f}, false-abstention delta "
                f"{-abstention_gain:+.6f}) with zero unsupported answers and all hard "
                "constraints satisfied.",
            )
        return (
            "C2",
            "Calibration retained frozen C2 because D did not materially improve required "
            "coverage or false abstention without violating a reliability guardrail.",
        )
