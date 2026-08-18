from datetime import UTC, datetime

import pytest

from rag_workbench.db.models import EvidenceBenchmarkLockRecord, ExperimentRunRecord
from rag_workbench.experiments.evidence_benchmark import (
    SPLIT_IDENTITY,
    EvidenceJudgeBenchmark,
    benchmark_candidates,
)
from rag_workbench.experiments.runner import ExperimentRunner


def test_exact_five_candidates_share_judge_decisions_by_provider() -> None:
    candidates = {item.code: item.config for item in benchmark_candidates()}
    assert tuple(candidates) == ("A", "B1", "B2", "C1", "C2")
    assert candidates["A"].answerability_gate is None
    assert candidates["B1"].answerability_gate.judge_provider == "qwen"
    assert candidates["B2"].answerability_gate.model_copy(
        update={"supporting_context_only": False}
    ) == candidates["B1"].answerability_gate
    assert candidates["C1"].answerability_gate.judge_provider == "openai"
    assert candidates["C2"].answerability_gate.model_copy(
        update={"supporting_context_only": False}
    ) == candidates["C1"].answerability_gate
    assert candidates["B1"].index_identity == candidates["C1"].index_identity


def test_holdout_requires_lock_and_is_one_shot(db_session) -> None:
    db_session.query(EvidenceBenchmarkLockRecord).filter_by(
        split_identity=SPLIT_IDENTITY
    ).delete()
    db_session.flush()
    benchmark = EvidenceJudgeBenchmark(db_session)
    with pytest.raises(ValueError, match="locked"):
        benchmark.run_holdout()
    selected = {item.code: item.config for item in benchmark.candidates.values()}["A"]
    db_session.add(
        EvidenceBenchmarkLockRecord(
            split_identity=SPLIT_IDENTITY,
            selected_candidate="A",
            selected_configuration_hash=selected.experiment_config_hash,
            threshold=0.28,
            supporting_context_only=False,
            calibration_metrics={"run_ids": {}, "candidates": {}},
            selection_reason="test",
            holdout_started_at=datetime.now(UTC),
        )
    )
    db_session.flush()
    with pytest.raises(ValueError, match="one-shot"):
        benchmark.run_holdout()
    with pytest.raises(ValueError, match="immutable"):
        benchmark.run_calibration()


def test_local_call_accounting_is_cumulative_across_judge_versions(db_session) -> None:
    runner = ExperimentRunner(db_session)
    qwen_v4 = {item.code: item.config for item in benchmark_candidates()}["B1"]
    qwen_v1 = qwen_v4.model_copy(
        update={
            "name": "diagnostic-v1",
            "answerability_gate": qwen_v4.answerability_gate.model_copy(
                update={"judge_version": "1"}
            ),
        }
    )
    for config, calls in ((qwen_v1, 37), (qwen_v4, 41)):
        db_session.add(
            ExperimentRunRecord(
                config_id=runner._config_record(config).id,
                name=config.name,
                status="failed",
                local_judge_calls=calls,
            )
        )
    db_session.flush()

    assert runner._recorded_judge_attempts("qwen", "qwen3:8b") == 78
