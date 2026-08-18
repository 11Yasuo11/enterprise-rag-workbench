from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.api.app import app
from rag_workbench.config import Settings
from rag_workbench.db.models import Chunk, ExperimentCaseResultRecord, ExperimentRunRecord
from rag_workbench.experiments.comparison import compare_experiment_runs
from rag_workbench.experiments.configs import load_experiment_config
from rag_workbench.experiments.runner import (
    ExperimentExecutionOptions,
    ExperimentRunner,
    plan_experiment,
)
from rag_workbench.providers.llm.base import GenerationRequest, GenerationResult

pytestmark = pytest.mark.integration


def test_experiment_runner_persists_cases_metrics_and_reuses_index(db_session: Session) -> None:
    baseline = load_experiment_config(Path("configs/baseline-hashing-eval-v1.yaml"))
    candidate = load_experiment_config(Path("configs/hashing-top10-eval-v1.yaml"))
    runner = ExperimentRunner(db_session)
    first = runner.run(baseline, Path("data/eval/eval_v1.json"))
    second = runner.run(candidate, Path("data/eval/eval_v1.json"))
    assert first.config.index_identity == second.config.index_identity
    assert first.report.case_count == 100
    assert first.report.category_metrics["multi_document"]["recall_at_k"] < 1.0
    assert first.report.category_metrics["access_control"]["security_success_rate"] == 1.0
    assert first.report.category_metrics["versioning"]["version_accuracy"] == 1.0
    assert db_session.scalar(select(func.count()).select_from(ExperimentRunRecord)) == 2
    assert (
        db_session.scalar(select(func.count()).select_from(ExperimentCaseResultRecord)) == 200
    )
    comparison = compare_experiment_runs(db_session, first.run_id, second.run_id)
    assert comparison["metrics"]["recall_at_k"]["delta"] == pytest.approx(0.01)
    assert comparison["candidate"]["config"]["retrieval"]["top_k"] == 10
    assert db_session.scalar(select(func.count()).select_from(Chunk)) == 24
    first_run = db_session.get(ExperimentRunRecord, first.run_id)
    second_run = db_session.get(ExperimentRunRecord, second.run_id)
    assert first_run.query_embedding_cache_misses > 0
    assert second_run.query_embedding_cache_misses == 0
    assert second_run.query_embedding_cache_hits == 100
    assert second_run.query_embedding_latency_ms == 0
    assert second_run.vector_search_latency_ms is not None
    assert second_run.acl_filter_latency_ms is not None
    assert second_run.context_construction_latency_ms is not None
    persisted_case = db_session.scalar(
        select(ExperimentCaseResultRecord).where(
            ExperimentCaseResultRecord.experiment_run_id == second.run_id
        )
    )
    assert persisted_case.query_embedding_cache_hit
    assert persisted_case.embedding_cache_lookup_latency_ms is not None
    assert persisted_case.vector_search_latency_ms is not None


class FailingGenerationProvider:
    provider_name = "failing-test"
    model_name = "failing-test-v1"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise RuntimeError("deliberate case failure")


def test_failed_case_is_persisted_and_api_returns_experiment(
    db_session: Session, monkeypatch
) -> None:
    config = load_experiment_config(Path("configs/default.yaml"))
    runner = ExperimentRunner(db_session)
    monkeypatch.setattr(
        runner,
        "_generation_provider",
        lambda config, options: FailingGenerationProvider(),
    )
    result = runner.run(
        config,
        Path("data/eval/initial.json"),
        ExperimentExecutionOptions(max_cases=1),
    )
    persisted = db_session.get(ExperimentRunRecord, result.run_id)
    assert persisted.status == "completed_with_errors"
    failed = db_session.scalar(
        select(ExperimentCaseResultRecord).where(
            ExperimentCaseResultRecord.experiment_run_id == result.run_id
        )
    )
    assert failed.failure_type == "UNKNOWN"
    assert failed.failure_types == ["UNKNOWN"]
    assert "deliberate case failure" in failed.error

    with TestClient(app) as client:
        listing = client.get("/experiments")
        detail = client.get(f"/experiments/{result.run_id}")
        created = client.post(
            "/experiments/run",
            json={
                "config": config.model_dump(mode="json"),
                "dataset": "initial.json",
                "options": {
                    "offline_only": True,
                    "max_cases": 1,
                    "max_configs": 20,
                    "confirm_external_calls": False,
                },
            },
        )
    assert listing.status_code == 200
    assert detail.status_code == 200
    assert created.status_code == 200
    assert created.json()["status"] == "completed"
    assert detail.json()["cases"][0]["failure_type"] == "UNKNOWN"


def test_semantic_plan_reports_calls_and_external_gate_is_fail_closed(
    db_session: Session,
) -> None:
    config = load_experiment_config(Path("configs/semantic-embedding-v1.yaml"))
    plan = plan_experiment(db_session, config, case_count=100)
    assert plan.document_embeddings_required == 24
    assert plan.document_embedding_calls == 16
    assert plan.query_embedding_calls == 100
    assert plan.expected_external_embedding_calls == 116

    runner = ExperimentRunner(
        db_session,
        settings=Settings(
            _env_file=None,
            embedding_api_key="mock-key",
            allow_external_calls=False,
            max_external_embedding_calls=116,
        ),
    )
    with pytest.raises(ValueError, match="ALLOW_EXTERNAL_CALLS"):
        runner.run(
            config,
            Path("data/eval/eval_v1.json"),
            ExperimentExecutionOptions(
                offline_only=False,
                max_cases=1,
                confirm_external_calls=True,
            ),
        )
