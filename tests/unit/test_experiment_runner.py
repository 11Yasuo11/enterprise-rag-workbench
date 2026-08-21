import hashlib
import json

import pytest

from rag_workbench.evaluation.eval_loop import (
    PrimaryError,
    classify_failure,
    compare_results,
    normalize_trace,
)
from rag_workbench.evaluation.experiment_runner import (
    execute_experiment,
    load_experiment_config,
    run_cases_checkpointed,
)


def _case(**overrides):
    case = {
        "case_id": "c1",
        "question": "question",
        "gold_answer": None,
        "gold_document_ids": ["gold-doc"],
        "gold_chunk_ids": [],
        "answerable": True,
        "category": "single_document",
        "difficulty": "unknown",
        "requires_acl": None,
        "requires_tenant_filter": None,
        "requires_version_filter": None,
        "requires_multi_hop": None,
        "required_facts": [],
        "raw_metadata": {},
    }
    case.update(overrides)
    return case


def _raw(**overrides):
    row = {
        "case_id": "c1",
        "candidate_ids": ["gold-chunk"],
        "retrieved_candidate_document_ids": ["gold-doc"],
        "reranked_ids": ["gold-chunk"],
        "reranked_document_ids": ["gold-doc"],
        "top_k_ids": ["gold-chunk"],
        "topk_document_ids": ["gold-doc"],
        "final_answer": "answer",
        "abstained": False,
        "answer_correct": True,
        "citation_ids": ["gold-chunk"],
        "stage_trace": {
            "filters": {
                "input_ids": ["gold-chunk", "denied"],
                "input_document_ids": ["gold-doc", "denied-doc"],
                "output_ids": ["gold-chunk"],
                "output_document_ids": ["gold-doc"],
                "removed_ids": ["denied"],
                "reasons": {"denied": ["ACL_DENIED"]},
            }
        },
    }
    row.update(overrides)
    return row


class FakeRunner:
    def __init__(self, rows=None, error=False):
        self.rows = rows or {}
        self.error = error
        self.calls = []

    def run_rag_case(self, case, config):
        del config
        self.calls.append(case["case_id"])
        if self.error:
            raise RuntimeError("boom")
        return self.rows.get(case["case_id"], _raw(case_id=case["case_id"]))


def test_runner_adapter(tmp_path) -> None:
    runner = FakeRunner()
    rows = run_cases_checkpointed(
        cases=[_case()], runner=runner, config={}, path=tmp_path / "raw.jsonl", resume=False
    )
    assert rows[0]["case_id"] == "c1"
    assert runner.calls == ["c1"]


def test_trace_capture() -> None:
    trace = normalize_trace(_case(), _raw())
    assert trace["retrieved_candidate_ids"] == ["gold-chunk"]
    assert trace["reranked_ids"] == ["gold-chunk"]
    assert trace["reranked_top_ids"] == ["gold-chunk"]


def test_document_stage_trace() -> None:
    trace = normalize_trace(_case(), _raw())
    evidence = trace["error_evidence"]
    assert evidence["present_before_filters"] is True
    assert evidence["present_after_filters"] is True
    assert evidence["present_in_candidates"] is True
    assert evidence["present_in_top_k"] is True


def test_filter_failure_attribution() -> None:
    raw = _raw(
        answer_correct=False,
        retrieved_candidate_document_ids=[],
        topk_document_ids=[],
        stage_trace={
            "filters": {
                "input_document_ids": ["gold-doc"],
                "output_document_ids": ["other"],
                "reasons": {"gold-chunk": ["ACL_DENIED"]},
            }
        },
    )
    trace = normalize_trace(_case(), raw)
    assert classify_failure(_case(), trace) == PrimaryError.FILTER_FAILURE
    assert trace["error_evidence"]["filter_reasons"]


def test_retrieval_failure_attribution() -> None:
    raw = _raw(
        answer_correct=False,
        retrieved_candidate_document_ids=["other"],
        topk_document_ids=["other"],
    )
    trace = normalize_trace(_case(), raw)
    assert trace["primary_error"] == PrimaryError.RETRIEVAL_FAILURE
    assert trace["error_evidence"]["missing_from_candidates"] == ["gold-doc"]


def test_ranking_failure_attribution() -> None:
    raw = _raw(answer_correct=False, topk_document_ids=["other"])
    trace = normalize_trace(_case(), raw)
    assert trace["primary_error"] == PrimaryError.RANKING_FAILURE
    assert trace["error_evidence"]["present_in_candidates"] is True


def test_unresolved_when_trace_missing() -> None:
    trace = normalize_trace(
        _case(),
        {
            "case_id": "c1",
            "abstained": True,
            "answer_correct": None,
            "execution_error": "PAID_SEMANTIC_EVAL_REQUIRED",
        },
    )
    assert trace["primary_error"] == PrimaryError.UNRESOLVED
    assert trace["unresolved_reason"] == "PAID_SEMANTIC_EVAL_REQUIRED"


def test_rag_execution_failure_is_checkpointed(tmp_path) -> None:
    path = tmp_path / "raw.jsonl"
    rows = run_cases_checkpointed(
        cases=[_case()], runner=FakeRunner(error=True), config={}, path=path, resume=False
    )
    assert rows[0]["execution_error"].startswith("RAG_EXECUTION_FAILURE")
    assert json.loads(path.read_text())["case_id"] == "c1"


def test_resume_after_interrupted_run(tmp_path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(_raw()) + "\n", encoding="utf-8")
    runner = FakeRunner()
    cases = [_case(), _case(case_id="c2")]
    rows = run_cases_checkpointed(cases=cases, runner=runner, config={}, path=path, resume=True)
    assert [row["case_id"] for row in rows] == ["c1", "c2"]
    assert runner.calls == ["c2"]


def test_partial_trace_with_gold_document_only() -> None:
    trace = normalize_trace(
        _case(),
        _raw(
            candidate_ids=[],
            retrieved_candidate_document_ids=["gold-doc"],
            reranked_ids=[],
            top_k_ids=[],
            topk_document_ids=[],
            answer_correct=False,
        ),
    )
    assert trace["retrieval_hit"] is True
    assert trace["topk_hit"] is None


def test_empty_retrieval() -> None:
    trace = normalize_trace(
        _case(),
        _raw(
            candidate_ids=[],
            retrieved_candidate_document_ids=["other"],
            reranked_ids=[],
            reranked_document_ids=[],
            top_k_ids=[],
            topk_document_ids=["other"],
            answer_correct=False,
            citation_ids=[],
        ),
    )
    assert trace["primary_error"] == PrimaryError.RETRIEVAL_FAILURE


def test_abstention_with_missing_semantic_cache_is_unresolved() -> None:
    trace = normalize_trace(
        _case(),
        _raw(
            final_answer=None,
            abstained=True,
            answer_correct=None,
            execution_error="PAID_SEMANTIC_EVAL_REQUIRED",
        ),
    )
    assert trace["primary_error"] == PrimaryError.UNRESOLVED


def test_resume_requires_complete_case_set(tmp_path) -> None:
    path = tmp_path / "raw.jsonl"
    runner = FakeRunner()
    run_cases_checkpointed(cases=[_case()], runner=runner, config={}, path=path, resume=False)
    assert path.exists()


def test_runner_protocol_failure_does_not_raise(tmp_path) -> None:
    rows = run_cases_checkpointed(
        cases=[_case()],
        runner=FakeRunner(error=True),
        config={},
        path=tmp_path / "raw.jsonl",
        resume=False,
    )
    assert rows[0]["abstained"] is True


def test_invalid_resume_json_raises(tmp_path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        run_cases_checkpointed(
            cases=[_case()], runner=FakeRunner(), config={}, path=path, resume=True
        )


def test_dataset_hash_guard(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
experiment_id: test
identity: {}
dataset: {}
retrieval: {}
reranking: {}
generation: {}
semantic_judge:
  enabled: true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="semantic_judge"):
        load_experiment_config(config)


def test_baseline_comparison() -> None:
    passing = normalize_trace(_case(), _raw())
    baseline = {
        "dataset_hash": "same",
        "metrics": {"precision": 1.0},
        "traces": [passing],
        "failure_census": {"counts": {"PASS": 1}},
        "slice_analysis": {"slices": {}},
    }
    result = compare_results(baseline, baseline, {"promotion_gate": {}}).payload
    assert result["decision"] == "PROMOTE"
    assert result["failure_comparison"]["PASS"]["delta"] == 0


def test_gate_execution() -> None:
    passing = normalize_trace(_case(), _raw())
    baseline = {
        "dataset_hash": "same",
        "metrics": {"precision": 1.0},
        "traces": [passing],
        "failure_census": {"counts": {}},
        "slice_analysis": {"slices": {}},
    }
    candidate = {**baseline, "metrics": {"precision": 0.9}}
    result = compare_results(
        baseline,
        candidate,
        {"promotion_gate": {"precision": {"min": 1.0}}},
    ).payload
    assert result["decision"] == "REJECT"


def test_experiment_runner(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "frozen.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "query_id": "c1",
                "question": "question",
                "category": "single_document",
                "expected_answerable": True,
                "should_abstain": False,
                "required_document_ids": ["gold-doc"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    historical = tmp_path / "historical.jsonl"
    historical.write_text("", encoding="utf-8")
    config = tmp_path / "candidate.yaml"
    config.write_text(
        f"""
experiment_id: fake-experiment
identity:
  phase5kr_results: {historical}
dataset:
  path: {dataset}
  sha256: {dataset_hash}
retrieval: {{}}
reranking: {{}}
generation: {{}}
semantic_judge:
  enabled: false
""",
        encoding="utf-8",
    )
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_trace = normalize_trace(_case(), _raw())
    (baseline_dir / "results.json").write_text(
        json.dumps(
            {
                "dataset_hash": dataset_hash,
                "metrics": {
                    "precision": 1.0,
                    "unsupported_answers": 0,
                    "citation_validity": 1.0,
                    "recall": 1.0,
                    "acl_accuracy": 1.0,
                    "tenant_accuracy": 1.0,
                    "unanswerable_accuracy": 1.0,
                },
                "traces": [baseline_trace],
                "failure_census": {"counts": {"PASS": 1}},
                "slice_analysis": {"slices": {}},
            }
        ),
        encoding="utf-8",
    )
    gate = tmp_path / "gate.yaml"
    gate.write_text("promotion_gate: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        "rag_workbench.evaluation.experiment_runner.Phase5CachedRunner",
        lambda session, config, baseline_rows: FakeRunner(),
    )
    output, regression = execute_experiment(
        config_path=config,
        baseline_path=baseline_dir,
        output_root=tmp_path / "artifacts",
        gate_path=gate,
        session=None,
        resume=False,
    )
    assert regression["decision"] == "PROMOTE"
    assert (output / "failure_evidence.jsonl").exists()
    assert (output / "report.md").exists()
