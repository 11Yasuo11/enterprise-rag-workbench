import json
from pathlib import Path

import pytest
import yaml

from rag_workbench.evaluation.auto_improvement import (
    FAILURE_STRATEGY_MAP,
    SUPPORTED_RANKING_PARAMETERS,
    UNSUPPORTED_RANKING_PARAMETERS,
    AutoImprovementController,
    Diagnosis,
    DisabledLLMPlanner,
    GridSearchPlanner,
    LLMPlanner,
    RuleBasedPlanner,
    _validate_run_config,
    change_one_thing,
    diagnose_failure,
    generate_ranking_candidates,
    rank_candidates,
)


def _trace(error="RANKING_FAILURE", candidate=True, top=False, category="three_document"):
    return {
        "case_id": "c1",
        "category": category,
        "primary_error": error,
        "reranked_top_document_ids": ["a", "a", "other", "other", "x"],
        "error_evidence": {
            "gold_document_ids": ["a", "b", "c"],
            "present_in_candidates": candidate,
            "present_in_top_k": top,
        },
        "stage_trace": {
            "reranker": {
                "items": [
                    {"document_id": "a", "rank": 1},
                    {"document_id": "a", "rank": 2},
                    {"document_id": "x", "rank": 3},
                    {"document_id": "x", "rank": 4},
                    {"document_id": "y", "rank": 5},
                    {"document_id": "b", "rank": 6},
                    {"document_id": "c", "rank": 10},
                ]
            }
        },
    }


def _diagnosis():
    return Diagnosis("RANKING_FAILURE", 23, True, "ranking_search", "KNOWN_PROBLEM", {})


def _baseline_config():
    return {
        "experiment_id": "baseline",
        "identity": {},
        "dataset": {},
        "retrieval": {},
        "reranking": {"top_k": 5, "selection_strategy": "pointwise"},
        "generation": {},
        "semantic_judge": {"enabled": False},
    }


def _run_config():
    return {
        "run_id": "test-run",
        "auto_improvement": {
            "enabled": True,
            "max_iterations": 1,
            "max_candidates_per_iteration": 3,
            "planner": {"mode": "rule_based"},
            "llm_planner": {"enabled": False},
            "semantic_judge": {"enabled": False},
            "paid_api_fallback": {"enabled": False},
            "candidate_search": {"mode": "grid"},
            "auto_promote": {"enabled": False},
        },
    }


def test_failure_strategy_mapping():
    assert FAILURE_STRATEGY_MAP["RANKING_FAILURE"] == "ranking_search"
    assert "GENERATOR_INCORRECT" not in FAILURE_STRATEGY_MAP


def test_known_problem_detection():
    diagnosis = diagnose_failure({"counts": {"RANKING_FAILURE": 1}}, [_trace()])
    assert diagnosis.known_problem is True
    assert diagnosis.strategy == "ranking_search"
    assert diagnosis.evidence["missing_gold_ce_rank_distribution"] == {6: 1, 10: 1}


def test_unknown_problem_stops_without_llm():
    diagnosis = diagnose_failure({"counts": {"UNRESOLVED": 2}}, [_trace("UNRESOLVED")])
    assert diagnosis.status == "NEEDS_HUMAN_OR_LLM_HYPOTHESIS"
    assert diagnosis.strategy is None


def test_ranking_search_space():
    assert SUPPORTED_RANKING_PARAMETERS["reranking.top_k"] == [7, 10]
    assert "reranking.diversity_weight" in UNSUPPORTED_RANKING_PARAMETERS


def test_candidate_generation():
    candidates = generate_ranking_candidates(_baseline_config(), _diagnosis(), 3)
    assert [row["candidate_id"] for row in candidates] == [
        "ranking-topk-7",
        "ranking-topk-10",
        "ranking-max-two-per-document",
    ]


def test_change_one_thing():
    baseline = _baseline_config()
    baseline["hypothesis"] = {"observation": "baseline instrumentation"}
    candidate = generate_ranking_candidates(baseline, _diagnosis(), 1)[0]
    candidate["experiment_id"] = "candidate"
    assert change_one_thing(baseline, candidate)
    candidate["retrieval"]["dense_k"] = 99
    assert not change_one_thing(baseline, candidate)


def test_target_failure_objective():
    rows = [
        {
            "candidate_id": "a",
            "target_failure_before": 23,
            "target_failure_after": 18,
            "critical_structural_regressions": 0,
            "structural_metrics": {
                "three_document_complete_coverage": 4,
                "gold_document_recall_at_5": 0.8,
                "gold_document_mrr": 0.5,
            },
        },
        {
            "candidate_id": "b",
            "target_failure_before": 23,
            "target_failure_after": 20,
            "critical_structural_regressions": 0,
            "structural_metrics": {
                "three_document_complete_coverage": 16,
                "gold_document_recall_at_5": 1.0,
                "gold_document_mrr": 1.0,
            },
        },
    ]
    ranked = rank_candidates(rows)
    assert ranked[0]["candidate_id"] == "a"


def test_candidate_ranking():
    rows = [
        {
            "candidate_id": candidate_id,
            "target_failure_before": 23,
            "target_failure_after": 18,
            "critical_structural_regressions": 0,
            "structural_metrics": {
                "three_document_complete_coverage": coverage,
                "gold_document_recall_at_5": 0.8,
                "gold_document_mrr": 0.5,
            },
        }
        for candidate_id, coverage in (("a", 2), ("b", 5))
    ]
    assert rank_candidates(rows)[0]["candidate_id"] == "b"


def test_llm_planner_disabled():
    assert LLMPlanner.enabled is False
    with pytest.raises(RuntimeError, match="LLM_PLANNER_DISABLED"):
        DisabledLLMPlanner().generate_hypotheses({}, {}, {})


def test_rule_based_grid_planner():
    planner = RuleBasedPlanner(GridSearchPlanner())
    candidates = planner.generate(_baseline_config(), _diagnosis(), 2)
    assert len(candidates) == 2


@pytest.mark.parametrize(
    "path",
    [
        ("semantic_judge", "enabled"),
        ("llm_planner", "enabled"),
        ("paid_api_fallback", "enabled"),
        ("auto_promote", "enabled"),
    ],
)
def test_zero_api_and_promotion_guards(path):
    config = _run_config()
    config["auto_improvement"][path[0]][path[1]] = True
    with pytest.raises(ValueError, match="ZERO_API_SAFETY_GUARD"):
        _validate_run_config(config, None)


def test_semantic_judge_disabled():
    assert _run_config()["auto_improvement"]["semantic_judge"]["enabled"] is False


def test_no_paid_api_fallback():
    assert _run_config()["auto_improvement"]["paid_api_fallback"]["enabled"] is False


def test_max_iterations_one():
    config = _run_config()
    config["auto_improvement"]["max_iterations"] = 2
    with pytest.raises(ValueError, match="max_iterations must be 1"):
        _validate_run_config(config, None)


def test_auto_promote_disabled():
    assert _run_config()["auto_improvement"]["auto_promote"]["enabled"] is False


def test_auto_improvement_end_to_end(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "experiment_config.yaml").write_text(
        yaml.safe_dump(_baseline_config()), encoding="utf-8"
    )
    baseline_results = {
        "experiment_id": "baseline",
        "dataset_hash": "hash",
        "metrics": {"accuracy": 0.5},
        "failure_census": {"counts": {"PASS": 0, "RANKING_FAILURE": 1}},
        "traces": [_trace()],
    }
    (baseline / "results.json").write_text(json.dumps(baseline_results), encoding="utf-8")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(_run_config()), encoding="utf-8")

    def fake_executor(**kwargs):
        config = yaml.safe_load(Path(kwargs["config_path"]).read_text())
        out = Path(kwargs["output_root"]) / config["experiment_id"]
        out.mkdir(parents=True)
        result = {
            "metrics": {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "gold_document_recall_at_5": 0.9,
                "gold_document_recall_at_10": 1.0,
                "gold_document_mrr": 0.6,
            },
            "failure_census": {"counts": {"RANKING_FAILURE": 0}},
            "traces": [_trace(error="PASS", top=True)],
        }
        (out / "results.json").write_text(json.dumps(result), encoding="utf-8")
        regression = {
            "decision": "MANUAL_REVIEW",
            "promotion_status": "NOT_ELIGIBLE",
            "reasons": ["semantic quality unavailable"],
        }
        (out / "regression.json").write_text(json.dumps(regression), encoding="utf-8")
        return out, regression

    output, result = AutoImprovementController(None, fake_executor).run(
        baseline=baseline,
        run_config_path=config_path,
        output_root=tmp_path / "auto",
        eval_output_root=tmp_path / "eval",
        gate_path=tmp_path / "gate.yaml",
    )
    assert result["status"] == "BEST_STRUCTURAL_CANDIDATE"
    assert result["promotion_eligibility"] == "MANUAL_REVIEW"
    assert result["external_api_calls"] == 0
    assert result["auto_promoted"] is False
    assert len(list((output / "candidates").glob("*.yaml"))) == 3
    assert (output / "report.md").exists()
