"""Zero-API hybrid auto-improvement controller.

Only deterministic diagnosis and bounded algorithmic search are enabled. The LLM planner is an
explicit disabled extension point; semantic cache misses are never filled by an external call.
"""

from __future__ import annotations

import copy
import json
import statistics
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml
from sqlalchemy.orm import Session

from rag_workbench.evaluation.experiment_runner import execute_experiment, write_json

FAILURE_STRATEGY_MAP = {
    "RETRIEVAL_FAILURE": "retrieval_search",
    "FILTER_FAILURE": "filter_diagnostics",
    "RANKING_FAILURE": "ranking_search",
    "JUDGE_FALSE_NEGATIVE": "judge_search",
    "JUDGE_FALSE_POSITIVE": "judge_search",
    "CITATION_FAILURE": "citation_search",
}
HUMAN_OR_LLM_FAILURES = frozenset(
    {"GENERATOR_INCORRECT", "GENERATOR_INCOMPLETE", "EVIDENCE_INSUFFICIENT", "UNRESOLVED", "OTHER"}
)
SUPPORTED_RANKING_PARAMETERS = {
    "reranking.top_k": [7, 10],
    "reranking.selection_strategy": ["max_two_chunks_per_document"],
}
UNSUPPORTED_RANKING_PARAMETERS = {
    "reranking.diversity_weight": "No production or experiment implementation exists.",
    "reranking.per_document_cap": (
        "Only the frozen max-two selector exists; arbitrary caps are unsupported."
    ),
    "reranker.model": "Model changes are outside the safe single-parameter structural search.",
}


class HypothesisPlanner(Protocol):
    def generate_hypotheses(
        self,
        failure_report: dict[str, Any],
        trace_summary: dict[str, Any],
        repository_context: dict[str, Any],
    ) -> list[dict[str, Any]]: ...


class CandidateSearch(Protocol):
    def generate(
        self,
        baseline_config: dict[str, Any],
        diagnosis: Diagnosis,
        max_candidates: int,
    ) -> list[dict[str, Any]]: ...


class LLMPlanner:
    """Future extension point that is deliberately non-operational."""

    enabled = False

    def generate_hypotheses(
        self,
        failure_report: dict[str, Any],
        trace_summary: dict[str, Any],
        repository_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        del failure_report, trace_summary, repository_context
        raise RuntimeError("LLM_PLANNER_DISABLED")


DisabledLLMPlanner = LLMPlanner


@dataclass(frozen=True)
class Diagnosis:
    largest_failure: str
    count: int
    known_problem: bool
    strategy: str | None
    status: str
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "largest_failure": self.largest_failure,
            "count": self.count,
            "known_problem": self.known_problem,
            "strategy": self.strategy,
            "status": self.status,
            "evidence": self.evidence,
        }


def analyze_ranking_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in traces if row.get("primary_error") == "RANKING_FAILURE"]
    missing_ranks: list[int] = []
    duplicate_counts: list[int] = []
    complete_at: dict[str, int] = {}
    three_complete_at: dict[str, int] = {}
    labeled = [row for row in traces if row.get("error_evidence", {}).get("gold_document_ids")]
    three = [row for row in traces if row.get("category") == "three_document"]
    for row in failures:
        gold = set(row["error_evidence"]["gold_document_ids"])
        items = row.get("stage_trace", {}).get("reranker", {}).get("items") or []
        best_rank: dict[str, int] = {}
        for item in items:
            best_rank.setdefault(str(item["document_id"]), int(item["rank"]))
        top_docs = list(row.get("reranked_top_document_ids") or [])
        missing_ranks.extend(best_rank[doc] for doc in gold - set(top_docs) if doc in best_rank)
        duplicate_counts.append(len(top_docs) - len(set(top_docs)))
    for k in (5, 7, 10):
        complete_at[str(k)] = sum(_complete_in_first_k(row, k) for row in labeled)
        three_complete_at[str(k)] = sum(_complete_in_first_k(row, k) for row in three)
    return {
        "ranking_failure_cases": len(failures),
        "missing_gold_ce_rank_distribution": dict(sorted(Counter(missing_ranks).items())),
        "missing_gold_ce_rank_min": min(missing_ranks) if missing_ranks else None,
        "missing_gold_ce_rank_median": statistics.median(missing_ranks) if missing_ranks else None,
        "missing_gold_ce_rank_max": max(missing_ranks) if missing_ranks else None,
        "same_document_duplicate_distribution_top5": dict(
            sorted(Counter(duplicate_counts).items())
        ),
        "complete_gold_coverage_by_top_k": complete_at,
        "three_document_complete_coverage_by_top_k": three_complete_at,
        "labeled_case_count": len(labeled),
        "three_document_case_count": len(three),
    }


def _complete_in_first_k(row: dict[str, Any], k: int) -> bool:
    gold = set(row.get("error_evidence", {}).get("gold_document_ids") or [])
    items = row.get("stage_trace", {}).get("reranker", {}).get("items") or []
    return bool(gold) and gold <= {str(item["document_id"]) for item in items[:k]}


def diagnose_failure(census: dict[str, Any], traces: list[dict[str, Any]]) -> Diagnosis:
    counts = {key: int(value) for key, value in census.get("counts", {}).items() if key != "PASS"}
    largest, count = max(counts.items(), key=lambda item: item[1], default=("NONE", 0))
    strategy = FAILURE_STRATEGY_MAP.get(largest)
    if largest == "RANKING_FAILURE":
        failures = [row for row in traces if row.get("primary_error") == largest]
        confirmed = bool(failures) and all(
            row.get("error_evidence", {}).get("present_in_candidates") is True
            and row.get("error_evidence", {}).get("present_in_top_k") is False
            for row in failures
        )
        evidence = analyze_ranking_traces(traces)
        return Diagnosis(
            largest,
            count,
            confirmed,
            strategy if confirmed else None,
            "KNOWN_PROBLEM" if confirmed else "NEEDS_HUMAN_OR_LLM_HYPOTHESIS",
            evidence,
        )
    known = bool(strategy) and largest not in HUMAN_OR_LLM_FAILURES
    return Diagnosis(
        largest,
        count,
        known,
        strategy if known else None,
        "KNOWN_PROBLEM" if known else "NEEDS_HUMAN_OR_LLM_HYPOTHESIS",
        {"reason": "No unambiguous deterministic stage rule is available."},
    )


def generate_ranking_candidates(
    baseline_config: dict[str, Any], diagnosis: Diagnosis, max_candidates: int
) -> list[dict[str, Any]]:
    if diagnosis.strategy != "ranking_search" or not diagnosis.known_problem:
        return []
    baseline_top_k = int(baseline_config["reranking"]["top_k"])
    baseline_strategy = str(baseline_config["reranking"]["selection_strategy"])
    specs = [
        ("ranking-topk-7", "reranking.top_k", baseline_top_k, 7),
        ("ranking-topk-10", "reranking.top_k", baseline_top_k, 10),
        (
            "ranking-max-two-per-document",
            "reranking.selection_strategy",
            baseline_strategy,
            "max_two_chunks_per_document",
        ),
    ]
    candidates: list[dict[str, Any]] = []
    for candidate_id, parameter, before, after in specs:
        if before == after:
            continue
        config = copy.deepcopy(baseline_config)
        section, key = parameter.split(".", 1)
        config[section][key] = after
        config["candidate_id"] = candidate_id
        config["parent_baseline"] = "current-baseline"
        config["change"] = {"parameter": parameter, "before": before, "after": after}
        config["target_failure"] = {"type": diagnosis.largest_failure, "count": diagnosis.count}
        config["hypothesis_source"] = {"type": "algorithmic_rule"}
        config["hypothesis"] = _hypothesis(diagnosis, parameter, before, after)
        candidates.append(config)
    return candidates[:max_candidates]


class GridSearchPlanner:
    """Bounded deterministic search; random/TPE/Bayesian planners can implement the protocol."""

    def generate(
        self,
        baseline_config: dict[str, Any],
        diagnosis: Diagnosis,
        max_candidates: int,
    ) -> list[dict[str, Any]]:
        return generate_ranking_candidates(baseline_config, diagnosis, max_candidates)


class RuleBasedPlanner:
    """Diagnose known stage failures and route them to a deterministic candidate search."""

    def __init__(self, search: CandidateSearch | None = None) -> None:
        self.search = search or GridSearchPlanner()

    def diagnose(self, census: dict[str, Any], traces: list[dict[str, Any]]) -> Diagnosis:
        return diagnose_failure(census, traces)

    def generate(
        self,
        baseline_config: dict[str, Any],
        diagnosis: Diagnosis,
        max_candidates: int,
    ) -> list[dict[str, Any]]:
        return self.search.generate(baseline_config, diagnosis, max_candidates)


def _hypothesis(diagnosis: Diagnosis, parameter: str, before: Any, after: Any) -> dict[str, str]:
    return {
        "observation": (
            f"{diagnosis.count} ranking failures; gold documents survive candidate retrieval "
            "but are incomplete in Top-K."
        ),
        "root_cause": "RANKING_STAGE",
        "hypothesis_source": "RULE_BASED",
        "change": f"{parameter}: {before} -> {after}",
        "expected_effect": "Increase complete gold-document coverage and reduce ranking failures.",
        "metrics_to_watch": (
            "RANKING_FAILURE, three_document completeness, gold-document Recall@K, MRR."
        ),
        "regression_risks": (
            "More or differently distributed evidence may reduce precision, citation validity, "
            "or unsupported-answer safety."
        ),
    }


def change_one_thing(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    change = candidate["change"]
    section, key = change["parameter"].split(".", 1)
    controlled = copy.deepcopy(candidate)
    metadata_fields = (
        "candidate_id",
        "parent_baseline",
        "change",
        "target_failure",
        "hypothesis_source",
        "hypothesis",
    )
    for metadata in metadata_fields:
        controlled.pop(metadata, None)
    expected = copy.deepcopy(baseline)
    for metadata in metadata_fields:
        expected.pop(metadata, None)
    expected[section][key] = change["after"]
    expected["experiment_id"] = controlled["experiment_id"]
    return controlled == expected


def rank_candidates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        structural = row["structural_metrics"]
        return (
            float(row["target_failure_before"] - row["target_failure_after"]),
            -float(row["critical_structural_regressions"]),
            float(structural["three_document_complete_coverage"]),
            float(structural.get("gold_document_recall_at_5") or 0),
            float(structural.get("gold_document_mrr") or 0),
        )

    ranked = sorted(results, key=key, reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _structural_result(
    candidate_id: str,
    change: dict[str, Any],
    output_dir: Path,
    baseline_target: int,
) -> dict[str, Any]:
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    regression = json.loads((output_dir / "regression.json").read_text(encoding="utf-8"))
    traces = results["traces"]
    metrics = results["metrics"]
    counts = results["failure_census"]["counts"]
    three = [row for row in traces if row.get("category") == "three_document"]
    three_complete = sum(
        row.get("error_evidence", {}).get("present_in_top_k") is True for row in three
    )
    critical_regressions = sum(
        counts.get(name, 0) > 0
        for name in ("FILTER_FAILURE", "RETRIEVAL_FAILURE", "CITATION_FAILURE")
    )
    quality_names = ("accuracy", "precision", "recall", "f1")
    quality_available = all(metrics.get(name) is not None for name in quality_names)
    return {
        "candidate_id": candidate_id,
        "change": change,
        "artifact": str(output_dir),
        "target_failure_before": baseline_target,
        "target_failure_after": int(counts.get("RANKING_FAILURE", 0)),
        "critical_structural_regressions": critical_regressions,
        "structural_metrics": {
            "three_document_complete_coverage": three_complete,
            "gold_document_recall_at_5": metrics.get("gold_document_recall_at_5"),
            "gold_document_recall_at_10": metrics.get("gold_document_recall_at_10"),
            "gold_document_mrr": metrics.get("gold_document_mrr"),
        },
        "semantic_quality_available": quality_available,
        "quality_metrics": {name: metrics.get(name) for name in quality_names},
        "gate": regression["decision"],
        "promotion_status": regression["promotion_status"],
        "reasons": regression["reasons"],
        "external_api_calls": 0,
        "semantic_judge_calls": 0,
        "llm_planner_calls": 0,
    }


ExperimentExecutor = Callable[..., tuple[Path, dict[str, Any]]]


@dataclass
class AutoImprovementController:
    session: Session
    experiment_executor: ExperimentExecutor = execute_experiment

    def run(
        self,
        *,
        baseline: Path,
        run_config_path: Path,
        output_root: Path,
        eval_output_root: Path,
        gate_path: Path,
        max_candidates: int | None = None,
        resume: bool = False,
    ) -> tuple[Path, dict[str, Any]]:
        config = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
        settings = _validate_run_config(config, max_candidates)
        run_id = config["run_id"]
        output_dir = output_root / run_id
        if output_dir.exists() and not resume:
            raise FileExistsError(f"auto-improvement artifact already exists: {output_dir}")
        candidates_dir = output_dir / "candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        baseline_results = json.loads((baseline / "results.json").read_text(encoding="utf-8"))
        baseline_config = yaml.safe_load(
            (baseline / "experiment_config.yaml").read_text(encoding="utf-8")
        )
        planner = RuleBasedPlanner()
        diagnosis = planner.diagnose(baseline_results["failure_census"], baseline_results["traces"])
        _write_common_artifacts(output_dir, run_config_path, baseline, baseline_results, diagnosis)
        if not diagnosis.known_problem:
            result = _stopped_result(
                run_id,
                diagnosis,
                baseline_results["experiment_id"],
                str(baseline),
            )
            _finalize(output_dir, result)
            return output_dir, result
        candidates = planner.generate(baseline_config, diagnosis, settings["max_candidates"])
        candidate_results: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            letter = chr(ord("a") + index)
            candidate["experiment_id"] = f"{run_id}-candidate-{letter}"
            if not change_one_thing(baseline_config, candidate):
                raise ValueError(f"CHANGE_ONE_THING_VIOLATION:{candidate['candidate_id']}")
            candidate_path = candidates_dir / f"candidate_{letter}.yaml"
            candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
            experiment_dir, _ = self.experiment_executor(
                config_path=candidate_path,
                baseline_path=baseline,
                output_root=eval_output_root,
                gate_path=gate_path,
                session=self.session,
                resume=resume,
            )
            candidate_results.append(
                _structural_result(
                    candidate["candidate_id"], candidate["change"], experiment_dir, diagnosis.count
                )
            )
        ranking = rank_candidates(candidate_results)
        best = ranking[0] if ranking else None
        promotion = (
            "PROMOTION_ELIGIBLE"
            if best and best["semantic_quality_available"] and best["gate"] == "PROMOTE"
            else "MANUAL_REVIEW"
            if best
            else "NOT_ELIGIBLE"
        )
        result = {
            "run_id": run_id,
            "status": "BEST_STRUCTURAL_CANDIDATE" if best else "NO_CANDIDATE",
            "baseline": {
                "experiment_id": baseline_results["experiment_id"],
                "path": str(baseline),
                "dataset_hash": baseline_results["dataset_hash"],
            },
            "planner": "RULE_BASED",
            "llm_planner": "OFF",
            "semantic_judge": "OFF",
            "external_api_calls": 0,
            "llm_tokens": 0,
            "diagnosis": diagnosis.as_dict(),
            "candidate_count": len(ranking),
            "ranking": ranking,
            "best_candidate": best,
            "structural_improvement_confirmed": bool(
                best and best["target_failure_after"] < best["target_failure_before"]
            ),
            "quality_review_required": bool(best and not best["semantic_quality_available"]),
            "promotion_eligibility": promotion,
            "auto_promoted": False,
        }
        _finalize(output_dir, result)
        return output_dir, result


def _validate_run_config(config: dict[str, Any], override: int | None) -> dict[str, int]:
    auto = config.get("auto_improvement") or {}
    guards = {
        "enabled": auto.get("enabled") is True,
        "llm_planner": auto.get("llm_planner", {}).get("enabled") is False,
        "semantic_judge": auto.get("semantic_judge", {}).get("enabled") is False,
        "paid_api_fallback": auto.get("paid_api_fallback", {}).get("enabled") is False,
        "auto_promote": auto.get("auto_promote", {}).get("enabled") is False,
        "rule_based": auto.get("planner", {}).get("mode") == "rule_based",
        "grid": auto.get("candidate_search", {}).get("mode") == "grid",
    }
    failed = [name for name, valid in guards.items() if not valid]
    if failed:
        raise ValueError(f"ZERO_API_SAFETY_GUARD:{','.join(failed)}")
    iterations = int(auto.get("max_iterations", 1))
    if iterations != 1:
        raise ValueError("max_iterations must be 1 in the current implementation")
    count = int(override or auto.get("max_candidates_per_iteration", 3))
    if not 1 <= count <= 3:
        raise ValueError("max_candidates must be between 1 and 3")
    return {"max_iterations": iterations, "max_candidates": count}


def _write_common_artifacts(
    output_dir: Path,
    config_path: Path,
    baseline: Path,
    baseline_results: dict[str, Any],
    diagnosis: Diagnosis,
) -> None:
    (output_dir / "run_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_json(
        output_dir / "baseline_summary.json",
        {
            "path": str(baseline),
            "experiment_id": baseline_results["experiment_id"],
            "dataset_hash": baseline_results["dataset_hash"],
            "metrics": baseline_results["metrics"],
        },
    )
    write_json(output_dir / "failure_summary.json", baseline_results["failure_census"])
    write_json(output_dir / "diagnosis.json", diagnosis.as_dict())
    write_json(
        output_dir / "strategy_selection.json",
        {
            "known_problem": diagnosis.known_problem,
            "selected_strategy": diagnosis.strategy,
            "status": diagnosis.status,
            "planner": "RULE_BASED",
            "llm_planner": "OFF",
        },
    )
    write_json(
        output_dir / "search_space.json",
        {
            "mode": "grid",
            "supported": SUPPORTED_RANKING_PARAMETERS,
            "unsupported": UNSUPPORTED_RANKING_PARAMETERS,
            "one_chunk_per_document": "SUPPORTED_BUT_EXCLUDED_ALREADY_TESTED_NO_TARGET_IMPROVEMENT",
        },
    )


def _stopped_result(
    run_id: str,
    diagnosis: Diagnosis,
    baseline_experiment_id: str,
    baseline_path: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "NEEDS_HUMAN_OR_LLM_HYPOTHESIS",
        "baseline": {"experiment_id": baseline_experiment_id, "path": baseline_path},
        "planner": "RULE_BASED",
        "llm_planner": "OFF",
        "semantic_judge": "OFF",
        "external_api_calls": 0,
        "llm_tokens": 0,
        "diagnosis": diagnosis.as_dict(),
        "candidate_count": 0,
        "ranking": [],
        "best_candidate": None,
        "structural_improvement_confirmed": False,
        "quality_review_required": True,
        "promotion_eligibility": "NOT_ELIGIBLE",
        "auto_promoted": False,
    }


def _finalize(output_dir: Path, result: dict[str, Any]) -> None:
    write_json(output_dir / "candidate_results.json", result["ranking"])
    write_json(output_dir / "ranking.json", result["ranking"])
    write_json(output_dir / "best_candidate.json", result["best_candidate"])
    (output_dir / "report.md").write_text(_render_report(result), encoding="utf-8")


def _render_report(result: dict[str, Any]) -> str:
    diagnosis = result["diagnosis"]
    lines = [
        "# AUTO IMPROVEMENT RUN",
        "",
        f"Baseline: `{result['baseline']['experiment_id']}` ({result['baseline']['path']})",
        f"Largest failure: `{diagnosis['largest_failure']}` ({diagnosis['count']})",
        f"Known problem: `{'YES' if diagnosis['known_problem'] else 'NO'}`",
        f"Selected strategy: `{diagnosis['strategy']}`",
        "Planner: `RULE_BASED`",
        "LLM calls: `0`",
        "Semantic Judge calls: `0`",
        f"Candidates generated: `{result['candidate_count']}`",
        "",
        "## Candidates",
        "",
    ]
    for row in result["ranking"]:
        lines.extend(
            [
                f"### {row['candidate_id']}",
                "",
                (
                    f"Change: `{row['change']['parameter']}: {row['change']['before']} -> "
                    f"{row['change']['after']}`"
                ),
                (
                    f"Target failure: `{row['target_failure_before']} -> "
                    f"{row['target_failure_after']}`"
                ),
                (
                    "Three-document complete coverage: "
                    f"`{row['structural_metrics']['three_document_complete_coverage']}`"
                ),
                (
                    "Semantic quality available: "
                    f"`{'YES' if row['semantic_quality_available'] else 'NO'}`"
                ),
                "",
            ]
        )
    best = result["best_candidate"]
    lines.extend(
        [
            "## Decision",
            "",
            f"Best structural candidate: `{best['candidate_id'] if best else None}`",
            (
                "Quality metrics available: "
                f"`{'YES' if best and best['semantic_quality_available'] else 'NO'}`"
            ),
            f"Promotion eligibility: `{result['promotion_eligibility']}`",
            f"Manual review required: `{'YES' if result['quality_review_required'] else 'NO'}`",
            "Auto promotion: `OFF`",
            "",
        ]
    )
    return "\n".join(lines)
