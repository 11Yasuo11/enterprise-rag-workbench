from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.db.models import ExperimentRunRecord
from rag_workbench.experiments.runner import ExperimentResult


@dataclass(frozen=True)
class RegressionThreshold:
    metric: str
    max_decrease: float | None = None
    min_value: float | None = None
    max_increase: float | None = None
    security_critical: bool = False


PROJECT_BENCHMARK_THRESHOLDS = (
    RegressionThreshold("recall_at_k", max_decrease=0.05),
    RegressionThreshold("abstention_f1", max_decrease=0.05),
    RegressionThreshold("security_success_rate", min_value=1.0, security_critical=True),
    RegressionThreshold("version_accuracy", min_value=1.0),
)


def load_regression_thresholds(path: Path) -> tuple[RegressionThreshold, ...]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(RegressionThreshold(**item) for item in payload.get("thresholds", []))


def metric_delta(
    baseline: ExperimentResult, candidate: ExperimentResult, metric: str
) -> float | None:
    baseline_value = baseline.report.metrics.get(metric)
    candidate_value = candidate.report.metrics.get(metric)
    if baseline_value is None or candidate_value is None:
        return None
    return candidate_value - baseline_value


def compare_experiment_runs(
    session: Session,
    baseline_id: str,
    candidate_id: str,
    thresholds: tuple[RegressionThreshold, ...] = PROJECT_BENCHMARK_THRESHOLDS,
) -> dict[str, Any]:
    baseline = session.get(ExperimentRunRecord, baseline_id)
    candidate = session.get(ExperimentRunRecord, candidate_id)
    if baseline is None or candidate is None:
        raise ValueError("baseline or candidate experiment was not found")
    metric_names = sorted(set(baseline.aggregate_metrics) | set(candidate.aggregate_metrics))
    metrics: dict[str, dict[str, float | None]] = {}
    for name in metric_names:
        baseline_value = baseline.aggregate_metrics.get(name)
        candidate_value = candidate.aggregate_metrics.get(name)
        delta = (
            candidate_value - baseline_value
            if isinstance(baseline_value, (int, float))
            and isinstance(candidate_value, (int, float))
            else None
        )
        metrics[name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
        }
    latency_delta = (
        candidate.total_latency_ms - baseline.total_latency_ms
        if baseline.total_latency_ms is not None and candidate.total_latency_ms is not None
        else None
    )
    regressions = detect_regressions(metrics, thresholds)
    return {
        "baseline": _run_summary(baseline),
        "candidate": _run_summary(candidate),
        "metrics": metrics,
        "latency_delta_ms": latency_delta,
        "regressions": regressions,
        "category_metrics": _category_comparison(
            baseline.category_metrics, candidate.category_metrics
        ),
        "configuration_differences": _configuration_differences(baseline, candidate),
        "hard_constraints": {
            "acl_safety_required": 1.0,
            "version_accuracy_required": 1.0,
            "candidate_acceptable": (
                candidate.aggregate_metrics.get("security_success_rate") == 1.0
                and candidate.aggregate_metrics.get("version_accuracy") == 1.0
            ),
        },
        "passed": not regressions,
        "threshold_policy": "project benchmark criteria; not production SLOs",
    }


def pareto_analysis(session: Session) -> dict[str, Any]:
    """Return transparent role winners and a non-dominated quality/latency frontier."""
    runs = list(
        session.scalars(
            select(ExperimentRunRecord).where(ExperimentRunRecord.status == "completed")
        ).all()
    )
    acceptable = [
        run
        for run in runs
        if run.aggregate_metrics.get("security_success_rate") == 1.0
        and run.aggregate_metrics.get("version_accuracy") == 1.0
    ]
    baseline = next(
        (run for run in acceptable if run.name == "baseline-hashing-eval-v1"), None
    )
    baseline_abstention = (
        baseline.aggregate_metrics.get("abstention_f1") if baseline is not None else None
    )
    balanced_candidates = [
        run
        for run in acceptable
        if baseline_abstention is None
        or (run.aggregate_metrics.get("abstention_f1") or -1) >= baseline_abstention
    ]
    frontier = [
        run for run in acceptable if not any(_dominates(other, run) for other in acceptable)
    ]
    return {
        "policy": (
            "ACL safety and version accuracy must equal 1.0. Best balanced maximizes Recall, "
            "then nDCG, then MRR, then minimizes latency, without reducing abstention F1 "
            "below baseline-hashing-eval-v1. No metrics are hidden in a composite score."
        ),
        "acceptable_run_count": len(acceptable),
        "rejected_run_count": len(runs) - len(acceptable),
        "roles": {
            "highest_recall": _winner(acceptable, "recall_at_k"),
            "best_abstention": _winner(acceptable, "abstention_f1"),
            "best_balanced": _balanced_winner(balanced_candidates),
            "lowest_latency": _latency_winner(acceptable),
        },
        "pareto_frontier": [_analysis_summary(run) for run in frontier],
    }


def detect_regressions(
    metrics: dict[str, dict[str, float | None]],
    thresholds: tuple[RegressionThreshold, ...],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for threshold in thresholds:
        values = metrics.get(threshold.metric)
        if values is None:
            continue
        candidate = values["candidate"]
        delta = values["delta"]
        reason: str | None = None
        if (
            threshold.max_decrease is not None
            and delta is not None
            and delta < -threshold.max_decrease
        ):
            reason = f"decreased by {-delta:.6f}, exceeding {threshold.max_decrease:.6f}"
        if (
            threshold.min_value is not None
            and candidate is not None
            and candidate < threshold.min_value
        ):
            reason = f"{candidate:.6f} is below {threshold.min_value:.6f}"
        if (
            threshold.max_increase is not None
            and delta is not None
            and delta > threshold.max_increase
        ):
            reason = f"increased by {delta:.6f}, exceeding {threshold.max_increase:.6f}"
        if reason:
            failures.append(
                {
                    "metric": threshold.metric,
                    "reason": reason,
                    "security_critical": threshold.security_critical,
                }
            )
    return failures


def _run_summary(run: ExperimentRunRecord) -> dict[str, Any]:
    return {
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "config": {
            "ingestion": run.config.ingestion_config,
            "retrieval": run.config.retrieval_config,
            "generation": run.config.generation_config,
        },
        "total_latency_ms": run.total_latency_ms,
        "latency": {
            "query_embedding_ms": run.query_embedding_latency_ms,
            "vector_search_ms": run.vector_search_latency_ms,
            "acl_filter_ms": run.acl_filter_latency_ms,
            "context_construction_ms": run.context_construction_latency_ms,
            "generation_ms": run.generation_latency_ms,
            "total_ms": run.total_latency_ms,
        },
        "cache": {
            "hits": run.query_embedding_cache_hits,
            "misses": run.query_embedding_cache_misses,
        },
    }


def _category_comparison(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, dict[str, float | None]]]:
    compared: dict[str, dict[str, dict[str, float | None]]] = {}
    for category in sorted(set(baseline) | set(candidate)):
        compared[category] = {}
        baseline_metrics = baseline.get(category, {})
        candidate_metrics = candidate.get(category, {})
        for metric in sorted(set(baseline_metrics) | set(candidate_metrics)):
            before = baseline_metrics.get(metric)
            after = candidate_metrics.get(metric)
            compared[category][metric] = {
                "baseline": before,
                "candidate": after,
                "delta": after - before
                if isinstance(before, (int, float)) and isinstance(after, (int, float))
                else None,
            }
    return compared


def _configuration_differences(
    baseline: ExperimentRunRecord, candidate: ExperimentRunRecord
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    sections = {
        "ingestion": (baseline.config.ingestion_config, candidate.config.ingestion_config),
        "retrieval": (baseline.config.retrieval_config, candidate.config.retrieval_config),
        "generation": (baseline.config.generation_config, candidate.config.generation_config),
    }
    for section, (before, after) in sections.items():
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                differences.append(
                    {
                        "field": f"{section}.{key}",
                        "baseline": before.get(key),
                        "candidate": after.get(key),
                    }
                )
    return differences


def _quality_tuple(run: ExperimentRunRecord) -> tuple[float, float, float, float]:
    metrics = run.aggregate_metrics
    return (
        float(metrics.get("recall_at_k") or -1),
        float(metrics.get("ndcg_at_k") or -1),
        float(metrics.get("reciprocal_rank") or -1),
        -float(run.total_latency_ms or float("inf")),
    )


def _balanced_winner(runs: list[ExperimentRunRecord]) -> dict[str, Any] | None:
    return _analysis_summary(max(runs, key=_quality_tuple)) if runs else None


def _winner(runs: list[ExperimentRunRecord], metric: str) -> dict[str, Any] | None:
    candidates = [run for run in runs if run.aggregate_metrics.get(metric) is not None]
    return (
        _analysis_summary(max(candidates, key=lambda run: run.aggregate_metrics[metric]))
        if candidates
        else None
    )


def _latency_winner(runs: list[ExperimentRunRecord]) -> dict[str, Any] | None:
    candidates = [run for run in runs if run.total_latency_ms is not None]
    return (
        _analysis_summary(min(candidates, key=lambda run: run.total_latency_ms))
        if candidates
        else None
    )


def _analysis_summary(run: ExperimentRunRecord) -> dict[str, Any]:
    return {
        "id": run.id,
        "name": run.name,
        "metrics": run.aggregate_metrics,
        "latency_ms": run.total_latency_ms,
        "config": {
            "ingestion": run.config.ingestion_config,
            "retrieval": run.config.retrieval_config,
        },
    }


def _dominates(left: ExperimentRunRecord, right: ExperimentRunRecord) -> bool:
    left_values = (
        left.aggregate_metrics.get("recall_at_k"),
        left.aggregate_metrics.get("reciprocal_rank"),
        left.aggregate_metrics.get("ndcg_at_k"),
        left.aggregate_metrics.get("abstention_f1"),
    )
    right_values = (
        right.aggregate_metrics.get("recall_at_k"),
        right.aggregate_metrics.get("reciprocal_rank"),
        right.aggregate_metrics.get("ndcg_at_k"),
        right.aggregate_metrics.get("abstention_f1"),
    )
    if any(value is None for value in left_values + right_values):
        return False
    quality_no_worse = all(a >= b for a, b in zip(left_values, right_values, strict=True))
    latency_no_worse = (left.total_latency_ms or float("inf")) <= (
        right.total_latency_ms or float("inf")
    )
    strictly_better = any(a > b for a, b in zip(left_values, right_values, strict=True)) or (
        (left.total_latency_ms or float("inf")) < (right.total_latency_ms or float("inf"))
    )
    return quality_no_worse and latency_no_worse and strictly_better
