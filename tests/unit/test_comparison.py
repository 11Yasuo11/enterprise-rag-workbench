from rag_workbench.experiments.comparison import RegressionThreshold, detect_regressions


def test_regression_detection_honors_quality_and_security_thresholds() -> None:
    metrics = {
        "recall_at_k": {"baseline": 0.8, "candidate": 0.7, "delta": -0.1},
        "security_success_rate": {"baseline": 1.0, "candidate": 0.9, "delta": -0.1},
    }
    failures = detect_regressions(
        metrics,
        (
            RegressionThreshold("recall_at_k", max_decrease=0.05),
            RegressionThreshold("security_success_rate", min_value=1.0, security_critical=True),
        ),
    )
    assert {failure["metric"] for failure in failures} == {
        "recall_at_k",
        "security_success_rate",
    }
    assert next(item for item in failures if item["security_critical"])["metric"] == (
        "security_success_rate"
    )
