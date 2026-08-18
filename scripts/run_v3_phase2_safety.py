#!/usr/bin/env python3
import argparse
import json

from rag_workbench.config import get_settings
from rag_workbench.db.session import session_factory
from rag_workbench.experiments.v3_phase2_safety import (
    V3Phase2SafetyBenchmark,
    hosted_preflight_estimate,
)
from rag_workbench.experiments.v3_phase2_safety_report import persist_v3_phase2_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("status", "initialize", "freeze-dataset", "preflight", "execute"),
    )
    phase = parser.parse_args().phase
    settings = get_settings()
    with session_factory()() as session:
        benchmark = V3Phase2SafetyBenchmark(session)
        if phase == "initialize":
            benchmark.initialize()
            payload = benchmark.status()
        elif phase == "freeze-dataset":
            payload = benchmark.freeze_dataset()
        elif phase == "preflight":
            benchmark.initialize()
            payload = hosted_preflight_estimate()
            if (
                not settings.allow_external_judge_calls
                or not settings.allow_external_calls
            ):
                payload = benchmark.authorization_stop(payload)
        elif phase == "execute":
            benchmark.initialize()
            preflight = hosted_preflight_estimate()
            if (
                not settings.allow_external_judge_calls
                or not settings.allow_external_calls
                or settings.max_external_judge_calls
                < (
                    preflight["missing_logical_calls"]["primary_judge"]
                    + preflight["missing_logical_calls"]["recovery_draft"]
                    + preflight["missing_logical_calls"]["claim_verifier"]
                )
            ):
                payload = benchmark.authorization_stop(preflight)
            else:
                payload = {
                    "error": (
                        "hosted validation path requires an authorized environment "
                        "with ingested corpus"
                    )
                }
        else:
            payload = benchmark.status()
        persist_v3_phase2_markdown(benchmark.status())
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
