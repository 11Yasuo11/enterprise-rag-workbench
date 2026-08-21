#!/usr/bin/env python3
import argparse
import json

from rag_workbench.db.models import V3Phase3ExperimentRecord
from rag_workbench.db.session import session_factory
from rag_workbench.experiments.v3_final_ab import (
    LOCK_ID,
    V3FinalAbBenchmark,
    verify_frozen_final_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("status", "verify-dataset", "preflight", "execute"))
    phase = parser.parse_args().phase
    if phase == "verify-dataset":
        print(json.dumps(verify_frozen_final_dataset(), indent=2, default=str))
        return
    with session_factory()() as session:
        benchmark = V3FinalAbBenchmark(session)
        if phase == "preflight":
            payload = benchmark.combined_preflight()
        elif phase == "execute":
            payload = benchmark.execute()
            payload = {
                key: payload[key]
                for key in (
                    "lock_id",
                    "experiment_id",
                    "dataset_id",
                    "dataset_hash",
                    "stop_code",
                    "required_ceilings",
                    "embedding_preflight",
                    "hosted_preflight",
                    "control_metrics",
                    "candidate_metrics",
                    "paired_deltas",
                    "recovery_funnel",
                    "instruction_boundary",
                    "prompt_injection",
                    "security",
                    "citations",
                    "failure_census",
                    "cost",
                    "usage",
                    "promotion_decision",
                    "v3_status",
                    "selected_strategy",
                    "primary_remaining_bottleneck",
                    "completed",
                )
                if key in payload
            }
        else:
            record = session.get(V3Phase3ExperimentRecord, LOCK_ID)
            payload = benchmark.status() if record is not None else {"exists": False}
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
