#!/usr/bin/env python3
import argparse
import json

from rag_workbench.db.models import V3Phase2ExperimentRecord
from rag_workbench.db.session import session_factory
from rag_workbench.experiments.v3_phase2_safety import LOCK_ID, V3Phase2SafetyBenchmark
from rag_workbench.experiments.v3_phase2_safety_report import persist_v3_phase2_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("status", "freeze-dataset", "execute"))
    phase = parser.parse_args().phase
    with session_factory()() as session:
        benchmark = V3Phase2SafetyBenchmark(session)
        if phase == "freeze-dataset":
            payload = benchmark.freeze_dataset()
        elif phase == "execute":
            payload = benchmark.execute()
            persist_v3_phase2_markdown(payload)
            payload = {
                key: payload[key]
                for key in (
                    "lock_id",
                    "dataset_id",
                    "dataset_hash",
                    "overlap_report",
                    "experiments",
                    "selected_safety_mechanism",
                    "qualification",
                    "development_replay",
                    "hosted_preflight",
                    "final_dataset",
                    "final_preflight",
                    "verdict",
                    "v3_status",
                    "primary_remaining_bottleneck",
                )
                if key in payload
            }
            replay = payload.get("development_replay") or {}
            payload["development_replay"] = {
                key: replay.get(key)
                for key in (
                    "notice",
                    "candidate_retained_historical_rescues",
                    "historical_fn_rescues_phase1",
                    "fv2_inj_02",
                    "fv2_inj_03",
                    "blocked_historical_rescues",
                )
            }
        else:
            record = session.get(V3Phase2ExperimentRecord, LOCK_ID)
            payload = {
                "exists": record is not None,
                "dataset_id": getattr(record, "dataset_id", None),
            }
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
