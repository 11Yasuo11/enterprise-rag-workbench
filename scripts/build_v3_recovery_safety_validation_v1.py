#!/usr/bin/env python3
import json

from rag_workbench.experiments.v3_phase2_safety_cases import write_dataset


def main() -> None:
    payload = write_dataset()
    print(
        json.dumps(
            {
                "dataset_id": payload["dataset_id"],
                "dataset_hash": payload["dataset_hash"],
                "overlap_report": payload["overlap_report"],
                "case_count": len(payload["cases"]),
                "freeze_timestamp": payload["freeze_timestamp"],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
