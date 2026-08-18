#!/usr/bin/env python3
"""Write the frozen unseen V3 Phase 1 dataset after diagnostic GO."""

from rag_workbench.experiments.v3_generate_verify_cases import write_dataset


def main() -> None:
    payload = write_dataset()
    print(payload["dataset_id"])
    print(payload["dataset_hash"])
    print(payload["overlap_report"])


if __name__ == "__main__":
    main()
