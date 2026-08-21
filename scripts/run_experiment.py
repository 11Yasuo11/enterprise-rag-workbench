#!/usr/bin/env python3
"""Run Candidate → RAG → trace → deterministic eval → comparison → gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rag_workbench.config import get_settings
from rag_workbench.evaluation.experiment_runner import execute_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--semantic-judge", choices=("off",), default="off")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--gate", type=Path, default=Path("configs/deterministic-eval-gate.yaml"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with Session(create_engine(get_settings().database_url)) as session:
        output, regression = execute_experiment(
            config_path=args.candidate,
            baseline_path=args.baseline,
            output_root=args.output_root,
            gate_path=args.gate,
            session=session,
            resume=args.resume,
        )
    print(f"Wrote {output}")
    print(regression["decision"])
    for reason in regression["reasons"]:
        print(f"Reason: {reason}")


if __name__ == "__main__":
    main()
