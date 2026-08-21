#!/usr/bin/env python3
"""Run one bounded, rule-based, zero-API auto-improvement iteration."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rag_workbench.config import get_settings
from rag_workbench.evaluation.auto_improvement import AutoImprovementController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/auto-improvement-zero-api.yaml")
    )
    parser.add_argument("--planner", choices=("rule_based",), default="rule_based")
    parser.add_argument("--llm-planner", choices=("off",), default="off")
    parser.add_argument("--semantic-judge", choices=("off",), default="off")
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--max-iterations", type=int, choices=(1,), default=1)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/auto_improvement"))
    parser.add_argument("--eval-output-root", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--gate", type=Path, default=Path("configs/deterministic-eval-gate.yaml"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    del args.planner, args.llm_planner, args.semantic_judge, args.max_iterations
    with Session(create_engine(get_settings().database_url)) as session:
        output, result = AutoImprovementController(session).run(
            baseline=args.baseline,
            run_config_path=args.config,
            output_root=args.output_root,
            eval_output_root=args.eval_output_root,
            gate_path=args.gate,
            max_candidates=args.max_candidates,
            resume=args.resume,
        )
    print(f"Wrote {output}")
    print(result["status"])
    print(result["promotion_eligibility"])


if __name__ == "__main__":
    main()
