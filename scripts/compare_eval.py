#!/usr/bin/env python3
"""Compare deterministic Control/Candidate artifacts and apply a promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_workbench.evaluation.eval_loop import compare_results, load_gate


def _load(path: Path) -> tuple[dict, Path]:
    target = path / "results.json" if path.is_dir() else path
    return json.loads(target.read_text(encoding="utf-8")), target.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=Path("configs/deterministic-eval-gate.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline, _ = _load(args.baseline)
    candidate, candidate_dir = _load(args.candidate)
    comparison = compare_results(baseline, candidate, load_gate(args.gate)).payload
    output = args.output or candidate_dir / "regression.json"
    output.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(comparison["decision"])
    for reason in comparison["reasons"]:
        print(f"Reason: {reason}")
    print("Metric          Baseline    Candidate    Delta")
    for name in ("precision", "recall", "f1", "hit_at_5", "citation_validity"):
        item = comparison["metric_comparison"].get(name)
        if item:
            values = f"{item['baseline']:>10.3f} {item['candidate']:>12.3f} {item['delta']:>+9.3f}"
            print(f"{name:<15} {values}")


if __name__ == "__main__":
    main()
