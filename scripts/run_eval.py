#!/usr/bin/env python3
"""Run deterministic evaluation from a frozen dataset and captured RAG traces."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from rag_workbench.evaluation.eval_loop import (
    DEFAULT_KS,
    evaluate,
    load_frozen_cases,
    load_traces,
    write_run,
)

DEFAULT_DATASET = Path("data/eval/phase5k/acmeai_enterprise_rag_v3_final_unseen_e2e_120.jsonl")
DEFAULT_TRACES = Path(
    "data/experiments/v3-phase5k-final-e2e/phase5kr_corrected_per_case_results.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--traces",
        type=Path,
        default=DEFAULT_TRACES,
        help="captured JSON/JSONL; no RAG or LLM is called",
    )
    parser.add_argument(
        "--arm", default="FINAL_R", help="optional arm filter for multi-arm trace files"
    )
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/eval"))
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--semantic-judge", choices=("off", "sol"), default="off")
    args = parser.parse_args()
    if args.semantic_judge != "off":
        parser.error(
            "semantic judge 'sol' requires a separately reviewed adapter; "
            "this deterministic CLI performs no paid calls"
        )
    ks = tuple(sorted(set(args.k)))
    if not ks or any(k < 1 for k in ks):
        parser.error("--k values must be positive")
    cases, dataset_hash = load_frozen_cases(args.dataset)
    raw_traces = load_traces(args.traces, args.arm or None)
    traces, metrics, census, slices = evaluate(cases, raw_traces, ks)
    experiment_id = (
        args.experiment_id or f"deterministic-baseline-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    output_dir = args.output_root / experiment_id
    write_run(
        output_dir=output_dir,
        experiment_id=experiment_id,
        dataset_path=args.dataset,
        dataset_hash=dataset_hash,
        traces_path=args.traces,
        traces=traces,
        metrics=metrics,
        census=census,
        slices=slices,
        semantic_judge=args.semantic_judge,
        ks=ks,
    )
    print(f"Wrote {output_dir}")
    print(f"F1: {metrics['f1']}")
    print("=== FAILURE CENSUS ===")
    for name, count in census["counts"].items():
        if count:
            print(f"{name:<24} {count:>4}  {census['percentages'][name]:>7.2%}")


if __name__ == "__main__":
    main()
