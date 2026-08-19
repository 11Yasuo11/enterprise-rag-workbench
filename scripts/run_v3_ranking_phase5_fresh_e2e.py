#!/usr/bin/env python3

import argparse
import json

from rag_workbench.db.session import session_factory
from rag_workbench.experiments.v3_ranking_phase5_fresh_e2e import (
    V3RankingPhase5FreshE2EBenchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "status",
            "freeze-dataset",
            "embedding-preflight",
            "execute",
        ),
    )
    args = parser.parse_args()

    with session_factory()() as session:
        benchmark = V3RankingPhase5FreshE2EBenchmark(session)
        if args.phase == "status":
            payload = benchmark.status()
        elif args.phase == "freeze-dataset":
            payload = benchmark.freeze_dataset()
        elif args.phase == "embedding-preflight":
            payload = benchmark.embedding_preflight()
        elif args.phase == "execute":
            payload = benchmark.execute()
        else:
            payload = benchmark.status()

        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

