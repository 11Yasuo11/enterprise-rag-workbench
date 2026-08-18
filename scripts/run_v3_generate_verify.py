#!/usr/bin/env python3
import argparse
import json

from rag_workbench.db.session import session_factory
from rag_workbench.experiments.v3_generate_verify import V3GenerateVerifyBenchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "status",
            "initialize",
            "diagnostic-preflight",
            "diagnostic",
            "freeze-dataset",
            "embedding-preflight",
            "retrieve",
            "hosted-preflight",
            "execute",
        ),
    )
    phase = parser.parse_args().phase
    with session_factory()() as session:
        benchmark = V3GenerateVerifyBenchmark(session)
        if phase == "initialize":
            benchmark.initialize()
            payload = benchmark.status()
        elif phase == "diagnostic-preflight":
            payload = benchmark.diagnostic_preflight()
        elif phase == "diagnostic":
            payload = benchmark.execute_diagnostic()
        elif phase == "freeze-dataset":
            payload = benchmark.freeze_unseen_dataset()
        elif phase == "embedding-preflight":
            payload = benchmark.embedding_preflight()
        elif phase == "retrieve":
            payload = benchmark.execute_retrieval()
        elif phase == "hosted-preflight":
            payload = benchmark.hosted_preflight()
        elif phase == "execute":
            payload = benchmark.execute()
        else:
            payload = benchmark.status(include_cases=True)
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
