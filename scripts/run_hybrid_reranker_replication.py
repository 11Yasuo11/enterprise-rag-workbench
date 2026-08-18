import argparse
import json

from rag_workbench.db.session import session_factory
from rag_workbench.experiments.hybrid_reranker_replication import (
    FinalV1Benchmark,
    HybridRerankerReplicationBenchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "status",
            "initialize",
            "embedding-preflight",
            "retrieve",
            "judge-preflight",
            "execute",
            "final-status",
            "final-initialize",
            "final-embedding-preflight",
            "final-retrieve",
            "final-judge-preflight",
            "final-execute",
        ),
    )
    phase = parser.parse_args().phase
    with session_factory()() as session:
        if phase.startswith("final"):
            benchmark = FinalV1Benchmark(session)
            action = phase.removeprefix("final-")
            if action == "initialize":
                benchmark.initialize()
                payload = benchmark.status()
            elif action == "embedding-preflight":
                payload = benchmark.embedding_preflight()
            elif action == "retrieve":
                payload = benchmark.retrieve()
            elif action == "judge-preflight":
                payload = benchmark.judge_preflight()
            elif action == "execute":
                payload = benchmark.execute()
            else:
                payload = benchmark.status(include_cases=True)
            print(json.dumps(payload, indent=2, default=str))
            return
        benchmark = HybridRerankerReplicationBenchmark(session)
        if phase == "initialize":
            benchmark.initialize()
            payload = benchmark.status()
        elif phase == "embedding-preflight":
            payload = benchmark.embedding_preflight()
        elif phase == "retrieve":
            payload = benchmark.retrieve()
        elif phase == "judge-preflight":
            payload = benchmark.judge_preflight()
        elif phase == "execute":
            payload = benchmark.execute()
        else:
            payload = benchmark.status(include_cases=True)
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
