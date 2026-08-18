import argparse
import json

from rag_workbench.db.session import session_factory
from rag_workbench.experiments.v2_final_benchmark import V2FinalBenchmark


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
        ),
    )
    phase = parser.parse_args().phase
    with session_factory()() as session:
        benchmark = V2FinalBenchmark(session)
        if phase == "initialize":
            benchmark.initialize()
            payload = benchmark.status()
        elif phase == "embedding-preflight":
            benchmark.initialize()
            payload = benchmark.embedding_preflight()
        elif phase == "retrieve":
            payload = benchmark.execute_retrieval()
        elif phase == "judge-preflight":
            payload = benchmark.judge_preflight()
        elif phase == "execute":
            payload = benchmark.execute()
        else:
            payload = benchmark.status(include_cases=True)
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
