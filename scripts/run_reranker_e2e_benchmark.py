import argparse
import json

from rag_workbench.db.session import session_factory
from rag_workbench.experiments.reranker_e2e_benchmark import RerankerEndToEndBenchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("status", "initialize", "prepare", "judge-preflight", "execute")
    )
    arguments = parser.parse_args()
    with session_factory()() as session:
        benchmark = RerankerEndToEndBenchmark(session)
        if arguments.phase == "initialize":
            benchmark.initialize()
            payload = benchmark.status()
        elif arguments.phase == "prepare":
            payload = benchmark.prepare()
        elif arguments.phase == "judge-preflight":
            payload = benchmark.judge_preflight()
        elif arguments.phase == "execute":
            payload = benchmark.execute()
        else:
            payload = benchmark.status(include_cases=True)
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
