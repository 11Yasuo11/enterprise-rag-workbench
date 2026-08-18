import argparse
import json

from rag_workbench.db.session import session_factory
from rag_workbench.experiments.hybrid_retrieval_benchmark import HybridRetrievalBenchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("status", "initialize", "calibration", "holdout"))
    arguments = parser.parse_args()
    with session_factory()() as session:
        benchmark = HybridRetrievalBenchmark(session)
        if arguments.phase == "initialize":
            benchmark.initialize()
            payload = benchmark.status(include_cases=False)
        elif arguments.phase == "calibration":
            payload = benchmark.run_calibration()
        elif arguments.phase == "holdout":
            payload = benchmark.run_holdout()
        else:
            payload = benchmark.status(include_cases=True)
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
