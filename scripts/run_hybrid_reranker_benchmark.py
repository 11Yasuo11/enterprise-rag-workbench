import argparse
import json

from rag_workbench.db.session import session_factory
from rag_workbench.experiments.hybrid_reranker_benchmark import HybridRerankerBenchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "status",
            "initialize",
            "embedding-preflight",
            "calibrate",
            "holdout-retrieve",
            "judge-preflight",
            "holdout-execute",
        ),
    )
    phase = parser.parse_args().phase
    with session_factory()() as session:
        benchmark = HybridRerankerBenchmark(session)
        if phase == "initialize":
            benchmark.initialize()
            payload = benchmark.status()
        elif phase == "embedding-preflight":
            payload = benchmark.embedding_preflight()
        elif phase == "calibrate":
            payload = benchmark.calibrate()
        elif phase == "holdout-retrieve":
            payload = benchmark.holdout_retrieve()
        elif phase == "judge-preflight":
            payload = benchmark.judge_preflight()
        elif phase == "holdout-execute":
            payload = benchmark.holdout_execute()
        else:
            payload = benchmark.status(include_cases=True)
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
