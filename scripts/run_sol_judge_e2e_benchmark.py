import argparse
import json

from rag_workbench.db.session import session_factory
from rag_workbench.experiments.sol_judge_e2e_benchmark import SolJudgeEndToEndBenchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("status", "initialize", "prepare", "judge-preflight", "execute")
    )
    phase = parser.parse_args().phase
    with session_factory()() as session:
        benchmark = SolJudgeEndToEndBenchmark(session)
        if phase == "initialize":
            benchmark.initialize()
            payload = benchmark.status()
        elif phase == "prepare":
            payload = benchmark.prepare()
        elif phase == "judge-preflight":
            payload = benchmark.judge_preflight()
        elif phase == "execute":
            payload = benchmark.execute()
        else:
            payload = benchmark.status(include_cases=True)
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
