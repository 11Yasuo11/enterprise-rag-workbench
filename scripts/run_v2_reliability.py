import argparse
import json

from rag_workbench.db.session import session_factory
from rag_workbench.experiments.v2_reliability import V2ReliabilityHardening


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("status", "initialize", "execute"))
    phase = parser.parse_args().phase
    with session_factory()() as session:
        benchmark = V2ReliabilityHardening(session)
        if phase == "initialize":
            benchmark.initialize()
            payload = benchmark.status()
        elif phase == "execute":
            payload = benchmark.execute()
        else:
            payload = benchmark.status(include_cases=True)
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
