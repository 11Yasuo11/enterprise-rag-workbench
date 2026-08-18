import argparse
import json

from rag_workbench.db.session import session_factory
from rag_workbench.experiments.v2_quality_recovery import V2QualityRecoveryBaseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("status", "initialize"))
    phase = parser.parse_args().phase
    with session_factory()() as session:
        baseline = V2QualityRecoveryBaseline(session)
        if phase == "initialize":
            baseline.initialize()
        payload = baseline.status(include_cases=True)
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
