"""P1 V2 entry point using the general citation-authorization contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    run_p1_version_temporal_targeted_real_api_confirmation_v1 as pipeline,
)

pipeline.OUT = (
    ROOT
    / "data/experiments/rag-release-pipeline-v4/p1-targeted-v2"
)
pipeline.EXPERIMENT_ID = "P1_VERSION_TEMPORAL_TARGETED_REAL_API_CONFIRMATION_V2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "execute", "score"))
    args = parser.parse_args()
    if args.phase == "preflight":
        pipeline.preflight()
    elif args.phase == "execute":
        pipeline.execute()
    else:
        pipeline.score_and_finalize()


if __name__ == "__main__":
    main()
