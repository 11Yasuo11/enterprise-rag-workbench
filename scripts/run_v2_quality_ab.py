import argparse
import json

from rag_workbench.db.session import session_factory
from rag_workbench.experiments.v2_quality_ab import V2QualityAbBenchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("status", "execute"))
    parser.add_argument(
        "--tiny-scale", action="store_true", help="use 64/128 synthetic scales for tests"
    )
    args = parser.parse_args()
    reranker_factory = None
    if not args.tiny_scale:

        def reranker_factory():
            from rag_workbench.reranking import CrossEncoderReranker

            return CrossEncoderReranker()

    with session_factory()() as session:
        scales = (64, 128) if args.tiny_scale else None
        benchmark = V2QualityAbBenchmark(session, reranker_factory=reranker_factory, scales=scales)
        payload = (
            benchmark.execute() if args.phase == "execute" else benchmark.status(include_cases=True)
        )
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
