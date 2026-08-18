import argparse
import json
from pathlib import Path

from sqlalchemy import select

from rag_workbench.api.dependencies import context_builder, embedding_provider, llm_provider
from rag_workbench.config import get_settings
from rag_workbench.db.models import ExperimentRunRecord
from rag_workbench.db.session import session_factory
from rag_workbench.evaluation import EvaluationRunner
from rag_workbench.evaluation.datasets import (
    load_evaluation_dataset,
    validate_evaluation_dataset,
)
from rag_workbench.experiments.comparison import (
    PROJECT_BENCHMARK_THRESHOLDS,
    compare_experiment_runs,
    load_regression_thresholds,
)
from rag_workbench.experiments.configs import load_experiment_config
from rag_workbench.experiments.evidence_benchmark import EvidenceJudgeBenchmark
from rag_workbench.experiments.multidoc_benchmark import MultiDocumentEvidenceBenchmark
from rag_workbench.experiments.reporting import (
    experiment_to_dict,
    export_benchmark_markdown,
)
from rag_workbench.experiments.runner import (
    ExperimentExecutionOptions,
    ExperimentRunner,
    plan_experiment,
    plan_grid,
)
from rag_workbench.generation import RagService
from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
from rag_workbench.ingestion.corpus_roots import corpus_roots_for_version
from rag_workbench.ingestion.pipeline import IngestionPipeline
from rag_workbench.retrieval.retriever import Retriever


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enterprise RAG Workbench local commands")
    subcommands = parser.add_subparsers(dest="command", required=True)
    ingest = subcommands.add_parser("ingest")
    ingest.add_argument("paths", nargs="+", type=Path)
    evaluate = subcommands.add_parser("eval")
    evaluate.add_argument("--dataset", type=Path, default=Path("data/eval/initial.json"))
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--score-threshold", type=float, default=0.2)

    validate = subcommands.add_parser("validate-dataset")
    validate.add_argument("dataset", type=Path)

    experiment = subcommands.add_parser("experiment")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    run = experiment_commands.add_parser("run")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--dataset", required=True, type=Path)
    run.add_argument("--max-cases", type=int)
    run.add_argument("--allow-external", action="store_true")
    run.add_argument("--confirm-external-calls", action="store_true")
    run.add_argument("--confirm-external-judge-calls", action="store_true")
    listing = experiment_commands.add_parser("list")
    listing.add_argument("--limit", type=int, default=50)
    compare = experiment_commands.add_parser("compare")
    compare.add_argument("baseline_id")
    compare.add_argument("candidate_id")
    compare.add_argument("--thresholds", type=Path)
    export = experiment_commands.add_parser("export")
    export.add_argument("run_id")
    export.add_argument("--output", type=Path, default=Path("BENCHMARK.md"))

    grid = subcommands.add_parser("grid")
    grid.add_argument("--config", required=True, type=Path)
    grid.add_argument("--dataset", required=True, type=Path)
    grid.add_argument("--max-configs", type=int, default=20)
    grid.add_argument("--max-cases", type=int)
    grid.add_argument("--dry-run", action="store_true")
    grid.add_argument("--allow-external", action="store_true")
    grid.add_argument("--confirm-external-calls", action="store_true")
    grid.add_argument("--confirm-external-judge-calls", action="store_true")
    evidence = subcommands.add_parser("evidence-benchmark")
    evidence.add_argument("action", choices=("calibration", "holdout", "status"))
    multidoc = subcommands.add_parser("multidoc-benchmark")
    multidoc.add_argument("action", choices=("initialize", "calibration", "holdout", "status"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    with session_factory()() as session:
        settings = get_settings()
        if args.command == "ingest":
            pipeline = IngestionPipeline(
                session,
                FixedTokenChunker(FixedTokenConfig(settings.chunk_size, settings.chunk_overlap)),
                embedding_provider(),
            )
            results = [pipeline.ingest_path(path) for path in args.paths]
            print(json.dumps([result.__dict__ for result in results], indent=2))
        elif args.command == "eval":
            rag = RagService(
                session,
                Retriever(session, embedding_provider()),
                context_builder(),
                llm_provider(),
            )
            report = EvaluationRunner(rag).run(
                load_evaluation_dataset(args.dataset), args.top_k, args.score_threshold
            )
            print(json.dumps(report.to_dict(), indent=2))
        elif args.command == "validate-dataset":
            dataset = load_evaluation_dataset(args.dataset, validate=False)
            result = validate_evaluation_dataset(
                dataset,
                corpus_roots_for_version(
                    dataset.corpus_version, data_root=args.dataset.parent.parent
                ),
            )
            print(result.model_dump_json(indent=2))
            if not result.valid:
                raise SystemExit(1)
        elif args.command == "experiment":
            _experiment_command(session, args)
        elif args.command == "grid":
            dataset = load_evaluation_dataset(args.dataset)
            case_count = min(len(dataset), args.max_cases) if args.max_cases else len(dataset)
            plan = plan_grid(
                session,
                args.config,
                case_count=case_count,
                max_configs=args.max_configs,
                queries=tuple(case.question for case in dataset.cases[:case_count]),
            )
            print(json.dumps(plan.to_dict(), indent=2))
            if not args.dry_run:
                runner = ExperimentRunner(session)
                options = _options(args)
                run_ids = [
                    runner.run(config, args.dataset, options).run_id
                    for config in plan.configurations
                ]
                print(json.dumps({"run_ids": run_ids}, indent=2))
        elif args.command == "evidence-benchmark":
            benchmark = EvidenceJudgeBenchmark(session)
            if args.action == "calibration":
                payload = benchmark.run_calibration()
            elif args.action == "holdout":
                payload = benchmark.run_holdout()
            else:
                payload = benchmark.status(include_cases=False)
            print(json.dumps(payload, indent=2, default=str))
        elif args.command == "multidoc-benchmark":
            benchmark = MultiDocumentEvidenceBenchmark(session)
            if args.action == "initialize":
                benchmark.initialize()
                payload = benchmark.status(include_cases=False)
            elif args.action == "calibration":
                payload = benchmark.run_calibration()
            elif args.action == "holdout":
                payload = benchmark.run_holdout()
            else:
                payload = benchmark.status(include_cases=False)
            print(json.dumps(payload, indent=2, default=str))


def _experiment_command(session, args: argparse.Namespace) -> None:
    if args.experiment_command == "run":
        config = load_experiment_config(args.config)
        dataset = load_evaluation_dataset(args.dataset)
        case_count = min(len(dataset), args.max_cases) if args.max_cases else len(dataset)
        print(
            json.dumps(
                {
                    "execution_plan": plan_experiment(
                        session,
                        config,
                        case_count=case_count,
                        queries=tuple(case.question for case in dataset.cases[:case_count]),
                    ).to_dict()
                },
                indent=2,
            )
        )
        result = ExperimentRunner(session).run(
            config, args.dataset, _options(args)
        )
        run = session.get(ExperimentRunRecord, result.run_id)
        print(
            json.dumps(experiment_to_dict(session, run, include_cases=False), indent=2, default=str)
        )
    elif args.experiment_command == "list":
        runs = session.scalars(
            select(ExperimentRunRecord)
            .order_by(ExperimentRunRecord.started_at.desc())
            .limit(args.limit)
        ).all()
        print(
            json.dumps(
                [experiment_to_dict(session, run, include_cases=False) for run in runs],
                indent=2,
                default=str,
            )
        )
    elif args.experiment_command == "compare":
        thresholds = (
            load_regression_thresholds(args.thresholds)
            if args.thresholds
            else PROJECT_BENCHMARK_THRESHOLDS
        )
        print(
            json.dumps(
                compare_experiment_runs(
                    session, args.baseline_id, args.candidate_id, thresholds
                ),
                indent=2,
            )
        )
    elif args.experiment_command == "export":
        print(export_benchmark_markdown(session, args.run_id, args.output))


def _options(args: argparse.Namespace) -> ExperimentExecutionOptions:
    return ExperimentExecutionOptions(
        offline_only=not args.allow_external,
        max_cases=args.max_cases,
        max_configs=getattr(args, "max_configs", 20),
        confirm_external_calls=args.confirm_external_calls,
        confirm_external_judge_calls=args.confirm_external_judge_calls,
    )


if __name__ == "__main__":
    main()
