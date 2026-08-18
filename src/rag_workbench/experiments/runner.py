import itertools
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import yaml
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.answerability.cache import CachedAnswerabilityGate
from rag_workbench.answerability.openai_compatible import OpenAICompatibleAnswerabilityGate
from rag_workbench.answerability.planning import (
    ExternalJudgeCallLimitGate,
    GateInput,
    LocalJudgeCallLimitGate,
    plan_judge_calls,
)
from rag_workbench.answerability.qwen import QwenAnswerabilityGate
from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import (
    STORED_EMBEDDING_DIMENSION,
    AnswerabilityGateCacheRecord,
    Chunk,
    ExperimentCaseResultRecord,
    ExperimentConfigRecord,
    ExperimentRunRecord,
    QueryEmbeddingCacheRecord,
)
from rag_workbench.evaluation.datasets import load_evaluation_dataset
from rag_workbench.evaluation.evaluator import EvaluationReport, EvaluationRunner
from rag_workbench.experiments.configs import ExperimentConfig, load_experiment_config
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.generation.generator import RagService
from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
from rag_workbench.ingestion.corpus_roots import collect_corpus_paths
from rag_workbench.ingestion.loaders import load_document
from rag_workbench.ingestion.pipeline import IngestionPipeline
from rag_workbench.providers.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from rag_workbench.providers.llm import ExtractiveGenerationProvider, OpenAICompatibleLLMProvider
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import Retriever


class ExperimentExecutionOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    offline_only: bool = True
    max_cases: int | None = Field(default=None, ge=1)
    max_configs: int = Field(default=20, ge=1)
    confirm_external_calls: bool = False
    confirm_external_judge_calls: bool = False


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    report: EvaluationReport
    run_id: str | None = None


@dataclass(frozen=True)
class GridPlan:
    configurations: tuple[ExperimentConfig, ...]
    reindex_config_names: tuple[str, ...]
    evaluation_executions: int
    documents_to_rechunk: int = 0
    document_embeddings_required: int = 0
    document_embedding_calls: int = 0
    query_embedding_calls: int = 0
    expected_embedding_calls: int = 0
    expected_external_embedding_calls: int = 0
    existing_cached_queries: int = 0
    missing_unique_query_embeddings: int = 0
    maximum_expected_external_calls: int = 0
    configured_external_call_ceiling: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration_count": len(self.configurations),
            "distinct_index_count": len(
                {config.index_identity for config in self.configurations}
            ),
            "reindex_configuration_count": len(self.reindex_config_names),
            "reindex_configurations": list(self.reindex_config_names),
            "evaluation_executions": self.evaluation_executions,
            "documents_to_rechunk": self.documents_to_rechunk,
            "document_embeddings_required": self.document_embeddings_required,
            "document_embedding_calls": self.document_embedding_calls,
            "query_embedding_calls": self.query_embedding_calls,
            "expected_embedding_calls": self.expected_embedding_calls,
            "expected_external_embedding_calls": self.expected_external_embedding_calls,
            "existing_cached_queries": self.existing_cached_queries,
            "missing_unique_query_embeddings": self.missing_unique_query_embeddings,
            "maximum_expected_external_calls": self.maximum_expected_external_calls,
            "configured_external_call_ceiling": self.configured_external_call_ceiling,
            "configurations": [
                {
                    "name": config.name,
                    "config_hash": config.experiment_config_hash,
                    "index_identity": config.index_identity,
                }
                for config in self.configurations
            ],
        }


class ExperimentRunner:
    def __init__(
        self,
        session: Session,
        *,
        settings: Settings | None = None,
        corpus_root: Path = Path("data/synthetic_company"),
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.corpus_root = corpus_root

    def run(
        self,
        config: ExperimentConfig,
        dataset_path: Path,
        options: ExperimentExecutionOptions | None = None,
        *,
        cases_override: tuple | None = None,
    ) -> ExperimentResult:
        options = options or ExperimentExecutionOptions()
        dataset = load_evaluation_dataset(dataset_path)
        if dataset.dataset_version != config.identity.evaluation_dataset_version:
            raise ValueError("experiment and evaluation dataset versions do not match")
        if dataset.corpus_version != config.identity.corpus_version:
            raise ValueError("experiment and evaluation corpus versions do not match")
        selected_cases = cases_override if cases_override is not None else dataset.cases
        cases = selected_cases[: options.max_cases] if options.max_cases else selected_cases
        self._validate_external_budget(config, tuple(case.question for case in cases), options)
        config_record = self._config_record(config)
        run = ExperimentRunRecord(config_id=config_record.id, name=config.name, status="running")
        self.session.add(run)
        self.session.commit()
        started = time.perf_counter()
        try:
            embedding = self._embedding_provider(config, options)
            generator = self._generation_provider(config, options)
            embedding_before = embedding.usage.input_tokens
            embedding_calls_before = embedding.usage.external_calls
            self._ensure_index(config, embedding)
            gate = self._answerability_gate(config, options, embedding, tuple(cases))
            rag = RagService(
                self.session,
                Retriever(self.session, embedding, config.index_identity),
                ContextBuilder(config.generation.context_budget),
                generator,
                gate,
                supporting_context_only=(
                    config.answerability_gate.supporting_context_only
                    if config.answerability_gate
                    else False
                ),
            )
            report = EvaluationRunner(rag).run(
                tuple(cases), config.retrieval.top_k, config.retrieval.score_threshold
            )
            self._persist_cases(run, report)
            run.aggregate_metrics = report.metrics
            run.category_metrics = report.category_metrics
            run.retrieval_latency_ms = self._mean_latency(report, "retrieval_latency_ms")
            run.query_embedding_latency_ms = self._mean_latency(
                report, "query_embedding_latency_ms"
            )
            run.embedding_cache_lookup_latency_ms = self._mean_latency(
                report, "embedding_cache_lookup_latency_ms"
            )
            run.vector_search_latency_ms = self._mean_latency(report, "vector_search_latency_ms")
            run.acl_filter_latency_ms = self._mean_latency(report, "acl_filter_latency_ms")
            run.context_construction_latency_ms = self._mean_latency(
                report, "context_construction_latency_ms"
            )
            run.answerability_gate_cache_lookup_latency_ms = self._mean_latency(
                report, "answerability_gate_cache_lookup_latency_ms"
            )
            run.answerability_judge_latency_ms = self._mean_latency(
                report, "answerability_judge_latency_ms"
            )
            run.context_pruning_latency_ms = self._mean_latency(
                report, "context_pruning_latency_ms"
            )
            run.generation_latency_ms = self._mean_latency(report, "generation_latency_ms")
            run.total_latency_ms = self._mean_latency(report, "total_latency_ms")
            run.input_tokens = sum(case.prompt_tokens or 0 for case in report.cases)
            run.output_tokens = sum(case.completion_tokens or 0 for case in report.cases)
            run.embedding_tokens = embedding.usage.input_tokens - embedding_before
            run.query_embedding_cache_hits = sum(
                case.query_embedding_cache_hit for case in report.cases if case.error is None
            )
            run.query_embedding_cache_misses = sum(
                not case.query_embedding_cache_hit for case in report.cases if case.error is None
            )
            run.external_embedding_calls = (
                embedding.usage.external_calls - embedding_calls_before
            )
            gate_cases = [case for case in report.cases if case.gate_cache_hit is not None]
            run.gate_cache_hits = sum(case.gate_cache_hit is True for case in gate_cases)
            run.gate_cache_misses = sum(case.gate_cache_hit is False for case in gate_cases)
            run.external_judge_calls = sum(case.external_judge_calls for case in report.cases)
            run.local_judge_calls = sum(case.local_judge_calls for case in report.cases)
            run.judge_prompt_tokens = sum(case.judge_prompt_tokens or 0 for case in report.cases)
            run.judge_completion_tokens = sum(
                case.judge_completion_tokens or 0 for case in report.cases
            )
            run.status = "completed_with_errors" if report.failed_case_count else "completed"
            run.completed_at = datetime.now(UTC)
            self.session.commit()
            return ExperimentResult(config=config, report=report, run_id=run.id)
        except Exception as exc:
            self.session.rollback()
            persisted = self.session.get(ExperimentRunRecord, run.id)
            if persisted is not None:
                persisted.status = "failed"
                persisted.error = f"{type(exc).__name__}: {exc}"
                persisted.completed_at = datetime.now(UTC)
                persisted.total_latency_ms = (time.perf_counter() - started) * 1000
                self.session.commit()
            raise

    def _config_record(self, config: ExperimentConfig) -> ExperimentConfigRecord:
        record = self.session.scalar(
            select(ExperimentConfigRecord).where(
                ExperimentConfigRecord.config_hash == config.experiment_config_hash
            )
        )
        if record is not None:
            return record
        record = ExperimentConfigRecord(
            name=config.name,
            config_hash=config.experiment_config_hash,
            ingestion_config_hash=config.ingestion_config_hash,
            retrieval_config_hash=config.retrieval_config_hash,
            generation_config_hash=config.generation_config_hash,
            gate_config_hash=config.gate_config_hash,
            index_identity=config.index_identity,
            corpus_version=config.identity.corpus_version,
            evaluation_dataset_version=config.identity.evaluation_dataset_version,
            ingestion_config=config.ingestion.stable_payload(),
            retrieval_config=config.retrieval.stable_payload(),
            generation_config=config.generation.stable_payload(),
            gate_config=(
                config.answerability_gate.stable_payload()
                if config.answerability_gate is not None
                else None
            ),
        )
        self.session.add(record)
        self.session.flush()
        return record

    def _embedding_provider(
        self, config: ExperimentConfig, options: ExperimentExecutionOptions
    ) -> EmbeddingProvider:
        ingestion = config.ingestion
        if ingestion.embedding_dimension != STORED_EMBEDDING_DIMENSION:
            raise ValueError(
                f"This deployment stores vector({STORED_EMBEDDING_DIMENSION}); configured "
                f"dimension {ingestion.embedding_dimension} is incompatible. Select a "
                "provider-supported output dimension or migrate and fully re-index; vectors "
                "are never padded or truncated."
            )
        if ingestion.embedding_provider == "hashing":
            expected_model = f"local-hashing-{ingestion.embedding_dimension}"
            if ingestion.embedding_model != expected_model:
                raise ValueError(
                    f"Hashing embedding_model must be {expected_model!r} for its dimension"
                )
            return HashingEmbeddingProvider(ingestion.embedding_dimension)
        if ingestion.embedding_provider in {"openai", "openai-compatible"}:
            self._allow_external(options, "embedding")
            return OpenAICompatibleEmbeddingProvider(
                api_key=self.settings.embedding_api_key or "",
                model=ingestion.embedding_model,
                dimension=ingestion.embedding_dimension,
                base_url=self.settings.embedding_base_url,
                version=ingestion.embedding_version,
                provider_name=ingestion.embedding_provider,
            )
        raise ValueError(f"Unsupported embedding provider: {ingestion.embedding_provider}")

    def _generation_provider(self, config: ExperimentConfig, options: ExperimentExecutionOptions):
        generation = config.generation
        if generation.llm_provider == "extractive":
            return ExtractiveGenerationProvider()
        if generation.llm_provider in {"openai", "openai-compatible"}:
            self._allow_external(options, "generation")
            return OpenAICompatibleLLMProvider(
                api_key=self.settings.openai_api_key or "",
                model=generation.llm_model,
                base_url=self.settings.openai_base_url,
            )
        raise ValueError(f"Unsupported generation provider: {generation.llm_provider}")

    def _answerability_gate(
        self,
        config: ExperimentConfig,
        options: ExperimentExecutionOptions,
        embedding: EmbeddingProvider,
        cases: tuple,
    ):
        gate_config = config.answerability_gate
        if gate_config is None:
            return None
        if gate_config.judge_provider not in {"qwen", "openai"}:
            raise ValueError(
                f"Unsupported answerability judge provider: {gate_config.judge_provider}"
            )
        is_hosted = gate_config.judge_provider == "openai"
        if is_hosted:
            if not options.confirm_external_judge_calls:
                raise ValueError(
                    "confirm_external_judge_calls is required for hosted answerability judging"
                )
            if options.offline_only:
                raise ValueError("offline_only blocks hosted answerability judge calls")
            if not self.settings.allow_external_judge_calls:
                raise ValueError("ALLOW_EXTERNAL_JUDGE_CALLS=true is required")
            if self.settings.judge_provider != gate_config.judge_provider:
                raise ValueError("JUDGE_PROVIDER must match the experiment configuration")
            if self.settings.judge_model != gate_config.judge_model:
                raise ValueError("JUDGE_MODEL must match the experiment configuration")
        else:
            if self.settings.local_judge_provider != gate_config.judge_provider:
                raise ValueError("LOCAL_JUDGE_PROVIDER must match the experiment configuration")
            if self.settings.local_judge_model != gate_config.judge_model:
                raise ValueError("LOCAL_JUDGE_MODEL must match the experiment configuration")

        retriever = Retriever(self.session, embedding, config.index_identity)
        inputs: list[GateInput] = []
        for case in cases:
            from rag_workbench.security.permissions import Principal

            results = retriever.retrieve(
                case.question,
                top_k=config.retrieval.top_k,
                score_threshold=config.retrieval.score_threshold,
                principal=Principal(
                    principal_id=case.principal.principal_id,
                    tenant_id=case.principal.tenant_id,
                    permission_groups=frozenset(case.principal.permission_groups),
                ),
            )
            inputs.append(
                GateInput(
                    question=case.question,
                    chunks=tuple(
                        GateEvidence(
                            chunk_id=item.chunk_id,
                            document_id=item.document_id,
                            document_version_id=item.document_version_id,
                            version=item.version,
                            text=item.text,
                            index_identity=config.index_identity,
                        )
                        for item in results
                    ),
                )
            )
        plan = plan_judge_calls(
            self.session,
            tuple(inputs),
            provider=gate_config.judge_provider,
            model=gate_config.judge_model,
            gate_version=gate_config.judge_version,
            prompt_version=gate_config.prompt_version,
            configured_ceiling=(
                self.settings.max_external_judge_calls
                if is_hosted
                else self.settings.max_local_judge_calls
            ),
        )
        print(plan.safe_summary())
        recorded_attempts = self._recorded_judge_attempts(
            gate_config.judge_provider,
            gate_config.judge_model,
        )
        remaining_capacity = max(0, plan.configured_ceiling - recorded_attempts)
        if plan.expected_maximum_calls > remaining_capacity:
            raise ValueError(
                f"Answerability judging has {recorded_attempts} persisted call attempts "
                f"and requires {plan.expected_maximum_calls} more, exceeding configured "
                f"ceiling {plan.configured_ceiling}"
            )
        if is_hosted:
            delegate = OpenAICompatibleAnswerabilityGate(
                api_key=self.settings.effective_judge_api_key or "",
                model=gate_config.judge_model,
                base_url=self.settings.judge_base_url,
                gate_version=gate_config.judge_version,
                prompt_version=gate_config.prompt_version,
                provider_name="openai",
            )
            limited = ExternalJudgeCallLimitGate(delegate, remaining_capacity)
        else:
            delegate = QwenAnswerabilityGate(
                api_key=self.settings.local_judge_api_key,
                model=gate_config.judge_model,
                base_url=self.settings.local_judge_base_url,
                gate_version=gate_config.judge_version,
                prompt_version=gate_config.prompt_version,
            )
            limited = LocalJudgeCallLimitGate(delegate, remaining_capacity)
        return CachedAnswerabilityGate(self.session, limited)

    def _recorded_judge_attempts(self, provider: str, model: str) -> int:
        runs = self.session.scalars(select(ExperimentRunRecord)).all()
        persisted_run_attempts = sum(
            (
                (run.external_judge_calls or 0)
                if provider == "openai"
                else (run.local_judge_calls or 0)
            )
            for run in runs
            if run.config.gate_config
            and run.config.gate_config.get("judge_provider") == provider
            and run.config.gate_config.get("judge_model") == model
        )
        cached_results = self.session.scalar(
            select(func.count())
            .select_from(AnswerabilityGateCacheRecord)
            .where(
                AnswerabilityGateCacheRecord.judge_provider == provider,
                AnswerabilityGateCacheRecord.judge_model == model,
            )
        ) or 0
        return max(persisted_run_attempts, cached_results)

    def _allow_external(self, options: ExperimentExecutionOptions, kind: str) -> None:
        if options.offline_only:
            raise ValueError(f"offline_only blocks external {kind} calls")
        if not options.confirm_external_calls:
            raise ValueError(f"confirm_external_calls is required for external {kind} calls")
        if not self.settings.allow_external_calls:
            raise ValueError(f"ALLOW_EXTERNAL_CALLS=true is required for external {kind} calls")

    def _validate_external_budget(
        self,
        config: ExperimentConfig,
        queries: tuple[str, ...],
        options: ExperimentExecutionOptions,
    ) -> None:
        if config.ingestion.embedding_provider not in {"openai", "openai-compatible"}:
            return
        self._allow_external(options, "embedding")
        index_exists = self.session.scalar(
            select(Chunk.id).where(Chunk.index_identity == config.index_identity).limit(1)
        )
        document_calls = 0 if index_exists is not None else len(self._corpus_paths(config))
        unique_queries = tuple(dict.fromkeys(queries))
        cached_keys = [
            self.session.get(
                QueryEmbeddingCacheRecord,
                query_embedding_cache_key(
                    query,
                    provider=config.ingestion.embedding_provider,
                    model=config.ingestion.embedding_model,
                    version=config.ingestion.embedding_version,
                    dimension=config.ingestion.embedding_dimension,
                ),
            )
            is not None
            for query in unique_queries
        ]
        cached_count = sum(cached_keys)
        missing_queries = len(unique_queries) - cached_count
        expected_calls = document_calls + missing_queries
        limit = self.settings.max_external_embedding_calls
        if limit < 1:
            raise ValueError(
                "MAX_EXTERNAL_EMBEDDING_CALLS must be a positive reviewed limit for external "
                "embedding experiments"
            )
        persisted_calls = sum(
            run.external_embedding_calls or 0
            for run in self.session.scalars(select(ExperimentRunRecord)).all()
        )
        cached_provider_queries = self.session.scalar(
            select(func.count())
            .select_from(QueryEmbeddingCacheRecord)
            .where(
                QueryEmbeddingCacheRecord.embedding_provider
                == config.ingestion.embedding_provider,
                QueryEmbeddingCacheRecord.embedding_model
                == config.ingestion.embedding_model,
                QueryEmbeddingCacheRecord.embedding_version
                == config.ingestion.embedding_version,
                QueryEmbeddingCacheRecord.embedding_dimension
                == config.ingestion.embedding_dimension,
            )
        ) or 0
        persisted_calls = max(persisted_calls, int(cached_provider_queries))
        remaining_capacity = max(0, limit - persisted_calls)
        if expected_calls > remaining_capacity:
            raise ValueError(
                f"Experiment has {persisted_calls} persisted embedding calls and requires "
                f"{expected_calls} more "
                f"({document_calls} document + {missing_queries} missing unique query), "
                "after safely reusing "
                f"{cached_count} cached queries, exceeding configured "
                f"limit {limit}"
            )

    def _ensure_index(self, config: ExperimentConfig, provider: EmbeddingProvider) -> None:
        indexed = self.session.scalar(
            select(Chunk.id).where(Chunk.index_identity == config.index_identity).limit(1)
        )
        if indexed is not None:
            return
        chunker = FixedTokenChunker(
            FixedTokenConfig(config.ingestion.chunk_size, config.ingestion.chunk_overlap)
        )
        pipeline = IngestionPipeline(
            self.session, chunker, provider, index_identity=config.index_identity
        )
        paths = self._corpus_paths(config)

        for path in paths:
            pipeline.ingest_path(path)

    def _corpus_paths(self, config: ExperimentConfig) -> list[Path]:
        paths = collect_corpus_paths(
            config.identity.corpus_version,
            self.corpus_root,
            require_manifest_complete=True,
        )
        if not paths:
            raise ValueError(f"No corpus documents found under {self.corpus_root}")
        return paths

    def _persist_cases(self, run: ExperimentRunRecord, report: EvaluationReport) -> None:
        for case in report.cases:
            self.session.add(
                ExperimentCaseResultRecord(
                    experiment_run_id=run.id,
                    rag_run_id=case.rag_run_id,
                    eval_case_id=case.case_id,
                    category=case.category,
                    question=case.question,
                    expected_answer=case.expected_answer,
                    expected_document_ids=list(case.expected_document_ids),
                    expected_chunk_ids=list(case.expected_chunk_ids),
                    forbidden_document_ids=list(case.forbidden_document_ids),
                    expected_versions=case.expected_versions,
                    principal=case.principal,
                    retrieved_document_ids=list(case.retrieved_document_ids),
                    retrieved_chunk_ids=list(case.retrieved_chunk_ids),
                    retrieval_trace=list(case.retrieval_trace),
                    answer=case.answer,
                    citations=list(case.citations),
                    expected_abstain=case.expected_abstain,
                    abstained=case.status == "abstained" if case.status != "error" else None,
                    metrics=case.metrics,
                    failure_type=case.failure_type,
                    failure_types=list(case.failure_types),
                    failure_details=case.failure_details,
                    security_passed=case.security_passed,
                    retrieval_latency_ms=case.retrieval_latency_ms,
                    query_embedding_latency_ms=case.query_embedding_latency_ms,
                    embedding_cache_lookup_latency_ms=case.embedding_cache_lookup_latency_ms,
                    vector_search_latency_ms=case.vector_search_latency_ms,
                    acl_filter_latency_ms=case.acl_filter_latency_ms,
                    context_construction_latency_ms=case.context_construction_latency_ms,
                    generation_latency_ms=case.generation_latency_ms,
                    supporting_chunk_ids=list(case.supporting_chunk_ids),
                    generation_context_chunk_ids=list(case.generation_context_chunk_ids),
                    answerability_result=case.answerability_result,
                    answerability_operational_error=case.answerability_operational_error,
                    answerability_gate_cache_lookup_latency_ms=(
                        case.answerability_gate_cache_lookup_latency_ms
                    ),
                    answerability_judge_latency_ms=case.answerability_judge_latency_ms,
                    context_pruning_latency_ms=case.context_pruning_latency_ms,
                    gate_cache_hit=case.gate_cache_hit,
                    external_judge_calls=case.external_judge_calls,
                    local_judge_calls=case.local_judge_calls,
                    judge_prompt_tokens=case.judge_prompt_tokens,
                    judge_completion_tokens=case.judge_completion_tokens,
                    query_embedding_cache_hit=case.query_embedding_cache_hit,
                    total_latency_ms=case.total_latency_ms,
                    error=case.error,
                )
            )

    @staticmethod
    def _mean_latency(report: EvaluationReport, field: str) -> float | None:
        values = [getattr(case, field) for case in report.cases if case.error is None]
        return mean(values) if values else None


def run_retrieval_experiment(
    config: ExperimentConfig, evaluator: EvaluationRunner, cases: list
) -> ExperimentResult:
    report = evaluator.run(
        cases,
        top_k=config.retrieval.top_k,
        score_threshold=config.retrieval.score_threshold,
    )
    return ExperimentResult(config=config, report=report)


GRID_PATHS = {
    "chunk_size": "ingestion.chunk_size",
    "chunk_overlap": "ingestion.chunk_overlap",
    "embedding_provider": "ingestion.embedding_provider",
    "embedding_model": "ingestion.embedding_model",
    "embedding_dimension": "ingestion.embedding_dimension",
    "top_k": "retrieval.top_k",
    "score_threshold": "retrieval.score_threshold",
}


def plan_grid(
    session: Session,
    path: Path,
    *,
    case_count: int,
    max_configs: int = 20,
    corpus_root: Path = Path("data/synthetic_company"),
    queries: tuple[str, ...] | None = None,
) -> GridPlan:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_path = (path.parent / payload["base_config"]).resolve()
    base = load_experiment_config(base_path)
    variants = payload.get("variants")
    if variants is not None:
        combinations = [tuple(item.items()) for item in variants]
    else:
        grid = payload.get("grid") or {}
        keys = list(grid)
        combinations = [
            tuple(zip(keys, values, strict=True))
            for values in itertools.product(*(grid[key] for key in keys))
        ]
    if len(combinations) > max_configs:
        raise ValueError(
            f"Grid has {len(combinations)} configurations; max_configs is {max_configs}"
        )
    configs: list[ExperimentConfig] = []
    for index, values in enumerate(combinations, start=1):
        config_payload = base.stable_payload()
        config_payload["name"] = f"{base.name}-grid-{index:03d}"
        for key, value in values:
            if key == "name":
                config_payload["name"] = str(value)
                continue
            dotted = GRID_PATHS.get(key, key)
            section, field = dotted.split(".", 1)
            config_payload[section][field] = value
        configs.append(ExperimentConfig.model_validate(config_payload))
    known_indexes = set(session.scalars(select(Chunk.index_identity).distinct()).all())
    reindex_names: list[str] = []
    new_indexes: list[ExperimentConfig] = []
    for config in configs:
        if config.index_identity not in known_indexes:
            reindex_names.append(config.name)
            new_indexes.append(config)
            known_indexes.add(config.index_identity)
    reindex = tuple(reindex_names)
    return plan_configurations(
        session,
        tuple(configs),
        case_count=case_count,
        corpus_root=corpus_root,
        reindex_names=reindex,
        new_indexes=tuple(new_indexes),
        queries=queries,
    )


def plan_experiment(
    session: Session,
    config: ExperimentConfig,
    *,
    case_count: int,
    corpus_root: Path = Path("data/synthetic_company"),
    queries: tuple[str, ...] | None = None,
) -> GridPlan:
    index_exists = session.scalar(
        select(Chunk.id).where(Chunk.index_identity == config.index_identity).limit(1)
    )
    new_indexes = () if index_exists is not None else (config,)
    reindex_names = () if index_exists is not None else (config.name,)
    return plan_configurations(
        session,
        (config,),
        case_count=case_count,
        corpus_root=corpus_root,
        reindex_names=reindex_names,
        new_indexes=new_indexes,
        queries=queries,
    )


def plan_configurations(
    session: Session,
    configs: tuple[ExperimentConfig, ...],
    *,
    case_count: int,
    corpus_root: Path,
    reindex_names: tuple[str, ...],
    new_indexes: tuple[ExperimentConfig, ...],
    queries: tuple[str, ...] | None = None,
) -> GridPlan:
    documents_to_rechunk = 0
    embeddings_required = 0
    document_calls = 0
    external_document_calls = 0
    for config in new_indexes:
        paths = _corpus_paths_for(config, corpus_root)
        documents_to_rechunk += len(paths)
        document_calls += len(paths)
        if config.ingestion.embedding_provider in {"openai", "openai-compatible"}:
            external_document_calls += len(paths)
        chunker = FixedTokenChunker(
            FixedTokenConfig(config.ingestion.chunk_size, config.ingestion.chunk_overlap)
        )
        embeddings_required += sum(len(chunker.chunk(load_document(item))) for item in paths)
    unique_queries = tuple(dict.fromkeys(queries or ()))
    cached_query_keys: set[str] = set()
    missing_query_keys: set[str] = set()
    for config in configs:
        if config.ingestion.embedding_provider not in {"openai", "openai-compatible"}:
            continue
        if unique_queries:
            for query in unique_queries:
                key = query_embedding_cache_key(
                    query,
                    provider=config.ingestion.embedding_provider,
                    model=config.ingestion.embedding_model,
                    version=config.ingestion.embedding_version,
                    dimension=config.ingestion.embedding_dimension,
                )
                if session.get(QueryEmbeddingCacheRecord, key) is None:
                    missing_query_keys.add(key)
                else:
                    cached_query_keys.add(key)
        else:
            missing_query_keys.update(
                f"unknown-{config.ingestion.config_hash}-{index}" for index in range(case_count)
            )
    external_query_calls = len(missing_query_keys)
    local_query_calls = (
        sum(
            config.ingestion.embedding_provider not in {"openai", "openai-compatible"}
            for config in configs
        )
        * case_count
    )
    query_calls = local_query_calls + external_query_calls
    settings = get_settings()
    return GridPlan(
        tuple(configs),
        reindex_names,
        len(configs) * case_count,
        documents_to_rechunk,
        embeddings_required,
        document_calls,
        query_calls,
        document_calls + query_calls,
        external_document_calls + external_query_calls,
        len(cached_query_keys),
        len(missing_query_keys),
        external_document_calls + external_query_calls,
        settings.max_external_embedding_calls,
    )


def _corpus_paths_for(config: ExperimentConfig, corpus_root: Path) -> list[Path]:
    return collect_corpus_paths(config.identity.corpus_version, corpus_root)
