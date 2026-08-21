from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from rag_workbench.api.dependencies import (
    answerability_gate,
    context_builder,
    embedding_provider,
    llm_provider,
    production_reranker,
)
from rag_workbench.api.schemas import (
    DocumentResponse,
    EvalRunRequest,
    ExperimentComparisonResponse,
    ExperimentDetailResponse,
    ExperimentRunRequest,
    ExperimentSummaryResponse,
    IngestRequest,
    IngestResultResponse,
    RagQueryRequest,
    RagQueryResponse,
    RetrievalResultResponse,
    RetrieveRequest,
)
from rag_workbench.config import get_settings
from rag_workbench.db.models import (
    Document,
    DocumentVersion,
    ExperimentRunRecord,
    RagRun,
    RetrievalResultRecord,
)
from rag_workbench.db.session import get_db
from rag_workbench.evaluation import EvaluationRunner
from rag_workbench.evaluation.datasets import load_evaluation_dataset
from rag_workbench.experiments.comparison import compare_experiment_runs, pareto_analysis
from rag_workbench.experiments.evidence_benchmark import EvidenceJudgeBenchmark
from rag_workbench.experiments.hybrid_reranker_benchmark import HybridRerankerBenchmark
from rag_workbench.experiments.hybrid_reranker_replication import (
    FinalV1Benchmark,
    HybridRerankerReplicationBenchmark,
)
from rag_workbench.experiments.hybrid_retrieval_benchmark import HybridRetrievalBenchmark
from rag_workbench.experiments.judge_e2e_benchmark import FrozenJudgeEndToEndBenchmark
from rag_workbench.experiments.multidoc_benchmark import MultiDocumentEvidenceBenchmark
from rag_workbench.experiments.reporting import (
    experiment_to_dict,
    render_benchmark_markdown,
)
from rag_workbench.experiments.reranker_e2e_benchmark import RerankerEndToEndBenchmark
from rag_workbench.experiments.reranking_benchmark import DenseCrossEncoderBenchmark
from rag_workbench.experiments.runner import ExperimentRunner
from rag_workbench.experiments.sol_judge_e2e_benchmark import SolJudgeEndToEndBenchmark
from rag_workbench.experiments.v2_document_diversity import V2DocumentDiversityBenchmark
from rag_workbench.experiments.v2_final_benchmark import V2FinalBenchmark
from rag_workbench.experiments.v2_quality_ab import V2QualityAbBenchmark
from rag_workbench.experiments.v2_quality_recovery import V2QualityRecoveryBaseline
from rag_workbench.experiments.v2_reliability import V2ReliabilityHardening
from rag_workbench.experiments.v2_soft_document_cap import V2SoftDocumentCapBenchmark
from rag_workbench.experiments.v2_sufficiency_fn import V2SufficiencyFnBenchmark
from rag_workbench.experiments.v3_generate_verify import V3GenerateVerifyBenchmark
from rag_workbench.generation import RagService
from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
from rag_workbench.ingestion.pipeline import IngestionConflictError, IngestionPipeline
from rag_workbench.logging import configure_logging
from rag_workbench.retrieval.filters import RetrievalFilters
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.runtime import (
    CanonicalRagRuntime,
    build_canonical_runtime,
    production_config_from_settings,
)
from rag_workbench.security.permissions import Principal

logger = structlog.get_logger(__name__)
SessionDependency = Annotated[Session, Depends(get_db)]


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging(get_settings().log_level)
    try:
        production_config_from_settings(get_settings()).validate_for_serving()
        logger.info("canonical_rag_config_validated")
    except RuntimeError as exc:
        logger.error("canonical_rag_config_invalid", error=str(exc))
        raise
    logger.info("api_started")
    yield


app = FastAPI(
    title="Enterprise RAG Workbench API",
    version="2.0.0",
    description="Inspectable, evaluation-first enterprise RAG baseline",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _principal(request: RetrieveRequest) -> Principal:
    return Principal(
        principal_id=request.principal.principal_id,
        tenant_id=request.principal.tenant_id,
        permission_groups=frozenset(request.principal.permission_groups),
    )


def _filters(request: RetrieveRequest) -> RetrievalFilters:
    return RetrievalFilters(
        document_ids=tuple(request.filters.document_ids),
        source_types=tuple(request.filters.source_types),
    )


def _retriever(session: Session) -> Retriever:
    return Retriever(session, embedding_provider())


def _rag_service(session: Session) -> RagService:
    """Legacy dense Top-5 service. Active /rag/query uses CanonicalRagRuntime."""
    return RagService(
        session,
        _retriever(session),
        context_builder(),
        llm_provider(),
        answerability_gate(session),
        supporting_context_only=True,
    )


def _canonical_runtime(session: Session) -> CanonicalRagRuntime:
    settings = get_settings()
    config = production_config_from_settings(settings)
    luna = None
    sol = None
    if settings.allow_external_judge_calls and settings.effective_judge_api_key:
        from rag_workbench.experiments.atomic_requirement_contract_v1.verifier import (
            FrozenEvidenceVerifier,
        )

        key = settings.effective_judge_api_key
        base = settings.judge_base_url or settings.openai_base_url
        luna = FrozenEvidenceVerifier(api_key=key, base_url=base, model=config.luna_model)
        sol = FrozenEvidenceVerifier(api_key=key, base_url=base, model=config.sol_model)
    return build_canonical_runtime(
        session,
        config=config,
        embedding_provider=embedding_provider(),
        reranker=production_reranker(),
        luna_verifier=luna,
        sol_verifier=sol,
    )


def _safe_corpus_path(raw_path: str) -> Path:
    corpus_root = (Path.cwd() / "data" / "synthetic_company").resolve()
    supplied = Path(raw_path)
    candidate = (
        (Path.cwd() / supplied).resolve()
        if supplied.parts[:1] == ("data",)
        else (corpus_root / supplied).resolve()
    )
    if not candidate.is_relative_to(corpus_root):
        raise HTTPException(status_code=400, detail="Only synthetic corpus paths may be ingested")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Document not found: {raw_path}")
    return candidate


def _safe_eval_path(raw_path: str) -> Path:
    eval_root = (Path.cwd() / "data" / "eval").resolve()
    dataset_path = (eval_root / raw_path).resolve()
    if not dataset_path.is_relative_to(eval_root) or not dataset_path.is_file():
        raise HTTPException(status_code=404, detail="Evaluation dataset not found")
    return dataset_path


@app.get("/health")
def health(session: SessionDependency) -> dict[str, str]:
    try:
        session.execute(text("SELECT 1"))
        vector_version = session.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
    except Exception as exc:
        logger.error("health_check_failed", error=type(exc).__name__)
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {"status": "ok", "database": "ok", "pgvector": str(vector_version)}


@app.post("/documents/ingest", response_model=list[IngestResultResponse])
def ingest_documents(request: IngestRequest, session: SessionDependency) -> list[Any]:
    settings = get_settings()
    pipeline = IngestionPipeline(
        session,
        FixedTokenChunker(FixedTokenConfig(settings.chunk_size, settings.chunk_overlap)),
        embedding_provider(),
    )
    try:
        results = [
            pipeline.ingest_path(_safe_corpus_path(path), request.tenant_id)
            for path in request.paths
        ]
    except IngestionConflictError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [asdict(result) for result in results]


@app.get("/documents", response_model=list[DocumentResponse])
def list_documents(session: SessionDependency, tenant_id: str = "acmeai") -> list[Any]:
    rows = session.execute(
        select(Document, DocumentVersion)
        .join(DocumentVersion, DocumentVersion.document_fk == Document.id)
        .where(Document.tenant_id == tenant_id)
        .order_by(Document.title, DocumentVersion.effective_at.desc().nullslast())
    ).all()
    return [
        {
            "document_id": document.document_id,
            "title": document.title,
            "source": document.source,
            "source_type": document.source_type,
            "visibility": document.visibility,
            "version": version.version,
            "is_active": version.is_active,
            "content_hash": version.content_hash,
            "updated_at": version.updated_at,
        }
        for document, version in rows
    ]


@app.post("/retrieve", response_model=list[RetrievalResultResponse])
def retrieve(request: RetrieveRequest, session: SessionDependency) -> list[Any]:
    results = _retriever(session).retrieve(
        request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        filters=_filters(request),
        principal=_principal(request),
    )
    session.commit()
    return [asdict(result) for result in results]


@app.post("/rag/query", response_model=RagQueryResponse)
def rag_query(request: RagQueryRequest, session: SessionDependency) -> dict[str, Any]:
    if request.include_debug and not get_settings().rag_admin_debug:
        raise HTTPException(status_code=403, detail="RAG debug mode is disabled")
    try:
        result = _canonical_runtime(session).query(
            request.query,
            principal=_principal(request),
            include_debug=request.include_debug,
        )
    except RuntimeError as exc:
        logger.error("canonical_runtime_config_error", error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    payload = result.to_api_dict(include_debug=request.include_debug)
    logger.info(
        "canonical_rag_query",
        request_id=result.request_id,
        route=result.route,
        status=result.status,
        error_class=result.error_class,
        latency_ms=result.trace.get("latency_ms") if result.trace else None,
    )
    return payload


@app.get("/runs/{run_id}")
def get_run(run_id: str, session: SessionDependency) -> dict[str, Any]:
    run = session.get(RagRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    retrieval = session.scalars(
        select(RetrievalResultRecord)
        .where(RetrievalResultRecord.run_id == run_id)
        .order_by(RetrievalResultRecord.rank)
    ).all()
    return {
        "run_id": run.id,
        "query": run.query,
        "status": run.status,
        "retrieval_config": run.retrieval_config,
        "retrieval_results": [
            {
                "chunk_id": item.chunk_id,
                "rank": item.rank,
                "score": item.score,
                "included_in_context": item.included_in_context,
            }
            for item in retrieval
        ],
        "context_references": run.context_references,
        "retrieved_chunk_ids": run.retrieved_chunk_ids,
        "supporting_chunk_ids": run.supporting_chunk_ids,
        "generation_context_chunk_ids": run.generation_context_chunk_ids,
        "answerability_result": run.answerability_result,
        "answerability_operational_error": run.answerability_operational_error,
        "provider": run.provider,
        "model": run.model,
        "answer": run.answer,
        "citations": run.citations,
        "latency_ms": run.latency_ms,
        "latency": {
            "retrieval_ms": run.retrieval_latency_ms,
            "query_embedding_ms": run.query_embedding_latency_ms,
            "embedding_cache_lookup_ms": run.embedding_cache_lookup_latency_ms,
            "vector_search_ms": run.vector_search_latency_ms,
            "acl_filter_ms": run.acl_filter_latency_ms,
            "context_construction_ms": run.context_construction_latency_ms,
            "answerability_gate_cache_lookup_ms": (run.answerability_gate_cache_lookup_latency_ms),
            "answerability_judge_ms": run.answerability_judge_latency_ms,
            "context_pruning_ms": run.context_pruning_latency_ms,
            "generation_ms": run.generation_latency_ms,
            "total_ms": run.latency_ms,
        },
        "query_embedding_cache_hit": run.query_embedding_cache_hit,
        "gate_cache_hit": run.gate_cache_hit,
        "external_judge_calls": run.external_judge_calls,
        "local_judge_calls": run.local_judge_calls,
        "judge_prompt_tokens": run.judge_prompt_tokens,
        "judge_completion_tokens": run.judge_completion_tokens,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "created_at": run.created_at,
    }


@app.post("/eval/run")
def run_evaluation(request: EvalRunRequest, session: SessionDependency) -> dict[str, object]:
    """Run evaluation cases through CanonicalRagRuntime (same inference as /rag/query)."""
    cases = load_evaluation_dataset(_safe_eval_path(request.dataset))
    # top_k / score_threshold retained on the request schema for API compatibility;
    # CanonicalRagRuntime ignores them and uses ProductionRagConfig stage depths.
    report = EvaluationRunner(_canonical_runtime(session)).run(
        cases, top_k=request.top_k, score_threshold=request.score_threshold
    )
    return report.to_dict()


@app.post("/experiments/run", response_model=ExperimentSummaryResponse)
def run_experiment(request: ExperimentRunRequest, session: SessionDependency) -> dict[str, Any]:
    try:
        result = ExperimentRunner(session).run(
            request.config, _safe_eval_path(request.dataset), request.options
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    run = session.get(ExperimentRunRecord, result.run_id)
    if run is None:
        raise HTTPException(status_code=500, detail="Experiment run was not persisted")
    return experiment_to_dict(session, run, include_cases=False)


@app.get("/experiments", response_model=list[ExperimentSummaryResponse])
def list_experiments(
    session: SessionDependency, limit: int = Query(default=50, ge=1, le=200)
) -> list[dict[str, Any]]:
    runs = session.scalars(
        select(ExperimentRunRecord).order_by(ExperimentRunRecord.started_at.desc()).limit(limit)
    ).all()
    return [experiment_to_dict(session, run, include_cases=False) for run in runs]


@app.get("/experiments/compare", response_model=ExperimentComparisonResponse)
def compare_experiments(
    session: SessionDependency, baseline_id: str, candidate_id: str
) -> dict[str, Any]:
    try:
        return compare_experiment_runs(session, baseline_id, candidate_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/experiments/analysis")
def analyze_experiments(session: SessionDependency) -> dict[str, Any]:
    return pareto_analysis(session)


@app.get("/experiments/evidence-judge")
def evidence_judge_benchmark(session: SessionDependency) -> dict[str, Any]:
    return EvidenceJudgeBenchmark(session).status(include_cases=True)


@app.get("/experiments/multidoc-coverage")
def multidoc_coverage_benchmark(session: SessionDependency) -> dict[str, Any]:
    return MultiDocumentEvidenceBenchmark(session).status(include_cases=True)


@app.get("/experiments/retrieval-comparison")
def retrieval_comparison_benchmark(session: SessionDependency) -> dict[str, Any]:
    return HybridRetrievalBenchmark(session).status(include_cases=True)


@app.get("/experiments/hybrid-reranker-comparison")
def hybrid_reranker_comparison_benchmark(session: SessionDependency) -> dict[str, Any]:
    return HybridRerankerBenchmark(session).status(include_cases=True)


@app.get("/experiments/hybrid-reranker-replication")
def hybrid_reranker_replication_benchmark(session: SessionDependency) -> dict[str, Any]:
    return HybridRerankerReplicationBenchmark(session).status(include_cases=True)


@app.get("/experiments/final-v1-benchmark")
def final_v1_benchmark(session: SessionDependency) -> dict[str, Any]:
    return FinalV1Benchmark(session).status(include_cases=True)


@app.get("/experiments/v1-architecture")
def v1_release_architecture(session: SessionDependency) -> dict[str, Any]:
    status = FinalV1Benchmark(session).status()
    architecture = status.get("release_architecture")
    if not architecture:
        raise HTTPException(status_code=409, detail="v1 release architecture is not frozen")
    return architecture


@app.get("/experiments/retrieval-architecture")
def retrieval_architecture(session: SessionDependency) -> dict[str, Any]:
    status = HybridRerankerReplicationBenchmark(session).status()
    architecture = status.get("architecture")
    if not architecture:
        raise HTTPException(status_code=409, detail="v1 retrieval architecture is not frozen")
    return architecture


@app.get(
    "/experiments/hybrid-reranker-comparison/benchmark.md",
    response_class=PlainTextResponse,
)
def hybrid_reranker_benchmark_export(session: SessionDependency) -> str:
    status = HybridRerankerBenchmark(session).status()
    if not status.get("e2e_completed"):
        raise HTTPException(status_code=409, detail="Hybrid reranker benchmark is not complete")
    return Path("BENCHMARK.md").read_text()


@app.get(
    "/experiments/hybrid-reranker-replication/benchmark.md",
    response_class=PlainTextResponse,
)
def hybrid_reranker_replication_export(session: SessionDependency) -> str:
    status = HybridRerankerReplicationBenchmark(session).status()
    if not status.get("e2e_completed") or not status.get("architecture_frozen"):
        raise HTTPException(status_code=409, detail="Hybrid reranker replication is not complete")
    return Path("BENCHMARK.md").read_text()


@app.get(
    "/experiments/final-v1-benchmark/benchmark.md",
    response_class=PlainTextResponse,
)
def final_v1_benchmark_export(session: SessionDependency) -> str:
    status = FinalV1Benchmark(session).status()
    if not status.get("completed"):
        raise HTTPException(status_code=409, detail="Final v1 benchmark is not complete")
    return Path("BENCHMARK.md").read_text()


@app.get("/experiments/v2-research")
def v2_research_baseline(session: SessionDependency) -> dict[str, Any]:
    return V2QualityRecoveryBaseline(session).status(include_cases=True)


@app.get("/experiments/v2-phase1")
def v2_phase1_document_diversity(session: SessionDependency) -> dict[str, Any]:
    return V2DocumentDiversityBenchmark(session).status(include_cases=True)


@app.get("/experiments/v2-phase2")
def v2_phase2_soft_document_cap(session: SessionDependency) -> dict[str, Any]:
    return V2SoftDocumentCapBenchmark(session).status(include_cases=True)


@app.get("/experiments/v2-phase3")
def v2_phase3_sufficiency_fn(session: SessionDependency) -> dict[str, Any]:
    return V2SufficiencyFnBenchmark(session).status(include_cases=True)


@app.get("/experiments/v2-phase4")
def v2_phase4_reliability(session: SessionDependency) -> dict[str, Any]:
    return V2ReliabilityHardening(session).status(include_cases=True)


@app.get("/experiments/v2-final-benchmark")
def v2_final_benchmark(session: SessionDependency) -> dict[str, Any]:
    return V2FinalBenchmark(session).status(include_cases=True)


@app.get("/experiments/v2-final-architecture")
def v2_final_architecture(session: SessionDependency) -> dict[str, Any]:
    status = V2FinalBenchmark(session).status()
    architecture = status.get("architecture_configuration") or status.get("architecture")
    if not architecture:
        raise HTTPException(status_code=409, detail="v2 final architecture is not frozen")
    return architecture


@app.get(
    "/experiments/v2-final-benchmark/benchmark.md",
    response_class=PlainTextResponse,
)
def v2_final_benchmark_export(session: SessionDependency) -> str:
    status = V2FinalBenchmark(session).status()
    if not status.get("completed"):
        raise HTTPException(status_code=409, detail="Final v2 benchmark is not complete")
    return Path("BENCHMARK.md").read_text()


@app.get("/experiments/v2-architecture")
def v2_research_architecture(session: SessionDependency) -> dict[str, Any]:
    status = V2QualityRecoveryBaseline(session).status()
    if not status.get("initialized"):
        raise HTTPException(status_code=409, detail="v2 research identity is not initialized")
    return status


@app.get(
    "/experiments/v2-research/benchmark.md",
    response_class=PlainTextResponse,
)
def v2_research_benchmark_export(session: SessionDependency) -> str:
    status = V2QualityRecoveryBaseline(session).status()
    if not status.get("initialized"):
        raise HTTPException(status_code=409, detail="v2 research identity is not initialized")
    return Path("BENCHMARK.md").read_text()


@app.get("/experiments/v2-quality-ab")
def v2_quality_ab(session: SessionDependency) -> dict[str, Any]:
    return V2QualityAbBenchmark(session).status(include_cases=True)


@app.get(
    "/experiments/v2-quality-ab/benchmark.md",
    response_class=PlainTextResponse,
)
def v2_quality_ab_export(session: SessionDependency) -> str:
    status = V2QualityAbBenchmark(session).status()
    if not status.get("completed"):
        raise HTTPException(status_code=409, detail="v2 quality A/B research is not complete")
    return Path("BENCHMARK.md").read_text()


@app.get("/experiments/v3-research")
def v3_research(session: SessionDependency) -> dict[str, Any]:
    return V3GenerateVerifyBenchmark(session).status(include_cases=True)


@app.get("/experiments/v3-phase1")
def v3_phase1(session: SessionDependency) -> dict[str, Any]:
    return V3GenerateVerifyBenchmark(session).status(include_cases=True)


@app.get(
    "/experiments/v3-phase1/benchmark.md",
    response_class=PlainTextResponse,
)
def v3_phase1_export(session: SessionDependency) -> str:
    status = V3GenerateVerifyBenchmark(session).status()
    if not status.get("initialized"):
        raise HTTPException(status_code=409, detail="v3 phase 1 research is not initialized")
    return Path("BENCHMARK.md").read_text()


@app.get("/experiments/reranking-comparison")
def reranking_comparison_benchmark(session: SessionDependency) -> dict[str, Any]:
    return DenseCrossEncoderBenchmark(session).status(include_cases=True)


@app.get("/experiments/reranker-e2e-comparison")
def reranker_e2e_comparison_benchmark(session: SessionDependency) -> dict[str, Any]:
    return RerankerEndToEndBenchmark(session).status(include_cases=True)


@app.get("/experiments/judge-e2e-comparison")
def judge_e2e_comparison_benchmark(session: SessionDependency) -> dict[str, Any]:
    return FrozenJudgeEndToEndBenchmark(session).status(include_cases=True)


@app.get("/experiments/sol-judge-e2e-comparison")
def sol_judge_e2e_comparison_benchmark(session: SessionDependency) -> dict[str, Any]:
    return SolJudgeEndToEndBenchmark(session).status(include_cases=True)


@app.get(
    "/experiments/sol-judge-e2e-comparison/benchmark.md",
    response_class=PlainTextResponse,
)
def sol_judge_e2e_benchmark_export(session: SessionDependency) -> str:
    status = SolJudgeEndToEndBenchmark(session).status()
    if not status["completed"]:
        raise HTTPException(status_code=409, detail="Sol judge benchmark is not complete")
    return Path("BENCHMARK.md").read_text()


@app.get(
    "/experiments/judge-e2e-comparison/benchmark.md",
    response_class=PlainTextResponse,
)
def judge_e2e_benchmark_export(session: SessionDependency) -> str:
    status = FrozenJudgeEndToEndBenchmark(session).status()
    if not status["completed"]:
        raise HTTPException(status_code=409, detail="Judge end-to-end benchmark is not complete")
    return Path("BENCHMARK.md").read_text()


@app.get(
    "/experiments/reranker-e2e-comparison/benchmark.md",
    response_class=PlainTextResponse,
)
def reranker_e2e_benchmark_export(session: SessionDependency) -> str:
    status = RerankerEndToEndBenchmark(session).status()
    if not status["completed"]:
        raise HTTPException(status_code=409, detail="End-to-end benchmark is not complete")
    return Path("BENCHMARK.md").read_text()


@app.get("/experiments/{experiment_id}", response_model=ExperimentDetailResponse)
def get_experiment(experiment_id: str, session: SessionDependency) -> dict[str, Any]:
    run = session.get(ExperimentRunRecord, experiment_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment_to_dict(session, run, include_cases=True)


@app.get("/experiments/{experiment_id}/export", response_class=PlainTextResponse)
def export_experiment(experiment_id: str, session: SessionDependency) -> str:
    try:
        return render_benchmark_markdown(session, experiment_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
