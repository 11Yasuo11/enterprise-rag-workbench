from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from rag_workbench.experiments.configs import ExperimentConfig
from rag_workbench.experiments.runner import ExperimentExecutionOptions


class PrincipalInput(BaseModel):
    principal_id: str = "local-developer"
    tenant_id: str = "acmeai"
    permission_groups: list[str] = Field(default_factory=lambda: ["employees"])


class RetrievalFilterInput(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float | None = Field(default=0.2, ge=-1, le=1)
    filters: RetrievalFilterInput = Field(default_factory=RetrievalFilterInput)
    principal: PrincipalInput = Field(default_factory=PrincipalInput)


class RetrievalResultResponse(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    rank: int
    score: float
    source: str
    source_type: str
    title: str
    version: str
    page: int | None
    section: str | None
    metadata: dict[str, Any]


class IngestRequest(BaseModel):
    paths: list[str] = Field(min_length=1)
    tenant_id: str = "acmeai"


class IngestResultResponse(BaseModel):
    document_id: str
    document_version_id: str
    version: str
    chunks_created: int
    status: str
    content_hash: str
    is_active: bool


class DocumentResponse(BaseModel):
    document_id: str
    title: str
    source: str
    source_type: str
    visibility: str
    version: str
    is_active: bool
    content_hash: str
    updated_at: datetime


class CitationResponse(BaseModel):
    document_id: str
    chunk_id: str
    source: str
    title: str
    version: str
    page: int | None
    section: str | None


class RagQueryResponse(BaseModel):
    request_id: str
    run_id: str
    status: Literal["answer", "abstain", "unavailable", "answered", "abstained"]
    answer: str | None
    citations: list[CitationResponse]
    requirements: list[dict[str, Any]] = Field(default_factory=list)
    route: Literal["deterministic", "luna", "sol", "abstain"] = "abstain"
    error_class: str | None = None
    retrieval_results: list[RetrievalResultResponse] = Field(default_factory=list)
    final_context: str | None = None
    answerability_result: dict[str, Any] | None = None
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    generation_context_chunk_ids: list[str] = Field(default_factory=list)
    answerability_operational_error: str | None = None
    question_plan: dict[str, Any] | None = None
    trace: dict[str, Any] | None = None


class RagQueryRequest(RetrieveRequest):
    include_debug: bool = False
    # Legacy top_k is ignored by CanonicalRagRuntime (explicit stage depths apply).
    top_k: int = Field(default=15, ge=1, le=100)


class EvalRunRequest(BaseModel):
    dataset: str = "initial.json"
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float | None = Field(default=0.2, ge=-1, le=1)


class ExperimentRunRequest(BaseModel):
    config: ExperimentConfig
    dataset: str = "eval_v1.json"
    options: ExperimentExecutionOptions = Field(default_factory=ExperimentExecutionOptions)


class ExperimentSummaryResponse(BaseModel):
    id: str
    name: str
    status: str
    started_at: datetime
    completed_at: datetime | None
    config: dict[str, Any]
    aggregate_metrics: dict[str, Any]
    category_metrics: dict[str, Any]
    latency: dict[str, float | None]
    usage: dict[str, int | None]
    failure_counts: dict[str, int]
    error: str | None


class ExperimentDetailResponse(ExperimentSummaryResponse):
    cases: list[dict[str, Any]]


class ExperimentComparisonResponse(BaseModel):
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    metrics: dict[str, dict[str, float | None]]
    latency_delta_ms: float | None
    regressions: list[dict[str, Any]]
    category_metrics: dict[str, Any]
    configuration_differences: list[dict[str, Any]]
    hard_constraints: dict[str, Any]
    passed: bool
    threshold_policy: str
