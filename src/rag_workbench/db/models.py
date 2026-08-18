import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

STORED_EMBEDDING_DIMENSION = 64


class Base(DeclarativeBase):
    pass


def uuid_string() -> str:
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("tenant_id", "document_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    document_id: Mapped[str] = mapped_column(String(200), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[str] = mapped_column(String(1000), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(2000))
    visibility: Mapped[str] = mapped_column(String(20), default="public", nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_fk", "version"),
        UniqueConstraint("document_fk", "content_hash"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    document_fk: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    document: Mapped[Document] = relationship(back_populates="versions")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document_version", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "index_identity",
            "chunk_index",
            name="uq_chunks_version_index_chunk",
        ),
        Index("ix_chunks_document_active_lookup", "document_fk", "document_version_id"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    document_fk: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    document_version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer)
    section: Mapped[str | None] = mapped_column(String(500))
    embedding: Mapped[list[float]] = mapped_column(VECTOR(STORED_EMBEDDING_DIMENSION))
    embedding_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)
    embedding_version: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    index_identity: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    document_version: Mapped[DocumentVersion] = relationship(back_populates="chunks")


class DocumentPermission(Base):
    __tablename__ = "document_permissions"
    __table_args__ = (UniqueConstraint("document_fk", "permission_group"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    document_fk: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    permission_group: Mapped[str] = mapped_column(String(100), nullable=False, index=True)


class RagRun(Base):
    __tablename__ = "rag_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    tenant_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    principal_id: Mapped[str] = mapped_column(String(200), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    retrieval_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    context_references: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    supporting_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    generation_context_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    answerability_result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    answerability_operational_error: Mapped[str | None] = mapped_column(String(100))
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    retrieval_latency_ms: Mapped[float | None] = mapped_column(Float)
    query_embedding_latency_ms: Mapped[float | None] = mapped_column(Float)
    embedding_cache_lookup_latency_ms: Mapped[float | None] = mapped_column(Float)
    vector_search_latency_ms: Mapped[float | None] = mapped_column(Float)
    acl_filter_latency_ms: Mapped[float | None] = mapped_column(Float)
    context_construction_latency_ms: Mapped[float | None] = mapped_column(Float)
    answerability_gate_cache_lookup_latency_ms: Mapped[float | None] = mapped_column(Float)
    answerability_judge_latency_ms: Mapped[float | None] = mapped_column(Float)
    context_pruning_latency_ms: Mapped[float | None] = mapped_column(Float)
    generation_latency_ms: Mapped[float | None] = mapped_column(Float)
    query_embedding_cache_hit: Mapped[bool | None] = mapped_column(Boolean)
    gate_cache_hit: Mapped[bool | None] = mapped_column(Boolean)
    external_judge_calls: Mapped[int | None] = mapped_column(Integer)
    local_judge_calls: Mapped[int | None] = mapped_column(Integer)
    judge_prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    judge_completion_tokens: Mapped[int | None] = mapped_column(Integer)
    embedding_tokens: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    retrieval_results: Mapped[list["RetrievalResultRecord"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class RetrievalResultRecord(Base):
    __tablename__ = "retrieval_results"
    __table_args__ = (UniqueConstraint("run_id", "rank"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    run_id: Mapped[str] = mapped_column(ForeignKey("rag_runs.id", ondelete="CASCADE"))
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"))
    document_fk: Mapped[str] = mapped_column(ForeignKey("documents.id"))
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    included_in_context: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    run: Mapped[RagRun] = relationship(back_populates="retrieval_results")


class ExperimentConfigRecord(Base):
    __tablename__ = "experiment_configs"
    __table_args__ = (
        UniqueConstraint("config_hash", name="experiment_configs_config_hash_key"),
        Index("ix_experiment_configs_config_hash", "config_hash"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    ingestion_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    retrieval_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    generation_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    gate_config_hash: Mapped[str | None] = mapped_column(String(64))
    index_identity: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    corpus_version: Mapped[str] = mapped_column(String(100), nullable=False)
    evaluation_dataset_version: Mapped[str] = mapped_column(String(100), nullable=False)
    ingestion_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    retrieval_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    generation_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    gate_config: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    runs: Mapped[list["ExperimentRunRecord"]] = relationship(back_populates="config")


class ExperimentRunRecord(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    config_id: Mapped[str] = mapped_column(
        ForeignKey("experiment_configs.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    aggregate_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    category_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    total_latency_ms: Mapped[float | None] = mapped_column(Float)
    retrieval_latency_ms: Mapped[float | None] = mapped_column(Float)
    query_embedding_latency_ms: Mapped[float | None] = mapped_column(Float)
    embedding_cache_lookup_latency_ms: Mapped[float | None] = mapped_column(Float)
    vector_search_latency_ms: Mapped[float | None] = mapped_column(Float)
    acl_filter_latency_ms: Mapped[float | None] = mapped_column(Float)
    context_construction_latency_ms: Mapped[float | None] = mapped_column(Float)
    answerability_gate_cache_lookup_latency_ms: Mapped[float | None] = mapped_column(Float)
    answerability_judge_latency_ms: Mapped[float | None] = mapped_column(Float)
    context_pruning_latency_ms: Mapped[float | None] = mapped_column(Float)
    generation_latency_ms: Mapped[float | None] = mapped_column(Float)
    query_embedding_cache_hits: Mapped[int | None] = mapped_column(Integer)
    query_embedding_cache_misses: Mapped[int | None] = mapped_column(Integer)
    external_embedding_calls: Mapped[int | None] = mapped_column(Integer)
    gate_cache_hits: Mapped[int | None] = mapped_column(Integer)
    gate_cache_misses: Mapped[int | None] = mapped_column(Integer)
    external_judge_calls: Mapped[int | None] = mapped_column(Integer)
    local_judge_calls: Mapped[int | None] = mapped_column(Integer)
    judge_prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    judge_completion_tokens: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    embedding_tokens: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    config: Mapped[ExperimentConfigRecord] = relationship(back_populates="runs")
    case_results: Mapped[list["ExperimentCaseResultRecord"]] = relationship(
        back_populates="experiment_run", cascade="all, delete-orphan"
    )


class ExperimentCaseResultRecord(Base):
    __tablename__ = "experiment_case_results"
    __table_args__ = (UniqueConstraint("experiment_run_id", "eval_case_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    experiment_run_id: Mapped[str] = mapped_column(
        ForeignKey("experiment_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rag_run_id: Mapped[str | None] = mapped_column(ForeignKey("rag_runs.id", ondelete="SET NULL"))
    eval_case_id: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text)
    expected_document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    expected_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    forbidden_document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    expected_versions: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    principal: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    retrieved_document_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    supporting_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    generation_context_chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    answerability_result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    answerability_operational_error: Mapped[str | None] = mapped_column(String(100))
    retrieval_trace: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    answer: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    expected_abstain: Mapped[bool] = mapped_column(Boolean, nullable=False)
    abstained: Mapped[bool | None] = mapped_column(Boolean)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    failure_type: Mapped[str | None] = mapped_column(String(100))
    failure_types: Mapped[list[str]] = mapped_column(JSON, default=list)
    failure_details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    security_passed: Mapped[bool | None] = mapped_column(Boolean)
    retrieval_latency_ms: Mapped[float | None] = mapped_column(Float)
    query_embedding_latency_ms: Mapped[float | None] = mapped_column(Float)
    embedding_cache_lookup_latency_ms: Mapped[float | None] = mapped_column(Float)
    vector_search_latency_ms: Mapped[float | None] = mapped_column(Float)
    acl_filter_latency_ms: Mapped[float | None] = mapped_column(Float)
    context_construction_latency_ms: Mapped[float | None] = mapped_column(Float)
    answerability_gate_cache_lookup_latency_ms: Mapped[float | None] = mapped_column(Float)
    answerability_judge_latency_ms: Mapped[float | None] = mapped_column(Float)
    context_pruning_latency_ms: Mapped[float | None] = mapped_column(Float)
    generation_latency_ms: Mapped[float | None] = mapped_column(Float)
    query_embedding_cache_hit: Mapped[bool | None] = mapped_column(Boolean)
    gate_cache_hit: Mapped[bool | None] = mapped_column(Boolean)
    external_judge_calls: Mapped[int | None] = mapped_column(Integer)
    local_judge_calls: Mapped[int | None] = mapped_column(Integer)
    judge_prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    judge_completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_latency_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    experiment_run: Mapped[ExperimentRunRecord] = relationship(back_populates="case_results")


class QueryEmbeddingCacheRecord(Base):
    __tablename__ = "query_embedding_cache"
    __table_args__ = (
        UniqueConstraint(
            "normalized_query",
            "embedding_provider",
            "embedding_model",
            "embedding_version",
            "embedding_dimension",
            name="uq_query_embedding_cache_identity",
        ),
    )

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    normalized_query: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(200), nullable=False)
    embedding_version: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(STORED_EMBEDDING_DIMENSION))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AnswerabilityGateCacheRecord(Base):
    __tablename__ = "answerability_gate_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    normalized_question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    judge_provider: Mapped[str] = mapped_column(String(100), nullable=False)
    judge_model: Mapped[str] = mapped_column(String(200), nullable=False)
    judge_version: Mapped[str] = mapped_column(String(100), nullable=False)
    judge_prompt_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_render_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    operational_error: Mapped[str | None] = mapped_column(String(100))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EvidenceBenchmarkLockRecord(Base):
    __tablename__ = "evidence_benchmark_locks"

    split_identity: Mapped[str] = mapped_column(String(64), primary_key=True)
    selected_candidate: Mapped[str] = mapped_column(String(2), nullable=False)
    selected_configuration_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    judge_provider: Mapped[str | None] = mapped_column(String(100))
    judge_model: Mapped[str | None] = mapped_column(String(200))
    judge_version: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(100))
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    supporting_context_only: Mapped[bool] = mapped_column(Boolean, nullable=False)
    calibration_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    selection_reason: Mapped[str] = mapped_column(Text, nullable=False)
    locked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    holdout_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    holdout_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    holdout_baseline_run_id: Mapped[str | None] = mapped_column(String(36))
    holdout_winner_run_id: Mapped[str | None] = mapped_column(String(36))


class MultiDocumentBenchmarkRecord(Base):
    __tablename__ = "multidoc_benchmark_locks"

    dataset_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    split_identity: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    split_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    calibration_case_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    holdout_case_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    control_configuration_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_configuration_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    candidate_prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    calibration_control_run_id: Mapped[str | None] = mapped_column(String(36))
    calibration_candidate_run_id: Mapped[str | None] = mapped_column(String(36))
    selected_candidate: Mapped[str | None] = mapped_column(String(2))
    selected_configuration_hash: Mapped[str | None] = mapped_column(String(64))
    calibration_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selection_reason: Mapped[str | None] = mapped_column(Text)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    holdout_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    holdout_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    holdout_control_run_id: Mapped[str | None] = mapped_column(String(36))
    holdout_selected_run_id: Mapped[str | None] = mapped_column(String(36))


class RetrievalBenchmarkRecord(Base):
    __tablename__ = "retrieval_benchmark_locks"

    dataset_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    split_identity: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    split_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    calibration_case_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    holdout_case_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    semantic_index_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    corpus_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    retrieval_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    selection_policy: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    selected_retrieval_mode: Mapped[str | None] = mapped_column(String(30))
    calibration_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selection_reason: Mapped[str | None] = mapped_column(Text)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    holdout_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    holdout_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    runs: Mapped[list["RetrievalBenchmarkRunRecord"]] = relationship(
        back_populates="benchmark", cascade="all, delete-orphan"
    )


class RetrievalBenchmarkRunRecord(Base):
    __tablename__ = "retrieval_benchmark_runs"
    __table_args__ = (UniqueConstraint("dataset_id", "partition", "retrieval_mode"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_benchmark_locks.dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    partition: Mapped[str] = mapped_column(String(20), nullable=False)
    retrieval_mode: Mapped[str] = mapped_column(String(30), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    category_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    case_results: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    branch_contribution: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    latency: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    usage: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    benchmark: Mapped[RetrievalBenchmarkRecord] = relationship(back_populates="runs")


class RerankingBenchmarkRecord(Base):
    __tablename__ = "reranking_benchmark_locks"

    dataset_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    split_identity: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    split_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    calibration_case_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    holdout_case_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    semantic_index_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    corpus_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    selection_policy: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    model_revision: Mapped[str | None] = mapped_column(String(100))
    model_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selected_mode: Mapped[str | None] = mapped_column(String(40))
    calibration_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selection_reason: Mapped[str | None] = mapped_column(Text)
    frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    holdout_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    holdout_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    runs: Mapped[list["RerankingBenchmarkRunRecord"]] = relationship(
        back_populates="benchmark", cascade="all, delete-orphan"
    )


class RerankingBenchmarkRunRecord(Base):
    __tablename__ = "reranking_benchmark_runs"
    __table_args__ = (UniqueConstraint("dataset_id", "partition", "mode"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("reranking_benchmark_locks.dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    partition: Mapped[str] = mapped_column(String(20), nullable=False)
    mode: Mapped[str] = mapped_column(String(40), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    category_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    candidate_pool_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    rerankable_subset_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    movement_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    case_results: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    latency: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    usage: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    benchmark: Mapped[RerankingBenchmarkRecord] = relationship(back_populates="runs")


class EndToEndBenchmarkRecord(Base):
    __tablename__ = "reranker_e2e_benchmark_locks"

    dataset_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    case_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    category_distribution: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    generation_method: Mapped[str] = mapped_column(String(100), nullable=False)
    maximum_prior_overlap: Mapped[float] = mapped_column(Float, nullable=False)
    semantic_index_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    corpus_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    reranker_revision: Mapped[str] = mapped_column(String(100), nullable=False)
    pipeline_a_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    pipeline_b_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    embedding_preflight: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    judge_preflight: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    prepared_cases: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    preparation_usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    paired_transitions: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    conversion_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    regression_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    production_retriever_status: Mapped[str | None] = mapped_column(String(40))
    frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    preparation_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    prepared_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    execution_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    runs: Mapped[list["EndToEndBenchmarkRunRecord"]] = relationship(
        back_populates="benchmark", cascade="all, delete-orphan"
    )


class EndToEndBenchmarkRunRecord(Base):
    __tablename__ = "reranker_e2e_benchmark_runs"
    __table_args__ = (UniqueConstraint("dataset_id", "mode"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("reranker_e2e_benchmark_locks.dataset_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    mode: Mapped[str] = mapped_column(String(40), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    retrieval_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    category_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    case_results: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    latency: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    usage: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    benchmark: Mapped[EndToEndBenchmarkRecord] = relationship(back_populates="runs")


class RetrievalArchitectureRecord(Base):
    __tablename__ = "retrieval_architecture_locks"

    architecture_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    selected_retriever: Mapped[str] = mapped_column(String(40), nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    selection_policy: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(100), nullable=False)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    retrieval_metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    immutable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ResearchArchitectureRecord(Base):
    __tablename__ = "research_architecture_locks"

    architecture_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    parent_architecture_id: Mapped[str] = mapped_column(String(100), nullable=False)
    selected_retriever: Mapped[str] = mapped_column(String(40), nullable=False)
    research_status: Mapped[str] = mapped_column(String(20), nullable=False)
    production_status: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    control_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    control_equivalence_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    diagnosis_dataset_id: Mapped[str] = mapped_column(String(100), nullable=False)
    diagnosis_dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    diagnosis_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    promotion_evidence_forbidden: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    security_guardrails: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    v1_preservation: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    failure_census: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    ranking_diagnostic: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    judge_false_negative_diagnostic: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    operational_diagnostic: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    primary_bottleneck: Mapped[str] = mapped_column(String(80), nullable=False)
    recommended_ranking_intervention: Mapped[str] = mapped_column(Text, nullable=False)
    selected_v2_ranking: Mapped[str | None] = mapped_column(String(40))
    phase1_dataset_id: Mapped[str | None] = mapped_column(String(100))
    phase2_dataset_id: Mapped[str | None] = mapped_column(String(100))
    ranking_research_status: Mapped[str | None] = mapped_column(String(40))
    selected_v2_judge: Mapped[str | None] = mapped_column(String(80))
    phase3_dataset_id: Mapped[str | None] = mapped_column(String(100))
    judge_research_status: Mapped[str | None] = mapped_column(String(40))
    phase4_lock_id: Mapped[str | None] = mapped_column(String(100))
    reliability_research_status: Mapped[str | None] = mapped_column(String(40))
    final_v2_architecture_id: Mapped[str | None] = mapped_column(String(100))
    final_v2_dataset_id: Mapped[str | None] = mapped_column(String(100))
    selected_v3_strategy: Mapped[str | None] = mapped_column(String(80))
    v3_phase1_dataset_id: Mapped[str | None] = mapped_column(String(100))
    v3_research_status: Mapped[str | None] = mapped_column(String(40))
    immutable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class V2Phase1ExperimentRecord(Base):
    __tablename__ = "v2_phase1_experiment_locks"

    dataset_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    architecture_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    case_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    category_distribution: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    generation_method: Mapped[str] = mapped_column(String(100), nullable=False)
    maximum_prior_overlap: Mapped[float] = mapped_column(Float, nullable=False)
    closest_previous_case: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    overlap_report: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    selection_policy: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    control_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    candidate_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    semantic_index_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    corpus_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_preflight: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    shared_traces: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    control_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    candidate_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    diversity_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    crowding_rescues: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    diversification_regressions: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    same_document_multichunk: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    three_document_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    exact_id_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    near_duplicate_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    semantic_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    version_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    security: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    latency: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    local_compute: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selection: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selected_ranking: Mapped[str | None] = mapped_column(String(40))
    primary_remaining_bottleneck: Mapped[str | None] = mapped_column(String(80))
    dataset_frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    selection_policy_frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    execution_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class V2Phase2ExperimentRecord(Base):
    __tablename__ = "v2_phase2_experiment_locks"

    dataset_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    architecture_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    case_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    category_distribution: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    generation_method: Mapped[str] = mapped_column(String(100), nullable=False)
    maximum_prior_overlap: Mapped[float] = mapped_column(Float, nullable=False)
    closest_previous_case: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    overlap_report: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    selection_policy: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    control_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    candidate_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    semantic_index_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    corpus_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_preflight: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    shared_traces: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    control_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    candidate_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    occupancy_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    crowding_rescues: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    diversification_regressions: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    same_document_two_chunk: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    three_document_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    missing_evidence_rank_distribution: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    exact_id_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    near_duplicate_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    semantic_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    version_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    security: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    latency: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    local_compute: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selection: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selected_ranking: Mapped[str | None] = mapped_column(String(40))
    ranking_research_status: Mapped[str | None] = mapped_column(String(40))
    primary_remaining_bottleneck: Mapped[str | None] = mapped_column(String(80))
    dataset_frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    selection_policy_frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    execution_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resume_checkpoint: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class V2Phase3ExperimentRecord(Base):
    __tablename__ = "v2_phase3_experiment_locks"

    dataset_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    architecture_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    case_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    category_distribution: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    generation_method: Mapped[str] = mapped_column(String(100), nullable=False)
    maximum_prior_overlap: Mapped[float] = mapped_column(Float, nullable=False)
    closest_previous_case: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    overlap_report: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    selection_policy: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    control_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    candidate_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    control_prompt_identity: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    candidate_prompt_identity: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    semantic_index_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    corpus_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_preflight: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    judge_preflight: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    shared_traces: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    control_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    candidate_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    retrieval_results: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    retrieval_complete_judge: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    all_case_answerability: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    exact_id_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    three_document_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    two_document_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    near_duplicate_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    semantic_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    version_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    false_negative_rescues: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    false_positive_regressions: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    answerable_to_abstain_regressions: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    supporting_id_quality: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    abstention_safety: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    prompt_injection_safety: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    security: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    end_to_end_confirmation: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    generation_failures: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    latency: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    cost: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selection: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selected_judge: Mapped[str | None] = mapped_column(String(80))
    judge_research_status: Mapped[str | None] = mapped_column(String(40))
    primary_remaining_bottleneck: Mapped[str | None] = mapped_column(String(80))
    dataset_frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    selection_policy_frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    retrieval_frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieval_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    judge_execution_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class V2Phase4ExperimentRecord(Base):
    __tablename__ = "v2_phase4_experiment_locks"

    lock_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    architecture_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    selected_v2_ranking: Mapped[str] = mapped_column(String(40), nullable=False)
    selected_v2_judge: Mapped[str] = mapped_column(String(80), nullable=False)
    ranking_research_status: Mapped[str] = mapped_column(String(40), nullable=False)
    judge_research_status: Mapped[str] = mapped_column(String(40), nullable=False)
    generator_parent: Mapped[str] = mapped_column(String(80), nullable=False)
    generator_revision: Mapped[dict[str, Any] | str] = mapped_column(JSON, nullable=False)
    fixture_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    historical_generator_failure: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    generator_root_cause: Mapped[str] = mapped_column(String(80), nullable=False)
    provider_failure: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    transport_retry_policy: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    taxonomy: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    security: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    reliability_status: Mapped[str | None] = mapped_column(String(40))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class V2FinalBenchmarkRecord(Base):
    __tablename__ = "v2_final_benchmark_locks"

    dataset_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    architecture_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    parent_architecture_id: Mapped[str] = mapped_column(String(100), nullable=False)
    architecture_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    architecture_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    case_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    category_distribution: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    generation_method: Mapped[str] = mapped_column(String(100), nullable=False)
    maximum_prior_overlap: Mapped[float] = mapped_column(Float, nullable=False)
    closest_previous_case: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    overlap_report: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    semantic_index_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    corpus_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    one_shot: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    dataset_frozen: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    embedding_preflight: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    judge_preflight: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    retrieval_traces: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    retrieval_results: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    case_results: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    end_to_end: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    category_results: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    stage_funnel: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    failure_taxonomy: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    generator_reliability: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    provider_reliability: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    security: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    citations: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    latency: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    cost: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    v1_comparison: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    primary_remaining_bottleneck: Mapped[str | None] = mapped_column(String(80))
    dataset_frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    architecture_frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    one_shot_locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieval_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieval_frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    execution_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class V2QualityAbExperimentRecord(Base):
    __tablename__ = "v2_quality_ab_experiment_locks"

    lock_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    architecture_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    parent_architecture_id: Mapped[str] = mapped_column(String(100), nullable=False)
    git_commit: Mapped[str | None] = mapped_column(String(64))
    baseline_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    corpus_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    selection_policy: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    baseline: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    experiments: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    failure_census: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    safety: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    verdicts: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    final_candidate: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    holdout: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    v1_preservation: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class RecoveryStageCacheRecord(Base):
    __tablename__ = "recovery_stage_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    stage: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    normalized_question: Mapped[str] = mapped_column(Text, nullable=False)
    ordered_top5: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(200), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    operational_error: Mapped[str | None] = mapped_column(String(100))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class V3Phase1ExperimentRecord(Base):
    __tablename__ = "v3_phase1_experiment_locks"

    lock_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    architecture_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    parent_architecture_id: Mapped[str] = mapped_column(String(100), nullable=False)
    production_status: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    diagnosis_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    dataset_id: Mapped[str | None] = mapped_column(String(100))
    dataset_hash: Mapped[str | None] = mapped_column(String(64))
    case_ids: Mapped[list[str] | None] = mapped_column(JSON)
    category_distribution: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    generation_method: Mapped[str | None] = mapped_column(String(100))
    maximum_prior_overlap: Mapped[float | None] = mapped_column(Float)
    closest_previous_case: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    overlap_report: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selection_policy: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    control_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    candidate_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    semantic_index_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    corpus_identity: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_preflight: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    hosted_preflight: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    diagnostic: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    shared_traces: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    control_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    candidate_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    recovery_funnel: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    valid_rescues: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    false_positive_recoveries: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    completeness_failures: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    category_results: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    security: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    citations: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    latency: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    cost: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selection: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    selected_strategy: Mapped[str | None] = mapped_column(String(80))
    go_nogo: Mapped[str | None] = mapped_column(String(40))
    primary_remaining_bottleneck: Mapped[str | None] = mapped_column(String(80))
    dataset_frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    selection_policy_frozen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    diagnostic_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    diagnostic_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieval_frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    execution_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

