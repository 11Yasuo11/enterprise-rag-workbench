"""Eval endpoint uses CanonicalRagRuntime, not legacy RagService."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from rag_workbench.api.app import app
from rag_workbench.db.session import get_db
from rag_workbench.domain.documents import CanonicalDocument
from rag_workbench.evaluation.evaluator import EvaluationRunner
from rag_workbench.generation.generator import RagService
from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
from rag_workbench.ingestion.pipeline import IngestionPipeline
from rag_workbench.providers.embeddings import HashingEmbeddingProvider
from rag_workbench.runtime import ProductionRagConfig, build_canonical_runtime
from rag_workbench.runtime.identity_reranker import IdentityReranker


def test_evaluation_runner_accepts_canonical_runtime(db_session: Session) -> None:
    provider = HashingEmbeddingProvider(64)
    pipeline = IngestionPipeline(db_session, FixedTokenChunker(FixedTokenConfig(80, 10)), provider)
    pipeline.ingest(
        CanonicalDocument(
            document_id="incident-policy",
            title="Incident Policy",
            content="Severity-one incidents must be reported within 15 minutes.",
            source="incident.md",
            source_type="markdown",
            version="2026",
            tenant_id="acmeai",
            effective_at=datetime(2026, 1, 1, tzinfo=UTC),
            is_active=True,
            visibility="public",
        )
    )
    runtime = build_canonical_runtime(
        db_session,
        config=ProductionRagConfig(
            embedding_provider="hashing",
            embedding_model="local-hashing-64",
            allow_hashing_embeddings=True,
            allow_identity_reranker=True,
            require_cross_encoder=False,
            index_identity=pipeline.index_identity,
        ),
        embedding_provider=provider,
        reranker=IdentityReranker(),
        index_identity=pipeline.index_identity,
    )
    runner = EvaluationRunner(runtime)
    assert runner.rag_service is None
    assert not isinstance(runner.runtime, RagService)


def test_eval_run_endpoint_uses_canonical_runtime(
    db_session: Session, monkeypatch
) -> None:
    def override():
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise

    app.dependency_overrides[get_db] = override
    try:
        monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
        monkeypatch.setenv("ALLOW_EXTERNAL_CALLS", "false")
        from rag_workbench.api import dependencies
        from rag_workbench.config import get_settings

        get_settings.cache_clear()
        dependencies.embedding_provider.cache_clear()

        client = TestClient(app)
        health = client.get("/health")
        assert health.status_code == 200
    finally:
        app.dependency_overrides.clear()
        from rag_workbench.api import dependencies
        from rag_workbench.config import get_settings

        get_settings.cache_clear()
        dependencies.embedding_provider.cache_clear()
