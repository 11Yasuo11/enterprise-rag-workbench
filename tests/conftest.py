from collections.abc import Generator

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from rag_workbench.api.app import app
from rag_workbench.db.models import (
    AnswerabilityGateCacheRecord,
    Chunk,
    Document,
    DocumentPermission,
    DocumentVersion,
    EndToEndBenchmarkRecord,
    EndToEndBenchmarkRunRecord,
    ExperimentCaseResultRecord,
    ExperimentConfigRecord,
    ExperimentRunRecord,
    MultiDocumentBenchmarkRecord,
    QueryEmbeddingCacheRecord,
    RagRun,
    RecoveryStageCacheRecord,
    RerankingBenchmarkRecord,
    RerankingBenchmarkRunRecord,
    ResearchArchitectureRecord,
    RetrievalArchitectureRecord,
    RetrievalBenchmarkRecord,
    RetrievalBenchmarkRunRecord,
    RetrievalResultRecord,
    V2FinalBenchmarkRecord,
    V2Phase1ExperimentRecord,
    V2Phase2ExperimentRecord,
    V2Phase3ExperimentRecord,
    V2Phase4ExperimentRecord,
    V2QualityAbExperimentRecord,
    V3Phase1ExperimentRecord,
)
from rag_workbench.db.session import get_db, get_engine


def _clear_inside_test_transaction(session: Session) -> None:
    for model in (
        EndToEndBenchmarkRunRecord,
        EndToEndBenchmarkRecord,
        V2FinalBenchmarkRecord,
        V2QualityAbExperimentRecord,
        V2Phase4ExperimentRecord,
        V2Phase3ExperimentRecord,
        V2Phase2ExperimentRecord,
        V2Phase1ExperimentRecord,
        V3Phase1ExperimentRecord,
        RecoveryStageCacheRecord,
        ResearchArchitectureRecord,
        RetrievalArchitectureRecord,
        RerankingBenchmarkRunRecord,
        RerankingBenchmarkRecord,
        RetrievalBenchmarkRunRecord,
        RetrievalBenchmarkRecord,
        MultiDocumentBenchmarkRecord,
        AnswerabilityGateCacheRecord,
        QueryEmbeddingCacheRecord,
        ExperimentCaseResultRecord,
        ExperimentRunRecord,
        ExperimentConfigRecord,
        RetrievalResultRecord,
        RagRun,
        Chunk,
        DocumentPermission,
        DocumentVersion,
        Document,
    ):
        session.execute(delete(model))
    session.flush()


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    # Services intentionally commit. An outer connection transaction keeps those commits
    # test-local and prevents the suite from deleting persisted benchmark evidence.
    connection = get_engine().connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    _clear_inside_test_transaction(session)

    def override_get_db() -> Generator[Session, None, None]:
        yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield session
    finally:
        app.dependency_overrides.pop(get_db, None)
        session.close()
        transaction.rollback()
        connection.close()
