import json
import re
from pathlib import Path

import pytest
from sqlalchemy import select

from rag_workbench.config import Settings
from rag_workbench.db.models import RetrievalBenchmarkRecord, RetrievalBenchmarkRunRecord
from rag_workbench.evaluation.hybrid_metrics import (
    aggregate_retrieval_metrics,
    retrieval_case_metrics,
)
from rag_workbench.evaluation.retrieval_dataset import RetrievalGroundTruthCase
from rag_workbench.experiments import hybrid_retrieval_benchmark as benchmark_module
from rag_workbench.experiments.hybrid_retrieval_benchmark import (
    CALIBRATION_CASES,
    DATASET,
    HOLDOUT_CASES,
    SPLIT_IDENTITY,
    SPLIT_SEED,
    HybridRetrievalBenchmark,
    deterministic_retrieval_split,
)
from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
from rag_workbench.ingestion.pipeline import IngestionPipeline
from rag_workbench.providers.embeddings import HashingEmbeddingProvider
from rag_workbench.retrieval.bm25 import BM25Retriever, tokenize_bm25
from rag_workbench.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal


class FakeSemanticEmbeddingProvider(HashingEmbeddingProvider):
    @property
    def provider_name(self) -> str:
        return "openai-compatible"

    @property
    def model_name(self) -> str:
        return "text-embedding-3-small"


def result(
    chunk_id: str,
    document_id: str,
    rank: int,
    *,
    source: str = "dense",
    score: float = 0.9,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version_id=f"v-{document_id}",
        text="text",
        rank=rank,
        score=score,
        source="source.md",
        source_type="markdown",
        title=document_id,
        version="1",
        retrieval_source=source,
        dense_score=score if source == "dense" else None,
        lexical_score=score if source == "bm25" else None,
        found_by_dense=source == "dense",
        found_by_bm25=source == "bm25",
    )


def ingest_for_bm25(db_session, *names: str, tenant_id: str = "acmeai"):
    provider = HashingEmbeddingProvider(64)
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        provider,
    )
    for name in names:
        pipeline.ingest_path(Path("data/synthetic_company") / name, tenant_id=tenant_id)
    retriever = Retriever(db_session, provider)
    bm25 = BM25Retriever(
        db_session,
        index_identity=retriever.index_identity,
        embedding_provider=provider.provider_name,
        embedding_model=provider.model_name,
        embedding_version=provider.version,
        embedding_dimension=provider.dimension,
    )
    return provider, retriever, bm25


def test_identifier_tokenization_preserves_whole_and_component_tokens() -> None:
    tokens = tokenize_bm25("POL-2026-004 Atlas-X17 API-V3 West-Runbook")
    for identifier in ("pol-2026-004", "atlas-x17", "api-v3", "west-runbook"):
        assert identifier in tokens
    assert {"pol", "2026", "004", "atlas", "x17", "api", "v3", "west", "runbook"} <= set(tokens)


def test_bm25_ranking_is_deterministic_and_exact_identifier_wins(db_session) -> None:
    _, _, bm25 = ingest_for_bm25(
        db_session, "recovery-east.md", "recovery-west.md"
    )
    principal = Principal("employee", "acmeai", frozenset({"employees"}))
    first = bm25.retrieve("OPS-REC-E17", top_k=5, principal=principal)
    second = bm25.retrieve("OPS-REC-E17", top_k=5, principal=principal)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert first[0].document_id == "recovery-runbook-east"
    assert first[0].lexical_score == first[0].score
    assert first[0].dense_score is None


def test_bm25_prefilters_acl_version_and_tenant_before_ranking(db_session) -> None:
    _, _, bm25 = ingest_for_bm25(
        db_session,
        "hr-benefits-private.md",
        "remote-work-policy-2025.md",
        "remote-work-policy-2026.md",
    )
    ingest_for_bm25(db_session, "recovery-east.md", tenant_id="other-tenant")
    employee = Principal("employee", "acmeai", frozenset({"employees"}))
    leadership = Principal("leader", "acmeai", frozenset({"hr-leadership"}))
    assert not bm25.retrieve("HR-BEN-771", principal=employee)
    assert bm25.retrieve("HR-BEN-771", principal=leadership)[0].document_id == "hr-benefits-private"
    current = bm25.retrieve("remote policy", top_k=10, principal=employee)
    assert current
    assert {
        item.version for item in current if item.document_id == "remote-work-policy"
    } == {"2026"}
    assert not bm25.retrieve("OPS-REC-E17", principal=employee)


def test_rrf_is_deterministic_deduplicated_and_tie_broken_by_chunk_id() -> None:
    dense = [result("c2", "d2", 1), result("shared", "both", 2)]
    lexical = [result("c1", "d1", 1, source="bm25"), result("shared", "both", 2, source="bm25")]
    first = reciprocal_rank_fusion(dense, lexical, top_k=5, rrf_k=60)
    second = reciprocal_rank_fusion(dense, lexical, top_k=5, rrf_k=60)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert [item.chunk_id for item in first].count("shared") == 1
    assert first[0].chunk_id == "shared"
    assert first[1].chunk_id == "c1"
    assert first[0].found_by_dense and first[0].found_by_bm25


def test_rrf_score_is_not_dense_score_or_subject_to_cosine_threshold() -> None:
    fused = reciprocal_rank_fusion(
        [result(f"d{i}", f"doc-d{i}", i) for i in range(1, 7)],
        [result(f"b{i}", f"doc-b{i}", i, source="bm25") for i in range(1, 7)],
        top_k=5,
        rrf_k=60,
    )
    assert len(fused) == 5
    assert all(item.fusion_score is not None and item.fusion_score < 0.28 for item in fused)
    assert all(item.score == item.fusion_score for item in fused)
    assert all(item.retrieval_source == "hybrid_rrf" for item in fused)


def test_hybrid_embeds_query_once_and_enforces_final_top_five(db_session) -> None:
    provider, dense, bm25 = ingest_for_bm25(
        db_session, "recovery-east.md", "recovery-west.md", "project-atlas-api.md"
    )
    calls = 0
    original = provider.embed_query

    def counted(text: str) -> list[float]:
        nonlocal calls
        calls += 1
        return original(text)

    provider.embed_query = counted  # type: ignore[method-assign]
    hybrid = HybridRetriever(dense, bm25)
    results = hybrid.retrieve(
        "regional recovery API",
        top_k=5,
        principal=Principal("employee", "acmeai", frozenset({"employees"})),
    )
    assert calls == 1
    assert len(results) <= 5
    assert hybrid.last_timing.external_embedding_calls == 0


@pytest.mark.parametrize(("category", "count"), (("multidoc_two", 2), ("multidoc_three", 3)))
def test_all_required_coverage_requires_every_document(category: str, count: int) -> None:
    required = tuple(f"doc-{index}" for index in range(count))
    case = RetrievalGroundTruthCase(
        case_id="coverage",
        question="question",
        required_document_ids=required,
        required_version_ids={item: "1" for item in required},
        expected_access_behavior="ALLOW_REQUIRED",
        expected_answerability=True,
        category=category,
    )
    partial = [result(f"c-{item}", item, rank) for rank, item in enumerate(required[:-1], 1)]
    metrics = retrieval_case_metrics(case, partial)
    assert metrics["required_evidence_recall_at_5"] == (count - 1) / count
    assert metrics["all_required_evidence_coverage_at_5"] == 0
    aggregate = aggregate_retrieval_metrics(
        [{"category": category, "expected_answerability": True, "metrics": metrics}]
    )
    expected_name = "two_document_coverage_at_5" if count == 2 else "three_document_coverage_at_5"
    assert aggregate[expected_name] == 0


def test_retrieval_split_is_reproducible_and_holdout_is_sealed() -> None:
    first = deterministic_retrieval_split(DATASET.cases, seed=SPLIT_SEED)
    second = deterministic_retrieval_split(DATASET.cases, seed=SPLIT_SEED)
    assert first == second
    assert first[2] == SPLIT_IDENTITY
    assert len(CALIBRATION_CASES) == 40
    assert len(HOLDOUT_CASES) == 20
    assert {item.case_id for item in CALIBRATION_CASES}.isdisjoint(
        item.case_id for item in HOLDOUT_CASES
    )


def test_new_questions_are_not_reused_or_close_token_paraphrases() -> None:
    historical = []
    for path in (
        Path("data/eval/eval_v1.json"),
        Path("data/eval/acmeai_multidoc_eval_v2.json"),
    ):
        historical.extend(json.loads(path.read_text(encoding="utf-8"))["cases"])

    def tokens(question: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", question.casefold()))

    old_questions = {item["question"] for item in historical}
    assert not {item.question for item in DATASET.cases} & old_questions
    maximum_overlap = max(
        len(tokens(case.question) & tokens(old["question"]))
        / len(tokens(case.question) | tokens(old["question"]))
        for case in DATASET.cases
        for old in historical
    )
    assert maximum_overlap < 0.5


def test_configuration_lock_blocks_holdout_reuse_and_post_holdout_tuning(
    db_session, monkeypatch
) -> None:
    _, dense, _ = ingest_for_bm25(db_session, "recovery-east.md")
    monkeypatch.setattr(benchmark_module, "SEMANTIC_INDEX_IDENTITY", dense.index_identity)
    benchmark = HybridRetrievalBenchmark(db_session)
    record = benchmark.initialize()
    with pytest.raises(ValueError, match="locked first"):
        benchmark.run_holdout()
    record.selected_retrieval_mode = "DENSE"
    record.locked_at = benchmark_module.datetime.now(benchmark_module.UTC)
    record.holdout_started_at = benchmark_module.datetime.now(benchmark_module.UTC)
    db_session.flush()
    with pytest.raises(ValueError, match="one-shot"):
        benchmark.run_holdout()
    with pytest.raises(ValueError, match="locked and immutable"):
        benchmark.run_calibration()
    persisted = db_session.scalar(select(RetrievalBenchmarkRecord))
    assert persisted.holdout_completed_at is None


def test_retrieval_benchmark_persists_branch_provenance_without_judge_calls(
    db_session, monkeypatch
) -> None:
    provider = FakeSemanticEmbeddingProvider(64)
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        provider,
    )
    pipeline.ingest_path(Path("data/synthetic_company/recovery-east.md"))
    monkeypatch.setattr(benchmark_module, "SEMANTIC_INDEX_IDENTITY", pipeline.index_identity)
    benchmark = HybridRetrievalBenchmark(
        db_session,
        settings=Settings(
            _env_file=None,
            allow_external_calls=True,
            embedding_api_key="test-only",
            max_external_embedding_calls=999,
        ),
    )
    monkeypatch.setattr(benchmark, "_embedding_provider", lambda: provider)
    benchmark.initialize()
    case = next(item for item in DATASET.cases if item.case_id == "hyb_vr_07")
    benchmark._run_partition("test", (case,))
    runs = db_session.scalars(select(RetrievalBenchmarkRunRecord)).all()
    assert len(runs) == 3
    assert all(run.usage["judge_calls"] == 0 for run in runs)
    hybrid = next(run for run in runs if run.retrieval_mode == "HYBRID_RRF")
    assert all(
        "found_by_dense" in item and "found_by_bm25" in item
        for item in hybrid.case_results[0]["retrieval_trace"]
    )
