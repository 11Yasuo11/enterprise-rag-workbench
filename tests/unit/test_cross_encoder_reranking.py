import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import select

from rag_workbench.config import Settings
from rag_workbench.db.models import RerankingBenchmarkRecord, RerankingBenchmarkRunRecord
from rag_workbench.evaluation.retrieval_dataset import RetrievalGroundTruthCase
from rag_workbench.experiments import reranking_benchmark as benchmark_module
from rag_workbench.experiments.reranking_benchmark import (
    CALIBRATION_CASES,
    CONFIGURATION,
    DATASET,
    HOLDOUT_CASES,
    SPLIT_IDENTITY,
    SPLIT_SEED,
    DenseCrossEncoderBenchmark,
    _aggregate_candidate_pool_metrics,
    _candidate_pool_case_metrics,
    deterministic_reranking_split,
)
from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
from rag_workbench.ingestion.pipeline import IngestionPipeline
from rag_workbench.providers.embeddings import HashingEmbeddingProvider
from rag_workbench.reranking.cross_encoder import CrossEncoderReranker
from rag_workbench.retrieval.vector_search import RetrievalResult


class FakeModel:
    def __init__(self, scores: list[float] | None = None) -> None:
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, **kwargs):
        del kwargs
        values = list(pairs)
        self.calls.append(values)
        return self.scores if self.scores is not None else list(range(len(values)))


class FakeSemanticProvider(HashingEmbeddingProvider):
    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        self.query_calls = 0

    @property
    def provider_name(self) -> str:
        return "openai-compatible"

    @property
    def model_name(self) -> str:
        return "text-embedding-3-small"

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return super().embed_query(text)


def result(chunk: str, document: str, rank: int, score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk,
        document_id=document,
        document_version_id=f"version-{document}",
        text=f"text {chunk}",
        rank=rank,
        score=score,
        source="source.md",
        source_type="markdown",
        title=document,
        version="1",
        dense_score=score,
        found_by_dense=True,
    )


def test_cross_encoder_deterministic_order_preserves_dense_trace_and_scores() -> None:
    model = FakeModel([0.1, 0.9, 0.9])
    reranker = CrossEncoderReranker(model=model)
    candidates = [result("c3", "d3", 1), result("c2", "d2", 2), result("c1", "d1", 3)]
    first = reranker.rerank("query", candidates)
    second = reranker.rerank("query", candidates)
    assert [item.result.chunk_id for item in first] == ["c2", "c1", "c3"]
    assert [item.result.chunk_id for item in second] == ["c2", "c1", "c3"]
    assert first[0].original_dense_rank == 2
    assert first[0].dense_score == 0.5
    assert first[0].reranker_score == 0.9
    assert first[0].reranked_rank == 1


def test_reranker_logits_are_not_cosine_thresholded_and_final_top5_is_explicit() -> None:
    candidates = [result(f"c{i}", f"d{i}", i) for i in range(1, 9)]
    reranked = CrossEncoderReranker(model=FakeModel([-10 + i for i in range(8)])).rerank(
        "query", candidates
    )
    assert len(reranked) == 8
    assert all(item.reranker_score < 0.28 for item in reranked[:5])
    assert len(reranked[:5]) == 5


def test_frozen_top20_pool_is_distinct_from_final_top5() -> None:
    case = RetrievalGroundTruthCase(
        case_id="pool",
        question="question",
        required_document_ids=("d1", "d6", "d20"),
        required_version_ids={"d1": "1", "d6": "1", "d20": "1"},
        expected_access_behavior="ALLOW_REQUIRED",
        expected_answerability=True,
        category="multidoc_three",
    )
    candidates = [result(f"c{i}", f"d{i}", i) for i in range(1, 21)]
    pool = _candidate_pool_case_metrics(case, candidates)
    assert CONFIGURATION["candidate_depth"] == 20
    assert CONFIGURATION["final_top_k"] == 5
    assert pool["required_evidence_recall_at_20"] == 1.0
    assert pool["all_required_evidence_coverage_at_20"] == 1.0
    assert (
        benchmark_module.retrieval_case_metrics(case, candidates)[
            "all_required_evidence_coverage_at_5"
        ]
        == 0.0
    )


def test_candidate_pool_version_metric_only_uses_version_cases() -> None:
    rows = [
        {
            "category": "version_region",
            "expected_answerability": True,
            "metrics": {
                "required_evidence_recall_at_20": 1.0,
                "all_required_evidence_coverage_at_20": 1.0,
                "version_correct_at_20": 1.0,
                "unauthorized_result_exposure_at_20": 0,
            },
        },
        {
            "category": "multidoc_two",
            "expected_answerability": True,
            "metrics": {
                "required_evidence_recall_at_20": 0.5,
                "all_required_evidence_coverage_at_20": 0.0,
                "version_correct_at_20": 0.0,
                "unauthorized_result_exposure_at_20": 0,
            },
        },
    ]
    aggregate = _aggregate_candidate_pool_metrics(rows)
    assert aggregate["version_correctness_at_20"] == 1.0
    assert aggregate["two_document_coverage_at_20"] == 0.0


def test_rank_movement_tracks_promotions_demotions_and_irrelevant_removal() -> None:
    case = RetrievalGroundTruthCase(
        case_id="move",
        question="question",
        required_document_ids=("required-a", "required-b", "required-c"),
        required_version_ids={"required-a": "1", "required-b": "1", "required-c": "1"},
        expected_access_behavior="ALLOW_REQUIRED",
        expected_answerability=True,
        category="multidoc_three",
    )
    dense = [
        result("a", "required-a", 1),
        result("x", "irrelevant", 2),
        result("b", "required-b", 3),
        result("y", "irrelevant-2", 4),
        result("z", "irrelevant-3", 5),
        result("c", "required-c", 6),
    ]
    reranked = [dense[5], dense[0], dense[1], dense[3], dense[4], dense[2]]
    reranked = [
        item.__class__(**{**item.__dict__, "rank": rank}) for rank, item in enumerate(reranked, 1)
    ]
    movement = DenseCrossEncoderBenchmark._movement(case, dense, reranked)
    assert sum(item["promoted_into_top5"] for item in movement["required_evidence"]) == 1
    assert sum(item["demoted_out_of_top5"] for item in movement["required_evidence"]) == 1
    assert movement["irrelevant_chunks_removed"] == 0


@pytest.mark.parametrize(
    ("category", "expected_failure"),
    (
        ("exact_identifier", "EXACT_IDENTIFIER_RERANK_REGRESSION"),
        ("semantic_paraphrase", "SEMANTIC_RERANK_REGRESSION"),
    ),
)
def test_reranker_regression_taxonomy_is_candidate_only(
    category: str, expected_failure: str
) -> None:
    case = RetrievalGroundTruthCase(
        case_id="regression",
        question="question",
        required_document_ids=("required",),
        required_version_ids={"required": "1"},
        expected_access_behavior="ALLOW_REQUIRED",
        expected_answerability=True,
        category=category,
    )
    dense = [result("required", "required", 1)] + [result(f"c{i}", f"d{i}", i) for i in range(2, 7)]
    reranked = [*dense[1:], dense[0]]
    reranked = [replace(item, rank=rank) for rank, item in enumerate(reranked, 1)]
    movement = DenseCrossEncoderBenchmark._movement(case, dense, reranked)
    dense_failures = DenseCrossEncoderBenchmark._failures(case, "DENSE", dense[:5], dense, movement)
    reranker_failures = DenseCrossEncoderBenchmark._failures(
        case, "DENSE_CROSS_ENCODER_RERANK", reranked[:5], dense, movement
    )
    assert expected_failure not in dense_failures
    assert expected_failure in reranker_failures


def _benchmark_with_fake_model(db_session, monkeypatch, paths: tuple[str, ...]):
    provider = FakeSemanticProvider(64)
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        provider,
    )
    for path in paths:
        pipeline.ingest_path(Path("data/synthetic_company") / path)
    monkeypatch.setattr(benchmark_module, "SEMANTIC_INDEX_IDENTITY", pipeline.index_identity)
    fake = FakeModel()
    reranker = CrossEncoderReranker(model=fake)
    benchmark = DenseCrossEncoderBenchmark(
        db_session,
        settings=Settings(
            _env_file=None,
            allow_external_calls=True,
            embedding_api_key="test-only",
            max_external_embedding_calls=999,
        ),
        reranker_factory=lambda: reranker,
    )
    monkeypatch.setattr(benchmark, "_embedding_provider", lambda: provider)
    benchmark.initialize()
    return benchmark, fake


def test_same_authorized_dense_top20_is_reused_and_persisted_without_external_calls(
    db_session, monkeypatch
) -> None:
    benchmark, fake = _benchmark_with_fake_model(
        db_session,
        monkeypatch,
        ("recovery-east.md", "recovery-west.md", "project-atlas-api.md"),
    )
    case = next(item for item in DATASET.cases if item.case_id == "rer_near_01")
    benchmark._run_partition("test", (case,))
    runs = {item.mode: item for item in db_session.scalars(select(RerankingBenchmarkRunRecord))}
    dense = runs["DENSE"].case_results[0]
    reranked = runs["DENSE_CROSS_ENCODER_RERANK"].case_results[0]
    candidate_ids = [item["chunk_id"] for item in dense["candidate_trace"]]
    assert candidate_ids == [item["chunk_id"] for item in reranked["candidate_trace"]]
    assert len(fake.calls) == 1
    assert [text for _, text in fake.calls[0]] == [
        item["text"]
        if "text" in item
        else next(
            chunk.text
            for chunk in db_session.scalars(select(benchmark_module.Chunk))
            if chunk.id == item["chunk_id"]
        )
        for item in dense["candidate_trace"]
    ]
    assert runs["DENSE"].usage["query_cache_misses"] == 1
    assert benchmark._embedding_provider().query_calls == 1
    assert runs["DENSE_CROSS_ENCODER_RERANK"].usage["external_reranker_calls"] == 0
    assert runs["DENSE_CROSS_ENCODER_RERANK"].usage["judge_calls"] == 0
    assert all("reranker_score" in item for item in reranked["reranking_trace"])
    assert reranked["rerankable"] is True


def test_acl_tenant_and_version_filters_precede_reranker_input(db_session, monkeypatch) -> None:
    benchmark, fake = _benchmark_with_fake_model(
        db_session,
        monkeypatch,
        (
            "hr-benefits-private.md",
            "remote-work-policy-2025.md",
            "remote-work-policy-2026.md",
            "recovery-east.md",
        ),
    )
    other_provider = FakeSemanticProvider(64)
    other = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        other_provider,
        index_identity=benchmark_module.SEMANTIC_INDEX_IDENTITY,
    )
    other.ingest_path(Path("data/synthetic_company/recovery-west.md"), tenant_id="other")
    case = next(item for item in DATASET.cases if item.case_id == "rer_acl_02")
    case = case.model_copy(
        update={
            "question": (
                "Under the current 2026 policy, how many remote-work days are allowed, "
                "and what are the private executive-benefit code and west recovery code?"
            )
        }
    )
    benchmark._run_partition("security", (case,))
    texts = " ".join(text for _, text in fake.calls[0])
    assert "HR-BEN-771" not in texts
    assert "three days per week" not in texts
    assert "OPS-REC-W29" not in texts
    assert "two days per week" in texts


def test_dataset_overlap_guard_and_split_reproducibility() -> None:
    historical = []
    for filename in (
        "eval_v1.json",
        "acmeai_multidoc_eval_v2.json",
        "acmeai_hybrid_retrieval_eval_v1.json",
    ):
        historical.extend(json.loads((Path("data/eval") / filename).read_text())["cases"])

    def tokens(question: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", question.casefold()))

    overlap = max(
        len(tokens(case.question) & tokens(old["question"]))
        / len(tokens(case.question) | tokens(old["question"]))
        for case in DATASET.cases
        for old in historical
    )
    assert overlap < 0.5
    first = deterministic_reranking_split(DATASET.cases, seed=SPLIT_SEED)
    second = deterministic_reranking_split(DATASET.cases, seed=SPLIT_SEED)
    assert first == second
    assert first[2] == SPLIT_IDENTITY
    assert len(CALIBRATION_CASES) == 40 and len(HOLDOUT_CASES) == 20


def test_selection_lock_seals_holdout_and_blocks_post_holdout_tuning(
    db_session, monkeypatch
) -> None:
    benchmark, _ = _benchmark_with_fake_model(db_session, monkeypatch, ("recovery-east.md",))
    record = db_session.get(RerankingBenchmarkRecord, benchmark_module.DATASET_ID)
    with pytest.raises(ValueError, match="locked before holdout"):
        benchmark.run_holdout()
    record.selected_mode = "DENSE"
    record.locked_at = benchmark_module.datetime.now(benchmark_module.UTC)
    record.holdout_started_at = benchmark_module.datetime.now(benchmark_module.UTC)
    db_session.flush()
    with pytest.raises(ValueError, match="one-shot"):
        benchmark.run_holdout()
    with pytest.raises(ValueError, match="locked and immutable"):
        benchmark.run_calibration()
