import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_workbench.answerability.openai_compatible import EVIDENCE_GATE_PROMPT_VERSION
from rag_workbench.config import Settings
from rag_workbench.db.models import RetrievalBenchmarkRecord
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    DENSE_DEPTH,
    FINAL_TOP_K,
    HOLDOUT_CASES,
    MODES,
    RRF_K,
    UNION_LIMIT,
    HybridRerankerBenchmark,
)
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    CONFIGURATION as HISTORICAL_CONFIGURATION,
)
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    DATASET_ID as HISTORICAL_DATASET_ID,
)
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    SELECTION_POLICY as HISTORICAL_SELECTION_POLICY,
)
from rag_workbench.experiments.hybrid_reranker_replication import (
    ARCHITECTURE_ID,
    CASES,
    CONFIGURATION,
    DATASET_HASH,
    DATASET_ID,
    MAXIMUM_PRIOR_OVERLAP,
    OVERLAP_CEILING,
    REPLICATION_POLICY,
    SOL_MODEL,
    SPLIT_IDENTITY,
    SPLIT_SEED,
    HybridRerankerReplicationBenchmark,
    apply_replication_policy,
    classify_b_three_document_failure,
    classify_retrieval_regression,
    maximum_prior_dataset_overlap,
)
from rag_workbench.experiments.judge_e2e_benchmark import (
    maximum_prior_dataset_overlap as judge_overlap,
)
from rag_workbench.experiments.reranker_e2e_benchmark import RERANKER_REVISION
from rag_workbench.experiments.reranker_e2e_benchmark import (
    maximum_prior_dataset_overlap as reranker_overlap,
)
from rag_workbench.experiments.sol_judge_e2e_benchmark import (
    maximum_prior_dataset_overlap as sol_overlap,
)
from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
from rag_workbench.ingestion.pipeline import IngestionPipeline
from rag_workbench.providers.embeddings import HashingEmbeddingProvider
from rag_workbench.reranking.cross_encoder import CrossEncoderReranker
from rag_workbench.retrieval.bm25 import BM25Config, tokenize_bm25
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.vector_search import RetrievalResult


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, **kwargs):
        del kwargs
        values = list(pairs)
        self.calls.append(values)
        return list(range(len(values), 0, -1))


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
        text=f"text {chunk_id}",
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


def test_replication_dataset_identity_distribution_and_overlap_guard() -> None:
    assert DATASET_ID == "acmeai-hybrid-reranker-replication-v1"
    assert len(CASES) == 80
    assert len({item.case_id for item in CASES}) == 80
    assert Counter(item.category for item in CASES) == {
        "multidoc_two": 10,
        "multidoc_three": 40,
        "near_duplicate": 8,
        "exact_identifier": 6,
        "version_region": 6,
        "semantic_paraphrase": 4,
        "acl_sensitive": 3,
        "partial_no_answer": 3,
    }
    assert DATASET_HASH == "f9144a0bbc6f15de030963199b8a784d9e23031a017258a8b6636b131cb04114"
    assert maximum_prior_dataset_overlap() == pytest.approx(MAXIMUM_PRIOR_OVERLAP)
    assert maximum_prior_dataset_overlap() < OVERLAP_CEILING
    assert SPLIT_SEED == 0
    questions = {item.question for item in CASES}
    assert len(questions) == 80
    historical = []
    for path in Path("data/eval").glob("*.json"):
        if path.name == "acmeai_hybrid_reranker_replication_v1.json":
            continue
        historical.extend(json.loads(path.read_text())["cases"])
    assert questions.isdisjoint(item["question"] for item in historical)
    assert {item.case_id for item in CASES}.isdisjoint(item.case_id for item in HOLDOUT_CASES)


def test_historical_overlap_guards_remain_frozen() -> None:
    assert sol_overlap() == pytest.approx(0.4444444444444444)
    assert judge_overlap() == pytest.approx(0.47058823529411764)
    assert reranker_overlap() == pytest.approx(0.391304347826087)
    from rag_workbench.experiments.hybrid_reranker_benchmark import (
        maximum_prior_dataset_overlap as hybrid_overlap,
    )

    assert hybrid_overlap() == pytest.approx(0.4)


def test_frozen_a_and_b_identities_match_historical_candidate() -> None:
    assert MODES == ("DENSE_CROSS_ENCODER_RERANK", "HYBRID_CROSS_ENCODER_RERANK")
    assert DENSE_DEPTH == BM25_DEPTH == 20
    assert UNION_LIMIT == 30
    assert RRF_K == 60
    assert FINAL_TOP_K == 5
    assert CONFIGURATION == HISTORICAL_CONFIGURATION
    assert CONFIGURATION["dense_threshold"] == 0.28
    assert CONFIGURATION["bm25"]["version"] == "bm25-okapi-v1"
    assert CONFIGURATION["bm25"]["k1"] == 1.2
    assert CONFIGURATION["bm25"]["b"] == 0.75
    assert CONFIGURATION["rrf_k"] == 60
    assert CONFIGURATION["candidate_union_limit"] == 30
    assert CONFIGURATION["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert CONFIGURATION["reranker_revision"] == RERANKER_REVISION
    assert CONFIGURATION["judge"]["model"] == SOL_MODEL
    assert CONFIGURATION["judge"]["prompt_version"] == EVIDENCE_GATE_PROMPT_VERSION
    assert CONFIGURATION["judge"]["used_for_retrieval_selection"] is False
    assert BM25Config().k1 == 1.2
    assert BM25Config().b == 0.75
    assert HISTORICAL_SELECTION_POLICY["all_required_coverage_minimum_gain"] == 0.10
    assert HISTORICAL_SELECTION_POLICY["three_document_coverage_minimum_gain"] == 0.20


def test_bm25_tokenizer_rrf_union_limit_and_same_cross_encoder_revision() -> None:
    tokens = tokenize_bm25("POL-2026-004 Atlas-X17")
    assert "pol-2026-004" in tokens
    fused = reciprocal_rank_fusion(
        [result("d1", "dense-doc", 1), result("shared", "both", 2)],
        [result("b1", "bm25-doc", 1, source="bm25"), result("shared", "both", 2, source="bm25")],
        top_k=10_000,
        rrf_k=60,
    )
    bounded = fused[:UNION_LIMIT]
    assert len(bounded) <= 30
    assert CONFIGURATION["reranker_revision"] == "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def test_replication_policy_is_frozen_and_not_the_original_calibration_rule() -> None:
    assert REPLICATION_POLICY["all_required_coverage_minimum_gain"] == 0.08
    assert REPLICATION_POLICY["three_document_coverage_minimum_gain"] == 0.15
    assert REPLICATION_POLICY["no_calibration_holdout_split"] is True
    assert REPLICATION_POLICY["previous_selection_rule_unchanged"] is True
    assert REPLICATION_POLICY["not_a_retroactive_change"] is True
    assert REPLICATION_POLICY["policy_frozen_before_retrieval"] is True
    dense = {
        "all_required_evidence_coverage_at_5": 0.70,
        "three_document_coverage_at_5": 0.50,
        "required_evidence_recall_at_5": 0.90,
        "exact_identifier_recall_at_5": 1.0,
        "semantic_success": 1.0,
        "version_correctness": 1.0,
        "acl_safety": 1.0,
        "unauthorized_result_exposure": 0,
    }
    promoted = {
        **dense,
        "all_required_evidence_coverage_at_5": 0.78,
        "three_document_coverage_at_5": 0.66,
    }
    assert apply_replication_policy(dense, promoted)["selected_mode"] == MODES[1]
    no_gain = {
        **dense,
        "all_required_evidence_coverage_at_5": 0.77,
        "three_document_coverage_at_5": 0.64,
    }
    assert apply_replication_policy(dense, no_gain)["selected_mode"] == MODES[0]
    recall_regression = {**promoted, "required_evidence_recall_at_5": 0.87}
    assert apply_replication_policy(dense, recall_regression)["selected_mode"] == MODES[0]


def test_three_document_cases_require_three_distinct_sources() -> None:
    for item in CASES:
        if item.category == "multidoc_three":
            assert item.expected_answerability
            assert len(set(item.required_document_ids)) == 3
        if item.category == "multidoc_two":
            assert len(set(item.required_document_ids)) == 2
        if item.category in {"acl_sensitive", "partial_no_answer"}:
            assert item.should_abstain
            assert not item.expected_answerability


def test_lexical_rescue_and_candidate_pool_helpers_remain_available() -> None:
    rows = [
        {
            "case_id": "rescue",
            "expected_answerability": True,
            "required_document_ids": ["missed", "kept"],
            "required_dense_ranks": {"missed": None, "kept": 1},
            "required_bm25_ranks": {"missed": 2, "kept": None},
            "required_rrf_ranks": {"missed": 8, "kept": 1},
            "required_ce_ranks": {"missed": 3, "kept": 1},
            "top5": [{"document_id": "missed"}, {"document_id": "kept"}],
            "dense_candidates": [{"document_id": "kept"}],
            "bm25_candidates": [{"document_id": "missed"}],
            "rrf_union": [{"document_id": "missed"}, {"document_id": "kept"}],
            "top5_complete": True,
            "category": "multidoc_three",
        }
    ]
    lexical = HybridRerankerBenchmark._lexical_rescues(rows)
    assert lexical["bm25_only_required_evidence"] == 1
    assert lexical["promoted_to_top5"] == 1
    assert classify_b_three_document_failure(rows[0]) is None


def test_three_document_failure_taxonomy() -> None:
    base = {
        "case_id": "fail",
        "top5_complete": False,
        "required_document_ids": ["a", "b", "c"],
        "dense_candidates": [],
        "bm25_candidates": [],
        "rrf_union": [],
        "top5": [],
    }
    assert classify_b_three_document_failure(base) == "BOTH_BRANCHES_MISS"
    truncated = {
        **base,
        "bm25_candidates": [{"document_id": "a"}, {"document_id": "b"}, {"document_id": "c"}],
        "rrf_union": [{"document_id": "a"}, {"document_id": "b"}],
    }
    assert classify_b_three_document_failure(truncated) == "RRF_TRUNCATION_LOSS"
    not_promoted = {
        **base,
        "bm25_candidates": [{"document_id": "a"}, {"document_id": "b"}, {"document_id": "c"}],
        "rrf_union": [{"document_id": "a"}, {"document_id": "b"}, {"document_id": "c"}],
        "top5": [{"document_id": "a"}, {"document_id": "b"}],
    }
    assert classify_b_three_document_failure(not_promoted) == "CROSS_ENCODER_FAILED_TO_PROMOTE"
    demoted = {
        **base,
        "dense_candidates": [{"document_id": "a"}, {"document_id": "b"}, {"document_id": "c"}],
        "bm25_candidates": [{"document_id": "a"}],
        "rrf_union": [{"document_id": "a"}, {"document_id": "b"}, {"document_id": "c"}],
        "top5": [{"document_id": "a"}, {"document_id": "b"}],
    }
    assert classify_b_three_document_failure(demoted) == "CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE"


def test_retrieval_regression_families() -> None:
    a_row = {
        "case_id": "sem",
        "top5_complete": True,
        "required_document_ids": ["doc"],
        "top5": [{"document_id": "doc"}],
    }
    b_row = {
        "case_id": "sem",
        "top5_complete": False,
        "required_document_ids": ["doc"],
        "top5": [{"document_id": "other"}],
    }
    found = classify_retrieval_regression("semantic_paraphrase", a_row, b_row)
    assert found is not None
    assert found["family"] == "semantic regression"
    assert classify_retrieval_regression("semantic_paraphrase", a_row, a_row) is None


def _prepare_benchmark(db_session, monkeypatch, documents: tuple[str, ...]):
    from rag_workbench.experiments import hybrid_reranker_benchmark as parent_module
    from rag_workbench.experiments import hybrid_reranker_replication as benchmark_module

    provider = FakeSemanticProvider(64)
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        provider,
    )
    for name in documents:
        pipeline.ingest_path(Path("data/synthetic_company") / name)
    monkeypatch.setattr(benchmark_module, "SEMANTIC_INDEX_IDENTITY", pipeline.index_identity)
    monkeypatch.setattr(parent_module, "SEMANTIC_INDEX_IDENTITY", pipeline.index_identity)
    fake = FakeModel()
    benchmark = HybridRerankerReplicationBenchmark(
        db_session,
        settings=Settings(
            _env_file=None,
            allow_external_calls=True,
            embedding_api_key="test-only",
            max_external_embedding_calls=999,
            allow_external_judge_calls=False,
            max_external_judge_calls=0,
        ),
        embedding_provider_factory=lambda: provider,
        reranker_factory=lambda: CrossEncoderReranker(
            model=fake, resolved_revision=RERANKER_REVISION
        ),
    )
    monkeypatch.setattr(benchmark, "_corpus_identity", lambda: benchmark_module.CORPUS_IDENTITY)
    monkeypatch.setattr(benchmark, "_verify_production_identities", lambda: None)
    return benchmark, provider, fake


def test_policy_is_persisted_before_retrieval_and_dataset_is_immutable(
    db_session, monkeypatch
) -> None:
    benchmark, _, _ = _prepare_benchmark(db_session, monkeypatch, ("recovery-east.md",))
    record = benchmark.initialize()
    assert record.selection_policy == REPLICATION_POLICY
    assert record.holdout_case_ids == []
    assert record.split_identity == SPLIT_IDENTITY
    assert record.locked_at is None
    persisted = db_session.get(RetrievalBenchmarkRecord, DATASET_ID)
    assert persisted.retrieval_configuration == CONFIGURATION
    persisted.selection_policy = {**REPLICATION_POLICY, "tampered": True}
    db_session.flush()
    with pytest.raises(ValueError, match="identity changed"):
        benchmark._verify(persisted)


def test_shared_query_embedding_security_and_no_sol_during_retrieval(
    db_session, monkeypatch
) -> None:
    benchmark, provider, fake = _prepare_benchmark(
        db_session, monkeypatch, ("recovery-east.md", "recovery-west.md", "project-atlas-api.md")
    )
    benchmark.initialize()
    sample = next(item for item in CASES if item.category == "multidoc_two")
    sample = sample.model_copy(
        update={
            "required_document_ids": ("recovery-runbook-east", "recovery-runbook-west"),
            "expected_document_ids": ("recovery-runbook-east", "recovery-runbook-west"),
            "forbidden_document_ids": (),
            "expected_access_behavior": "ALLOW_REQUIRED",
        }
    )
    judge_before = benchmark._historical_judge_calls()
    with benchmark._bind_dataset():
        analysis = benchmark._run_partition("replication-sample", (sample,))
    assert provider.query_calls == 1
    assert benchmark._historical_judge_calls() == judge_before
    dense = analysis[MODES[0]]["cases"][0]
    hybrid = analysis[MODES[1]]["cases"][0]
    assert dense["candidate_count_sent_to_reranker"] <= DENSE_DEPTH
    assert hybrid["candidate_count_sent_to_reranker"] <= UNION_LIMIT
    assert all(len(fake_call) <= UNION_LIMIT for fake_call in fake.calls)


def test_acl_content_never_reaches_cross_encoder(db_session, monkeypatch) -> None:
    benchmark, _, fake = _prepare_benchmark(
        db_session,
        monkeypatch,
        ("hr-benefits-private.md", "remote-work-policy-2026.md", "recovery-east.md"),
    )
    sample = next(item for item in CASES if item.category == "acl_sensitive")
    sample = sample.model_copy(
        update={
            "question": (
                "What is the private executive benefit code and the current "
                "remote-work allowance?"
            ),
            "forbidden_document_ids": ("hr-benefits-private",),
            "expected_access_behavior": "EXCLUDE_FORBIDDEN",
            "expected_answerability": False,
            "should_abstain": True,
            "required_document_ids": (),
            "expected_document_ids": (),
        }
    )
    benchmark.initialize()
    with benchmark._bind_dataset():
        benchmark._run_partition("acl", (sample,))
    texts = " ".join(text for call in fake.calls for _, text in call)
    assert "HR-BEN-771" not in texts


def test_retrieval_is_one_shot_and_architecture_lock_is_immutable(db_session, monkeypatch) -> None:
    benchmark, _, _ = _prepare_benchmark(db_session, monkeypatch, ("recovery-east.md",))
    record = benchmark.initialize()
    record.selected_retrieval_mode = MODES[0]
    record.locked_at = datetime.now(UTC)
    record.calibration_metrics = {"modes": {}}
    db_session.flush()
    with pytest.raises(ValueError, match="already locked"):
        benchmark.retrieve()
    first = benchmark.freeze_architecture()
    assert first.architecture_id == ARCHITECTURE_ID
    assert first.selected_retriever == MODES[0]
    assert first.configuration["no_further_v1_retrieval_tuning"] is True
    assert first.immutable is True
    first.selected_retriever = MODES[1]
    db_session.flush()
    with pytest.raises(ValueError, match="immutable"):
        benchmark.freeze_architecture()


def test_historical_dataset_id_is_not_reused() -> None:
    assert DATASET_ID != HISTORICAL_DATASET_ID
    assert DATASET_ID != "acmeai-hybrid-reranker-eval-v1"


def test_canonical_dedup_keeps_one_shared_chunk() -> None:
    fused = reciprocal_rank_fusion(
        [result("shared", "both", 1), result("d2", "dense-doc", 2)],
        [result("shared", "both", 1, source="bm25"), result("b2", "bm25-doc", 2, source="bm25")],
        top_k=10_000,
        rrf_k=60,
    )
    assert [item.chunk_id for item in fused].count("shared") == 1
    assert fused[0].found_by_dense and fused[0].found_by_bm25


def test_embedding_preflight_ceiling_is_current_plus_missing(
    db_session, monkeypatch
) -> None:
    benchmark, _, _ = _prepare_benchmark(db_session, monkeypatch, ("recovery-east.md",))
    preflight = benchmark.embedding_preflight()
    assert preflight["new_unique_queries"] == 80
    assert preflight["expected_cumulative_ending_usage"] == (
        preflight["current_cumulative_embedding_calls"] + preflight["missing_embeddings"]
    )
    benchmark.settings.max_external_embedding_calls = (
        preflight["current_cumulative_embedding_calls"]
    )
    with (
        pytest.raises(ValueError, match="exceeds the embedding-call ceiling"),
        benchmark._bind_dataset(),
    ):
        benchmark._run_partition("replication-sample", CASES[:1])


def test_live_and_cached_sol_latency_are_separated() -> None:
    from types import SimpleNamespace

    prepared = [
        {
            "timing": {
                "query_embedding_ms": 10.0,
                "cache_lookup_ms": 1.0,
                "dense_ms": 2.0,
                "bm25_ms": 3.0,
                "rrf_ms": 0.5,
                "reranker_a_ms": 20.0,
                "reranker_b_ms": 40.0,
            }
        },
        {
            "timing": {
                "query_embedding_ms": 10.0,
                "cache_lookup_ms": 1.0,
                "dense_ms": 2.0,
                "bm25_ms": 3.0,
                "rrf_ms": 0.5,
                "reranker_a_ms": 20.0,
                "reranker_b_ms": 40.0,
            }
        },
    ]
    cases = (
        SimpleNamespace(
            answerability_judge_latency_ms=2000.0,
            generation_latency_ms=5.0,
            total_latency_ms=2010.0,
            gate_cache_hit=False,
        ),
        SimpleNamespace(
            answerability_judge_latency_ms=0.0,
            generation_latency_ms=5.0,
            total_latency_ms=10.0,
            gate_cache_hit=True,
        ),
    )
    latency = HybridRerankerReplicationBenchmark._e2e_latency(MODES[1], cases, prepared)
    assert latency["live_judge_count"] == 1.0
    assert latency["cached_replay_judge_count"] == 1.0
    assert latency["live_judge_mean_ms"] == 2000.0
    assert latency["cached_replay_judge_mean_ms"] == 0.0
    assert latency["judge_mean_ms"] == 2000.0
    assert latency["cached_replay_excluded_from_live_judge_mean"] == 1.0


def test_benchmark_md_records_final_replication_and_architecture_freeze() -> None:
    text = Path("BENCHMARK.md").read_text()
    section = text.split("## Final Hybrid+Cross-Encoder Replication", 1)[1]
    assert (
        "This was the final Dense-vs-Hybrid replication for Enterprise RAG Workbench v1."
        in section
    )
    assert "HYBRID_CROSS_ENCODER_RERANK" in section
    assert "enterprise-rag-v1-retriever" in section
    assert "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1" in section
    assert "cached replay" in section.lower() or "Cached replay" in section


def test_benchmark_md_records_single_final_frozen_end_to_end_section() -> None:
    from rag_workbench.experiments.hybrid_reranker_replication import (
        FINAL_DATASET_HASH,
        FINAL_DATASET_ID,
        RELEASE_ARCHITECTURE_ID,
    )

    text = Path("BENCHMARK.md").read_text()
    heading = "## Enterprise RAG Workbench v1 — Final Frozen End-to-End Benchmark"
    assert text.count(heading) == 1
    section = text.split(heading, 1)[1]
    assert FINAL_DATASET_ID in section
    assert FINAL_DATASET_HASH in section
    assert RELEASE_ARCHITECTURE_ID in section
    assert "HYBRID_CROSS_ENCODER_RERANK" in section
    assert "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1" in section
    assert "Resume added 0 embedding calls and 0 Sol calls." in section


def test_sol_configuration_remains_frozen() -> None:
    assert CONFIGURATION["judge"]["model"] == "gpt-5.6-sol"
    assert CONFIGURATION["judge"]["prompt_version"] == EVIDENCE_GATE_PROMPT_VERSION
    assert CONFIGURATION["judge"]["used_for_retrieval_selection"] is False
    assert SOL_MODEL == "gpt-5.6-sol"


def test_final_v1_dataset_identity_distribution_and_overlap_guard() -> None:
    from rag_workbench.experiments.hybrid_reranker_replication import (
        EXPECTED_FINAL_DISTRIBUTION,
        FINAL_CASES,
        FINAL_DATASET_HASH,
        FINAL_DATASET_ID,
        FINAL_DATASET_PATH,
        FINAL_OVERLAP_CEILING,
        classify_final_root_cause,
        final_dataset_overlap_report,
        historical_overlap_identities_hold,
    )

    assert FINAL_DATASET_ID == "acmeai-enterprise-rag-v1-final-eval"
    assert len(FINAL_CASES) == 80
    assert len({item.case_id for item in FINAL_CASES}) == 80
    assert Counter(item.category for item in FINAL_CASES) == EXPECTED_FINAL_DISTRIBUTION
    assert hashlib.sha256(FINAL_DATASET_PATH.read_bytes()).hexdigest() == FINAL_DATASET_HASH
    overlap = final_dataset_overlap_report()
    assert overlap["maximum_normalized_overlap"] < FINAL_OVERLAP_CEILING
    assert overlap["pass"] is True
    assert historical_overlap_identities_hold() is True
    three = [item for item in FINAL_CASES if item.category == "multidoc_three"]
    assert len(three) == 20
    assert all(len(set(item.required_document_ids)) == 3 for item in three)
    assert all(len(item.required_fact_ids) == 3 for item in three)
    injection = [item for item in FINAL_CASES if item.category == "prompt_injection"]
    assert len(injection) == 4
    assert all("prompt_injection" in item.security_checks for item in injection)
    historical = []
    for path in Path("data/eval").glob("*.json"):
        if path == FINAL_DATASET_PATH:
            continue
        historical.extend(json.loads(path.read_text())["cases"])
    assert {item.question for item in FINAL_CASES}.isdisjoint(
        item["question"] for item in historical
    )
    assert {item.case_id for item in FINAL_CASES}.isdisjoint(
        item["case_id"] for item in historical
    )
    result = SimpleNamespace(
        security_passed=True,
        version_correct=1.0,
        expected_abstain=False,
        status="abstained",
        answerability_result={"answerable": False},
        supporting_context_loss=0.0,
        supporting_chunk_ids=(),
        retrieved_chunk_ids=(),
        citation_correctness=1.0,
    )
    retrieval = {
        "top5_complete": True,
        "metrics": {"unauthorized_result_exposure": 0},
        "failure_taxonomy": "NONE",
    }
    case = next(item for item in FINAL_CASES if item.expected_answerability)
    assert classify_final_root_cause(case, retrieval, result) == "EVIDENCE_GATE_FALSE_NEGATIVE"


def test_final_v1_judge_and_single_pipeline_remain_frozen() -> None:
    from rag_workbench.experiments.hybrid_reranker_replication import (
        RELEASE_ARCHITECTURE_ID,
        SCHEMA_IDENTITY,
    )

    assert CONFIGURATION["judge"]["model"] == "gpt-5.6-sol"
    assert CONFIGURATION["judge"]["prompt_version"] == EVIDENCE_GATE_PROMPT_VERSION
    assert SCHEMA_IDENTITY == "6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621"
    assert RELEASE_ARCHITECTURE_ID == "enterprise-rag-workbench-v1"
    assert ARCHITECTURE_ID == "enterprise-rag-v1-retriever"
    assert RELEASE_ARCHITECTURE_ID != ARCHITECTURE_ID


def test_final_v1_initialize_requires_frozen_replication_and_is_one_shot(
    db_session, monkeypatch
) -> None:
    from rag_workbench.db.models import EndToEndBenchmarkRecord, RetrievalArchitectureRecord
    from rag_workbench.experiments.hybrid_reranker_replication import (
        FINAL_DATASET_ID,
        FinalV1Benchmark,
    )
    from rag_workbench.experiments.reranker_e2e_benchmark import CORPUS_IDENTITY
    from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
    from rag_workbench.ingestion.pipeline import IngestionPipeline
    from rag_workbench.providers.embeddings import HashingEmbeddingProvider

    benchmark = FinalV1Benchmark(
        db_session,
        settings=Settings(
            _env_file=None,
            allow_external_calls=True,
            embedding_api_key="test-only",
            max_external_embedding_calls=999,
            allow_external_judge_calls=False,
            max_external_judge_calls=0,
        ),
    )
    with pytest.raises(ValueError, match="PARTIAL"):
        benchmark.initialize()
    db_session.add(
        RetrievalArchitectureRecord(
            architecture_id=ARCHITECTURE_ID,
            selected_retriever=MODES[1],
            configuration={"immutable": True},
            selection_policy={},
            dataset_id=DATASET_ID,
            dataset_hash=DATASET_HASH,
            retrieval_metrics={},
            immutable=True,
        )
    )
    db_session.flush()
    monkeypatch.setattr(
        benchmark,
        "previous_replication_state",
        lambda: {
            "replication_verdict": "COMPLETE",
            "final_selected_retriever": MODES[1],
            "retrieval_architecture_frozen": True,
            "replication_rerun": False,
            "complete": True,
        },
    )
    monkeypatch.setattr(benchmark, "_corpus_identity", lambda: CORPUS_IDENTITY)

    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        HashingEmbeddingProvider(64),
    )
    pipeline.ingest_path(Path("data/synthetic_company") / "recovery-east.md")
    monkeypatch.setattr(
        "rag_workbench.experiments.hybrid_reranker_replication.SEMANTIC_INDEX_IDENTITY",
        pipeline.index_identity,
    )
    first = benchmark.initialize()
    assert first.dataset_id == FINAL_DATASET_ID
    assert first.production_retriever_status == MODES[1]
    again = benchmark.initialize()
    assert again.dataset_id == first.dataset_id
    assert db_session.get(EndToEndBenchmarkRecord, FINAL_DATASET_ID) is not None


def _seed_frozen_replication(db_session, monkeypatch):
    from rag_workbench.db.models import RetrievalArchitectureRecord
    from rag_workbench.experiments.hybrid_reranker_replication import (
        FINAL_CASES,
        FinalV1Benchmark,
    )
    from rag_workbench.experiments.reranker_e2e_benchmark import CORPUS_IDENTITY

    db_session.add(
        RetrievalArchitectureRecord(
            architecture_id=ARCHITECTURE_ID,
            selected_retriever=MODES[1],
            configuration={"immutable": True},
            selection_policy={},
            dataset_id=DATASET_ID,
            dataset_hash=DATASET_HASH,
            retrieval_metrics={},
            immutable=True,
        )
    )
    db_session.flush()
    provider = FakeSemanticProvider(64)
    fake = FakeModel()
    benchmark = FinalV1Benchmark(
        db_session,
        settings=Settings(
            _env_file=None,
            allow_external_calls=True,
            embedding_api_key="test-only",
            max_external_embedding_calls=999,
            allow_external_judge_calls=False,
            max_external_judge_calls=0,
        ),
        embedding_provider_factory=lambda: provider,
        reranker_factory=lambda: CrossEncoderReranker(
            model=fake, resolved_revision=RERANKER_REVISION
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "previous_replication_state",
        lambda: {
            "replication_verdict": "COMPLETE",
            "final_selected_retriever": MODES[1],
            "retrieval_architecture_frozen": True,
            "replication_rerun": False,
            "complete": True,
        },
    )
    monkeypatch.setattr(benchmark, "_corpus_identity", lambda: CORPUS_IDENTITY)
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        provider,
    )
    pipeline.ingest_path(Path("data/synthetic_company") / "recovery-east.md")
    monkeypatch.setattr(
        "rag_workbench.experiments.hybrid_reranker_replication.SEMANTIC_INDEX_IDENTITY",
        pipeline.index_identity,
    )
    return benchmark, provider, fake, FINAL_CASES


def _fake_production_analysis(cases):
    rows = []
    for case in cases:
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "top5": [],
                "timing": {
                    "query_embedding_ms": 1.0,
                    "cache_lookup_ms": 0.0,
                    "dense_ms": 1.0,
                    "bm25_ms": 1.0,
                    "rrf_ms": 0.1,
                    "reranker_ms": 1.0,
                },
                "embedding_cache_hit": False,
                "dense_candidates": [],
                "bm25_candidates": [],
                "rrf_union": [],
                "cross_encoder_pairs": 0,
                "required_ce_ranks": {},
                "top5_complete": False,
                "failure_taxonomy": "NONE",
                "pool": {},
            }
        )
    return {
        "cases": rows,
        "usage": {"new_query_embedding_calls": 0},
        "metrics": {},
        "candidate_pool": {},
        "three_document": {},
        "category_metrics": {},
        "latency": {},
        "lexical_rescues": {},
        "cross_encoder_conversion": {},
    }


def test_final_v1_retrieve_and_execute_are_one_shot(db_session, monkeypatch) -> None:
    from rag_workbench.experiments.hybrid_reranker_replication import FINAL_CASES

    benchmark, _, _, _ = _seed_frozen_replication(db_session, monkeypatch)
    monkeypatch.setattr(
        benchmark, "_run_production", lambda mode: _fake_production_analysis(FINAL_CASES)
    )
    benchmark.retrieve()
    with pytest.raises(ValueError, match="one-shot"):
        benchmark.retrieve()
    record = benchmark.initialize()
    record.execution_started_at = datetime.now(UTC)
    db_session.flush()
    with pytest.raises(ValueError, match="one-shot"):
        benchmark.execute()


def test_final_v1_acl_never_reaches_cross_encoder_and_sol_is_not_used_in_retrieve(
    db_session, monkeypatch
) -> None:
    from rag_workbench.experiments.hybrid_reranker_replication import FINAL_CASES

    benchmark, _, fake, _ = _seed_frozen_replication(db_session, monkeypatch)
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        FakeSemanticProvider(64),
    )
    pipeline.ingest_path(Path("data/synthetic_company") / "hr-compensation.md")
    pipeline.ingest_path(Path("data/synthetic_company") / "hr-benefits-private.md")
    monkeypatch.setattr(
        "rag_workbench.experiments.hybrid_reranker_replication.SEMANTIC_INDEX_IDENTITY",
        pipeline.index_identity,
    )
    sample = next(item for item in FINAL_CASES if item.case_id == "fv1_acl_01")
    monkeypatch.setattr(
        "rag_workbench.experiments.hybrid_reranker_replication.FINAL_CASES",
        (sample,),
    )
    judge_before = benchmark._historical_judge_calls()
    analysis = benchmark._run_production(MODES[1])
    assert benchmark._historical_judge_calls() == judge_before
    texts = " ".join(text for call in fake.calls for _, text in call)
    assert "HR-COMP-900" not in texts
    assert analysis["usage"]["unauthorized_chunks_to_cross_encoder"] == 0


def test_final_v1_release_architecture_is_a_new_immutable_record(
    db_session, monkeypatch
) -> None:
    from rag_workbench.db.models import RetrievalArchitectureRecord
    from rag_workbench.experiments.hybrid_reranker_replication import (
        FINAL_DATASET_ID,
        RELEASE_ARCHITECTURE_ID,
    )

    benchmark, _, _, _ = _seed_frozen_replication(db_session, monkeypatch)
    record = benchmark.initialize()
    record.completed_at = datetime.now(UTC)
    db_session.flush()
    first = benchmark.freeze_release_architecture()
    assert first.architecture_id == RELEASE_ARCHITECTURE_ID
    assert first.architecture_id != ARCHITECTURE_ID
    assert first.immutable is True
    original = db_session.get(RetrievalArchitectureRecord, ARCHITECTURE_ID)
    assert original is not None
    assert original.dataset_id == DATASET_ID
    again = benchmark.freeze_release_architecture()
    assert again.architecture_id == first.architecture_id
    first.selected_retriever = MODES[0]
    db_session.flush()
    with pytest.raises(ValueError, match="immutable"):
        benchmark.freeze_release_architecture()
    assert db_session.get(RetrievalArchitectureRecord, FINAL_DATASET_ID) is None

