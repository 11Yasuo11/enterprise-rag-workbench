import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from rag_workbench.answerability.cache import gate_cache_key
from rag_workbench.answerability.openai_compatible import EVIDENCE_GATE_PROMPT_VERSION
from rag_workbench.config import Settings
from rag_workbench.db.models import RetrievalBenchmarkRecord, RetrievalBenchmarkRunRecord
from rag_workbench.evaluation.retrieval_dataset import RetrievalGroundTruthCase
from rag_workbench.experiments import hybrid_reranker_benchmark as benchmark_module
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    CALIBRATION_CASES,
    CASES,
    CONFIGURATION,
    DATASET_HASH,
    DATASET_ID,
    DENSE_DEPTH,
    FINAL_TOP_K,
    HOLDOUT_CASES,
    MAXIMUM_PRIOR_OVERLAP,
    MODES,
    RRF_K,
    SELECTION_POLICY,
    SOL_MODEL,
    SPLIT_IDENTITY,
    SPLIT_SEED,
    UNION_LIMIT,
    HybridRerankerBenchmark,
    apply_selection_policy,
    compare_three_document,
    deterministic_hybrid_reranker_split,
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


def case(**updates: object) -> RetrievalGroundTruthCase:
    payload = {
        "case_id": "case",
        "question": "question",
        "required_document_ids": ("doc-a", "doc-b", "doc-c"),
        "required_version_ids": {"doc-a": "1", "doc-b": "1", "doc-c": "1"},
        "expected_access_behavior": "ALLOW_REQUIRED",
        "expected_answerability": True,
        "category": "multidoc_three",
    }
    payload.update(updates)
    return RetrievalGroundTruthCase(**payload)  # type: ignore[arg-type]


def test_dataset_identity_distribution_split_and_overlap_guard() -> None:
    assert DATASET_ID == "acmeai-hybrid-reranker-eval-v1"
    assert len(CASES) == 60
    assert len({item.case_id for item in CASES}) == 60
    assert Counter(item.category for item in CASES) == {
        "multidoc_two": 8,
        "multidoc_three": 26,
        "near_duplicate": 6,
        "exact_identifier": 6,
        "version_region": 6,
        "semantic_paraphrase": 4,
        "acl_sensitive": 2,
        "partial_no_answer": 2,
    }
    assert DATASET_HASH == "d5b69e67a4b322d03e8f9530e0526142c1176bf4112febc3cc5c61635ba9a451"
    assert maximum_prior_dataset_overlap() == pytest.approx(MAXIMUM_PRIOR_OVERLAP)
    assert maximum_prior_dataset_overlap() == pytest.approx(0.4)
    first = deterministic_hybrid_reranker_split(CASES, seed=SPLIT_SEED)
    second = deterministic_hybrid_reranker_split(CASES, seed=SPLIT_SEED)
    assert first == second
    assert first[2] == SPLIT_IDENTITY
    assert len(CALIBRATION_CASES) == 40
    assert len(HOLDOUT_CASES) == 20
    assert {item.case_id for item in CALIBRATION_CASES}.isdisjoint(
        item.case_id for item in HOLDOUT_CASES
    )
    questions = {item.question for item in CASES}
    assert len(questions) == 60
    historical = []
    for path in Path("data/eval").glob("*.json"):
        if path.name == "acmeai_hybrid_reranker_eval_v1.json":
            continue
        historical.extend(json.loads(path.read_text())["cases"])
    assert questions.isdisjoint(item["question"] for item in historical)


def test_historical_overlap_guards_remain_frozen() -> None:
    assert sol_overlap() == pytest.approx(0.4444444444444444)
    assert judge_overlap() == pytest.approx(0.47058823529411764)
    assert reranker_overlap() == pytest.approx(0.391304347826087)


def test_frozen_candidate_and_cross_encoder_configuration() -> None:
    assert MODES == ("DENSE_CROSS_ENCODER_RERANK", "HYBRID_CROSS_ENCODER_RERANK")
    assert DENSE_DEPTH == BM25_DEPTH == 20
    assert UNION_LIMIT == 30
    assert RRF_K == 60
    assert FINAL_TOP_K == 5
    assert CONFIGURATION["dense_threshold"] == 0.28
    assert CONFIGURATION["bm25"]["k1"] == 1.2
    assert CONFIGURATION["bm25"]["b"] == 0.75
    assert CONFIGURATION["bm25"]["version"] == "bm25-okapi-v1"
    assert CONFIGURATION["rrf_k"] == 60
    assert CONFIGURATION["candidate_union_limit"] == 30
    assert CONFIGURATION["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L6-v2"
    assert CONFIGURATION["reranker_revision"] == RERANKER_REVISION
    assert CONFIGURATION["judge"]["model"] == SOL_MODEL
    assert CONFIGURATION["judge"]["prompt_version"] == EVIDENCE_GATE_PROMPT_VERSION
    assert CONFIGURATION["judge"]["used_for_retrieval_selection"] is False
    assert SELECTION_POLICY["all_required_coverage_minimum_gain"] == 0.10
    assert SELECTION_POLICY["three_document_coverage_minimum_gain"] == 0.20
    assert BM25Config().k1 == 1.2
    assert BM25Config().b == 0.75


def test_bm25_tokenizer_and_rrf_k_are_historical() -> None:
    tokens = tokenize_bm25("POL-2026-004 Atlas-X17")
    assert "pol-2026-004" in tokens
    assert {"pol", "2026", "004", "atlas", "x17"} <= set(tokens)
    fused = reciprocal_rank_fusion(
        [result("d1", "dense-doc", 1), result("shared", "both", 2)],
        [result("b1", "bm25-doc", 1, source="bm25"), result("shared", "both", 2, source="bm25")],
        top_k=30,
        rrf_k=60,
    )
    assert len(fused) == 3
    assert fused[0].chunk_id == "shared"
    assert fused[0].found_by_dense and fused[0].found_by_bm25
    assert all(item.fusion_score is not None and item.fusion_score < 0.28 for item in fused)


def test_selection_policy_requires_material_gain_without_safety_regression() -> None:
    dense = {
        "all_required_evidence_coverage_at_5": 0.40,
        "three_document_coverage_at_5": 0.20,
        "required_evidence_recall_at_5": 0.70,
        "exact_identifier_recall_at_5": 1.0,
        "semantic_success": 1.0,
        "version_correctness": 1.0,
        "acl_safety": 1.0,
    }
    hybrid = {
        **dense,
        "all_required_evidence_coverage_at_5": 0.51,
        "three_document_coverage_at_5": 0.45,
    }
    assert apply_selection_policy(dense, hybrid)["selected_mode"] == MODES[1]
    no_gain = {
        **hybrid,
        "all_required_evidence_coverage_at_5": 0.45,
        "three_document_coverage_at_5": 0.30,
    }
    assert apply_selection_policy(dense, no_gain)["selected_mode"] == MODES[0]
    recall_regression = {**hybrid, "required_evidence_recall_at_5": 0.67}
    assert apply_selection_policy(dense, recall_regression)["selected_mode"] == MODES[0]


def test_lexical_rescue_and_cross_encoder_conversion_accounting() -> None:
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
        },
        {
            "case_id": "truncated",
            "expected_answerability": True,
            "required_document_ids": ["lost"],
            "required_dense_ranks": {"lost": None},
            "required_bm25_ranks": {"lost": 19},
            "required_rrf_ranks": {"lost": 31},
            "required_ce_ranks": {},
            "top5": [{"document_id": "other"}],
        },
    ]
    lexical = HybridRerankerBenchmark._lexical_rescues(rows)
    assert lexical["bm25_only_required_evidence"] == 2
    assert lexical["retained_by_rrf"] == 2
    assert lexical["sent_to_cross_encoder"] == 1
    assert lexical["promoted_to_top5"] == 1
    assert lexical["lost"] == 1
    conversion = HybridRerankerBenchmark._conversion(rows)
    assert conversion["lexical_promoted"] == 1
    assert conversion["dense_only_required_retained"] == 1
    assert conversion["dense_only_required_lost"] == 0


def test_three_document_effect_classifies_fixed_worsened_and_impossible() -> None:
    effect = compare_three_document(
        [
            {"case_id": "fixed", "top5_complete": False},
            {"case_id": "worse", "top5_complete": True},
            {"case_id": "still", "top5_complete": False},
        ],
        [
            {"case_id": "fixed", "top5_complete": True},
            {"case_id": "worse", "top5_complete": False},
            {"case_id": "still", "top5_complete": False},
        ],
    )
    assert effect["fixed_by_hybrid_reranker"] == ["fixed"]
    assert effect["worsened_by_hybrid_reranker"] == ["worse"]
    assert effect["still_impossible_after_candidate_expansion"] == ["still"]


def test_taxonomy_uses_traces() -> None:
    required = case()
    bm25 = [
        result("c-a", "doc-a", 1, source="bm25"),
        result("c-b", "doc-b", 2, source="bm25"),
        result("c-c", "doc-c", 3, source="bm25"),
    ]
    union = reciprocal_rank_fusion([], bm25, top_k=30, rrf_k=60)
    assert HybridRerankerBenchmark._taxonomy(required, [], [], [], []) == "BOTH_BRANCHES_MISS"
    assert (
        HybridRerankerBenchmark._taxonomy(required, [], bm25, union, union[:2])
        == "CROSS_ENCODER_FAILED_TO_PROMOTE"
    )
    truncated = reciprocal_rank_fusion([], bm25, top_k=2, rrf_k=60)
    assert (
        HybridRerankerBenchmark._taxonomy(required, [], bm25, truncated, truncated)
        == "RRF_TRUNCATION_LOSS"
    )


def test_gate_cache_isolates_different_ordered_top5() -> None:
    evidence = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "document_version_id": "v1",
            "version": "1",
            "text": "first",
        },
        {
            "chunk_id": "c2",
            "document_id": "d2",
            "document_version_id": "v2",
            "version": "1",
            "text": "second",
        },
    ]
    from rag_workbench.experiments.reranker_e2e_benchmark import RerankerEndToEndBenchmark

    same = RerankerEndToEndBenchmark._gate_evidence(evidence)
    reordered = RerankerEndToEndBenchmark._gate_evidence(list(reversed(evidence)))
    options = {
        "provider": "openai",
        "model": SOL_MODEL,
        "gate_version": "1",
        "prompt_version": EVIDENCE_GATE_PROMPT_VERSION,
    }
    assert gate_cache_key("q", same, **options)[0] == gate_cache_key("q", same, **options)[0]
    assert gate_cache_key("q", same, **options)[0] != gate_cache_key("q", reordered, **options)[0]


def _prepare_benchmark(db_session, monkeypatch, documents: tuple[str, ...]):
    provider = FakeSemanticProvider(64)
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        provider,
    )
    for name in documents:
        pipeline.ingest_path(Path("data/synthetic_company") / name)
    monkeypatch.setattr(benchmark_module, "SEMANTIC_INDEX_IDENTITY", pipeline.index_identity)
    fake = FakeModel()
    benchmark = HybridRerankerBenchmark(
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


def test_shared_query_embedding_and_no_sol_during_calibration(db_session, monkeypatch) -> None:
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
    analysis = benchmark._run_partition("calibration-sample", (sample,))
    assert provider.query_calls == 1
    assert benchmark._historical_judge_calls() == judge_before
    dense = analysis[MODES[0]]["cases"][0]
    hybrid = analysis[MODES[1]]["cases"][0]
    assert dense["candidate_count_sent_to_reranker"] <= DENSE_DEPTH
    assert hybrid["candidate_count_sent_to_reranker"] <= UNION_LIMIT
    assert all(len(fake_call) <= UNION_LIMIT for fake_call in fake.calls)
    runs = db_session.scalars(select(RetrievalBenchmarkRunRecord)).all()
    assert all((run.usage or {}).get("document_embedding_calls") == 0 for run in runs)
    assert all((run.usage or {}).get("external_reranker_calls") == 0 for run in runs)


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
    benchmark._run_partition("acl", (sample,))
    texts = " ".join(text for call in fake.calls for _, text in call)
    assert "HR-BEN-771" not in texts


def test_holdout_requires_lock_and_is_one_shot(db_session, monkeypatch) -> None:
    benchmark, _, _ = _prepare_benchmark(db_session, monkeypatch, ("recovery-east.md",))
    record = benchmark.initialize()
    with pytest.raises(ValueError, match="locked before holdout"):
        benchmark.holdout_retrieve()
    record.selected_retrieval_mode = MODES[0]
    record.locked_at = datetime.now(UTC)
    record.holdout_started_at = datetime.now(UTC)
    db_session.flush()
    with pytest.raises(ValueError, match="one-shot"):
        benchmark.holdout_retrieve()
    persisted = db_session.get(RetrievalBenchmarkRecord, DATASET_ID)
    assert persisted.holdout_completed_at is None
    with pytest.raises(ValueError, match="already locked"):
        benchmark.calibrate()


def test_union_bound_and_dense_threshold_are_not_applied_to_rrf() -> None:
    dense = [result(f"d{index}", f"dense-{index}", index) for index in range(1, 21)]
    lexical = [result(f"b{index}", f"bm25-{index}", index, source="bm25") for index in range(1, 21)]
    fused = reciprocal_rank_fusion(dense, lexical, top_k=10_000, rrf_k=60)
    bounded = fused[:UNION_LIMIT]
    assert len({item.chunk_id for item in fused}) == 40
    assert len(bounded) == 30
    assert all(item.score == item.fusion_score for item in bounded)
    assert all(item.fusion_score < 0.28 for item in bounded)


def test_answerable_three_document_cases_require_three_sources() -> None:
    for item in CASES:
        if item.category == "multidoc_three":
            assert item.expected_answerability
            assert len(set(item.required_document_ids)) == 3
        if item.category == "multidoc_two":
            assert len(set(item.required_document_ids)) == 2
        if item.category in {"acl_sensitive", "partial_no_answer"}:
            assert item.should_abstain
            assert not item.expected_answerability


def test_paired_rescue_conversion_and_abstention_safety() -> None:
    def row(
        case_id: str,
        *,
        complete: bool,
        abstain: bool,
        status: str,
        category: str = "multidoc_three",
    ):
        return SimpleNamespace(
            case_id=case_id,
            category=category,
            expected_abstain=abstain,
            retrieval_coverage_complete=complete,
            status=status,
        )

    results = {
        MODES[0]: [
            row("rescue", complete=False, abstain=False, status="abstained"),
            row("safe", complete=False, abstain=True, status="abstained", category="acl_sensitive"),
        ],
        MODES[1]: [
            row("rescue", complete=True, abstain=False, status="answered"),
            row("safe", complete=False, abstain=True, status="abstained", category="acl_sensitive"),
        ],
    }
    analysis = HybridRerankerBenchmark._holdout_analysis(
        HybridRerankerBenchmark.__new__(HybridRerankerBenchmark), results
    )
    assert analysis["hybrid_retrieval_rescues"] == 1
    assert analysis["rescues_to_correct_answer"] == 1
    assert analysis["abstention_safety"][MODES[1]]["unsupported_answers"] == 0
    assert analysis["three_document"][MODES[1]]["correct_answer_rate"] == 1.0


def test_unauthorized_sol_guard_applies_only_to_exclude_forbidden() -> None:
    required = case(
        expected_access_behavior="PREFER_REQUIRED_EXCLUDE_SIMILAR",
        forbidden_document_ids=("near-dup",),
        required_document_ids=("doc-a",),
        category="near_duplicate",
    )
    top5 = [result("c-a", "doc-a", 1), result("c-n", "near-dup", 2)]
    assert HybridRerankerBenchmark._taxonomy(required, top5, [], top5, top5) == "NONE"
    acl = case(
        expected_access_behavior="EXCLUDE_FORBIDDEN",
        forbidden_document_ids=("hr-benefits-private",),
        required_document_ids=(),
        expected_answerability=False,
        category="acl_sensitive",
    )
    leaked = [result("c-hr", "hr-benefits-private", 1)]
    assert HybridRerankerBenchmark._taxonomy(acl, leaked, leaked, leaked, leaked) == "ACL_FAILURE"
