import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_workbench.config import Settings
from rag_workbench.db.models import (
    EndToEndBenchmarkRecord,
    EndToEndBenchmarkRunRecord,
    ResearchArchitectureRecord,
    RetrievalArchitectureRecord,
    RetrievalBenchmarkRecord,
    RetrievalBenchmarkRunRecord,
)
from rag_workbench.experiments import v2_document_diversity as diversity_module
from rag_workbench.experiments.hybrid_reranker_benchmark import MODES, HybridRerankerCase
from rag_workbench.experiments.hybrid_reranker_replication import (
    ARCHITECTURE_ID,
    FINAL_DATASET_HASH,
    FINAL_DATASET_ID,
    RELEASE_ARCHITECTURE_ID,
)
from rag_workbench.experiments.reranker_e2e_benchmark import (
    CORPUS_IDENTITY,
    RERANKER_REVISION,
    SEMANTIC_INDEX_IDENTITY,
)
from rag_workbench.experiments.v2_document_diversity import (
    CANDIDATE_MODE,
    CASES,
    CONTROL_MODE,
    DATASET_HASH,
    DATASET_ID,
    DATASET_PATH,
    EXPECTED_DISTRIBUTION,
    PHASE1_BENCHMARK_HEADING,
    SELECTION_POLICY,
    DocumentDiversityCase,
    apply_selection_policy,
    dataset_overlap_report,
    marker_hits,
    ranking_candidate,
    retrieval_complete,
    unique_document_stats,
)
from rag_workbench.experiments.v2_quality_recovery import (
    FROZEN_V1_ARCHITECTURE_HASH,
    V1_BENCHMARK_HEADING,
    V2_BENCHMARK_HEADING,
    V2_RESEARCH_ARCHITECTURE_ID,
    expected_v1_pipeline_control,
)
from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
from rag_workbench.ingestion.pipeline import IngestionPipeline
from rag_workbench.providers.embeddings import HashingEmbeddingProvider
from rag_workbench.reranking.base import RerankedResult
from rag_workbench.reranking.cross_encoder import CrossEncoderReranker
from rag_workbench.reranking.document_diversity import (
    EVALUATION_LABEL_FIELDS,
    select_document_diversified_top5,
)
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


def _hit(document_id: str, rank: int, chunk_id: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or f"{document_id}-{rank}",
        document_id=document_id,
        document_version_id=f"v-{document_id}",
        text=f"text {document_id} {rank}",
        rank=rank,
        score=1.0 - rank * 0.01,
        source="source.md",
        source_type="markdown",
        title=document_id,
        version="1",
    )


def _reranked(results: list[RetrievalResult]) -> list[RerankedResult]:
    return [
        RerankedResult(
            result=item,
            original_dense_rank=item.rank,
            dense_score=item.score,
            reranker_score=item.score,
            reranked_rank=index,
        )
        for index, item in enumerate(results, start=1)
    ]


def test_dataset_identity_distribution_and_overlap_guard() -> None:
    assert DATASET_ID == "acmeai-v2-document-diversity-eval-v1"
    assert hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() == DATASET_HASH
    assert len(CASES) == 80
    distribution = {
        name: sum(item.category == name for item in CASES) for name in EXPECTED_DISTRIBUTION
    }
    assert distribution == EXPECTED_DISTRIBUTION
    assert sum(item.requires_multiple_chunks_same_document for item in CASES) >= 8
    overlap = dataset_overlap_report()
    assert overlap["pass"] is True
    assert overlap["maximum_normalized_overlap"] < 0.5


def test_historical_overlap_identities_remain_frozen() -> None:
    from rag_workbench.experiments.hybrid_reranker_benchmark import (
        maximum_prior_dataset_overlap as hybrid_overlap,
    )
    from rag_workbench.experiments.hybrid_reranker_replication import (
        HISTORICAL_OVERLAP_IDENTITIES,
        final_dataset_overlap_report,
    )
    from rag_workbench.experiments.hybrid_reranker_replication import (
        maximum_prior_dataset_overlap as replication_overlap,
    )
    from rag_workbench.experiments.judge_e2e_benchmark import (
        maximum_prior_dataset_overlap as judge_overlap,
    )
    from rag_workbench.experiments.reranker_e2e_benchmark import (
        maximum_prior_dataset_overlap as reranker_overlap,
    )
    from rag_workbench.experiments.sol_judge_e2e_benchmark import (
        maximum_prior_dataset_overlap as sol_overlap,
    )

    assert hybrid_overlap() == pytest.approx(HISTORICAL_OVERLAP_IDENTITIES["hybrid_reranker"])
    assert replication_overlap() == pytest.approx(HISTORICAL_OVERLAP_IDENTITIES["replication"])
    assert sol_overlap() == pytest.approx(HISTORICAL_OVERLAP_IDENTITIES["sol_judge"])
    assert judge_overlap() == pytest.approx(HISTORICAL_OVERLAP_IDENTITIES["judge_e2e"])
    assert reranker_overlap() == pytest.approx(HISTORICAL_OVERLAP_IDENTITIES["reranker_e2e"])
    assert final_dataset_overlap_report()["pass"] is True


def test_canonical_document_id_keeps_highest_chunk_and_is_deterministic() -> None:
    ranked = _reranked(
        [
            _hit("doc-a", 1, "a1"),
            _hit("doc-a", 2, "a2"),
            _hit("doc-b", 3, "b1"),
            _hit("doc-a", 4, "a3"),
            _hit("doc-c", 5, "c1"),
            _hit("doc-d", 6, "d1"),
            _hit("doc-e", 7, "e1"),
        ]
    )
    selected = select_document_diversified_top5(ranked, top_k=5)
    assert [item.result.chunk_id for item in selected] == ["a1", "b1", "c1", "d1", "e1"]
    assert [item.result.document_id for item in selected] == [
        "doc-a",
        "doc-b",
        "doc-c",
        "doc-d",
        "doc-e",
    ]
    assert select_document_diversified_top5(ranked, top_k=5)[0].result.chunk_id == "a1"
    assert len(selected) <= 5


def test_duplicate_document_chunks_are_removed_and_topk_caps() -> None:
    ranked = [
        ranking_candidate(item) for item in _reranked([_hit("doc-a", rank) for rank in range(1, 8)])
    ]
    selected = select_document_diversified_top5(ranked, top_k=5)
    assert [item["document_id"] for item in selected] == ["doc-a"]
    assert len(selected) == 1
    assert len(selected) <= 5


def test_no_evaluation_label_leakage_into_diversification() -> None:
    import inspect

    source = inspect.getsource(select_document_diversified_top5)
    for field in EVALUATION_LABEL_FIELDS:
        assert field not in source
    ranked = [
        {
            **ranking_candidate(_reranked([_hit("doc-a", 1)])[0]),
            "required_document_ids": ["secret"],
            "expected_answer": "leak",
            "category": "multidoc_three",
        }
    ]
    with pytest.raises(ValueError, match="evaluation labels leaked"):
        select_document_diversified_top5(ranked, top_k=5)


def test_same_document_multi_chunk_metric_and_regression() -> None:
    case = DocumentDiversityCase.model_validate(
        {
            "case_id": "same",
            "category": "single_document",
            "question": "Need two chunks",
            "required_document_ids": ["doc-a"],
            "required_chunk_markers": ["alpha-marker", "beta-marker"],
            "requires_multiple_chunks_same_document": True,
            "expected_access_behavior": "ALLOW_REQUIRED",
            "expected_answerability": True,
            "should_abstain": False,
            "expected_document_ids": ["doc-a"],
        }
    )
    control = [
        {"chunk_id": "a1", "document_id": "doc-a", "text": "alpha-marker first"},
        {"chunk_id": "a2", "document_id": "doc-a", "text": "beta-marker second"},
    ]
    candidate = [{"chunk_id": "a1", "document_id": "doc-a", "text": "alpha-marker first"}]
    assert marker_hits(case, control)["all_required_evidence_coverage_at_5"] == 1.0
    assert marker_hits(case, candidate)["all_required_evidence_coverage_at_5"] == 0.0
    assert retrieval_complete(case, control) is True
    assert retrieval_complete(case, candidate) is False


def test_crowding_rescue_and_selection_policy_guardrails() -> None:
    control = {
        "all_required_evidence_coverage_at_5": 0.70,
        "three_document_coverage_at_5": 0.25,
        "required_evidence_recall_at_5": 0.90,
        "exact_identifier_recall_at_5": 1.0,
        "semantic_success": 1.0,
        "version_correctness": 1.0,
        "acl_safety": 1.0,
    }
    candidate = {
        **control,
        "all_required_evidence_coverage_at_5": 0.85,
        "three_document_coverage_at_5": 0.50,
    }
    same_doc = {
        "control_all_required_coverage_at_5": 1.0,
        "candidate_all_required_coverage_at_5": 0.95,
    }
    won = apply_selection_policy(control, candidate, regressions=2, rescues=5, same_doc=same_doc)
    assert won["selected_ranking"] == CANDIDATE_MODE
    too_many_regressions = apply_selection_policy(
        control, candidate, regressions=6, rescues=5, same_doc=same_doc
    )
    assert too_many_regressions["selected_ranking"] == CONTROL_MODE
    exact_fail = apply_selection_policy(
        control,
        {**candidate, "exact_identifier_recall_at_5": 0.90},
        regressions=0,
        rescues=5,
        same_doc=same_doc,
    )
    assert exact_fail["selected_ranking"] == CONTROL_MODE
    same_doc_fail = apply_selection_policy(
        control,
        candidate,
        regressions=0,
        rescues=5,
        same_doc={
            "control_all_required_coverage_at_5": 1.0,
            "candidate_all_required_coverage_at_5": 0.80,
        },
    )
    assert same_doc_fail["selected_ranking"] == CONTROL_MODE


def test_unique_document_stats() -> None:
    stats = unique_document_stats(
        [
            {"document_id": "a"},
            {"document_id": "a"},
            {"document_id": "b"},
            {"document_id": "b"},
            {"document_id": "c"},
        ]
    )
    assert stats["unique_document_ids"] == 3
    assert stats["repeated_document_chunks"] == 2
    assert stats["duplicate_slot_fraction"] == pytest.approx(0.4)


def test_benchmark_md_preserves_history_and_adds_phase1_heading() -> None:
    text = Path("BENCHMARK.md").read_text()
    assert text.count(V1_BENCHMARK_HEADING) == 1
    assert text.count(V2_BENCHMARK_HEADING) == 1
    assert text.count(PHASE1_BENCHMARK_HEADING) == 1
    v1 = text.split(V1_BENCHMARK_HEADING, 1)[1].split(V2_BENCHMARK_HEADING, 1)[0]
    assert FROZEN_V1_ARCHITECTURE_HASH in v1
    assert "HYBRID_CROSS_ENCODER_RERANK" in v1
    p0 = text.split(V2_BENCHMARK_HEADING, 1)[1].split(PHASE1_BENCHMARK_HEADING, 1)[0]
    assert "Do not implement it." in p0
    phase1 = text.split(PHASE1_BENCHMARK_HEADING, 1)[1]
    assert CONTROL_MODE in phase1
    assert CANDIDATE_MODE in phase1
    assert "Enterprise RAG Workbench v1 remained frozen" in phase1


def _seed_v1(db_session, cases, prepared, traces) -> None:
    control = expected_v1_pipeline_control()
    db_session.add(
        RetrievalArchitectureRecord(
            architecture_id=RELEASE_ARCHITECTURE_ID,
            selected_retriever=MODES[1],
            configuration={**control, "architecture_hash": FROZEN_V1_ARCHITECTURE_HASH},
            selection_policy={},
            dataset_id=FINAL_DATASET_ID,
            dataset_hash=FINAL_DATASET_HASH,
            retrieval_metrics={},
            immutable=True,
        )
    )
    db_session.add(
        RetrievalArchitectureRecord(
            architecture_id=ARCHITECTURE_ID,
            selected_retriever=MODES[1],
            configuration=control,
            selection_policy={},
            dataset_id=FINAL_DATASET_ID,
            dataset_hash=FINAL_DATASET_HASH,
            retrieval_metrics={},
            immutable=True,
        )
    )
    db_session.add(
        RetrievalBenchmarkRecord(
            dataset_id=FINAL_DATASET_ID,
            dataset_hash=FINAL_DATASET_HASH,
            split_identity="final",
            split_seed=0,
            calibration_case_ids=[],
            holdout_case_ids=[item.case_id for item in cases],
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=CORPUS_IDENTITY,
            retrieval_configuration=control,
            selection_policy={},
        )
    )
    db_session.add(
        RetrievalBenchmarkRunRecord(
            dataset_id=FINAL_DATASET_ID,
            partition="final",
            retrieval_mode=MODES[1],
            metrics={},
            category_metrics={},
            case_results=[{"question_id": item.case_id} for item in cases],
            branch_contribution={},
            latency={},
            usage={},
        )
    )
    db_session.add(
        EndToEndBenchmarkRecord(
            dataset_id=FINAL_DATASET_ID,
            dataset_hash=FINAL_DATASET_HASH,
            case_ids=[item.case_id for item in cases],
            category_distribution={},
            generation_method="manual-corpus-grounded-v1",
            maximum_prior_overlap=0.3,
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=CORPUS_IDENTITY,
            reranker_revision=RERANKER_REVISION,
            pipeline_a_configuration={},
            pipeline_b_configuration={},
            prepared_cases=prepared,
            completed_at=datetime.now(UTC),
        )
    )
    db_session.add(
        EndToEndBenchmarkRunRecord(
            dataset_id=FINAL_DATASET_ID,
            mode=MODES[1],
            metrics={},
            retrieval_metrics={},
            category_metrics={},
            case_results=traces,
            latency={},
            usage={},
        )
    )
    db_session.flush()


def test_initialize_freezes_policy_before_results(db_session, monkeypatch) -> None:
    from rag_workbench.experiments.v2_document_diversity import V2DocumentDiversityBenchmark
    from rag_workbench.experiments.v2_quality_recovery import V2QualityRecoveryBaseline

    case = HybridRerankerCase.model_validate(
        {
            "case_id": "fv1_three_01",
            "category": "multidoc_three",
            "question": "Need three pamphlets",
            "required_document_ids": ["doc-a", "doc-b", "doc-c"],
            "expected_access_behavior": "ALLOW_REQUIRED",
            "expected_answerability": True,
            "should_abstain": False,
            "expected_document_ids": ["doc-a", "doc-b", "doc-c"],
        }
    )
    prepared = [
        {
            "case_id": case.case_id,
            "retrieval_row": {
                "case_id": case.case_id,
                "top5": [],
                "rrf_union": [],
                "pool": {"all_required_evidence_coverage": 1.0},
                "top5_complete": False,
                "required_ce_ranks": {},
            },
        }
    ]
    traces = [
        {
            "question_id": case.case_id,
            "status": "abstained",
            "answerability_result": {"answerable": False, "reason_code": "MISSING_REQUIRED_FACT"},
            "supporting_chunk_ids": [],
            "retrieved_chunk_ids": [],
            "generation_context_chunk_ids": [],
            "security_passed": True,
            "version_correct": 1.0,
        }
    ]
    _seed_v1(db_session, (case,), prepared, traces)
    monkeypatch.setattr(
        "rag_workbench.experiments.hybrid_reranker_replication.FINAL_CASES",
        (case,),
    )
    V2QualityRecoveryBaseline(db_session).initialize()
    provider = FakeSemanticProvider(64)
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        provider,
    )
    pipeline.ingest_path(Path("data/synthetic_company") / "operations-continuity.md")
    monkeypatch.setattr(diversity_module, "SEMANTIC_INDEX_IDENTITY", pipeline.index_identity)
    v1_before = db_session.get(RetrievalArchitectureRecord, RELEASE_ARCHITECTURE_ID).configuration
    benchmark = V2DocumentDiversityBenchmark(db_session)
    monkeypatch.setattr(benchmark, "_corpus_identity", lambda: CORPUS_IDENTITY)
    record = benchmark.initialize()
    assert record.dataset_id == DATASET_ID
    assert record.selection_policy == SELECTION_POLICY
    assert record.selection_policy_frozen_at is not None
    assert record.dataset_frozen_at is not None
    assert record.completed_at is None
    assert record.shared_traces is None
    assert record.selected_ranking is None
    again = benchmark.initialize()
    assert again.dataset_id == record.dataset_id
    record.selection_policy = {**SELECTION_POLICY, "tampered": True}
    db_session.flush()
    with pytest.raises(ValueError, match="selection policy changed"):
        V2DocumentDiversityBenchmark(db_session).initialize()
    assert db_session.get(RetrievalArchitectureRecord, RELEASE_ARCHITECTURE_ID).configuration == (
        v1_before
    )
    assert db_session.get(ResearchArchitectureRecord, RELEASE_ARCHITECTURE_ID) is None
    research = db_session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
    assert research.production_status is False
    assert research.phase1_dataset_id == DATASET_ID


def _prepare_run(db_session, monkeypatch, documents: tuple[str, ...], cases):
    from rag_workbench.experiments.v2_document_diversity import V2DocumentDiversityBenchmark

    provider = FakeSemanticProvider(64)
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        provider,
    )
    for name in documents:
        pipeline.ingest_path(Path("data/synthetic_company") / name)
    monkeypatch.setattr(diversity_module, "CASES", cases)
    monkeypatch.setattr(diversity_module, "SEMANTIC_INDEX_IDENTITY", pipeline.index_identity)
    monkeypatch.setattr(
        diversity_module,
        "EXPECTED_DISTRIBUTION",
        {
            name: sum(item.category == name for item in cases)
            for name in sorted({item.category for item in cases})
        },
    )
    fake = FakeModel()
    benchmark = V2DocumentDiversityBenchmark(
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
    monkeypatch.setattr(benchmark, "_corpus_identity", lambda: CORPUS_IDENTITY)
    return benchmark, provider, fake


def test_shared_pool_scores_embedding_and_no_sol(db_session, monkeypatch) -> None:
    subset = tuple(item for item in CASES if item.requires_multiple_chunks_same_document)
    benchmark, provider, fake = _prepare_run(
        db_session,
        monkeypatch,
        (
            "operations-continuity.md",
            "engineering-deployments.md",
            "data-retention.md",
            "finance-expenses.md",
            "support-escalation.md",
            "project-atlas-api.md",
            "recovery-east.md",
        ),
        subset,
    )
    judge_before = benchmark._historical_judge_calls()
    analysis = benchmark._run()
    assert provider.query_calls == len(subset)
    assert benchmark._historical_judge_calls() == judge_before
    assert analysis["usage"]["new_sol_calls"] == 0
    assert analysis["usage"]["generator_calls"] == 0
    assert analysis["usage"]["document_embedding_calls"] == 0
    assert analysis["usage"]["external_reranker_calls"] == 0
    assert all(item["shared_query_embedding"] for item in analysis["shared_traces"])
    assert all(item["shared_cross_encoder_scores"] for item in analysis["shared_traces"])
    assert all(len(item["control_top5"]) <= 5 for item in analysis["shared_traces"])
    assert all(len(item["candidate_top5"]) <= 5 for item in analysis["shared_traces"])
    assert fake.calls
    assert all(len(call) <= 30 for call in fake.calls)


def test_acl_and_version_filtering_precede_ranking(db_session, monkeypatch) -> None:
    acl_case = next(item for item in CASES if item.category == "acl_sensitive")
    version_case = next(item for item in CASES if item.category == "version_region")
    benchmark, _provider, fake = _prepare_run(
        db_session,
        monkeypatch,
        (
            "hr-compensation.md",
            "remote-work-policy-2026.md",
            "remote-work-policy-2025.md",
            "security-incident-policy-2026.md",
            "security-incident-policy-2025.md",
        ),
        (acl_case, version_case),
    )
    analysis = benchmark._run()
    acl_trace = next(
        item for item in analysis["shared_traces"] if item["case_id"] == acl_case.case_id
    )
    for item in acl_trace["cross_encoder_ranked"]:
        assert item["document_id"] != "hr-compensation-bands"
    version_trace = next(
        item for item in analysis["shared_traces"] if item["case_id"] == version_case.case_id
    )
    for item in version_trace["cross_encoder_ranked"]:
        if item["document_id"] == "remote-work-policy":
            assert item["version"] == "2026"
        if item["document_id"] == "security-incident-policy":
            assert item["version"] == "2026"
    assert analysis["usage"]["unauthorized_chunks_to_cross_encoder"] == 0
    assert analysis["usage"]["unauthorized_chunks_entering_candidate_b"] == 0
    assert fake.calls
