import hashlib
from collections import Counter
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
    V2Phase1ExperimentRecord,
)
from rag_workbench.experiments import v2_soft_document_cap as cap_module
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
    PHASE1_BENCHMARK_HEADING,
)
from rag_workbench.experiments.v2_document_diversity import (
    SELECTION_POLICY as PHASE1_POLICY,
)
from rag_workbench.experiments.v2_quality_recovery import (
    FROZEN_V1_ARCHITECTURE_HASH,
    V1_BENCHMARK_HEADING,
    V2_BENCHMARK_HEADING,
    V2_RESEARCH_ARCHITECTURE_ID,
    expected_v1_pipeline_control,
)
from rag_workbench.experiments.v2_soft_document_cap import (
    CANDIDATE_MODE,
    CASES,
    CONTROL_MODE,
    DATASET_HASH,
    DATASET_ID,
    DATASET_PATH,
    EXPECTED_DISTRIBUTION,
    PHASE2_BENCHMARK_HEADING,
    SELECTION_POLICY,
    SoftDocumentCapCase,
    apply_selection_policy,
    classify_regression,
    dataset_overlap_report,
    occupancy_stats,
    slots_freed_in_top5,
)
from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
from rag_workbench.ingestion.pipeline import IngestionPipeline
from rag_workbench.providers.embeddings import HashingEmbeddingProvider
from rag_workbench.reranking.base import RerankedResult
from rag_workbench.reranking.cross_encoder import CrossEncoderReranker
from rag_workbench.reranking.document_diversity import (
    EVALUATION_LABEL_FIELDS,
    select_max_2_chunks_per_document_top5,
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
    assert hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() == DATASET_HASH
    assert len(CASES) == 80
    assert dict(sorted(Counter(item.category for item in CASES).items())) == EXPECTED_DISTRIBUTION
    assert sum(item.requires_exactly_two_chunks_same_document for item in CASES) >= 8
    assert sum(item.crowding_relevant_same_document for item in CASES) >= 4
    report = dataset_overlap_report()
    assert report["pass"] is True
    assert report["maximum_normalized_overlap"] < 0.5


def test_max_two_keeps_first_two_skips_later_and_is_deterministic() -> None:
    ranked = _reranked(
        [
            _hit("doc-a", 1, "a1"),
            _hit("doc-a", 2, "a2"),
            _hit("doc-a", 3, "a3"),
            _hit("doc-b", 4, "b1"),
            _hit("doc-c", 5, "c1"),
            _hit("doc-d", 6, "d1"),
        ]
    )
    selected = select_max_2_chunks_per_document_top5(ranked, top_k=5)
    assert [item.result.chunk_id for item in selected] == ["a1", "a2", "b1", "c1", "d1"]
    assert [item.result.document_id for item in selected] == [
        "doc-a",
        "doc-a",
        "doc-b",
        "doc-c",
        "doc-d",
    ]
    assert select_max_2_chunks_per_document_top5(ranked, top_k=5)[2].result.chunk_id == "b1"
    assert len(selected) <= 5
    with pytest.raises(ValueError, match="frozen at two"):
        select_max_2_chunks_per_document_top5(ranked, top_k=5, max_chunks_per_document=3)


def test_no_evaluation_label_leakage_into_soft_cap() -> None:
    import inspect

    source = inspect.getsource(select_max_2_chunks_per_document_top5)
    for field in EVALUATION_LABEL_FIELDS:
        assert field not in source
    ranked = [
        {
            "document_id": "doc-a",
            "chunk_id": "a1",
            "rank": 1,
            "score": 1.0,
            "required_document_ids": ["secret"],
            "category": "multidoc_three",
        }
    ]
    with pytest.raises(ValueError, match="evaluation labels leaked"):
        select_max_2_chunks_per_document_top5(ranked, top_k=5)


def test_occupancy_and_slots_freed() -> None:
    top5 = [
        {"document_id": "a"},
        {"document_id": "a"},
        {"document_id": "a"},
        {"document_id": "b"},
        {"document_id": "c"},
    ]
    stats = occupancy_stats(top5)
    assert stats["unique_document_ids"] == 3
    assert stats["max_chunks_from_one_document"] == 3
    assert stats["has_3plus"] == 1
    assert slots_freed_in_top5(top5) == 1


def test_soft_cap_rescue_and_regression_policy() -> None:
    same = {
        "control_all_required_coverage_at_5": 1.0,
        "candidate_all_required_coverage_at_5": 1.0,
    }
    control = {
        "required_evidence_recall_at_5": 0.90,
        "all_required_evidence_coverage_at_5": 0.80,
        "three_document_coverage_at_5": 0.70,
        "exact_identifier_recall_at_5": 1.0,
        "semantic_success": 1.0,
        "near_duplicate_preferred_source_success": 1.0,
        "version_correctness": 1.0,
        "acl_safety": 1.0,
    }
    candidate = {
        **control,
        "all_required_evidence_coverage_at_5": 0.86,
        "three_document_coverage_at_5": 0.82,
        "required_evidence_recall_at_5": 0.91,
    }
    won = apply_selection_policy(control, candidate, regressions=1, rescues=2, same_doc=same)
    assert won["selected_ranking"] == CANDIDATE_MODE
    lost = apply_selection_policy(control, candidate, regressions=3, rescues=2, same_doc=same)
    assert lost["selected_ranking"] == CONTROL_MODE
    case = SoftDocumentCapCase.model_validate(
        {
            "case_id": "two",
            "category": "single_document",
            "question": "Need two chunks",
            "required_document_ids": ["doc-a"],
            "requires_exactly_two_chunks_same_document": True,
            "expected_access_behavior": "ALLOW_REQUIRED",
            "expected_answerability": True,
            "should_abstain": False,
            "expected_document_ids": ["doc-a"],
        }
    )
    assert classify_regression(case) == "TWO_REQUIRED_CHUNKS_SAME_DOCUMENT"


def test_benchmark_md_preserves_history_and_adds_phase2_heading() -> None:
    text = Path("BENCHMARK.md").read_text()
    assert text.count(V1_BENCHMARK_HEADING) == 1
    assert text.count(V2_BENCHMARK_HEADING) == 1
    assert text.count(PHASE1_BENCHMARK_HEADING) == 1
    assert text.count(PHASE2_BENCHMARK_HEADING) == 1
    p0 = text.split(V2_BENCHMARK_HEADING, 1)[1].split(PHASE1_BENCHMARK_HEADING, 1)[0]
    assert "Do not implement it." in p0
    phase1 = text.split(PHASE1_BENCHMARK_HEADING, 1)[1].split(PHASE2_BENCHMARK_HEADING, 1)[0]
    assert "DOCUMENT_DIVERSIFIED_TOP5" in phase1
    phase2 = text.split(PHASE2_BENCHMARK_HEADING, 1)[1]
    assert CONTROL_MODE in phase2
    assert CANDIDATE_MODE in phase2
    assert "Frozen v1 was not modified" in phase2
    assert "Results pending live execution" not in phase2
    assert "FROZEN_FOR_CURRENT_V2_CYCLE" in phase2
    assert "Selected ranking" in phase2 or "Selected V2 ranking" in phase2
    assert PHASE1_POLICY["primary_condition_a_three_document_coverage_gain"] == 0.15


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


def _seed_phase1(db_session) -> None:
    now = datetime.now(UTC)
    db_session.add(
        V2Phase1ExperimentRecord(
            dataset_id="acmeai-v2-document-diversity-eval-v1",
            architecture_id=V2_RESEARCH_ARCHITECTURE_ID,
            dataset_hash="79229d49eac1091f648632531019064f73e0e877038c0d1d88618c783dd502c1",
            case_ids=["vdv_three_01"],
            category_distribution={"multidoc_three": 1},
            generation_method="manual-corpus-grounded-v1",
            maximum_prior_overlap=0.3,
            overlap_report={"pass": True},
            selection_policy=PHASE1_POLICY,
            control_configuration={},
            candidate_configuration={},
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=CORPUS_IDENTITY,
            dataset_frozen_at=now,
            selection_policy_frozen_at=now,
            completed_at=now,
        )
    )
    db_session.flush()


def test_initialize_freezes_policy_before_results(db_session, monkeypatch) -> None:
    from rag_workbench.experiments.v2_quality_recovery import V2QualityRecoveryBaseline
    from rag_workbench.experiments.v2_soft_document_cap import V2SoftDocumentCapBenchmark

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
    _seed_phase1(db_session)
    provider = FakeSemanticProvider(64)
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        provider,
    )
    pipeline.ingest_path(Path("data/synthetic_company") / "operations-continuity.md")
    monkeypatch.setattr(cap_module, "SEMANTIC_INDEX_IDENTITY", pipeline.index_identity)
    v1_before = db_session.get(RetrievalArchitectureRecord, RELEASE_ARCHITECTURE_ID).configuration
    benchmark = V2SoftDocumentCapBenchmark(db_session)
    monkeypatch.setattr(benchmark, "_corpus_identity", lambda: CORPUS_IDENTITY)
    record = benchmark.initialize()
    assert record.dataset_id == DATASET_ID
    assert record.selection_policy == SELECTION_POLICY
    assert record.selection_policy["independent_from_phase1_policy"] is True
    assert record.completed_at is None
    assert record.shared_traces is None
    again = benchmark.initialize()
    assert again.dataset_id == record.dataset_id
    record.selection_policy = {**SELECTION_POLICY, "tampered": True}
    db_session.flush()
    with pytest.raises(ValueError, match="selection policy changed"):
        V2SoftDocumentCapBenchmark(db_session).initialize()
    assert db_session.get(RetrievalArchitectureRecord, RELEASE_ARCHITECTURE_ID).configuration == (
        v1_before
    )
    research = db_session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
    assert research.production_status is False
    assert research.phase2_dataset_id == DATASET_ID


def _prepare_run(db_session, monkeypatch, documents: tuple[str, ...], cases):
    from rag_workbench.experiments.v2_soft_document_cap import V2SoftDocumentCapBenchmark

    provider = FakeSemanticProvider(64)
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        provider,
    )
    for name in documents:
        pipeline.ingest_path(Path("data/synthetic_company") / name)
    monkeypatch.setattr(cap_module, "CASES", cases)
    monkeypatch.setattr(cap_module, "SEMANTIC_INDEX_IDENTITY", pipeline.index_identity)
    fake = FakeModel()
    benchmark = V2SoftDocumentCapBenchmark(
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
    subset = tuple(item for item in CASES if item.requires_exactly_two_chunks_same_document)
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
            "project-atlas-launch.md",
            "recovery-east.md",
            "recovery-west.md",
            "remote-work-policy-2026.md",
            "security-incident-policy-2026.md",
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
