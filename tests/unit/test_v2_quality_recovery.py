from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_workbench.db.models import (
    EndToEndBenchmarkRecord,
    EndToEndBenchmarkRunRecord,
    ResearchArchitectureRecord,
    RetrievalArchitectureRecord,
    RetrievalBenchmarkRecord,
    RetrievalBenchmarkRunRecord,
)
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
from rag_workbench.experiments.v2_quality_recovery import (
    CENSUS_CATEGORIES,
    FROZEN_V1_ARCHITECTURE_HASH,
    PRIMARY_V2_BOTTLENECK,
    RECOMMENDED_RANKING_INTERVENTION,
    V1_BENCHMARK_HEADING,
    V2_BENCHMARK_HEADING,
    V2_RESEARCH_ARCHITECTURE_ID,
    V2_SECURITY_GUARDRAILS,
    V2QualityRecoveryBaseline,
    build_failure_census,
    build_judge_false_negative_diagnostic,
    build_operational_diagnostic,
    build_ranking_diagnostic,
    classify_census_category,
    classify_judge_false_negative,
    classify_ranking_appearance,
    expected_v1_pipeline_control,
    pipeline_control,
    stable_hash,
    v2_control_configuration,
    verify_v1_file_identities,
)


def _case(**overrides) -> HybridRerankerCase:
    payload = {
        "case_id": "fv1_three_01",
        "category": "multidoc_three",
        "question": "Need three pamphlets",
        "required_document_ids": ["doc-a", "doc-b", "doc-c"],
        "expected_access_behavior": "ALLOW_REQUIRED",
        "expected_answerability": True,
        "should_abstain": False,
        "expected_document_ids": ["doc-a", "doc-b", "doc-c"],
    }
    payload.update(overrides)
    return HybridRerankerCase.model_validate(payload)


def _hit(document_id: str, rank: int, chunk_id: str | None = None) -> dict:
    return {
        "rank": rank,
        "chunk_id": chunk_id or f"{document_id}-{rank}",
        "document_id": document_id,
        "score": 1.0 - rank * 0.01,
    }


def _retrieval(*, top5, union=None, taxonomy="CROSS_ENCODER_FAILED_TO_PROMOTE", **overrides):
    union = union or top5
    required = overrides.pop("required_document_ids", ["doc-a", "doc-b", "doc-c"])
    top5_docs = {item["document_id"] for item in top5}
    payload = {
        "case_id": "fv1_three_01",
        "category": "multidoc_three",
        "expected_answerability": True,
        "required_document_ids": required,
        "dense_candidates": union,
        "bm25_candidates": union,
        "rrf_union": union,
        "top5": top5,
        "pool": {"all_required_evidence_coverage": 1.0, "required_evidence_recall": 1.0},
        "required_dense_ranks": {item: index + 1 for index, item in enumerate(required)},
        "required_bm25_ranks": {item: index + 1 for index, item in enumerate(required)},
        "required_rrf_ranks": {item: index + 1 for index, item in enumerate(required)},
        "required_ce_ranks": {"doc-a": 1, "doc-b": 2, "doc-c": 6},
        "top5_complete": set(required) <= top5_docs,
        "failure_taxonomy": taxonomy if set(required) - top5_docs else "NONE",
        "metrics": {"unauthorized_result_exposure": 0},
    }
    payload.update(overrides)
    return payload


def _trace(*, answerable=False, status="abstained", **overrides):
    payload = {
        "question_id": "fv1_three_01",
        "status": status,
        "answerability_result": {
            "answerable": answerable,
            "reason_code": "SUFFICIENT_EVIDENCE" if answerable else "MISSING_REQUIRED_FACT",
            "supporting_chunk_ids": ["doc-a-1"] if answerable else [],
        },
        "supporting_chunk_ids": ["doc-a-1"] if answerable else [],
        "retrieved_chunk_ids": ["doc-a-1", "doc-a-2", "doc-b-1", "noise-1", "noise-2"],
        "generation_context_chunk_ids": ["doc-a-1"] if answerable else [],
        "supporting_context_loss": 0.0,
        "citation_correctness": 1.0,
        "version_correct": 1.0,
        "security_passed": True,
        "answerability_operational_error": None,
        "answer": "yes" if status == "answered" else None,
    }
    payload.update(overrides)
    return payload


def test_v1_file_identities_remain_frozen() -> None:
    checks = verify_v1_file_identities()
    assert checks["final_v1_architecture_record"] == RELEASE_ARCHITECTURE_ID
    assert checks["final_v1_benchmark_dataset_hash"] is True
    assert checks["v1_experiment_rows_preserved"] is True
    assert V2_RESEARCH_ARCHITECTURE_ID != RELEASE_ARCHITECTURE_ID


def test_benchmark_md_preserves_v1_and_adds_v2_research_section() -> None:
    text = Path("BENCHMARK.md").read_text()
    assert text.count(V1_BENCHMARK_HEADING) == 1
    assert text.count(V2_BENCHMARK_HEADING) == 1
    v1 = text.split(V1_BENCHMARK_HEADING, 1)[1].split(V2_BENCHMARK_HEADING, 1)[0]
    v2 = text.split(V2_BENCHMARK_HEADING, 1)[1]
    assert "HYBRID_CROSS_ENCODER_RERANK" in v1
    assert "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1" in v1
    assert FROZEN_V1_ARCHITECTURE_HASH in v1
    assert V2_RESEARCH_ARCHITECTURE_ID in v2
    assert "research status = ACTIVE" in v2
    assert "production status = FALSE" in v2
    assert "diagnosis-only" in v2
    assert "promotion evidence" in v2
    assert PRIMARY_V2_BOTTLENECK in v2
    assert "DOCUMENT_DIVERSIFIED_TOP5" in v2
    assert "Do not implement it." in v2


def test_same_document_crowding_is_detected() -> None:
    case = _case()
    retrieval = _retrieval(
        top5=[
            _hit("doc-a", 1, "doc-a-1"),
            _hit("doc-a", 2, "doc-a-2"),
            _hit("doc-a", 3, "doc-a-3"),
            _hit("doc-b", 4, "doc-b-1"),
            _hit("doc-b", 5, "doc-b-2"),
        ],
        union=[
            _hit("doc-a", 1),
            _hit("doc-b", 2),
            _hit("doc-c", 3),
            _hit("noise", 4),
        ],
        required_ce_ranks={"doc-a": 1, "doc-b": 4, "doc-c": 6},
    )
    crowding = classify_ranking_appearance(case, retrieval)
    assert crowding["appearance"] == "SAME_DOCUMENT_CROWDING"
    assert crowding["required_evidence_displaced_from_top5"] == ["doc-c"]
    assert crowding["number_of_required_documents"] == 3
    assert crowding["number_of_duplicate_same_document_chunks_in_top5"] == 3
    assert crowding["cross_encoder_rank_of_missing_required_evidence"]["doc-c"] == 6


def test_pointwise_misordering_has_no_duplicate_slots() -> None:
    case = _case()
    retrieval = _retrieval(
        top5=[
            _hit("doc-a", 1),
            _hit("noise-1", 2),
            _hit("noise-2", 3),
            _hit("noise-3", 4),
            _hit("noise-4", 5),
        ],
        required_ce_ranks={"doc-a": 1, "doc-b": 6, "doc-c": 7},
    )
    crowding = classify_ranking_appearance(case, retrieval)
    assert crowding["appearance"] == "POINTWISE_RELEVANCE_MISORDERING"
    assert crowding["number_of_duplicate_same_document_chunks_in_top5"] == 0


def test_near_duplicate_crowding_uses_forbidden_twin() -> None:
    case = _case(
        case_id="fv1_dup_01",
        category="near_duplicate",
        required_document_ids=["recovery-runbook-east"],
        expected_document_ids=["recovery-runbook-east"],
        forbidden_document_ids=["recovery-runbook-west"],
        expected_access_behavior="PREFER_REQUIRED_EXCLUDE_SIMILAR",
    )
    retrieval = _retrieval(
        top5=[
            _hit("recovery-runbook-west", 1),
            _hit("noise-1", 2),
            _hit("noise-2", 3),
            _hit("noise-3", 4),
            _hit("noise-4", 5),
        ],
        required_document_ids=["recovery-runbook-east"],
        required_ce_ranks={"recovery-runbook-east": 6},
        taxonomy="CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE",
    )
    crowding = classify_ranking_appearance(case, retrieval)
    assert crowding["appearance"] == "NEAR_DUPLICATE_CROWDING"
    assert crowding["near_duplicate_crowding"] == 1


def test_census_separates_request_error_from_judge_false_negative() -> None:
    ranking_case = _case()
    ranking_retrieval = _retrieval(
        top5=[
            _hit("doc-a", 1),
            _hit("doc-a", 2),
            _hit("doc-a", 3),
            _hit("doc-b", 4),
            _hit("doc-b", 5),
        ]
    )
    id_case = _case(
        case_id="fv1_id_01",
        category="exact_identifier",
        required_document_ids=["engineering-deployment-handbook"],
        expected_document_ids=["engineering-deployment-handbook"],
    )
    id_retrieval = _retrieval(
        top5=[_hit("engineering-deployment-handbook", 1)],
        required_document_ids=["engineering-deployment-handbook"],
        taxonomy="NONE",
        required_ce_ranks={"engineering-deployment-handbook": 1},
    )
    dup_case = _case(
        case_id="fv1_dup_05",
        category="near_duplicate",
        required_document_ids=["recovery-runbook-east"],
        expected_document_ids=["recovery-runbook-east"],
        forbidden_document_ids=["recovery-runbook-west"],
    )
    dup_retrieval = _retrieval(
        top5=[_hit("recovery-runbook-east", 1)],
        required_document_ids=["recovery-runbook-east"],
        taxonomy="NONE",
        required_ce_ranks={"recovery-runbook-east": 1},
    )
    gen_case = _case(
        case_id="fv1_ver_03",
        category="version_region",
        required_document_ids=["remote-work-policy"],
        expected_document_ids=["remote-work-policy"],
        security_checks=("version",),
    )
    gen_retrieval = _retrieval(
        top5=[_hit("remote-work-policy", 1)],
        required_document_ids=["remote-work-policy"],
        taxonomy="NONE",
        required_ce_ranks={"remote-work-policy": 1},
    )
    cases = (ranking_case, id_case, dup_case, gen_case)
    prepared = [
        {"case_id": ranking_case.case_id, "retrieval_row": ranking_retrieval},
        {"case_id": id_case.case_id, "retrieval_row": id_retrieval},
        {"case_id": dup_case.case_id, "retrieval_row": dup_retrieval},
        {"case_id": gen_case.case_id, "retrieval_row": gen_retrieval},
    ]
    traces = [
        _trace(question_id=ranking_case.case_id),
        _trace(
            question_id=id_case.case_id,
            retrieved_chunk_ids=["engineering-deployment-handbook-1"],
        ),
        _trace(
            question_id=dup_case.case_id,
            answerability_operational_error="JUDGE_REQUEST_ERROR",
            retrieved_chunk_ids=["recovery-runbook-east-1"],
        ),
        _trace(
            question_id=gen_case.case_id,
            answerable=True,
            status="abstained",
            supporting_chunk_ids=["remote-work-policy-1"],
            retrieved_chunk_ids=["remote-work-policy-1"],
            generation_context_chunk_ids=["remote-work-policy-1"],
        ),
    ]
    assert classify_census_category(ranking_case, ranking_retrieval, traces[0]) == (
        "CROSS_ENCODER_FAILED_TO_PROMOTE"
    )
    assert classify_census_category(id_case, id_retrieval, traces[1]) == (
        "EVIDENCE_GATE_FALSE_NEGATIVE"
    )
    assert classify_census_category(dup_case, dup_retrieval, traces[2]) == "JUDGE_REQUEST_ERROR"
    assert classify_census_category(gen_case, gen_retrieval, traces[3]) == "GENERATION_FAILURE"
    census = build_failure_census(cases, prepared, traces)
    assert census["promotion_evidence"] is False
    assert census["failed_answerable_case_count"] == 4
    assert set(census["counts"]) == set(CENSUS_CATEGORIES)
    assert census["counts"]["CROSS_ENCODER_FAILED_TO_PROMOTE"] == 1
    assert census["counts"]["EVIDENCE_GATE_FALSE_NEGATIVE"] == 1
    assert census["counts"]["JUDGE_REQUEST_ERROR"] == 1
    assert census["counts"]["GENERATION_FAILURE"] == 1
    ranking = build_ranking_diagnostic(cases, prepared)
    assert ranking["pool_complete_top5_incomplete_count"] == 1
    assert ranking["primary_appearance"] == "SAME_DOCUMENT_CROWDING"
    assert ranking["ranking_fix_implemented"] is False
    judge = build_judge_false_negative_diagnostic(cases, prepared, traces)
    assert judge["retrieval_complete_sol_false_negative_count"] == 2
    assert judge["class_counts"]["EXACT_IDENTIFIER_FALSE_NEGATIVE"] == 1
    assert judge["class_counts"]["REQUEST_ERROR"] == 1
    assert judge["judge_modified"] is False
    assert classify_judge_false_negative(id_case, id_retrieval, traces[1])["empty_support"] is True
    assert classify_judge_false_negative(id_case, id_retrieval, traces[1])[
        "all_required_evidence_actually_present"
    ]
    operational = build_operational_diagnostic(cases, prepared, traces)
    assert operational["historical_calls_retried"] is False
    classes = {item["case_id"]: item["class"] for item in operational["cases"]}
    assert classes["fv1_ver_03"] == "GENERATOR_FAILURE"
    assert classes["fv1_dup_05"] == "PROVIDER_REQUEST_FAILURE"


def test_v2_control_configuration_is_pipeline_equivalent_and_not_production() -> None:
    v1 = {
        **expected_v1_pipeline_control(),
        "architecture_id": RELEASE_ARCHITECTURE_ID,
        "release_name": "enterprise-rag-workbench-v1",
        "architecture_hash": FROZEN_V1_ARCHITECTURE_HASH,
    }
    v2 = v2_control_configuration(v1)
    assert pipeline_control(v2) == pipeline_control(v1)
    assert v2["architecture_id"] == V2_RESEARCH_ARCHITECTURE_ID
    assert v2["parent_architecture"] == RELEASE_ARCHITECTURE_ID
    assert v2["research_status"] == "ACTIVE"
    assert v2["production_status"] is False
    assert v2["diagnosis_only"] is True
    assert v2["promotion_evidence_forbidden"] is True
    assert v2["control_equivalence_hash"] == stable_hash(pipeline_control(v1))
    assert V2_SECURITY_GUARDRAILS["acl_safety"] == 1.0
    assert V2_SECURITY_GUARDRAILS["unauthorized_chunks_to_reranker"] == 0
    assert V2_SECURITY_GUARDRAILS["unauthorized_chunks_to_judge"] == 0


def _seed_v1(db_session, cases, prepared, traces) -> None:
    control = expected_v1_pipeline_control()
    release_configuration = {
        **control,
        "architecture_id": RELEASE_ARCHITECTURE_ID,
        "release_name": "enterprise-rag-workbench-v1",
        "final_benchmark_dataset_id": FINAL_DATASET_ID,
        "final_benchmark_dataset_hash": FINAL_DATASET_HASH,
        "final_benchmark_run_id": FINAL_DATASET_ID,
        "immutable": True,
        "no_v1_quality_tuning": True,
        "architecture_hash": FROZEN_V1_ARCHITECTURE_HASH,
    }
    db_session.add(
        RetrievalArchitectureRecord(
            architecture_id=ARCHITECTURE_ID,
            selected_retriever=MODES[1],
            configuration={"immutable": True},
            selection_policy={},
            dataset_id="acmeai-hybrid-reranker-replication-v1",
            dataset_hash="f9144a0bbc6f15de030963199b8a784d9e23031a017258a8b6636b131cb04114",
            retrieval_metrics={},
            immutable=True,
        )
    )
    db_session.add(
        RetrievalArchitectureRecord(
            architecture_id=RELEASE_ARCHITECTURE_ID,
            selected_retriever=MODES[1],
            configuration=release_configuration,
            selection_policy={"no_architecture_selection": True},
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
            split_identity="final-v1",
            split_seed=0,
            calibration_case_ids=[item.case_id for item in cases],
            holdout_case_ids=[],
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


def test_v2_initialize_preserves_v1_and_is_not_production(db_session, monkeypatch) -> None:
    case = _case()
    prepared = [
        {
            "case_id": case.case_id,
            "retrieval_row": _retrieval(
                top5=[
                    _hit("doc-a", 1),
                    _hit("doc-a", 2),
                    _hit("doc-a", 3),
                    _hit("doc-b", 4),
                    _hit("doc-b", 5),
                ]
            ),
        }
    ]
    traces = [_trace(question_id=case.case_id)]
    _seed_v1(db_session, (case,), prepared, traces)
    monkeypatch.setattr(
        "rag_workbench.experiments.hybrid_reranker_replication.FINAL_CASES",
        (case,),
    )
    v1_before = db_session.get(RetrievalArchitectureRecord, RELEASE_ARCHITECTURE_ID)
    v1_hash = dict(v1_before.configuration)
    baseline = V2QualityRecoveryBaseline(db_session)
    first = baseline.initialize()
    assert first.architecture_id == V2_RESEARCH_ARCHITECTURE_ID
    assert first.parent_architecture_id == RELEASE_ARCHITECTURE_ID
    assert first.research_status == "ACTIVE"
    assert first.production_status is False
    assert first.diagnosis_only is True
    assert first.promotion_evidence_forbidden is True
    assert first.control_equivalence_hash == stable_hash(expected_v1_pipeline_control())
    assert first.primary_bottleneck == PRIMARY_V2_BOTTLENECK
    assert first.recommended_ranking_intervention == RECOMMENDED_RANKING_INTERVENTION
    assert db_session.get(RetrievalArchitectureRecord, RELEASE_ARCHITECTURE_ID).configuration == (
        v1_hash
    )
    again = baseline.initialize()
    assert again.architecture_id == first.architecture_id
    first.production_status = True
    db_session.flush()
    with pytest.raises(ValueError, match="immutable"):
        baseline.initialize()
    assert db_session.get(ResearchArchitectureRecord, RELEASE_ARCHITECTURE_ID) is None
    assert db_session.get(RetrievalArchitectureRecord, RELEASE_ARCHITECTURE_ID) is not None
    assert RECOMMENDED_RANKING_INTERVENTION.startswith("DOCUMENT_DIVERSIFIED_TOP5")
