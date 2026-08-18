from collections import Counter
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from rag_workbench.answerability.cache import gate_cache_key
from rag_workbench.config import Settings
from rag_workbench.db.models import EndToEndBenchmarkRecord
from rag_workbench.experiments.reranker_e2e_benchmark import (
    COMMON_CONFIGURATION,
    DATASET,
    DATASET_HASH,
    DATASET_ID,
    MAXIMUM_PRIOR_OVERLAP,
    MODES,
    PIPELINE_A,
    PIPELINE_B,
    RERANKER_REVISION,
    SUCCESS_POLICY,
    RerankerEndToEndBenchmark,
    maximum_prior_dataset_overlap,
)


def test_new_dataset_has_frozen_identity_and_requested_distribution() -> None:
    assert DATASET.dataset_version == DATASET_ID
    assert len(DATASET.cases) == 60
    assert len({case.case_id for case in DATASET.cases}) == 60
    assert Counter(case.category for case in DATASET.cases) == {
        "multidoc_two": 12,
        "multidoc_three": 16,
        "near_duplicate": 8,
        "exact_identifier": 6,
        "version_region": 6,
        "semantic_paraphrase": 4,
        "acl_sensitive": 4,
        "partial_no_answer": 4,
    }
    assert DATASET_HASH == "288b26b0f1adc9c75b2b1d2c617c362f9aa0540ebb5591157bd728ff94fa2e16"
    assert maximum_prior_dataset_overlap() == pytest.approx(MAXIMUM_PRIOR_OVERLAP)


def test_answerable_multidoc_cases_really_require_distinct_sources() -> None:
    for case in DATASET.cases:
        if case.category == "multidoc_two":
            assert not case.should_abstain
            assert len(set(case.expected_document_ids)) == 2
        if case.category == "multidoc_three":
            assert not case.should_abstain
            assert len(set(case.expected_document_ids)) == 3


def test_pipelines_freeze_only_the_selected_retrieval_difference() -> None:
    assert PIPELINE_A["retrieval_mode"] == "DENSE"
    assert PIPELINE_A["final_top_k"] == 5
    assert PIPELINE_B["retrieval_mode"] == "DENSE_CROSS_ENCODER_RERANK"
    assert PIPELINE_B["dense_candidate_depth"] == 20
    assert PIPELINE_B["final_top_k"] == 5
    assert PIPELINE_B["reranker_revision"] == RERANKER_REVISION
    for key, value in COMMON_CONFIGURATION.items():
        assert PIPELINE_A[key] == value
        assert PIPELINE_B[key] == value
    assert SUCCESS_POLICY["minimum_correct_answer_gain"] == 3
    assert SUCCESS_POLICY["maximum_unsupported_answer_increase"] == 0


def test_gate_cache_reuses_identical_ordered_evidence_but_not_changed_order() -> None:
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
    gate_evidence = RerankerEndToEndBenchmark._gate_evidence(evidence)
    same = RerankerEndToEndBenchmark._gate_evidence(list(evidence))
    reordered = RerankerEndToEndBenchmark._gate_evidence(list(reversed(evidence)))
    options = {
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "gate_version": "1",
        "prompt_version": "evidence-sufficiency-v1",
    }
    assert (
        gate_cache_key("question", gate_evidence, **options)[0]
        == gate_cache_key("question", same, **options)[0]
    )
    assert (
        gate_cache_key("question", gate_evidence, **options)[0]
        != gate_cache_key("question", reordered, **options)[0]
    )


def test_paired_analysis_records_retrieval_rescue_conversion_and_regression() -> None:
    def case(case_id: str, coverage: bool, status: str):
        return SimpleNamespace(
            case_id=case_id,
            expected_abstain=False,
            retrieval_coverage_complete=coverage,
            status=status,
        )

    results = {
        MODES[0]: [case("rescue", False, "abstained"), case("regress", True, "answered")],
        MODES[1]: [case("rescue", True, "answered"), case("regress", False, "abstained")],
    }
    transitions, conversion, regression = RerankerEndToEndBenchmark._paired_analysis(results, [])
    assert transitions == {
        "CORRECT_ANSWER→INCORRECT_ABSTENTION": 1,
        "INCORRECT_ABSTENTION→CORRECT_ANSWER": 1,
    }
    assert conversion["retrieval_rescue_to_correct_answer_count"] == 1
    assert conversion["retrieval_to_answer_conversion_rate"] == 1.0
    assert regression["new_incorrect_abstentions"] == 1


def test_sealed_record_blocks_preparation_reuse(db_session, monkeypatch) -> None:
    benchmark = RerankerEndToEndBenchmark(db_session, Settings(_env_file=None))
    monkeypatch.setattr(benchmark, "_verify_frozen_dependencies", lambda: None)
    record = benchmark.initialize()
    record.preparation_started_at = datetime.now(UTC)
    db_session.commit()
    with pytest.raises(ValueError, match="preparation is one-shot"):
        benchmark.prepare()
    persisted = db_session.get(EndToEndBenchmarkRecord, DATASET_ID)
    assert persisted.pipeline_a_configuration["success_policy"] == SUCCESS_POLICY
    assert persisted.pipeline_b_configuration["success_policy"] == SUCCESS_POLICY
