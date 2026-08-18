from collections import Counter
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from rag_workbench.answerability.cache import gate_cache_key
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_COVERAGE_PROMPT_VERSION,
    EVIDENCE_GATE_PROMPT_VERSION,
    evidence_coverage_template_hash,
)
from rag_workbench.config import Settings
from rag_workbench.experiments.judge_e2e_benchmark import (
    DATASET,
    DATASET_HASH,
    DATASET_ID,
    JUDGE_A,
    JUDGE_B,
    MODES,
    SHARED_RETRIEVAL,
    SUCCESS_POLICY,
    V2_TEMPLATE_HASH,
    FrozenJudgeEndToEndBenchmark,
    maximum_prior_dataset_overlap,
)
from rag_workbench.experiments.reranker_e2e_benchmark import RerankerEndToEndBenchmark


def test_dataset_identity_distribution_and_overlap_guard() -> None:
    assert DATASET.dataset_version == DATASET_ID
    assert len(DATASET.cases) == 60
    assert len({case.case_id for case in DATASET.cases}) == 60
    assert Counter(case.category for case in DATASET.cases) == {
        "multidoc_two": 12,
        "multidoc_three": 18,
        "near_duplicate": 6,
        "exact_identifier": 6,
        "version_region": 6,
        "semantic_paraphrase": 4,
        "acl_sensitive": 4,
        "partial_no_answer": 4,
    }
    assert DATASET_HASH == "7d9b0e6f45e7d385be92ce5c37d229a1602957316e85bfe0c5f828cf514f9651"
    assert maximum_prior_dataset_overlap() == pytest.approx(0.47058823529411764)
    assert maximum_prior_dataset_overlap() < 0.5


def test_frozen_v2_identity_and_shared_retrieval_configuration() -> None:
    assert evidence_coverage_template_hash() == V2_TEMPLATE_HASH
    assert JUDGE_A["prompt_version"] == EVIDENCE_GATE_PROMPT_VERSION
    assert JUDGE_B["prompt_version"] == EVIDENCE_COVERAGE_PROMPT_VERSION
    assert JUDGE_B["statuses"] == ["SUPPORTED", "PARTIAL", "MISSING", "CONFLICTING"]
    assert SHARED_RETRIEVAL["dense_candidate_depth"] == 20
    assert SHARED_RETRIEVAL["final_top_k"] == 5
    assert SHARED_RETRIEVAL["dense_threshold"] == 0.28
    assert SUCCESS_POLICY["minimum_false_negative_fixes"] == 4


def test_prompt_versions_are_cache_isolated_for_identical_evidence() -> None:
    evidence = RerankerEndToEndBenchmark._gate_evidence(
        [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "document_version_id": "v1",
                "version": "1",
                "text": "evidence",
            }
        ]
    )
    common = {
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "gate_version": "1",
    }
    a = gate_cache_key("question", evidence, prompt_version=EVIDENCE_GATE_PROMPT_VERSION, **common)[
        0
    ]
    b = gate_cache_key(
        "question", evidence, prompt_version=EVIDENCE_COVERAGE_PROMPT_VERSION, **common
    )[0]
    assert a != b


def test_false_negative_rescue_false_positive_and_transition_accounting() -> None:
    def case(case_id: str, *, abstain: bool, complete: bool, status: str):
        return SimpleNamespace(
            case_id=case_id,
            expected_abstain=abstain,
            retrieval_coverage_complete=complete,
            status=status,
        )

    results = {
        MODES[0]: [
            case("rescue", abstain=False, complete=True, status="abstained"),
            case("unsafe", abstain=True, complete=False, status="abstained"),
        ],
        MODES[1]: [
            case("rescue", abstain=False, complete=True, status="answered"),
            case("unsafe", abstain=True, complete=False, status="answered"),
        ],
    }
    analysis = FrozenJudgeEndToEndBenchmark._paired_analysis(results)
    assert analysis["false_negative_rescues"] == 1
    assert analysis["false_positive_regressions"]["count"] == 1
    assert analysis["false_positive_regressions"]["answers_despite_incomplete_retrieval"] == [
        "unsafe"
    ]
    assert analysis["transitions"] == {
        "CORRECT_ABSTENTION→UNSUPPORTED_ANSWER": 1,
        "INCORRECT_ABSTENTION→CORRECT_ANSWER": 1,
    }


def test_judge_decision_metrics_are_separate_from_generation_behavior() -> None:
    cases = (
        SimpleNamespace(
            expected_abstain=False,
            answerability_result={"answerable": True},
            status="abstained",
        ),
        SimpleNamespace(
            expected_abstain=False,
            answerability_result={"answerable": False},
            status="abstained",
        ),
        SimpleNamespace(
            expected_abstain=True,
            answerability_result={"answerable": False},
            status="abstained",
        ),
    )
    metrics = FrozenJudgeEndToEndBenchmark._judge_decision_metrics(cases)
    assert metrics == {
        "case_count": 3,
        "true_positive": 1,
        "false_negative": 1,
        "false_positive": 0,
        "true_negative": 1,
        "accuracy": pytest.approx(2 / 3),
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
    }


def test_root_cause_distinguishes_generation_from_judge_failure() -> None:
    common = {
        "expected_abstain": False,
        "retrieval_coverage_complete": True,
        "status": "abstained",
        "supporting_context_loss": False,
    }
    generation = SimpleNamespace(
        **common, answerability_result={"answerable": True, "requirements": []}
    )
    decomposition = SimpleNamespace(
        **common, answerability_result={"answerable": False, "requirements": []}
    )
    assert (
        FrozenJudgeEndToEndBenchmark._root_cause(generation, MODES[0])
        == "GENERATION_FAILURE"
    )
    assert (
        FrozenJudgeEndToEndBenchmark._root_cause(decomposition, MODES[1])
        == "V2_REQUIREMENT_DECOMPOSITION_FALSE_NEGATIVE"
    )


def test_sealed_dataset_blocks_preparation_reuse(db_session, monkeypatch) -> None:
    benchmark = FrozenJudgeEndToEndBenchmark(db_session, Settings(_env_file=None))
    monkeypatch.setattr(benchmark, "_verify_dependencies", lambda: None)
    record = benchmark.initialize()
    record.preparation_started_at = datetime.now(UTC)
    db_session.commit()
    with pytest.raises(ValueError, match="preparation is one-shot"):
        benchmark.prepare()
    assert record.pipeline_a_configuration["prompt_version"] == EVIDENCE_GATE_PROMPT_VERSION
    assert record.pipeline_b_configuration["template_hash"] == V2_TEMPLATE_HASH
