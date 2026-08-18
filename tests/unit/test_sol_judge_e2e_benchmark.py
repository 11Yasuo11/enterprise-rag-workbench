from collections import Counter
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from rag_workbench.answerability.cache import gate_cache_key
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    evidence_gate_schema,
    hosted_judge_chat_payload,
    hosted_judge_request_settings,
)
from rag_workbench.config import Settings
from rag_workbench.experiments.judge_e2e_benchmark import (
    DATASET as PREVIOUS_JUDGE_DATASET,
)
from rag_workbench.experiments.judge_e2e_benchmark import (
    DATASET_HASH as PREVIOUS_JUDGE_HASH,
)
from rag_workbench.experiments.judge_e2e_benchmark import (
    SHARED_RETRIEVAL,
    V2_TEMPLATE_HASH,
)
from rag_workbench.experiments.judge_e2e_benchmark import (
    maximum_prior_dataset_overlap as previous_judge_overlap,
)
from rag_workbench.experiments.reranker_e2e_benchmark import RerankerEndToEndBenchmark
from rag_workbench.experiments.sol_judge_e2e_benchmark import (
    DATASET,
    DATASET_HASH,
    DATASET_ID,
    HISTORICAL_SCHEMA_IDENTITY,
    JUDGE_A,
    JUDGE_B,
    LUNA_MODEL,
    MODES,
    SCHEMA_IDENTITY,
    SOL_MODEL,
    SUCCESS_POLICY,
    SolJudgeEndToEndBenchmark,
    maximum_prior_dataset_overlap,
    request_parameter_parity,
)


def test_dataset_identity_distribution_and_overlap_guard() -> None:
    assert DATASET.dataset_version == DATASET_ID
    assert len(DATASET.cases) == 60
    assert len({case.case_id for case in DATASET.cases}) == 60
    assert Counter(case.category for case in DATASET.cases) == {
        "multidoc_two": 10,
        "multidoc_three": 20,
        "near_duplicate": 6,
        "exact_identifier": 6,
        "version_region": 6,
        "semantic_paraphrase": 4,
        "acl_sensitive": 4,
        "partial_no_answer": 4,
    }
    assert DATASET_HASH == "10c947941644ab9e29ac6e4dc8d59ae87872859cee032eb5b8fb89aaadc2cf55"
    assert maximum_prior_dataset_overlap() == pytest.approx(0.4444444444444444)
    assert maximum_prior_dataset_overlap() < 0.5
    previous_ids = {case.case_id for case in PREVIOUS_JUDGE_DATASET.cases}
    assert previous_ids.isdisjoint({case.case_id for case in DATASET.cases})
    previous_questions = {case.question for case in PREVIOUS_JUDGE_DATASET.cases}
    assert previous_questions.isdisjoint({case.question for case in DATASET.cases})


def test_historical_judge_dataset_and_v2_identities_remain_frozen() -> None:
    assert PREVIOUS_JUDGE_HASH == "7d9b0e6f45e7d385be92ce5c37d229a1602957316e85bfe0c5f828cf514f9651"
    assert previous_judge_overlap() == pytest.approx(0.47058823529411764)
    assert V2_TEMPLATE_HASH == "dbcce103d21a48254c69819959297fbf981464fb99b1760e5f01d316a34a9940"
    assert SCHEMA_IDENTITY == HISTORICAL_SCHEMA_IDENTITY
    assert SCHEMA_IDENTITY == (
        "6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621"
    )


def test_shared_retrieval_prompt_schema_and_request_parity() -> None:
    assert SHARED_RETRIEVAL["dense_candidate_depth"] == 20
    assert SHARED_RETRIEVAL["final_top_k"] == 5
    assert SHARED_RETRIEVAL["dense_threshold"] == 0.28
    assert JUDGE_A["prompt_version"] == EVIDENCE_GATE_PROMPT_VERSION
    assert JUDGE_B["prompt_version"] == EVIDENCE_GATE_PROMPT_VERSION
    assert JUDGE_A["schema_identity"] == JUDGE_B["schema_identity"] == SCHEMA_IDENTITY
    assert JUDGE_A["model"] == LUNA_MODEL
    assert JUDGE_B["model"] == SOL_MODEL
    assert JUDGE_A["request_settings"] == JUDGE_B["request_settings"]
    settings = hosted_judge_request_settings()
    assert settings["temperature"] == 0
    assert settings["reasoning_effort"] == "none"
    assert settings["max_completion_tokens"] == 160
    assert settings["pro_mode"] is False
    parity = request_parameter_parity()
    assert parity["matched"] is True
    dummy = RerankerEndToEndBenchmark._gate_evidence(
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
    luna = hosted_judge_chat_payload(model=LUNA_MODEL, question="q", chunks=dummy)
    sol = hosted_judge_chat_payload(model=SOL_MODEL, question="q", chunks=dummy)
    assert luna["model"] != sol["model"]
    luna.pop("model")
    sol.pop("model")
    assert luna == sol
    assert SUCCESS_POLICY["minimum_false_negative_fixes"] == 4


def test_model_specific_cache_isolation_and_same_model_reuse() -> None:
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
        "gate_version": "1",
        "prompt_version": EVIDENCE_GATE_PROMPT_VERSION,
    }
    luna = gate_cache_key("question", evidence, model=LUNA_MODEL, **common)[0]
    sol = gate_cache_key("question", evidence, model=SOL_MODEL, **common)[0]
    luna_again = gate_cache_key("question", evidence, model=LUNA_MODEL, **common)[0]
    assert luna != sol
    assert luna == luna_again


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
            case("kept", abstain=False, complete=True, status="answered"),
        ],
        MODES[1]: [
            case("rescue", abstain=False, complete=True, status="answered"),
            case("unsafe", abstain=True, complete=False, status="answered"),
            case("kept", abstain=False, complete=True, status="abstained"),
        ],
    }
    analysis = SolJudgeEndToEndBenchmark._paired_analysis(results)
    assert analysis["false_negative_rescues"] == 1
    assert analysis["luna_correct_to_sol_false_abstention"] == 1
    assert analysis["net_false_negative_improvement"] == 0
    assert analysis["false_positive_regressions"]["count"] == 1
    assert analysis["false_positive_regressions"]["answers_despite_incomplete_retrieval"] == [
        "unsafe"
    ]
    assert analysis["transitions"] == {
        "CORRECT_ABSTENTION→UNSUPPORTED_ANSWER": 1,
        "CORRECT_ANSWER→INCORRECT_ABSTENTION": 1,
        "INCORRECT_ABSTENTION→CORRECT_ANSWER": 1,
    }


def test_root_cause_and_generation_failure_are_separate() -> None:
    common = {
        "expected_abstain": False,
        "retrieval_coverage_complete": True,
        "status": "abstained",
        "supporting_context_loss": False,
        "failure_types": (),
    }
    generation = SimpleNamespace(
        **common,
        answerability_result={"answerable": True},
        answerability_operational_error=None,
    )
    luna_fn = SimpleNamespace(
        **common,
        answerability_result={"answerable": False},
        answerability_operational_error=None,
    )
    invalid = SimpleNamespace(
        **common,
        answerability_result={"answerable": False},
        answerability_operational_error="INVALID_SUPPORTING_ID",
    )
    assert SolJudgeEndToEndBenchmark._root_cause(generation, MODES[0]) == "GENERATION_FAILURE"
    assert (
        SolJudgeEndToEndBenchmark._root_cause(luna_fn, MODES[0])
        == "LUNA_EVIDENCE_GATE_FALSE_NEGATIVE"
    )
    assert (
        SolJudgeEndToEndBenchmark._root_cause(luna_fn, MODES[1])
        == "SOL_EVIDENCE_GATE_FALSE_NEGATIVE"
    )
    assert SolJudgeEndToEndBenchmark._root_cause(invalid, MODES[1]) == "INVALID_SUPPORTING_ID"


def test_retrieval_complete_judge_metrics_are_separate_from_generation() -> None:
    from rag_workbench.experiments.judge_e2e_benchmark import FrozenJudgeEndToEndBenchmark

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
    judged = FrozenJudgeEndToEndBenchmark._judge_decision_metrics(cases)
    assert judged["true_positive"] == 1
    assert judged["false_negative"] == 1
    assert judged["false_positive"] == 0
    assert judged["f1"] == pytest.approx(2 / 3)


def test_sealed_dataset_blocks_preparation_and_execution_reuse(db_session, monkeypatch) -> None:
    benchmark = SolJudgeEndToEndBenchmark(db_session, Settings(_env_file=None))
    monkeypatch.setattr(benchmark, "_verify_dependencies", lambda: None)
    record = benchmark.initialize()
    record.preparation_started_at = datetime.now(UTC)
    db_session.commit()
    with pytest.raises(ValueError, match="preparation is one-shot"):
        benchmark.prepare()
    record.prepared_at = datetime.now(UTC)
    record.prepared_cases = [{"unused": True}]
    record.execution_started_at = datetime.now(UTC)
    db_session.commit()
    with pytest.raises(ValueError, match="evaluation is one-shot"):
        benchmark.execute()
    assert record.pipeline_a_configuration["model"] == LUNA_MODEL
    assert record.pipeline_b_configuration["model"] == SOL_MODEL
    assert record.pipeline_a_configuration["prompt_version"] == EVIDENCE_GATE_PROMPT_VERSION
    assert evidence_gate_schema()["additionalProperties"] is False


def test_schema_identity_is_computed_from_frozen_sufficiency_schema() -> None:
    import hashlib
    import json

    assert (
        hashlib.sha256(
            json.dumps(evidence_gate_schema(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == SCHEMA_IDENTITY
    )
