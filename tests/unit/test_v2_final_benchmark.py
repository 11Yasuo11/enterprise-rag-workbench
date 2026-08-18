import hashlib
import inspect
from collections import Counter
from pathlib import Path

from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    evidence_sufficiency_schema_identity,
    evidence_sufficiency_template_hash,
    hosted_judge_request_settings,
)
from rag_workbench.experiments.hybrid_reranker_replication import RELEASE_ARCHITECTURE_ID
from rag_workbench.experiments.v2_document_diversity import PHASE1_BENCHMARK_HEADING
from rag_workbench.experiments.v2_final_benchmark import (
    CASES,
    DATASET_HASH,
    DATASET_ID,
    DATASET_PATH,
    EXPECTED_DISTRIBUTION,
    HIDDEN_GROUND_TRUTH_FIELDS,
    V2_ARCHITECTURE_ID,
    V2FinalBenchmark,
    dataset_overlap_report,
    v2_architecture_configuration,
)
from rag_workbench.experiments.v2_final_benchmark_report import V2_FINAL_BENCHMARK_HEADING
from rag_workbench.experiments.v2_quality_ab_report import QUALITY_AB_BENCHMARK_HEADING
from rag_workbench.experiments.v2_quality_recovery import (
    FROZEN_V1_ARCHITECTURE_HASH,
    V1_BENCHMARK_HEADING,
    V2_BENCHMARK_HEADING,
)
from rag_workbench.experiments.v2_reliability import PHASE4_BENCHMARK_HEADING
from rag_workbench.experiments.v2_soft_document_cap import PHASE2_BENCHMARK_HEADING
from rag_workbench.experiments.v2_sufficiency_fn import (
    CONTROL_JUDGE,
    CONTROL_MODE,
    FROZEN_V1_TEMPLATE_HASH,
    PHASE3_BENCHMARK_HEADING,
    SCHEMA_IDENTITY,
    SOL_MODEL,
)
from rag_workbench.providers.llm.extractive import EXTRACTIVE_REVISION
from rag_workbench.reranking.document_diversity import EVALUATION_LABEL_FIELDS


def test_dataset_identity_distribution_and_overlap_guard() -> None:
    assert DATASET_ID == "acmeai-enterprise-rag-v2-final-eval"
    assert hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() == DATASET_HASH
    assert len(CASES) == 100
    assert len({item.question for item in CASES}) == 100
    assert dict(sorted(Counter(item.category for item in CASES).items())) == EXPECTED_DISTRIBUTION
    report = dataset_overlap_report()
    assert report["pass"] is True
    assert report["maximum_normalized_overlap"] < 0.5


def test_three_document_cases_require_three_sources() -> None:
    three = [item for item in CASES if item.category == "multidoc_three"]
    assert len(three) == 30
    for item in three:
        assert len(set(item.required_document_ids)) == 3
        assert len(item.required_fact_ids) == 3


def test_exact_identifier_and_injection_cases() -> None:
    exact = [item for item in CASES if item.category == "exact_identifier"]
    assert len(exact) == 10
    assert all(item.expected_facts for item in exact)
    injection = [item for item in CASES if item.category == "prompt_injection"]
    assert len(injection) == 4
    assert all("prompt_injection" in item.security_checks for item in injection)


def test_hidden_evaluator_labels_are_not_runtime_fields() -> None:
    from rag_workbench.experiments import v2_final_benchmark as module

    assert "preferred_source_id" in HIDDEN_GROUND_TRUTH_FIELDS
    assert "preferred_source_id" in EVALUATION_LABEL_FIELDS
    assert "_gate_evidence(top5)" in inspect.getsource(V2FinalBenchmark._run_generation)
    assert "preferred_source_id" not in inspect.getsource(module._gate_evidence)


def test_selected_ranking_and_judge_remain_frozen() -> None:
    configuration = v2_architecture_configuration()
    assert configuration["architecture_id"] == V2_ARCHITECTURE_ID
    assert configuration["parent_architecture_id"] == RELEASE_ARCHITECTURE_ID
    assert configuration["selected_retriever"] == CONTROL_MODE
    assert CONTROL_MODE == "POINTWISE_CROSS_ENCODER_TOP5"
    assert configuration["judge"]["id"] == CONTROL_JUDGE
    assert CONTROL_JUDGE == "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1"
    assert configuration["judge"]["model"] == SOL_MODEL
    assert configuration["judge"]["prompt"] == EVIDENCE_GATE_PROMPT_VERSION
    assert configuration["judge"]["prompt_hash"] == FROZEN_V1_TEMPLATE_HASH
    assert configuration["judge"]["schema_identity"] == SCHEMA_IDENTITY
    assert (
        evidence_sufficiency_template_hash(EVIDENCE_GATE_PROMPT_VERSION)
        == FROZEN_V1_TEMPLATE_HASH
    )
    assert evidence_sufficiency_schema_identity() == SCHEMA_IDENTITY
    settings = hosted_judge_request_settings(EVIDENCE_GATE_PROMPT_VERSION)
    assert settings["temperature"] == 0
    assert settings["max_completion_tokens"] == 160
    assert configuration["generator"] == EXTRACTIVE_REVISION
    assert EXTRACTIVE_REVISION["id"] == "deterministic-extractive-v1.1"
    assert EXTRACTIVE_REVISION["parent"] == "deterministic-extractive-v1"
    assert EXTRACTIVE_REVISION["semantic_policy_changed"] is False
    assert configuration["retry_policy"]["max_total_attempts"] == 2
    assert configuration["retry_policy"]["quality_outcomes_retried"] is False


def test_v1_identities_remain_in_benchmark_md() -> None:
    text = Path("BENCHMARK.md").read_text()
    assert text.count(V1_BENCHMARK_HEADING) == 1
    assert FROZEN_V1_ARCHITECTURE_HASH in text
    assert "enterprise-rag-workbench-v1" in text
    assert "HYBRID_CROSS_ENCODER_RERANK" in text
    assert text.count(V2_BENCHMARK_HEADING) == 1
    assert text.count(PHASE1_BENCHMARK_HEADING) == 1
    assert text.count(PHASE2_BENCHMARK_HEADING) == 1
    assert text.count(PHASE3_BENCHMARK_HEADING) == 1
    assert text.count(PHASE4_BENCHMARK_HEADING) == 1
    assert text.count(V2_FINAL_BENCHMARK_HEADING) == 1
    assert text.count(QUALITY_AB_BENCHMARK_HEADING) == 1


def test_one_shot_and_judge_after_retrieval_lock() -> None:
    retrieve = inspect.getsource(V2FinalBenchmark.execute_retrieval)
    generate = inspect.getsource(V2FinalBenchmark.execute_generation)
    execute = inspect.getsource(V2FinalBenchmark.execute)
    assert "one_shot_locked_at" in retrieve
    assert "do not invoke the Judge until retrieval traces are frozen" in generate
    assert "execute_retrieval" in execute
    assert "execute_generation" in execute
    assert "A/B" not in execute


def test_freeze_architecture_requires_persisted_v1(db_session) -> None:
    benchmark = V2FinalBenchmark(db_session)
    try:
        benchmark.freeze_architecture()
    except ValueError as exc:
        assert "frozen v1" in str(exc) or "v2 research identity" in str(exc)
    else:
        raise AssertionError("expected freeze to fail without persisted v1")
