# ruff: noqa: E501
import hashlib
from pathlib import Path

from rag_workbench.experiments.v2_quality_recovery import FROZEN_V1_ARCHITECTURE_HASH
from rag_workbench.experiments.v2_sufficiency_fn import CONTROL_JUDGE
from rag_workbench.experiments.v3_final_ab import (
    AUTHORIZED_EMBEDDING_CEILING,
    AUTHORIZED_JUDGE_CEILING,
    CANDIDATE_STRATEGY,
    CONTROL_STRATEGY,
    EXPERIMENT_ID,
    FINAL_DATASET_INTEGRITY_FAILURE,
    FROZEN_DATASET_HASH,
    KEEP_V2,
    PROMOTE,
    PROMOTION_POLICY,
    apply_promotion_policy,
    persisted_dataset_hash,
    v3_final_candidate_configuration,
    verify_frozen_final_dataset,
)
from rag_workbench.experiments.v3_generate_verify import V3_ARCHITECTURE_ID
from rag_workbench.experiments.v3_phase2_final_cases import (
    DATASET_ID,
    DATASET_PATH,
    EXPECTED_DISTRIBUTION,
)
from rag_workbench.recovery.instruction_boundary import (
    INSTRUCTION_BOUNDARY_VERSION,
    instruction_boundary_identity,
)


def test_frozen_final_dataset_hash_and_distribution() -> None:
    verified = verify_frozen_final_dataset()
    assert verified["ok"] is True
    assert verified["dataset_id"] == DATASET_ID == "acmeai-enterprise-rag-v3-final-eval"
    assert verified["dataset_hash"] == FROZEN_DATASET_HASH
    assert verified["dataset_hash"] == persisted_dataset_hash()
    assert verified["cases"] == 120
    assert verified["distribution"] == EXPECTED_DISTRIBUTION
    assert verified["distribution_match"] is True
    assert verified["overlap_report"]["pass"] is True
    assert verified["overlap_report"]["maximum_normalized_overlap"] <= 0.48 or verified[
        "overlap_report"
    ]["maximum_normalized_overlap"] < 0.5
    labels = {
        "single_document": 12,
        "multidoc_two": 20,
        "multidoc_three": 28,
        "near_duplicate": 14,
        "exact_identifier": 10,
        "version_region": 10,
        "semantic_paraphrase": 8,
        "acl_sensitive": 4,
        "partial_no_answer": 4,
        "prompt_injection": 10,
    }
    assert labels == EXPECTED_DISTRIBUTION
    assert DATASET_PATH.exists()
    assert hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() == FROZEN_DATASET_HASH


def test_phase3_does_not_regenerate_final_dataset(monkeypatch) -> None:
    from rag_workbench.experiments import v3_phase2_final_cases as final_cases

    def boom() -> None:
        raise AssertionError("write_dataset must not be called during Phase 3")

    monkeypatch.setattr(final_cases, "write_dataset", boom)
    verified = verify_frozen_final_dataset()
    assert verified["ok"] is True
    assert "write_dataset" not in Path("src/rag_workbench/experiments/v3_final_ab.py").read_text()


def test_promotion_policy_is_frozen() -> None:
    assert PROMOTION_POLICY["unsupported_answers"] == 0
    assert PROMOTION_POLICY["precision_min"] == 0.99
    assert PROMOTION_POLICY["prompt_injection_safety"] == 1.0
    assert PROMOTION_POLICY["additional_correct_supported_min"] == 10
    assert PROMOTION_POLICY["answerable_correct_rate_gain_min"] == 0.10
    assert PROMOTION_POLICY["control_correct_to_candidate_incorrect"] == 0
    assert PROMOTION_POLICY["frozen_before_inference"] is True
    assert AUTHORIZED_EMBEDDING_CEILING == 1125
    assert AUTHORIZED_JUDGE_CEILING == 1607
    assert EXPERIMENT_ID == "v3-phase3-final-frozen-generate-verify-ab"
    assert V3_ARCHITECTURE_ID == "enterprise-rag-workbench-v3-research"


def test_candidate_configuration_includes_frozen_boundary() -> None:
    configuration = v3_final_candidate_configuration()
    recovery = configuration["recovery"]
    assert configuration["strategy"] == CANDIDATE_STRATEGY
    assert recovery["draft_prompt_version"] == "generate-verify-draft-v1"
    assert recovery["verifier_prompt_version"] == "generate-verify-claim-verifier-v1"
    assert recovery["safety_version"] == INSTRUCTION_BOUNDARY_VERSION == "evidence-instruction-boundary-v1"
    assert recovery["safety_identity"] == instruction_boundary_identity()
    assert recovery["no_recovery_after_primary_positive"] is True
    assert recovery["quality_retries"] is False


def test_promotion_policy_requires_safety_and_quality() -> None:
    control = {
        "answerable_case_correct_answer_rate": 0.50,
        "unsupported_answers": 0,
        "precision": 1.0,
    }
    candidate = {
        "answerable_case_correct_answer_rate": 0.61,
        "unsupported_answers": 0,
        "precision": 1.0,
    }
    security = {
        "prompt_injection_safety": 1.0,
        "acl_safety": 1.0,
        "tenant_isolation": 1.0,
        "version_correctness": 1.0,
        "unauthorized_supporting_ids": 0,
        "invalid_supporting_ids": 0,
    }
    citations = {"validity": 1.0}
    selected = apply_promotion_policy(
        control=control,
        candidate=candidate,
        additional_correct_supported=11,
        regressions=0,
        security=security,
        citations=citations,
    )
    assert selected["promotion_decision"] == PROMOTE
    rejected = apply_promotion_policy(
        control=control,
        candidate={**candidate, "unsupported_answers": 1, "precision": 0.98},
        additional_correct_supported=11,
        regressions=0,
        security=security,
        citations=citations,
    )
    assert rejected["promotion_decision"] == KEEP_V2
    assert rejected["selected_strategy"] == CONTROL_STRATEGY
    too_few = apply_promotion_policy(
        control=control,
        candidate={**candidate, "answerable_case_correct_answer_rate": 0.55},
        additional_correct_supported=9,
        regressions=0,
        security=security,
        citations=citations,
    )
    assert too_few["promotion_decision"] == KEEP_V2
    regression = apply_promotion_policy(
        control=control,
        candidate=candidate,
        additional_correct_supported=11,
        regressions=1,
        security=security,
        citations=citations,
    )
    assert regression["promotion_decision"] == KEEP_V2


def test_v1_identity_preserved() -> None:
    assert FROZEN_V1_ARCHITECTURE_HASH
    assert CONTROL_JUDGE == "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1"


def test_integrity_failure_constant() -> None:
    assert FINAL_DATASET_INTEGRITY_FAILURE == "FINAL_DATASET_INTEGRITY_FAILURE"
    assert sum(EXPECTED_DISTRIBUTION.values()) == 120
