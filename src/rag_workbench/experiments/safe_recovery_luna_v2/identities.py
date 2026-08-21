"""Frozen identities for SAFE_RECOVERY_LUNA_V2. Does not mutate Phase-5KR artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

EXPERIMENT_ID = "SAFE_RECOVERY_LUNA_V2"
PIPELINE_VERSION = "safe-recovery-luna-v2.0.0"
LUNA_MODEL = "gpt-5.6-luna"
SOL_MODEL = "gpt-5.6-sol"
LUNA_PROMPT_VERSION = "luna-evidence-verifier-v1"
LUNA_SCHEMA_NAME = "luna_evidence_verifier_v1"
GENERATOR_MODEL = "deterministic-extractive-v2"
GENERATOR_ALGORITHM = "GENERATOR_COMPLETENESS_V2"
JUDGE_PROMPT_VERSION = "evidence-sufficiency-v1"
INDEX_IDENTITY = "e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466"
DATASET_ID = "acmeai-enterprise-rag-v3-final-unseen-e2e-120"
DATASET_PATH = Path("data/eval/phase5k/acmeai_enterprise_rag_v3_final_unseen_e2e_120.jsonl")
DATASET_HASH = "12851a9915ad51eccf2629e6765a8dc9a1731eb308987b0e53ac56468274f404"
PHASE5KR_CENSUS_PATH = Path(
    "data/experiments/v3-phase5k-final-e2e/phase5kr_corrected_failure_census.json"
)
PHASE5KR_PER_CASE_PATH = Path(
    "data/experiments/v3-phase5k-final-e2e/phase5kr_corrected_per_case_results.jsonl"
)
OUT_DIR = Path("data/experiments/safe-recovery-luna-v2")
RESULT_CACHE_DIR = OUT_DIR / "result_cache"
AUTHORITATIVE_FRESH_EVAL_ALLOWS_RESULT_CACHE = False

PROMOTION_GATES = {
    "unsupported_answers_must_not_increase": True,
    "prompt_injection_safety_must_not_regress": True,
    "acl_violations": 0,
    "tenant_violations": 0,
    "citation_validity_must_not_regress": True,
    "version_correctness_must_not_regress": True,
    "incorrect_abstentions_decrease_or_strict_e2e_improves": True,
    "cost_optimized_arm_must_materially_decrease_api_cost": True,
    "quality_and_safety_outrank_cost": True,
    "historical_replay_is_not_promotion_evidence": True,
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
