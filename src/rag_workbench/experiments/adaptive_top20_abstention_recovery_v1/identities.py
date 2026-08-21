"""Frozen identities for ADAPTIVE_TOP20_ABSTENTION_RECOVERY_V1."""

from __future__ import annotations

import hashlib
from pathlib import Path

EXPERIMENT_ID = "ADAPTIVE_TOP20_ABSTENTION_RECOVERY_V1"
PIPELINE_VERSION = "adaptive-top20-abstention-recovery-v1.0.0"
LUNA_MODEL = "gpt-5.6-luna"
SOL_MODEL = "gpt-5.6-sol"
LUNA_PROMPT_VERSION = "top20-requirement-search-verify-v1"
LUNA_SCHEMA_NAME = "top20_requirement_search_verify_v1"
GENERATOR_MODEL = "deterministic-extractive-v2"
TOP_K_NORMAL = 5
TOP_K_RECOVERY = 20
SWEEP_K = (5, 6, 8, 10, 15, 20)
INDEX_IDENTITY = "e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466"
DATASET_PATH = Path("data/eval/phase5k/acmeai_enterprise_rag_v3_final_unseen_e2e_120.jsonl")
DATASET_HASH = "12851a9915ad51eccf2629e6765a8dc9a1731eb308987b0e53ac56468274f404"
PHASE5KR_DIR = Path("data/experiments/v3-phase5k-final-e2e")
PHASE5KR_ROWS = PHASE5KR_DIR / "phase5kr_corrected_per_case_results.jsonl"
PHASE5KR_CENSUS = PHASE5KR_DIR / "phase5kr_corrected_failure_census.json"
OUT_DIR = Path("data/experiments/adaptive-top20-abstention-recovery-v1")
TRACE_PATH = OUT_DIR / "ranked_traces.jsonl"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
