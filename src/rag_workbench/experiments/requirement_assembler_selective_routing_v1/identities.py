"""Frozen identities for REQUIREMENT_ASSEMBLER_SELECTIVE_ROUTING_V1."""

from pathlib import Path

EXPERIMENT_ID = "REQUIREMENT_ASSEMBLER_SELECTIVE_ROUTING_V1"
PIPELINE_VERSION = "requirement-assembler-selective-routing-v1.0.0"
ASSEMBLER_ID = "deterministic-requirement-assembler-v3"
RECOVERY_TOP_K = 15
LUNA_MODEL = "gpt-5.6-luna"
SOL_MODEL = "gpt-5.6-sol"
OUT_DIR = Path("data/experiments/requirement-assembler-selective-routing-v1")
ADAPTIVE_DIR = Path("data/experiments/adaptive-top20-abstention-recovery-v1")
LUNA_FIRST_DIR = Path("data/experiments/safe-recovery-luna-v2")
PHASE5KR_DIR = Path("data/experiments/v3-phase5k-final-e2e")
DATASET_PATH = Path("data/eval/phase5k/acmeai_enterprise_rag_v3_final_unseen_e2e_120.jsonl")
