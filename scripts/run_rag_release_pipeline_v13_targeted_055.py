"""Run the frozen V13 candidate against fresh_v2_055 only."""

from pathlib import Path

from rag_workbench.evaluation.stage_guard import StageGuardConfig
from scripts import run_rag_release_pipeline_v4_evaluation as runner

ROOT = Path(__file__).resolve().parents[1]
V13 = ROOT / "data/experiments/rag-release-pipeline-v13"

runner.BASE = V13
runner.OUT = V13
runner.LOCAL = V13
runner.QUESTIONS = (
    ROOT / "data/experiments/rag-release-pipeline-v6/fresh-holdout-v2/fresh_v2_questions.json"
)
runner.GOLD = ROOT / "data/experiments/rag-release-pipeline-v6/fresh-holdout-v2/fresh_v2_gold.json"
runner.PLAN_BASELINE = V13 / "fresh_v2_055_question_plan_baseline.jsonl"
runner.EXPERIMENT_ID = "TARGETED_FRESH_V2_055_V13"
runner.RC_ID = "RC_FINAL_PRE_API_V13"
runner.FREEZE_MANIFEST_NAME = "rc_final_pre_api_v13_freeze_manifest.json"
runner.ACTIVE_STAGE = "TARGETED_ONE"
runner.ARTIFACT_PREFIX = "fresh_v2_055"
runner.TARGETS = ("fresh_v2_055",)
runner.ACTIVE_STAGE_CONFIG = StageGuardConfig("TARGETED_ONE", 1, 1, 1, 0.02)


if __name__ == "__main__":
    runner.main()
