"""Run the frozen V12 candidate against fresh_v2_032 only."""

from pathlib import Path

from rag_workbench.evaluation.stage_guard import StageGuardConfig
from scripts import run_rag_release_pipeline_v4_evaluation as runner

ROOT = Path(__file__).resolve().parents[1]
V12 = ROOT / "data/experiments/rag-release-pipeline-v12"

runner.BASE = V12
runner.OUT = V12
runner.LOCAL = V12
runner.QUESTIONS = (
    ROOT / "data/experiments/rag-release-pipeline-v6/fresh-holdout-v2/fresh_v2_questions.json"
)
runner.GOLD = ROOT / "data/experiments/rag-release-pipeline-v6/fresh-holdout-v2/fresh_v2_gold.json"
runner.PLAN_BASELINE = V12 / "fresh_v2_032_question_plan_baseline.jsonl"
runner.EXPERIMENT_ID = "TARGETED_FRESH_V2_032_V12"
runner.RC_ID = "RC_FINAL_PRE_API_V12"
runner.FREEZE_MANIFEST_NAME = "rc_final_pre_api_v12_freeze_manifest.json"
runner.ACTIVE_STAGE = "TARGETED_ONE"
runner.ARTIFACT_PREFIX = "fresh_v2_032"
runner.TARGETS = ("fresh_v2_032",)
runner.ACTIVE_STAGE_CONFIG = StageGuardConfig("TARGETED_ONE", 1, 1, 1, 0.03)


if __name__ == "__main__":
    runner.main()
