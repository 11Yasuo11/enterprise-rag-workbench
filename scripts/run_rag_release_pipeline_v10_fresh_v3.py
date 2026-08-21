"""Run the frozen V10 candidate against the separately frozen Fresh V3 holdout."""

from pathlib import Path

from rag_workbench.evaluation.stage_guard import StageGuardConfig
from scripts import run_rag_release_pipeline_v4_evaluation as runner

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT / "data/experiments/rag-release-pipeline-v10"

runner.BASE = V10
runner.OUT = V10
runner.LOCAL = V10
runner.QUESTIONS = V10 / "fresh_v3_questions.json"
runner.GOLD = V10 / "fresh_v3_gold.json"
runner.PLAN_BASELINE = V10 / "fresh_v3_question_plan_baseline.jsonl"
runner.EXPERIMENT_ID = "FRESH_UNSEEN_HOLDOUT_V3"
runner.ACTIVE_STAGE = "FRESH_HOLDOUT_V3"
runner.ARTIFACT_PREFIX = "fresh_v3"
runner.TARGETS = tuple(f"fresh_v3_{case_id:03d}" for case_id in range(1, 61))
runner.ACTIVE_STAGE_CONFIG = StageGuardConfig("FRESH_HOLDOUT_V3", 60, 1, 1, 0.30)


if __name__ == "__main__":
    runner.main()
