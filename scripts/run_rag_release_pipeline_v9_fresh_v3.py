"""Run the frozen V9 candidate against the separately frozen Fresh V3 holdout."""

from pathlib import Path

from rag_workbench.evaluation.stage_guard import StageGuardConfig
from scripts import run_rag_release_pipeline_v4_evaluation as runner

ROOT = Path(__file__).resolve().parents[1]
V9 = ROOT / "data/experiments/rag-release-pipeline-v9"

runner.BASE = V9
runner.OUT = V9
runner.LOCAL = V9
runner.QUESTIONS = V9 / "fresh_v3_questions.json"
runner.GOLD = V9 / "fresh_v3_gold.json"
runner.PLAN_BASELINE = V9 / "fresh_v3_question_plan_baseline.jsonl"
runner.EXPERIMENT_ID = "FRESH_UNSEEN_HOLDOUT_V3"
runner.ACTIVE_STAGE = "FRESH_HOLDOUT_V3"
runner.ARTIFACT_PREFIX = "fresh_v3"
runner.TARGETS = tuple(f"fresh_v3_{case_id:03d}" for case_id in range(1, 61))
runner.ACTIVE_STAGE_CONFIG = StageGuardConfig("FRESH_HOLDOUT_V3", 60, 1, 1, 0.30)


if __name__ == "__main__":
    runner.main()
