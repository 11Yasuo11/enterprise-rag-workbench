"""Run the frozen V10 candidate against the three V9 V2 failures."""

from pathlib import Path

from rag_workbench.evaluation.stage_guard import StageGuardConfig
from scripts import run_rag_release_pipeline_v4_evaluation as runner

ROOT = Path(__file__).resolve().parents[1]
V10 = ROOT / "data/experiments/rag-release-pipeline-v10"

runner.BASE = V10
runner.OUT = V10
runner.LOCAL = V10
runner.QUESTIONS = (
    ROOT / "data/experiments/rag-release-pipeline-v6/fresh-holdout-v2/fresh_v2_questions.json"
)
runner.GOLD = ROOT / "data/experiments/rag-release-pipeline-v6/fresh-holdout-v2/fresh_v2_gold.json"
runner.PLAN_BASELINE = V10 / "targeted_three_question_plan_baseline.jsonl"
runner.EXPERIMENT_ID = "TARGETED_THREE_V10"
runner.ACTIVE_STAGE = "TARGETED_THREE"
runner.ARTIFACT_PREFIX = "targeted_three"
runner.TARGETS = ("fresh_v2_010", "fresh_v2_021", "fresh_v2_023")
runner.ACTIVE_STAGE_CONFIG = StageGuardConfig("TARGETED_THREE", 3, 1, 1, 0.08)


if __name__ == "__main__":
    runner.main()
