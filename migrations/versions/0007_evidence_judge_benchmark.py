"""Add dual-judge accounting, operational errors, and persistent holdout lock."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "answerability_gate_cache",
        sa.Column("prompt_render_sha256", sa.String(64), nullable=False, server_default="legacy"),
    )
    for table in ("rag_runs", "experiment_case_results"):
        op.add_column(table, sa.Column("answerability_operational_error", sa.String(100)))
        op.add_column(table, sa.Column("local_judge_calls", sa.Integer()))
    op.add_column("experiment_runs", sa.Column("local_judge_calls", sa.Integer()))
    op.create_table(
        "evidence_benchmark_locks",
        sa.Column("split_identity", sa.String(64), primary_key=True),
        sa.Column("selected_candidate", sa.String(2), nullable=False),
        sa.Column("selected_configuration_hash", sa.String(64), nullable=False),
        sa.Column("judge_provider", sa.String(100)),
        sa.Column("judge_model", sa.String(200)),
        sa.Column("judge_version", sa.String(100)),
        sa.Column("prompt_version", sa.String(100)),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("supporting_context_only", sa.Boolean(), nullable=False),
        sa.Column("calibration_metrics", sa.JSON(), nullable=False),
        sa.Column("selection_reason", sa.Text(), nullable=False),
        sa.Column(
            "locked_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("holdout_started_at", sa.DateTime(timezone=True)),
        sa.Column("holdout_completed_at", sa.DateTime(timezone=True)),
        sa.Column("holdout_baseline_run_id", sa.String(36)),
        sa.Column("holdout_winner_run_id", sa.String(36)),
    )


def downgrade() -> None:
    op.drop_table("evidence_benchmark_locks")
    op.drop_column("experiment_runs", "local_judge_calls")
    for table in ("experiment_case_results", "rag_runs"):
        op.drop_column(table, "local_judge_calls")
        op.drop_column(table, "answerability_operational_error")
    op.drop_column("answerability_gate_cache", "prompt_render_sha256")
