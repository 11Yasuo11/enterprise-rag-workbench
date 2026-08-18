"""Persist the frozen multi-document coverage benchmark and one-shot lock."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "multidoc_benchmark_locks",
        sa.Column("dataset_id", sa.String(100), primary_key=True),
        sa.Column("dataset_hash", sa.String(64), nullable=False),
        sa.Column("split_identity", sa.String(64), nullable=False, unique=True),
        sa.Column("split_seed", sa.Integer(), nullable=False),
        sa.Column("calibration_case_ids", sa.JSON(), nullable=False),
        sa.Column("holdout_case_ids", sa.JSON(), nullable=False),
        sa.Column("control_configuration_hash", sa.String(64), nullable=False),
        sa.Column("candidate_configuration_hash", sa.String(64), nullable=False),
        sa.Column("candidate_prompt_hash", sa.String(64), nullable=False),
        sa.Column(
            "frozen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("calibration_control_run_id", sa.String(36)),
        sa.Column("calibration_candidate_run_id", sa.String(36)),
        sa.Column("selected_candidate", sa.String(2)),
        sa.Column("selected_configuration_hash", sa.String(64)),
        sa.Column("calibration_metrics", sa.JSON()),
        sa.Column("selection_reason", sa.Text()),
        sa.Column("locked_at", sa.DateTime(timezone=True)),
        sa.Column("holdout_started_at", sa.DateTime(timezone=True)),
        sa.Column("holdout_completed_at", sa.DateTime(timezone=True)),
        sa.Column("holdout_control_run_id", sa.String(36)),
        sa.Column("holdout_selected_run_id", sa.String(36)),
    )


def downgrade() -> None:
    op.drop_table("multidoc_benchmark_locks")
