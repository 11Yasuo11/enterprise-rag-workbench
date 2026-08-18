"""Persist the v2 quality A/B research lock. Additive only; v1 and frozen v2 rows are untouched.

Revision ID: 0021
Revises: 0020
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0021"
down_revision: str | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "v2_quality_ab_experiment_locks",
        sa.Column("lock_id", sa.String(length=100), primary_key=True),
        sa.Column("architecture_id", sa.String(length=100), nullable=False),
        sa.Column("parent_architecture_id", sa.String(length=100), nullable=False),
        sa.Column("git_commit", sa.String(length=64), nullable=True),
        sa.Column("baseline_config_hash", sa.String(length=64), nullable=False),
        sa.Column("dataset_hash", sa.String(length=64), nullable=False),
        sa.Column("corpus_hash", sa.String(length=64), nullable=False),
        sa.Column("selection_policy", sa.JSON(), nullable=False),
        sa.Column("baseline", sa.JSON(), nullable=True),
        sa.Column("experiments", sa.JSON(), nullable=True),
        sa.Column("failure_census", sa.JSON(), nullable=True),
        sa.Column("safety", sa.JSON(), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("verdicts", sa.JSON(), nullable=True),
        sa.Column("final_candidate", sa.JSON(), nullable=True),
        sa.Column("holdout", sa.JSON(), nullable=True),
        sa.Column("v1_preservation", sa.JSON(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_v2_quality_ab_experiment_locks_architecture_id",
        "v2_quality_ab_experiment_locks",
        ["architecture_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_v2_quality_ab_experiment_locks_architecture_id",
        table_name="v2_quality_ab_experiment_locks",
    )
    op.drop_table("v2_quality_ab_experiment_locks")
