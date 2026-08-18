"""Persist the v2 Phase 4 generation and provider reliability lock.

Revision ID: 0019
Revises: 0018
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "research_architecture_locks",
        sa.Column("phase4_lock_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "research_architecture_locks",
        sa.Column("reliability_research_status", sa.String(length=40), nullable=True),
    )
    op.create_table(
        "v2_phase4_experiment_locks",
        sa.Column("lock_id", sa.String(length=100), primary_key=True),
        sa.Column("architecture_id", sa.String(length=100), nullable=False),
        sa.Column("selected_v2_ranking", sa.String(length=40), nullable=False),
        sa.Column("selected_v2_judge", sa.String(length=80), nullable=False),
        sa.Column("ranking_research_status", sa.String(length=40), nullable=False),
        sa.Column("judge_research_status", sa.String(length=40), nullable=False),
        sa.Column("generator_parent", sa.String(length=80), nullable=False),
        sa.Column("generator_revision", sa.JSON(), nullable=False),
        sa.Column("fixture_hash", sa.String(length=64), nullable=False),
        sa.Column("historical_generator_failure", sa.JSON(), nullable=False),
        sa.Column("generator_root_cause", sa.String(length=80), nullable=False),
        sa.Column("provider_failure", sa.JSON(), nullable=False),
        sa.Column("transport_retry_policy", sa.JSON(), nullable=False),
        sa.Column("taxonomy", sa.JSON(), nullable=True),
        sa.Column("security", sa.JSON(), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("reliability_status", sa.String(length=40), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_v2_phase4_experiment_locks_architecture_id",
        "v2_phase4_experiment_locks",
        ["architecture_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_v2_phase4_experiment_locks_architecture_id",
        table_name="v2_phase4_experiment_locks",
    )
    op.drop_table("v2_phase4_experiment_locks")
    op.drop_column("research_architecture_locks", "reliability_research_status")
    op.drop_column("research_architecture_locks", "phase4_lock_id")
