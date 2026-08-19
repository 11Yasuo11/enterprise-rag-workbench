"""Persist V3 Phase 5B ranking E2E experiment tables. Additive only.

Revision ID: 0025
Revises: 0024
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0025"
down_revision: str | None = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "v3_phase5b_experiment_locks",
        sa.Column("lock_id", sa.String(length=100), primary_key=True),
        sa.Column("experiment_id", sa.String(length=120), nullable=False),
        sa.Column("architecture_id", sa.String(length=100), nullable=False),
        sa.Column("parent_architecture_id", sa.String(length=100), nullable=False),
        sa.Column("production_status", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("dataset_id", sa.String(length=100), nullable=True),
        sa.Column("dataset_hash", sa.String(length=64), nullable=True),
        sa.Column("case_ids", sa.JSON(), nullable=True),
        sa.Column("category_distribution", sa.JSON(), nullable=True),
        sa.Column("generation_method", sa.String(length=100), nullable=True),
        sa.Column("maximum_prior_overlap", sa.Float(), nullable=True),
        sa.Column("closest_previous_case", sa.JSON(), nullable=True),
        sa.Column("overlap_report", sa.JSON(), nullable=True),
        sa.Column("selection_policy", sa.JSON(), nullable=False),
        sa.Column("control_configuration", sa.JSON(), nullable=False),
        sa.Column("candidate_configuration", sa.JSON(), nullable=False),
        sa.Column("control_architecture_hash", sa.String(length=64), nullable=True),
        sa.Column("candidate_architecture_hash", sa.String(length=64), nullable=True),
        sa.Column("promotion_policy_hash", sa.String(length=64), nullable=True),
        sa.Column("semantic_index_identity", sa.String(length=64), nullable=False),
        sa.Column("corpus_identity", sa.String(length=64), nullable=False),
        sa.Column("embedding_preflight", sa.JSON(), nullable=True),
        sa.Column("hosted_preflight", sa.JSON(), nullable=True),
        sa.Column("shared_traces", sa.JSON(), nullable=True),
        sa.Column("control_rows", sa.JSON(), nullable=True),
        sa.Column("candidate_rows", sa.JSON(), nullable=True),
        sa.Column("control_metrics", sa.JSON(), nullable=True),
        sa.Column("candidate_metrics", sa.JSON(), nullable=True),
        sa.Column("paired_deltas", sa.JSON(), nullable=True),
        sa.Column("recovery_funnel", sa.JSON(), nullable=True),
        sa.Column("instruction_boundary", sa.JSON(), nullable=True),
        sa.Column("prompt_injection", sa.JSON(), nullable=True),
        sa.Column("category_results", sa.JSON(), nullable=True),
        sa.Column("retrieval_metrics", sa.JSON(), nullable=True),
        sa.Column("failure_census", sa.JSON(), nullable=True),
        sa.Column("security", sa.JSON(), nullable=True),
        sa.Column("citations", sa.JSON(), nullable=True),
        sa.Column("reliability", sa.JSON(), nullable=True),
        sa.Column("latency", sa.JSON(), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("cost", sa.JSON(), nullable=True),
        sa.Column("selection", sa.JSON(), nullable=True),
        sa.Column("selected_strategy", sa.String(length=120), nullable=True),
        sa.Column("promotion_decision", sa.String(length=80), nullable=True),
        sa.Column("v3_status", sa.String(length=40), nullable=True),
        sa.Column("primary_remaining_bottleneck", sa.String(length=80), nullable=True),
        # Phase 5B ranking-specific columns
        sa.Column("reference_configuration", sa.JSON(), nullable=True),
        sa.Column("reference_rows", sa.JSON(), nullable=True),
        sa.Column("reference_metrics", sa.JSON(), nullable=True),
        sa.Column("candidate_b_rows", sa.JSON(), nullable=True),
        sa.Column("candidate_b_metrics", sa.JSON(), nullable=True),
        sa.Column("ranking_metrics", sa.JSON(), nullable=True),
        sa.Column("ranking_comparison", sa.JSON(), nullable=True),
        sa.Column("candidate_ranking_hash", sa.String(length=64), nullable=True),
        sa.Column("dataset_frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "selection_policy_frozen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("retrieval_frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("execution_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_v3_phase5b_experiment_locks_architecture_id",
        "v3_phase5b_experiment_locks",
        ["architecture_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_v3_phase5b_experiment_locks_architecture_id",
        table_name="v3_phase5b_experiment_locks",
    )
    op.drop_table("v3_phase5b_experiment_locks")
