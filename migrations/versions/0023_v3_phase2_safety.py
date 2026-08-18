"""Persist V3 Phase-2 safety-research tables. Additive only; frozen v1/v2 rows are untouched.

Revision ID: 0023
Revises: 0022
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0023"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "research_architecture_locks",
        sa.Column("v3_phase2_dataset_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "research_architecture_locks",
        sa.Column("v3_phase2_selected_candidate", sa.String(length=80), nullable=True),
    )
    op.create_table(
        "v3_phase2_experiment_locks",
        sa.Column("lock_id", sa.String(length=100), primary_key=True),
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
        sa.Column("experiment_plan", sa.JSON(), nullable=False),
        sa.Column("control_configuration", sa.JSON(), nullable=False),
        sa.Column("candidate_configurations", sa.JSON(), nullable=False),
        sa.Column("ledger", sa.JSON(), nullable=True),
        sa.Column("selected_candidate", sa.String(length=80), nullable=True),
        sa.Column("selected_configuration", sa.JSON(), nullable=True),
        sa.Column("validation_results", sa.JSON(), nullable=True),
        sa.Column("development_results", sa.JSON(), nullable=True),
        sa.Column("shared_traces", sa.JSON(), nullable=True),
        sa.Column("embedding_preflight", sa.JSON(), nullable=True),
        sa.Column("hosted_preflight", sa.JSON(), nullable=True),
        sa.Column("cumulative_budget", sa.JSON(), nullable=True),
        sa.Column("security", sa.JSON(), nullable=True),
        sa.Column("freeze_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "selection_policy_frozen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("dataset_frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retrieval_frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stop_reason", sa.String(length=80), nullable=True),
    )
    op.create_index(
        "ix_v3_phase2_experiment_locks_architecture_id",
        "v3_phase2_experiment_locks",
        ["architecture_id"],
    )
    op.create_table(
        "v3_phase2_experiment_ledger",
        sa.Column("experiment_id", sa.String(length=80), primary_key=True),
        sa.Column("parent_candidate", sa.String(length=80), nullable=True),
        sa.Column("hypothesis", sa.Text(), nullable=False),
        sa.Column("independent_variable", sa.Text(), nullable=False),
        sa.Column("configuration_hash", sa.String(length=64), nullable=False),
        sa.Column("dataset_hash", sa.String(length=64), nullable=True),
        sa.Column("prompt_hash", sa.String(length=64), nullable=True),
        sa.Column("schema_hash", sa.String(length=64), nullable=True),
        sa.Column("external_calls", sa.JSON(), nullable=True),
        sa.Column("latency", sa.JSON(), nullable=True),
        sa.Column("cost", sa.JSON(), nullable=True),
        sa.Column("quality_metrics", sa.JSON(), nullable=True),
        sa.Column("safety_metrics", sa.JSON(), nullable=True),
        sa.Column("failure_census", sa.JSON(), nullable=True),
        sa.Column("verdict", sa.String(length=40), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("v3_phase2_experiment_ledger")
    op.drop_index(
        "ix_v3_phase2_experiment_locks_architecture_id",
        table_name="v3_phase2_experiment_locks",
    )
    op.drop_table("v3_phase2_experiment_locks")
    op.drop_column("research_architecture_locks", "v3_phase2_selected_candidate")
    op.drop_column("research_architecture_locks", "v3_phase2_dataset_id")
