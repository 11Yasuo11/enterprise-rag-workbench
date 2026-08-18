"""Persist the v3 research identity tables. Additive only; frozen v2 rows are untouched.

Revision ID: 0022
Revises: 0021
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "research_architecture_locks",
        sa.Column("selected_v3_strategy", sa.String(length=80), nullable=True),
    )
    op.add_column(
        "research_architecture_locks",
        sa.Column("v3_phase1_dataset_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "research_architecture_locks",
        sa.Column("v3_research_status", sa.String(length=40), nullable=True),
    )
    op.create_table(
        "recovery_stage_cache",
        sa.Column("cache_key", sa.String(length=64), primary_key=True),
        sa.Column("stage", sa.String(length=40), nullable=False),
        sa.Column("normalized_question", sa.Text(), nullable=False),
        sa.Column("ordered_top5", sa.JSON(), nullable=False),
        sa.Column("provider", sa.String(length=100), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("prompt_version", sa.String(length=100), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("schema_identity", sa.String(length=64), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("operational_error", sa.String(length=100), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_recovery_stage_cache_stage", "recovery_stage_cache", ["stage"])
    op.create_table(
        "v3_phase1_experiment_locks",
        sa.Column("lock_id", sa.String(length=100), primary_key=True),
        sa.Column("architecture_id", sa.String(length=100), nullable=False),
        sa.Column("parent_architecture_id", sa.String(length=100), nullable=False),
        sa.Column("production_status", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("diagnosis_only", sa.Boolean(), nullable=False, server_default=sa.true()),
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
        sa.Column("semantic_index_identity", sa.String(length=64), nullable=False),
        sa.Column("corpus_identity", sa.String(length=64), nullable=False),
        sa.Column("embedding_preflight", sa.JSON(), nullable=True),
        sa.Column("hosted_preflight", sa.JSON(), nullable=True),
        sa.Column("diagnostic", sa.JSON(), nullable=True),
        sa.Column("shared_traces", sa.JSON(), nullable=True),
        sa.Column("control_metrics", sa.JSON(), nullable=True),
        sa.Column("candidate_metrics", sa.JSON(), nullable=True),
        sa.Column("recovery_funnel", sa.JSON(), nullable=True),
        sa.Column("valid_rescues", sa.JSON(), nullable=True),
        sa.Column("false_positive_recoveries", sa.JSON(), nullable=True),
        sa.Column("completeness_failures", sa.JSON(), nullable=True),
        sa.Column("category_results", sa.JSON(), nullable=True),
        sa.Column("security", sa.JSON(), nullable=True),
        sa.Column("citations", sa.JSON(), nullable=True),
        sa.Column("latency", sa.JSON(), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("cost", sa.JSON(), nullable=True),
        sa.Column("selection", sa.JSON(), nullable=True),
        sa.Column("selected_strategy", sa.String(length=80), nullable=True),
        sa.Column("go_nogo", sa.String(length=40), nullable=True),
        sa.Column("primary_remaining_bottleneck", sa.String(length=80), nullable=True),
        sa.Column("dataset_frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "selection_policy_frozen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("diagnostic_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("diagnostic_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retrieval_frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("execution_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_v3_phase1_experiment_locks_architecture_id",
        "v3_phase1_experiment_locks",
        ["architecture_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_v3_phase1_experiment_locks_architecture_id",
        table_name="v3_phase1_experiment_locks",
    )
    op.drop_table("v3_phase1_experiment_locks")
    op.drop_index("ix_recovery_stage_cache_stage", table_name="recovery_stage_cache")
    op.drop_table("recovery_stage_cache")
    op.drop_column("research_architecture_locks", "v3_research_status")
    op.drop_column("research_architecture_locks", "v3_phase1_dataset_id")
    op.drop_column("research_architecture_locks", "selected_v3_strategy")
