"""Persist the frozen V2 final architecture and one-shot benchmark lock.

Revision ID: 0020
Revises: 0019
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "research_architecture_locks",
        sa.Column("final_v2_architecture_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "research_architecture_locks",
        sa.Column("final_v2_dataset_id", sa.String(length=100), nullable=True),
    )
    op.create_table(
        "v2_final_benchmark_locks",
        sa.Column("dataset_id", sa.String(length=100), primary_key=True),
        sa.Column("architecture_id", sa.String(length=100), nullable=False),
        sa.Column("parent_architecture_id", sa.String(length=100), nullable=False),
        sa.Column("architecture_hash", sa.String(length=64), nullable=False),
        sa.Column("architecture_configuration", sa.JSON(), nullable=False),
        sa.Column("dataset_hash", sa.String(length=64), nullable=False),
        sa.Column("case_ids", sa.JSON(), nullable=False),
        sa.Column("category_distribution", sa.JSON(), nullable=False),
        sa.Column("generation_method", sa.String(length=100), nullable=False),
        sa.Column("maximum_prior_overlap", sa.Float(), nullable=False),
        sa.Column("closest_previous_case", sa.JSON(), nullable=True),
        sa.Column("overlap_report", sa.JSON(), nullable=False),
        sa.Column("semantic_index_identity", sa.String(length=64), nullable=False),
        sa.Column("corpus_identity", sa.String(length=64), nullable=False),
        sa.Column("one_shot", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("dataset_frozen", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("embedding_preflight", sa.JSON(), nullable=True),
        sa.Column("judge_preflight", sa.JSON(), nullable=True),
        sa.Column("retrieval_traces", sa.JSON(), nullable=True),
        sa.Column("retrieval_results", sa.JSON(), nullable=True),
        sa.Column("case_results", sa.JSON(), nullable=True),
        sa.Column("end_to_end", sa.JSON(), nullable=True),
        sa.Column("category_results", sa.JSON(), nullable=True),
        sa.Column("stage_funnel", sa.JSON(), nullable=True),
        sa.Column("failure_taxonomy", sa.JSON(), nullable=True),
        sa.Column("generator_reliability", sa.JSON(), nullable=True),
        sa.Column("provider_reliability", sa.JSON(), nullable=True),
        sa.Column("security", sa.JSON(), nullable=True),
        sa.Column("citations", sa.JSON(), nullable=True),
        sa.Column("latency", sa.JSON(), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("cost", sa.JSON(), nullable=True),
        sa.Column("v1_comparison", sa.JSON(), nullable=True),
        sa.Column("primary_remaining_bottleneck", sa.String(length=80), nullable=True),
        sa.Column(
            "dataset_frozen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("architecture_frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("one_shot_locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retrieval_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retrieval_frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("execution_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_v2_final_benchmark_locks_architecture_id",
        "v2_final_benchmark_locks",
        ["architecture_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_v2_final_benchmark_locks_architecture_id",
        table_name="v2_final_benchmark_locks",
    )
    op.drop_table("v2_final_benchmark_locks")
    op.drop_column("research_architecture_locks", "final_v2_dataset_id")
    op.drop_column("research_architecture_locks", "final_v2_architecture_id")
