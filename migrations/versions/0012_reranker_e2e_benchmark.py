"""Persist the sealed paired reranker end-to-end benchmark.

Revision ID: 0012
Revises: 0011
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "reranker_e2e_benchmark_locks",
        sa.Column("dataset_id", sa.String(length=100), primary_key=True),
        sa.Column("dataset_hash", sa.String(length=64), nullable=False),
        sa.Column("case_ids", sa.JSON(), nullable=False),
        sa.Column("category_distribution", sa.JSON(), nullable=False),
        sa.Column("generation_method", sa.String(length=100), nullable=False),
        sa.Column("maximum_prior_overlap", sa.Float(), nullable=False),
        sa.Column("semantic_index_identity", sa.String(length=64), nullable=False),
        sa.Column("corpus_identity", sa.String(length=64), nullable=False),
        sa.Column("reranker_revision", sa.String(length=100), nullable=False),
        sa.Column("pipeline_a_configuration", sa.JSON(), nullable=False),
        sa.Column("pipeline_b_configuration", sa.JSON(), nullable=False),
        sa.Column("embedding_preflight", sa.JSON()),
        sa.Column("judge_preflight", sa.JSON()),
        sa.Column("prepared_cases", sa.JSON()),
        sa.Column("preparation_usage", sa.JSON()),
        sa.Column("paired_transitions", sa.JSON()),
        sa.Column("conversion_analysis", sa.JSON()),
        sa.Column("regression_analysis", sa.JSON()),
        sa.Column("production_retriever_status", sa.String(length=40)),
        sa.Column(
            "frozen_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("preparation_started_at", sa.DateTime(timezone=True)),
        sa.Column("prepared_at", sa.DateTime(timezone=True)),
        sa.Column("execution_started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "reranker_e2e_benchmark_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "dataset_id",
            sa.String(length=100),
            sa.ForeignKey("reranker_e2e_benchmark_locks.dataset_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("mode", sa.String(length=40), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("retrieval_metrics", sa.JSON(), nullable=False),
        sa.Column("category_metrics", sa.JSON(), nullable=False),
        sa.Column("case_results", sa.JSON(), nullable=False),
        sa.Column("latency", sa.JSON(), nullable=False),
        sa.Column("usage", sa.JSON(), nullable=False),
        sa.Column(
            "completed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("dataset_id", "mode"),
    )
    op.create_index(
        "ix_reranker_e2e_benchmark_runs_dataset_id",
        "reranker_e2e_benchmark_runs",
        ["dataset_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_reranker_e2e_benchmark_runs_dataset_id",
        table_name="reranker_e2e_benchmark_runs",
    )
    op.drop_table("reranker_e2e_benchmark_runs")
    op.drop_table("reranker_e2e_benchmark_locks")
