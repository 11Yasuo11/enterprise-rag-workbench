"""Persist the sealed Dense cross-encoder reranking benchmark."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "reranking_benchmark_locks",
        sa.Column("dataset_id", sa.String(100), primary_key=True),
        sa.Column("dataset_hash", sa.String(64), nullable=False),
        sa.Column("split_identity", sa.String(64), nullable=False, unique=True),
        sa.Column("split_seed", sa.Integer(), nullable=False),
        sa.Column("calibration_case_ids", sa.JSON(), nullable=False),
        sa.Column("holdout_case_ids", sa.JSON(), nullable=False),
        sa.Column("semantic_index_identity", sa.String(64), nullable=False),
        sa.Column("corpus_identity", sa.String(64), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("selection_policy", sa.JSON(), nullable=False),
        sa.Column("model_revision", sa.String(100)),
        sa.Column("model_metadata", sa.JSON()),
        sa.Column("selected_mode", sa.String(40)),
        sa.Column("calibration_metrics", sa.JSON()),
        sa.Column("selection_reason", sa.Text()),
        sa.Column(
            "frozen_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("locked_at", sa.DateTime(timezone=True)),
        sa.Column("holdout_started_at", sa.DateTime(timezone=True)),
        sa.Column("holdout_completed_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "reranking_benchmark_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "dataset_id",
            sa.String(100),
            sa.ForeignKey("reranking_benchmark_locks.dataset_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("partition", sa.String(20), nullable=False),
        sa.Column("mode", sa.String(40), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("category_metrics", sa.JSON(), nullable=False),
        sa.Column("candidate_pool_metrics", sa.JSON(), nullable=False),
        sa.Column("rerankable_subset_metrics", sa.JSON(), nullable=False),
        sa.Column("movement_metrics", sa.JSON(), nullable=False),
        sa.Column("case_results", sa.JSON(), nullable=False),
        sa.Column("latency", sa.JSON(), nullable=False),
        sa.Column("usage", sa.JSON(), nullable=False),
        sa.Column(
            "completed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("dataset_id", "partition", "mode"),
    )
    op.create_index(
        "ix_reranking_benchmark_runs_dataset_id", "reranking_benchmark_runs", ["dataset_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_reranking_benchmark_runs_dataset_id", table_name="reranking_benchmark_runs")
    op.drop_table("reranking_benchmark_runs")
    op.drop_table("reranking_benchmark_locks")
