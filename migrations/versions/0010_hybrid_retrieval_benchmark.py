"""Persist the sealed dense/BM25/hybrid retrieval benchmark."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "retrieval_benchmark_locks",
        sa.Column("dataset_id", sa.String(100), primary_key=True),
        sa.Column("dataset_hash", sa.String(64), nullable=False),
        sa.Column("split_identity", sa.String(64), nullable=False, unique=True),
        sa.Column("split_seed", sa.Integer(), nullable=False),
        sa.Column("calibration_case_ids", sa.JSON(), nullable=False),
        sa.Column("holdout_case_ids", sa.JSON(), nullable=False),
        sa.Column("semantic_index_identity", sa.String(64), nullable=False),
        sa.Column("corpus_identity", sa.String(64), nullable=False),
        sa.Column("retrieval_configuration", sa.JSON(), nullable=False),
        sa.Column("selection_policy", sa.JSON(), nullable=False),
        sa.Column(
            "frozen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("selected_retrieval_mode", sa.String(30)),
        sa.Column("calibration_metrics", sa.JSON()),
        sa.Column("selection_reason", sa.Text()),
        sa.Column("locked_at", sa.DateTime(timezone=True)),
        sa.Column("holdout_started_at", sa.DateTime(timezone=True)),
        sa.Column("holdout_completed_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "retrieval_benchmark_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "dataset_id",
            sa.String(100),
            sa.ForeignKey("retrieval_benchmark_locks.dataset_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("partition", sa.String(20), nullable=False),
        sa.Column("retrieval_mode", sa.String(30), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("category_metrics", sa.JSON(), nullable=False),
        sa.Column("case_results", sa.JSON(), nullable=False),
        sa.Column("branch_contribution", sa.JSON(), nullable=False),
        sa.Column("latency", sa.JSON(), nullable=False),
        sa.Column("usage", sa.JSON(), nullable=False),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("dataset_id", "partition", "retrieval_mode"),
    )
    op.create_index(
        "ix_retrieval_benchmark_runs_dataset_id",
        "retrieval_benchmark_runs",
        ["dataset_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_retrieval_benchmark_runs_dataset_id",
        table_name="retrieval_benchmark_runs",
    )
    op.drop_table("retrieval_benchmark_runs")
    op.drop_table("retrieval_benchmark_locks")
