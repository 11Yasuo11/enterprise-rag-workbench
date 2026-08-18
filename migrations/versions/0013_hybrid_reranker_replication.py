"""Persist the final Hybrid+Cross-Encoder replication architecture freeze.

Revision ID: 0013
Revises: 0012
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "retrieval_architecture_locks",
        sa.Column("architecture_id", sa.String(length=100), primary_key=True),
        sa.Column("selected_retriever", sa.String(length=40), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("selection_policy", sa.JSON(), nullable=False),
        sa.Column("dataset_id", sa.String(length=100), nullable=False),
        sa.Column("dataset_hash", sa.String(length=64), nullable=False),
        sa.Column("retrieval_metrics", sa.JSON(), nullable=False),
        sa.Column("immutable", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "frozen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("retrieval_architecture_locks")
