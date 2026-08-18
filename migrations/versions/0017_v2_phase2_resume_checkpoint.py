"""Persist Phase 2 resume-checkpoint audit without mutating frozen results.

Revision ID: 0017
Revises: 0016
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "v2_phase2_experiment_locks",
        sa.Column("resume_checkpoint", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("v2_phase2_experiment_locks", "resume_checkpoint")
