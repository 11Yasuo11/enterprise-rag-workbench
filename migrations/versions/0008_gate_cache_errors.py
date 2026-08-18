"""Persist fail-closed gate errors so paired candidates reuse them."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("answerability_gate_cache", sa.Column("operational_error", sa.String(100)))


def downgrade() -> None:
    op.drop_column("answerability_gate_cache", "operational_error")
