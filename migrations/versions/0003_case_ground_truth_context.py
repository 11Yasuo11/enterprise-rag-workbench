"""Persist ACL principal and version/security ground truth for experiment cases."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "experiment_case_results",
        sa.Column(
            "forbidden_document_ids",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
    )
    op.add_column(
        "experiment_case_results",
        sa.Column(
            "expected_versions",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
    )
    op.add_column(
        "experiment_case_results",
        sa.Column(
            "principal",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
    )


def downgrade() -> None:
    op.drop_column("experiment_case_results", "principal")
    op.drop_column("experiment_case_results", "expected_versions")
    op.drop_column("experiment_case_results", "forbidden_document_ids")
