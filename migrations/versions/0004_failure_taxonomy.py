"""Persist structured evaluation failure taxonomy and diagnostics."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "experiment_case_results",
        sa.Column("failure_types", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
    )
    op.add_column(
        "experiment_case_results",
        sa.Column(
            "failure_details",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'::json"),
        ),
    )
    op.execute(
        """
        UPDATE experiment_case_results
        SET failure_types = CASE
          WHEN failure_type IS NULL THEN '[]'::json
          WHEN failure_type = 'retrieval_failure' THEN '[\"RETRIEVAL_MISS\"]'::json
          WHEN failure_type = 'version_failure' THEN '[\"VERSION_FAILURE\"]'::json
          WHEN failure_type = 'security_failure' THEN '[\"ACL_FAILURE\"]'::json
          WHEN failure_type = 'generation_failure' THEN '[\"GENERATION_FAILURE\"]'::json
          ELSE '[\"UNKNOWN\"]'::json
        END
        """
    )


def downgrade() -> None:
    op.drop_column("experiment_case_results", "failure_details")
    op.drop_column("experiment_case_results", "failure_types")
