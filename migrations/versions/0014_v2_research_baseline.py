"""Persist the v2 research identity and quality-recovery census.

Revision ID: 0014
Revises: 0013
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "research_architecture_locks",
        sa.Column("architecture_id", sa.String(length=100), primary_key=True),
        sa.Column("parent_architecture_id", sa.String(length=100), nullable=False),
        sa.Column("selected_retriever", sa.String(length=40), nullable=False),
        sa.Column("research_status", sa.String(length=20), nullable=False),
        sa.Column("production_status", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("control_configuration", sa.JSON(), nullable=False),
        sa.Column("control_equivalence_hash", sa.String(length=64), nullable=False),
        sa.Column("diagnosis_dataset_id", sa.String(length=100), nullable=False),
        sa.Column("diagnosis_dataset_hash", sa.String(length=64), nullable=False),
        sa.Column("diagnosis_only", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "promotion_evidence_forbidden",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column("security_guardrails", sa.JSON(), nullable=False),
        sa.Column("v1_preservation", sa.JSON(), nullable=False),
        sa.Column("failure_census", sa.JSON(), nullable=False),
        sa.Column("ranking_diagnostic", sa.JSON(), nullable=False),
        sa.Column("judge_false_negative_diagnostic", sa.JSON(), nullable=False),
        sa.Column("operational_diagnostic", sa.JSON(), nullable=False),
        sa.Column("primary_bottleneck", sa.String(length=80), nullable=False),
        sa.Column("recommended_ranking_intervention", sa.Text(), nullable=False),
        sa.Column("immutable", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "frozen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("research_architecture_locks")
