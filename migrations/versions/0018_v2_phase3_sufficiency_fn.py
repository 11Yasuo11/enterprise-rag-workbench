"""Persist the v2 Phase 3 evidence-sufficiency false-negative reduction experiment.

Revision ID: 0018
Revises: 0017
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "research_architecture_locks",
        sa.Column("selected_v2_judge", sa.String(length=80), nullable=True),
    )
    op.add_column(
        "research_architecture_locks",
        sa.Column("phase3_dataset_id", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "research_architecture_locks",
        sa.Column("judge_research_status", sa.String(length=40), nullable=True),
    )
    op.create_table(
        "v2_phase3_experiment_locks",
        sa.Column("dataset_id", sa.String(length=100), primary_key=True),
        sa.Column("architecture_id", sa.String(length=100), nullable=False),
        sa.Column("dataset_hash", sa.String(length=64), nullable=False),
        sa.Column("case_ids", sa.JSON(), nullable=False),
        sa.Column("category_distribution", sa.JSON(), nullable=False),
        sa.Column("generation_method", sa.String(length=100), nullable=False),
        sa.Column("maximum_prior_overlap", sa.Float(), nullable=False),
        sa.Column("closest_previous_case", sa.JSON(), nullable=True),
        sa.Column("overlap_report", sa.JSON(), nullable=False),
        sa.Column("selection_policy", sa.JSON(), nullable=False),
        sa.Column("control_configuration", sa.JSON(), nullable=False),
        sa.Column("candidate_configuration", sa.JSON(), nullable=False),
        sa.Column("control_prompt_identity", sa.JSON(), nullable=False),
        sa.Column("candidate_prompt_identity", sa.JSON(), nullable=False),
        sa.Column("semantic_index_identity", sa.String(length=64), nullable=False),
        sa.Column("corpus_identity", sa.String(length=64), nullable=False),
        sa.Column("embedding_preflight", sa.JSON(), nullable=True),
        sa.Column("judge_preflight", sa.JSON(), nullable=True),
        sa.Column("shared_traces", sa.JSON(), nullable=True),
        sa.Column("control_metrics", sa.JSON(), nullable=True),
        sa.Column("candidate_metrics", sa.JSON(), nullable=True),
        sa.Column("retrieval_results", sa.JSON(), nullable=True),
        sa.Column("retrieval_complete_judge", sa.JSON(), nullable=True),
        sa.Column("all_case_answerability", sa.JSON(), nullable=True),
        sa.Column("exact_id_analysis", sa.JSON(), nullable=True),
        sa.Column("three_document_analysis", sa.JSON(), nullable=True),
        sa.Column("two_document_analysis", sa.JSON(), nullable=True),
        sa.Column("near_duplicate_analysis", sa.JSON(), nullable=True),
        sa.Column("semantic_analysis", sa.JSON(), nullable=True),
        sa.Column("version_analysis", sa.JSON(), nullable=True),
        sa.Column("false_negative_rescues", sa.JSON(), nullable=True),
        sa.Column("false_positive_regressions", sa.JSON(), nullable=True),
        sa.Column("answerable_to_abstain_regressions", sa.JSON(), nullable=True),
        sa.Column("supporting_id_quality", sa.JSON(), nullable=True),
        sa.Column("abstention_safety", sa.JSON(), nullable=True),
        sa.Column("prompt_injection_safety", sa.JSON(), nullable=True),
        sa.Column("security", sa.JSON(), nullable=True),
        sa.Column("end_to_end_confirmation", sa.JSON(), nullable=True),
        sa.Column("generation_failures", sa.JSON(), nullable=True),
        sa.Column("latency", sa.JSON(), nullable=True),
        sa.Column("usage", sa.JSON(), nullable=True),
        sa.Column("cost", sa.JSON(), nullable=True),
        sa.Column("selection", sa.JSON(), nullable=True),
        sa.Column("selected_judge", sa.String(length=80), nullable=True),
        sa.Column("judge_research_status", sa.String(length=40), nullable=True),
        sa.Column("primary_remaining_bottleneck", sa.String(length=80), nullable=True),
        sa.Column(
            "dataset_frozen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "selection_policy_frozen_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("retrieval_frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retrieval_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("judge_execution_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_v2_phase3_experiment_locks_architecture_id",
        "v2_phase3_experiment_locks",
        ["architecture_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_v2_phase3_experiment_locks_architecture_id",
        table_name="v2_phase3_experiment_locks",
    )
    op.drop_table("v2_phase3_experiment_locks")
    op.drop_column("research_architecture_locks", "judge_research_status")
    op.drop_column("research_architecture_locks", "phase3_dataset_id")
    op.drop_column("research_architecture_locks", "selected_v2_judge")
