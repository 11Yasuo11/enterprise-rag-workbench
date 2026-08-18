"""Add evidence-sufficiency gate persistence and cache."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "answerability_gate_cache",
        sa.Column("cache_key", sa.String(64), primary_key=True),
        sa.Column("normalized_question", sa.Text(), nullable=False),
        sa.Column("retrieved_evidence", sa.JSON(), nullable=False),
        sa.Column("judge_provider", sa.String(100), nullable=False),
        sa.Column("judge_model", sa.String(200), nullable=False),
        sa.Column("judge_version", sa.String(100), nullable=False),
        sa.Column("judge_prompt_version", sa.String(100), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer()),
        sa.Column("completion_tokens", sa.Integer()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.add_column("experiment_configs", sa.Column("gate_config_hash", sa.String(64)))
    op.add_column("experiment_configs", sa.Column("gate_config", sa.JSON()))

    op.add_column(
        "rag_runs",
        sa.Column("retrieved_chunk_ids", sa.JSON(), nullable=False, server_default="[]"),
    )
    for table in ("rag_runs", "experiment_case_results"):
        op.add_column(
            table,
            sa.Column("supporting_chunk_ids", sa.JSON(), nullable=False, server_default="[]"),
        )
        op.add_column(
            table,
            sa.Column(
                "generation_context_chunk_ids", sa.JSON(), nullable=False, server_default="[]"
            ),
        )
        op.add_column(table, sa.Column("answerability_result", sa.JSON()))
        op.add_column(table, sa.Column("answerability_gate_cache_lookup_latency_ms", sa.Float()))
        op.add_column(table, sa.Column("answerability_judge_latency_ms", sa.Float()))
        op.add_column(table, sa.Column("context_pruning_latency_ms", sa.Float()))
        op.add_column(table, sa.Column("gate_cache_hit", sa.Boolean()))
        op.add_column(table, sa.Column("external_judge_calls", sa.Integer()))
        op.add_column(table, sa.Column("judge_prompt_tokens", sa.Integer()))
        op.add_column(table, sa.Column("judge_completion_tokens", sa.Integer()))

    for column in (
        "answerability_gate_cache_lookup_latency_ms",
        "answerability_judge_latency_ms",
        "context_pruning_latency_ms",
    ):
        op.add_column("experiment_runs", sa.Column(column, sa.Float()))
    for column in (
        "gate_cache_hits",
        "gate_cache_misses",
        "external_judge_calls",
        "judge_prompt_tokens",
        "judge_completion_tokens",
    ):
        op.add_column("experiment_runs", sa.Column(column, sa.Integer()))


def downgrade() -> None:
    for column in (
        "judge_completion_tokens",
        "judge_prompt_tokens",
        "external_judge_calls",
        "gate_cache_misses",
        "gate_cache_hits",
        "context_pruning_latency_ms",
        "answerability_judge_latency_ms",
        "answerability_gate_cache_lookup_latency_ms",
    ):
        op.drop_column("experiment_runs", column)
    for table in ("experiment_case_results", "rag_runs"):
        for column in (
            "judge_completion_tokens",
            "judge_prompt_tokens",
            "external_judge_calls",
            "gate_cache_hit",
            "context_pruning_latency_ms",
            "answerability_judge_latency_ms",
            "answerability_gate_cache_lookup_latency_ms",
            "answerability_result",
            "generation_context_chunk_ids",
            "supporting_chunk_ids",
        ):
            op.drop_column(table, column)
    op.drop_column("rag_runs", "retrieved_chunk_ids")
    op.drop_column("experiment_configs", "gate_config")
    op.drop_column("experiment_configs", "gate_config_hash")
    op.drop_table("answerability_gate_cache")
