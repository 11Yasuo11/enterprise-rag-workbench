"""Add safe query embedding cache and decomposed latency fields."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


LATENCY_COLUMNS = (
    "query_embedding_latency_ms",
    "embedding_cache_lookup_latency_ms",
    "vector_search_latency_ms",
    "acl_filter_latency_ms",
    "context_construction_latency_ms",
)


def upgrade() -> None:
    op.create_table(
        "query_embedding_cache",
        sa.Column("cache_key", sa.String(64), primary_key=True),
        sa.Column("normalized_query", sa.Text(), nullable=False),
        sa.Column("embedding_provider", sa.String(100), nullable=False),
        sa.Column("embedding_model", sa.String(200), nullable=False),
        sa.Column("embedding_version", sa.String(100), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("embedding", VECTOR(64), nullable=False),
        sa.Column("input_tokens", sa.Integer()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "normalized_query",
            "embedding_provider",
            "embedding_model",
            "embedding_version",
            "embedding_dimension",
            name="uq_query_embedding_cache_identity",
        ),
    )
    for table in ("rag_runs", "experiment_case_results", "experiment_runs"):
        for column in LATENCY_COLUMNS:
            op.add_column(table, sa.Column(column, sa.Float()))
    op.add_column("rag_runs", sa.Column("query_embedding_cache_hit", sa.Boolean()))
    op.add_column(
        "experiment_case_results", sa.Column("query_embedding_cache_hit", sa.Boolean())
    )
    op.add_column("experiment_runs", sa.Column("query_embedding_cache_hits", sa.Integer()))
    op.add_column("experiment_runs", sa.Column("query_embedding_cache_misses", sa.Integer()))
    op.add_column("experiment_runs", sa.Column("external_embedding_calls", sa.Integer()))


def downgrade() -> None:
    op.drop_column("experiment_runs", "external_embedding_calls")
    op.drop_column("experiment_runs", "query_embedding_cache_misses")
    op.drop_column("experiment_runs", "query_embedding_cache_hits")
    op.drop_column("experiment_case_results", "query_embedding_cache_hit")
    op.drop_column("rag_runs", "query_embedding_cache_hit")
    for table in ("experiment_runs", "experiment_case_results", "rag_runs"):
        for column in reversed(LATENCY_COLUMNS):
            op.drop_column(table, column)
    op.drop_table("query_embedding_cache")
