"""Add embedding/index identity and persistent experiment results."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("embedding_provider", sa.String(100), nullable=False, server_default="hashing"),
    )
    op.add_column(
        "chunks",
        sa.Column(
            "index_identity", sa.String(64), nullable=False, server_default="legacy-baseline-v1"
        ),
    )
    op.drop_constraint("chunks_document_version_id_chunk_index_key", "chunks", type_="unique")
    op.create_unique_constraint(
        "uq_chunks_version_index_chunk",
        "chunks",
        ["document_version_id", "index_identity", "chunk_index"],
    )
    op.create_index("ix_chunks_index_identity", "chunks", ["index_identity"])
    op.add_column("rag_runs", sa.Column("retrieval_latency_ms", sa.Float()))
    op.add_column("rag_runs", sa.Column("generation_latency_ms", sa.Float()))
    op.add_column("rag_runs", sa.Column("embedding_tokens", sa.Integer()))

    op.create_table(
        "experiment_configs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("ingestion_config_hash", sa.String(64), nullable=False),
        sa.Column("retrieval_config_hash", sa.String(64), nullable=False),
        sa.Column("generation_config_hash", sa.String(64), nullable=False),
        sa.Column("index_identity", sa.String(64), nullable=False),
        sa.Column("corpus_version", sa.String(100), nullable=False),
        sa.Column("evaluation_dataset_version", sa.String(100), nullable=False),
        sa.Column("ingestion_config", sa.JSON(), nullable=False),
        sa.Column("retrieval_config", sa.JSON(), nullable=False),
        sa.Column("generation_config", sa.JSON(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_experiment_configs_config_hash", "experiment_configs", ["config_hash"])
    op.create_index(
        "ix_experiment_configs_index_identity", "experiment_configs", ["index_identity"]
    )
    op.create_table(
        "experiment_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "config_id",
            sa.String(36),
            sa.ForeignKey("experiment_configs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column(
            "aggregate_metrics", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")
        ),
        sa.Column(
            "category_metrics", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")
        ),
        sa.Column("total_latency_ms", sa.Float()),
        sa.Column("retrieval_latency_ms", sa.Float()),
        sa.Column("generation_latency_ms", sa.Float()),
        sa.Column("input_tokens", sa.Integer()),
        sa.Column("output_tokens", sa.Integer()),
        sa.Column("embedding_tokens", sa.Integer()),
        sa.Column("error", sa.Text()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_experiment_runs_config_id", "experiment_runs", ["config_id"])
    op.create_index("ix_experiment_runs_status", "experiment_runs", ["status"])
    op.create_table(
        "experiment_case_results",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "experiment_run_id",
            sa.String(36),
            sa.ForeignKey("experiment_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rag_run_id", sa.String(36), sa.ForeignKey("rag_runs.id", ondelete="SET NULL")),
        sa.Column("eval_case_id", sa.String(200), nullable=False),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text()),
        sa.Column(
            "expected_document_ids", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")
        ),
        sa.Column(
            "expected_chunk_ids", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")
        ),
        sa.Column(
            "retrieved_document_ids",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
        sa.Column(
            "retrieved_chunk_ids", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")
        ),
        sa.Column(
            "retrieval_trace", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")
        ),
        sa.Column("answer", sa.Text()),
        sa.Column("citations", sa.JSON(), nullable=False, server_default=sa.text("'[]'::json")),
        sa.Column("expected_abstain", sa.Boolean(), nullable=False),
        sa.Column("abstained", sa.Boolean()),
        sa.Column("metrics", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
        sa.Column("failure_type", sa.String(100)),
        sa.Column("security_passed", sa.Boolean()),
        sa.Column("retrieval_latency_ms", sa.Float()),
        sa.Column("generation_latency_ms", sa.Float()),
        sa.Column("total_latency_ms", sa.Float()),
        sa.Column("error", sa.Text()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("experiment_run_id", "eval_case_id"),
    )
    op.create_index(
        "ix_experiment_case_results_experiment_run_id",
        "experiment_case_results",
        ["experiment_run_id"],
    )
    op.create_index("ix_experiment_case_results_category", "experiment_case_results", ["category"])


def downgrade() -> None:
    op.drop_table("experiment_case_results")
    op.drop_table("experiment_runs")
    op.drop_table("experiment_configs")
    op.drop_column("rag_runs", "embedding_tokens")
    op.drop_column("rag_runs", "generation_latency_ms")
    op.drop_column("rag_runs", "retrieval_latency_ms")
    op.drop_index("ix_chunks_index_identity", table_name="chunks")
    op.drop_constraint("uq_chunks_version_index_chunk", "chunks", type_="unique")
    op.create_unique_constraint(None, "chunks", ["document_version_id", "chunk_index"])
    op.drop_column("chunks", "index_identity")
    op.drop_column("chunks", "embedding_provider")
