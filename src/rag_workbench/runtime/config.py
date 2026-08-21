"""Production RAG configuration — explicit stage Top-K, no ambiguous top_k=5."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rag_workbench.config import Settings, get_settings
from rag_workbench.reranking.cross_encoder import MODEL_ID as CROSS_ENCODER_MODEL_ID

# Frozen research semantic corpus identity (text-embedding-3-small, dim=64).
# Not recomputed from IngestionConfig — pin explicitly for production serving.
CANONICAL_SEMANTIC_INDEX_IDENTITY = (
    "e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466"
)


@dataclass(frozen=True)
class ProductionRagConfig:
    """Canonical serving/evaluation config (validated research depths).

    Validated by V4–V14 ``retrieve_trace`` + release runner:
    dense 20 + BM25 20 → RRF k=60 → union ≤30 → CE → Top-15 internal pool.
    """

    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 64
    embedding_version: str = "1"
    dense_top_k: int = 20
    bm25_top_k: int = 20
    dense_score_threshold: float = 0.28
    rrf_k: int = 60
    fused_candidate_cap: int = 30
    cross_encoder_model: str = CROSS_ENCODER_MODEL_ID
    cross_encoder_top_k: int = 15
    luna_model: str = "gpt-5.6-luna"
    sol_model: str = "gpt-5.6-sol"
    final_generation_llm: str = "disabled"
    assembler_id: str = "deterministic-requirement-assembler-v3"
    max_luna_calls_per_request: int = 1
    max_sol_calls_per_request: int = 1
    allow_hashing_embeddings: bool = False
    require_cross_encoder: bool = True
    allow_identity_reranker: bool = False
    index_identity: str | None = CANONICAL_SEMANTIC_INDEX_IDENTITY

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate_for_serving(self) -> None:
        if self.embedding_provider == "hashing" and not self.allow_hashing_embeddings:
            raise RuntimeError(
                "RAG_CONFIG_INVALID: hashing embeddings are not allowed in production mode"
            )
        if (
            self.embedding_provider in {"openai", "openai-compatible"}
            and self.embedding_model != "text-embedding-3-small"
            and not self.allow_hashing_embeddings
        ):
            raise RuntimeError(
                "RAG_CONFIG_INVALID: production embedding model must be text-embedding-3-small"
            )
        if self.dense_top_k != 20 or self.bm25_top_k != 20:
            raise RuntimeError("RAG_CONFIG_INVALID: dense/bm25 top_k must be 20")
        if self.rrf_k != 60 or self.fused_candidate_cap != 30:
            raise RuntimeError("RAG_CONFIG_INVALID: rrf_k must be 60 and fused_candidate_cap 30")
        if self.cross_encoder_top_k != 15:
            raise RuntimeError("RAG_CONFIG_INVALID: cross_encoder_top_k must be 15")
        if self.final_generation_llm != "disabled":
            raise RuntimeError("RAG_CONFIG_INVALID: final_generation_llm must be disabled")
        if self.require_cross_encoder and not self.cross_encoder_model:
            raise RuntimeError("RAG_CONFIG_INVALID: cross_encoder_model required")
        if self.allow_identity_reranker and not self.allow_hashing_embeddings:
            raise RuntimeError(
                "RAG_CONFIG_INVALID: identity reranker only allowed with hashing/local mode"
            )
        if (
            self.embedding_provider in {"openai", "openai-compatible"}
            and not self.allow_hashing_embeddings
            and self.index_identity != CANONICAL_SEMANTIC_INDEX_IDENTITY
        ):
            raise RuntimeError(
                "RAG_CONFIG_INVALID: production semantic index_identity must be the frozen "
                "canonical text-embedding-3-small corpus identity"
            )


def production_config_from_settings(settings: Settings | None = None) -> ProductionRagConfig:
    """Build config from environment. Hashing is allowed only when explicitly configured."""
    settings = settings or get_settings()
    provider = settings.embedding_provider
    allow_hashing = provider == "hashing"
    index_identity = None if allow_hashing else CANONICAL_SEMANTIC_INDEX_IDENTITY
    return ProductionRagConfig(
        embedding_provider=provider,
        embedding_model=(
            settings.embedding_model if allow_hashing else "text-embedding-3-small"
        ),
        embedding_dimension=settings.embedding_dimension,
        embedding_version=settings.embedding_version,
        allow_hashing_embeddings=allow_hashing,
        allow_identity_reranker=allow_hashing,
        require_cross_encoder=not allow_hashing,
        index_identity=index_identity,
        luna_model=settings.answerability_gate_model or "gpt-5.6-luna",
        sol_model=settings.judge_model or "gpt-5.6-sol",
    )
