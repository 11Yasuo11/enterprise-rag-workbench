"""Canonical production RAG runtime (shared by Web serving and evaluation)."""

from rag_workbench.runtime.canonical_runtime import (
    CanonicalRagRuntime,
    build_canonical_runtime,
)
from rag_workbench.runtime.config import (
    CANONICAL_SEMANTIC_INDEX_IDENTITY,
    ProductionRagConfig,
    production_config_from_settings,
)
from rag_workbench.runtime.types import CanonicalQueryResult

__all__ = [
    "CANONICAL_SEMANTIC_INDEX_IDENTITY",
    "CanonicalQueryResult",
    "CanonicalRagRuntime",
    "ProductionRagConfig",
    "build_canonical_runtime",
    "production_config_from_settings",
]
