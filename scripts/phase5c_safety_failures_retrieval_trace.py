# ruff: noqa: E501
"""
Extract retrieval traces for Phase-5C safety failures.

We only re-run retrieval (Dense/BM25/RRF/Cross-Encoder Top-5) for a small
set of query_ids discovered from phase5c_safety_failures_detailed.jsonl.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import create_engine

from rag_workbench.config import get_settings
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    DENSE_DEPTH,
    FINAL_TOP_K,
    RRF_K,
    UNION_LIMIT,
    SEMANTIC_INDEX_IDENTITY,
)
from rag_workbench.experiments.reranker_e2e_benchmark import RERANKER_REVISION
from rag_workbench.experiments.v2_document_diversity import ranking_candidate
from rag_workbench.experiments.v3_generate_verify import V3GenerateVerifyBenchmark
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal


PHASE5C_DATASET_PATH = Path("data/eval/phase5c/v3_clean_120_cases.jsonl")
FAILURES_PATH = Path(
    "data/experiments/v3-phase5d-safe-generator-candidate/phase5c_safety_failures_detailed.jsonl"
)
OUT_DIR = Path("data/experiments/v3-phase5d-safe-generator-candidate")
OUT_PATH = OUT_DIR / "phase5c_safety_failures_retrieval_trace.jsonl"


def load_cases() -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for line in PHASE5C_DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        cases[d["query_id"]] = d
    return cases


def main() -> None:
    failures = json.loads(FAILURES_PATH.read_text(encoding="utf-8"))
    query_ids = sorted({item["query_id"] for item in failures})
    cases_by_id = load_cases()

    settings = get_settings()
    engine = create_engine(settings.database_url)
    # V3GenerateVerifyBenchmark builds its own reranker + retrieval infra.
    session = engine.connect()

    # We need the benchmark to get the frozen reranker revision identity.
    # (V3GenerateVerifyBenchmark expects an ORM Session; we’ll use a Session
    # created via SQLAlchemy sessionmaker by reusing its internal construction.)
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=engine)
    orm_session = SessionLocal()
    try:
        inner = V3GenerateVerifyBenchmark(orm_session, settings)
        inner._ensure_v3_identity()
        reranker = inner.reranker_factory()
        if reranker.resolved_revision != RERANKER_REVISION:
            raise RuntimeError("Cross-encoder revision mismatch")

        dense_retriever = Retriever(
            orm_session, inner._embedding_provider(), SEMANTIC_INDEX_IDENTITY
        )
        bm25 = BM25Retriever(
            orm_session,
            index_identity=SEMANTIC_INDEX_IDENTITY,
            embedding_provider="openai-compatible",
            embedding_model="text-embedding-3-small",
            embedding_version="1",
            embedding_dimension=64,
            config=BM25Config(),
        )

        outputs: list[dict[str, Any]] = []
        for qid in query_ids:
            case = cases_by_id[qid]
            # Reconstruct a minimal principal object.
            principal = Principal(
                principal_id=case["principal"]["principal_id"],
                tenant_id=case["principal"]["tenant_id"],
                permission_groups=tuple(case["principal"].get("permission_groups") or ()),
            )

            # Dense branch
            embedding = dense_retriever.query_embedding_cache.get_or_embed(case["question"])
            orm_session.commit()
            dense_candidates = dense_retriever.retrieve_with_embedding(
                embedding,
                top_k=DENSE_DEPTH,
                score_threshold=0.28,
                principal=principal,
            )

            # BM25 branch
            bm25_candidates = bm25.retrieve(
                case["question"], top_k=BM25_DEPTH, principal=principal
            )

            union_all = reciprocal_rank_fusion(
                dense_candidates, bm25_candidates, top_k=10_000, rrf_k=RRF_K
            )
            union = union_all[:UNION_LIMIT]

            reranked = reranker.rerank(case["question"], union)
            top5 = [
                ranking_candidate(item)
                | {
                    "rank": item.reranked_rank,
                    "score": item.reranker_score,
                    "retrieval_source": "pointwise_cross_encoder_top5",
                }
                for item in reranked[:FINAL_TOP_K]
            ]

            def _simplify(items: list[RetrievalResult]) -> list[dict[str, Any]]:
                return [
                    {
                        "chunk_id": item.chunk_id,
                        "document_id": item.document_id,
                        "version": item.version,
                        "document_version_id": item.document_version_id,
                        "score": getattr(item, "score", None),
                        "text_preview": (item.text or "")[:120],
                    }
                    for item in items
                ]

            outputs.append(
                {
                    "query_id": qid,
                    "dense_candidates": _simplify(dense_candidates),
                    "bm25_candidates": _simplify(bm25_candidates),
                    "rrf_union": _simplify(union),
                    "cross_encoder_top5": top5,
                }
            )

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote retrieval traces for {len(outputs)} cases to {OUT_PATH}")
    finally:
        orm_session.close()


if __name__ == "__main__":
    main()

