from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from rag_workbench.reranking.base import RerankedResult
from rag_workbench.retrieval.vector_search import RetrievalResult

RUNTIME_ALLOWED_FIELDS = frozenset(
    {
        "document_id",
        "chunk_id",
        "rank",
        "score",
        "reranker_score",
        "reranked_rank",
        "original_dense_rank",
        "dense_score",
        "lexical_score",
        "fusion_score",
        "text",
        "version",
        "section",
        "title",
        "source",
        "source_type",
        "page",
        "metadata",
        "document_version_id",
        "retrieval_source",
        "found_by_dense",
        "found_by_bm25",
        "found_by_both",
        "dense_rank",
        "bm25_rank",
    }
)
EVALUATION_LABEL_FIELDS = frozenset(
    {
        "required_document_ids",
        "required_chunk_ids",
        "required_chunk_markers",
        "required_fact_ids",
        "required_version_ids",
        "expected_answer",
        "expected_answerability",
        "expected_facts",
        "expected_document_ids",
        "category",
        "requires_multiple_chunks_same_document",
        "requires_exactly_two_chunks_same_document",
        "crowding_relevant_same_document",
        "number_of_required_documents",
        "should_abstain",
        "forbidden_document_ids",
        "preferred_source_id",
        "expected_acl_behavior",
        "expected_prompt_injection_behavior",
        "expected_versions",
        "expected_access_behavior",
        "security_checks",
    }
)


def canonical_document_id(candidate: Any) -> str:
    if isinstance(candidate, RerankedResult):
        return candidate.result.document_id
    if isinstance(candidate, RetrievalResult):
        return candidate.document_id
    if isinstance(candidate, dict):
        document_id = candidate.get("document_id")
        if document_id:
            return str(document_id)
        nested = candidate.get("result") or {}
        return str(nested.get("document_id"))
    return str(candidate.document_id)


def assert_no_evaluation_label_leakage(candidate: Any) -> None:
    payload = candidate if isinstance(candidate, dict) else getattr(candidate, "__dict__", {})
    if not isinstance(payload, dict):
        return
    leaked = EVALUATION_LABEL_FIELDS.intersection(payload)
    if leaked:
        raise ValueError(f"evaluation labels leaked into ranking: {sorted(leaked)}")


def select_document_diversified_top5(
    ranked_candidates: Sequence[Any],
    *,
    top_k: int = 5,
) -> list[Any]:
    """Keep the highest-ranked chunk per canonical document_id, then cut Top-K.

    The input order is the frozen Cross-Encoder ranking. This function does not
    rescore, apply MMR, or read evaluation labels.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    selected: list[Any] = []
    seen_document_ids: set[str] = set()
    for candidate in ranked_candidates:
        assert_no_evaluation_label_leakage(candidate)
        document_id = canonical_document_id(candidate)
        if document_id in seen_document_ids:
            continue
        selected.append(candidate)
        seen_document_ids.add(document_id)
        if len(selected) == top_k:
            break
    return selected


def select_max_2_chunks_per_document_top5(
    ranked_candidates: Sequence[Any],
    *,
    top_k: int = 5,
    max_chunks_per_document: int = 2,
) -> list[Any]:
    """Keep at most two Cross-Encoder-ranked chunks per canonical document_id.

    The input order is the frozen Cross-Encoder ranking. This function does not
    rescore, apply MMR, change the cap, or read evaluation labels. The cap is
    precommitted at two and is not a search parameter.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if max_chunks_per_document != 2:
        raise ValueError("Phase 2 cap is frozen at two chunks per document")
    selected: list[Any] = []
    document_counts: dict[str, int] = {}
    for candidate in ranked_candidates:
        assert_no_evaluation_label_leakage(candidate)
        document_id = canonical_document_id(candidate)
        if document_counts.get(document_id, 0) >= 2:
            continue
        selected.append(candidate)
        document_counts[document_id] = document_counts.get(document_id, 0) + 1
        if len(selected) == top_k:
            break
    return selected


def as_retrieval_result(candidate: Any, *, rank: int, retrieval_source: str) -> RetrievalResult:
    if isinstance(candidate, RerankedResult):
        return replace(
            candidate.result,
            rank=rank,
            score=candidate.reranker_score,
            retrieval_source=retrieval_source,
        )
    if isinstance(candidate, RetrievalResult):
        return replace(candidate, rank=rank, retrieval_source=retrieval_source)
    payload = dict(candidate)
    payload["rank"] = rank
    payload["retrieval_source"] = retrieval_source
    allowed = {key: payload[key] for key in RUNTIME_ALLOWED_FIELDS if key in payload}
    return RetrievalResult(**allowed)
