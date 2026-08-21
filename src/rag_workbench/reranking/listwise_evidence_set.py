# ruff: noqa: E501
"""Listwise evidence-set Top-5 selection.

Instead of taking the five highest-scoring individual chunks, this algorithm
greedily selects a *set* of at most five chunks that jointly maximises
relevance while penalising redundancy and rewarding complementary evidence.

The selection objective for each greedy step is:

    score(chunk | already_selected) =
        relevance(chunk)
        - λ_redundancy · max_similarity(chunk, already_selected)
        + λ_complement · complement_bonus(chunk, already_selected, query)

where:
    relevance    = normalised Cross-Encoder score (shifted to [0, 1] within candidate set)
    redundancy   = maximum token-Jaccard similarity between chunk text and any already-selected chunk
    complement   = fraction of *new* query uni-grams covered by the chunk that are not yet
                   covered by any already-selected chunk (query-conditioned, not just diversity)

All signals are computed from runtime-available data only:
    query text, chunk text, Cross-Encoder score, document_id, version.

No ground-truth labels (required_document_ids, expected_answer, etc.) are used.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from rag_workbench.reranking.base import RerankedResult
from rag_workbench.reranking.document_diversity import (
    assert_no_evaluation_label_leakage,
)
from rag_workbench.retrieval.vector_search import RetrievalResult

ALGORITHM_ID = "LISTWISE_EVIDENCE_SET_TOP5"
ALGORITHM_VERSION = "1.0"

LAMBDA_REDUNDANCY = 0.30
LAMBDA_COMPLEMENT = 0.25

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.casefold()))


def _chunk_text(candidate: Any) -> str:
    if isinstance(candidate, RerankedResult):
        return candidate.result.text
    if isinstance(candidate, RetrievalResult):
        return candidate.text
    if isinstance(candidate, dict):
        return str(candidate.get("text") or "")
    return str(getattr(candidate, "text", ""))


def _chunk_score(candidate: Any) -> float:
    if isinstance(candidate, RerankedResult):
        return candidate.reranker_score
    if isinstance(candidate, dict):
        return float(candidate.get("score") or 0.0)
    return float(getattr(candidate, "score", 0.0))


def _normalise_scores(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    span = hi - lo
    if span < 1e-12:
        return [0.5] * len(scores)
    return [(s - lo) / span for s in scores]


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def select_listwise_evidence_set_top5(
    ranked_candidates: Sequence[Any],
    query: str,
    *,
    top_k: int = 5,
    lambda_redundancy: float = LAMBDA_REDUNDANCY,
    lambda_complement: float = LAMBDA_COMPLEMENT,
) -> list[Any]:
    """Greedy query-conditioned evidence-set selection.

    Parameters
    ----------
    ranked_candidates
        CE-ranked candidates (highest-score first). May be ``RerankedResult``,
        ``RetrievalResult``, or dicts.
    query
        The original user question.
    top_k
        Maximum evidence set size (default 5).
    lambda_redundancy
        Weight for the redundancy penalty.
    lambda_complement
        Weight for the complement bonus.

    Returns
    -------
    list
        Up to *top_k* candidates forming the selected evidence set, in order
        of selection (highest marginal value first).
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    candidates = list(ranked_candidates)
    if not candidates:
        return []
    for c in candidates:
        assert_no_evaluation_label_leakage(c)

    query_tokens = _tokenize(query)
    raw_scores = [_chunk_score(c) for c in candidates]
    norm_scores = _normalise_scores(raw_scores)
    chunk_tokens = [_tokenize(_chunk_text(c)) for c in candidates]

    selected: list[Any] = []
    selected_tokens: list[set[str]] = []
    covered_query_tokens: set[str] = set()
    used_indices: set[int] = set()

    for _ in range(min(top_k, len(candidates))):
        best_idx = -1
        best_score = float("-inf")
        for idx in range(len(candidates)):
            if idx in used_indices:
                continue
            relevance = norm_scores[idx]
            ct = chunk_tokens[idx]

            redundancy = max(_jaccard(ct, st) for st in selected_tokens) if selected_tokens else 0.0

            uncovered = query_tokens - covered_query_tokens
            new_coverage = len(ct & uncovered) / len(uncovered) if uncovered else 0.0

            marginal = relevance - lambda_redundancy * redundancy + lambda_complement * new_coverage

            if marginal > best_score:
                best_score = marginal
                best_idx = idx

        if best_idx < 0:
            break
        selected.append(candidates[best_idx])
        selected_tokens.append(chunk_tokens[best_idx])
        covered_query_tokens |= (chunk_tokens[best_idx] & query_tokens)
        used_indices.add(best_idx)

    return selected


def listwise_configuration(
    *,
    lambda_redundancy: float = LAMBDA_REDUNDANCY,
    lambda_complement: float = LAMBDA_COMPLEMENT,
) -> dict[str, Any]:
    return {
        "algorithm_id": ALGORITHM_ID,
        "algorithm_version": ALGORITHM_VERSION,
        "lambda_redundancy": lambda_redundancy,
        "lambda_complement": lambda_complement,
        "selection_objective": (
            "score(chunk | selected) = normalised_CE_relevance "
            "- λ_redundancy * max_jaccard(chunk, selected) "
            "+ λ_complement * query_token_new_coverage(chunk, selected)"
        ),
        "query_conditioned_complementarity": True,
        "respects_same_document_multi_chunk": True,
        "no_fixed_document_quota": True,
        "no_ground_truth_labels": True,
    }
