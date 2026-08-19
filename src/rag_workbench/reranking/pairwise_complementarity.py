# ruff: noqa: E501
"""Pairwise complementarity rerank.

Preserves original Cross-Encoder relevance as primary signal but applies
stronger pairwise redundancy penalties with query-conditioned complementarity
scoring during greedy evidence-set construction.

Key differences from listwise_evidence_set v1.0:
  - Stronger redundancy penalty (λ=0.45 vs 0.30)
  - Stronger complement bonus (λ=0.35 vs 0.25)
  - Pairwise token-Jaccard between the *query-relevant* token subsets of
    chunk pairs, not the full chunk text. This penalises overlap in
    query-relevant content more than incidental textual similarity.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from rag_workbench.reranking.document_diversity import (
    assert_no_evaluation_label_leakage,
)
from rag_workbench.reranking.listwise_evidence_set import (
    _chunk_score,
    _chunk_text,
    _normalise_scores,
)

ALGORITHM_ID = "PAIRWISE_COMPLEMENTARITY_RERANK"
ALGORITHM_VERSION = "1.0"

LAMBDA_REDUNDANCY = 0.45
LAMBDA_COMPLEMENT = 0.35

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.casefold()))


def _query_relevant_tokens(chunk_tokens: set[str], query_tokens: set[str]) -> set[str]:
    return chunk_tokens & query_tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def select_pairwise_complementarity_top5(
    ranked_candidates: Sequence[Any],
    query: str,
    *,
    top_k: int = 5,
    lambda_redundancy: float = LAMBDA_REDUNDANCY,
    lambda_complement: float = LAMBDA_COMPLEMENT,
) -> list[Any]:
    """Greedy pairwise complementarity selection.

    Keeps CE relevance as primary signal. Penalises redundancy based on
    query-relevant token overlap with already-selected chunks. Rewards
    complementary evidence based on uncovered query tokens.

    Same-document chunks with distinct query-relevant content are preserved.
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
    chunk_qr_tokens = [ct & query_tokens for ct in chunk_tokens]

    selected: list[Any] = []
    selected_qr_tokens: list[set[str]] = []
    selected_all_tokens: list[set[str]] = []
    covered_query_tokens: set[str] = set()
    used_indices: set[int] = set()

    for _ in range(min(top_k, len(candidates))):
        best_idx = -1
        best_score = float("-inf")
        for idx in range(len(candidates)):
            if idx in used_indices:
                continue
            relevance = norm_scores[idx]
            qr = chunk_qr_tokens[idx]
            ct = chunk_tokens[idx]

            if selected_qr_tokens:
                redundancy = max(_jaccard(qr, sq) for sq in selected_qr_tokens)
            else:
                redundancy = 0.0

            if selected_all_tokens:
                text_redundancy = max(_jaccard(ct, st) for st in selected_all_tokens)
            else:
                text_redundancy = 0.0

            combined_redundancy = 0.6 * redundancy + 0.4 * text_redundancy

            uncovered = query_tokens - covered_query_tokens
            new_coverage = len(ct & uncovered) / len(uncovered) if uncovered else 0.0

            marginal = (
                relevance
                - lambda_redundancy * combined_redundancy
                + lambda_complement * new_coverage
            )

            if marginal > best_score:
                best_score = marginal
                best_idx = idx

        if best_idx < 0:
            break
        selected.append(candidates[best_idx])
        selected_qr_tokens.append(chunk_qr_tokens[best_idx])
        selected_all_tokens.append(chunk_tokens[best_idx])
        covered_query_tokens |= (chunk_tokens[best_idx] & query_tokens)
        used_indices.add(best_idx)

    return selected


def pairwise_configuration(
    *,
    lambda_redundancy: float = LAMBDA_REDUNDANCY,
    lambda_complement: float = LAMBDA_COMPLEMENT,
) -> dict[str, Any]:
    return {
        "algorithm_id": ALGORITHM_ID,
        "algorithm_version": ALGORITHM_VERSION,
        "lambda_redundancy": lambda_redundancy,
        "lambda_complement": lambda_complement,
        "redundancy_mode": "pairwise_query_relevant_token_jaccard",
        "combined_redundancy_weights": {"query_relevant": 0.6, "full_text": 0.4},
        "selection_objective": (
            "score(chunk | selected) = normalised_CE_relevance "
            "- λ_redundancy * (0.6·qr_jaccard + 0.4·text_jaccard) "
            "+ λ_complement * query_token_new_coverage"
        ),
        "query_conditioned_complementarity": True,
        "respects_same_document_multi_chunk": True,
        "no_fixed_document_quota": True,
        "no_ground_truth_labels": True,
    }
