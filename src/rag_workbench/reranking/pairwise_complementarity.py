from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag_workbench.reranking.document_diversity import (
    assert_no_evaluation_label_leakage,
    select_max_2_chunks_per_document_top5,
)

PAIRWISE_COMPLEMENTARITY_RERANK_V1 = "PAIRWISE_COMPLEMENTARITY_RERANK v1.0"
PAIRWISE_COMPLEMENTARITY_RERANK_V1_CONFIG_HASH = (
    "527afb76a0226018e158291c0212e16cfdc31e0d9990cfd4686d72c1d3df69cd"
)


def pairwise_complementarity_rerank_top5(
    ranked_candidates: Sequence[Any],
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Frozen Candidate-B Top-5 selection.

    Runtime-safe: selection is purely structural over the frozen
    cross-encoder-ranked candidate list and does *not* use any evaluation
    labels (expected answers, required ids, etc).

    The current qualified Candidate-B mapping uses the frozen rule:
    "at most two chunks per canonical document_id", preserving input order.
    """
    # Defensive check: ensure callers didn't smuggle evaluation labels into
    # the ranking candidates (selection must be label-free).
    for item in ranked_candidates:
        assert_no_evaluation_label_leakage(item)

    selected = select_max_2_chunks_per_document_top5(
        ranked_candidates, top_k=top_k, max_chunks_per_document=2
    )
    # The selector returns the original dicts in input order. We only add a
    # retrieval_source marker so downstream traces can distinguish arms.
    for item in selected:
        if not isinstance(item, dict):
            raise TypeError(
                "pairwise_complementarity_rerank_top5 expects dict candidates; "
                f"got {type(item).__name__}"
            )

    return [
        {**item, "retrieval_source": "pairwise_complementarity_rerank_top5"}
        for item in selected
    ]

