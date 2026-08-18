from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from rag_workbench.evaluation.retrieval_dataset import RetrievalGroundTruthCase
from rag_workbench.evaluation.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_workbench.retrieval.vector_search import RetrievalResult


def first_document_rank(results: list[RetrievalResult], document_id: str) -> int | None:
    return next((item.rank for item in results if item.document_id == document_id), None)


def retrieval_case_metrics(
    case: RetrievalGroundTruthCase,
    results: list[RetrievalResult],
) -> dict[str, Any]:
    documents = [item.document_id for item in results]
    relevant = set(case.required_document_ids)
    retrieved_relevant = relevant & set(documents[:5])
    version_correct = all(
        any(
            item.document_id == document_id and item.version == version
            for item in results[:5]
        )
        for document_id, version in case.required_version_ids.items()
    )
    forbidden_ranks = {
        document_id: first_document_rank(results, document_id)
        for document_id in case.forbidden_document_ids
    }
    required_ranks = {
        document_id: first_document_rank(results, document_id)
        for document_id in case.required_document_ids
    }
    preferred_success = bool(relevant) and all(rank is not None for rank in required_ranks.values())
    if preferred_success and forbidden_ranks:
        highest_required = max(rank for rank in required_ranks.values() if rank is not None)
        preferred_success = all(
            rank is None or highest_required < rank for rank in forbidden_ranks.values()
        )
    return {
        "hit_at_5": hit_at_k(documents, relevant, 5) if relevant else None,
        "recall_at_5": recall_at_k(documents, relevant, 5) if relevant else None,
        "reciprocal_rank": reciprocal_rank(documents, relevant) if relevant else None,
        "ndcg_at_5": ndcg_at_k(documents, relevant, 5) if relevant else None,
        "required_evidence_count": len(relevant),
        "required_evidence_retrieved": len(retrieved_relevant),
        "required_evidence_recall_at_5": (
            len(retrieved_relevant) / len(relevant) if relevant else None
        ),
        "all_required_evidence_coverage_at_5": (
            float(relevant <= set(documents[:5])) if case.expected_answerability else None
        ),
        "version_correct": float(version_correct),
        "preferred_source_success": float(preferred_success) if relevant else None,
        "unauthorized_result_exposure": (
            sum(item.document_id in set(case.forbidden_document_ids) for item in results)
            if case.expected_access_behavior == "EXCLUDE_FORBIDDEN"
            else 0
        ),
        "required_document_ranks": required_ranks,
        "forbidden_document_ranks": forbidden_ranks,
    }


def aggregate_retrieval_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in case_results if item["expected_answerability"]]

    def average(name: str, rows: list[dict[str, Any]] = answerable) -> float:
        values = [row["metrics"][name] for row in rows if row["metrics"][name] is not None]
        return mean(values) if values else 0.0

    two = [item for item in answerable if item["category"] == "multidoc_two"]
    three = [item for item in answerable if item["category"] == "multidoc_three"]
    exact = [item for item in answerable if item["category"] == "exact_identifier"]
    version = [item for item in answerable if item["category"] == "version_region"]
    duplicate = [item for item in answerable if item["category"] == "near_duplicate"]
    unauthorized = sum(
        item["metrics"]["unauthorized_result_exposure"] for item in case_results
    )
    return {
        "hit_at_5": average("hit_at_5"),
        "recall_at_5": average("recall_at_5"),
        "mrr": average("reciprocal_rank"),
        "ndcg_at_5": average("ndcg_at_5"),
        "required_evidence_recall_at_5": average("required_evidence_recall_at_5"),
        "all_required_evidence_coverage_at_5": average(
            "all_required_evidence_coverage_at_5"
        ),
        "two_document_coverage_at_5": average(
            "all_required_evidence_coverage_at_5", two
        ),
        "three_document_coverage_at_5": average(
            "all_required_evidence_coverage_at_5", three
        ),
        "exact_identifier_recall_at_5": average("required_evidence_recall_at_5", exact),
        "version_sensitive_recall_at_5": average("required_evidence_recall_at_5", version),
        "version_correctness": average("version_correct", version),
        "near_duplicate_preferred_source_success": average(
            "preferred_source_success", duplicate
        ),
        "acl_safety": float(unauthorized == 0),
        "unauthorized_result_exposure": unauthorized,
    }


def category_retrieval_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in case_results:
        grouped[item["category"]].append(item)
    return {
        category: aggregate_retrieval_metrics(rows)
        for category, rows in sorted(grouped.items())
    }
