"""Deterministic, artifact-first evaluation and error-analysis loop.

This module is deliberately independent from the production RAG path.  It consumes frozen
ground truth plus captured traces, never calls an LLM, and preserves unknown values rather
than inventing labels.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from rag_workbench.evaluation.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

DEFAULT_KS = (1, 3, 5, 10)
PASS_BEHAVIORS = {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"}


class PrimaryError(StrEnum):
    PASS = "PASS"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    RANKING_FAILURE = "RANKING_FAILURE"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    JUDGE_FALSE_NEGATIVE = "JUDGE_FALSE_NEGATIVE"
    JUDGE_FALSE_POSITIVE = "JUDGE_FALSE_POSITIVE"
    GENERATOR_INCOMPLETE = "GENERATOR_INCOMPLETE"
    GENERATOR_INCORRECT = "GENERATOR_INCORRECT"
    CITATION_FAILURE = "CITATION_FAILURE"
    FILTER_FAILURE = "FILTER_FAILURE"
    OTHER = "OTHER"
    UNRESOLVED = "UNRESOLVED"


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _first(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("cases", "traces", "results", "per_case"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"cannot find rows in {path}")


def load_frozen_cases(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load legacy or Phase-5 JSON/JSONL without changing the source dataset."""
    rows = _load_rows(path)
    cases: list[dict[str, Any]] = []
    for raw in rows:
        should_abstain = _first(raw, "should_abstain")
        answerable = _first(raw, "answerable", "expected_answerable")
        if answerable is None and should_abstain is not None:
            answerable = not bool(should_abstain)
        cases.append(
            {
                "case_id": _first(raw, "case_id", "query_id"),
                "question": raw.get("question"),
                "gold_answer": _first(raw, "gold_answer", "expected_answer"),
                "gold_document_ids": _list(
                    _first(
                        raw, "gold_document_ids", "expected_document_ids", "required_document_ids"
                    )
                ),
                "gold_chunk_ids": _list(
                    _first(raw, "gold_chunk_ids", "expected_chunk_ids", "required_chunk_ids")
                ),
                "answerable": answerable,
                "category": raw.get("category", "unknown"),
                "difficulty": raw.get("difficulty", "unknown"),
                "requires_acl": _flag(raw, "requires_acl", "acl"),
                "requires_tenant_filter": _flag(raw, "requires_tenant_filter", "tenant"),
                "requires_version_filter": _flag(raw, "requires_version_filter", "version"),
                "requires_multi_hop": _flag(raw, "requires_multi_hop", "multi_hop"),
                "required_facts": _list(raw.get("required_facts")),
                "raw_metadata": raw,
            }
        )
    missing_ids = [i for i, case in enumerate(cases) if not case["case_id"]]
    if missing_ids:
        raise ValueError(f"cases missing case_id/query_id at indexes {missing_ids[:5]}")
    duplicate_ids = [
        key for key, count in Counter(c["case_id"] for c in cases).items() if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"duplicate frozen case IDs: {duplicate_ids[:5]}")
    return cases, hashlib.sha256(path.read_bytes()).hexdigest()


def _flag(raw: dict[str, Any], field: str, marker: str) -> bool | None:
    if field in raw:
        return raw[field] if isinstance(raw[field], bool) else None
    category = str(raw.get("category", "")).lower()
    checks = {str(v).lower() for v in _list(raw.get("security_checks"))}
    return True if marker in category or marker in checks else None


def load_traces(path: Path, arm: str | None = None) -> list[dict[str, Any]]:
    rows = _load_rows(path)
    if arm is not None:
        rows = [row for row in rows if row.get("arm") == arm]
    return rows


def normalize_trace(case: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    candidate_ids = _list(_first(raw, "retrieved_candidate_ids", "candidate_ids", "retrieved_ids"))
    reranked_ids = _list(raw.get("reranked_ids"))
    top_ids = _list(
        _first(raw, "reranked_top_ids", "retrieved_top_k_ids", "top_k_ids", "chunk_ids")
    )
    candidate_docs = _list(
        _first(raw, "retrieved_candidate_document_ids", "candidate_document_ids")
    )
    reranked_docs = _list(raw.get("reranked_document_ids"))
    top_docs = _list(
        _first(
            raw,
            "topk_document_ids",
            "reranked_top_document_ids",
            "retrieved_document_ids",
        )
    )
    final_answer = _first(raw, "final_answer", "answer", default="") or ""
    abstained = bool(_first(raw, "abstained", default=raw.get("status") == "abstained"))
    citation_ids = _list(raw.get("citation_ids"))
    answer_correct = _answer_correct(raw)
    citation_exists = None if not citation_ids else all(cid in set(top_ids) for cid in citation_ids)
    citation_valid = _first(
        raw, "citation_valid", "citation_validity_pass", default=citation_exists
    )
    topk_hit = _hit_or_none(top_ids, top_docs, case)
    legacy_recall = raw.get("retrieval_recall5")
    if topk_hit is None and case["gold_document_ids"] and isinstance(legacy_recall, (int, float)):
        topk_hit = legacy_recall > 0
    trace = {
        "case_id": case["case_id"],
        "question": case["question"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "answerable": case["answerable"],
        "retrieved_candidate_ids": candidate_ids,
        "retrieved_candidate_document_ids": candidate_docs,
        "reranked_ids": reranked_ids,
        "reranked_document_ids": reranked_docs,
        "reranked_top_ids": top_ids,
        "reranked_top_document_ids": top_docs,
        "final_answer": final_answer,
        "abstained": abstained,
        "citation_ids": citation_ids,
        "cited_document_ids": _list(raw.get("cited_document_ids")),
        "latency_ms": _first(raw, "latency_ms", "total_latency_ms", default=0) or 0,
        "token_usage": raw.get("token_usage") or _legacy_tokens(raw),
        "api_calls": _first(raw, "api_calls", "external_calls"),
        "retrieval_hit": _hit_or_none(candidate_ids, candidate_docs, case),
        "topk_hit": topk_hit,
        "answer_correct": answer_correct,
        "citation_present": bool(citation_ids),
        "citation_id_exists": citation_exists,
        "citation_valid": citation_valid,
        "citation_gold_chunk_match": _overlap_or_none(citation_ids, case["gold_chunk_ids"]),
        "citation_gold_document_match": _overlap_or_none(
            _list(raw.get("cited_document_ids")), case["gold_document_ids"]
        ),
        "stage_trace": raw.get("stage_trace") or _legacy_stage_trace(raw),
        "behavior": raw.get("behavior"),
        "judge_answerable": raw.get("judge_answerable"),
        "evidence_sufficient": _first(raw, "evidence_sufficient", "top5_complete_evidence"),
        "filter_failure": raw.get("filter_failure"),
        "facts_missing": _list(raw.get("facts_missing")),
        "document_count": len(case["gold_document_ids"]) or None,
        "raw_trace": raw,
        "execution_error": raw.get("execution_error"),
    }
    trace["error_evidence"] = failure_evidence(case, trace)
    trace["primary_error"] = classify_failure(case, trace).value
    trace["unresolved_reason"] = unresolved_reason(case, trace)
    trace["secondary_tags"] = secondary_tags(case, trace)
    return trace


def _answer_correct(raw: dict[str, Any]) -> bool | None:
    explicit = raw.get("answer_correct")
    if isinstance(explicit, bool):
        return explicit
    behavior = raw.get("behavior")
    if behavior in PASS_BEHAVIORS:
        return True
    if behavior in {
        "INCORRECT_ANSWER",
        "INCORRECT_ABSTENTION",
        "UNSUPPORTED_ANSWER",
    }:
        return False
    return None


def _legacy_tokens(raw: dict[str, Any]) -> dict[str, int]:
    values = {
        "prompt": raw.get("prompt_tokens") or raw.get("judge_prompt_tokens"),
        "completion": raw.get("completion_tokens") or raw.get("judge_completion_tokens"),
    }
    return {key: int(value) for key, value in values.items() if isinstance(value, (int, float))}


def _legacy_stage_trace(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: raw[key]
        for key in (
            "acl_filter",
            "tenant_filter",
            "version_filter",
            "dense",
            "bm25",
            "rrf",
            "cross_encoder",
            "judge_answerable",
            "judge_supporting_chunk_ids",
            "generation_path",
            "status",
        )
        if key in raw
    }


def _overlap_or_none(actual: list[str], gold: list[str]) -> bool | None:
    if not gold:
        return None
    return bool(set(actual) & set(gold))


def _hit_or_none(ids: list[str], docs: list[str], case: dict[str, Any]) -> bool | None:
    chunk_gold = case["gold_chunk_ids"]
    doc_gold = case["gold_document_ids"]
    if chunk_gold and ids:
        return set(chunk_gold) <= set(ids)
    if doc_gold and docs:
        return set(doc_gold) <= set(docs)
    return None


def failure_evidence(case: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Persist the exact stage facts used for attribution."""
    gold_docs = set(case["gold_document_ids"])
    stages = trace.get("stage_trace") or {}
    filters = stages.get("filters") or {}
    filter_input_docs = set(_list(filters.get("input_document_ids")))
    filter_output_docs = set(_list(filters.get("output_document_ids")))
    candidates = set(trace["retrieved_candidate_document_ids"])
    reranked = set(trace["reranked_document_ids"])
    topk = set(trace["reranked_top_document_ids"])
    return {
        "gold_document_ids": sorted(gold_docs),
        "present_before_filters": _contains_or_none(filter_input_docs, gold_docs),
        "present_after_filters": _contains_or_none(filter_output_docs, gold_docs),
        "present_in_candidates": _contains_or_none(candidates, gold_docs),
        "present_after_reranking": _contains_or_none(reranked, gold_docs),
        "present_in_top_k": _contains_or_none(topk, gold_docs),
        "missing_from_candidates": sorted(gold_docs - candidates) if candidates else [],
        "missing_from_top_k": sorted(gold_docs - topk) if topk else [],
        "filter_reasons": filters.get("reasons") or {},
        "execution_error": trace.get("execution_error"),
    }


def _contains_or_none(actual: set[str], expected: set[str]) -> bool | None:
    if not expected or not actual:
        return None
    return expected <= actual


def classify_failure(case: dict[str, Any], trace: dict[str, Any]) -> PrimaryError:
    execution_error = trace.get("execution_error")
    if execution_error and str(execution_error).startswith("RAG_EXECUTION_FAILURE"):
        return PrimaryError.UNRESOLVED
    if trace.get("behavior") in PASS_BEHAVIORS or trace.get("answer_correct") is True:
        return PrimaryError.PASS
    evidence = trace.get("error_evidence") or failure_evidence(case, trace)
    if trace.get("filter_failure") is True or (
        case.get("answerable") is True
        and evidence["present_before_filters"] is True
        and evidence["present_after_filters"] is False
    ):
        return PrimaryError.FILTER_FAILURE
    if evidence["present_after_filters"] is True and evidence["present_in_candidates"] is False:
        return PrimaryError.RETRIEVAL_FAILURE
    if evidence["present_in_candidates"] is True and evidence["present_in_top_k"] is False:
        return PrimaryError.RANKING_FAILURE
    if trace.get("retrieval_hit") is False:
        return PrimaryError.RETRIEVAL_FAILURE
    if trace.get("retrieval_hit") is True and trace.get("topk_hit") is False:
        return PrimaryError.RANKING_FAILURE
    if trace.get("evidence_sufficient") is False:
        return PrimaryError.EVIDENCE_INSUFFICIENT
    if execution_error:
        return PrimaryError.UNRESOLVED
    answerable = case.get("answerable")
    if answerable is True and trace.get("abstained"):
        if (
            trace.get("evidence_sufficient") in (True, 1, 1.0)
            and trace.get("judge_answerable") is False
        ):
            return PrimaryError.JUDGE_FALSE_NEGATIVE
        return PrimaryError.OTHER
    if answerable is False and not trace.get("abstained"):
        return PrimaryError.JUDGE_FALSE_POSITIVE
    if trace.get("citation_valid") is False:
        return PrimaryError.CITATION_FAILURE
    if trace.get("facts_missing") or trace.get("behavior") == "UNSUPPORTED_ANSWER":
        return PrimaryError.GENERATOR_INCOMPLETE
    if trace.get("answer_correct") is False:
        return PrimaryError.GENERATOR_INCORRECT
    return PrimaryError.UNRESOLVED


def unresolved_reason(case: dict[str, Any], trace: dict[str, Any]) -> str | None:
    if trace["primary_error"] not in {
        PrimaryError.OTHER.value,
        PrimaryError.UNRESOLVED.value,
    }:
        return None
    if trace.get("execution_error"):
        return str(trace["execution_error"])
    evidence = trace.get("error_evidence") or {}
    unknown = [
        name
        for name in (
            "present_before_filters",
            "present_after_filters",
            "present_in_candidates",
            "present_in_top_k",
        )
        if evidence.get(name) is None
    ]
    if unknown:
        return "MISSING_STAGE_TRACE:" + ",".join(unknown)
    if not case["gold_document_ids"] and not case["gold_chunk_ids"]:
        return "INSUFFICIENT_GOLD_LABEL"
    return "NO_DETERMINISTIC_RULE_MATCHED"


def secondary_tags(case: dict[str, Any], trace: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    count = len(case["gold_document_ids"])
    if count == 1:
        tags.append("SINGLE_DOCUMENT")
    elif count == 2:
        tags.append("MULTI_DOCUMENT")
    elif count == 3:
        tags.extend(("MULTI_DOCUMENT", "THREE_DOCUMENT"))
    elif count > 3:
        tags.append("MULTI_DOCUMENT")
    for field, tag in (
        ("requires_version_filter", "VERSION_SENSITIVE"),
        ("requires_acl", "ACL_SENSITIVE"),
        ("requires_tenant_filter", "TENANT_SENSITIVE"),
    ):
        if case.get(field) is True:
            tags.append(tag)
    if (
        not case["gold_chunk_ids"]
        and not case["gold_document_ids"]
        and case.get("answerable") is not False
    ):
        tags.append("INSUFFICIENT_GOLD_LABEL")
    if trace.get("raw_trace", {}).get("conflicting_documents") is True:
        tags.append("CONFLICTING_DOCUMENTS")
    if not tags:
        tags.append("UNCLASSIFIED")
    return tags


def evaluate(
    cases: list[dict[str, Any]], raw_traces: list[dict[str, Any]], ks: tuple[int, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    by_id = {_first(row, "case_id", "query_id"): row for row in raw_traces}
    missing = [case["case_id"] for case in cases if case["case_id"] not in by_id]
    if missing:
        raise ValueError(f"missing traces for {len(missing)} frozen cases: {missing[:5]}")
    extra = sorted(set(by_id) - {case["case_id"] for case in cases})
    if extra:
        raise ValueError(f"trace case IDs are not in frozen dataset: {extra[:5]}")
    traces = [normalize_trace(case, by_id[case["case_id"]]) for case in cases]
    metrics = aggregate_metrics(cases, traces, ks)
    census = failure_census(traces)
    slices = slice_analysis(cases, traces, min_size=5)
    return traces, metrics, census, slices


def aggregate_metrics(
    cases: list[dict[str, Any]], traces: list[dict[str, Any]], ks: tuple[int, ...] = DEFAULT_KS
) -> dict[str, Any]:
    total = len(traces)
    answered = [t for t in traces if not t["abstained"]]
    correct_answered = [t for t in answered if t["answer_correct"] is True]
    correct = [t for t in traces if t["answer_correct"] is True]
    known = [t for t in traces if t["answer_correct"] is not None]
    answerable = [t for t in traces if t["answerable"] is True]
    unanswerable = [t for t in traces if t["answerable"] is False]
    false_positive = sum(t["answer_correct"] is False for t in answered)
    false_negative = sum(t["abstained"] and t["answerable"] is True for t in traces)
    precision = len(correct_answered) / len(answered) if answered else None
    # Preserve FINAL_E2E_SCORER_V2: incorrect answers are FP and abstentions are FN.
    recall_denominator = len(correct_answered) + false_negative
    recall = len(correct_answered) / recall_denominator if recall_denominator else None
    f1 = _f1(precision, recall)
    result: dict[str, Any] = {
        "case_count": total,
        "scored_case_count": len(known),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": len(correct) / len(known) if known else None,
        "answer_rate": len(answered) / total if total else None,
        "abstention_rate": (total - len(answered)) / total if total else None,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "answered_correctly": len(correct_answered),
        "answered_incorrectly": false_positive,
        "correct_abstention": sum(
            t["abstained"] and t["answerable"] is False and t["answer_correct"] is True
            for t in traces
        ),
        "incorrect_abstention": false_negative,
        "unsupported_answers": sum(t.get("behavior") == "UNSUPPORTED_ANSWER" for t in traces),
        "citation_present_rate": _mean([float(t["citation_present"]) for t in answered]),
        "citation_id_exists_rate": _known_bool_mean(t["citation_id_exists"] for t in traces),
        "citation_validity": _known_bool_mean(t["citation_valid"] for t in traces),
        "citation_gold_chunk_match": _known_bool_mean(
            t["citation_gold_chunk_match"] for t in traces
        ),
        "citation_gold_document_match": _known_bool_mean(
            t["citation_gold_document_match"] for t in traces
        ),
    }
    for k in ks:
        chunk_rows = [(t, c) for t, c in zip(traces, cases, strict=True) if c["gold_chunk_ids"]]
        doc_rows = [
            (t, c)
            for t, c in zip(traces, cases, strict=True)
            if c["gold_document_ids"] and t["reranked_top_document_ids"]
        ]
        result[f"hit_at_{k}"] = _mean(
            [hit_at_k(t["reranked_top_ids"], set(c["gold_chunk_ids"]), k) for t, c in chunk_rows]
        )
        result[f"recall_at_{k}"] = _mean(
            [recall_at_k(t["reranked_top_ids"], set(c["gold_chunk_ids"]), k) for t, c in chunk_rows]
        )
        result[f"ndcg_at_{k}"] = _mean(
            [ndcg_at_k(t["reranked_top_ids"], set(c["gold_chunk_ids"]), k) for t, c in chunk_rows]
        )
        result[f"gold_document_hit_at_{k}"] = _mean(
            [
                hit_at_k(t["reranked_top_document_ids"], set(c["gold_document_ids"]), k)
                for t, c in doc_rows
            ]
        )
        result[f"gold_document_recall_at_{k}"] = _mean(
            [
                recall_at_k(t["reranked_top_document_ids"], set(c["gold_document_ids"]), k)
                for t, c in doc_rows
            ]
        )
    rr = [
        reciprocal_rank(t["reranked_top_ids"], set(c["gold_chunk_ids"]))
        for t, c in zip(traces, cases, strict=True)
        if c["gold_chunk_ids"]
    ]
    result["mrr"] = _mean(rr)
    result["gold_document_mrr"] = _mean(
        [
            reciprocal_rank(t["reranked_top_document_ids"], set(c["gold_document_ids"]))
            for t, c in zip(traces, cases, strict=True)
            if c["gold_document_ids"] and t["reranked_top_document_ids"]
        ]
    )
    candidate_recalls = [
        recall_at_k(
            t["retrieved_candidate_ids"],
            set(c["gold_chunk_ids"]),
            len(t["retrieved_candidate_ids"]),
        )
        for t, c in zip(traces, cases, strict=True)
        if c["gold_chunk_ids"] and t["retrieved_candidate_ids"]
    ]
    result["candidate_recall"] = _mean(candidate_recalls)
    result["top_k_recall"] = result.get(f"recall_at_{max(ks)}")
    result["retrieval_recall"] = result.get("gold_document_recall_at_5")
    result["gold_chunk_labeled_cases"] = sum(bool(c["gold_chunk_ids"]) for c in cases)
    result["gold_document_labeled_cases"] = sum(bool(c["gold_document_ids"]) for c in cases)
    result["unknown_answerability_cases"] = total - len(answerable) - len(unanswerable)
    result["cost"] = cost_summary(traces)
    return result


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _known_bool_mean(values: Any) -> float | None:
    known = [float(v) for v in values if isinstance(v, bool)]
    return _mean(known)


def cost_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    token_totals: Counter[str] = Counter()
    for trace in traces:
        token_totals.update({k: int(v) for k, v in trace["token_usage"].items()})
    recorded_calls = [int(t["api_calls"]) for t in traces if t["api_calls"] is not None]
    rag_calls = sum(recorded_calls) if len(recorded_calls) == len(traces) else None
    return {
        "rag_execution_api_calls": rag_calls,
        "rag_execution_tokens": dict(token_totals),
        "deterministic_evaluation_api_calls": 0,
        "semantic_judge_api_calls": 0,
        "semantic_judge_tokens": {},
        "total_api_calls": rag_calls,
    }


def failure_census(traces: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(t["primary_error"] for t in traces)
    total = len(traces)
    details = []
    for trace in traces:
        if trace["primary_error"] not in {PrimaryError.OTHER.value, PrimaryError.UNRESOLVED.value}:
            continue
        details.append(
            {
                "case_id": trace["case_id"],
                "question_type": trace["category"],
                "retrieval_status": _status(trace["retrieval_hit"]),
                "ranking_status": _status(trace["topk_hit"]),
                "judge_decision": trace["judge_answerable"],
                "generator_output": trace["final_answer"],
                "citation_state": {
                    "present": trace["citation_present"],
                    "valid": trace["citation_valid"],
                },
                "version_sensitivity": "VERSION_SENSITIVE" in trace["secondary_tags"],
                "document_count": trace["document_count"],
                "secondary_tags": trace["secondary_tags"],
                "unresolved_reason": trace.get("unresolved_reason"),
                "error_evidence": trace.get("error_evidence"),
            }
        )
    return {
        "total": total,
        "counts": {key: counts.get(key, 0) for key in PrimaryError},
        "percentages": {key: counts.get(key, 0) / total if total else 0.0 for key in PrimaryError},
        "other_unresolved_cases": details,
    }


def _status(value: bool | None) -> str:
    return "HIT" if value is True else "MISS" if value is False else "UNKNOWN"


def slice_analysis(
    cases: list[dict[str, Any]], traces: list[dict[str, Any]], min_size: int = 5
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, (case, _trace) in enumerate(zip(cases, traces, strict=True)):
        groups[f"category:{case['category']}"].append(i)
        if case["difficulty"] != "unknown":
            groups[f"difficulty:{case['difficulty']}"].append(i)
        doc_count = len(case["gold_document_ids"])
        if doc_count == 1:
            groups["single_document"].append(i)
        elif doc_count > 1:
            groups["multi_document"].append(i)
        for field, name in (
            ("requires_version_filter", "version_sensitive"),
            ("requires_acl", "acl"),
            ("requires_tenant_filter", "tenant"),
        ):
            if case[field] is True:
                groups[name].append(i)
        if case["answerable"] is True:
            groups["answerable"].append(i)
        elif case["answerable"] is False:
            groups["unanswerable"].append(i)
    output: dict[str, Any] = {}
    for name, indexes in sorted(groups.items()):
        selected = [traces[i] for i in indexes]
        answered = [t for t in selected if not t["abstained"]]
        correct_answered = sum(t["answer_correct"] is True for t in answered)
        incorrect_abstentions = sum(t["abstained"] and t["answerable"] is True for t in selected)
        precision = correct_answered / len(answered) if answered else None
        recall_denominator = correct_answered + incorrect_abstentions
        recall = correct_answered / recall_denominator if recall_denominator else None
        known = [t for t in selected if t["answer_correct"] is not None]
        output[name] = {
            "case_count": len(indexes),
            "small_sample": len(indexes) < min_size,
            "minimum_recommended_size": min_size,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "accuracy": sum(t["answer_correct"] is True for t in known) / len(known)
            if known
            else None,
            "failure_counts": dict(Counter(t["primary_error"] for t in selected)),
        }
    return {"min_slice_size": min_size, "slices": output}


@dataclass(frozen=True)
class Comparison:
    payload: dict[str, Any]


def compare_results(
    baseline: dict[str, Any], candidate: dict[str, Any], gate: dict[str, Any]
) -> Comparison:
    if baseline.get("dataset_hash") != candidate.get("dataset_hash"):
        return Comparison(
            {
                "decision": "MANUAL_REVIEW",
                "reasons": [
                    "dataset hashes differ; Control and Candidate must use the same frozen dataset"
                ],
                "metric_comparison": {},
                "case_comparison": {},
            }
        )
    bmetrics, cmetrics = baseline["metrics"], candidate["metrics"]
    metric_names = sorted(set(bmetrics) & set(cmetrics) - {"cost"})
    comparisons = {}
    for name in metric_names:
        b, c = bmetrics[name], cmetrics[name]
        if isinstance(b, (int, float)) and isinstance(c, (int, float)) and not isinstance(b, bool):
            comparisons[name] = {"baseline": b, "candidate": c, "delta": c - b}
    bcases = {r["case_id"]: r for r in baseline["traces"]}
    ccases = {r["case_id"]: r for r in candidate["traces"]}
    fixed, regressed, unchanged, new_failures = [], [], [], []
    for case_id in sorted(bcases.keys() & ccases.keys()):
        bp = bcases[case_id]["primary_error"] == PrimaryError.PASS
        cp = ccases[case_id]["primary_error"] == PrimaryError.PASS
        if not bp and cp:
            fixed.append(case_id)
        elif bp and not cp:
            regressed.append(case_id)
            new_failures.append(case_id)
        elif not bp and not cp:
            unchanged.append(case_id)
    reasons = apply_gate(comparisons, gate)
    unavailable = any("unavailable" in reason for reason in reasons)
    decision = "MANUAL_REVIEW" if unavailable else "REJECT" if reasons else "PROMOTE"
    failure_comparison = _count_comparison(
        baseline.get("failure_census", {}).get("counts", {}),
        candidate.get("failure_census", {}).get("counts", {}),
    )
    slice_comparison = _slice_comparison(
        baseline.get("slice_analysis", {}).get("slices", {}),
        candidate.get("slice_analysis", {}).get("slices", {}),
    )
    return Comparison(
        {
            "decision": decision,
            "reasons": reasons,
            "metric_comparison": comparisons,
            "case_comparison": {
                "fixed_cases": fixed,
                "regressed_cases": regressed,
                "unchanged_failures": unchanged,
                "new_failures": new_failures,
            },
            "failure_comparison": failure_comparison,
            "slice_comparison": slice_comparison,
            "gate": gate,
        }
    )


def _count_comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "baseline": baseline.get(name, 0),
            "candidate": candidate.get(name, 0),
            "delta": candidate.get(name, 0) - baseline.get(name, 0),
        }
        for name in sorted(set(baseline) | set(candidate))
    }


def _slice_comparison(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in sorted(set(baseline) & set(candidate)):
        metrics: dict[str, Any] = {}
        for metric in ("accuracy", "precision", "recall", "f1"):
            before = baseline[name].get(metric)
            after = candidate[name].get(metric)
            metrics[metric] = {
                "baseline": before,
                "candidate": after,
                "delta": (
                    after - before
                    if isinstance(before, (int, float)) and isinstance(after, (int, float))
                    else None
                ),
            }
        output[name] = metrics
    return output


def apply_gate(comparisons: dict[str, dict[str, float]], gate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    rules = gate.get("promotion_gate", gate)
    for metric, constraints in rules.items():
        measured = comparisons.get(metric)
        if measured is None:
            reasons.append(f"{metric}: required gate metric is unavailable")
            continue
        if "min" in constraints and measured["candidate"] < constraints["min"]:
            reasons.append(
                f"{metric}: candidate {measured['candidate']:.6f} < min {constraints['min']:.6f}"
            )
        if "min_delta" in constraints and measured["delta"] < constraints["min_delta"]:
            delta = measured["delta"]
            threshold = constraints["min_delta"]
            reasons.append(f"{metric}: delta {delta:+.6f} < min_delta {threshold:+.6f}")
        if (
            "max_regression" in constraints
            and measured["candidate"] - measured["baseline"] > constraints["max_regression"]
        ):
            reasons.append(
                f"{metric}: increased {measured['baseline']:.6f} -> {measured['candidate']:.6f}"
            )
    return reasons


def load_gate(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_run(
    *,
    output_dir: Path,
    experiment_id: str,
    dataset_path: Path,
    dataset_hash: str,
    traces_path: Path,
    traces: list[dict[str, Any]],
    metrics: dict[str, Any],
    census: dict[str, Any],
    slices: dict[str, Any],
    semantic_judge: str,
    ks: tuple[int, ...],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": "deterministic-eval-v1",
        "experiment_id": experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_hash": dataset_hash,
        "trace_source": str(traces_path),
        "ks": list(ks),
        "semantic_judge": semantic_judge,
        "semantic_judge_note": "OFF; deterministic evaluation made zero judge calls"
        if semantic_judge == "off"
        else "semantic results must be supplied by a separately reviewed adapter",
    }
    results = {
        **metadata,
        "metrics": metrics,
        "failure_census": census,
        "slice_analysis": slices,
        "traces": traces,
    }
    _write_json(output_dir / "results.json", results)
    _write_jsonl(output_dir / "traces.jsonl", traces)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "failure_census.json", census)
    _write_json(output_dir / "slice_analysis.json", slices)
    _write_json(
        output_dir / "regression.json",
        {"status": "NOT_RUN", "reason": "run compare_eval.py with a frozen baseline"},
    )
    (output_dir / "report.md").write_text(
        render_report(metadata, metrics, census, slices), encoding="utf-8"
    )
    return results


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def render_report(
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    census: dict[str, Any],
    slices: dict[str, Any],
) -> str:
    lines = [
        f"# Deterministic Eval: {metadata['experiment_id']}",
        "",
        f"Frozen dataset SHA-256: `{metadata['dataset_hash']}`",
        f"Semantic judge: `{metadata['semantic_judge']}`",
        "",
        "## Overall metrics",
        "",
    ]
    for key in ("precision", "recall", "f1", "accuracy", "answer_rate", "abstention_rate"):
        value = metrics.get(key)
        lines.append(
            f"- {key}: `{value:.6f}`" if isinstance(value, float) else f"- {key}: `{value}`"
        )
    lines.extend(["", "## Failure census", ""])
    for key, count in census["counts"].items():
        lines.append(f"- {key}: {count} / {census['total']} ({census['percentages'][key]:.2%})")
    lines.extend(["", "## Slice analysis", ""])
    for name, item in slices["slices"].items():
        note = " (small sample)" if item["small_sample"] else ""
        lines.append(f"- {name}: n={item['case_count']}, F1={item['f1']}{note}")
    cost = metrics["cost"]
    lines.extend(
        [
            "",
            "## Cost",
            "",
            f"- RAG execution API calls recorded in traces: {cost['rag_execution_api_calls']}",
            "- Deterministic evaluation API calls: 0",
            "- Semantic judge API calls: 0",
            "",
        ]
    )
    return "\n".join(lines)
