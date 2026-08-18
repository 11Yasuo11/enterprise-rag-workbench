"""Diagnostic rollup for V3 Phase 1. Does not mutate frozen V2 rows."""

from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Any

from rag_workbench.experiments.reranker_e2e_benchmark import _percentile
from rag_workbench.experiments.sol_judge_e2e_benchmark import official_token_cost
from rag_workbench.experiments.v2_sufficiency_fn import SOL_MODEL

NO_GO = "NO_GO_FOR_UNSEEN_V3_PHASE1"
NO_GO_ALIASES = frozenset({NO_GO, "NO_GO_FOR_UNSEEN_EXPERIMENT"})
CANNOT_DRAFT = "CANNOT_DRAFT_SUPPORTED_ANSWER"
DIAGNOSTIC_GO_POLICY = {
    "historical_judge_fn_rescues_min": 6,
    "historical_should_abstain_false_positive_recoveries": 0,
    "unsupported_recovered_answers": 0,
    "unauthorized_evidence_used": 0,
    "unauthorized_supporting_ids": 0,
    "invalid_citation_ids": 0,
    "version_violations": 0,
}


def diagnostic_is_go(value: str | None) -> bool:
    return value == "GO"


def classify_recovery_failure(
    typed_failure: str | None, claim_states: list[str] | None = None
) -> str | None:
    if not typed_failure:
        return None
    if typed_failure in {CANNOT_DRAFT, "DRAFT_CANNOT_ANSWER"}:
        return "DRAFT_CANNOT_ANSWER"
    if typed_failure in {"CLAIM_NOT_ALL_SUPPORTED", "CLAIM_NOT_SUPPORTED"}:
        if "CONTRADICTED" in set(claim_states or []):
            return "CLAIM_CONTRADICTED"
        return "CLAIM_NOT_SUPPORTED"
    if typed_failure == "CLAIM_CONTRADICTED":
        return "CLAIM_CONTRADICTED"
    if typed_failure == "COMPLETENESS_FAILURE":
        return "COMPLETENESS_FAILURE"
    if typed_failure in {"UNAUTHORIZED_OR_INVALID_CITATION", "INVALID_CITATION"}:
        return "INVALID_CITATION"
    if typed_failure == "INVALID_SUPPORTING_ID":
        return "INVALID_SUPPORTING_ID"
    if typed_failure in {"INACTIVE_VERSION", "VERSION_MISMATCH", "VERSION_FAILURE"}:
        return "VERSION_FAILURE"
    if typed_failure in {"UNAUTHORIZED_SUPPORTING_ID", "ACL_FAILURE", "SECURITY_FAILURE"}:
        return "SECURITY_FAILURE"
    if typed_failure in {
        "DRAFT_REQUEST_ERROR",
        "VERIFIER_REQUEST_ERROR",
        "PROVIDER_REQUEST_ERROR",
        "JUDGE_REQUEST_ERROR",
    }:
        return "PROVIDER_REQUEST_ERROR"
    return "UNKNOWN"


def diagnostic_rollup(diagnostic: dict[str, Any]) -> dict[str, Any]:
    rows = list(diagnostic.get("cases") or [])
    fn_rows = [item for item in rows if item.get("cohort") == "FN"]
    safety_rows = [item for item in rows if item.get("cohort") == "SAFETY"]
    incremental = [
        (item.get("draft_latency_ms") or 0) + (item.get("verifier_latency_ms") or 0)
        for item in rows
        if item.get("recovery_triggered")
    ]
    draft_tokens_in = sum(
        item.get("draft_prompt_tokens") or 0 for item in rows if item.get("draft_live")
    )
    draft_tokens_out = sum(
        item.get("draft_completion_tokens") or 0 for item in rows if item.get("draft_live")
    )
    verifier_tokens_in = sum(
        item.get("verifier_prompt_tokens") or 0 for item in rows if item.get("draft_success")
    )
    verifier_tokens_out = sum(
        item.get("verifier_completion_tokens") or 0 for item in rows if item.get("draft_success")
    )
    input_tokens = draft_tokens_in + verifier_tokens_in
    output_tokens = draft_tokens_out + verifier_tokens_out
    category_rescues = Counter(item["category"] for item in fn_rows if item.get("valid_rescue"))
    category_fn = Counter(item["category"] for item in fn_rows)
    failure_census = Counter(
        classify_recovery_failure(item.get("typed_failure"), item.get("claim_states"))
        for item in rows
        if item.get("typed_failure")
    )
    cost_usd = official_token_cost(
        model=SOL_MODEL, input_tokens=input_tokens, output_tokens=output_tokens
    )
    go = diagnostic.get("go_nogo")
    if go in NO_GO_ALIASES:
        go = NO_GO
    return {
        "go_nogo": go,
        "historical_fn_cases": len(fn_rows),
        "draft_success_count": sum(1 for item in fn_rows if item.get("draft_success")),
        "verification_pass_count": sum(1 for item in fn_rows if item.get("verification_pass")),
        "valid_rescue_count": sum(1 for item in fn_rows if item.get("valid_rescue")),
        "still_abstain_count": sum(1 for item in fn_rows if not item.get("answered")),
        "unsupported_recovery_count": sum(
            1 for item in rows if item.get("behavior") == "UNSUPPORTED_ANSWER"
        ),
        "safety_control_false_positives": sum(
            1 for item in safety_rows if item.get("false_positive_recovery")
        ),
        "near_duplicate_rescues": category_rescues.get("near_duplicate", 0),
        "near_duplicate_fn": category_fn.get("near_duplicate", 0),
        "two_document_rescues": category_rescues.get("multidoc_two", 0),
        "three_document_rescues": category_rescues.get("multidoc_three", 0),
        "multi_document_rescues": category_rescues.get("multidoc_two", 0)
        + category_rescues.get("multidoc_three", 0),
        "exact_id_rescues": category_rescues.get("exact_identifier", 0),
        "version_region_rescues": category_rescues.get("version_region", 0),
        "failure_census": dict(failure_census),
        "version_violations": sum(
            1
            for item in rows
            if classify_recovery_failure(item.get("typed_failure")) == "VERSION_FAILURE"
        ),
        "recovery_funnel": {
            "primary_judge_negatives": len(rows),
            "recovery_triggers": sum(1 for item in rows if item.get("recovery_triggered")),
            "draft_successes": sum(1 for item in rows if item.get("draft_success")),
            "verification_passes": sum(1 for item in rows if item.get("verification_pass")),
            "completeness_passes": sum(
                1 for item in rows if item.get("completeness_state") == "COMPLETE"
            ),
            "valid_rescues": sum(1 for item in fn_rows if item.get("valid_rescue")),
            "false_positive_recoveries": sum(
                1 for item in safety_rows if item.get("false_positive_recovery")
            ),
            "completeness_failures": sum(
                1 for item in rows if item.get("typed_failure") == "COMPLETENESS_FAILURE"
            ),
        },
        "usage": {
            "additional_draft_calls": sum(1 for item in rows if item.get("draft_external_calls")),
            "additional_verifier_calls": sum(
                item.get("verifier_external_calls") or 0 for item in rows
            ),
            "recovery_input_tokens": input_tokens,
            "recovery_output_tokens": output_tokens,
            "new_query_embedding_calls": 0,
            "new_sol_judge_calls": 0,
            "external_reranker_calls": 0,
            "fallback_trigger_rate": 1.0 if rows else 0.0,
        },
        "cost": {
            "sol_recovery_usd": cost_usd,
            "additional_cost_usd": cost_usd,
            "embedding_cost": 0.0,
            "pricing_source": "official OpenAI gpt-5.6-sol short-context list prices",
        },
        "latency": {
            "incremental": {
                "mean_ms": mean(incremental) if incremental else 0.0,
                "p50_ms": median(incremental) if incremental else 0.0,
                "p95_ms": _percentile(incremental, 0.95) if incremental else 0.0,
                "worst_case_ms": max(incremental) if incremental else 0.0,
                "count": len(incremental),
            }
        },
    }
