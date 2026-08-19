# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path
from typing import Any

V3_PHASE3_BENCHMARK_HEADING = "## V3 Final Frozen Generate→Verify A/B"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _latency_line(payload: dict[str, Any] | None) -> str:
    values = payload or {}
    return (
        f"mean {_fmt(values.get('mean_ms'))} / p50 {_fmt(values.get('p50_ms'))} / "
        f"p95 {_fmt(values.get('p95_ms'))} / max {_fmt(values.get('max_ms'))} (n={values.get('count', 0)})"
    )


def _replace_heading_section(text: str, heading: str, section: str) -> str:
    if heading not in text:
        return text.rstrip() + "\n\n" + section.rstrip() + "\n"
    prefix, _sep, rest = text.partition(heading)
    next_idx = rest.find("\n## ")
    suffix = rest[next_idx:] if next_idx != -1 else ""
    return prefix.rstrip() + "\n\n" + section.rstrip() + ("\n" + suffix if suffix else "\n")


def render_v3_final_ab_markdown(status: dict[str, Any]) -> str:
    control = status.get("control_metrics") or {}
    candidate = status.get("candidate_metrics") or {}
    paired = status.get("paired_deltas") or {}
    funnel = status.get("recovery_funnel") or {}
    boundary = status.get("instruction_boundary") or {}
    injection = status.get("prompt_injection") or {}
    security = status.get("security") or {}
    citations = status.get("citations") or {}
    latency = status.get("latency") or {}
    usage = status.get("usage") or {}
    cost = status.get("cost") or {}
    census = status.get("failure_census") or {}
    retrieval = status.get("retrieval_metrics") or {}
    categories = status.get("category_results") or {}
    target = categories.get("target_95") or {}
    overlap = status.get("overlap_report") or {}
    closest = status.get("closest_previous_case") or overlap.get("closest_previous_case") or {}
    recovery = ((status.get("candidate_configuration") or {}).get("recovery") or {})
    lines = [
        V3_PHASE3_BENCHMARK_HEADING,
        "",
        "This section is V3 research. It does not replace frozen Enterprise RAG Workbench v2.",
        "Architecture `enterprise-rag-workbench-v3-research`. Parent `enterprise-rag-workbench-v2`. Production `false`.",
        "",
        "### Dataset identity",
        "",
        f"Dataset `{status.get('dataset_id')}`. Hash `{status.get('dataset_hash')}`. Cases `{len(status.get('case_ids') or [])}`.",
        f"Generation method `{status.get('generation_method')}`. Freeze `{status.get('dataset_frozen_at')}`.",
        f"Maximum prior overlap {_fmt(status.get('maximum_prior_overlap'))} vs `{closest.get('prior_dataset')}` / `{closest.get('prior_case_id') or closest.get('prior_id')}`.",
        f"Independence pass `{overlap.get('pass')}`. Overlap ceiling `{overlap.get('overlap_threshold')}`.",
        "",
        "### Candidate architecture",
        "",
        "Control A is official frozen V2 Judge-first. Candidate B uses the same Top-5 and Primary Judge. Recovery runs only after a schema-valid Judge `answerable=false`: `generate-verify-draft-v1` → `generate-verify-claim-verifier-v1` → completeness → deterministic ACL/tenant/version/citation validation → `evidence-instruction-boundary-v1`.",
        f"Safety mechanism `{recovery.get('safety_version')}`. Draft `{recovery.get('draft_prompt_version')}`. Verifier `{recovery.get('verifier_prompt_version')}`.",
        "",
        "### Control A metrics",
        "",
        f"Cases {control.get('cases')}. Answerable {control.get('answerable_cases')}. Should-abstain {control.get('should_abstain_cases')}.",
        f"Correct answers {control.get('correct_answers')}. Correct abstentions {control.get('correct_abstentions')}. Incorrect abstentions {control.get('incorrect_abstentions')}. Unsupported {control.get('unsupported_answers')}.",
        f"Accuracy {_fmt(control.get('accuracy'))}. Precision {_fmt(control.get('answer_precision') or control.get('precision'))}. Recall {_fmt(control.get('answer_recall') or control.get('recall'))}. F1 {_fmt(control.get('f1'))}.",
        f"Answerable-case correct-answer rate {_fmt(control.get('answerable_case_correct_answer_rate'))}.",
        "",
        "### Candidate B metrics",
        "",
        f"Cases {candidate.get('cases')}. Answerable {candidate.get('answerable_cases')}. Should-abstain {candidate.get('should_abstain_cases')}.",
        f"Correct answers {candidate.get('correct_answers')}. Correct abstentions {candidate.get('correct_abstentions')}. Incorrect abstentions {candidate.get('incorrect_abstentions')}. Unsupported {candidate.get('unsupported_answers')}.",
        f"Accuracy {_fmt(candidate.get('accuracy'))}. Precision {_fmt(candidate.get('answer_precision') or candidate.get('precision'))}. Recall {_fmt(candidate.get('answer_recall') or candidate.get('recall'))}. F1 {_fmt(candidate.get('f1'))}.",
        f"Answerable-case correct-answer rate {_fmt(candidate.get('answerable_case_correct_answer_rate'))}.",
        "",
        "### Paired deltas",
        "",
        f"Candidate correct − Control correct {paired.get('candidate_correct_minus_control_correct')}.",
        f"Incorrect abstention delta {paired.get('candidate_incorrect_abstention_minus_control')}. Unsupported delta {paired.get('candidate_unsupported_minus_control')}.",
        f"Answerable correct-rate delta {_fmt(paired.get('absolute_answerable_correct_rate_delta'))}. F1 delta {_fmt(paired.get('f1_delta'))}.",
        f"A abstain → B correct {paired.get('a_abstain_to_b_correct')}. A abstain → B unsupported {paired.get('a_abstain_to_b_unsupported')}.",
        f"A correct → B correct {paired.get('a_correct_to_b_correct')}. A correct → B incorrect {paired.get('a_correct_to_b_incorrect')}.",
        f"A correct abstain → B correct abstain {paired.get('a_correct_abstain_to_b_correct_abstain')}. A correct abstain → B answer {paired.get('a_correct_abstain_to_b_answer')}.",
        f"Additional correct supported {paired.get('additional_correct_supported')}. Rescue IDs `{paired.get('rescue_ids')}`.",
        "",
        "### Safety",
        "",
        f"ACL {_fmt(security.get('acl_safety'))}. Tenant {_fmt(security.get('tenant_isolation'))}. Version {_fmt(security.get('version_correctness'))}. Prompt-injection {_fmt(security.get('prompt_injection_safety'))}.",
        f"Citation validity {_fmt(citations.get('validity') or security.get('citation_validity'))}. Citation correctness {_fmt(citations.get('correctness') or security.get('citation_correctness'))}.",
        f"Unauthorized supporting IDs {security.get('unauthorized_supporting_ids', 0)}. Invalid supporting IDs {security.get('invalid_supporting_ids', 0)}. Content identity failures {security.get('content_identity_failures', 0)}.",
        f"Instruction-boundary invocations {boundary.get('boundary_invocations')}. PASS {boundary.get('boundary_pass')}. FAIL {boundary.get('boundary_fail')}. SAFE_RECOVERY_BLOCKED {boundary.get('SAFE_RECOVERY_BLOCKED')}.",
        f"Prompt-injection cases safe {(injection.get('safe_count'))} / {injection.get('required')}.",
        "",
        "### Cost",
        "",
        f"New query embeddings {usage.get('new_query_embeddings')}. Embedding tokens {usage.get('embedding_tokens')}.",
        f"Primary Judge logical {usage.get('primary_judge_logical_calls')} / physical {usage.get('primary_judge_physical_attempts')} / live {usage.get('new_sol_judge_calls')}.",
        f"Recovery draft logical {usage.get('recovery_draft_logical_calls')}. Verifier logical {usage.get('recovery_verifier_logical_calls')}. Transport retries {usage.get('transport_retries')}.",
        f"Judge tokens in/out {usage.get('judge_input_tokens')}/{usage.get('judge_output_tokens')}. Recovery tokens in/out {usage.get('recovery_input_tokens')}/{usage.get('recovery_output_tokens')}.",
        f"Embedding USD {_fmt(cost.get('embedding_usd'))}. Judge USD {_fmt(cost.get('primary_judge_usd'))}. Recovery USD {_fmt(cost.get('recovery_draft_and_verifier_usd'))}. Total final benchmark USD {_fmt(cost.get('total_final_benchmark_usd'))}.",
        "Historical diagnostic cost is excluded from this total.",
        "",
        "### Latency",
        "",
        f"Embedding {_latency_line(latency.get('embedding'))}.",
        f"Dense {_latency_line(latency.get('dense'))}. BM25 {_latency_line(latency.get('bm25'))}. RRF {_latency_line(latency.get('rrf'))}. Cross-Encoder {_latency_line(latency.get('cross_encoder'))}.",
        f"Primary Judge {_latency_line(latency.get('primary_judge'))}. Draft {_latency_line(latency.get('recovery_draft'))}. Verifier {_latency_line(latency.get('recovery_verifier'))}. Boundary {_latency_line(latency.get('instruction_boundary'))}.",
        f"Control total {_latency_line(latency.get('control_total'))}. Candidate total {_latency_line(latency.get('candidate_total'))}.",
        f"Recovery-triggered Candidate {_latency_line(latency.get('recovery_triggered_candidate'))}. Non-recovery Candidate {_latency_line(latency.get('non_recovery_candidate'))}.",
        "",
        "### Failure census",
        "",
        f"`{(census.get('counts'))}`.",
        f"Primary remaining bottleneck `{status.get('primary_remaining_bottleneck')}`.",
        "",
        "### Retrieval",
        "",
        f"Hit@5 {_fmt(retrieval.get('hit_at_5'))}. Recall@5 {_fmt(retrieval.get('recall_at_5'))}. MRR {_fmt(retrieval.get('mrr'))}. nDCG {_fmt(retrieval.get('ndcg'))}.",
        f"Required Evidence Recall {_fmt(retrieval.get('required_evidence_recall'))}. All Required Evidence Coverage@5 {_fmt(retrieval.get('all_required_evidence_coverage_at_5'))}.",
        f"Pool Required Evidence Recall {_fmt(retrieval.get('candidate_pool_required_evidence_recall'))}. Pool All Required Evidence Coverage {_fmt(retrieval.get('candidate_pool_all_required_evidence_coverage'))}.",
        f"Pool-complete / Top-5-incomplete {retrieval.get('pool_complete_top5_incomplete')}.",
        "",
        "### Recovery funnel",
        "",
        f"Judge negatives {funnel.get('primary_judge_negatives')} → triggered {funnel.get('recovery_triggered')} → draft {funnel.get('draft_success')} → verify {funnel.get('claim_verification_pass')} → completeness {funnel.get('completeness_pass')} → boundary PASS {funnel.get('instruction_boundary_pass')} → correct supported recovery {funnel.get('correct_supported_recovery')}.",
        f"Losses `{funnel.get('losses')}`.",
        "",
        "### 95% target",
        "",
        f"Answerable {target.get('answerable_cases')}. Correct supported {target.get('correct_supported_answers')}. Rate {_fmt(target.get('correct_answer_rate'))}.",
        f"Minimum correct for >=95% {target.get('minimum_correct_answers_for_95')}. Additional still required {target.get('additional_correct_answers_still_required')}. Claimed 95 `{target.get('claimed_95')}`.",
        "",
        "### Promotion decision",
        "",
        f"`{status.get('promotion_decision')}`. V3 status `{status.get('v3_status')}`. Selected `{status.get('selected_strategy')}`.",
        "",
        "### Known limitations",
        "",
        "This is a one-shot evaluation of an already-frozen candidate. Failures discovered here are future research, not a license to retune prompts, the instruction boundary, retrieval, the Judge, or the promotion policy. Official public v2 was not modified.",
        "",
    ]
    return "\n".join(lines)


def persist_v3_final_ab_markdown(status: dict[str, Any], path: Path | None = None) -> None:
    target = path or Path("BENCHMARK.md")
    rendered = render_v3_final_ab_markdown(status)
    current = target.read_text() if target.exists() else ""
    target.write_text(_replace_heading_section(current, V3_PHASE3_BENCHMARK_HEADING, rendered))
