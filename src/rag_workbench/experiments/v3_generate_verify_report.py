# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path
from typing import Any

V3_PHASE1_BENCHMARK_HEADING = (
    "## Enterprise RAG Workbench v3 Research — Phase 1 Generate-Then-Verify Recovery"
)


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
        f"p95 {_fmt(values.get('p95_ms'))} (n={values.get('count', 0)})"
    )


def render_v3_phase1_markdown(status: dict[str, Any]) -> str:
    diagnostic = status.get("diagnostic") or {}
    control = status.get("control_metrics") or {}
    candidate = status.get("candidate_metrics") or {}
    funnel = status.get("recovery_funnel") or {}
    security = status.get("security") or {}
    citations = status.get("citations") or {}
    latency = status.get("latency") or {}
    usage = status.get("usage") or {}
    cost = status.get("cost") or {}
    categories = status.get("category_results") or {}
    selection = status.get("selection") or {}
    overlap = status.get("overlap_report") or {}
    closest = status.get("closest_previous_case") or overlap.get("closest_previous_case") or {}
    go = status.get("go_nogo")
    diagnostic_cases = diagnostic.get("cases") or []
    if not funnel:
        funnel = {
            "primary_judge_negatives": diagnostic.get("historical_fn_count", 0)
            + len(diagnostic.get("safety_controls") or []),
            "recovery_triggers": sum(1 for item in diagnostic_cases if item.get("recovery_triggered")),
            "draft_successes": sum(1 for item in diagnostic_cases if item.get("draft_success")),
            "verification_passes": sum(1 for item in diagnostic_cases if item.get("verification_pass")),
            "valid_rescues": diagnostic.get("rescue_count", 0),
            "false_positive_recoveries": diagnostic.get("false_positive_count", 0),
            "completeness_failures": sum(
                1 for item in diagnostic_cases if item.get("typed_failure") == "COMPLETENESS_FAILURE"
            ),
        }
    selection_reason = (status.get("selection") or {}).get("reason") or ""
    return "\n".join(
        [
            V3_PHASE1_BENCHMARK_HEADING,
            "",
            "This section is V3 research. It does not replace frozen Enterprise RAG Workbench v2.",
            "Architecture `enterprise-rag-workbench-v3-research`. Parent `enterprise-rag-workbench-v2`. Production `false`.",
            "",
            "### Frozen v2 control",
            "",
            "```text",
            "Dense + BM25 → RRF → POINTWISE_CROSS_ENCODER_TOP5",
            "Judge GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1 / evidence-sufficiency-v1",
            "Generator deterministic-extractive-v1.1",
            "```",
            "",
            "### Candidate B",
            "",
            "```text",
            "V2 + schema-valid negative-decision recovery:",
            "Judge no → grounded draft → claim verification → completeness → PASS answer / FAIL abstain",
            "```",
            "",
            f"Draft prompt `{((status.get('candidate_configuration') or {}).get('recovery') or {}).get('draft_prompt_version')}`.",
            f"Verifier prompt `{((status.get('candidate_configuration') or {}).get('recovery') or {}).get('verifier_prompt_version')}`.",
            "Same Sol model family; separate prompt, schema, cache, and logical-call identities.",
            "",
            "### Diagnostic replay",
            "",
            "Diagnosis-only. Not promotion evidence. Historical V2 labels were not rewritten.",
            f"Historical FN cases {diagnostic.get('historical_fn_count', 0)}.",
            f"Rescues {diagnostic.get('rescue_count', 0)}: `{diagnostic.get('rescues')}`.",
            f"Safety controls {len(diagnostic.get('safety_controls') or [])}.",
            f"False positives {diagnostic.get('false_positive_count', 0)}: `{diagnostic.get('false_positives')}`.",
            f"Unauthorized evidence {diagnostic.get('unauthorized_evidence_used', 0)}. Invalid citation IDs {diagnostic.get('invalid_citation_ids', 0)}.",
            f"Unsupported recovered answers {(diagnostic.get('rollup') or {}).get('unsupported_recovery_count', diagnostic.get('unsupported_recovered_answers'))}.",
            f"GO / NO_GO: `{'NO_GO_FOR_UNSEEN_V3_PHASE1' if (go or '') in {'NO_GO_FOR_UNSEEN_EXPERIMENT', 'NO_GO_FOR_UNSEEN_V3_PHASE1'} else go}`.",
            "",
            "Diagnostic category rescues are historical FN traces only and are not promotion evidence.",
            f"Near-duplicate {(diagnostic.get('rollup') or {}).get('near_duplicate_rescues')} / {(diagnostic.get('rollup') or {}).get('near_duplicate_fn')}.",
            f"Two-document {(diagnostic.get('rollup') or {}).get('two_document_rescues')}. Three-document {(diagnostic.get('rollup') or {}).get('three_document_rescues')}. Multi-document {(diagnostic.get('rollup') or {}).get('multi_document_rescues')}.",
            f"Exact-ID {(diagnostic.get('rollup') or {}).get('exact_id_rescues')}. Version/region {(diagnostic.get('rollup') or {}).get('version_region_rescues')}.",
            f"Failure census `{(diagnostic.get('rollup') or {}).get('failure_census')}`.",
            "",
            "### New dataset",
            "",
            f"Dataset `{status.get('dataset_id')}`. Hash `{status.get('dataset_hash')}`.",
            f"Generation method `{status.get('generation_method')}`. Freeze `{status.get('dataset_frozen_at')}`.",
            f"Maximum prior overlap {_fmt(status.get('maximum_prior_overlap'))} vs `{closest.get('prior_dataset')}` / `{closest.get('prior_case_id')}`.",
            f"Independence pass `{overlap.get('pass')}`.",
            "",
            "### Precommitted policy",
            "",
            "Candidate wins only if answerable-case correct-answer rate improves >= +0.10 or valid rescues >= 8, and hard safety/regression gates hold. Frozen before first unseen result.",
            "",
            "### Primary results",
            "",
            "| Metric | Control A | Candidate B |",
            "|---|---:|---:|",
            f"| Correct answers | {control.get('correct_answers')} | {candidate.get('correct_answers')} |",
            f"| Correct abstentions | {control.get('correct_abstentions')} | {candidate.get('correct_abstentions')} |",
            f"| Incorrect abstentions | {control.get('incorrect_abstentions')} | {candidate.get('incorrect_abstentions')} |",
            f"| Unsupported answers | {control.get('unsupported_answers')} | {candidate.get('unsupported_answers')} |",
            f"| Accuracy | {_fmt(control.get('accuracy'))} | {_fmt(candidate.get('accuracy'))} |",
            f"| Precision | {_fmt(control.get('precision'))} | {_fmt(candidate.get('precision'))} |",
            f"| Recall | {_fmt(control.get('recall'))} | {_fmt(candidate.get('recall'))} |",
            f"| F1 | {_fmt(control.get('f1'))} | {_fmt(candidate.get('f1'))} |",
            f"| Answerable-case correct-answer rate | {_fmt(control.get('answerable_case_correct_answer_rate'))} | {_fmt(candidate.get('answerable_case_correct_answer_rate'))} |",
            "",
            "### Recovery funnel",
            "",
            f"Primary Judge negatives {funnel.get('primary_judge_negatives')} · triggers {funnel.get('recovery_triggers')} · draft successes {funnel.get('draft_successes')} · verification passes {funnel.get('verification_passes')} · valid rescues {funnel.get('valid_rescues')} · false-positive recoveries {funnel.get('false_positive_recoveries')} · completeness failures {funnel.get('completeness_failures')}.",
            f"Rescue IDs `{(status.get('valid_rescues') or {}).get('case_ids') or diagnostic.get('rescues')}`.",
            f"False-positive IDs `{(status.get('false_positive_recoveries') or {}).get('case_ids') or diagnostic.get('false_positives')}`.",
            f"Completeness-failure IDs `{(status.get('completeness_failures') or {}).get('case_ids')}`.",
            "",
            "### Category results",
            "",
            *[
                (
                    f"{label}: control correct {(categories.get(key) or {}).get('control', {}).get('correct_answers')} / "
                    f"incorrect abstentions {(categories.get(key) or {}).get('control', {}).get('incorrect_abstentions')}; "
                    f"candidate correct {(categories.get(key) or {}).get('candidate', {}).get('correct_answers')} / "
                    f"incorrect abstentions {(categories.get(key) or {}).get('candidate', {}).get('incorrect_abstentions')}."
                )
                for label, key in (
                    ("Near-duplicate", "near_duplicate"),
                    ("Two-document", "multidoc_two"),
                    ("Three-document", "multidoc_three"),
                    ("Exact-ID", "exact_identifier"),
                    ("Version/region", "version_region"),
                    ("Semantic", "semantic_paraphrase"),
                    ("Single-document", "single_document"),
                    ("Should-abstain", "should_abstain"),
                )
            ],
            "",
            "### Security and citations",
            "",
            f"ACL {_fmt(security.get('acl_safety'))}. Tenant {_fmt(security.get('tenant_isolation'))}. Version {_fmt(security.get('version_correctness'))}. Prompt-injection {_fmt(security.get('prompt_injection_boundary'))}.",
            f"Invalid supporting IDs {security.get('invalid_supporting_ids', 0)}. Unauthorized supporting IDs {security.get('unauthorized_supporting_ids', 0)}.",
            f"Citation validity {_fmt(citations.get('validity'))}. Invalid citations {citations.get('invalid_citation_count', 0)}.",
            "",
            "### Latency, usage, cost",
            "",
            f"Incremental fallback {_latency_line(latency.get('incremental'))}. Candidate total {_latency_line(latency.get('candidate_total'))}.",
            f"Fallback trigger rate {_fmt(usage.get('fallback_trigger_rate'))}. Additional draft calls {usage.get('additional_draft_calls', 0)}. Additional verifier calls {usage.get('additional_verifier_calls', 0)}.",
            f"Additional recovery tokens in/out {usage.get('recovery_input_tokens', 0)}/{usage.get('recovery_output_tokens', 0)}. Additional Sol cost USD {_fmt(cost.get('additional_cost_usd'))}. Embedding cost {cost.get('embedding_cost')}.",
            "",
            "### Selection",
            "",
            f"Selected `{status.get('selected_strategy')}`. Reason `{selection_reason or selection.get('reason')}`. Primary quality {selection.get('primary_quality')}. Hard gates {selection.get('hard_gates')}. Regression gate {selection.get('regression_gate')}.",
            f"Remaining bottleneck `{status.get('primary_remaining_bottleneck')}`.",
            "",
            "Official frozen v2 was not modified.",
        ]
    )


def persist_v3_markdown(status: dict[str, Any], *, replace: bool = True) -> None:
    path = Path("BENCHMARK.md")
    text = path.read_text()
    count = text.count(V3_PHASE1_BENCHMARK_HEADING)
    section = render_v3_phase1_markdown(status)
    if count > 1:
        raise ValueError("duplicate v3 phase 1 heading")
    if count == 1:
        if not replace:
            return
        prefix = text.split(V3_PHASE1_BENCHMARK_HEADING, 1)[0]
        path.write_text(prefix.rstrip() + "\n\n" + section + "\n")
        return
    if not status.get("completed") and not status.get("diagnostic"):
        return
    path.write_text(text.rstrip() + "\n\n" + section + "\n")
