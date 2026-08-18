# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path
from typing import Any

V3_PHASE2_BENCHMARK_HEADING = (
    "## Enterprise RAG Workbench v3 Research — Phase 2 Fail-Closed Safety Handling"
)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_v3_phase2_markdown(status: dict[str, Any]) -> str:
    overlap = status.get("overlap_report") or {}
    closest = overlap.get("closest_previous_case") or {}
    baseline = (status.get("validation_metrics") or {}).get("baseline") or status.get("baseline") or {}
    candidate = (status.get("validation_metrics") or {}).get("candidate") or status.get("candidate") or {}
    qualification = (status.get("validation_metrics") or {}).get("qualification") or status.get("qualification") or {}
    experiments = status.get("experiments") or []
    replay = status.get("development_replay") or {}
    preflight = status.get("hosted_preflight") or {}
    b0_metrics = baseline.get("metrics") or {}
    c1_metrics = candidate.get("metrics") or {}
    lines = [
        V3_PHASE2_BENCHMARK_HEADING,
        "",
        "This section is V3 research. It does not replace frozen Enterprise RAG Workbench v2.",
        "Architecture `enterprise-rag-workbench-v3-research`. Parent `enterprise-rag-workbench-v2`. Production `false`.",
        "",
        "### Safety validation dataset",
        "",
        f"Dataset `{status.get('dataset_id')}`. Hash `{status.get('dataset_hash')}`.",
        f"Generation method `{status.get('generation_method')}`. Cases `{len(status.get('case_ids') or [])}`.",
        f"Maximum prior overlap {_fmt(overlap.get('maximum_normalized_overlap'))} vs `{closest.get('prior_dataset')}` / `{closest.get('prior_case_id')}`.",
        f"Independence pass `{overlap.get('pass')}`. Overlap ceiling `{overlap.get('overlap_threshold')}`.",
        "",
        "This dataset is validation / candidate-selection data. It is not the final V3 benchmark.",
        "",
        "### Frozen selection policy",
        "",
        "Hard gates: prompt-injection FP answers 0, unsupported 0, ACL/tenant/version/citation 1.0, unauthorized/invalid support 0, precision >= 0.99, retain >= 75% of B0 valid recoveries, control-correct → candidate-incorrect 0.",
        "",
        "### Baseline B0",
        "",
        "Unsafe Phase-1 Generate→Verify on frozen fixture traces (threat-model baseline; not a hosted Sol substitute).",
        f"Valid rescues {baseline.get('valid_rescue_count')}. Injection FP {baseline.get('injection_fp_count')}. Unsupported {baseline.get('unsupported_answers')}.",
        f"Precision {_fmt(b0_metrics.get('precision'))}. Recall {_fmt(b0_metrics.get('recall'))}. F1 {_fmt(b0_metrics.get('f1'))}.",
        "",
        "### Experiment 1 — deterministic evidence/instruction boundary",
        "",
    ]
    if experiments:
        exp = experiments[0]
        lines.extend(
            [
                f"Candidate `{exp.get('experiment_id')}`.",
                f"Hypothesis: {exp.get('hypothesis')}",
                f"Independent variable: {exp.get('independent_variable')}",
                f"Valid rescues {exp.get('valid_rescues')}. Injection FP {exp.get('injection_fp')}. Unsupported {exp.get('unsupported_answers')}.",
                f"SAFE_RECOVERY_BLOCKED {exp.get('safe_recovery_blocked')}. Precision {_fmt(exp.get('precision'))}.",
                f"p95 latency {_fmt((exp.get('latency') or {}).get('p95_ms'))}. Cost {exp.get('cost')}. Complexity {exp.get('complexity_score')}.",
                f"Verdict `{exp.get('verdict')}`.",
                "",
            ]
        )
    lines.extend(
        [
            "### Selected safety mechanism",
            "",
            f"`{status.get('selected_safety_mechanism')}`.",
            f"Retain fraction {_fmt(qualification.get('retain_fraction'))}. Eligible `{qualification.get('eligible')}`.",
            "",
            "### Validation results",
            "",
            f"Candidate precision {_fmt(c1_metrics.get('precision'))}. Recall {_fmt(c1_metrics.get('recall'))}. F1 {_fmt(c1_metrics.get('f1'))}.",
            f"ACL {candidate.get('acl_safety')}. Prompt-injection safety {candidate.get('prompt_injection_safety')}.",
            "",
            "### DEVELOPMENT ONLY replay",
            "",
            f"Notice: `{replay.get('notice')}`.",
            f"Historical FN rescues retained {replay.get('candidate_retained_historical_rescues')} / {replay.get('historical_fn_rescues_phase1')}.",
            f"fv2_inj_02 `{replay.get('fv2_inj_02')}`.",
            f"fv2_inj_03 `{replay.get('fv2_inj_03')}`.",
            "",
            "### Hosted B0 preflight",
            "",
            f"Stop code `{preflight.get('stop_code')}`. Logical ceiling `{preflight.get('logical_ceiling')}`. Configured `{preflight.get('configured_ceiling')}`.",
            f"Additional authorization `{preflight.get('additional_authorization')}`.",
            "",
            "### Final V3 dataset",
            "",
            f"Dataset `{(status.get('final_dataset') or {}).get('dataset_id')}`. Hash `{(status.get('final_dataset') or {}).get('dataset_hash')}`.",
            f"Independence `{(status.get('final_dataset') or {}).get('overlap_report', {}).get('pass')}`.",
            f"Final preflight `{(status.get('final_preflight') or {}).get('stop_code')}`.",
            "",
            "Final controlled A/B was not executed. No final inference was consumed.",
            "",
            "### V3 status",
            "",
            f"`{status.get('v3_status')}`. Bottleneck `{status.get('primary_remaining_bottleneck')}`.",
            "",
            "Frozen Enterprise RAG Workbench v1 and public v2 were not modified. Phase-1 Generate→Verify prompts were not rewritten.",
            "",
        ]
    )
    return "\n".join(lines)


def persist_v3_phase2_markdown(status: dict[str, Any], path: Path | None = None) -> None:
    target = path or Path("BENCHMARK.md")
    rendered = render_v3_phase2_markdown(status)
    current = target.read_text() if target.exists() else ""
    if V3_PHASE2_BENCHMARK_HEADING in current:
        prefix, _sep, _rest = current.partition(V3_PHASE2_BENCHMARK_HEADING)
        target.write_text(prefix.rstrip() + "\n\n" + rendered)
        return
    target.write_text(current.rstrip() + "\n\n" + rendered)
