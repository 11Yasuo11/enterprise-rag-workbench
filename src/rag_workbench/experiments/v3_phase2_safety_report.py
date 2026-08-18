# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path
from typing import Any

V3_PHASE2_BENCHMARK_HEADING = (
    "## Enterprise RAG Workbench v3 Research — Phase 2 Recovery Safety"
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
    development = status.get("development_results") or {}
    ledger = status.get("ledger") or []
    selected = status.get("selected_candidate")
    preflight = status.get("hosted_preflight") or {}
    validation = status.get("validation_results") or {}
    exp1 = validation.get("V3_P2_EXP1_UNTRUSTED_INSTRUCTION_BOUNDARY") or {}
    lines = [
        V3_PHASE2_BENCHMARK_HEADING,
        "",
        "This section is V3 research. It does not replace frozen Enterprise RAG Workbench v2.",
        "Architecture `enterprise-rag-workbench-v3-research`. Parent `enterprise-rag-workbench-v2`. Production `false`.",
        "",
        "### Research objective",
        "",
        "Preserve Generate→Verify evidence-utilization gains while restoring the frozen V2 safety profile.",
        "Independent variable: one safety mechanism on the frozen Phase-1 recovery baseline.",
        "Draft prompt `generate-verify-draft-v1` and verifier prompt `generate-verify-claim-verifier-v1` were not retuned.",
        "",
        "### Safety validation dataset",
        "",
        f"Dataset `{status.get('dataset_id')}`. Hash `{status.get('dataset_hash')}`.",
        f"Maximum prior overlap {_fmt(overlap.get('maximum_normalized_overlap'))} vs `{closest.get('prior_dataset')}` / `{closest.get('prior_case_id')}`.",
        f"Independence pass `{overlap.get('pass')}`. Threshold `{overlap.get('overlap_threshold')}`.",
        "",
        "### Development-only historical replay",
        "",
        "DEVELOPMENT ONLY. Not unseen promotion evidence.",
        f"fv2_inj_02 illocution `{development.get('fv2_inj_02_illocution')}`. fv2_inj_03 illocution `{development.get('fv2_inj_03_illocution')}`.",
        f"Question-illocution blocks fv2_inj_02 `{development.get('fv2_inj_02_blocked_by_question_illocution')}` and fv2_inj_03 `{development.get('fv2_inj_03_blocked_by_question_illocution')}`.",
        "",
        "### Offline gold-span audit (not promotion)",
        "",
        f"Label `{(development.get('offline_gold_boundary_audit') or {}).get('label')}`.",
        f"Legitimate gold answers blocked `{(development.get('offline_gold_boundary_audit') or {}).get('legitimate_block_count')}`.",
        f"Simulated injection FPs unblocked `{(development.get('offline_gold_boundary_audit') or {}).get('injection_unblock_count')}`.",
        "",
        "### Experiment ledger",
        "",
    ]
    if not ledger:
        lines.append("No hosted candidate evaluation rows yet.")
        lines.append("")
    for row in ledger:
        lines.extend(
            [
                f"Experiment `{row.get('experiment_id')}`.",
                f"Hypothesis: {row.get('hypothesis')}",
                f"Independent variable: {row.get('independent_variable')}",
                f"Verdict `{row.get('verdict')}`. Configuration hash `{row.get('configuration_hash')}`.",
                "",
            ]
        )
    lines.extend(
        [
            "### Selected safety mechanism",
            "",
            f"`{selected if selected is not None else 'NONE_HOSTED_EVALUATION_NOT_RUN'}`.",
            "",
            "Budget stop is not `NO_SAFE_GENERATE_VERIFY_CANDIDATE`.",
            "",
            "### Experiment 1 validation snapshot",
            "",
            f"Injection FP `{exp1.get('injection_false_positives')}`. Unsupported `{exp1.get('unsupported_answers')}`. Precision {_fmt(exp1.get('precision'))}.",
            f"Rescue fraction of recoverable {_fmt(exp1.get('rescue_fraction_of_recoverable'))}. SAFE_RECOVERY_BLOCKED `{exp1.get('SAFE_RECOVERY_BLOCKED')}`.",
            "",
            "### Hosted preflight / stop",
            "",
            f"Stop reason `{status.get('stop_reason')}`. Estimated USD `{_fmt((preflight or {}).get('estimated_usd_sol'))}`.",
            "",
            "Official frozen v2 was not modified.",
        ]
    )
    return "\n".join(lines)


def _replace_heading_section(text: str, heading: str, section: str) -> str:
    count = text.count(heading)
    if count > 1:
        raise ValueError(f"duplicate heading {heading}")
    if count == 0:
        return text.rstrip() + "\n\n" + section + "\n"
    prefix, remainder = text.split(heading, 1)
    if "\n" in remainder:
        _, rest = remainder.split("\n", 1)
    else:
        rest = ""
    suffix = ""
    offset = 0
    for line in rest.splitlines(keepends=True):
        if offset > 0 and line.startswith("## "):
            suffix = rest[offset:]
            break
        offset += len(line)
    else:
        suffix = ""
    rendered = section.rstrip() + "\n"
    if suffix:
        return prefix.rstrip() + "\n\n" + rendered + "\n" + suffix.lstrip()
    return prefix.rstrip() + "\n\n" + rendered


def persist_v3_phase2_markdown(status: dict[str, Any]) -> None:
    path = Path("BENCHMARK.md")
    path.write_text(_replace_heading_section(path.read_text(), V3_PHASE2_BENCHMARK_HEADING, render_v3_phase2_markdown(status)))
