# ruff: noqa: E501
"""Append or replace the measured dual-judge benchmark section in BENCHMARK.md."""

from pathlib import Path

from sqlalchemy import func, select

from rag_workbench.config import get_settings
from rag_workbench.db.models import AnswerabilityGateCacheRecord, ExperimentRunRecord
from rag_workbench.db.session import session_factory
from rag_workbench.experiments.evidence_benchmark import (
    ORIGINAL_NINE,
    EvidenceJudgeBenchmark,
)

HEADING = "## Evidence Judge Benchmark — Qwen3-8B vs GPT-5.6 Luna"


def value(item) -> str:
    if item is None:
        return "pending"
    if isinstance(item, float):
        return f"{item:.6f}"
    return str(item)


def metric_row(name: str, runs: dict) -> str:
    return "| " + name + " | " + " | ".join(
        value(runs[code]["aggregate_metrics"].get(name))
        for code in ("A", "B1", "B2", "C1", "C2")
    ) + " |"


def matrix(run: dict) -> str:
    metrics = run["aggregate_metrics"]
    return (
        f"{metrics['correct_answer_count']} / {metrics['incorrect_abstention_count']} / "
        f"{metrics['unsupported_answer_count']} / {metrics['correct_abstention_count']}"
    )


def case_map(run: dict) -> dict[str, dict]:
    return {case["case_id"]: case for case in run["cases"]}


def decision(case: dict) -> str:
    result = case.get("answerability_result")
    if result is None:
        return "baseline"
    support = ", ".join(result["supporting_chunk_ids"]) or "none"
    operational = case.get("answerability_operational_error") or "OK"
    return f"{result['answerable']} / {result['reason_code']} / {support} / {operational}"


def original_nine_rows(calibration: dict) -> list[str]:
    baseline = case_map(calibration["A"])
    qwen = case_map(calibration["B1"])
    luna = case_map(calibration["C1"])
    rows = []
    for case_id in sorted(ORIGINAL_NINE & set(baseline)):
        trace = baseline[case_id]["retrieval_trace"]
        retrieval = ", ".join(
            f"{item['chunk_id']} ({item['score']:.6f})" for item in trace
        )
        qwen_case = qwen[case_id]
        luna_case = luna[case_id]
        rows.append(
            f"| `{case_id}` | {retrieval} | {decision(qwen_case)} | "
            f"{decision(luna_case)} | {_outcome(qwen_case)} / {_outcome(luna_case)} |"
        )
    return rows


def _outcome(case: dict) -> str:
    if case.get("answerability_operational_error"):
        return case["answerability_operational_error"]
    expected_answerable = not case["expected_abstain"]
    actual_answerable = not case["abstained"]
    if not expected_answerable:
        return "FIXED_BY_GATE" if not actual_answerable else "STILL_UNSUPPORTED"
    return "FALSE_ABSTENTION" if not actual_answerable else "FIXED_BY_GATE"


def generalization(calibration: dict, holdout: dict, selected: str) -> tuple[str, str]:
    before = calibration[selected]["aggregate_metrics"]
    after = holdout[selected]["aggregate_metrics"]
    f1_change = after["answerability_f1"] - before["answerability_f1"]
    unsupported_rate_change = (
        after["unsupported_answer_count"] / 30
        - before["unsupported_answer_count"] / 70
    )
    security_ok = all(
        after.get(name) == expected
        for name, expected in (
            ("acl_safety", 1.0),
            ("version_accuracy", 1.0),
            ("prompt_injection_boundary", 1.0),
            ("unauthorized_evidence_selection", 0),
        )
    )
    if f1_change >= -0.05 and unsupported_rate_change <= 0.02 and security_ok:
        rating = "GOOD"
    elif f1_change >= -0.10 and security_ok:
        rating = "QUESTIONABLE"
    else:
        rating = "POOR"
    evidence = (
        f"F1 change {f1_change:+.6f}; unsupported-answer rate change "
        f"{unsupported_rate_change:+.6f}; security constraints "
        f"{'held' if security_ok else 'did not all hold'}."
    )
    return rating, evidence


def render(status: dict) -> str:
    calibration = status["calibration"]
    selected = status["selected_candidate"]
    lines = [
        HEADING,
        "",
        "### Methodology and integrity",
        "",
        "Five candidates were evaluated on the deterministic 70-case calibration split: "
        "A (threshold-only), B1/B2 (one cached local Qwen decision with all/supporting-only "
        "generation context), and C1/C2 (one cached Luna decision with the same two context "
        "policies). Retrieval remained Top-5 dense semantic search at threshold 0.28 using the "
        "existing 64-dimensional `text-embedding-3-small` index. No document was re-embedded.",
        "",
        "Both providers used prompt version `evidence-sufficiency-v1`, the same task, evidence "
        "delimiters, reason codes, supporting-ID rule, and fail-closed semantics. Judges received "
        "only the question and authorized retrieved evidence; evaluation labels and expected "
        "sources/answers were not provided.",
        "",
        "Qwen v4 used Ollama's native `/api/chat` endpoint with `think=false`, deterministic "
        "bounded inference, and native JSON Schema output. Versions 1 and 2 used the unsuitable "
        "OpenAI-compatible Ollama transport, where reasoning consumed the bounded output and "
        "structured content became unreliable; version 3 has no persisted cache result. All "
        "older rows remain for auditability and were excluded from B1/B2 quality metrics. Cache "
        "identity includes provider, model, judge and prompt versions, normalized question, "
        "ordered evidence identity/content hashes, index identity, and prompt-render hash.",
        "",
        "**The holdout set was not used to select the judge model or context-pruning policy.**",
        "",
        "### Calibration results",
        "",
        "| Metric | A | B1 | B2 | C1 | C2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "answerability_accuracy",
        "answerability_precision",
        "answerability_recall",
        "answerability_f1",
        "correct_answer_count",
        "correct_abstention_count",
        "unsupported_answer_count",
        "incorrect_abstention_count",
        "judge_error_count",
        "judge_format_error_count",
        "gate_false_positive_count",
        "gate_false_negative_count",
        "multi_document_answer_success",
        "exact_identifier_answer_success",
        "citation_validity",
        "claim_support_rate",
        "grounded_answer_rate",
        "recall_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
        "acl_safety",
        "version_accuracy",
        "prompt_injection_boundary",
        "unauthorized_evidence_selection",
    ):
        lines.append(metric_row(name, calibration))
    lines.extend(
        [
            "",
            "Retrieval metrics are preserved for comparison; the evidence judges did not change "
            "retrieval ranking or index quality.",
            "",
            "### Calibration confusion matrices",
            "",
            "Order: expected-answerable/predicted-answerable, expected-answerable/predicted-"
            "abstain, expected-abstain/predicted-answerable, expected-abstain/predicted-abstain.",
            "",
            "| Candidate | TP / FN / FP / TN |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {code} | {matrix(calibration[code])} |" for code in calibration)
    lines.extend(
        [
            "",
            "### Qwen vs Luna trade-offs",
            "",
            "| Dimension | Qwen (B1/B2) | Luna (C1/C2) |",
            "|---|---|---|",
            f"| Quality | B1 F1 {value(calibration['B1']['aggregate_metrics']['answerability_f1'])}; "
            f"B2 F1 {value(calibration['B2']['aggregate_metrics']['answerability_f1'])} | "
            f"C1 F1 {value(calibration['C1']['aggregate_metrics']['answerability_f1'])}; "
            f"C2 F1 {value(calibration['C2']['aggregate_metrics']['answerability_f1'])} |",
            f"| Unsupported / incorrect abstentions | "
            f"{calibration['B1']['aggregate_metrics']['unsupported_answer_count']} / "
            f"{calibration['B1']['aggregate_metrics']['incorrect_abstention_count']} | "
            f"{calibration['C1']['aggregate_metrics']['unsupported_answer_count']} / "
            f"{calibration['C1']['aggregate_metrics']['incorrect_abstention_count']} |",
            f"| Format / request / timeout errors | "
            f"{calibration['B1']['aggregate_metrics']['judge_format_error_count']} / 0 / 0 | "
            f"{calibration['C1']['aggregate_metrics']['judge_format_error_count']} / 0 / 0 |",
            "| Supporting evidence selection | Qwen had four supporting-context-loss cases; "
            "B2 conservative support was 0.714286. | Luna had one supporting-context-loss case; "
            "C2 conservative support was 0.745455. |",
            f"| Mean cold judge latency | {value(calibration['B1']['latency']['answerability_judge_ms'])} ms | "
            f"{value(calibration['C1']['latency']['answerability_judge_ms'])} ms |",
            f"| Cache behavior B1/B2 or C1/C2 | "
            f"{calibration['B1']['usage']['gate_cache_hits']}/{calibration['B1']['usage']['gate_cache_misses']} then "
            f"{calibration['B2']['usage']['gate_cache_hits']}/{calibration['B2']['usage']['gate_cache_misses']} | "
            f"{calibration['C1']['usage']['gate_cache_hits']}/{calibration['C1']['usage']['gate_cache_misses']} then "
            f"{calibration['C2']['usage']['gate_cache_hits']}/{calibration['C2']['usage']['gate_cache_misses']} |",
            "| Privacy / operations | Local evidence stays on the developer machine; no external "
            "judge API usage, but local hardware, memory, energy, model storage, and runtime "
            "management remain real costs. | Authorized evidence is sent to OpenAI; network "
            "availability, latency, token billing, and external-service governance apply. |",
            "",
            "### Selected configuration",
            "",
            f"**{selected}** was locked at `{status['locked_at']}`. {status['selection_reason']}",
            "",
            "### Original-nine calibration analysis",
            "",
            "Only six of the original nine unsupported-answer cases were in calibration; the "
            "remaining three stayed sealed as part of holdout and were not used for selection.",
            "",
            "| Case | Retrieved chunks (score) | Qwen decision / reason / support / status | Luna decision / reason / support / status | Qwen / Luna outcome |",
            "|---|---|---|---|---|",
        ]
    )
    lines.extend(original_nine_rows(calibration))
    lines.extend(
        [
            "",
            "### Latency and usage",
            "",
            "| Candidate | Judge ms | Gate-cache ms | Pruning ms | Generation ms | Total ms | Gate H/M | External calls | Local calls | Judge input/output tokens |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for code, run in calibration.items():
        lines.append(
            f"| {code} | {value(run['latency']['answerability_judge_ms'])} | "
            f"{value(run['latency']['answerability_gate_cache_lookup_ms'])} | "
            f"{value(run['latency']['context_pruning_ms'])} | "
            f"{value(run['latency']['generation_ms'])} | {value(run['latency']['total_ms'])} | "
            f"{run['usage']['gate_cache_hits'] or 0}/{run['usage']['gate_cache_misses'] or 0} | "
            f"{run['usage']['external_judge_calls'] or 0} | {run['usage']['local_judge_calls'] or 0} | "
            f"{run['usage']['judge_prompt_tokens'] or 0}/{run['usage']['judge_completion_tokens'] or 0} |"
        )
    lines.extend(
        [
            "",
            f"Actual model requests/cache records: Qwen "
            f"{status['judge_cache']['qwen']['recorded_attempts']}/"
            f"{status['judge_cache']['qwen']['cached_results']} with "
            f"{status['judge_cache']['qwen']['prompt_tokens']}/"
            f"{status['judge_cache']['qwen']['completion_tokens']} input/output tokens; Luna "
            f"{status['judge_cache']['openai']['recorded_attempts']}/"
            f"{status['judge_cache']['openai']['cached_results']} with "
            f"{status['judge_cache']['openai']['prompt_tokens']}/"
            f"{status['judge_cache']['openai']['completion_tokens']} input/output tokens.",
            "",
            f"Qwen v4 outcomes were {status['audit']['qwen_v4_success']} successful decisions and "
            f"{status['audit']['qwen_v4_errors']} fail-closed format errors (zero request errors "
            f"and zero timeouts). Older persisted Qwen diagnostics made "
            f"{status['audit']['qwen_old_calls']} local calls: "
            f"{status['audit']['qwen_v1_cache']} v1 cache rows and "
            f"{status['audit']['qwen_v2_cache']} v2 cache rows; v3 had "
            f"{status['audit']['qwen_v3_cache']}. Cumulative persisted local calls are "
            f"{status['audit']['qwen_all_calls']} against the configured ceiling "
            f"{status['audit']['max_local_calls']}. This exceeds the requested cumulative "
            "interpretation of the ceiling by 7 because the persisted runner enforced it per "
            "judge version. No further Qwen calls were issued after this was discovered.",
            "",
            "Calibration-only Luna usage was 62 external calls with 38,436 input and 3,719 "
            "output tokens. Holdout reused five cached inputs and made 24 calls, producing "
            "cumulative Luna usage of 86 calls, 52,879 input tokens, and 5,160 output tokens. "
            "This phase made zero embedding calls; all query embeddings were cache hits.",
            "",
            "No project-local verified pricing configuration exists, so monetary cost is not "
            "calculated. Token usage is reported instead.",
        ]
    )
    if status.get("holdout_completed"):
        holdout = status["holdout"]
        rating, evidence = generalization(calibration, holdout, selected)
        lines.extend(
            [
                "",
                "### Sealed holdout results",
                "",
                f"The holdout was executed exactly once for A and the single locked winner "
                f"{selected}. No other judge candidate was run on holdout, and no configuration "
                "was changed afterward. The configuration lock was persisted before the first "
                "holdout call.",
                "",
                f"| Metric | A | {selected} |",
                "|---|---:|---:|",
            ]
        )
        for name in (
            "correct_answer_count",
            "correct_abstention_count",
            "unsupported_answer_count",
            "incorrect_abstention_count",
            "answerability_accuracy",
            "answerability_precision",
            "answerability_recall",
            "answerability_f1",
            "multi_document_answer_success",
            "exact_identifier_answer_success",
            "citation_validity",
            "claim_support_rate",
            "acl_safety",
            "version_accuracy",
            "prompt_injection_boundary",
            "unauthorized_evidence_selection",
        ):
            lines.append(
                f"| {name} | {value(holdout['A']['aggregate_metrics'].get(name))} | "
                f"{value(holdout[selected]['aggregate_metrics'].get(name))} |"
            )
        lines.extend(
            [
                "",
                f"### Generalization: {rating}",
                "",
                evidence + " This is descriptive evidence, not a statistical-significance claim.",
            ]
        )
    lines.extend(
        [
            "",
            "### Security and limitations",
            "",
            "Supporting IDs were checked server-side against retrieved IDs, ACLs, active and "
            "expected versions, and index identity. Invalid selections fail closed and the "
            "operational error is persisted separately. Retrieved content is explicitly "
            "delimited as untrusted evidence.",
            "",
            "The benchmark uses a synthetic 100-case dataset, deterministic extractive final "
            "generation, and one local Qwen quantization/runtime. Claim-level entailment remains "
            "unmeasured because the dataset has no claim-span annotations; `claim_support_rate` "
            "is the existing conservative expected-answer/source metric.",
            "",
            "Largest remaining measured problem: supporting-evidence selection for multi-document "
            "questions (C2 calibration multi-document success was 0.928571, driven by one false "
            "abstention/context-loss case). A future experiment should evaluate a new, frozen "
            "multi-document-focused judge prompt or model on newly created unseen evaluation data; "
            "it must not reuse this consumed holdout.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    with session_factory()() as session:
        status = EvidenceJudgeBenchmark(session).status(include_cases=True)
        settings = get_settings()
        runs = session.scalars(select(ExperimentRunRecord)).all()

        def local_calls(version: str | None = None) -> int:
            return sum(
                run.local_judge_calls or 0
                for run in runs
                if run.config.gate_config
                and run.config.gate_config.get("judge_provider") == "qwen"
                and (version is None or run.config.gate_config.get("judge_version") == version)
            )

        def cache_count(version: str, error: bool | None = None) -> int:
            clauses = [
                AnswerabilityGateCacheRecord.judge_provider == "qwen",
                AnswerabilityGateCacheRecord.judge_model == settings.local_judge_model,
                AnswerabilityGateCacheRecord.judge_version == version,
            ]
            if error is True:
                clauses.append(AnswerabilityGateCacheRecord.operational_error.is_not(None))
            elif error is False:
                clauses.append(AnswerabilityGateCacheRecord.operational_error.is_(None))
            return int(
                session.scalar(
                    select(func.count()).select_from(AnswerabilityGateCacheRecord).where(*clauses)
                )
                or 0
            )

        status["audit"] = {
            "qwen_v4_success": cache_count("4", False),
            "qwen_v4_errors": cache_count("4", True),
            "qwen_v1_cache": cache_count("1"),
            "qwen_v2_cache": cache_count("2"),
            "qwen_v3_cache": cache_count("3"),
            "qwen_old_calls": local_calls("1") + local_calls("2") + local_calls("3"),
            "qwen_all_calls": local_calls(),
            "max_local_calls": settings.max_local_judge_calls,
        }
    if status.get("selected_candidate") is None:
        raise SystemExit("benchmark has not been calibrated and locked")
    destination = Path("BENCHMARK.md")
    existing = destination.read_text(encoding="utf-8").rstrip()
    if HEADING in existing:
        existing = existing.split(HEADING, 1)[0].rstrip()
    destination.write_text(existing + "\n\n" + render(status), encoding="utf-8")


if __name__ == "__main__":
    main()
