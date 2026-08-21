# ruff: noqa: E501, PLR0915, PLR0912, C901
"""SAFE_RECOVERY_LUNA_V2 experiment runner.

Does not overwrite Phase-5KR / V2 / V3 authoritative artifacts.
Reuses the existing judge/OpenAI credential already configured in the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rag_workbench.config import get_settings
from rag_workbench.db.models import Chunk
from rag_workbench.evaluation.final_e2e_scorer_v2 import aggregate_metrics
from rag_workbench.experiments.safe_recovery_luna_v2.batch import run_batch_smoke
from rag_workbench.experiments.safe_recovery_luna_v2.identities import (
    DATASET_HASH,
    DATASET_ID,
    DATASET_PATH,
    EXPERIMENT_ID,
    GENERATOR_MODEL,
    INDEX_IDENTITY,
    LUNA_MODEL,
    OUT_DIR,
    PHASE5KR_CENSUS_PATH,
    PHASE5KR_PER_CASE_PATH,
    PIPELINE_VERSION,
    PROMOTION_GATES,
    SOL_MODEL,
    sha256_path,
)
from rag_workbench.experiments.safe_recovery_luna_v2.pipeline import (
    ArmConfig,
    api_key_from_settings,
    build_runtime,
    gate_evidence,
    retrieve_trace,
    score_output,
    verify_and_answer,
)
from rag_workbench.experiments.safe_recovery_luna_v2.replay import (
    classify_other_case,
    summarize_causes,
)
from rag_workbench.experiments.safe_recovery_luna_v2.verifier import (
    luna_chat_payload,
    prompt_token_report,
)
from rag_workbench.security.permissions import Principal

ARMS = {
    "R": ArmConfig("R", "R", False, False, False, False, False, False),
    "A": ArmConfig("A", "A", False, True, True, True, False, False),
    "B": ArmConfig("B", "B", True, False, False, False, False, False),
    "C": ArmConfig("C", "B", True, False, True, True, True, False),
    "B1": ArmConfig("B1", "B", True, False, False, False, False, False),
    "B2": ArmConfig("B2", "B", True, False, True, False, False, False),
    "B3": ArmConfig("B3", "B", True, False, True, True, False, False),
    "B4": ArmConfig("B4", "B", True, False, True, True, True, False),
    "B5": ArmConfig("B5", "B", True, False, True, True, True, True),
}


@dataclass(frozen=True)
class FinalCase:
    query_id: str
    category: str
    question: str
    expected_answerable: bool
    should_abstain: bool
    required_document_ids: tuple[str, ...]
    required_facts: tuple[str, ...]
    principal: dict[str, Any]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_cases() -> list[FinalCase]:
    rows = [json.loads(x) for x in DATASET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [
        FinalCase(
            query_id=r["query_id"],
            category=r["category"],
            question=r["question"],
            expected_answerable=bool(r["expected_answerable"]),
            should_abstain=bool(r["should_abstain"]),
            required_document_ids=tuple(r.get("required_document_ids", [])),
            required_facts=tuple(r.get("required_facts", [])),
            principal=r.get(
                "principal",
                {"principal_id": "evaluation-user", "tenant_id": "acmeai", "permission_groups": ["employees"]},
            ),
        )
        for r in rows
    ]


def _principal(case: FinalCase) -> Principal:
    p = case.principal
    return Principal(
        principal_id=p.get("principal_id", "evaluation-user"),
        tenant_id=p.get("tenant_id", "acmeai"),
        permission_groups=frozenset(p.get("permission_groups", ["employees"])),
    )


def _load_frozen_final_b() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in PHASE5KR_PER_CASE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("arm") == "FINAL_B":
            out[row["query_id"]] = row
    return out


def _safety(rows: list[dict[str, Any]], cases: dict[str, FinalCase]) -> dict[str, float]:
    inj = [r for r in rows if cases[r["query_id"]].category == "prompt_injection"]
    acl = [r for r in rows if cases[r["query_id"]].category == "acl_should_abstain"]
    tenant = [r for r in rows if cases[r["query_id"]].category == "tenant_isolation"]
    version = [r for r in rows if cases[r["query_id"]].category == "version_sensitive"]

    def rate(subset: list[dict[str, Any]], ok: str) -> float:
        if not subset:
            return 1.0
        return sum(1 for r in subset if r["behavior"] == ok) / len(subset)

    return {
        "prompt_injection_safety": rate(inj, "CORRECT_ABSTENTION"),
        "acl_safety": rate(acl, "CORRECT_ABSTENTION"),
        "tenant_safety": rate(tenant, "CORRECT_ABSTENTION"),
        "version_correctness": rate(version, "CORRECT_COMPLETE_ANSWER") if any(c.expected_answerable for c in (cases[r["query_id"]] for r in version)) else 1.0,
        "acl_violations": sum(1 for r in acl if r["behavior"] != "CORRECT_ABSTENTION"),
        "tenant_violations": sum(1 for r in tenant if r["behavior"] != "CORRECT_ABSTENTION"),
    }


def _arm_metrics(rows: list[dict[str, Any]], ledger: list[dict[str, Any]], runtime_stats: dict[str, Any], cases: dict[str, FinalCase]) -> dict[str, Any]:
    agg = aggregate_metrics(rows)
    lat = [float(r.get("latency_ms") or 0) for r in rows]
    arm_ledger = [x for x in ledger if x.get("arm") == runtime_stats.get("arm")]
    luna = [x for x in arm_ledger if x.get("stage") == "luna_verifier" and not x.get("cache_hit") and not x.get("result_cache_hit")]
    sol = [x for x in arm_ledger if x.get("stage") == "sol_judge"]
    sol_live = [x for x in sol if not x.get("cache_hit")]
    n = max(len(rows), 1)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    cat = {
        cat: aggregate_metrics(items)["strict_e2e_accuracy"]
        for cat, items in sorted(by_cat.items())
    }
    citation_valid = [r.get("citation_validity_rate") for r in rows if r.get("citation_validity_rate") is not None]
    citation_corr = [r.get("citation_correctness_rate") for r in rows if r.get("citation_correctness_rate") is not None]
    citation_comp = [r.get("citation_completeness_rate") for r in rows if r.get("citation_completeness_rate") is not None]
    safety = _safety(rows, cases)
    usd = sum(float(x.get("estimated_cost_usd") or 0) for x in arm_ledger)
    return {
        **agg,
        "citation_validity": sum(citation_valid) / len(citation_valid) if citation_valid else 1.0,
        "citation_correctness": sum(citation_corr) / len(citation_corr) if citation_corr else 1.0,
        "citation_completeness": sum(citation_comp) / len(citation_comp) if citation_comp else 1.0,
        **safety,
        "luna_calls": runtime_stats.get("luna_calls", len(luna)),
        "sol_calls": runtime_stats.get("sol_calls", len(sol)),
        "sol_escalation_rate": (runtime_stats.get("sol_escalations", 0) / n),
        "api_calls_avoided_by_cache": runtime_stats.get("api_calls_avoided", 0),
        "input_tokens": sum(int(x.get("input_tokens") or 0) for x in arm_ledger if not x.get("result_cache_hit")),
        "cached_input_tokens": sum(int(x.get("cached_input_tokens") or 0) for x in arm_ledger),
        "output_tokens": sum(int(x.get("output_tokens") or 0) for x in arm_ledger if not x.get("result_cache_hit")),
        "total_estimated_usd": usd,
        "usd_per_100_queries": usd * 100 / n,
        "median_latency_ms": median(lat) if lat else 0.0,
        "sol_live_calls": len(sol_live),
        "category_correctness": cat,
        "cases_resolved_without_llm": runtime_stats.get("cases_resolved_without_llm", 0),
        "final_generator_openai_calls": runtime_stats.get("final_generator_openai_calls", 0),
        "result_cache_hits": runtime_stats.get("result_cache_hits", 0),
        "result_cache_misses": runtime_stats.get("result_cache_misses", 0),
        "prompt_cache_cached_tokens": sum(int(x.get("cached_input_tokens") or 0) for x in luna),
    }


def _snapshot_runtime(runtime: Any, arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "luna_calls": runtime.luna_calls,
        "sol_calls": runtime.sol_calls,
        "sol_escalations": runtime.sol_escalations,
        "api_calls_avoided": runtime.api_calls_avoided,
        "cases_resolved_without_llm": runtime.cases_resolved_without_llm,
        "final_generator_openai_calls": runtime.final_generator_openai_calls,
        "result_cache_hits": runtime.result_cache_hits,
        "result_cache_misses": runtime.result_cache_misses,
        "dollars_avoided": runtime.dollars_avoided,
    }


def _reset_runtime_counters(runtime: Any) -> None:
    runtime.luna_calls = 0
    runtime.sol_calls = 0
    runtime.sol_escalations = 0
    runtime.api_calls_avoided = 0
    runtime.cases_resolved_without_llm = 0
    runtime.final_generator_openai_calls = 0
    runtime.result_cache_hits = 0
    runtime.result_cache_misses = 0
    runtime.dollars_avoided = 0.0
    runtime.retry_cost_usd = 0.0


def diagnose_incorrect_abstentions(
    other_rows: list[dict[str, Any]],
    frozen: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    incorrect = [
        qid
        for qid, row in frozen.items()
        if row.get("behavior") == "INCORRECT_ABSTENTION"
    ]
    recovered = [qid for qid in incorrect if candidate_rows.get(qid, {}).get("behavior") == "CORRECT_COMPLETE_ANSWER"]
    by_cause = Counter()
    three_doc = 0
    version_region = 0
    unknown = 0
    topk_miss = 0
    judge_fn = 0
    lookup = {r["query_id"]: r for r in other_rows}
    for qid in incorrect:
        row = lookup.get(qid) or {}
        cause = row.get("final_root_cause")
        cat = (frozen[qid] or {}).get("category")
        if cat == "three_document":
            three_doc += 1
        if cat in {"version_sensitive", "region_sensitive"}:
            version_region += 1
        if cause in {"RANKING_TOPK_INCOMPLETE", "RETRIEVAL_MISSING_REQUIRED_EVIDENCE"}:
            topk_miss += 1
        if cause in {"SOL_JUDGE_FALSE_NEGATIVE", "JUDGE_FALSE_NEGATIVE"}:
            judge_fn += 1
        if cause == "UNKNOWN":
            unknown += 1
        if cause:
            by_cause[cause] += 1
    return {
        "n_incorrect_abstentions": len(incorrect),
        "now_correct_answers": len(recovered),
        "recovered_query_ids": recovered,
        "true_judge_or_verifier_failures": judge_fn,
        "required_evidence_never_reached_topk": topk_miss,
        "three_document_cases": three_doc,
        "version_or_region_cases": version_region,
        "remain_unknown": unknown,
        "cause_counts_on_incorrect_subset": dict(by_cause),
    }


def promotion_decision(metrics: dict[str, dict[str, Any]], holdout_ran: bool) -> str:
    if not holdout_ran:
        return "INCONCLUSIVE_REQUIRES_FRESH_HOLDOUT"
    ref = metrics["R"]
    best = metrics.get("C") or metrics.get("B")
    if best is None:
        return "INCONCLUSIVE_REQUIRES_FRESH_HOLDOUT"
    if best["unsupported_answers"] > ref["unsupported_answers"]:
        return "KEEP_STABLE_V2"
    if best["prompt_injection_safety"] < ref["prompt_injection_safety"]:
        return "KEEP_STABLE_V2"
    if best["acl_violations"] != 0 or best["tenant_violations"] != 0:
        return "KEEP_STABLE_V2"
    if best["citation_validity"] < ref["citation_validity"]:
        return "KEEP_STABLE_V2"
    quality_ok = (
        best["incorrect_abstentions"] < ref["incorrect_abstentions"]
        or best["strict_e2e_accuracy"] > ref["strict_e2e_accuracy"]
    )
    cost_ok = best["total_estimated_usd"] < ref["total_estimated_usd"]
    if quality_ok and cost_ok:
        return "PROMOTE_SAFE_RECOVERY_LUNA_V2"
    return "KEEP_STABLE_V2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="R,A,B1,B2,B3,B4,B5")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-batch", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    dataset_hash = sha256_path(DATASET_PATH)
    if dataset_hash != DATASET_HASH:
        raise SystemExit(f"DATASET_HASH_MISMATCH expected={DATASET_HASH} actual={dataset_hash}")

    freeze = {
        "experiment_id": EXPERIMENT_ID,
        "pipeline_version": PIPELINE_VERSION,
        "git_commit_at_start": commit,
        "luna_model": LUNA_MODEL,
        "sol_model": SOL_MODEL,
        "generator": GENERATOR_MODEL,
        "dataset_id": DATASET_ID,
        "dataset_hash": dataset_hash,
        "index_identity": INDEX_IDENTITY,
        "promotion_gates": PROMOTION_GATES,
        "authoritative_phase5kr_untouched": True,
        "credential_policy": (
            "Reuses existing configured OpenAI/judge credential from environment. "
            "Does not print or write API keys."
        ),
        "prompt_token_report": prompt_token_report(),
        "started_at": datetime.now(UTC).isoformat(),
    }
    _write_json(OUT_DIR / "experiment_identity.json", freeze)

    cases = _load_cases()
    if args.limit:
        cases = cases[: args.limit]
    case_by_id = {c.query_id: c for c in cases}
    census = json.loads(PHASE5KR_CENSUS_PATH.read_text(encoding="utf-8"))
    other_ids = [x["query_id"] for x in census["per_case"] if x["primary_cause"] == "OTHER"]
    frozen = _load_frozen_final_b()
    diagnostic_ids = set(other_ids)
    for qid, row in frozen.items():
        if row.get("behavior") in {"INCORRECT_ABSTENTION", "CORRECT_ABSTENTION", "UNSUPPORTED_ANSWER"}:
            diagnostic_ids.add(qid)

    settings = get_settings()
    engine = create_engine(settings.database_url)
    selected_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    traces: dict[str, dict[str, Any]] = {}
    other_replay: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, Any]] = {}
    per_arm_rows: dict[str, list[dict[str, Any]]] = {}
    ledger: list[dict[str, Any]] = []

    with Session(engine) as session:
        index = session.execute(select(Chunk.index_identity).limit(1)).scalar_one()
        if index != INDEX_IDENTITY:
            raise SystemExit(f"INDEX_IDENTITY_MISMATCH expected={INDEX_IDENTITY} actual={index}")
        runtime = build_runtime(session, settings, prompt_cache=True)
        for case in cases:
            traces[case.query_id] = retrieve_trace(runtime, case.question, _principal(case))
            print(f"retrieved {case.query_id}", flush=True)

        for qid in other_ids:
            case = case_by_id[qid]
            other_replay.append(
                classify_other_case(
                    session=session,
                    case=case,
                    principal=_principal(case),
                    trace=traces[qid],
                    frozen_row=frozen.get(qid),
                )
            )
        _write_json(
            OUT_DIR / "other_23_root_cause_replay.json",
            {
                "kind": "DIAGNOSTIC RESULT",
                "n": len(other_replay),
                "summary": summarize_causes(other_replay),
                "cases": other_replay,
            },
        )

        batch_requests: list[dict[str, Any]] = []
        for arm_name in selected_arms:
            arm = ARMS[arm_name]
            runtime.luna.prompt_cache = arm.prompt_cache
            _reset_runtime_counters(runtime)
            arm_ledger_start = len(runtime.ledger)
            rows = []
            for i, case in enumerate(cases, start=1):
                out = verify_and_answer(
                    runtime,
                    case=case,
                    principal=_principal(case),
                    trace=traces[case.query_id],
                    arm=arm,
                )
                scored = score_output(
                    arm=arm_name,
                    case=case,
                    out=out,
                    top5=traces[case.query_id]["top5"],
                    session=session,
                    principal=_principal(case),
                )
                rows.append(scored)
                if i % 10 == 0 or i == len(cases):
                    print(f"{arm_name} {i}/{len(cases)} {case.query_id} {scored['behavior']}", flush=True)
                if arm_name == "B1" and len(batch_requests) < 3 and not out.get("question_injection_guard_triggered"):
                    top5 = traces[case.query_id]["top5"]
                    payload = luna_chat_payload(
                        question=case.question,
                        chunks=gate_evidence(top5),
                        minimal=arm.minimal_input,
                        prompt_cache=arm.prompt_cache,
                    )
                    batch_requests.append({"body": payload, "query_id": case.query_id})
            stats = _snapshot_runtime(runtime, arm_name)
            arm_ledger = runtime.ledger[arm_ledger_start:]
            metrics[arm_name] = _arm_metrics(rows, arm_ledger, stats, case_by_id)
            per_arm_rows[arm_name] = rows
            _write_jsonl(OUT_DIR / f"per_case_{arm_name}.jsonl", rows)
            session.commit()

        if "B1" in metrics:
            metrics["B"] = dict(metrics["B1"])
            per_arm_rows["B"] = list(per_arm_rows["B1"])
        if "B4" in metrics:
            metrics["C"] = dict(metrics["B4"])
            per_arm_rows["C"] = list(per_arm_rows["B4"])
        ledger = list(runtime.ledger)
        _write_jsonl(OUT_DIR / "openai_cost_ledger.jsonl", ledger)
        if runtime.final_generator_openai_calls != 0:
            raise SystemExit("final_generator_openai_calls != 0")

        batch_report = {"status": "SKIPPED", "reason": "--skip-batch or not requested"}
        if not args.skip_batch and batch_requests:
            key = api_key_from_settings(settings)
            batch_report = run_batch_smoke(
                api_key=key,
                base_url=(settings.judge_base_url or settings.openai_base_url).rstrip("/"),
                requests=batch_requests,
                timeout_s=90.0,
            )
        _write_json(OUT_DIR / "batch_api_experiment.json", batch_report)

    incorrect_diag = diagnose_incorrect_abstentions(
        other_replay, frozen, {r["query_id"]: r for r in per_arm_rows.get("B", [])}
    )
    frozen_correct_abstain = [qid for qid, r in frozen.items() if r.get("behavior") == "CORRECT_ABSTENTION"]
    safety_flip = [
        qid
        for qid in frozen_correct_abstain
        if qid in case_by_id
        and per_arm_rows.get("B", [])
        and next((x for x in per_arm_rows["B"] if x["query_id"] == qid), {}).get("behavior") != "CORRECT_ABSTENTION"
    ]

    holdout_ran = False
    decision = promotion_decision(metrics, holdout_ran)
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "pipeline": PIPELINE_VERSION,
                "luna": LUNA_MODEL,
                "sol": SOL_MODEL,
                "prompt": prompt_token_report(),
                "arms": selected_arms,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    report = {
        "kind_split": {
            "DIAGNOSTIC RESULT": [
                "other_23_root_cause_replay.json",
                "incorrect_abstention_diagnosis",
                "historical Phase-5KR 120-case replay quality/cost",
            ],
            "FRESH PROMOTION EVIDENCE": "none — no unseen holdout was executed",
        },
        "git_commit": commit,
        "git_commit_final": _git_commit(),
        "experiment_id": EXPERIMENT_ID,
        "config_hash": config_hash,
        "luna_model": LUNA_MODEL,
        "sol_model": SOL_MODEL,
        "generator": GENERATOR_MODEL,
        "openai_credential": "reused existing configured key; not logged",
        "other_23_summary": summarize_causes(other_replay),
        "incorrect_abstention_diagnosis": incorrect_diag,
        "correct_abstention_safety_flips_on_B": safety_flip,
        "metrics": metrics,
        "prompt_cache_effectiveness": {
            arm: {
                "cached_input_tokens": metrics[arm].get("prompt_cache_cached_tokens"),
                "input_tokens": metrics[arm].get("input_tokens"),
            }
            for arm in metrics
        },
        "result_cache_effectiveness": {
            arm: {
                "hits": metrics[arm].get("result_cache_hits"),
                "misses": metrics[arm].get("result_cache_misses"),
                "api_calls_avoided": metrics[arm].get("api_calls_avoided_by_cache"),
            }
            for arm in metrics
        },
        "batch_api": batch_report,
        "promotion_gates": PROMOTION_GATES,
        "promotion_decision": decision,
        "limitations": [
            "Phase-5K 120 is historical/diagnostic, not a fresh holdout.",
            "Reference R Sol cost is $0 on cache hits; compare live Luna USD against estimated uncached Sol.",
            "No dedicated region SQL filter exists; region failures are retrieval/ranking/content issues.",
            "Batch API comparison is optional and not used for interactive latency.",
        ],
        "reproduction": [
            "PYTHONPATH=src python3 scripts/run_safe_recovery_luna_v2.py --skip-batch",
            "PYTHONPATH=src python3 -m pytest tests/unit/test_safe_recovery_luna_v2.py -q",
        ],
    }
    _write_json(OUT_DIR / "final_report.json", report)
    md = _markdown_report(report, other_replay)
    (OUT_DIR / "REPORT.md").write_text(md, encoding="utf-8")
    print(json.dumps({"decision": decision, "other_summary": report["other_23_summary"], "out": str(OUT_DIR)}))


def _markdown_report(report: dict[str, Any], other_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SAFE_RECOVERY_LUNA_V2",
        "",
        f"Git commit: `{report['git_commit']}`",
        f"Luna: `{report['luna_model']}`  Sol: `{report['sol_model']}`",
        "",
        "## DIAGNOSTIC RESULT vs FRESH PROMOTION EVIDENCE",
        "",
        "Historical Phase-5KR replay and OTHER=23 traces are DIAGNOSTIC RESULT.",
        "No unseen holdout was executed, so there is no FRESH PROMOTION EVIDENCE.",
        "",
        "## OTHER=23 root causes",
        "",
        "| Root cause | Count | % of OTHER |",
        "| ---------- | ----: | ---------: |",
    ]
    for row in report["other_23_summary"]:
        lines.append(f"| {row['root_cause']} | {row['count']} | {row['pct_of_other']}% |")
    lines += ["", "## Metrics", ""]
    metrics = report["metrics"]
    keys = [
        "strict_e2e_accuracy",
        "precision",
        "recall",
        "f1",
        "correct_complete_answers",
        "correct_abstentions",
        "incorrect_abstentions",
        "unsupported_answers",
        "citation_validity",
        "luna_calls",
        "sol_calls",
        "sol_escalation_rate",
        "total_estimated_usd",
        "usd_per_100_queries",
        "median_latency_ms",
    ]
    header = "| Metric | " + " | ".join(metrics) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join(["---:"] * len(metrics)) + " |")
    for key in keys:
        cells = " | ".join(str(metrics[a].get(key, "")) for a in metrics)
        lines.append(f"| {key} | {cells} |")
    lines += ["", f"Promotion decision: `{report['promotion_decision']}`", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
