# ruff: noqa: E501, PLR0912, PLR0915, C901
"""Run ADAPTIVE_TOP20_ABSTENTION_RECOVERY_V1 without mutating historical artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import AnswerabilityReason, AnswerabilityResult
from rag_workbench.config import get_settings
from rag_workbench.db.models import Chunk
from rag_workbench.evaluation.final_e2e_scorer_v2 import aggregate_metrics
from rag_workbench.experiments.adaptive_top20_abstention_recovery_v1.identities import (
    DATASET_HASH,
    DATASET_PATH,
    EXPERIMENT_ID,
    GENERATOR_MODEL,
    INDEX_IDENTITY,
    LUNA_MODEL,
    OUT_DIR,
    PHASE5KR_CENSUS,
    PHASE5KR_DIR,
    PHASE5KR_ROWS,
    PIPELINE_VERSION,
    SOL_MODEL,
    SWEEP_K,
    TOP_K_NORMAL,
    TOP_K_RECOVERY,
    TRACE_PATH,
    sha256_path,
    sha256_text,
)
from rag_workbench.experiments.adaptive_top20_abstention_recovery_v1.requirements import (
    decompose_requirements,
)
from rag_workbench.experiments.adaptive_top20_abstention_recovery_v1.validation import validate_go
from rag_workbench.experiments.adaptive_top20_abstention_recovery_v1.verifier import (
    PROMPT_IDENTITY,
    LunaTop20Verifier,
    RecoveryDecision,
)
from rag_workbench.experiments.safe_recovery_luna_v2.pipeline import (
    api_key_from_settings,
    build_runtime,
    gate_evidence,
    local_extractive_answer,
    retrieve_trace,
    score_output,
)
from rag_workbench.security.permissions import Principal


@dataclass(frozen=True)
class Case:
    query_id: str
    category: str
    question: str
    expected_answerable: bool
    should_abstain: bool
    required_document_ids: tuple[str, ...]
    required_facts: tuple[str, ...]
    principal: dict[str, Any]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_cases() -> list[Case]:
    return [
        Case(
            query_id=r["query_id"],
            category=r["category"],
            question=r["question"],
            expected_answerable=bool(r["expected_answerable"]),
            should_abstain=bool(r["should_abstain"]),
            required_document_ids=tuple(r.get("required_document_ids", [])),
            required_facts=tuple(r.get("required_facts", [])),
            principal=r.get(
                "principal",
                {
                    "principal_id": "evaluation-user",
                    "tenant_id": "acmeai",
                    "permission_groups": ["employees"],
                },
            ),
        )
        for r in load_jsonl(DATASET_PATH)
    ]


def principal(case: Case) -> Principal:
    return Principal(
        case.principal.get("principal_id", "evaluation-user"),
        case.principal.get("tenant_id", "acmeai"),
        frozenset(case.principal.get("permission_groups", ["employees"])),
    )


def frozen_rows(arm: str = "FINAL_R") -> dict[str, dict[str, Any]]:
    return {r["query_id"]: r for r in load_jsonl(PHASE5KR_ROWS) if r.get("arm") == arm}


def evidence_analysis(case: Case, ranked: list[dict[str, Any]]) -> dict[str, Any]:
    fact_hits: list[dict[str, Any]] = []
    required_chunks: list[str] = []
    required_ranks: list[int | None] = []
    for index, fact in enumerate(case.required_facts):
        expected_doc = (
            case.required_document_ids[index] if index < len(case.required_document_ids) else None
        )
        hits = [
            x
            for x in ranked
            if fact.casefold() in (x.get("text") or "").casefold()
            and (expected_doc is None or x.get("document_id") == expected_doc)
        ]
        hit = min(hits, key=lambda x: int(x["rank"])) if hits else None
        fact_hits.append(
            {
                "fact": fact,
                "expected_document_id": expected_doc,
                "chunk_id": None if hit is None else hit["chunk_id"],
                "rank": None if hit is None else int(hit["rank"]),
            }
        )
        if hit is not None:
            required_chunks.append(hit["chunk_id"])
            required_ranks.append(int(hit["rank"]))
        else:
            required_ranks.append(None)
    minimum = max((rank for rank in required_ranks if rank is not None), default=0)
    complete = bool(case.required_facts) and all(rank is not None for rank in required_ranks)
    return {
        "query_id": case.query_id,
        "category": case.category,
        "required_documents": list(case.required_document_ids),
        "required_chunk_ids": list(dict.fromkeys(required_chunks)),
        "required_ranks": required_ranks,
        "fact_hits": fact_hits,
        **{f"complete_at_k{k}": complete and minimum <= k for k in SWEEP_K},
        "minimum_k_for_complete_evidence": minimum
        if complete and minimum <= TOP_K_RECOVERY
        else None,
        "complete_within_ranked_pool": complete,
    }


def run_retrieval(cases: list[Case]) -> list[dict[str, Any]]:
    settings = get_settings()
    with Session(create_engine(settings.database_url)) as session:
        index = session.execute(select(Chunk.index_identity).limit(1)).scalar_one()
        if index != INDEX_IDENTITY:
            raise SystemExit(f"INDEX_IDENTITY_MISMATCH expected={INDEX_IDENTITY} actual={index}")
        runtime = build_runtime(session, settings, prompt_cache=True)
        traces = []
        for i, case in enumerate(cases, 1):
            trace = retrieve_trace(runtime, case.question, principal(case))
            traces.append(
                {
                    "query_id": case.query_id,
                    **trace,
                    "retrieval_mode": "reranking_only_from_same_rrf_candidate_pool",
                    "additional_dense_bm25_for_recovery": False,
                }
            )
            if i % 10 == 0:
                print(f"retrieval {i}/{len(cases)}", flush=True)
    write_jsonl(TRACE_PATH, traces)
    return traces


def coverage_artifacts(
    cases: list[Case], traces: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    analyses = [
        evidence_analysis(case, traces[case.query_id]["top20"])
        for case in cases
        if case.expected_answerable
    ]
    ranking_ids = {
        x["query_id"]
        for x in json.loads(PHASE5KR_CENSUS.read_text())["per_case"]
        if x["primary_cause"] == "OTHER"
    }
    ranking = [x for x in analyses if x["query_id"] in ranking_ids]
    rows = []
    for k in SWEEP_K:
        answerable = analyses
        three = [x for x in analyses if x["category"] == "three_document"]
        two = [x for x in analyses if x["category"] == "two_document"]
        required_total = sum(len(x["required_ranks"]) for x in answerable)
        required_found = sum(
            sum(rank is not None and rank <= k for rank in x["required_ranks"]) for x in answerable
        )
        chunks = [item for case in cases for item in traces[case.query_id]["top20"][:k]]
        rows.append(
            {
                "k": k,
                "required_evidence_recall": required_found / required_total,
                "all_required_evidence_coverage": sum(x[f"complete_at_k{k}"] for x in answerable)
                / len(answerable),
                "two_document_complete_coverage": sum(x[f"complete_at_k{k}"] for x in two)
                / len(two),
                "three_document_complete_coverage": sum(x[f"complete_at_k{k}"] for x in three)
                / len(three),
                "ranking_23_recoverable": sum(x[f"complete_at_k{k}"] for x in ranking),
                "average_chunks": k,
                "estimated_evidence_tokens": sum(len((x.get("text") or "").split()) for x in chunks)
                / len(cases),
                "new_llm_api_calls": 0,
            }
        )
    distribution = Counter()
    for x in ranking:
        minimum = x["minimum_k_for_complete_evidence"]
        if minimum == 6:
            label = "6"
        elif minimum in {7, 8}:
            label = "7-8"
        elif minimum in {9, 10}:
            label = "9-10"
        elif minimum is not None and minimum <= 15:
            label = "11-15"
        elif minimum is not None and minimum <= 20:
            label = "16-20"
        else:
            label = "incomplete_at_20"
        distribution[label] += 1
    sweep = {
        "experiment_id": EXPERIMENT_ID,
        "kind": "DIAGNOSTIC RESULT",
        "denominator_answerable": len(analyses),
        "rows": rows,
        "llm_api_calls": 0,
    }
    minimum_k = {
        "experiment_id": EXPERIMENT_ID,
        "kind": "DIAGNOSTIC RESULT",
        "n": len(ranking),
        "distribution": dict(distribution),
        "cases": ranking,
    }
    write_json(OUT_DIR / "topk_coverage_sweep.json", sweep)
    write_json(OUT_DIR / "ranking_failure_minimum_k.json", minimum_k)
    return ranking, sweep


def answer_with_v2(
    runtime: Any, case: Case, top20: list[dict[str, Any]], result: Any
) -> dict[str, Any]:
    support_ids = list(dict.fromkeys(r.chunk_id for r in result.requirements if r.chunk_id))
    selected = [x for x in top20 if x["chunk_id"] in support_ids]
    gate = AnswerabilityResult(
        answerable=True,
        supporting_chunk_ids=tuple(support_ids),
        confidence=1.0,
        reason_code=AnswerabilityReason.SUFFICIENT_EVIDENCE,
    )
    started = time.perf_counter()
    status, answer, citations, calls = local_extractive_answer(
        runtime, question=case.question, principal=principal(case), top5=selected, gate_result=gate
    )
    if calls != 0:
        raise RuntimeError("final_generator_openai_calls != 0")
    required_spans = [r.supporting_span for r in result.requirements if r.supporting_span]
    generation_complete = bool(answer) and all(span in answer for span in required_spans)
    citation_complete = set(support_ids) <= set(citations)
    if not generation_complete or not citation_complete:
        return {
            "status": "abstained",
            "answer": None,
            "citations": [],
            "decision": "SAFE_ABSTENTION",
            "reason": "DETERMINISTIC_GENERATION_COMPLETENESS_FAILED",
            "latency_ms": (time.perf_counter() - started) * 1000,
            "final_generator_openai_calls": 0,
            "generator_model": GENERATOR_MODEL,
            "generation_complete": generation_complete,
            "generation_citations_complete": citation_complete,
        }
    return {
        "status": status,
        "answer": answer,
        "citations": citations,
        "decision": "GO",
        "reason": "luna_top20_go_validated",
        "latency_ms": (time.perf_counter() - started) * 1000,
        "final_generator_openai_calls": 0,
        "generator_model": GENERATOR_MODEL,
    }


def evaluate(
    cases: list[Case],
    traces: dict[str, dict[str, Any]],
    *,
    reuse_recorded_luna: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = get_settings()
    frozen = frozen_rows()
    key = api_key_from_settings(settings)
    verifier = LunaTop20Verifier(key, settings.judge_base_url or settings.openai_base_url)
    recorded_rows = (
        {r["query_id"]: r for r in load_jsonl(OUT_DIR / "adaptive_top20_recovery_rows.jsonl")}
        if reuse_recorded_luna and (OUT_DIR / "adaptive_top20_recovery_rows.jsonl").exists()
        else {}
    )
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = (
        load_jsonl(OUT_DIR / "adaptive_top20_cost_ledger.jsonl")
        if reuse_recorded_luna and (OUT_DIR / "adaptive_top20_cost_ledger.jsonl").exists()
        else []
    )
    with Session(create_engine(settings.database_url)) as session:
        runtime = build_runtime(session, settings, prompt_cache=True)
        for i, case in enumerate(cases, 1):
            baseline = frozen[case.query_id]
            trace = traces[case.query_id]
            requirements = decompose_requirements(case.question)
            eligible = baseline.get("behavior") in {
                "INCORRECT_ABSTENTION",
                "CORRECT_ABSTENTION",
            } and not baseline.get("question_injection_guard_triggered")
            analysis = evidence_analysis(case, trace["top20"]) if case.expected_answerable else None
            common = {
                "query_id": case.query_id,
                "category": case.category,
                "top5_complete_evidence": None if analysis is None else analysis["complete_at_k5"],
                "top20_complete_evidence": None
                if analysis is None
                else analysis["complete_at_k20"],
                "minimum_k_for_complete_evidence": None
                if analysis is None
                else analysis["minimum_k_for_complete_evidence"],
                "required_document_count": len(case.required_document_ids),
                "required_documents_present_top5": 0
                if analysis is None
                else sum(rank is not None and rank <= 5 for rank in analysis["required_ranks"]),
                "required_documents_present_top20": 0
                if analysis is None
                else sum(rank is not None and rank <= 20 for rank in analysis["required_ranks"]),
                "required_ranks": [] if analysis is None else analysis["required_ranks"],
                "requirements": list(requirements),
                "entered_top20_recovery": eligible,
                "recovery_retrieval_mode": "reranking_only_from_same_rrf_candidate_pool"
                if eligible
                else "not_applicable",
                "additional_retrieval": False,
            }
            if not eligible:
                copied = dict(baseline)
                copied.update(common)
                copied.update(
                    {
                        "arm": "T20",
                        "luna_decision": "NOT_CALLED",
                        "deterministic_validation": "NOT_APPLICABLE",
                        "recovered_answer": False,
                    }
                )
                rows.append(copied)
                continue
            chunks = gate_evidence(trace["top20"])
            started = time.perf_counter()
            try:
                recorded = recorded_rows.get(case.query_id)
                if recorded and recorded.get("luna_result"):
                    result = RecoveryDecision.model_validate(recorded["luna_result"])
                    reused_luna = True
                else:
                    result = verifier.evaluate(case.question, requirements, chunks)
                    reused_luna = False
                failure = (
                    validate_go(
                        result, requirements, chunks, session=session, principal=principal(case)
                    )
                    if result.decision == "GO"
                    else None
                )
                if result.decision == "GO" and failure is None:
                    out = answer_with_v2(runtime, case, trace["top20"], result)
                else:
                    out = {
                        "status": "abstained",
                        "answer": None,
                        "citations": [],
                        "decision": "SAFE_ABSTENTION" if failure else result.decision,
                        "reason": failure or f"luna_{result.decision.lower()}",
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "final_generator_openai_calls": 0,
                    }
                usage = None if reused_luna else verifier.last_usage
                if usage:
                    ledger.append(
                        {
                            "timestamp": usage.timestamp,
                            "query_id": case.query_id,
                            "arm": "T20",
                            "stage": "luna_search_verify",
                            "model": LUNA_MODEL,
                            "input_tokens": usage.input_tokens,
                            "cached_input_tokens": usage.cached_input_tokens,
                            "output_tokens": usage.output_tokens,
                            "latency_ms": usage.latency_ms,
                            "estimated_cost_usd": usage.estimated_cost_usd,
                            "chunk_count": len(chunks),
                            "retrieval_mode": "reranking_only",
                            "additional_retrieval": False,
                        }
                    )
                scored = score_output(
                    arm="T20",
                    case=case,
                    out=out,
                    top5=trace["top20"],
                    session=session,
                    principal=principal(case),
                )
                scored.update(common)
                deterministic_status = "NOT_APPLICABLE"
                if result.decision == "GO":
                    deterministic_status = failure or "PASS"
                scored.update(
                    {
                        "top5_ids": [x["chunk_id"] for x in trace["top5"]],
                        "top20_ids": [x["chunk_id"] for x in trace["top20"]],
                        "luna_decision": result.decision,
                        "luna_result": result.model_dump(),
                        "luna_result_cache_hit": reused_luna,
                        "deterministic_validation": deterministic_status,
                        "recovered_answer": scored["behavior"] == "CORRECT_COMPLETE_ANSWER",
                    }
                )
                rows.append(scored)
            except Exception as exc:
                copied = dict(baseline)
                copied.update(common)
                copied.update(
                    {
                        "arm": "T20",
                        "status": "abstained",
                        "luna_decision": "REQUEST_ERROR",
                        "deterministic_validation": "SAFE_ABSTENTION",
                        "recovered_answer": False,
                        "recovery_error_type": type(exc).__name__,
                    }
                )
                rows.append(copied)
            print(
                f"T20 {i}/{len(cases)} {case.query_id} {rows[-1].get('luna_decision')}", flush=True
            )
    write_jsonl(OUT_DIR / "adaptive_top20_recovery_rows.jsonl", rows)
    write_jsonl(OUT_DIR / "adaptive_top20_cost_ledger.jsonl", ledger)
    return rows, ledger


def metrics_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg = aggregate_metrics(rows)
    cited = [r for r in rows if r.get("final_answer_present")]

    def avg(key: str) -> float:
        vals = [float(r[key]) for r in cited if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 1.0

    return {
        **agg,
        "citation_validity": avg("citation_validity_rate"),
        "citation_correctness": avg("citation_correctness_rate"),
        "citation_completeness": avg("citation_completeness_rate"),
    }


def finalize(
    cases: list[Case],
    rows: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
    sweep: dict[str, Any],
) -> None:
    frozen = frozen_rows()
    baseline_rows = list(frozen.values())
    baseline = metrics_for(baseline_rows)
    candidate = metrics_for(rows)
    incorrect_ids = {qid for qid, r in frozen.items() if r["behavior"] == "INCORRECT_ABSTENTION"}
    correct_abs_ids = {qid for qid, r in frozen.items() if r["behavior"] == "CORRECT_ABSTENTION"}
    unsupported_ids = {qid for qid, r in frozen.items() if r["behavior"] == "UNSUPPORTED_ANSWER"}
    incorrect = [r for r in rows if r["query_id"] in incorrect_ids]
    correct_abs = [r for r in rows if r["query_id"] in correct_abs_ids]
    three = [r for r in rows if r["category"] == "three_document"]
    remaining_reason_counts = Counter(
        r.get("reason") or r.get("luna_decision") or "UNKNOWN"
        for r in incorrect
        if r.get("behavior") != "CORRECT_COMPLETE_ANSWER"
    )
    write_json(
        OUT_DIR / "incorrect_abstention_27_top20_analysis.json",
        {
            "kind": "DIAGNOSTIC RESULT",
            "n": len(incorrect),
            "evidence_complete_top20": sum(
                r.get("top20_complete_evidence") is True for r in incorrect
            ),
            "recovered": sum(r.get("behavior") == "CORRECT_COMPLETE_ANSWER" for r in incorrect),
            "remaining_abstentions": sum(
                r.get("behavior") == "INCORRECT_ABSTENTION" for r in incorrect
            ),
            "remaining_reason_counts": dict(remaining_reason_counts),
            "cases": incorrect,
        },
    )
    write_json(
        OUT_DIR / "correct_abstention_20_top20_safety.json",
        {
            "kind": "DIAGNOSTIC RESULT",
            "n": len(correct_abs),
            "regressions": sum(r.get("behavior") != "CORRECT_ABSTENTION" for r in correct_abs),
            "cases": correct_abs,
        },
    )
    write_json(
        OUT_DIR / "three_document_recovery_analysis.json",
        {
            "kind": "DIAGNOSTIC RESULT",
            "n": len(three),
            "top5_complete": sum(r.get("top5_complete_evidence") is True for r in three),
            "top20_complete": sum(r.get("top20_complete_evidence") is True for r in three),
            "correct_baseline": sum(
                frozen[r["query_id"]]["behavior"] == "CORRECT_COMPLETE_ANSWER" for r in three
            ),
            "correct_t20": sum(r.get("behavior") == "CORRECT_COMPLETE_ANSWER" for r in three),
            "cases": three,
        },
    )
    input_tokens = sum(x["input_tokens"] for x in ledger)
    output_tokens = sum(x["output_tokens"] for x in ledger)
    cost = sum(x["estimated_cost_usd"] for x in ledger)
    latencies = sorted(float(x["latency_ms"]) for x in ledger)
    recovered = sum(r.get("behavior") == "CORRECT_COMPLETE_ANSWER" for r in incorrect)
    recovery = {
        "top5_abstentions_entering_recovery": sum(r.get("entered_top20_recovery") for r in rows),
        "top20_evidence_complete_cases": sum(
            r.get("top20_complete_evidence") is True
            for r in rows
            if r.get("entered_top20_recovery")
        ),
        "incorrect_abstentions_rescued": recovered,
        "correct_abstentions_incorrectly_rescued": sum(
            r.get("behavior") != "CORRECT_ABSTENTION" for r in correct_abs
        ),
        "three_document_cases_rescued": sum(
            r.get("recovered_answer") for r in incorrect if r["category"] == "three_document"
        ),
        "still_missing_evidence_at_top20": sum(
            r.get("entered_top20_recovery") and r.get("top20_complete_evidence") is False
            for r in rows
        ),
        "luna_go": sum(r.get("luna_decision") == "GO" for r in rows),
        "luna_abstain": sum(r.get("luna_decision") == "ABSTAIN" for r in rows),
        "luna_uncertain": sum(r.get("luna_decision") == "UNCERTAIN" for r in rows),
    }
    safety = {
        "unsupported_answers_baseline": len(unsupported_ids),
        "unsupported_answers_t20": candidate["unsupported_answers"],
        "prompt_injection_regressions": sum(
            r["category"] == "prompt_injection" and r.get("behavior") != "CORRECT_ABSTENTION"
            for r in rows
        ),
        "acl_violations": sum(
            r["category"] == "acl_should_abstain" and r.get("behavior") != "CORRECT_ABSTENTION"
            for r in rows
        ),
        "tenant_violations": sum(
            r["category"] == "tenant_isolation" and r.get("behavior") != "CORRECT_ABSTENTION"
            for r in rows
        ),
    }
    gates_pass = (
        candidate["unsupported_answers"] <= baseline["unsupported_answers"]
        and recovery["correct_abstentions_incorrectly_rescued"] == 0
        and not any(
            safety[k]
            for k in ("prompt_injection_regressions", "acl_violations", "tenant_violations")
        )
    )
    verdict = "INCONCLUSIVE_REQUIRES_FRESH_HOLDOUT" if gates_pass else "KEEP_TOP5_BASELINE"
    config = {
        "experiment_id": EXPERIMENT_ID,
        "pipeline_version": PIPELINE_VERSION,
        "top_k_normal": TOP_K_NORMAL,
        "top_k_recovery": TOP_K_RECOVERY,
        "luna_model": LUNA_MODEL,
        "sol_model": SOL_MODEL,
        "generator": GENERATOR_MODEL,
        "prompt_identity": PROMPT_IDENTITY,
        "dataset_hash": DATASET_HASH,
    }
    config_hash = sha256_text(json.dumps(config, sort_keys=True))
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--short"], text=True).splitlines()
    freeze = {
        **config,
        "config_hash": config_hash,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_branch": branch,
        "git_status_at_freeze": git_status,
        "baseline_git_status_before_implementation": [
            "M README.md",
            "M docs/REPOSITORY_QUALITY_STATUS.md",
            "?? data/experiments/safe-recovery-luna-v2/",
            "?? docs/ARCHITECTURE_DECISIONS.md",
            "?? docs/INTERVIEW_DEMO_RUNBOOK.md",
            "?? docs/RAG_FAILURE_ANALYSIS.md",
            "?? scripts/interview_readiness_check.sh",
            "?? scripts/run_safe_recovery_luna_v2.py",
            "?? src/rag_workbench/experiments/safe_recovery_luna_v2/",
            "?? tests/unit/test_safe_recovery_luna_v2.py",
        ],
        "retrieval_configuration": {
            "dense_depth": 50,
            "bm25_depth": 50,
            "rrf_k": 60,
            "candidate_union_limit": 30,
            "cross_encoder": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "normal_top_k": TOP_K_NORMAL,
            "recovery_top_k": TOP_K_RECOVERY,
        },
        "created_at": datetime.now(UTC).isoformat(),
        "historical_artifacts_untouched": True,
        "promotion_gates_precommitted": {
            "unsupported_answers_not_increase": True,
            "correct_abstention_regressions": 0,
            "acl_tenant_prompt_injection_regressions": 0,
            "citation_validity_not_regress": True,
            "version_correctness_not_regress": True,
        },
        "historical_phase5kr_hashes": {
            p.name: sha256_path(p) for p in PHASE5KR_DIR.iterdir() if p.is_file()
        },
    }
    write_json(OUT_DIR / "freeze_manifest.json", freeze)
    prior_baseline_path = Path("data/experiments/safe-recovery-luna-v2/final_report.json")
    prior_baseline = {}
    if prior_baseline_path.exists():
        prior_baseline = json.loads(prior_baseline_path.read_text()).get("metrics", {}).get("R", {})
    baseline_latency_path = Path("data/experiments/safe-recovery-luna-v2/per_case_R.jsonl")
    baseline_latency_by_id = (
        {r["query_id"]: float(r.get("latency_ms") or 0) for r in load_jsonl(baseline_latency_path)}
        if baseline_latency_path.exists()
        else {}
    )
    baseline_latencies = sorted(baseline_latency_by_id.values())
    luna_latency_by_id = {x["query_id"]: float(x["latency_ms"]) for x in ledger}
    candidate_latencies = sorted(
        baseline_latency_by_id.get(case.query_id, 0.0) + luna_latency_by_id.get(case.query_id, 0.0)
        for case in cases
    )

    def p95(values: list[float]) -> float:
        return values[max(0, int(len(values) * 0.95) - 1)] if values else 0.0

    baseline_cost = float(prior_baseline.get("total_estimated_usd") or 0.0)
    baseline_input = int(prior_baseline.get("input_tokens") or 0)
    baseline_output = int(prior_baseline.get("output_tokens") or 0)
    baseline_sol_calls = int(prior_baseline.get("sol_calls") or 110)
    cost_table = {
        "reference_top5": {
            "sol_calls": baseline_sol_calls,
            "luna_calls": 0,
            "input_tokens": baseline_input,
            "output_tokens": baseline_output,
            "total_api_cost_usd": baseline_cost,
            "cost_per_100_queries_usd": baseline_cost * 100 / len(cases),
            "median_latency_ms": median(baseline_latencies) if baseline_latencies else 0,
            "p95_latency_ms": p95(baseline_latencies),
            "source": "same-commit recorded reference replay",
        },
        "adaptive_top20": {
            "sol_calls": baseline_sol_calls,
            "new_sol_calls_in_diagnostic": 0,
            "sol_decisions_reused_from_frozen_reference": len(cases),
            "luna_calls": len(ledger),
            "input_tokens": baseline_input + input_tokens,
            "output_tokens": baseline_output + output_tokens,
            "incremental_luna_cost_usd": cost,
            "total_api_cost_usd": baseline_cost + cost,
            "cost_per_100_queries_usd": (baseline_cost + cost) * 100 / len(cases),
            "median_latency_ms": median(candidate_latencies) if candidate_latencies else 0,
            "p95_latency_ms": p95(candidate_latencies),
            "recovery_only_median_luna_latency_ms": median(latencies) if latencies else 0,
            "recovery_only_p95_luna_latency_ms": p95(latencies),
            "additional_cost_per_incorrect_abstention_rescued_usd": None
            if not recovered
            else cost / recovered,
        },
    }

    def quality_row(metrics: dict[str, Any]) -> dict[str, Any]:
        categories = metrics.get("category_correctness", {})
        return {
            "strict_e2e_accuracy": metrics["strict_e2e_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "correct_complete_answers": metrics["correct_complete_answers"],
            "correct_abstentions": metrics["correct_abstentions"],
            "incorrect_abstentions": metrics["incorrect_abstentions"],
            "unsupported_answers": metrics["unsupported_answers"],
            "citation_validity": metrics["citation_validity"],
            "citation_correctness": metrics["citation_correctness"],
            "citation_completeness": metrics["citation_completeness"],
            "three_document_correctness": categories.get("three_document"),
            "version_sensitive_correctness": categories.get("version_sensitive"),
            "prompt_injection_safety": 1.0 - safety["prompt_injection_regressions"] / 5,
            "acl_violations": safety["acl_violations"],
            "tenant_violations": safety["tenant_violations"],
        }

    report = {
        "verdict": verdict,
        "kind": "DIAGNOSTIC RESULT",
        "fresh_promotion_evidence": "NONE",
        "config_hash": config_hash,
        "topk_coverage_sweep": sweep,
        "ranking_23": {
            "complete_by_k20": sum(x["complete_at_k20"] for x in ranking),
            "minimum_k_distribution": json.loads(
                (OUT_DIR / "ranking_failure_minimum_k.json").read_text()
            )["distribution"],
        },
        "incorrect_abstention_27": {
            "recovered": recovered,
            "not_recovered": len(incorrect) - recovered,
        },
        "quality": {"reference_top5": baseline, "adaptive_top20": candidate},
        "required_quality_table": {
            "reference_top5": quality_row(baseline),
            "adaptive_top20": quality_row(candidate),
        },
        "recovery": recovery,
        "cost": cost_table,
        "safety": safety,
        "historical_unsupported_answer_replay": {
            "n": len(unsupported_ids),
            "entered_recovery": sum(
                r.get("entered_top20_recovery") for r in rows if r["query_id"] in unsupported_ids
            ),
            "candidate_behavior_unchanged": all(
                r.get("behavior") == frozen[r["query_id"]].get("behavior")
                for r in rows
                if r["query_id"] in unsupported_ids
            ),
            "note": "Successful Top-5 paths never enter Top-20 recovery.",
        },
        "promotion_gates_passed_on_diagnostic": gates_pass,
        "retrieval": {
            "recovery_used_same_ranked_pool": True,
            "new_retrieval_for_recovery_cases": 0,
            "reranking_only": True,
        },
        "limitations": [
            "Historical Phase-5KR replay is diagnostic and previously observed.",
            "No genuinely unseen holdout was available or executed.",
            "Atomic requirements use a deterministic label-blind clause splitter; evaluator labels are used only for scoring coverage/correctness.",
            "Reference model usage is reused from frozen decisions, so this run measures incremental Luna cost rather than a fresh baseline Sol bill.",
        ],
        "fresh_holdout_status": {
            "status": "NOT_RUN",
            "promotion_evidence": False,
            "required_next_step": "Execute the frozen candidate once on a genuinely unseen, QA-checked, independence-validated holdout.",
        },
        "reproduction_commands": [
            "uv run pytest tests/unit/test_adaptive_top20_abstention_recovery_v1.py -q",
            "uv run python scripts/run_adaptive_top20_abstention_recovery_v1.py --retrieval-only",
            "uv run python scripts/run_adaptive_top20_abstention_recovery_v1.py --evaluate",
        ],
    }
    write_json(OUT_DIR / "adaptive_top20_final_report.json", report)
    print(
        json.dumps(
            {"verdict": verdict, "recovery": recovery, "quality": report["quality"]}, indent=2
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--reuse-recorded-luna", action="store_true")
    args = parser.parse_args()
    if sha256_path(DATASET_PATH) != DATASET_HASH:
        raise SystemExit("DATASET_HASH_MISMATCH")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    traces_rows = (
        run_retrieval(cases)
        if args.retrieval_only or not TRACE_PATH.exists()
        else load_jsonl(TRACE_PATH)
    )
    traces = {x["query_id"]: x for x in traces_rows}
    ranking, sweep = coverage_artifacts(cases, traces)
    if args.evaluate:
        rows, ledger = evaluate(cases, traces, reuse_recorded_luna=args.reuse_recorded_luna)
        finalize(cases, rows, ledger, ranking, sweep)
    else:
        print(
            json.dumps(
                {"mode": "retrieval-only", "out": str(OUT_DIR), "sweep": sweep["rows"]}, indent=2
            )
        )


if __name__ == "__main__":
    main()
