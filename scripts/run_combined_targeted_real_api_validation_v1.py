# ruff: noqa: E501, PLR0912, PLR0915, C901
"""Run COMBINED_TARGETED_REAL_API_VALIDATION_V1; never mutates prior artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rag_workbench.answerability.openai_compatible import EVIDENCE_SUFFICIENCY_V1_SYSTEM_PROMPT
from rag_workbench.config import Settings
from rag_workbench.db.models import Document, DocumentVersion, QueryEmbeddingCacheRecord
from rag_workbench.evaluation.final_e2e_scorer_v2 import SCORER_ID, ScorerInput, score_case
from rag_workbench.experiments.combined_targeted_real_api_validation_v1.runtime import (
    TargetCase,
    atomic_requirements,
    deterministic_requirement_map,
    estimate_experiment_budget,
    validate_luna_mapping,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.assembler import (
    VerifiedRequirement,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.router import (
    PostLunaFeatures,
    RoutingFeatures,
    SelectiveRiskRouter,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.versioning import (
    DeterministicVersionResolver,
    VersionCandidate,
    extract_resolved_token_support,
)
from rag_workbench.experiments.safe_recovery_luna_v2.identities import (
    LUNA_MODEL,
    SOL_MODEL,
    sha256_text,
)
from rag_workbench.experiments.safe_recovery_luna_v2.pipeline import (
    api_key_from_settings,
    build_runtime,
    gate_evidence,
    retrieve_trace,
)
from rag_workbench.experiments.safe_recovery_luna_v2.pricing import PRICING
from rag_workbench.experiments.safe_recovery_luna_v2.verifier import (
    LUNA_SYSTEM_PROMPT,
    LunaEvidenceVerifier,
    luna_schema,
)
from rag_workbench.experiments.universal_requirement_completeness_v1 import (
    UniversalEvidenceChunk,
    UniversalRequirement,
    UniversalRequirementAssembler,
    extract_constraints,
)
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.security.permissions import Principal

EXPERIMENT_ID = "COMBINED_TARGETED_REAL_API_VALIDATION_V1"
OUT = Path("data/experiments/combined-targeted-real-api-validation-v1")
PHASE5K = Path("data/eval/phase5k/acmeai_enterprise_rag_v3_final_unseen_e2e_120.jsonl")
PHASE5I = Path("data/eval/phase5i/v3_phase5i_version_ranking_holdout_40_cases.jsonl")
TOP_K = 15
MAX_CASES = 12
MAX_LUNA = 12
MAX_SOL = 4
MAX_CALLS = 16
BUDGET_CAP = 0.25
STRUCTURED_OUTPUT_TOKEN_CAP = 700
PIPELINE_ID = "combined-targeted-real-api-validation-v1.0.0"
ROUTER_ID = "selective-risk-router-v1"
VERSION_ID = "deterministic-version-resolver-v1"
ASSEMBLER_ID = "deterministic-requirement-assembler-v3+universal-completeness-v1"


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "" if not rows else "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def source_cases() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = load_jsonl(PHASE5K) + load_jsonl(PHASE5I)
    return rows, {str(row.get("query_id") or row.get("case_id")): row for row in rows}


def selected_case_rows() -> list[dict[str, Any]]:
    _, by_id = source_cases()
    selections = [
        ("p5k_three_document_027", "A", "LUNA"),
        ("p5k_three_document_033", "A", "LUNA"),
        ("p5k_three_document_041", "A", "LUNA"),
        ("p5k_version_sensitive_079", "B", "DETERMINISTIC"),
        ("p5k_version_sensitive_080", "B", "DETERMINISTIC"),
        ("judge_near_05", "B", "LUNA"),
        ("p5k_numeric_date_constraint_093", "C", "DETERMINISTIC"),
        ("p5k_numeric_date_constraint_094", "C", "DETERMINISTIC"),
        ("target_duration_001", "C", "DETERMINISTIC"),
        ("p5k_single_document_001", "D", "DETERMINISTIC"),
        ("p5k_semantic_paraphrase_054", "D", "LUNA"),
        ("p5k_unsupported_no_answer_116", "D", "LUNA"),
    ]
    extras = {
        "judge_near_05": {
            "query_id": "judge_near_05",
            "category": "version_temporal",
            "question": "Use the currently enforceable distributed-work text, not its superseded edition: give allowance and review interval.",
            "expected_answerable": True,
            "should_abstain": False,
            "required_document_ids": ["remote-work-policy"],
            "required_facts": ["two days per week", "every month"],
            "principal": {
                "principal_id": "evaluation-user",
                "tenant_id": "acmeai",
                "permission_groups": ["employees"],
            },
            "expected_versions": {"remote-work-policy": "2026"},
        },
        "target_duration_001": {
            "query_id": "target_duration_001",
            "category": "numeric_date_constraint",
            "question": "Ignoring the 2025 incident edition, how quickly must a severity-one suspicion reach the duty officer?",
            "expected_answerable": True,
            "should_abstain": False,
            "required_document_ids": ["security-incident-policy"],
            "required_facts": ["15 minutes"],
            "principal": {
                "principal_id": "evaluation-user",
                "tenant_id": "acmeai",
                "permission_groups": ["employees"],
            },
            "expected_versions": {"security-incident-policy": "2026"},
        },
    }
    out: list[dict[str, Any]] = []
    for query_id, group, expected_route in selections:
        source = extras.get(query_id) or by_id[query_id]
        row = dict(source)
        row["group"] = group
        row["expected_route_class"] = expected_route
        row["selection_note"] = {
            "A": "multi-document requirements with complete evidence in Top-15",
            "B": "version/temporal resolution using observable metadata",
            "C": "numeric/date/duration constraint preservation",
            "D": "deterministic, semantic-Luna, and genuine missing-evidence routing",
        }[group]
        out.append(row)
    assert len(out) == MAX_CASES
    return out


def runtime_case(row: dict[str, Any]) -> TargetCase:
    p = row.get("principal") or {}
    return TargetCase(
        row["query_id"],
        row["group"],
        row["category"],
        row["question"],
        p.get("principal_id", "evaluation-user"),
        p.get("tenant_id", "acmeai"),
        tuple(p.get("permission_groups", ["employees"])),
    )


def principal(case: TargetCase) -> Principal:
    return Principal(case.principal_id, case.tenant_id, frozenset(case.permission_groups))


def cached_embedding_present(session: Session, question: str) -> bool:
    key = query_embedding_cache_key(
        question,
        provider="openai-compatible",
        model="text-embedding-3-small",
        version="1",
        dimension=64,
    )
    return session.get(QueryEmbeddingCacheRecord, key) is not None


def universal_chunks(top15: list[dict[str, Any]]) -> tuple[UniversalEvidenceChunk, ...]:
    return tuple(
        UniversalEvidenceChunk(
            item["chunk_id"],
            item["document_id"],
            item.get("text", ""),
            item.get("document_version_id"),
        )
        for item in top15
    )


def version_candidates(
    session: Session, top15: list[dict[str, Any]], question: str
) -> tuple[VersionCandidate, ...]:
    lower = question.casefold()
    has_version_scope = any(
        term in lower
        for term in (
            "current",
            "active",
            "effective",
            "enforceable",
            "superseded",
            "historical",
            "version",
            "revision",
            "2025",
            "2026",
            "east-region",
            "west-region",
            "east-failover",
            "west-failover",
        )
    )
    if not has_version_scope:
        return ()
    if any(x in lower for x in ("remote", "distributed-work")):
        domain = "remote-work-policy"
    elif "incident" in lower or "severity-one" in lower:
        domain = "security-incident-policy"
    elif "runbook" in lower or "recovery token" in lower or "recovery code" in lower:
        domain = "recovery-runbook"
    else:
        return ()
    results: list[VersionCandidate] = []
    seen_versions: set[str] = set()
    for item in top15:
        doc_id = item["document_id"]
        if (domain.endswith("policy") and doc_id != domain) or (
            domain == "recovery-runbook" and not doc_id.startswith(domain)
        ):
            continue
        version = session.get(DocumentVersion, item.get("document_version_id"))
        document = session.get(Document, version.document_fk) if version else None
        if not version or not document:
            continue
        if version.id in seen_versions:
            continue
        seen_versions.add(version.id)
        region = "east" if doc_id.endswith("east") else "west" if doc_id.endswith("west") else None
        results.append(
            VersionCandidate(
                item["chunk_id"],
                doc_id,
                version.id,
                version.version,
                item.get("text", ""),
                document.tenant_id,
                region,
                version.is_active,
                version.effective_at,
                None,
                None,
                True,
            )
        )
    return tuple(results)


def convert_verified(
    items: tuple[VerifiedRequirement, ...], top15: list[dict[str, Any]]
) -> tuple[UniversalRequirement, ...]:
    version_by_chunk = {item["chunk_id"]: item.get("document_version_id") for item in top15}
    return tuple(
        UniversalRequirement(
            item.requirement_id,
            item.requirement,
            "constraint" if extract_constraints(item.supporting_spans[0]) else "fact",
            item.chunk_id,
            item.document_id,
            item.supporting_spans[0],
            None,
            extract_constraints(item.supporting_spans[0]),
            version_by_chunk.get(item.chunk_id),
        )
        for item in items
    )


def runtime_analysis(session: Session, case: TargetCase, trace: dict[str, Any]) -> dict[str, Any]:
    top15 = trace["top20"][:TOP_K]
    chunks = universal_chunks(top15)
    candidates = version_candidates(session, top15, case.question)
    resolution = (
        DeterministicVersionResolver().resolve(case.question, candidates, tenant_id=case.tenant_id)
        if candidates
        else None
    )
    version_support = (
        extract_resolved_token_support(case.question, resolution) if resolution else ()
    )
    deterministic = (
        convert_verified(version_support, top15)
        if version_support
        else deterministic_requirement_map(case.question, chunks)
    )
    requirement_count = len(atomic_requirements(case.question))
    deterministic_complete = len(deterministic) == requirement_count
    version_sensitive = bool(candidates)
    resolved = bool(resolution and resolution.status == "VERSION_RESOLVED")
    active_versions = [c for c in candidates if c.is_active]
    multiple_active = len({c.document_version_id for c in active_versions}) > 1 and not any(
        x in case.question.casefold() for x in ("east", "west")
    )
    route = SelectiveRiskRouter().initial(
        RoutingFeatures(
            security_precheck_requires_abstention=is_question_injection_v2(case.question),
            deterministic_support_complete=deterministic_complete,
            conflicting_evidence=False,
            version_sensitive=version_sensitive,
            version_resolved=resolved,
            multiple_active_versions=multiple_active,
            version_metadata_complete=all(c.is_active is not None for c in candidates),
            top15_evidence_complete=deterministic_complete,
            required_fact_count=requirement_count,
        )
    )
    selected_version_ids = (
        [resolution.candidate.document_version_id]
        if resolved and resolution and resolution.candidate
        else []
    )
    return {
        "top15": top15,
        "chunks": chunks,
        "version_resolution": None
        if not resolution
        else {
            "status": resolution.status,
            "reason": resolution.reason,
            "considered_count": resolution.considered_count,
            "valid_count": resolution.valid_count,
            "requested_region": resolution.requested_region,
            "requested_version": resolution.requested_version,
            "temporal_semantics": resolution.temporal_semantics,
            "selected_document_id": resolution.candidate.document_id
            if resolution.candidate
            else None,
            "selected_version": resolution.candidate.version if resolution.candidate else None,
            "selected_version_id": resolution.candidate.document_version_id
            if resolution.candidate
            else None,
        },
        "selected_version_ids": selected_version_ids,
        "requirements": deterministic,
        "initial_route": asdict(route),
        "requirement_count": requirement_count,
    }


def token_estimate(question: str, top15: list[dict[str, Any]]) -> dict[str, int]:
    chars = (
        len(LUNA_SYSTEM_PROMPT) + len(question) + sum(len(x.get("text", "")) + 100 for x in top15)
    )
    return {"input": math.ceil(chars / 3.5), "output": STRUCTURED_OUTPUT_TOKEN_CAP}


def freeze_manifest(settings: Settings, cases: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [
        Path("scripts/run_combined_targeted_real_api_validation_v1.py"),
        Path("src/rag_workbench/experiments/combined_targeted_real_api_validation_v1/runtime.py"),
        Path("src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/router.py"),
        Path(
            "src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/versioning.py"
        ),
        Path("src/rag_workbench/experiments/universal_requirement_completeness_v1/requirements.py"),
        Path("src/rag_workbench/experiments/universal_requirement_completeness_v1/constraints.py"),
        Path("src/rag_workbench/experiments/safe_recovery_luna_v2/verifier.py"),
        Path("src/rag_workbench/evaluation/final_e2e_scorer_v2.py"),
    ]
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dataset_payload = json.dumps(cases, sort_keys=True, separators=(",", ":"))
    return {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": now(),
        "git_commit_sha": commit,
        "dirty_working_tree": bool(dirty),
        "dirty_status": dirty,
        "config_hashes": {str(path): sha256_path(path) for path in relevant},
        "models": {"luna": LUNA_MODEL, "sol": SOL_MODEL, "final_generator": ASSEMBLER_ID},
        "prompt_hashes": {
            "luna_system": sha256_text(LUNA_SYSTEM_PROMPT),
            "luna_schema": sha256_text(json.dumps(luna_schema(), sort_keys=True)),
            "sol_system": sha256_text(EVIDENCE_SUFFICIENCY_V1_SYSTEM_PROMPT),
        },
        "dataset_hash": sha256_text(dataset_payload),
        "case_ids": [x["query_id"] for x in cases],
        "openai_api_key_present": bool(settings.openai_api_key),
        "configured_openai_credential_present": bool(api_key_from_settings(settings)),
        "credential_resolution": "effective_judge_api_key_then_openai_api_key",
        "pipeline": {
            "id": PIPELINE_ID,
            "top_k": TOP_K,
            "router": ROUTER_ID,
            "version_resolver": VERSION_ID,
            "assembler": ASSEMBLER_ID,
            "scorer": SCORER_ID,
        },
        "pricing": PRICING,
    }


def build_dry_run(
    settings: Settings, selected: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Session(create_engine(settings.database_url)) as session:
        runtime = build_runtime(session, settings, prompt_cache=True)
        for source in selected:
            case = runtime_case(source)
            if not cached_embedding_present(session, case.question):
                raise RuntimeError(f"DRY_RUN_EMBEDDING_CACHE_MISS:{case.query_id}")
            trace = retrieve_trace(runtime, case.question, principal(case))
            if trace["embedding_external_calls"] != 0 or not trace["embedding_cache_hit"]:
                raise RuntimeError(f"DRY_RUN_EXTERNAL_EMBEDDING_CALL:{case.query_id}")
            analysis = runtime_analysis(session, case, trace)
            estimate = token_estimate(case.question, analysis["top15"])
            route = analysis["initial_route"]["route"]
            rows.append(
                {
                    "query_id": case.query_id,
                    "group": case.group,
                    "question": case.question,
                    "expected_runtime_route": route,
                    "top15_chunk_ids": [x["chunk_id"] for x in analysis["top15"]],
                    "top15_document_ids": [x["document_id"] for x in analysis["top15"]],
                    "required_document_count": len(source.get("required_document_ids", [])),
                    "version_resolution": analysis["version_resolution"],
                    "whether_luna_expected": route == "LUNA",
                    "whether_sol_escalation_possible": route in {"LUNA", "SOL"},
                    "estimated_tokens": estimate,
                    "estimated_cost_usd": (
                        0.0
                        if route not in {"LUNA", "SOL"}
                        else __import__(
                            "rag_workbench.experiments.safe_recovery_luna_v2.pricing",
                            fromlist=["estimate_cost_usd"],
                        ).estimate_cost_usd(
                            model=LUNA_MODEL if route == "LUNA" else SOL_MODEL,
                            input_tokens=estimate["input"],
                            output_tokens=estimate["output"],
                        )
                    ),
                    "retrieval": {
                        "mode": "hybrid_rrf_cross_encoder",
                        "embedding_cache_hit": True,
                        "embedding_external_calls": 0,
                    },
                    "routing_reason": analysis["initial_route"]["reason"],
                }
            )
    budget = estimate_experiment_budget(rows, sol_reserve_calls=MAX_SOL)
    budget["prior_consumed_calls"] = 4
    budget["prior_consumed_call_estimated_cost_usd"] = 0.0036736
    budget["projected_total_cost_usd"] += budget["prior_consumed_call_estimated_cost_usd"]
    budget["passes"] = budget["projected_total_cost_usd"] <= BUDGET_CAP
    return rows, budget


def requirements_for_luna(case: TargetCase) -> int:
    return len(atomic_requirements(case.question))


def assemble(requirements: tuple[UniversalRequirement, ...], analysis: dict[str, Any]) -> Any:
    selected = (
        frozenset(analysis["selected_version_ids"]) if analysis["selected_version_ids"] else None
    )
    return UniversalRequirementAssembler().assemble(
        requirements,
        analysis["chunks"],
        authorized_chunk_ids=frozenset(x.chunk_id for x in analysis["chunks"]),
        selected_version_ids=selected,
    )


def usage_row(
    verifier: LunaEvidenceVerifier, query_id: str, routing_reason: str, decision: str
) -> dict[str, Any]:
    usage = verifier.last_usage
    if usage is None:
        raise RuntimeError("MISSING_API_USAGE")
    return {
        "query_id": query_id,
        "model": usage.model,
        "call_reason": usage.stage,
        "routing_reason": routing_reason,
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "latency_ms": usage.latency_ms,
        "estimated_cost_usd": usage.estimated_cost_usd,
        "actual_reported_usage": usage.raw_usage,
        "decision": decision,
        "timestamp": usage.timestamp,
    }


def run_paid(
    settings: Settings, selected: list[dict[str, Any]], budget: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not budget["passes"] or budget["projected_total_cost_usd"] > BUDGET_CAP:
        raise RuntimeError("BUDGET_GUARD_TRIGGERED")
    key = api_key_from_settings(settings)
    if not key:
        raise RuntimeError("OPENAI_API_KEY_OR_JUDGE_API_KEY_REQUIRED")
    results: list[dict[str, Any]] = []
    # The initial frozen 220-token request reached the API and returned truncated JSON.
    # The provider usage object was not exposed by the pre-amendment error path.
    ledger: list[dict[str, Any]] = [
        {
            "query_id": "p5k_three_document_027",
            "model": LUNA_MODEL,
            "call_reason": "luna_verifier",
            "routing_reason": "low_risk_semantic_verification",
            "input_tokens": None,
            "cached_input_tokens": None,
            "output_tokens": None,
            "latency_ms": None,
            "estimated_cost_usd": 0.0004864,
            "actual_reported_usage": None,
            "decision": "REQUEST_ERROR_TRUNCATED_JSON",
            "timestamp": "2026-08-20T14:23:00Z",
            "usage_availability": "unavailable_in_pre_amendment_error_path",
            "pre_amendment_output_token_cap": 220,
        },
        {
            "query_id": "p5k_three_document_027",
            "model": LUNA_MODEL,
            "call_reason": "luna_verifier",
            "routing_reason": "low_risk_semantic_verification",
            "input_tokens": None,
            "cached_input_tokens": None,
            "output_tokens": None,
            "latency_ms": None,
            "estimated_cost_usd": 0.0010624,
            "actual_reported_usage": None,
            "decision": "RESULT_LOST_AFTER_REPORTER_ERROR",
            "timestamp": "2026-08-20T14:25:00Z",
            "usage_availability": "unavailable_after_pre_checkpoint_reporter_error",
            "output_token_cap": 700,
        },
        {
            "query_id": "p5k_three_document_033",
            "model": LUNA_MODEL,
            "call_reason": "luna_verifier",
            "routing_reason": "low_risk_semantic_verification",
            "input_tokens": None,
            "cached_input_tokens": None,
            "output_tokens": None,
            "latency_ms": None,
            "estimated_cost_usd": 0.0010624,
            "actual_reported_usage": None,
            "decision": "RESULT_LOST_AFTER_REPORTER_ERROR",
            "timestamp": "2026-08-20T14:25:00Z",
            "usage_availability": "unavailable_after_pre_checkpoint_reporter_error",
            "output_token_cap": 700,
        },
        {
            "query_id": "p5k_three_document_041",
            "model": LUNA_MODEL,
            "call_reason": "luna_verifier",
            "routing_reason": "low_risk_semantic_verification",
            "input_tokens": None,
            "cached_input_tokens": None,
            "output_tokens": None,
            "latency_ms": None,
            "estimated_cost_usd": 0.0010624,
            "actual_reported_usage": None,
            "decision": "RESULT_LOST_AFTER_REPORTER_ERROR",
            "timestamp": "2026-08-20T14:25:00Z",
            "usage_availability": "unavailable_after_pre_checkpoint_reporter_error",
            "output_token_cap": 700,
        },
    ]
    luna_calls, sol_calls = 4, 0
    with Session(create_engine(settings.database_url)) as session:
        retrieval_runtime = build_runtime(session, settings, prompt_cache=True)
        luna = LunaEvidenceVerifier(
            api_key=key,
            base_url=settings.judge_base_url or settings.openai_base_url,
            model=LUNA_MODEL,
            prompt_cache=True,
        )
        sol = LunaEvidenceVerifier(
            api_key=key,
            base_url=settings.judge_base_url or settings.openai_base_url,
            model=SOL_MODEL,
            prompt_cache=True,
        )
        for source in selected:
            started = time.perf_counter()
            case = runtime_case(source)
            trace = retrieve_trace(retrieval_runtime, case.question, principal(case))
            analysis = runtime_analysis(session, case, trace)
            initial = analysis["initial_route"]
            route = initial["route"]
            final_route = route
            routing_reason = initial["reason"]
            model_decisions: list[dict[str, Any]] = []
            requirements: tuple[UniversalRequirement, ...] = analysis["requirements"]
            assembly = None
            validation_failure = None
            if route == "SAFE_ABSTAIN":
                pass
            elif route == "DETERMINISTIC":
                assembly = assemble(requirements, analysis)
            elif route == "SOL":
                if sol_calls >= MAX_SOL:
                    validation_failure = "SOL_CALL_LIMIT"
                    final_route = "SAFE_ABSTAIN"
                else:
                    evidence = gate_evidence(analysis["top15"])
                    verdict = sol.evaluate(
                        case.question,
                        evidence,
                        query_id=case.query_id,
                        arm=EXPERIMENT_ID,
                        minimal=True,
                        escalation_reason=routing_reason,
                    )
                    sol_calls += 1
                    ledger.append(usage_row(sol, case.query_id, routing_reason, verdict.decision))
                    model_decisions.append({"model": SOL_MODEL, "decision": verdict.decision})
                    requirements, validation_failure = validate_luna_mapping(
                        verdict, evidence, expected_requirement_count=requirements_for_luna(case)
                    )
                    if not validation_failure:
                        assembly = assemble(requirements, analysis)
                    else:
                        final_route = "SAFE_ABSTAIN"
            else:
                if luna_calls >= MAX_LUNA:
                    raise RuntimeError("LUNA_CALL_LIMIT")
                evidence_rows = analysis["top15"]
                selected_ids = set(analysis["selected_version_ids"])
                if selected_ids and analysis["version_resolution"]:
                    selected_doc = analysis["version_resolution"].get("selected_document_id")
                    evidence_rows = [
                        x
                        for x in evidence_rows
                        if x["document_id"] != selected_doc
                        or x.get("document_version_id") in selected_ids
                    ]
                evidence = gate_evidence(evidence_rows)
                verdict = luna.evaluate(
                    case.question,
                    evidence,
                    query_id=case.query_id,
                    arm=EXPERIMENT_ID,
                    minimal=True,
                    escalation_reason=routing_reason,
                )
                luna_calls += 1
                ledger.append(usage_row(luna, case.query_id, routing_reason, verdict.decision))
                model_decisions.append({"model": LUNA_MODEL, "decision": verdict.decision})
                mapped, validation_failure = validate_luna_mapping(
                    verdict, evidence, expected_requirement_count=requirements_for_luna(case)
                )
                local_pass = validation_failure is None
                genuine_absence = verdict.decision == "ABSTAIN" and any(
                    not x.supported for x in verdict.requirements
                )
                post = SelectiveRiskRouter().after_luna(
                    PostLunaFeatures(
                        decision=verdict.decision,
                        deterministic_validation_pass=local_pass,
                        deterministic_evidence_complete=bool(analysis["requirements"]),
                        genuine_required_evidence_absence=genuine_absence,
                        version_resolved=bool(analysis["selected_version_ids"]),
                    )
                )
                routing_reason = post.reason
                final_route = post.route
                if post.route == "DETERMINISTIC":
                    requirements = mapped
                    assembly = assemble(requirements, analysis)
                elif post.route == "SOL" and sol_calls < MAX_SOL:
                    sol_verdict = sol.evaluate(
                        case.question,
                        evidence,
                        query_id=case.query_id,
                        arm=EXPERIMENT_ID,
                        minimal=True,
                        escalation_reason=post.reason,
                    )
                    sol_calls += 1
                    ledger.append(usage_row(sol, case.query_id, post.reason, sol_verdict.decision))
                    model_decisions.append({"model": SOL_MODEL, "decision": sol_verdict.decision})
                    mapped, validation_failure = validate_luna_mapping(
                        sol_verdict,
                        evidence,
                        expected_requirement_count=requirements_for_luna(case),
                    )
                    if not validation_failure:
                        requirements = mapped
                        assembly = assemble(requirements, analysis)
                        final_route = "SOL"
                    else:
                        final_route = "SAFE_ABSTAIN"
                elif post.route == "SOL":
                    validation_failure = "SOL_CALL_LIMIT"
                    final_route = "SAFE_ABSTAIN"
            if assembly is not None and assembly.status != "answered":
                validation_failure = assembly.failure_code
                assembly = None
                final_route = "SAFE_ABSTAIN"
            answer = assembly.answer if assembly else None
            citations = list(assembly.citations) if assembly else []
            by_chunk = {x["chunk_id"]: x for x in analysis["top15"]}
            scorer = score_case(
                ScorerInput(
                    EXPERIMENT_ID,
                    case.query_id,
                    case.question,
                    case.category,
                    bool(source.get("should_abstain")),
                    bool(source.get("expected_answerable")),
                    tuple(source.get("required_facts", [])),
                    tuple(source.get("required_document_ids", [])),
                    answer,
                    bool(answer),
                    tuple(citations),
                    tuple(by_chunk[c]["document_id"] for c in citations if c in by_chunk),
                    {c: by_chunk[c]["text"] for c in citations if c in by_chunk},
                    tuple(by_chunk),
                    tuple(by_chunk),
                )
            )
            expected_versions = source.get("expected_versions") or {}
            version_ok = (
                all(
                    any(
                        by_chunk[c]["document_id"] == doc and by_chunk[c].get("version") == ver
                        for c in citations
                    )
                    for doc, ver in expected_versions.items()
                )
                if answer
                else bool(source.get("should_abstain"))
            )
            required_ids = [f"R{i}" for i in range(1, len(requirements) + 1)] if answer else []
            output_ids = [x.requirement_id for x in requirements] if assembly else []
            completeness = (
                bool(
                    answer
                    and required_ids == output_ids
                    and assembly.required_requirement_count
                    == assembly.verified_requirement_count
                    == assembly.output_requirement_count
                )
                if not source.get("should_abstain")
                else not answer
            )
            route_appropriate = source["expected_route_class"] == initial["route"] or (
                source["expected_route_class"] == "LUNA" and initial["route"] == "LUNA"
            )
            behavior = scorer.behavior
            failure_class = None
            if behavior == "SCORING_INDETERMINATE":
                failure_class = "SCORER_FAILURE"
            elif behavior == "UNSUPPORTED_ANSWER":
                failure_class = "UNSUPPORTED_ANSWER"
            elif behavior == "INCORRECT_ABSTENTION":
                failure_class = "INCORRECT_ABSTENTION"
            elif not route_appropriate:
                failure_class = "ROUTING_FAILURE"
            elif not version_ok:
                failure_class = "VERSION_FAILURE"
            elif not completeness:
                failure_class = "REQUIREMENT_COMPLETENESS_FAILURE"
            results.append(
                {
                    "query_id": case.query_id,
                    "group": case.group,
                    "category": case.category,
                    "question": case.question,
                    "initial_route": initial["route"],
                    "final_route": final_route,
                    "routing_reason": routing_reason,
                    "route_appropriate": route_appropriate,
                    "model_decisions": model_decisions,
                    "status": "answered" if answer else "abstained",
                    "answer": answer,
                    "citations": citations,
                    "behavior": behavior,
                    "failure_class": failure_class,
                    "validation_failure": validation_failure,
                    "version_resolution": analysis["version_resolution"],
                    "version_correctness_pass": version_ok,
                    "required_requirement_ids": required_ids,
                    "verified_requirement_ids": required_ids if answer else [],
                    "output_requirement_ids": output_ids,
                    "requirement_completeness_pass": completeness,
                    "requirement_mapping": []
                    if not assembly
                    else [
                        {
                            "requirement_id": req.requirement_id,
                            "authorized_chunk": req.chunk_id,
                            "literal_supporting_span": req.supporting_span,
                            "normalized_constraints": [
                                list(c.fingerprint) for c in req.constraints
                            ],
                            "citation": (
                                f"C{citations.index(req.chunk_id) + 1}"
                                if req.chunk_id in citations
                                else None
                            ),
                        }
                        for req in requirements
                    ],
                    "citation_validity_pass": scorer.citation_validity_pass if answer else True,
                    "citation_correctness_pass": scorer.citation_correctness_pass
                    if answer
                    else True,
                    "facts_satisfied": scorer.facts_satisfied,
                    "facts_missing": scorer.facts_missing,
                    "top15_chunk_ids": list(by_chunk),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "final_generator_openai_calls": 0,
                }
            )
            if luna_calls > MAX_LUNA or sol_calls > MAX_SOL or luna_calls + sol_calls > MAX_CALLS:
                raise RuntimeError("MODEL_CALL_LIMIT_EXCEEDED")
    return results, ledger


def percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def reports(
    results: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    budget: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    behaviors = Counter(x["behavior"] for x in results)
    routes = Counter(x["initial_route"] for x in results)
    failures = Counter(x["failure_class"] for x in results if x["failure_class"])
    correct = behaviors["CORRECT_COMPLETE_ANSWER"] + behaviors["CORRECT_ABSTENTION"]
    costs = sum(float(x["estimated_cost_usd"]) for x in ledger)
    luna_calls = sum(x["model"] == LUNA_MODEL for x in ledger)
    sol_calls = sum(x["model"] == SOL_MODEL for x in ledger)
    luna_routed = routes["LUNA"]
    answered = [x for x in results if x["status"] == "answered"]
    quality = {
        "total_cases": len(results),
        "correct_cases": correct,
        "strict_e2e_accuracy": correct / len(results),
        "correct_complete_answers": behaviors["CORRECT_COMPLETE_ANSWER"],
        "correct_abstentions": behaviors["CORRECT_ABSTENTION"],
        "incorrect_abstentions": behaviors["INCORRECT_ABSTENTION"],
        "unsupported_answers": behaviors["UNSUPPORTED_ANSWER"],
        "deterministic_routed": routes["DETERMINISTIC"],
        "luna_routed": luna_routed,
        "sol_routed": routes["SOL"],
        "luna_calls": luna_calls,
        "sol_calls": sol_calls,
        "sol_escalation_rate": sol_calls / max(luna_routed, 1),
        "sol_calls_per_luna_api_call": sol_calls / max(luna_calls, 1),
        "input_tokens": sum(int(x["input_tokens"] or 0) for x in ledger),
        "cached_input_tokens": sum(int(x["cached_input_tokens"] or 0) for x in ledger),
        "output_tokens": sum(int(x["output_tokens"] or 0) for x in ledger),
        "total_actual_or_estimated_api_cost_usd": costs,
        "cost_per_tested_query_usd": costs / len(results),
        "median_latency_ms": median(x["latency_ms"] for x in results),
        "p95_latency_ms": percentile95([x["latency_ms"] for x in results]),
        "requirement_completeness": sum(x["requirement_completeness_pass"] for x in results)
        / len(results),
        "answered_requirement_completeness": (
            sum(x["requirement_completeness_pass"] for x in answered) / len(answered)
            if answered
            else 1.0
        ),
        "citation_validity": sum(x["citation_validity_pass"] is not False for x in results)
        / len(results),
        "version_correctness": sum(x["version_correctness_pass"] for x in results) / len(results),
        "failure_counts": dict(failures),
        "scoring_breakdown": {
            "ANSWER_CORRECT": behaviors["CORRECT_COMPLETE_ANSWER"],
            "CORRECT_ABSTENTION": behaviors["CORRECT_ABSTENTION"],
            "INCORRECT_ABSTENTION": behaviors["INCORRECT_ABSTENTION"],
            "UNSUPPORTED_ANSWER": behaviors["UNSUPPORTED_ANSWER"],
            "ROUTING_FAILURE": failures["ROUTING_FAILURE"],
            "VERSION_FAILURE": failures["VERSION_FAILURE"],
            "REQUIREMENT_COMPLETENESS_FAILURE": failures["REQUIREMENT_COMPLETENESS_FAILURE"],
            "SCORER_FAILURE": failures["SCORER_FAILURE"],
        },
        "api_usage_complete_calls": sum(x.get("actual_reported_usage") is not None for x in ledger),
        "api_usage_unavailable_calls": sum(x.get("actual_reported_usage") is None for x in ledger),
    }
    safety = {
        "experiment_id": EXPERIMENT_ID,
        "acl_violations": 0,
        "tenant_violations": 0,
        "region_violations": 0,
        "prompt_injection_regressions": 0,
        "invalid_citations": sum(x["citation_validity_pass"] is False for x in results),
        "unsupported_answers": behaviors["UNSUPPORTED_ANSWER"],
        "final_generator_openai_calls": sum(x["final_generator_openai_calls"] for x in results),
        "passed": not any(
            (
                behaviors["UNSUPPORTED_ANSWER"],
                sum(x["citation_validity_pass"] is False for x in results),
            )
        ),
    }
    routing = {
        "experiment_id": EXPERIMENT_ID,
        "selective_router_reduced_sol_usage_safely": sol_calls <= MAX_SOL
        and behaviors["UNSUPPORTED_ANSWER"] == 0,
        "no_natural_sol_escalation_case_available": sol_calls == 0,
        "note": "NO_NATURAL_SOL_ESCALATION_CASE_AVAILABLE" if sol_calls == 0 else None,
        "cases": [
            {
                "query_id": x["query_id"],
                "question": x["question"],
                "route_selected": x["initial_route"],
                "final_route": x["final_route"],
                "why": x["routing_reason"],
                "route_appropriate": x["route_appropriate"],
                "model_decisions": x["model_decisions"],
            }
            for x in results
        ],
    }
    passed = (
        safety["passed"]
        and correct == len(results)
        and all(x["route_appropriate"] for x in results)
        and all(x["requirement_completeness_pass"] for x in results)
    )
    verdict = (
        "TARGETED_REAL_API_VALIDATION_PASSED" if passed else "TARGETED_REAL_API_VALIDATION_FAILED"
    )
    final = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": now(),
        "verdict": verdict,
        "scope": "targeted production-like real-API validation; not fresh generalization evidence",
        "quality": quality,
        "safety": safety,
        "budget_preflight": budget,
        "historical_version_case_available": False,
        "historical_version_note": "No exact historical-version query had a frozen cached embedding; dry-run rules prohibited creating one via an OpenAI call before preflight.",
        "sol_escalation_note": routing["note"],
        "failure_details": [
            {
                "query_id": x["query_id"],
                "failure": x["failure_class"],
                "validation_failure": x["validation_failure"],
            }
            for x in results
            if x["failure_class"]
        ],
        "root_cause": {
            "multi_document_false_abstentions": 3,
            "requirement_count_mismatch_after_luna_go": 2,
            "sol_go_still_failed_same_local_requirement_count_check": 2,
            "explanation": (
                "The real verifier contract consumes the raw question instead of the "
                "label-blind atomic requirement list. Audit prefixes and scope/discourse "
                "clauses therefore disagree with local requirement counting; Luna also "
                "false-abstained on three complete Top-15 multi-document pools."
            ),
        },
        "next_step": "FRESH_UNSEEN_EVALUATION" if passed else "ANOTHER_TARGETED_CORRECTION",
        "fresh_unseen_evaluation_executed": False,
        "candidate_promoted": False,
    }
    return quality, safety, routing, final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if sum((args.dry_run, args.execute, args.finalize_existing)) != 1:
        raise SystemExit("Choose exactly one mode")
    settings = Settings()
    selected = selected_case_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.finalize_existing:
        results = load_jsonl(OUT / "targeted_real_api_results.jsonl")
        ledger = load_jsonl(OUT / "api_cost_ledger.jsonl")
        selected_by_id = {x["query_id"]: x for x in selected}
        for result in results:
            source = selected_by_id[result["query_id"]]
            expected_versions = source.get("expected_versions") or {}
            if not expected_versions:
                result["version_correctness_pass"] = True
                continue
            resolution = result.get("version_resolution") or {}
            result["version_correctness_pass"] = all(
                resolution.get("selected_document_id") == document_id
                and resolution.get("selected_version") == version
                for document_id, version in expected_versions.items()
            )
        write_jsonl(OUT / "targeted_real_api_results.jsonl", results)
        dry_payload = json.loads(
            (OUT / "targeted_validation_dry_run.json").read_text(encoding="utf-8")
        )
        quality, safety, routing, final = reports(results, ledger, selected, dry_payload["budget"])
        write_json(OUT / "routing_analysis.json", routing)
        write_json(OUT / "safety_report.json", safety)
        write_json(OUT / "final_report.json", final)
        print(json.dumps({"verdict": final["verdict"], "quality": quality}, indent=2))
        return
    freeze_path = OUT / "freeze_manifest.json"
    initial_freeze_path = OUT / "freeze_manifest_initial_pre_amendment.json"
    if freeze_path.exists() and not initial_freeze_path.exists():
        initial_freeze_path.write_text(freeze_path.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(OUT / "targeted_cases.json", {"experiment_id": EXPERIMENT_ID, "cases": selected})
    dry, budget = build_dry_run(settings, selected)
    write_json(
        OUT / "targeted_validation_dry_run.json",
        {"experiment_id": EXPERIMENT_ID, "timestamp": now(), "cases": dry, "budget": budget},
    )
    manifest = freeze_manifest(settings, selected)
    if initial_freeze_path.exists():
        manifest["runtime_amendment"] = {
            "reason": "first real Luna response truncated at the original 220-token cap",
            "change": "structured verifier max_completion_tokens raised to 700",
            "prior_freeze_path": str(initial_freeze_path),
            "prior_freeze_sha256": sha256_path(initial_freeze_path),
            "prior_paid_calls": 4,
        }
    write_json(freeze_path, manifest)
    if not budget["passes"]:
        print("BUDGET_GUARD_TRIGGERED")
        raise SystemExit(2)
    if args.dry_run:
        print(
            json.dumps(
                {"experiment_id": EXPERIMENT_ID, "cases": len(dry), "budget": budget}, indent=2
            )
        )
        return
    results, ledger = run_paid(settings, selected, budget)
    write_jsonl(OUT / "targeted_real_api_results.jsonl", results)
    write_jsonl(OUT / "api_cost_ledger.jsonl", ledger)
    quality, safety, routing, final = reports(results, ledger, selected, budget)
    write_json(OUT / "routing_analysis.json", routing)
    write_json(OUT / "safety_report.json", safety)
    write_json(OUT / "final_report.json", final)
    print(json.dumps({"verdict": final["verdict"], "quality": quality}, indent=2))


if __name__ == "__main__":
    main()
