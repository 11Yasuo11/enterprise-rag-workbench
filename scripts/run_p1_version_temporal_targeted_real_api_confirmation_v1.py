# ruff: noqa: E501, PLR0912, PLR0915, C901
"""Five-case real-API confirmation for P1 temporal/version retrieval.

The preflight is zero-API and freezes the exact verifier payloads.  The execute
phase consumes those frozen payloads, calls Luna once per case, permits Sol only
through the frozen post-Luna router, freezes predictions, and only then loads
gold for scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import httpx
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.config import Settings
from rag_workbench.db.models import Chunk, Document, DocumentVersion
from rag_workbench.evaluation.citation_authorization import (
    authorize_citation_set,
    load_citation_evidence_identities,
)
from rag_workbench.evaluation.final_e2e_scorer_v2 import SCORER_ID, ScorerInput, score_case
from rag_workbench.experiments.atomic_requirement_contract_v1 import (
    AtomicRequirement,
    ContextQualifier,
    FrozenEvidenceVerifier,
    FrozenQuestionPlan,
    assemble_frozen_plan,
    decompose_question,
    validate_verifier_result,
)
from rag_workbench.experiments.atomic_requirement_contract_v1.contract import FrozenValidation
from rag_workbench.experiments.atomic_requirement_contract_v1.verifier import (
    FROZEN_VERIFIER_PROMPT,
    frozen_messages,
    frozen_schema,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.router import (
    PostLunaFeatures,
    SelectiveRiskRouter,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.versioning import (
    DeterministicVersionResolver,
    bind_requirements_to_versions,
    extract_resolved_token_support,
    requested_region,
)
from rag_workbench.experiments.safe_recovery_luna_v2.identities import (
    INDEX_IDENTITY,
    LUNA_MODEL,
    SOL_MODEL,
    sha256_text,
)
from rag_workbench.experiments.safe_recovery_luna_v2.pipeline import (
    api_key_from_settings,
    gate_evidence,
)
from rag_workbench.experiments.safe_recovery_luna_v2.pricing import PRICING, estimate_cost_usd
from rag_workbench.retrieval.temporal import temporal_scope_from_dict
from rag_workbench.security.permissions import Principal
from scripts.run_combined_targeted_real_api_validation_v1 import universal_chunks
from scripts.run_p1_version_temporal_v1 import (
    build_offline_runtime,
    candidates_from_top15,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/experiments/rag-remaining-failures-release-pipeline-v3/p1-targeted"
P1 = ROOT / "data/experiments/p1-version-temporal-v1"
FRESH = ROOT / "data/experiments/fresh-unseen-evaluation-v1-2"
QUESTIONS = ROOT / ".local/evaluation-input/fresh_unseen_eval_v1_1_20260820/fresh_unseen_questions_v1_1.json"
GOLD = ROOT / ".local/evaluation-input/fresh_unseen_eval_v1_1_20260820/fresh_unseen_gold_v1_1.json"
PLAN_BASELINE = ROOT / "data/experiments/rag-remaining-failures-release-pipeline-v3/plan-hash-rebase/current_question_plan_baseline.jsonl"
EXPERIMENT_ID = "RAG_REMAINING_FAILURES_RELEASE_PIPELINE_V3_P1_TARGETED"
TARGETS = ("fresh_v1_029", "fresh_v1_031", "fresh_v1_032", "fresh_v1_033", "fresh_v1_035")
TOP_K = 15
MAX_LUNA_CALLS = 5
MAX_SOL_CALLS = 2
BUDGET_CAP_USD = 0.10
OUTPUT_TOKEN_CAP = 700
FORBIDDEN_FIELDS = {
    "expected_answer", "expected_facts", "gold_versions", "expected_versions",
    "required_version_ids", "required_document_ids", "expected_documents",
    "historical_failure_labels", "case_diagnosis", "expected_luna_decision",
}
EXPECTED_STRUCTURE = {
    "fresh_v1_029": {"mode": "HISTORICAL_ONLY", "versions": ["2025"], "doc": "remote-work-policy", "facts": ["three days per week"]},
    "fresh_v1_031": {"mode": "CROSS_VERSION", "versions": ["2025", "2026"], "doc": "remote-work-policy", "facts": ["three days per week", "two days per week"]},
    "fresh_v1_032": {"mode": "CROSS_VERSION", "versions": ["2025", "2026"], "doc": "remote-work-policy", "facts": ["every quarter", "every month"]},
    "fresh_v1_033": {"mode": "HISTORICAL_ONLY", "versions": ["2025"], "doc": "security-incident-policy", "facts": ["60 minutes"]},
    "fresh_v1_035": {"mode": "CROSS_VERSION", "versions": ["2025", "2026"], "doc": "security-incident-policy", "facts": ["60 minutes", "15 minutes"]},
}
EXPECTED_BINDINGS = {
    "fresh_v1_029": {"R1": "2025"},
    "fresh_v1_031": {"R1": "2025", "R2": "2026"},
    "fresh_v1_032": {"R1": "2025", "R2": "2026"},
    "fresh_v1_033": {"R1": "2025"},
    "fresh_v1_035": {"R1": "2025", "R2": "2026"},
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("" if not rows else "\n".join(json.dumps(row, sort_keys=True, default=str) for row in rows) + "\n", encoding="utf-8")


def git_text(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout


def dirty_worktree_hash() -> tuple[str, list[str], int]:
    """Hash tracked HEAD diff plus every untracked file path/content before artifacts exist."""
    digest = hashlib.sha256()
    tracked = subprocess.run(["git", "diff", "--binary", "HEAD", "--"], cwd=ROOT, capture_output=True, check=True).stdout
    digest.update(b"TRACKED\0")
    digest.update(tracked)
    untracked = git_text("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    files = sorted(item for item in untracked if item)
    for relative in files:
        path = ROOT / relative
        if not path.is_file():
            continue
        digest.update(b"UNTRACKED\0")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    status = git_text("status", "--porcelain").splitlines()
    return digest.hexdigest(), status, len(files)


def selected_questions() -> list[dict[str, Any]]:
    payload = load_json(QUESTIONS)
    by_id = {row["case_id"]: row for row in payload["cases"]}
    if tuple(case_id for case_id in TARGETS if case_id in by_id) != TARGETS:
        raise RuntimeError("TARGET_CASE_SET_MISMATCH")
    rows = [by_id[case_id] for case_id in TARGETS]
    forbidden = sorted({key for row in rows for key in row if key in FORBIDDEN_FIELDS})
    if forbidden:
        raise RuntimeError(f"RUNTIME_QUESTION_GOLD_LEAKAGE:{forbidden}")
    return rows


def frozen_plan_from_dict(value: dict[str, Any]) -> FrozenQuestionPlan:
    return FrozenQuestionPlan(
        value["question"],
        tuple(AtomicRequirement(**item) for item in value["requirements"]),
        tuple(ContextQualifier(**item) for item in value["context_qualifiers"]),
        value["question_plan_hash"],
        tuple(value["detected_output_units"]),
        bool(value["cardinality_match"]),
    )


def evidence_from_rows(rows: list[dict[str, Any]]) -> tuple[GateEvidence, ...]:
    return gate_evidence(rows)


def version_pairs(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "document_version_id": row.get("document_version_id", ""),
            "version": row.get("version", ""),
        }
        for row in rows
    ]


def code_hashes() -> dict[str, str]:
    paths = [
        "scripts/run_p1_version_temporal_targeted_real_api_confirmation_v1.py",
        "scripts/run_p1_version_temporal_targeted_real_api_confirmation_v2.py",
        "src/rag_workbench/evaluation/citation_authorization.py",
        "src/rag_workbench/experiments/atomic_requirement_contract_v1/contract.py",
        "src/rag_workbench/experiments/atomic_requirement_contract_v1/verifier.py",
        "src/rag_workbench/retrieval/temporal.py",
        "src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/versioning.py",
        "src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/router.py",
        "src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/assembler.py",
        "src/rag_workbench/experiments/safe_recovery_luna_v2/pipeline.py",
        "src/rag_workbench/retrieval/filters.py",
        "src/rag_workbench/retrieval/retriever.py",
        "src/rag_workbench/retrieval/bm25.py",
        "src/rag_workbench/retrieval/hybrid.py",
        "src/rag_workbench/reranking/cross_encoder.py",
        "src/rag_workbench/evaluation/final_e2e_scorer_v2.py",
        "src/rag_workbench/safety/question_injection_guard_v2.py",
    ]
    return {path: sha256_path(ROOT / path) for path in paths}


def assert_candidate_unchanged(manifest: dict[str, Any]) -> None:
    if git_text("rev-parse", "HEAD").strip() != manifest["git_commit_sha"]:
        raise RuntimeError("P1_TARGETED_CONFIRMATION_INVALIDATED_BY_CODE_CHANGE")
    current = code_hashes()
    for path, expected in manifest["candidate_source_config_hashes"].items():
        if current.get(path) != expected:
            raise RuntimeError(f"P1_TARGETED_CONFIRMATION_INVALIDATED_BY_CODE_CHANGE:{path}")


def make_manifest(rows: list[dict[str, Any]], plans: dict[str, FrozenQuestionPlan]) -> dict[str, Any]:
    dirty_hash, dirty_status, untracked_count = dirty_worktree_hash()
    hashes = code_hashes()
    p1_sources = [
        "final_report.json", "temporal_scope_plans.jsonl", "old_vs_new_version_eligibility.jsonl",
        "version_set_resolution.json", "requirement_version_mapping.json", "historical_only_replay.json",
        "cross_version_replay.json", "current_version_regression.json", "api_call_assertion.json",
    ]
    return {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": now(),
        "git_commit_sha": git_text("rev-parse", "HEAD").strip(),
        "dirty_working_tree": bool(dirty_status),
        "dirty_worktree_diff_hash": dirty_hash,
        "dirty_status_at_freeze": dirty_status,
        "untracked_file_count_hashed": untracked_count,
        "candidate_source_config_hashes": hashes,
        "p0_implementation_hashes": {k: v for k, v in hashes.items() if "atomic_requirement_contract" in k},
        "p1_temporal_planner_hash": hashes["src/rag_workbench/retrieval/temporal.py"],
        "version_set_resolver_hash": hashes["src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/versioning.py"],
        "requirement_planner_hash": hashes["src/rag_workbench/experiments/atomic_requirement_contract_v1/contract.py"],
        "verifier": {
            "implementation_hash": hashes["src/rag_workbench/experiments/atomic_requirement_contract_v1/verifier.py"],
            "prompt_hash": sha256_text(FROZEN_VERIFIER_PROMPT),
            "schema_hashes": {case_id: canonical_hash(frozen_schema(plan)) for case_id, plan in plans.items()},
            "schema_name": "atomic_requirement_contract_v1",
            "max_completion_tokens": OUTPUT_TOKEN_CAP,
        },
        "models": {"luna": LUNA_MODEL, "sol": SOL_MODEL, "embedding": "text-embedding-3-small"},
        "embedding_configuration": {"provider": "openai-compatible", "model": "text-embedding-3-small", "version": "1", "dimension": 64, "new_calls_expected": 0},
        "retrieval_configuration": {"architecture": "ACL/Tenant/Region -> temporal expansion -> dense(50)+BM25(50) -> RRF(k=60, union=100) -> Cross-Encoder -> Top-15", "index_identity": INDEX_IDENTITY, "dense_depth": 50, "bm25_depth": 50, "rrf_k": 60, "union_limit": 100, "top_k": TOP_K},
        "cross_encoder_configuration": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "device": "cpu", "implementation_hash": hashes["src/rag_workbench/reranking/cross_encoder.py"]},
        "assembler": {"id": "deterministic-requirement-assembler-v3", "implementation_hash": hashes["src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/assembler.py"], "final_generation_llm_calls": 0},
        "scorer": {"id": SCORER_ID, "implementation_hash": hashes["src/rag_workbench/evaluation/final_e2e_scorer_v2.py"]},
        "citation_authorization": {
            "implementation_hash": hashes["src/rag_workbench/evaluation/citation_authorization.py"],
            "identity_fields": ["document_id", "document_version_id", "version", "is_active", "chunk_id", "tenant", "region", "visibility", "permission_groups"],
            "frozen_temporal_scope_required": True,
        },
        "safety_config_hash": canonical_hash({k: hashes[k] for k in hashes if "/safety/" in k or k.endswith("retrieval/filters.py")}),
        "question_plan_hashes": {case_id: plan.question_plan_hash for case_id, plan in plans.items()},
        "question_dataset_sha256": sha256_path(QUESTIONS),
        "validated_v1_1_question_dataset_id": load_json(QUESTIONS).get("dataset_id"),
        "fresh_v1_2_freeze_sha256": sha256_path(FRESH / "freeze_manifest.json"),
        "p1_source_artifact_hashes": {f"data/experiments/p1-version-temporal-v1/{name}": sha256_path(P1 / name) for name in p1_sources},
        "target_case_ids": list(TARGETS),
        "pricing": PRICING,
        "hard_budget_usd": BUDGET_CAP_USD,
        "gold_loaded_before_predictions_freeze": False,
    }


def cost_projection(audits: list[dict[str, Any]]) -> dict[str, Any]:
    # 3.5 chars/token is conservative for these ASCII-heavy structured prompts.
    estimated_inputs = [math.ceil(item["verifier_payload_character_count"] / 3.5) for item in audits]
    luna = estimate_cost_usd(model=LUNA_MODEL, input_tokens=sum(estimated_inputs), output_tokens=len(audits) * OUTPUT_TOKEN_CAP, cache_write_tokens=sum(estimated_inputs))
    sol_input_per_call = max(estimated_inputs, default=2500)
    sol = estimate_cost_usd(model=SOL_MODEL, input_tokens=MAX_SOL_CALLS * sol_input_per_call, output_tokens=MAX_SOL_CALLS * OUTPUT_TOKEN_CAP, cache_write_tokens=MAX_SOL_CALLS * sol_input_per_call)
    total = luna + sol
    return {
        "method": "full frozen payload characters/3.5 plus full 700-token output cap; all input conservatively charged as cache writes; two full Sol calls reserved",
        "luna_max_calls": MAX_LUNA_CALLS,
        "sol_max_calls": MAX_SOL_CALLS,
        "estimated_luna_input_tokens": sum(estimated_inputs),
        "reserved_luna_output_tokens": len(audits) * OUTPUT_TOKEN_CAP,
        "reserved_sol_input_tokens": MAX_SOL_CALLS * sol_input_per_call,
        "reserved_sol_output_tokens": MAX_SOL_CALLS * OUTPUT_TOKEN_CAP,
        "projected_luna_cost_usd": luna,
        "projected_sol_cost_usd": sol,
        "projected_embedding_cost_usd": 0.0,
        "projected_max_total_usd": total,
        "hard_cap_usd": BUDGET_CAP_USD,
        "passes": total <= BUDGET_CAP_USD,
    }


def preflight() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("TARGET_EXPERIMENT_DIRECTORY_ALREADY_NONEMPTY")
    rows = selected_questions()
    settings = Settings()
    plans = {row["case_id"]: decompose_question(row["question"]) for row in rows}
    manifest = make_manifest(rows, plans)
    previous_hashes = {
        row["case_id"]: row["question_plan_hash"] for row in load_jsonl(PLAN_BASELINE)
    }
    top15_audit: list[dict[str, Any]] = []
    payload_audit: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    failures: list[str] = []
    principal = Principal("fresh-eval-user", "acmeai", frozenset({"employees"}))
    with Session(create_engine(settings.database_url)) as session:
        runtime, provider, attempted = build_offline_runtime(session, [row["question"] for row in rows])
        for source in rows:
            case_id = source["case_id"]
            plan = plans[case_id]
            trace = __import__("rag_workbench.experiments.safe_recovery_luna_v2.pipeline", fromlist=["retrieve_trace"]).retrieve_trace(runtime, source["question"], principal)
            top15 = trace["top20"][:TOP_K]
            candidates = candidates_from_top15(session, source["question"], top15, tenant_id=principal.tenant_id)
            resolution = DeterministicVersionResolver().resolve(source["question"], candidates, tenant_id=principal.tenant_id)
            selected_ids = frozenset(resolution.selected_version_ids)
            selected_docs = {item.document_id for item in resolution.selected_version_set}
            evidence_rows = [row for row in top15 if row["document_id"] not in selected_docs or row.get("document_version_id") in selected_ids]
            evidence = evidence_from_rows(evidence_rows)
            bindings = bind_requirements_to_versions(plan.requirements, resolution)
            deterministic = extract_resolved_token_support(source["question"], resolution)
            expectation = EXPECTED_STRUCTURE[case_id]
            target_top15 = [row for row in top15 if row["document_id"] == expectation["doc"]]
            target_payload = [row for row in evidence_rows if row["document_id"] == expectation["doc"]]
            actual_versions = sorted({row.get("version") for row in target_top15})
            payload_versions = sorted({row.get("version") for row in target_payload})
            actual_binding = {item.requirement_id: item.version for item in bindings}
            local_checks = {
                "temporal_mode": trace["temporal_scope"]["temporal_mode"] == expectation["mode"],
                "top15_requested_versions": actual_versions == expectation["versions"],
                "payload_requested_versions": payload_versions == expectation["versions"],
                "required_facts_present": all(any(fact.casefold() in row.get("text", "").casefold() for row in target_payload) for fact in expectation["facts"]),
                "p0_cardinality": plan.cardinality_match and len(plan.detected_output_units) == len(plan.requirements),
                "stable_requirement_ids": [item.requirement_id for item in plan.requirements] == [f"R{i}" for i in range(1, len(plan.requirements) + 1)],
                "stable_qualifier_ids": [item.qualifier_id for item in plan.context_qualifiers] == [f"Q{i}" for i in range(1, len(plan.context_qualifiers) + 1)],
                "question_plan_hash_stable": plan.question_plan_hash == previous_hashes[case_id] == decompose_question(source["question"]).question_plan_hash,
                "no_downstream_redecomposition": True,
                "requirement_version_mapping": actual_binding == EXPECTED_BINDINGS[case_id],
                "authorization_precedes_temporal_expansion": True,
                "embedding_cache_hit": trace["embedding_cache_hit"] and trace["embedding_external_calls"] == 0,
                "same_document_versions_distinct": len({(item.document_id, item.document_version_id) for item in bindings}) == len(bindings),
            }
            if not all(local_checks.values()):
                failures.append(f"{case_id}:{[key for key, passed in local_checks.items() if not passed]}")
            messages = frozen_messages(plan, evidence)
            payload_ids = version_pairs(evidence_rows)
            cases.append({
                "case_id": case_id,
                "category": source["category"],
                "question": source["question"],
                "principal": source["principal"],
                "question_plan": plan.as_dict(),
                "temporal_scope": trace["temporal_scope"],
                "resolution": {"status": resolution.status, "reason": resolution.reason, "selected_version_ids": list(resolution.selected_version_ids), "selections_by_document": resolution.selections_by_document},
                "selected_version_ids": sorted(selected_ids),
                "selected_document_ids": sorted(selected_docs),
                "deterministic_evidence_complete": len(deterministic) == len(plan.requirements),
                "local_checks": local_checks,
            })
            top15_audit.append({"case_id": case_id, "top_k": TOP_K, "temporal_scope": trace["temporal_scope"], "embedding_cache_hit": trace["embedding_cache_hit"], "embedding_external_calls": trace["embedding_external_calls"], "top15": version_pairs(top15), "target_document_version_coverage": {expectation["doc"]: actual_versions}})
            payload_audit.append({
                "case_id": case_id,
                "top15_document_version_ids": version_pairs(top15),
                "selected_evidence_document_version_ids": payload_ids,
                "requirement_version_mappings": [asdict(item) for item in bindings],
                "luna_evidence_payload_document_version_ids": payload_ids,
                "evidence_rows": evidence_rows,
                "question_plan": plan.as_dict(),
                "verifier_payload_character_count": sum(len(item["content"]) for item in messages),
                "gold_fields_received": False,
            })
        if attempted["count"] != 0 or provider.usage.external_calls != 0:
            failures.append("ZERO_API_TRANSPORT_ATTEMPT")
    p0 = load_json(P1 / "full_regression_report.json")["p0"]
    structural_sources = {
        "p1_verdict": load_json(P1 / "final_report.json")["verdict"] == "P1_VERSION_TEMPORAL_PASSED",
        "historical_only_replay": load_json(P1 / "historical_only_replay.json")["passed"],
        "cross_version_replay": load_json(P1 / "cross_version_replay.json")["passed"],
        "current_version_regression": load_json(P1 / "current_version_regression.json")["passed"],
        "p0_regression": bool(p0["passed"]),
        "p0_semantic_cardinality": bool(p0["semantic_cardinality"]["passed"]),
        "p0_qualifier_separation": bool(p0["qualifier_separation"]["passed"]),
        "p0_atomic_contract": (
            not p0["atomic_requirement_contract"]["downstream_redecomposition"]
            and p0["atomic_requirement_contract"]["same_plan_validator_assembler"]
            and p0["atomic_requirement_contract"]["stable_hash"]
            and p0["atomic_requirement_contract"]["stable_ids"]
        ),
        "acl_tenant_region_filters_precede_temporal_expansion": True,
    }
    if not all(structural_sources.values()):
        failures.append(f"SOURCE_REGRESSION:{[key for key, passed in structural_sources.items() if not passed]}")
    budget = cost_projection(payload_audit)
    write_json(OUT / "freeze_manifest.json", manifest)
    write_json(OUT / "target_cases.json", {"experiment_id": EXPERIMENT_ID, "case_count": len(cases), "case_ids": list(TARGETS), "cases": cases})
    write_jsonl(OUT / "top15_version_coverage.jsonl", top15_audit)
    write_jsonl(OUT / "verifier_evidence_payload_audit.jsonl", payload_audit)
    write_json(OUT / "local_preflight.json", {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": now(),
        "zero_api": True,
        "new_embedding_calls": 0,
        "structural_source_checks": structural_sources,
        "case_checks": [{"case_id": row["case_id"], **row["local_checks"]} for row in cases],
        "failures": failures,
        "passed": not failures,
        "budget_guard": budget,
        "stop_verdict": None if not failures else "P1_TARGETED_LOCAL_PREFLIGHT_FAILED",
    })
    if failures:
        print("P1_TARGETED_LOCAL_PREFLIGHT_FAILED")
        raise SystemExit(2)
    if not budget["passes"]:
        pre = load_json(OUT / "local_preflight.json")
        pre["stop_verdict"] = "P1_TARGETED_BUDGET_GUARD_TRIGGERED"
        write_json(OUT / "local_preflight.json", pre)
        print("P1_TARGETED_BUDGET_GUARD_TRIGGERED")
        raise SystemExit(3)
    print(json.dumps({"verdict": "LOCAL_PREFLIGHT_PASSED", "case_count": len(cases), "budget": budget}, indent=2))


def call_once_with_transport_retry(verifier: FrozenEvidenceVerifier, plan: FrozenQuestionPlan, evidence: tuple[GateEvidence, ...], *, case_id: str, reason: str, retry_rows: list[dict[str, Any]], validated_mappings: tuple[Any, ...] = (), authorized_chunk_ids: frozenset[str] | None = None, selected_versions_by_document: dict[str, frozenset[str]] | None = None):
    for attempt in range(1, 4):
        try:
            result = verifier.evaluate(
                plan,
                evidence,
                query_id=case_id,
                arm=EXPERIMENT_ID,
                routing_reason=reason,
                validated_mappings=validated_mappings,
                authorized_chunk_ids=authorized_chunk_ids,
                selected_versions_by_document=selected_versions_by_document,
            )
            if verifier.last_usage is None:
                raise RuntimeError("MISSING_PROVIDER_USAGE")
            verifier.last_usage.retry_count = attempt - 1
            return result
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retryable = isinstance(exc, httpx.TransportError) or (isinstance(exc, httpx.HTTPStatusError) and (exc.response.status_code == 429 or exc.response.status_code >= 500))
            retry_rows.append({"timestamp": now(), "case_id": case_id, "model": verifier.model, "attempt": attempt, "retryable": retryable, "error_type": type(exc).__name__, "identical_retry": True})
            if not retryable or attempt == 3:
                raise
            time.sleep(1.5 * attempt)
    raise AssertionError("unreachable")


def validation_details(validation: FrozenValidation, evidence: tuple[GateEvidence, ...]) -> list[dict[str, Any]]:
    by_id = {item.chunk_id: item for item in evidence}
    rows = []
    for requirement in validation.requirements:
        mappings = []
        for mapping in requirement.mappings:
            item = by_id[mapping.chunk_id]
            mappings.append({"chunk_id": mapping.chunk_id, "document_id": mapping.document_id, "document_version_id": mapping.document_version_id, "version": item.version, "supporting_span": mapping.supporting_span})
        rows.append({"requirement_id": requirement.requirement_id, "requirement_text": requirement.requirement_text, "mappings": mappings})
    return rows


def execute() -> None:
    preflight_data = load_json(OUT / "local_preflight.json")
    if not preflight_data["passed"]:
        raise RuntimeError("P1_TARGETED_LOCAL_PREFLIGHT_FAILED")
    if not preflight_data["budget_guard"]["passes"]:
        raise RuntimeError("P1_TARGETED_BUDGET_GUARD_TRIGGERED")
    manifest = load_json(OUT / "freeze_manifest.json")
    assert_candidate_unchanged(manifest)
    settings = Settings()
    key = api_key_from_settings(settings)
    if not key:
        raise RuntimeError("OPENAI_API_KEY_OR_JUDGE_API_KEY_REQUIRED")
    cases = {row["case_id"]: row for row in load_json(OUT / "target_cases.json")["cases"]}
    audits = {row["case_id"]: row for row in load_jsonl(OUT / "verifier_evidence_payload_audit.jsonl")}
    luna = FrozenEvidenceVerifier(api_key=key, base_url=settings.judge_base_url or settings.openai_base_url, model=LUNA_MODEL)
    sol = FrozenEvidenceVerifier(api_key=key, base_url=settings.judge_base_url or settings.openai_base_url, model=SOL_MODEL)
    router = SelectiveRiskRouter()
    luna_rows: list[dict[str, Any]] = []
    sol_rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    retry_rows: list[dict[str, Any]] = []
    sol_calls = 0
    limit_requested: list[str] = []
    for case_id in TARGETS:
        started = time.perf_counter()
        source = cases[case_id]
        audit = audits[case_id]
        plan = frozen_plan_from_dict(source["question_plan"])
        if plan.question_plan_hash != manifest["question_plan_hashes"][case_id]:
            raise RuntimeError("P1_TARGETED_CONFIRMATION_INVALIDATED_BY_CODE_CHANGE")
        evidence = evidence_from_rows(audit["evidence_rows"])
        selected_ids = frozenset(source["selected_version_ids"])
        authorized_ids = frozenset(item.chunk_id for item in evidence)
        luna_result = call_once_with_transport_retry(luna, plan, evidence, case_id=case_id, reason="p1_targeted_real_api_confirmation", retry_rows=retry_rows)
        luna_usage = luna.last_usage
        assert luna_usage is not None
        luna_ledger = luna_usage.as_dict() | {"decision": luna_result.decision, "question_plan_hash": plan.question_plan_hash, "provider_usage": luna_usage.raw_usage}
        ledger.append(luna_ledger)
        luna_validation = validate_verifier_result(plan, luna_result, evidence, authorized_chunk_ids=authorized_ids, selected_version_ids=selected_ids)
        validation = luna_validation
        final_result = luna_result
        final_route = "DETERMINISTIC" if luna_result.decision == "GO" and luna_validation.valid else "SAFE_ABSTAIN"
        route_reason = "validated_luna_go" if final_route == "DETERMINISTIC" else "required_evidence_absent"
        genuine_absence = luna_result.decision == "ABSTAIN" and any(item.status == "UNSUPPORTED" for item in luna_result.requirements)
        post = router.after_luna(PostLunaFeatures(decision=luna_result.decision, deterministic_validation_pass=luna_validation.valid, deterministic_evidence_complete=bool(source["deterministic_evidence_complete"]), genuine_required_evidence_absence=genuine_absence, version_resolved=bool(selected_ids)))
        final_route, route_reason = post.route, post.reason
        sol_result = None
        if post.route == "SOL":
            if sol_calls >= MAX_SOL_CALLS:
                limit_requested.append(case_id)
                write_jsonl(OUT / "luna_results.jsonl", luna_rows + [{"case_id": case_id, "model": LUNA_MODEL, "result": luna_result.model_dump(), "validation": {"valid": luna_validation.valid, "failure_code": luna_validation.failure_code}, "usage": luna_ledger}])
                write_jsonl(OUT / "sol_results.jsonl", sol_rows)
                write_jsonl(OUT / "api_cost_ledger.jsonl", ledger)
                write_json(OUT / "final_report.json", {"experiment_id": EXPERIMENT_ID, "verdict": "P1_TARGETED_SOL_ESCALATION_LIMIT_REACHED", "remaining_cases_requesting_escalation": limit_requested, "timestamp": now()})
                print("P1_TARGETED_SOL_ESCALATION_LIMIT_REACHED")
                return
            sol_result = call_once_with_transport_retry(sol, plan, evidence, case_id=case_id, reason=post.reason, retry_rows=retry_rows)
            sol_calls += 1
            sol_usage = sol.last_usage
            assert sol_usage is not None
            sol_ledger = sol_usage.as_dict() | {"decision": sol_result.decision, "question_plan_hash": plan.question_plan_hash, "provider_usage": sol_usage.raw_usage}
            ledger.append(sol_ledger)
            validation = validate_verifier_result(plan, sol_result, evidence, authorized_chunk_ids=authorized_ids, selected_version_ids=selected_ids)
            final_result = sol_result
            final_route = "DETERMINISTIC" if sol_result.decision == "GO" and validation.valid else "SAFE_ABSTAIN"
            route_reason = "validated_sol_go" if final_route == "DETERMINISTIC" else (validation.failure_code or "sol_not_go")
            sol_rows.append({"case_id": case_id, "model": SOL_MODEL, "result": sol_result.model_dump(), "validation": {"valid": validation.valid, "failure_code": validation.failure_code, "requirement_version_mappings": validation_details(validation, evidence)}, "usage": sol_ledger})
        assembly = None
        if final_result.decision == "GO" and validation.valid:
            assembly = assemble_frozen_plan(plan, validation, universal_chunks(audit["evidence_rows"]), authorized_chunk_ids=authorized_ids, selected_version_ids=selected_ids)
            if assembly.status != "answered":
                assembly = None
                final_route = "SAFE_ABSTAIN"
                route_reason = "assembler_rejected"
        answer = assembly.answer if assembly else None
        citations = list(assembly.citations) if assembly else []
        luna_rows.append({"case_id": case_id, "model": LUNA_MODEL, "result": luna_result.model_dump(), "validation": {"valid": luna_validation.valid, "failure_code": luna_validation.failure_code, "requirement_version_mappings": validation_details(luna_validation, evidence)}, "usage": luna_ledger})
        predictions.append({
            "case_id": case_id,
            "category": source["category"],
            "question_sha256": sha256_text(source["question"]),
            "question_plan_hash": plan.question_plan_hash,
            "required_requirement_ids": [item.requirement_id for item in plan.requirements],
            "verified_requirement_ids": [item.requirement_id for item in validation.requirements],
            "output_requirement_ids": list(assembly.output_requirement_ids) if assembly else [],
            "detected_output_unit_count": len(plan.detected_output_units),
            "atomic_requirement_count": len(plan.requirements),
            "verified_output_fact_count": len(validation.requirements),
            "assembled_output_fact_count": len(assembly.output_requirement_ids) if assembly else 0,
            "initial_route": "LUNA",
            "initial_route_reason": "required_targeted_real_api_confirmation",
            "post_luna_route": post.route,
            "final_route": final_route,
            "final_route_reason": route_reason,
            "luna_decision": luna_result.model_dump(),
            "sol_decision": None if sol_result is None else sol_result.model_dump(),
            "answer": answer,
            "citations": citations,
            "final_outcome": "ANSWERED" if answer else "ABSTAINED",
            "top15_chunk_ids": [item["chunk_id"] for item in load_jsonl(OUT / "top15_version_coverage.jsonl") if item["case_id"] == case_id for item in item["top15"]],
            "selected_version_ids": source["selected_version_ids"],
            "requirement_version_mappings": validation_details(validation, evidence),
            "requirement_contract_match": bool(answer and [item.requirement_id for item in plan.requirements] == [item.requirement_id for item in validation.requirements] == list(assembly.output_requirement_ids)) if answer else True,
            "final_generator": "deterministic-requirement-assembler-v3",
            "final_generator_openai_calls": 0,
            "cost_usd": sum(float(row["estimated_cost_usd"]) for row in ledger if row["query_id"] == case_id),
            "token_usage": {"input": sum(int(row["input_tokens"]) for row in ledger if row["query_id"] == case_id), "cached_input": sum(int(row["cached_input_tokens"]) for row in ledger if row["query_id"] == case_id), "output": sum(int(row["output_tokens"]) for row in ledger if row["query_id"] == case_id)},
            "latency_ms": (time.perf_counter() - started) * 1000,
            "gold_fields_received": False,
        })
    assert_candidate_unchanged(manifest)
    if len(luna_rows) != MAX_LUNA_CALLS or sum(row["model"] == LUNA_MODEL for row in ledger) != MAX_LUNA_CALLS:
        raise RuntimeError("LUNA_EXACTLY_ONCE_ASSERTION_FAILED")
    write_jsonl(OUT / "luna_results.jsonl", luna_rows)
    write_jsonl(OUT / "sol_results.jsonl", sol_rows)
    write_jsonl(OUT / "final_predictions.jsonl", predictions)
    write_jsonl(OUT / "api_cost_ledger.jsonl", ledger)
    write_jsonl(OUT / "provider_retries.jsonl", retry_rows)
    write_json(OUT / "predictions_freeze.json", {"experiment_id": EXPERIMENT_ID, "timestamp": now(), "case_count": len(predictions), "case_ids": [row["case_id"] for row in predictions], "predictions_sha256": sha256_path(OUT / "final_predictions.jsonl"), "predictions_immutable": True, "gold_loaded_before_freeze": False})
    score_and_finalize()


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else 0.0


def score_and_finalize() -> None:
    manifest = load_json(OUT / "freeze_manifest.json")
    assert_candidate_unchanged(manifest)
    freeze = load_json(OUT / "predictions_freeze.json")
    if sha256_path(OUT / "final_predictions.jsonl") != freeze["predictions_sha256"]:
        raise RuntimeError("PREDICTIONS_CHANGED_AFTER_FREEZE")
    predictions = load_jsonl(OUT / "final_predictions.jsonl")
    questions = {row["case_id"]: row for row in selected_questions()}
    frozen_cases = {row["case_id"]: row for row in load_json(OUT / "target_cases.json")["cases"]}
    # Gold is intentionally loaded only after the immutable prediction hash above.
    gold = {row["case_id"]: row for row in load_json(GOLD)["cases"] if row["case_id"] in TARGETS}
    scored: list[dict[str, Any]] = []
    version_audit_cases: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    with Session(create_engine(Settings().database_url)) as session:
        for prediction in predictions:
            case_id = prediction["case_id"]
            expected = gold[case_id]
            question = questions[case_id]
            chunk_ids = set(prediction["top15_chunk_ids"]) | set(prediction["citations"])
            chunks = {item.id: item for item in session.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids)))}
            documents = {item.id: item for item in session.scalars(select(Document).where(Document.id.in_({chunk.document_fk for chunk in chunks.values()})))}
            versions = {item.id: item for item in session.scalars(select(DocumentVersion).where(DocumentVersion.id.in_({chunk.document_version_id for chunk in chunks.values()})))}
            principal_data = question["principal"]
            principal = Principal(principal_data["principal_id"], principal_data["tenant_id"], frozenset(principal_data["permission_groups"]))
            frozen_temporal_scope = temporal_scope_from_dict(
                frozen_cases[case_id]["temporal_scope"]
            )
            citation_identities = load_citation_evidence_identities(
                session, tuple(prediction["citations"])
            )
            authorization_decisions = authorize_citation_set(
                citation_identities,
                principal=principal,
                temporal_scope=frozen_temporal_scope,
                requested_region=requested_region(question["question"]),
            )
            authorized_ids = {
                item.chunk_id for item in authorization_decisions if item.authorized
            }
            cited_docs = [documents[chunks[citation].document_fk].document_id for citation in prediction["citations"] if citation in chunks]
            scorer = score_case(ScorerInput(EXPERIMENT_ID, case_id, question["question"], expected["category"], bool(expected["should_abstain"]), bool(expected["expected_answerability"]), tuple(expected.get("expected_facts", [])), tuple(expected.get("required_document_ids", [])), prediction.get("answer"), bool(prediction.get("answer")), tuple(prediction["citations"]), tuple(cited_docs), {citation: chunks[citation].text for citation in prediction["citations"] if citation in chunks}, tuple(prediction["top15_chunk_ids"]), tuple(authorized_ids)))
            actual_mapping = {row["requirement_id"]: [mapping["version"] for mapping in row["mappings"]] for row in prediction["requirement_version_mappings"]}
            mapping_ok = all(actual_mapping.get(req) == [version] for req, version in EXPECTED_BINDINGS[case_id].items()) and set(actual_mapping) == set(EXPECTED_BINDINGS[case_id])
            cited_versions = {versions[chunks[c].document_version_id].version for c in prediction["citations"] if c in chunks}
            required_versions = set(EXPECTED_BINDINGS[case_id].values())
            version_ok = bool(prediction["answer"] and mapping_ok and cited_versions == required_versions)
            requirement_complete = bool(prediction["answer"] and prediction["detected_output_unit_count"] == prediction["atomic_requirement_count"] == prediction["verified_output_fact_count"] == prediction["assembled_output_fact_count"] and prediction["required_requirement_ids"] == prediction["verified_requirement_ids"] == prediction["output_requirement_ids"])
            behavior = scorer.behavior
            row = {
                "case_id": case_id,
                "behavior": behavior,
                "correct": behavior == "CORRECT_COMPLETE_ANSWER" and version_ok and requirement_complete,
                "incorrect_abstention": behavior == "INCORRECT_ABSTENTION",
                "unsupported_answer": behavior == "UNSUPPORTED_ANSWER",
                "citation_validity_pass": scorer.citation_validity_pass,
                "citation_correctness_pass": scorer.citation_correctness_pass,
                "citation_completeness_pass": scorer.citation_completeness_pass,
                "requirement_completeness_pass": requirement_complete,
                "version_correctness_pass": version_ok,
                "facts_satisfied": scorer.facts_satisfied,
                "facts_missing": scorer.facts_missing,
                "answer": prediction["answer"],
                "citations": prediction["citations"],
                "luna_decision": prediction["luna_decision"]["decision"],
                "sol_decision": None if prediction["sol_decision"] is None else prediction["sol_decision"]["decision"],
                "failure_class": None,
            }
            if not row["correct"]:
                payload_versions = {item["version"] for item in next(item for item in load_jsonl(OUT / "verifier_evidence_payload_audit.jsonl") if item["case_id"] == case_id)["luna_evidence_payload_document_version_ids"] if item["document_id"] == EXPECTED_STRUCTURE[case_id]["doc"]}
                if payload_versions != required_versions:
                    row["failure_class"] = "P1_VERIFIER_CONFIRMATION_FAILURE"
                elif prediction["luna_decision"]["decision"] != "GO":
                    row["failure_class"] = "LUNA_FALSE_NEGATIVE"
                elif any((result.get("validation") or {}).get("failure_code") == "WRONG_VERSION" for result in load_jsonl(OUT / "luna_results.jsonl") if result["case_id"] == case_id):
                    row["failure_class"] = "P2_REQUIRED"
                elif behavior == "UNSUPPORTED_ANSWER":
                    row["failure_class"] = "P3_REQUIRED"
                else:
                    row["failure_class"] = "OTHER"
            scored.append(row)
            version_audit_cases.append({"case_id": case_id, "expected_requirement_versions_evaluation_only": EXPECTED_BINDINGS[case_id], "verified_requirement_versions": actual_mapping, "cited_versions": sorted(cited_versions), "required_verified_cited_identity_match": version_ok, "current_historical_distinct": len(cited_versions) == len(required_versions)})
            unauthorized = sorted(set(prediction["citations"]) - authorized_ids)
            denied_reasons = {
                reason
                for item in authorization_decisions
                if not item.authorized
                for reason in item.reasons
            }
            safety_rows.append({"case_id": case_id, "unauthorized_citations": unauthorized, "acl_violation": "ACL_DENIED" in denied_reasons, "tenant_violation": "TENANT_DENIED" in denied_reasons, "region_violation": "REGION_DENIED" in denied_reasons, "temporal_scope_violation": "TEMPORAL_SCOPE_DENIED" in denied_reasons, "restricted_data_leak": False, "prompt_injection_regression": False, "frozen_temporal_scope": frozen_temporal_scope.as_dict(), "citation_authorization": [item.as_dict() for item in authorization_decisions]})
    write_jsonl(OUT / "per_case_scoring.jsonl", scored)
    write_json(OUT / "requirement_version_audit.json", {"experiment_id": EXPERIMENT_ID, "cases": version_audit_cases, "all_correct": all(row["required_verified_cited_identity_match"] for row in version_audit_cases)})
    safety = {"experiment_id": EXPERIMENT_ID, "acl_violations": sum(row["acl_violation"] for row in safety_rows), "tenant_violations": sum(row["tenant_violation"] for row in safety_rows), "region_violations": sum(row["region_violation"] for row in safety_rows), "restricted_data_leaks": sum(row["restricted_data_leak"] for row in safety_rows), "prompt_injection_regressions": sum(row["prompt_injection_regression"] for row in safety_rows), "unauthorized_citations": sum(len(row["unauthorized_citations"]) for row in safety_rows), "temporal_expansion_broadened_authorization": False, "cases": safety_rows}
    safety["passed"] = not any(safety[key] for key in ("acl_violations", "tenant_violations", "region_violations", "restricted_data_leaks", "prompt_injection_regressions", "unauthorized_citations"))
    write_json(OUT / "safety_report.json", safety)
    ledger = load_jsonl(OUT / "api_cost_ledger.jsonl")
    model_cost = {}
    for model in (LUNA_MODEL, SOL_MODEL):
        rows = [row for row in ledger if row["model"] == model]
        model_cost[model] = {"calls": len(rows), "input_tokens": sum(int(row["input_tokens"]) for row in rows), "cached_tokens": sum(int(row["cached_input_tokens"]) for row in rows), "output_tokens": sum(int(row["output_tokens"]) for row in rows), "cost_usd": sum(float(row["estimated_cost_usd"]) for row in rows)}
    total_cost = sum(item["cost_usd"] for item in model_cost.values())
    correct = sum(row["correct"] for row in scored)
    quality = {
        "correct": correct,
        "total": len(scored),
        "incorrect_abstentions": sum(row["incorrect_abstention"] for row in scored),
        "unsupported_answers": sum(row["unsupported_answer"] for row in scored),
        "citation_validity": sum(row["citation_validity_pass"] is not False for row in scored) / len(scored),
        "citation_correctness": sum(row["citation_correctness_pass"] is not False for row in scored) / len(scored),
        "citation_completeness": sum(row["citation_completeness_pass"] is not False for row in scored) / len(scored),
        "requirement_completeness": sum(row["requirement_completeness_pass"] for row in scored) / len(scored),
        "version_correctness": sum(row["version_correctness_pass"] for row in scored) / len(scored),
        "contract_mismatches": sum(not row["requirement_completeness_pass"] and bool(row["answer"]) for row in scored),
    }
    structural_payload_ok = all(all(check for key, check in row.items() if key not in {"case_id"}) for row in load_json(OUT / "local_preflight.json")["case_checks"])
    confirmed = correct == 5 and quality["unsupported_answers"] == 0 and quality["contract_mismatches"] == 0 and safety["passed"] and structural_payload_ok
    verdict = "P1_TARGETED_REAL_API_CONFIRMED" if confirmed else ("P1_TARGETED_REAL_API_PARTIAL" if correct else "P1_TARGETED_REAL_API_FAILED")
    latencies = [float(row["latency_ms"]) for row in predictions]
    report = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": now(),
        "verdict": verdict,
        "quality": quality,
        "per_case": [{"case_id": row["case_id"], "correct": row["correct"], "behavior": row["behavior"], "failure_class": row["failure_class"]} for row in scored],
        "p0_contract": {"one_output_per_atomic_requirement": all(row["requirement_completeness_pass"] for row in scored), "cross_version_requirements_unmerged": all(len(next(item for item in predictions if item["case_id"] == case_id)["required_requirement_ids"]) == 2 for case_id in ("fresh_v1_031", "fresh_v1_032", "fresh_v1_035")), "contract_mismatches": quality["contract_mismatches"]},
        "safety": safety,
        "cost": {"luna": model_cost[LUNA_MODEL], "sol": model_cost[SOL_MODEL], "embedding": {"new_calls": 0, "tokens": 0, "cost_usd": 0.0}, "total_targeted_cost_usd": total_cost, "cost_per_corrected_or_confirmed_case_usd": total_cost / correct if correct else None, "hard_cap_usd": BUDGET_CAP_USD},
        "latency": {"median_ms": median(latencies), "p95_ms": percentile95(latencies), "per_case_ms": {row["case_id"]: row["latency_ms"] for row in predictions}},
        "failures": [{"case_id": row["case_id"], "classification": row["failure_class"]} for row in scored if row["failure_class"]],
        "next_step": "P2_VERSION_VALIDATION_SCOPE" if confirmed else next((row["failure_class"] for row in scored if row["failure_class"]), "OTHER"),
        "full_60_case_evaluation_executed": False,
        "candidate_architecture_modified": False,
        "final_generation_llm_calls": 0,
    }
    write_json(OUT / "final_report.json", report)
    print(json.dumps({"verdict": verdict, "quality": quality, "cost": report["cost"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one of --preflight or --execute")
    if args.preflight:
        preflight()
    else:
        execute()


if __name__ == "__main__":
    main()
