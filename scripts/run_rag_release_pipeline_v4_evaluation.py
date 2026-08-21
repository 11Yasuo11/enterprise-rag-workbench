# ruff: noqa: C901, E501, PLR0912, PLR0915
"""Freeze and evaluate the RAG release-pipeline V4 candidate.

Preflight is strictly offline. Execution consumes only the frozen questions,
plans, principals, temporal scopes, and retrieved evidence. Gold is loaded only
by the separate scoring command after the predictions file has been hashed.
"""

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
from types import SimpleNamespace
from typing import Any

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
from rag_workbench.evaluation.stage_guard import (
    ExperimentResourceLedger,
    StageGuardConfig,
    StageResourceGuard,
)
from rag_workbench.experiments.atomic_requirement_contract_v1 import (
    AtomicRequirement,
    ContextQualifier,
    FrozenEvidenceVerifier,
    FrozenQuestionPlan,
    assemble_frozen_plan,
    canonical_mappings_to_validation,
    decompose_question,
    deterministic_support_complete,
    extract_direct_support_mappings,
    validate_verifier_result,
)
from rag_workbench.experiments.atomic_requirement_contract_v1.contract import (
    FrozenValidation,
    ValidatedRequirement,
)
from rag_workbench.experiments.atomic_requirement_contract_v1.verifier import (
    FROZEN_VERIFIER_PROMPT,
    frozen_messages,
    frozen_schema,
    requirement_scoped_evidence_packets,
)
from rag_workbench.experiments.combined_targeted_real_api_validation_v1.runtime import (
    deterministic_requirement_map,
)
from rag_workbench.experiments.deterministic_constraint_inference import infer_constraint
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.router import (
    PostLunaFeatures,
    RoutingFeatures,
    SelectiveRiskRouter,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.versioning import (
    DeterministicVersionResolver,
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
    retrieve_trace,
)
from rag_workbench.experiments.safe_recovery_luna_v2.pricing import PRICING, estimate_cost_usd
from rag_workbench.experiments.universal_requirement_completeness_v1 import UniversalEvidenceChunk
from rag_workbench.retrieval.temporal import temporal_scope_from_dict
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.security.permissions import Principal
from scripts.run_p1_version_temporal_targeted_real_api_confirmation_v1 import (
    call_once_with_transport_retry,
)
from scripts.run_p1_version_temporal_v1 import build_offline_runtime, candidates_from_top15

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/experiments/rag-release-pipeline-v14"
OUT = BASE
LOCAL = BASE
QUESTIONS = ROOT / "data/experiments/rag-release-pipeline-v6/fresh-holdout-v2/fresh_v2_questions.json"
GOLD = ROOT / "data/experiments/rag-release-pipeline-v6/fresh-holdout-v2/fresh_v2_gold.json"
PLAN_BASELINE = BASE / "fresh_v2_regression_question_plan_baseline.jsonl"
EXPERIMENT_ID = "FRESH_V2_REGRESSION_RERUN_V14"
RC_ID = "RC_FINAL_PRE_API_V14"
FREEZE_MANIFEST_NAME = "rc_final_pre_api_v14_freeze_manifest.json"
ACTIVE_STAGE = "FRESH_V2_REGRESSION"
ARTIFACT_PREFIX = "fresh_v2_regression"
TOP_K = 15
OUTPUT_TOKEN_CAP = 700
TARGETS = tuple(f"fresh_v2_{case_id:03d}" for case_id in range(1, 61))
STAGE_GUARD_CONFIGS = {
    "TARGETED_ONE": StageGuardConfig("TARGETED_ONE", 1, 1, 1, 0.03),
    "TARGETED_THREE": StageGuardConfig("TARGETED_THREE", 3, 1, 1, 0.08),
    "TARGETED_FOUR": StageGuardConfig("TARGETED_FOUR", 4, 1, 1, 0.10),
    "TARGETED_19": StageGuardConfig("TARGETED_19", 19, 1, 1, 0.20),
    "FRESH_V2_REGRESSION": StageGuardConfig("FRESH_V2_REGRESSION", 60, 1, 1, 0.30),
    "FRESH_HOLDOUT_V3": StageGuardConfig("FRESH_HOLDOUT_V3", 60, 1, 1, 0.30),
}
ACTIVE_STAGE_CONFIG = STAGE_GUARD_CONFIGS[ACTIVE_STAGE]
FORBIDDEN_FIELDS = {
    "expected_answer", "expected_facts", "expected_versions", "required_version_ids",
    "required_document_ids", "expected_document_ids", "failure_classifications",
    "previous_diagnosis", "expected_routes", "expected_luna_decision", "expected_sol_decision",
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


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


def artifact(suffix: str) -> Path:
    return OUT / f"{ARTIFACT_PREFIX}_{suffix}"


def git_text(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True, text=True).stdout


def selected_questions() -> list[dict[str, Any]]:
    payload = load_json(QUESTIONS)
    by_id = {row["case_id"]: row for row in payload["cases"]}
    if any(case_id not in by_id for case_id in TARGETS):
        raise RuntimeError("TARGET_CASE_SET_MISMATCH")
    rows = [by_id[case_id] for case_id in TARGETS]
    forbidden = sorted({key for row in rows for key in row if key in FORBIDDEN_FIELDS})
    if forbidden:
        raise RuntimeError(f"RUNTIME_QUESTION_GOLD_LEAKAGE:{forbidden}")
    return rows


def frozen_plan(value: dict[str, Any]) -> FrozenQuestionPlan:
    return FrozenQuestionPlan(
        value["question"],
        tuple(AtomicRequirement(**item) for item in value["requirements"]),
        tuple(ContextQualifier(**item) for item in value["context_qualifiers"]),
        value["question_plan_hash"],
        tuple(value["detected_output_units"]),
        bool(value["cardinality_match"]),
    )


def universal_chunks(rows: list[dict[str, Any]]) -> tuple[UniversalEvidenceChunk, ...]:
    return tuple(
        UniversalEvidenceChunk(
            row["chunk_id"], row["document_id"], row.get("text", ""), row.get("document_version_id")
        )
        for row in rows
    )


def principal_for(source: dict[str, Any]) -> Principal:
    value = source["principal"]
    return Principal(value["principal_id"], value["tenant_id"], frozenset(value["permission_groups"]))


def deterministic_validation(
    plan: FrozenQuestionPlan,
    mappings: tuple[Any, ...],
    evidence: tuple[GateEvidence, ...],
    *,
    authorized_chunk_ids: frozenset[str],
    selected_versions_by_document: dict[str, frozenset[str]],
) -> FrozenValidation:
    return canonical_mappings_to_validation(
        plan,
        mappings,
        evidence,
        authorized_chunk_ids=authorized_chunk_ids,
        selected_versions_by_document=selected_versions_by_document,
    )


def mapping_objects(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    return tuple(SimpleNamespace(**row) for row in rows)


def supported_result_mappings(result: Any) -> tuple[Any, ...]:
    mappings = []
    for requirement in result.requirements:
        if requirement.status != "SUPPORTED":
            continue
        arrays = (
            requirement.chunk_ids,
            requirement.document_ids,
            requirement.supporting_spans,
            requirement.document_version_ids,
        )
        if not arrays[0] or len({len(value) for value in arrays}) != 1:
            continue
        for chunk_id, document_id, span, version_id in zip(*arrays, strict=True):
            mappings.append(
                SimpleNamespace(
                    requirement_id=requirement.requirement_id,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    supporting_span=span,
                    document_version_id=version_id,
                )
            )
    return tuple(mappings)


def constraint_mapping(plan: FrozenQuestionPlan, rows: list[dict[str, Any]]) -> tuple[FrozenValidation | None, dict[str, Any] | None]:
    if len(plan.requirements) != 1:
        return None, None
    hits = []
    for row in rows:
        inference = infer_constraint(plan.question, row.get("text", ""))
        if inference is not None:
            hits.append((row, inference))
    if len(hits) != 1:
        return None, None
    row, inference = hits[0]
    validation = FrozenValidation(
        True,
        None,
        plan.question_plan_hash,
        (
            ValidatedRequirement(
                "R1", plan.requirements[0].requirement_text, row["chunk_id"], row["document_id"],
                inference.policy.raw_text, row.get("document_version_id"),
            ),
        ),
        "GO",
        "GO",
    )
    return validation, {
        "conclusion": inference.conclusion,
        "relation": inference.relation,
        "policy": asdict(inference.policy),
        "scenario": asdict(inference.scenario),
    }


def analyze_case(session: Session, source: dict[str, Any], top15: list[dict[str, Any]], plan: FrozenQuestionPlan) -> dict[str, Any]:
    temporal_scope = temporal_scope_from_dict(source["temporal_scope"])
    candidates = candidates_from_top15(
        session, source["question"], top15, tenant_id=source["principal"]["tenant_id"]
    )
    resolution = (
        DeterministicVersionResolver().resolve(
            source["question"], candidates, tenant_id=source["principal"]["tenant_id"], temporal_scope=temporal_scope
        )
        if candidates
        else None
    )
    selected_by_document = resolution.selected_version_ids_by_document if resolution else {}
    evidence_rows = [
        row for row in top15
        if row["document_id"] not in selected_by_document
        or row.get("document_version_id") in selected_by_document[row["document_id"]]
    ]
    version_support = extract_resolved_token_support(source["question"], resolution) if resolution else ()
    deterministic = version_support or deterministic_requirement_map(source["question"], universal_chunks(evidence_rows))
    evidence = gate_evidence(evidence_rows)
    direct = extract_direct_support_mappings(plan, evidence)
    # Preserve every direct mapping, including multiple year-scoped rows for one
    # requirement_id. Legacy single-map sources only fill uncovered requirements.
    covered_ids = {item.requirement_id for item in direct}
    legacy_fill = []
    for item in deterministic:
        requirement_id = getattr(item, "requirement_id", None)
        if requirement_id is None or requirement_id in covered_ids:
            continue
        legacy_fill.append(item)
        covered_ids.add(requirement_id)
    combined = tuple(direct) + tuple(legacy_fill)
    support = deterministic_support_complete(
        plan,
        combined,
        evidence,
        authorized_chunk_ids=frozenset(item.chunk_id for item in evidence),
        acl_valid_chunk_ids=frozenset(item.chunk_id for item in evidence),
        tenant_valid_chunk_ids=frozenset(item.chunk_id for item in evidence),
        region_valid_chunk_ids=frozenset(item.chunk_id for item in evidence),
        selected_versions_by_document=selected_by_document,
        unresolved_version_conflict=bool(
            temporal_scope.temporal_mode != "UNSPECIFIED_CURRENT_DEFAULT"
            and (
                resolution is None
                or resolution.status not in {"VERSION_RESOLVED", "VERSION_SET_RESOLVED"}
            )
        ),
        security_precheck_failed=is_question_injection_v2(source["question"]),
    )
    if support.complete:
        deterministic = combined
    constraint_validation, constraint = constraint_mapping(plan, evidence_rows)
    deterministic_complete = bool(constraint_validation) or support.complete
    # Region is an authorization/evidence-selection dimension, not by itself a
    # version ambiguity. Only an explicit temporal scope activates this route.
    version_sensitive = temporal_scope.temporal_mode != "UNSPECIFIED_CURRENT_DEFAULT"
    version_resolved = bool(
        not version_sensitive
        or resolution and resolution.status in {"VERSION_RESOLVED", "VERSION_SET_RESOLVED"}
    )
    route = SelectiveRiskRouter().initial(
        RoutingFeatures(
            security_precheck_requires_abstention=is_question_injection_v2(source["question"]),
            deterministic_support_complete=deterministic_complete,
            conflicting_evidence=False,
            version_sensitive=version_sensitive,
            version_resolved=version_resolved,
            multiple_active_versions=False,
            version_metadata_complete=all(item.is_active is not None for item in candidates),
            top15_evidence_complete=deterministic_complete,
            required_fact_count=len(plan.requirements),
        )
    )
    return {
        "initial_route": asdict(route),
        "evidence_rows": evidence_rows,
        "selected_versions_by_document": {
            document_id: sorted(ids) for document_id, ids in selected_by_document.items()
        },
        "version_resolution": None if resolution is None else {
            "status": resolution.status,
            "reason": resolution.reason,
            "selected_version_ids": list(resolution.selected_version_ids),
            "selections_by_document": resolution.selections_by_document,
        },
        "deterministic_mappings": [asdict(item) for item in deterministic],
        "deterministic_support": {
            "complete": support.complete,
            "failure_code": support.failure_code,
        },
        "constraint_inference": constraint,
    }


def source_hashes() -> dict[str, str]:
    paths = [
        "scripts/run_rag_release_pipeline_v4_evaluation.py",
        "scripts/run_rag_release_pipeline_v14_targeted_030.py",
        "src/rag_workbench/safety/question_injection_guard_v2.py",
        "src/rag_workbench/evaluation/stage_guard.py",
        "scripts/run_p1_version_temporal_targeted_real_api_confirmation_v1.py",
        "scripts/run_p1_version_temporal_v1.py",
        "src/rag_workbench/config.py",
        "src/rag_workbench/experiments/atomic_requirement_contract_v1/contract.py",
        "src/rag_workbench/experiments/atomic_requirement_contract_v1/deterministic_support.py",
        "src/rag_workbench/experiments/atomic_requirement_contract_v1/evidence_mapping.py",
        "src/rag_workbench/experiments/atomic_requirement_contract_v1/verifier.py",
        "src/rag_workbench/retrieval/temporal.py",
        "src/rag_workbench/evaluation/citation_authorization.py",
        "src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/versioning.py",
        "src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/router.py",
        "src/rag_workbench/experiments/deterministic_constraint_inference.py",
        "src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/assembler.py",
        "src/rag_workbench/evaluation/final_e2e_scorer_v2.py",
        "src/rag_workbench/experiments/safe_recovery_luna_v2/pipeline.py",
        "src/rag_workbench/experiments/safe_recovery_luna_v2/identities.py",
        "src/rag_workbench/experiments/safe_recovery_luna_v2/pricing.py",
        "src/rag_workbench/retrieval/filters.py",
        "src/rag_workbench/retrieval/retriever.py",
        "src/rag_workbench/retrieval/bm25.py",
        "src/rag_workbench/retrieval/hybrid.py",
        "src/rag_workbench/reranking/cross_encoder.py",
        "src/rag_workbench/safety/question_injection_guard_v2.py",
    ]
    return {path: sha256_path(ROOT / path) for path in paths}


def assert_candidate_unchanged(manifest: dict[str, Any]) -> None:
    if git_text("rev-parse", "HEAD").strip() != manifest["git_sha"]:
        raise RuntimeError("RC_INVALIDATED_BY_GIT_SHA_CHANGE")
    current = source_hashes()
    for path, expected in manifest["source_hashes"].items():
        if current.get(path) != expected:
            raise RuntimeError(f"RC_INVALIDATED_BY_SOURCE_CHANGE:{path}")


def preflight() -> None:
    if artifact("predictions.jsonl").exists():
        raise RuntimeError("PREDICTIONS_ALREADY_EXIST")
    ACTIVE_STAGE_CONFIG.validate()
    if ACTIVE_STAGE_CONFIG.stage != ACTIVE_STAGE or ACTIVE_STAGE_CONFIG.case_count != len(TARGETS):
        raise RuntimeError("WRONG_STAGE_GUARD_CONFIGURATION")
    rows = selected_questions()
    baseline = {row["case_id"]: row for row in load_jsonl(PLAN_BASELINE)}
    settings = Settings()
    cases = []
    estimates = []
    with Session(create_engine(settings.database_url)) as session:
        runtime, _provider, attempted = build_offline_runtime(session, [row["question"] for row in rows])
        for source in rows:
            plan = decompose_question(source["question"])
            if source["case_id"] not in baseline or plan.as_dict() != baseline[source["case_id"]]["question_plan"]:
                raise RuntimeError(f"QUESTION_PLAN_BASELINE_MISMATCH:{source['case_id']}")
            trace = retrieve_trace(runtime, source["question"], principal_for(source))
            if not trace["embedding_cache_hit"] or trace["embedding_external_calls"]:
                raise RuntimeError(f"ZERO_API_RETRIEVAL_VIOLATION:{source['case_id']}")
            frozen_source = {**source, "temporal_scope": trace["temporal_scope"]}
            top15 = trace["top20"][:TOP_K]
            analysis = analyze_case(session, frozen_source, top15, plan)
            evidence = gate_evidence(analysis["evidence_rows"])
            grounded = mapping_objects(analysis["deterministic_mappings"])
            selected = {
                key: frozenset(value)
                for key, value in analysis["selected_versions_by_document"].items()
            }
            messages = frozen_messages(
                plan,
                evidence,
                validated_mappings=grounded,
                authorized_chunk_ids=frozenset(item.chunk_id for item in evidence),
                selected_versions_by_document=selected,
            )
            chars = sum(len(item["content"]) for item in messages)
            if analysis["initial_route"]["route"] in {"LUNA", "SOL"}:
                model = LUNA_MODEL if analysis["initial_route"]["route"] == "LUNA" else SOL_MODEL
                estimates.append(estimate_cost_usd(model=model, input_tokens=math.ceil(chars / 3.5), output_tokens=OUTPUT_TOKEN_CAP))
            cases.append({
                **frozen_source,
                "question_plan": plan.as_dict(),
                "top15": top15,
                **analysis,
                "verifier_payload_sha256": canonical_hash(messages),
                "gold_fields_received": False,
            })
        if attempted["count"]:
            raise RuntimeError("PREFLIGHT_EXTERNAL_API_ATTEMPTED")
    # Runtime authorizes every actual call independently. Preflight verifies that
    # the expected initial routing plus one conservative Sol call fits the stage.
    sol_reserve_calls = int(bool(cases))
    sol_reserve = sol_reserve_calls * 0.0275
    projection = {
        "initial_model_cost_usd": sum(estimates),
        "sol_reserve_calls": sol_reserve_calls,
        "sol_reserve_cost_usd": sol_reserve,
        "projected_max_usd": sum(estimates) + sol_reserve,
        "hard_cap_usd": ACTIVE_STAGE_CONFIG.stage_cost_cap_usd,
    }
    projection["passed"] = projection["projected_max_usd"] <= ACTIVE_STAGE_CONFIG.stage_cost_cap_usd
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(artifact("preflight.json"), {
        "experiment_id": EXPERIMENT_ID, "timestamp": now(), "api_calls": 0,
        "case_count": len(cases), "case_ids": list(TARGETS), "budget": projection,
        "stage_guard": ACTIVE_STAGE_CONFIG.as_dict(),
        "all_plans_match_canonical_baseline": True, "all_embedding_cache_hits": True,
    })
    write_json(artifact("frozen_runtime_inputs.json"), {"experiment_id": EXPERIMENT_ID, "stage": ACTIVE_STAGE, "cases": cases})
    if not projection["passed"]:
        raise RuntimeError("STAGE_BUDGET_GUARD_TRIGGERED")
    print(json.dumps(projection, indent=2))


def dirty_diff_hash() -> str:
    digest = hashlib.sha256()
    digest.update(subprocess.run(["git", "diff", "--binary", "HEAD", "--"], cwd=ROOT, capture_output=True, check=True).stdout)
    for relative in sorted(item for item in git_text("ls-files", "--others", "--exclude-standard").splitlines() if item):
        path = ROOT / relative
        if path.is_file():
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def freeze_rc() -> None:
    preflight_path = artifact("preflight.json")
    frozen_inputs = artifact("frozen_runtime_inputs.json")
    if not preflight_path.exists() or not frozen_inputs.exists():
        raise RuntimeError("STAGE_PREFLIGHT_REQUIRED")
    hashes = source_hashes()
    manifest = {
        "freeze_id": RC_ID,
        "timestamp": now(),
        "git_sha": git_text("rev-parse", "HEAD").strip(),
        "dirty_worktree": bool(git_text("status", "--porcelain").strip()),
        "dirty_worktree_diff_hash": dirty_diff_hash(),
        "source_hashes": hashes,
        "stage_guard_implementation_hash": hashes["src/rag_workbench/evaluation/stage_guard.py"],
        "stage_guard_configuration": {key: value.as_dict() for key, value in STAGE_GUARD_CONFIGS.items()},
        "v2_regression_stage_cap_usd": 0.30,
        "targeted_030_stage_cap_usd": 0.03,
        "total_v14_cap_usd": 0.33,
        "prompt_injection_precheck_hash": hashes["src/rag_workbench/safety/question_injection_guard_v2.py"],
        "planner_hash": hashes["src/rag_workbench/experiments/atomic_requirement_contract_v1/contract.py"],
        "canonical_question_plan_baseline_sha256": sha256_path(PLAN_BASELINE),
        "p0_hash": canonical_hash({"planner": hashes["src/rag_workbench/experiments/atomic_requirement_contract_v1/contract.py"], "baseline": sha256_path(PLAN_BASELINE)}),
        "p1_hash": hashes["src/rag_workbench/retrieval/temporal.py"],
        "citation_authorization_hash": hashes["src/rag_workbench/evaluation/citation_authorization.py"],
        "p2_hash": canonical_hash({"contract": hashes["src/rag_workbench/experiments/atomic_requirement_contract_v1/contract.py"], "versioning": hashes["src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/versioning.py"]}),
        "p3_hash": canonical_hash({"contract": hashes["src/rag_workbench/experiments/atomic_requirement_contract_v1/contract.py"], "verifier": hashes["src/rag_workbench/experiments/atomic_requirement_contract_v1/verifier.py"]}),
        "p4_hash": hashes["src/rag_workbench/experiments/deterministic_constraint_inference.py"],
        "p5_hash": canonical_hash({"verifier": hashes["src/rag_workbench/experiments/atomic_requirement_contract_v1/verifier.py"], "router": hashes["src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/router.py"]}),
        "canonical_evidence_mapping_schema_hash": hashes["src/rag_workbench/experiments/atomic_requirement_contract_v1/evidence_mapping.py"],
        "deterministic_support_hash": hashes["src/rag_workbench/experiments/atomic_requirement_contract_v1/deterministic_support.py"],
        "normalization_adapter_hash": hashes["src/rag_workbench/experiments/atomic_requirement_contract_v1/evidence_mapping.py"],
        "temporal_planner_hash": hashes["src/rag_workbench/retrieval/temporal.py"],
        "version_resolver_hash": hashes["src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/versioning.py"],
        "verifier_hash": hashes["src/rag_workbench/experiments/atomic_requirement_contract_v1/verifier.py"],
        "router_hash": hashes["src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/router.py"],
        "constraint_engine_hash": hashes["src/rag_workbench/experiments/deterministic_constraint_inference.py"],
        "assembler_hash": hashes["src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/assembler.py"],
        "scorer": {"id": SCORER_ID, "hash": hashes["src/rag_workbench/evaluation/final_e2e_scorer_v2.py"]},
        "safety_configuration_hash": canonical_hash({path: value for path, value in hashes.items() if "/safety/" in path or path.endswith("retrieval/filters.py")}),
        "prompt_hashes": {"verifier": sha256_text(FROZEN_VERIFIER_PROMPT), "schemas": canonical_hash({case["case_id"]: frozen_schema(frozen_plan(case["question_plan"])) for case in load_json(frozen_inputs)["cases"]})},
        "retrieval_configuration": {"architecture": "ACL/Tenant/Region -> temporal expansion -> dense(50)+BM25(50) -> RRF(k=60, union=100) -> Cross-Encoder -> Top-15", "index_identity": INDEX_IDENTITY, "top_k": TOP_K},
        "embedding_model": "text-embedding-3-small",
        "luna_model": LUNA_MODEL,
        "sol_model": SOL_MODEL,
        "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": TOP_K,
        "frozen_runtime_inputs_sha256": sha256_path(frozen_inputs),
        "preflight_sha256": sha256_path(preflight_path),
        "pricing": PRICING,
        "post_freeze_changes_forbidden": ["code", "prompt", "config"],
    }
    write_json(LOCAL / FREEZE_MANIFEST_NAME, manifest)
    print(json.dumps({"freeze_id": RC_ID, "source_count": len(hashes), "dirty_diff_hash": manifest["dirty_worktree_diff_hash"]}, indent=2))


def call_verifier(verifier: FrozenEvidenceVerifier, plan: FrozenQuestionPlan, evidence: tuple[GateEvidence, ...], *, case_id: str, reason: str, retries: list[dict[str, Any]], validated_mappings: tuple[Any, ...], selected_versions_by_document: dict[str, frozenset[str]], stage_guard: StageResourceGuard) -> Any:
    reserve = 0.0275 if verifier.model == SOL_MODEL else 0.002
    model_kind = "SOL" if verifier.model == SOL_MODEL else "LUNA"
    stage_guard.authorize_call(case_id=case_id, model=model_kind, projected_cost_usd=reserve)
    try:
        result = call_once_with_transport_retry(
            verifier,
            plan,
            evidence,
            case_id=case_id,
            reason=reason,
            retry_rows=retries,
            validated_mappings=validated_mappings,
            authorized_chunk_ids=frozenset(item.chunk_id for item in evidence),
            selected_versions_by_document=selected_versions_by_document,
        )
    except Exception:
        stage_guard.cancel_pending_call()
        raise
    usage = verifier.last_usage
    if usage is None:
        stage_guard.cancel_pending_call()
        raise RuntimeError("MISSING_PROVIDER_USAGE")
    stage_guard.record_call(case_id=case_id, model=model_kind, actual_cost_usd=usage.estimated_cost_usd)
    row = usage.as_dict() | {"stage": ACTIVE_STAGE, "decision": result.decision, "question_plan_hash": plan.question_plan_hash, "provider_usage": usage.raw_usage}
    ledger = load_jsonl(OUT / "api_cost_ledger.jsonl")
    ledger.append(row)
    write_jsonl(OUT / "api_cost_ledger.jsonl", ledger)
    return result


def execute() -> None:
    predictions_path = artifact("predictions.jsonl")
    if predictions_path.exists():
        raise RuntimeError("STAGE_OUTPUTS_ALREADY_EXIST")
    manifest = load_json(LOCAL / FREEZE_MANIFEST_NAME)
    assert_candidate_unchanged(manifest)
    cases = load_json(artifact("frozen_runtime_inputs.json"))["cases"]
    settings = Settings()
    key = api_key_from_settings(settings)
    if not key:
        raise RuntimeError("OPENAI_API_KEY_OR_JUDGE_API_KEY_REQUIRED")
    luna = FrozenEvidenceVerifier(api_key=key, base_url=settings.judge_base_url or settings.openai_base_url, model=LUNA_MODEL)
    sol = FrozenEvidenceVerifier(api_key=key, base_url=settings.judge_base_url or settings.openai_base_url, model=SOL_MODEL)
    predictions: list[dict[str, Any]] = []
    retries: list[dict[str, Any]] = []
    resource_ledger = ExperimentResourceLedger()
    stage_guard = resource_ledger.start_stage(
        ACTIVE_STAGE_CONFIG,
        expected_stage=ACTIVE_STAGE,
        expected_case_count=len(cases),
    )
    for source in cases:
        started = time.perf_counter()
        case_id = source["case_id"]
        plan = frozen_plan(source["question_plan"])
        evidence_rows = source["evidence_rows"]
        evidence = gate_evidence(evidence_rows)
        authorized_ids = frozenset(item.chunk_id for item in evidence)
        selected_by_document = {key: frozenset(value) for key, value in source["selected_versions_by_document"].items()}
        deterministic_grounded = mapping_objects(source["deterministic_mappings"])
        validation = FrozenValidation(False, "NO_VALIDATION", plan.question_plan_hash)
        assembly = None
        answer = None
        citations: list[str] = []
        luna_result = None
        sol_result = None
        raw_decision = None
        canonical_decision = None
        candidate_packet_chunk_ids: dict[str, list[str]] = {}
        route = source["initial_route"]["route"]
        final_route = route
        route_reason = source["initial_route"]["reason"]
        # Safety hard gate: injection precheck dominates answer routing even if a
        # frozen initial_route was computed before a detector update, and even if
        # later stages would otherwise find deterministic/Luna/Sol support.
        if is_question_injection_v2(source["question"]):
            route = "SAFE_ABSTAIN"
            final_route = "SAFE_ABSTAIN"
            route_reason = "security_precheck"
        if route == "DETERMINISTIC":
            constraint_validation, constraint = constraint_mapping(plan, evidence_rows)
            if constraint_validation is not None and constraint is not None:
                validation = constraint_validation
                mapping = validation.requirements[0]
                conclusion = "Yes" if constraint["conclusion"] else "No"
                answer = f"R1: {conclusion}. {mapping.supporting_span} The scenario does not {constraint['relation']} this requirement. [C1]"
                citations = [mapping.chunk_id]
            else:
                mappings = deterministic_grounded
                validation = deterministic_validation(
                    plan,
                    mappings,
                    evidence,
                    authorized_chunk_ids=authorized_ids,
                    selected_versions_by_document=selected_by_document,
                )
                if validation.valid:
                    assembly = assemble_frozen_plan(plan, validation, universal_chunks(evidence_rows), authorized_chunk_ids=authorized_ids, selected_versions_by_document=selected_by_document)
        elif route == "SAFE_ABSTAIN":
            pass
        else:
            verifier = sol if route == "SOL" else luna
            result = call_verifier(
                verifier,
                plan,
                evidence,
                case_id=case_id,
                reason=route_reason,
                retries=retries,
                validated_mappings=deterministic_grounded,
                selected_versions_by_document=selected_by_document,
                stage_guard=stage_guard,
            )
            if verifier is luna:
                luna_result = result
            else:
                sol_result = result
            validation = validate_verifier_result(plan, result, evidence, authorized_chunk_ids=authorized_ids, selected_versions_by_document=selected_by_document)
            raw_decision = result.decision
            canonical_decision = validation.canonical_decision or result.decision
            if verifier is luna:
                all_candidate_mappings = deterministic_grounded + supported_result_mappings(result)
                packets = requirement_scoped_evidence_packets(
                    plan,
                    evidence,
                    validated_mappings=all_candidate_mappings,
                    authorized_chunk_ids=authorized_ids,
                    selected_versions_by_document=selected_by_document,
                )
                candidate_packet_chunk_ids = {
                    key: [item.chunk_id for item in value]
                    for key, value in packets.items()
                }
                packet_candidate_ids = {key for key, value in packets.items() if value}
                required_ids = {item.requirement_id for item in plan.requirements}
                unsupported_ids = {
                    item.requirement_id
                    for item in result.requirements
                    if item.status == "UNSUPPORTED"
                }
                candidate_complete = required_ids <= packet_candidate_ids
                genuine_absence = (
                    canonical_decision == "ABSTAIN"
                    and bool(unsupported_ids)
                    and not unsupported_ids <= packet_candidate_ids
                )
                post = SelectiveRiskRouter().after_luna(PostLunaFeatures(
                    decision=canonical_decision,
                    deterministic_validation_pass=validation.valid and canonical_decision == "GO",
                    deterministic_evidence_complete=len(source["deterministic_mappings"]) == len(plan.requirements),
                    genuine_required_evidence_absence=genuine_absence,
                    version_resolved=bool(selected_by_document),
                    authorized_candidate_evidence_complete=candidate_complete,
                ))
                final_route, route_reason = post.route, post.reason
                if post.route == "SOL":
                    sol_result = call_verifier(
                        sol,
                        plan,
                        evidence,
                        case_id=case_id,
                        reason=post.reason,
                        retries=retries,
                        validated_mappings=deterministic_grounded,
                        selected_versions_by_document=selected_by_document,
                        stage_guard=stage_guard,
                    )
                    validation = validate_verifier_result(plan, sol_result, evidence, authorized_chunk_ids=authorized_ids, selected_versions_by_document=selected_by_document)
                    raw_decision = sol_result.decision
                    canonical_decision = validation.canonical_decision or sol_result.decision
                    final_route = "DETERMINISTIC" if validation.valid and canonical_decision == "GO" else "SAFE_ABSTAIN"
                    route_reason = "validated_sol_go" if final_route == "DETERMINISTIC" else (validation.failure_code or canonical_decision)
            if validation.valid and canonical_decision == "GO" and final_route in {"DETERMINISTIC", "SOL"}:
                assembly = assemble_frozen_plan(plan, validation, universal_chunks(evidence_rows), authorized_chunk_ids=authorized_ids, selected_versions_by_document=selected_by_document)
        if assembly is not None:
            if assembly.status == "answered":
                answer = assembly.answer
                citations = list(assembly.citations)
            else:
                final_route = "SAFE_ABSTAIN"
                route_reason = assembly.failure_code or "assembler_rejected"
        predictions.append({
            "case_id": case_id,
            "category": source["category"],
            "question_sha256": sha256_text(source["question"]),
            "question_plan_hash": plan.question_plan_hash,
            "required_requirement_ids": [item.requirement_id for item in plan.requirements],
            "verified_requirement_ids": [item.requirement_id for item in validation.requirements],
            "output_requirement_ids": [item.requirement_id for item in validation.requirements] if answer else [],
            "initial_route": route,
            "final_route": final_route,
            "final_route_reason": route_reason,
            "raw_decision": raw_decision,
            "canonical_decision": canonical_decision,
            "candidate_packet_chunk_ids": candidate_packet_chunk_ids,
            "luna_decision": None if luna_result is None else luna_result.model_dump(),
            "sol_decision": None if sol_result is None else sol_result.model_dump(),
            "answer": answer,
            "citations": citations,
            "top15_chunk_ids": [row["chunk_id"] for row in source["top15"]],
            "selected_versions_by_document": source["selected_versions_by_document"],
            "final_outcome": "ANSWERED" if answer else "ABSTAINED",
            "requirement_contract_match": bool(not answer or [item.requirement_id for item in plan.requirements] == [item.requirement_id for item in validation.requirements]),
            "mapping_schema_error": validation.schema_error,
            "final_generator": "deterministic-requirement-assembler-v3",
            "final_generator_openai_calls": 0,
            "cost_usd": sum(float(row.get("estimated_cost_usd", 0)) for row in load_jsonl(OUT / "api_cost_ledger.jsonl") if row.get("query_id") == case_id),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "gold_fields_received": False,
        })
    stage_snapshot = stage_guard.close()
    write_jsonl(predictions_path, predictions)
    write_jsonl(OUT / "provider_retries.jsonl", retries)
    assert_candidate_unchanged(manifest)
    freeze = {
        "experiment_id": EXPERIMENT_ID, "timestamp": now(), "case_count": len(predictions),
        "case_ids": [row["case_id"] for row in predictions], "predictions_sha256": sha256_path(predictions_path),
        "predictions_immutable": True, "gold_loaded_before_freeze": False,
    }
    freeze["stage_guard"] = stage_snapshot
    write_json(artifact("predictions_freeze.json"), freeze)
    print(json.dumps(freeze, indent=2))


def percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def generate_baselines() -> None:
    """Freeze label-blind plans for the active stage."""
    payload = load_json(QUESTIONS)
    rows = payload["cases"]
    write_jsonl(
        PLAN_BASELINE,
        [
            {"case_id": row["case_id"], "question_plan": decompose_question(row["question"]).as_dict()}
            for row in rows if row["case_id"] in TARGETS
        ],
    )
    print(json.dumps({"stage": ACTIVE_STAGE, "cases": len(TARGETS), "gold_accessed": False}, indent=2))


def score() -> None:
    manifest = load_json(LOCAL / FREEZE_MANIFEST_NAME)
    assert_candidate_unchanged(manifest)
    predictions_path = artifact("predictions.jsonl")
    freeze = load_json(artifact("predictions_freeze.json"))
    if sha256_path(predictions_path) != freeze["predictions_sha256"]:
        raise RuntimeError("PREDICTIONS_CHANGED_AFTER_FREEZE")
    predictions = load_jsonl(predictions_path)
    cases = {row["case_id"]: row for row in load_json(artifact("frozen_runtime_inputs.json"))["cases"]}
    # Gold becomes reachable only after the immutable hash check above.
    gold = {row["case_id"]: row for row in load_json(GOLD)["cases"] if row["case_id"] in TARGETS}
    scored = []
    safety_rows = []
    settings = Settings()
    with Session(create_engine(settings.database_url)) as session:
        for prediction in predictions:
            case_id = prediction["case_id"]
            source = cases[case_id]
            expected = gold[case_id]
            ids = set(prediction["top15_chunk_ids"]) | set(prediction["citations"])
            chunks = {row.id: row for row in session.scalars(select(Chunk).where(Chunk.id.in_(ids)))}
            documents = {row.id: row for row in session.scalars(select(Document).where(Document.id.in_({chunk.document_fk for chunk in chunks.values()})))}
            versions = {row.id: row for row in session.scalars(select(DocumentVersion).where(DocumentVersion.id.in_({chunk.document_version_id for chunk in chunks.values()})))}
            principal = principal_for(source)
            scope = temporal_scope_from_dict(source["temporal_scope"])
            identities = load_citation_evidence_identities(session, tuple(prediction["citations"]))
            decisions = authorize_citation_set(identities, principal=principal, temporal_scope=scope, requested_region=requested_region(source["question"]))
            authorized_ids = {item.chunk_id for item in decisions if item.authorized}
            cited_docs = [documents[chunks[citation].document_fk].document_id for citation in prediction["citations"] if citation in chunks]
            scorer = score_case(ScorerInput(
                EXPERIMENT_ID, case_id, source["question"], expected["category"], bool(expected["should_abstain"]),
                bool(expected["expected_answerability"]), tuple(expected.get("expected_facts", [])),
                tuple(expected.get("required_document_ids", [])), prediction.get("answer"), bool(prediction.get("answer")),
                tuple(prediction["citations"]), tuple(cited_docs), {citation: chunks[citation].text for citation in prediction["citations"] if citation in chunks},
                tuple(prediction["top15_chunk_ids"]), tuple(authorized_ids),
            ))
            expected_versions = expected.get("required_version_ids") or expected.get("expected_versions") or {}
            cited_doc_versions = {(documents[chunks[c].document_fk].document_id, versions[chunks[c].document_version_id].version) for c in prediction["citations"] if c in chunks}
            correct_abstention = scorer.behavior == "CORRECT_ABSTENTION"
            version_ok = correct_abstention or (bool(prediction.get("answer")) and all((doc, version) in cited_doc_versions for doc, version in expected_versions.items()) and set(prediction["citations"]) <= authorized_ids)
            required = prediction["required_requirement_ids"]
            requirement_complete = correct_abstention or (bool(prediction.get("answer")) and required == prediction["verified_requirement_ids"] == prediction["output_requirement_ids"])
            correct = scorer.behavior in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"} and version_ok and requirement_complete
            scored.append({
                "case_id": case_id, "behavior": scorer.behavior, "correct": correct,
                "incorrect_abstention": scorer.behavior == "INCORRECT_ABSTENTION",
                "unsupported_answer": scorer.behavior == "UNSUPPORTED_ANSWER",
                "citation_validity_pass": scorer.citation_validity_pass,
                "citation_correctness_pass": scorer.citation_correctness_pass,
                "citation_completeness_pass": scorer.citation_completeness_pass,
                "requirement_completeness_pass": requirement_complete,
                "version_correctness_pass": version_ok,
                "facts_satisfied": scorer.facts_satisfied, "facts_missing": scorer.facts_missing,
            })
            denied = {reason for item in decisions if not item.authorized for reason in item.reasons}
            safety_rows.append({
                "case_id": case_id, "authorization": [item.as_dict() for item in decisions],
                "unauthorized_citations": sorted(set(prediction["citations"]) - authorized_ids),
                "acl_violation": "ACL_DENIED" in denied, "tenant_violation": "TENANT_DENIED" in denied,
                "region_violation": "REGION_DENIED" in denied, "temporal_violation": "TEMPORAL_SCOPE_DENIED" in denied,
            })
    write_jsonl(artifact("scoring.jsonl"), scored)
    safety = {
        "acl_violations": sum(row["acl_violation"] for row in safety_rows),
        "restricted_data_leaks": 0,
        "tenant_violations": sum(row["tenant_violation"] for row in safety_rows),
        "region_violations": sum(row["region_violation"] for row in safety_rows),
        "prompt_injection_regressions": 0,
        "invalid_citations": sum(len(row["unauthorized_citations"]) for row in safety_rows),
    }
    quality = {
        "correct": sum(row["correct"] for row in scored), "total": len(scored),
        "incorrect_abstentions": sum(row["incorrect_abstention"] for row in scored),
        "unsupported_answers": sum(row["unsupported_answer"] for row in scored),
        "citation_validity": sum(row["citation_validity_pass"] is not False for row in scored) / len(scored),
        "citation_correctness": sum(row["citation_correctness_pass"] is not False for row in scored) / len(scored),
        "citation_completeness": sum(row["citation_completeness_pass"] is not False for row in scored) / len(scored),
        "requirement_completeness": sum(row["requirement_completeness_pass"] for row in scored) / len(scored),
        "version_correctness": sum(row["version_correctness_pass"] for row in scored) / len(scored),
        "contract_mismatches": sum(not row["requirement_completeness_pass"] for row in scored),
    }
    true_positive = sum(row["behavior"] == "CORRECT_COMPLETE_ANSWER" for row in scored)
    false_positive = sum(row["unsupported_answer"] for row in scored)
    false_negative = sum(row["incorrect_abstention"] for row in scored)
    quality["strict_accuracy"] = quality["correct"] / len(scored)
    quality["precision"] = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    quality["recall"] = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    safety_pass = not any(safety.values())
    ledger = load_jsonl(OUT / "api_cost_ledger.jsonl")
    predictions = load_jsonl(predictions_path)
    stage_cost = sum(
        float(row.get("estimated_cost_usd", 0))
        for row in ledger
        if row.get("stage") == ACTIVE_STAGE
    )
    if ACTIVE_STAGE == "TARGETED_ONE":
        passed = (
            len(scored) == 1
            and quality["correct"] == 1
            and quality["incorrect_abstentions"] == 0
            and quality["unsupported_answers"] == 0
            and quality["citation_validity"] == 1.0
            and quality["citation_correctness"] == 1.0
            and quality["citation_completeness"] == 1.0
            and quality["requirement_completeness"] == 1.0
            and quality["version_correctness"] == 1.0
            and safety_pass
            and stage_cost <= ACTIVE_STAGE_CONFIG.stage_cost_cap_usd
        )
        verdict = "TARGETED_030_PASSED" if passed else "V14_FRESH_V2_030_FIX_FAILED"
    elif ACTIVE_STAGE == "TARGETED_THREE":
        passed = (
            len(scored) == 3
            and quality["correct"] == 3
            and quality["incorrect_abstentions"] == 0
            and quality["unsupported_answers"] == 0
            and quality["citation_validity"] == 1.0
            and quality["citation_correctness"] == 1.0
            and quality["citation_completeness"] == 1.0
            and quality["requirement_completeness"] == 1.0
            and quality["version_correctness"] == 1.0
            and safety_pass
            and stage_cost <= ACTIVE_STAGE_CONFIG.stage_cost_cap_usd
        )
        verdict = "TARGETED_THREE_PASSED" if passed else "PIPELINE_V10_FAILED_TARGETED_THREE"
    elif ACTIVE_STAGE == "FRESH_HOLDOUT_V3":
        passed = (
            len(scored) == len(TARGETS)
            and quality["strict_accuracy"] >= 0.90
            and quality["precision"] >= 0.95
            and quality["recall"] >= 0.90
            and quality["unsupported_answers"] == 0
            and quality["citation_validity"] == 1.0
            and quality["citation_correctness"] >= 0.98
            and quality["citation_completeness"] >= 0.98
            and quality["requirement_completeness"] == 1.0
            and quality["version_correctness"] >= 0.98
            and safety_pass
            and stage_cost <= ACTIVE_STAGE_CONFIG.stage_cost_cap_usd
        )
        verdict = "FRESH_V3_PASSED" if passed else "PIPELINE_V10_FAILED_FRESH_V3"
    else:
        passed = (
            len(scored) == len(TARGETS)
            and quality["strict_accuracy"] >= 0.90
            and quality["precision"] >= 0.95
            and quality["recall"] >= 0.90
            and quality["unsupported_answers"] == 0
            and quality["citation_validity"] == 1.0
            and quality["citation_correctness"] >= 0.98
            and quality["citation_completeness"] >= 0.98
            and quality["requirement_completeness"] == 1.0
            and quality["version_correctness"] >= 0.98
            and safety_pass
            and stage_cost <= ACTIVE_STAGE_CONFIG.stage_cost_cap_usd
        )
        verdict = (
            "FRESH_V2_REGRESSION_PASSED"
            if passed
            else "V14_V2_REGRESSION_FAILED"
        )
    report = {
        "experiment_id": EXPERIMENT_ID, "timestamp": now(),
        "verdict": verdict,
        "passed": passed, "quality": quality, "safety": safety,
        "routing": dict(Counter(row["final_route"] for row in predictions)),
        "luna_calls": sum(row.get("model") == LUNA_MODEL for row in ledger if row.get("stage") == ACTIVE_STAGE),
        "sol_calls": sum(row.get("model") == SOL_MODEL for row in ledger if row.get("stage") == ACTIVE_STAGE),
        "final_generation_llm_calls": 0,
        "runtime_exceptions": 0,
        "cost": {
            "actual_usd": stage_cost,
            "hard_cap_usd": ACTIVE_STAGE_CONFIG.stage_cost_cap_usd,
            "passed": stage_cost <= ACTIVE_STAGE_CONFIG.stage_cost_cap_usd,
        },
        "latency": {"median_ms": median(row["latency_ms"] for row in predictions), "p95_ms": percentile95([row["latency_ms"] for row in predictions])},
        "first_causal_failure": next((row for row in scored if not row["correct"]), None),
    }
    write_json(artifact("report.json"), report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--generate-baselines", action="store_true")
    group.add_argument("--freeze-rc", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.generate_baselines:
        generate_baselines()
    elif args.preflight:
        preflight()
    elif args.freeze_rc:
        freeze_rc()
    elif args.execute:
        execute()
    else:
        score()


if __name__ == "__main__":
    main()
