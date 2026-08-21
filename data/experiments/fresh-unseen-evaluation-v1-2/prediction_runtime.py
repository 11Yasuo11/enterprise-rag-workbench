from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rag_workbench.config import Settings
from rag_workbench.experiments.atomic_requirement_contract_v1 import (
    FrozenEvidenceVerifier,
    assemble_frozen_plan,
    decompose_question,
    validate_verifier_result,
)
from rag_workbench.experiments.atomic_requirement_contract_v1.contract import (
    FrozenValidation,
    ValidatedRequirement,
)
from rag_workbench.experiments.combined_targeted_real_api_validation_v1.runtime import (
    TargetCase,
    deterministic_requirement_map,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.router import (
    PostLunaFeatures,
    RoutingFeatures,
    SelectiveRiskRouter,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.versioning import (
    DeterministicVersionResolver,
    extract_resolved_token_support,
)
from rag_workbench.experiments.safe_recovery_luna_v2.identities import LUNA_MODEL, SOL_MODEL
from rag_workbench.experiments.safe_recovery_luna_v2.pipeline import (
    api_key_from_settings,
    build_runtime,
    gate_evidence,
    is_authorized,
    retrieve_trace,
    support_meta,
)
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.security.permissions import Principal
from scripts.run_combined_targeted_real_api_validation_v1 import (
    convert_verified,
    universal_chunks,
    version_candidates,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
QUESTIONS = (
    ROOT
    / ".local/evaluation-input/fresh_unseen_eval_v1_1_20260820"
    / "fresh_unseen_questions_v1_1.json"
)
PREDICTIONS = OUT / "fresh_predictions.jsonl"
LEDGER = OUT / "api_cost_ledger.jsonl"
RETRIES = OUT / "provider_retries.jsonl"
TOP_K = 15
MAX_SOL_CALLS = 12
TOTAL_BUDGET_USD = 0.50
EMBEDDING_COST_USD = 0.0000208
FORBIDDEN_FIELDS = {
    "expected_answer",
    "expected_facts",
    "required_document_ids",
    "required_versions",
    "required_fact_ids",
    "required_chunk_markers",
    "should_abstain",
    "expected_answerability",
    "forbidden_document_ids",
    "security_checks",
    "gold_notes",
}


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def walk_keys(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def candidate_unchanged() -> None:
    freeze = json.loads((OUT / "freeze_manifest.json").read_text())
    for path, expected in freeze["candidate_source_config_hashes"].items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"FRESH_EVAL_INVALIDATED_BY_CODE_CHANGE:{path}")


def version_state(session: Session, top15: list[dict], question: str, tenant_id: str):
    candidates = version_candidates(session, top15, question)
    if not candidates:
        return None, None, candidates
    resolution = DeterministicVersionResolver().resolve(
        question, candidates, tenant_id=tenant_id
    )
    detail = {
        "status": resolution.status,
        "reason": resolution.reason,
        "selected_document_id": resolution.candidate.document_id
        if resolution.candidate
        else None,
        "selected_version": resolution.candidate.version if resolution.candidate else None,
        "selected_version_id": resolution.candidate.document_version_id
        if resolution.candidate
        else None,
        "requested_region": resolution.requested_region,
        "requested_version": resolution.requested_version,
        "temporal_semantics": resolution.temporal_semantics,
    }
    selected = (
        frozenset({resolution.candidate.document_version_id})
        if resolution.status == "VERSION_RESOLVED" and resolution.candidate
        else frozenset()
    )
    return selected, detail, candidates


def initial_analysis(session: Session, source: dict, top15: list[dict], plan):
    selected, resolution, candidates = version_state(
        session, top15, source["question"], source["principal"]["tenant_id"]
    )
    chunks = universal_chunks(top15)
    resolved_object = (
        DeterministicVersionResolver().resolve(
            source["question"],
            candidates,
            tenant_id=source["principal"]["tenant_id"],
        )
        if candidates
        else None
    )
    version_support = (
        extract_resolved_token_support(source["question"], resolved_object)
        if resolved_object
        else ()
    )
    deterministic = (
        convert_verified(version_support, top15)
        if version_support
        else deterministic_requirement_map(source["question"], chunks)
    )
    complete = len(deterministic) == len(plan.requirements)
    active = [candidate for candidate in candidates if candidate.is_active]
    multiple_active = len({candidate.document_version_id for candidate in active}) > 1 and not any(
        token in source["question"].casefold() for token in ("east", "west")
    )
    route = SelectiveRiskRouter().initial(
        RoutingFeatures(
            security_precheck_requires_abstention=is_question_injection_v2(source["question"]),
            deterministic_support_complete=complete,
            conflicting_evidence=False,
            version_sensitive=bool(candidates),
            version_resolved=bool(selected),
            multiple_active_versions=multiple_active,
            version_metadata_complete=all(candidate.is_active is not None for candidate in candidates),
            top15_evidence_complete=complete,
            required_fact_count=len(plan.requirements),
        )
    )
    return route, deterministic, selected, resolution


def deterministic_validation(plan, deterministic) -> FrozenValidation:
    if len(deterministic) != len(plan.requirements):
        return FrozenValidation(False, "DETERMINISTIC_REQUIREMENT_COUNT_MISMATCH", plan.question_plan_hash)
    mapped = tuple(
        ValidatedRequirement(
            planned.requirement_id,
            planned.requirement_text,
            item.chunk_id,
            item.document_id,
            item.supporting_span,
            item.document_version_id,
        )
        for planned, item in zip(plan.requirements, deterministic, strict=True)
    )
    return FrozenValidation(True, None, plan.question_plan_hash, mapped)


def ledger_cost() -> float:
    return sum(float(row.get("estimated_cost_usd", 0.0)) for row in load_jsonl(LEDGER))


def call_verifier(verifier, plan, evidence, *, case_id: str, reason: str):
    model_reserve = 0.04 if verifier.model == SOL_MODEL else 0.004
    if EMBEDDING_COST_USD + ledger_cost() + model_reserve > TOTAL_BUDGET_USD:
        raise RuntimeError("FRESH_EVAL_BUDGET_GUARD_TRIGGERED")
    for attempt in range(3):
        try:
            result = verifier.evaluate(
                plan,
                evidence,
                query_id=case_id,
                arm="FRESH_UNSEEN_EVALUATION_V1_2",
                routing_reason=reason,
            )
            usage = verifier.last_usage
            if usage is None:
                raise RuntimeError("MISSING_PROVIDER_USAGE")
            row = usage.as_dict()
            row.update(
                {
                    "decision": result.decision,
                    "question_plan_hash": plan.question_plan_hash,
                    "routing_reason": reason,
                    "attempt": attempt + 1,
                }
            )
            append_jsonl(LEDGER, row)
            return result
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retryable = isinstance(exc, httpx.TransportError) or (
                isinstance(exc, httpx.HTTPStatusError)
                and (exc.response.status_code == 429 or exc.response.status_code >= 500)
            )
            append_jsonl(
                RETRIES,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "case_id": case_id,
                    "model": verifier.model,
                    "attempt": attempt + 1,
                    "retryable": retryable,
                    "error_type": type(exc).__name__,
                },
            )
            if not retryable or attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    candidate_unchanged()
    payload = json.loads(QUESTIONS.read_text())
    leakage = sorted(set(walk_keys(payload)) & FORBIDDEN_FIELDS)
    if leakage:
        raise RuntimeError(f"FRESH_EVAL_GOLD_LEAKAGE_DETECTED:{leakage}")
    if payload.get("dataset_id") != "acmeai-fresh-unseen-eval-v1.1-20260820":
        raise RuntimeError("QUESTION_DATASET_ID_MISMATCH")
    existing = load_jsonl(PREDICTIONS)
    completed = {row["case_id"] for row in existing}
    if len(completed) != len(existing):
        raise RuntimeError("DUPLICATE_PREDICTION_CASE")
    settings = Settings()
    key = api_key_from_settings(settings)
    if not key:
        raise RuntimeError("OPENAI_API_KEY_OR_JUDGE_API_KEY_REQUIRED")
    luna = FrozenEvidenceVerifier(
        api_key=key,
        base_url=settings.judge_base_url or settings.openai_base_url,
        model=LUNA_MODEL,
    )
    sol = FrozenEvidenceVerifier(
        api_key=key,
        base_url=settings.judge_base_url or settings.openai_base_url,
        model=SOL_MODEL,
    )
    with Session(create_engine(settings.database_url)) as session:
        runtime = build_runtime(session, settings, prompt_cache=True)
        for source in payload["cases"]:
            if source["case_id"] in completed:
                continue
            started = time.perf_counter()
            principal_data = source["principal"]
            principal = Principal(
                principal_data["principal_id"],
                principal_data["tenant_id"],
                frozenset(principal_data["permission_groups"]),
            )
            plan = decompose_question(source["question"])
            trace = retrieve_trace(runtime, source["question"], principal)
            if not trace["embedding_cache_hit"] or trace["embedding_external_calls"]:
                raise RuntimeError(f"UNEXPECTED_EMBEDDING_CALL:{source['case_id']}")
            top15 = trace["top20"][:TOP_K]
            route, deterministic, selected_versions, version_resolution = initial_analysis(
                session, source, top15, plan
            )
            evidence_rows = top15
            if selected_versions and version_resolution:
                selected_document = version_resolution["selected_document_id"]
                evidence_rows = [
                    row
                    for row in top15
                    if row["document_id"] != selected_document
                    or row.get("document_version_id") in selected_versions
                ]
            evidence = gate_evidence(evidence_rows)
            authorized_ids = frozenset(item.chunk_id for item in evidence)
            validation = FrozenValidation(False, "NO_VALIDATION", plan.question_plan_hash)
            assembly = None
            luna_result = None
            sol_result = None
            final_route = route.route
            route_reason = route.reason
            if route.route == "SAFE_ABSTAIN":
                pass
            elif route.route == "DETERMINISTIC":
                validation = deterministic_validation(plan, deterministic)
                if validation.valid:
                    assembly = assemble_frozen_plan(
                        plan,
                        validation,
                        universal_chunks(evidence_rows),
                        authorized_chunk_ids=authorized_ids,
                        selected_version_ids=selected_versions,
                    )
            elif route.route == "SOL":
                if sum(row.get("model") == SOL_MODEL for row in load_jsonl(LEDGER)) >= MAX_SOL_CALLS:
                    raise RuntimeError("FRESH_EVAL_BUDGET_GUARD_TRIGGERED")
                sol_result = call_verifier(
                    sol, plan, evidence, case_id=source["case_id"], reason=route.reason
                )
                validation = validate_verifier_result(
                    plan,
                    sol_result,
                    evidence,
                    authorized_chunk_ids=authorized_ids,
                    selected_version_ids=selected_versions,
                )
                if validation.valid and sol_result.decision == "GO":
                    assembly = assemble_frozen_plan(
                        plan,
                        validation,
                        universal_chunks(evidence_rows),
                        authorized_chunk_ids=authorized_ids,
                        selected_version_ids=selected_versions,
                    )
            else:
                luna_result = call_verifier(
                    luna, plan, evidence, case_id=source["case_id"], reason=route.reason
                )
                validation = validate_verifier_result(
                    plan,
                    luna_result,
                    evidence,
                    authorized_chunk_ids=authorized_ids,
                    selected_version_ids=selected_versions,
                )
                genuine_absence = luna_result.decision == "ABSTAIN" and any(
                    item.status == "UNSUPPORTED" for item in luna_result.requirements
                )
                post = SelectiveRiskRouter().after_luna(
                    PostLunaFeatures(
                        decision=luna_result.decision,
                        deterministic_validation_pass=validation.valid,
                        deterministic_evidence_complete=bool(deterministic),
                        genuine_required_evidence_absence=genuine_absence,
                        version_resolved=bool(selected_versions),
                    )
                )
                final_route = post.route
                route_reason = post.reason
                if post.route == "DETERMINISTIC" and validation.valid:
                    assembly = assemble_frozen_plan(
                        plan,
                        validation,
                        universal_chunks(evidence_rows),
                        authorized_chunk_ids=authorized_ids,
                        selected_version_ids=selected_versions,
                    )
                elif post.route == "SOL":
                    if sum(row.get("model") == SOL_MODEL for row in load_jsonl(LEDGER)) >= MAX_SOL_CALLS:
                        raise RuntimeError("FRESH_EVAL_BUDGET_GUARD_TRIGGERED")
                    sol_result = call_verifier(
                        sol, plan, evidence, case_id=source["case_id"], reason=post.reason
                    )
                    validation = validate_verifier_result(
                        plan,
                        sol_result,
                        evidence,
                        authorized_chunk_ids=authorized_ids,
                        selected_version_ids=selected_versions,
                    )
                    if validation.valid and sol_result.decision == "GO":
                        assembly = assemble_frozen_plan(
                            plan,
                            validation,
                            universal_chunks(evidence_rows),
                            authorized_chunk_ids=authorized_ids,
                            selected_version_ids=selected_versions,
                        )
            if assembly is not None and assembly.status != "answered":
                assembly = None
            answer = assembly.answer if assembly else None
            citations = list(assembly.citations) if assembly else []
            metadata, permissions = support_meta(session, set(citations))
            authorized = all(
                citation in metadata
                and is_authorized(
                    metadata[citation],
                    principal,
                    permissions.get(metadata[citation]["document_fk"], set()),
                )
                for citation in citations
            )
            prediction = {
                "case_id": source["case_id"],
                "category": source["category"],
                "question_sha256": hashlib.sha256(source["question"].encode()).hexdigest(),
                "initial_route": route.route,
                "initial_route_reason": route.reason,
                "final_route": final_route,
                "final_route_reason": route_reason,
                "top15_chunk_ids": [row["chunk_id"] for row in top15],
                "top15_document_ids": [row["document_id"] for row in top15],
                "selected_version_ids": sorted(selected_versions) if selected_versions else [],
                "version_resolution": version_resolution,
                "question_plan_hash": plan.question_plan_hash,
                "requirement_count": len(plan.requirements),
                "required_requirement_ids": [item.requirement_id for item in plan.requirements],
                "verified_requirement_ids": [item.requirement_id for item in validation.requirements],
                "output_requirement_ids": list(assembly.output_requirement_ids) if assembly else [],
                "luna_decision": None if luna_result is None else luna_result.model_dump(),
                "sol_decision": None if sol_result is None else sol_result.model_dump(),
                "final_outcome": "ANSWERED" if answer else "ABSTAINED",
                "answer": answer,
                "citations": citations,
                "requirement_contract_match": bool(
                    not answer
                    or (
                        [item.requirement_id for item in plan.requirements]
                        == [item.requirement_id for item in validation.requirements]
                        == list(assembly.output_requirement_ids)
                    )
                ),
                "final_generator": "deterministic-requirement-assembler-v3",
                "final_generator_openai_calls": 0,
                "token_usage": {
                    "input": sum(
                        int(row.get("input_tokens", 0))
                        for row in load_jsonl(LEDGER)
                        if row.get("query_id") == source["case_id"]
                    ),
                    "cached_input": sum(
                        int(row.get("cached_input_tokens", 0))
                        for row in load_jsonl(LEDGER)
                        if row.get("query_id") == source["case_id"]
                    ),
                    "output": sum(
                        int(row.get("output_tokens", 0))
                        for row in load_jsonl(LEDGER)
                        if row.get("query_id") == source["case_id"]
                    ),
                },
                "cost_usd": sum(
                    float(row.get("estimated_cost_usd", 0.0))
                    for row in load_jsonl(LEDGER)
                    if row.get("query_id") == source["case_id"]
                ),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "safety_decision": "SAFE_ABSTAIN_SECURITY_PRECHECK"
                if route.reason == "security_precheck"
                else "AUTHORIZED_RUNTIME",
                "citations_authorized": authorized,
                "tenant_valid": authorized,
                "region_valid": authorized,
                "gold_fields_received": False,
            }
            append_jsonl(PREDICTIONS, prediction)
            completed.add(source["case_id"])
    rows = load_jsonl(PREDICTIONS)
    if len(rows) != 60 or {row["case_id"] for row in rows} != {
        case["case_id"] for case in payload["cases"]
    }:
        raise RuntimeError("INCOMPLETE_PREDICTIONS")
    candidate_unchanged()
    digest = hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest()
    freeze = {
        "experiment_id": "FRESH_UNSEEN_EVALUATION_V1_2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_count": 60,
        "predictions_path": str(PREDICTIONS.relative_to(ROOT)),
        "predictions_sha256": digest,
        "questions_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(),
        "gold_loaded_before_freeze": False,
        "predictions_immutable": True,
    }
    (OUT / "predictions_freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
