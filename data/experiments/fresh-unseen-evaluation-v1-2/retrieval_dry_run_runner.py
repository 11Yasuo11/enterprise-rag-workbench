from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rag_workbench.config import Settings
from rag_workbench.experiments.atomic_requirement_contract_v1.contract import decompose_question
from rag_workbench.experiments.atomic_requirement_contract_v1.verifier import FROZEN_VERIFIER_PROMPT
from rag_workbench.experiments.combined_targeted_real_api_validation_v1.runtime import (
    TargetCase,
    deterministic_requirement_map,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.router import (
    RoutingFeatures,
    SelectiveRiskRouter,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.versioning import (
    DeterministicVersionResolver,
    extract_resolved_token_support,
)
from rag_workbench.experiments.safe_recovery_luna_v2.pipeline import (
    build_runtime,
    retrieve_trace,
)
from rag_workbench.experiments.safe_recovery_luna_v2.pricing import estimate_cost_usd
from rag_workbench.experiments.safe_recovery_luna_v2.identities import LUNA_MODEL, SOL_MODEL
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.security.permissions import Principal
from scripts.run_combined_targeted_real_api_validation_v1 import (
    convert_verified,
    universal_chunks,
    version_candidates,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("retrieval_dry_run.json")
QUESTIONS = (
    ROOT
    / ".local/evaluation-input/fresh_unseen_eval_v1_1_20260820"
    / "fresh_unseen_questions_v1_1.json"
)
TOP_K = 15
OUTPUT_CAP = 700


def main() -> None:
    payload = json.loads(QUESTIONS.read_text())
    rows = []
    started = time.perf_counter()
    settings = Settings()
    with Session(create_engine(settings.database_url)) as session:
        runtime = build_runtime(session, settings, prompt_cache=True)
        for source in payload["cases"]:
            case_started = time.perf_counter()
            principal_data = source["principal"]
            principal = Principal(
                principal_data["principal_id"],
                principal_data["tenant_id"],
                frozenset(principal_data["permission_groups"]),
            )
            trace = retrieve_trace(runtime, source["question"], principal)
            if not trace["embedding_cache_hit"] or trace["embedding_external_calls"]:
                raise RuntimeError(f"NONZERO_API_RETRIEVAL_DRY_RUN:{source['case_id']}")
            top15 = trace["top20"][:TOP_K]
            plan = decompose_question(source["question"])
            chunks = universal_chunks(top15)
            target = TargetCase(
                source["case_id"],
                "FRESH",
                source["category"],
                source["question"],
                principal.principal_id,
                principal.tenant_id,
                tuple(principal.permission_groups),
            )
            candidates = version_candidates(session, top15, source["question"])
            resolution = (
                DeterministicVersionResolver().resolve(
                    source["question"], candidates, tenant_id=principal.tenant_id
                )
                if candidates
                else None
            )
            version_support = (
                extract_resolved_token_support(source["question"], resolution)
                if resolution
                else ()
            )
            deterministic = (
                convert_verified(version_support, top15)
                if version_support
                else deterministic_requirement_map(source["question"], chunks)
            )
            deterministic_complete = len(deterministic) == len(plan.requirements)
            version_sensitive = bool(candidates)
            version_resolved = bool(
                resolution and resolution.status == "VERSION_RESOLVED"
            )
            active_versions = [candidate for candidate in candidates if candidate.is_active]
            multiple_active = len(
                {candidate.document_version_id for candidate in active_versions}
            ) > 1 and not any(
                value in source["question"].casefold() for value in ("east", "west")
            )
            route = SelectiveRiskRouter().initial(
                RoutingFeatures(
                    security_precheck_requires_abstention=is_question_injection_v2(
                        source["question"]
                    ),
                    deterministic_support_complete=deterministic_complete,
                    conflicting_evidence=False,
                    version_sensitive=version_sensitive,
                    version_resolved=version_resolved,
                    multiple_active_versions=multiple_active,
                    version_metadata_complete=all(
                        candidate.is_active is not None for candidate in candidates
                    ),
                    top15_evidence_complete=deterministic_complete,
                    required_fact_count=len(plan.requirements),
                )
            )
            estimated_input = math.ceil(
                (
                    len(FROZEN_VERIFIER_PROMPT)
                    + len(json.dumps(plan.as_dict(), sort_keys=True))
                    + sum(len(item.get("text", "")) + 150 for item in top15)
                )
                / 3.5
            )
            rows.append(
                {
                    "case_id": source["case_id"],
                    "category": source["category"],
                    "query_embedding_cache_hit": True,
                    "embedding_external_calls": 0,
                    "dense_retrieval_success": bool(trace["dense_ids"]),
                    "bm25_success": bool(trace["bm25_ids"]),
                    "rrf_success": bool(trace["union"]),
                    "cross_encoder_success": bool(trace["top20"]),
                    "top15_success": 0 < len(top15) <= TOP_K,
                    "top15_pool_count": len(top15),
                    "dense_result_count": len(trace["dense_ids"]),
                    "bm25_result_count": len(trace["bm25_ids"]),
                    "rrf_union_count": len(trace["union"]),
                    "top15_chunk_ids": [item["chunk_id"] for item in top15],
                    "top15_document_ids": [item["document_id"] for item in top15],
                    "question_plan_hash": plan.question_plan_hash,
                    "requirement_count": len(plan.requirements),
                    "deterministic_support_count": len(deterministic),
                    "version_resolution": None if resolution is None else {
                        "status": resolution.status,
                        "reason": resolution.reason,
                        "selected_document_id": resolution.candidate.document_id
                        if resolution.candidate
                        else None,
                        "selected_version": resolution.candidate.version
                        if resolution.candidate
                        else None,
                        "selected_version_id": resolution.candidate.document_version_id
                        if resolution.candidate
                        else None,
                    },
                    "route_projection": asdict(route),
                    "estimated_model_tokens": {
                        "input": estimated_input,
                        "output_cap": OUTPUT_CAP,
                    },
                    "latency_ms": (time.perf_counter() - case_started) * 1000,
                }
            )
    failures = [
        row["case_id"]
        for row in rows
        if not all(
            row[key]
            for key in (
                "dense_retrieval_success",
                "bm25_success",
                "rrf_success",
                "cross_encoder_success",
                "top15_success",
            )
        )
    ]
    routes = Counter(row["route_projection"]["route"] for row in rows)
    luna_rows = [row for row in rows if row["route_projection"]["route"] == "LUNA"]
    direct_sol_rows = [row for row in rows if row["route_projection"]["route"] == "SOL"]
    sol_reserve = max(4, math.ceil(len(luna_rows) * 0.20))
    luna_input = sum(row["estimated_model_tokens"]["input"] for row in luna_rows)
    luna_output = len(luna_rows) * OUTPUT_CAP
    direct_sol_input = sum(
        row["estimated_model_tokens"]["input"] for row in direct_sol_rows
    )
    direct_sol_output = len(direct_sol_rows) * OUTPUT_CAP
    max_luna_input = max(
        (row["estimated_model_tokens"]["input"] for row in luna_rows), default=2500
    )
    projected_luna = estimate_cost_usd(
        model=LUNA_MODEL, input_tokens=luna_input, output_tokens=luna_output
    )
    projected_direct_sol = estimate_cost_usd(
        model=SOL_MODEL,
        input_tokens=direct_sol_input,
        output_tokens=direct_sol_output,
    )
    projected_sol_reserve = estimate_cost_usd(
        model=SOL_MODEL,
        input_tokens=sol_reserve * max_luna_input,
        output_tokens=sol_reserve * OUTPUT_CAP,
    )
    embedding = json.loads(
        Path(__file__).with_name("embedding_warmup_results.json").read_text()
    )
    projected_total = (
        embedding["cost_usd"]
        + projected_luna
        + projected_direct_sol
        + projected_sol_reserve
    )
    report = {
        "experiment_id": "FRESH_UNSEEN_EVALUATION_V1_2",
        "stage": "RETRIEVAL_DRY_RUN",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "openai_calls": 0,
        "case_count": len(rows),
        "query_embedding_coverage": len(rows),
        "dense_retrieval_success_count": sum(row["dense_retrieval_success"] for row in rows),
        "bm25_success_count": sum(row["bm25_success"] for row in rows),
        "rrf_success_count": sum(row["rrf_success"] for row in rows),
        "cross_encoder_success_count": sum(row["cross_encoder_success"] for row in rows),
        "top15_success_count": sum(row["top15_success"] for row in rows),
        "route_projections": dict(routes),
        "deterministic_route_count": routes.get("DETERMINISTIC", 0),
        "expected_luna_count": routes.get("LUNA", 0),
        "direct_sol_count": routes.get("SOL", 0),
        "safe_abstain_count": routes.get("SAFE_ABSTAIN", 0),
        "possible_sol_escalation_reserve": sol_reserve,
        "cost_projection": {
            "embedding_actual_usd": embedding["cost_usd"],
            "expected_luna_usd": projected_luna,
            "direct_sol_usd": projected_direct_sol,
            "sol_escalation_reserve_usd": projected_sol_reserve,
            "projected_total_usd": projected_total,
            "hard_cap_usd": 0.50,
        },
        "failed_case_ids": failures,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "gate_passed": not failures,
        "cases": rows,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in report if key != "cases"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
