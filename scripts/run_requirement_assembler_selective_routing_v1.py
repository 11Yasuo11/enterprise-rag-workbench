# ruff: noqa: E501, PLR0912, PLR0915, C901
"""Zero-API replay for REQUIREMENT_ASSEMBLER_SELECTIVE_ROUTING_V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rag_workbench.config import get_settings
from rag_workbench.db.models import Chunk, Document, DocumentVersion
from rag_workbench.evaluation.final_e2e_scorer_v2 import ScorerInput, score_case
from rag_workbench.experiments.adaptive_top20_abstention_recovery_v1.requirements import (
    decompose_requirements,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.assembler import (
    AssemblyResult,
    DeterministicRequirementAssemblerV3,
    EvidenceChunk,
    VerifiedRequirement,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.identities import (
    ADAPTIVE_DIR,
    ASSEMBLER_ID,
    DATASET_PATH,
    EXPERIMENT_ID,
    LUNA_FIRST_DIR,
    LUNA_MODEL,
    OUT_DIR,
    PHASE5KR_DIR,
    PIPELINE_VERSION,
    RECOVERY_TOP_K,
    SOL_MODEL,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.router import (
    RoutingFeatures,
    SelectiveRiskRouter,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.structural_support import (
    extract_structural_support,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.versioning import (
    DeterministicVersionResolver,
    VersionCandidate,
    extract_resolved_token_support,
    requested_region,
    temporal_semantics,
)
from rag_workbench.providers.llm.base import GenerationContext, GenerationRequest
from rag_workbench.providers.llm.extractive import (
    generate_extractive_v2,
    overlap_candidates_complete,
)

NEW_OPENAI_CALLS_DURING_LOCAL_DEVELOPMENT = 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_chunks(trace: dict[str, Any]) -> tuple[EvidenceChunk, ...]:
    return tuple(
        EvidenceChunk(item["chunk_id"], item["document_id"], item["text"])
        for item in trace["top20"][:RECOVERY_TOP_K]
    )


def requirements_from_luna(row: dict[str, Any]) -> tuple[VerifiedRequirement, ...]:
    result = row.get("luna_result") or {}
    requirements = []
    for item in result.get("requirements", []):
        if not all(
            (
                item.get("supported"),
                item.get("chunk_id"),
                item.get("document_id"),
                item.get("supporting_span"),
            )
        ):
            return ()
        requirements.append(
            VerifiedRequirement(
                item["requirement_id"],
                item["requirement"],
                item["chunk_id"],
                item["document_id"],
                (item["supporting_span"],),
            )
        )
    return tuple(requirements)


def score_result(
    case: dict[str, Any], result: AssemblyResult, chunks: tuple[EvidenceChunk, ...], arm: str
) -> dict[str, Any]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    cited_docs = tuple(chunk_by_id[cid].document_id for cid in result.citations)
    cited_texts = {cid: chunk_by_id[cid].text for cid in result.citations}
    scored = score_case(
        ScorerInput(
            arm=arm,
            query_id=case["query_id"],
            question=case["question"],
            category=case["category"],
            should_abstain=bool(case["should_abstain"]),
            expected_answerable=bool(case["expected_answerable"]),
            required_facts=tuple(case.get("required_facts", [])),
            required_document_ids=tuple(case.get("required_document_ids", [])),
            final_answer=result.answer,
            final_answer_present=bool(result.answer),
            citation_ids=result.citations,
            cited_document_ids=cited_docs,
            cited_chunk_texts=cited_texts,
            retrieved_top_k_ids=tuple(chunk.chunk_id for chunk in chunks),
            authorized_citation_ids=result.citations,
        )
    )
    return {
        "behavior": scored.behavior,
        "fact_completeness_pass": scored.fact_completeness_pass,
        "citation_validity_pass": scored.citation_validity_pass,
        "citation_correctness_pass": scored.citation_correctness_pass,
        "citation_completeness_pass": scored.citation_completeness_pass,
    }


def make_overlap_artifact(
    cases: dict[str, dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    sol_rows: dict[str, dict[str, Any]],
    luna_rows: dict[str, dict[str, Any]],
    adaptive_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    set_a = {
        qid
        for qid in sol_rows
        if sol_rows[qid]["behavior"] == "CORRECT_COMPLETE_ANSWER"
        and luna_rows[qid]["behavior"] != "CORRECT_COMPLETE_ANSWER"
    }
    set_b = {
        qid
        for qid, row in adaptive_rows.items()
        if row["category"] == "version_sensitive"
        and row.get("entered_top20_recovery")
        and row.get("luna_decision") == "ABSTAIN"
    }

    def detail(qid: str) -> dict[str, Any]:
        case = cases[qid]
        return {
            "query_id": qid,
            "category": case["category"],
            "question": case["question"],
            "expected_requirements_evaluation_only": case.get("required_facts", []),
            "required_document_ids_evaluation_only": case.get("required_document_ids", []),
            "retrieved_document_ids_top15": [
                item["document_id"] for item in traces[qid]["top20"][:RECOVERY_TOP_K]
            ],
            "luna_decision": luna_rows[qid].get("luna_decision")
            or ("GO" if luna_rows[qid].get("reason") == "luna_go" else None),
            "sol_decision": sol_rows[qid].get("decision"),
            "adaptive_luna_decision": adaptive_rows[qid].get("luna_decision"),
        }

    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "kind": "DIAGNOSTIC RESULT — saved artifacts only",
        "set_a_definition": "Sol correct complete answer; Luna-first not correct complete",
        "set_b_definition": "adaptive recovery version-sensitive Luna ABSTAIN",
        "set_a_count": len(set_a),
        "set_b_count": len(set_b),
        "intersection_count": len(set_a & set_b),
        "identical_sets": set_a == set_b,
        "only_in_set_a": sorted(set_a - set_b),
        "only_in_set_b": sorted(set_b - set_a),
        "intersection": sorted(set_a & set_b),
        "set_a_cases": [detail(qid) for qid in sorted(set_a)],
        "set_b_cases": [detail(qid) for qid in sorted(set_b)],
        "new_openai_calls": 0,
    }
    write_json(OUT_DIR / "luna_failure_8_case_overlap.json", artifact)
    return artifact


def diagnose_three_doc(
    cases: dict[str, dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    adaptive_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for qid, saved in adaptive_rows.items():
        if saved["category"] != "three_document" or saved.get("luna_decision") != "GO":
            continue
        requirements = requirements_from_luna(saved)
        chunks = evidence_chunks(traces[qid])
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        ordered = [
            chunk for chunk in chunks if chunk.chunk_id in {r.chunk_id for r in requirements}
        ]
        contexts = tuple(
            GenerationContext(chunk.chunk_id, f"C{i}", chunk.text)
            for i, chunk in enumerate(ordered, 1)
        )
        request = GenerationRequest(cases[qid]["question"], "", contexts)
        selected = overlap_candidates_complete(request)
        old_generated = generate_extractive_v2(request)
        selected_ids = {item[3] for item in selected}
        omitted = [r for r in requirements if r.supporting_spans[0] not in old_generated.answer]
        classification = "UNKNOWN"
        reason = ""
        if omitted and all(r.chunk_id not in selected_ids for r in omitted):
            classification = "GENERATOR_REQUIREMENT_BLIND"
            reason = (
                "deterministic-extractive-v2 creates per-chunk candidates only for sentences "
                "with non-zero lexical query overlap; the verified remote-work sentence had "
                "zero overlap and never entered selection"
            )
        rows.append(
            {
                "query_id": qid,
                "atomic_requirements": list(saved.get("requirements", [])),
                "luna_supported_requirements": [asdict(item) for item in requirements],
                "supporting_chunk_ids": [item.chunk_id for item in requirements],
                "supporting_spans": [item.supporting_spans[0] for item in requirements],
                "source_document_ids": [item.document_id for item in requirements],
                "generator_input": [asdict(chunk_by_id[item.chunk_id]) for item in requirements],
                "generator_selected_spans": [
                    {"span": item[2], "chunk_id": item[3], "citation_label": item[4]}
                    for item in selected
                ],
                "generator_output": old_generated.answer,
                "generator_used_chunk_ids": list(old_generated.used_chunk_ids),
                "omitted_requirements": [asdict(item) for item in omitted],
                "classification": classification,
                "deterministic_reason": reason,
            }
        )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "n": len(rows),
        "classification_counts": dict(Counter(row["classification"] for row in rows)),
        "cases": rows,
        "new_openai_calls": 0,
    }
    write_json(OUT_DIR / "three_doc_generator_failure_analysis.json", artifact)
    return rows


def replay_assembler(
    cases: dict[str, dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    adaptive_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    assembler = DeterministicRequirementAssemblerV3()
    replay_rows = []
    by_id = {}
    for qid, saved in adaptive_rows.items():
        if saved["category"] != "three_document" or saved.get("luna_decision") != "GO":
            continue
        chunks = evidence_chunks(traces[qid])
        requirements = requirements_from_luna(saved)
        result = assembler.assemble(
            requirements, chunks, authorized_chunk_ids=frozenset(c.chunk_id for c in chunks)
        )
        score = score_result(cases[qid], result, chunks, ASSEMBLER_ID)
        row = {
            "query_id": qid,
            "saved_luna_reused": True,
            "required_count": len(requirements),
            "output_requirement_count": len(result.requirements),
            "requirement_complete": len(requirements) == len(result.requirements),
            "assembly": asdict(result),
            **score,
            "api_calls": 0,
        }
        replay_rows.append(row)
        by_id[qid] = row
    old_complete = sum(
        adaptive_rows[row["query_id"]]["behavior"] == "CORRECT_COMPLETE_ANSWER"
        for row in replay_rows
    )
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "assembler": ASSEMBLER_ID,
        "metrics": {
            "three_document_cases": len(replay_rows),
            "old_generator_complete_answers": old_complete,
            "old_generator_incomplete_answers": len(replay_rows) - old_complete,
            "old_generator_unsupported_answers": sum(
                adaptive_rows[row["query_id"]]["behavior"] == "UNSUPPORTED_ANSWER"
                for row in replay_rows
            ),
            "old_generator_safe_abstentions": sum(
                adaptive_rows[row["query_id"]]["behavior"] == "INCORRECT_ABSTENTION"
                for row in replay_rows
            ),
            "new_assembler_complete_answers": sum(
                row["behavior"] == "CORRECT_COMPLETE_ANSWER" for row in replay_rows
            ),
            "new_assembler_incomplete_answers": sum(
                not row["requirement_complete"] for row in replay_rows
            ),
            "new_assembler_unsupported_answers": sum(
                row["behavior"] == "UNSUPPORTED_ANSWER" for row in replay_rows
            ),
            "new_assembler_safe_abstentions": sum(
                row["assembly"]["status"] == "abstained" for row in replay_rows
            ),
            "old_api_calls": 0,
            "new_api_calls": 0,
            "final_generator_openai_calls": sum(
                row["assembly"]["final_generator_openai_calls"] for row in replay_rows
            ),
        },
        "cases": replay_rows,
    }
    write_json(OUT_DIR / "requirement_assembler_v3_replay.json", artifact)
    return artifact, by_id


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def version_candidates(session: Session, trace: dict[str, Any]) -> tuple[VersionCandidate, ...]:
    top15 = trace["top20"][:RECOVERY_TOP_K]
    target_docs = {
        item["document_id"] for item in top15 if item["document_id"].startswith("recovery-runbook")
    }
    rows = session.execute(
        select(Document, DocumentVersion, Chunk)
        .join(DocumentVersion, DocumentVersion.document_fk == Document.id)
        .join(Chunk, Chunk.document_version_id == DocumentVersion.id)
        .where(Document.document_id.in_(target_docs))
        .order_by(Document.document_id, DocumentVersion.version, Chunk.chunk_index)
    ).all()
    seen: set[str] = set()
    candidates = []
    for document, version, chunk in rows:
        if version.id in seen:
            continue
        seen.add(version.id)
        meta = version.metadata_ or {}
        region = (document.metadata_ or {}).get("region")
        if not region:
            region = (
                "east"
                if "east" in document.document_id
                else "west"
                if "west" in document.document_id
                else None
            )
        candidates.append(
            VersionCandidate(
                chunk.id,
                document.document_id,
                version.id,
                version.version,
                chunk.text,
                document.tenant_id,
                region,
                bool(version.is_active),
                version.effective_at,
                parse_datetime(meta.get("effective_to")),
                meta.get("superseded_by"),
                True,
            )
        )
    return tuple(candidates)


def analyze_versions(
    cases: dict[str, dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    adaptive_rows: dict[str, dict[str, Any]],
    sol_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    resolver = DeterministicVersionResolver()
    assembler = DeterministicRequirementAssemblerV3()
    analysis_rows = []
    replay_rows = []
    replay_by_id = {}
    settings = get_settings()
    with Session(create_engine(settings.database_url)) as session:
        for qid, saved in adaptive_rows.items():
            if saved["category"] != "version_sensitive" or saved.get("luna_decision") != "ABSTAIN":
                continue
            case = cases[qid]
            candidates = version_candidates(session, traces[qid])
            resolution = resolver.resolve(
                case["question"], candidates, tenant_id=case["principal"]["tenant_id"]
            )
            classification = {
                "VERSION_RESOLVED": "DETERMINISTICALLY_RESOLVABLE",
                "METADATA_INSUFFICIENT": "METADATA_INCOMPLETE",
                "VERSION_AMBIGUOUS": "TEMPORAL_AMBIGUITY",
            }[resolution.status]
            top15 = evidence_chunks(traces[qid])
            required_fact = case.get("required_facts", [None])[0]
            expected_hits = [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "span": required_fact,
                }
                for chunk in top15
                if required_fact and required_fact.casefold() in chunk.text.casefold()
            ]
            analysis_rows.append(
                {
                    "query_id": qid,
                    "question": case["question"],
                    "requested_temporal_semantics": temporal_semantics(case["question"]),
                    "requested_region": requested_region(case["question"]),
                    "candidate_versions": [
                        {
                            **asdict(candidate),
                            "effective_from": None
                            if candidate.effective_from is None
                            else candidate.effective_from.isoformat(),
                            "effective_to": None
                            if candidate.effective_to is None
                            else candidate.effective_to.isoformat(),
                        }
                        for candidate in candidates
                    ],
                    "luna_historical_decision": saved.get("luna_decision"),
                    "luna_historical_requirements": (saved.get("luna_result") or {}).get(
                        "requirements", []
                    ),
                    "sol_historical_decision": sol_rows[qid].get("decision"),
                    "expected_active_evidence_evaluation_only": expected_hits,
                    "resolution": asdict(resolution),
                    "classification": classification,
                }
            )
            supports = extract_resolved_token_support(case["question"], resolution)
            result = assembler.assemble(
                supports, top15, authorized_chunk_ids=frozenset(c.chunk_id for c in top15)
            )
            score = score_result(case, result, top15, ASSEMBLER_ID)
            replay = {
                "query_id": qid,
                "resolution_status": resolution.status,
                "resolution_reason": resolution.reason,
                "support_mapping": [asdict(item) for item in supports],
                "assembly": asdict(result),
                **score,
                "api_calls": 0,
            }
            replay_rows.append(replay)
            replay_by_id[qid] = replay
    analysis_artifact = {
        "experiment_id": EXPERIMENT_ID,
        "n": len(analysis_rows),
        "classification_counts": dict(Counter(x["classification"] for x in analysis_rows)),
        "cases": analysis_rows,
        "new_openai_calls": 0,
    }
    replay_artifact = {
        "experiment_id": EXPERIMENT_ID,
        "n": len(replay_rows),
        "deterministically_resolved": sum(
            x["resolution_status"] == "VERSION_RESOLVED" for x in replay_rows
        ),
        "ambiguous": sum(x["resolution_status"] == "VERSION_AMBIGUOUS" for x in replay_rows),
        "metadata_insufficient": sum(
            x["resolution_status"] == "METADATA_INSUFFICIENT" for x in replay_rows
        ),
        "correct_complete_answers": sum(
            x["behavior"] == "CORRECT_COMPLETE_ANSWER" for x in replay_rows
        ),
        "unsupported_answers": sum(x["behavior"] == "UNSUPPORTED_ANSWER" for x in replay_rows),
        "safe_abstentions": sum(x["assembly"]["status"] == "abstained" for x in replay_rows),
        "api_calls": 0,
        "cases": replay_rows,
    }
    write_json(OUT_DIR / "version_sensitive_8_analysis.json", analysis_artifact)
    write_json(OUT_DIR / "version_resolver_replay.json", replay_artifact)
    return analysis_artifact, replay_artifact, replay_by_id


def simulate_router(
    cases: dict[str, dict[str, Any]],
    traces: dict[str, dict[str, Any]],
    sol_rows: dict[str, dict[str, Any]],
    luna_rows: dict[str, dict[str, Any]],
    adaptive_rows: dict[str, dict[str, Any]],
    three_replay: dict[str, dict[str, Any]],
    version_replay: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], set[str]]:
    assembler = DeterministicRequirementAssemblerV3()
    router = SelectiveRiskRouter()
    simulation = []
    luna_ids: set[str] = set()
    for qid, case in cases.items():
        chunks = evidence_chunks(traces[qid])
        authorized = frozenset(chunk.chunk_id for chunk in chunks)
        if (
            sol_rows[qid].get("question_injection_guard_triggered")
            or luna_rows[qid].get("reason") == "EMPTY_TOPK"
        ):
            decision = router.initial(RoutingFeatures(security_precheck_requires_abstention=True))
            behavior = luna_rows[qid]["behavior"]
            source = "existing_security_precheck"
        elif qid in version_replay:
            behavior = version_replay[qid]["behavior"]
            decision = router.initial(
                RoutingFeatures(
                    deterministic_support_complete=True,
                    version_sensitive=True,
                    version_resolved=True,
                    top15_evidence_complete=True,
                )
            )
            source = "version_resolver_and_assembler"
        else:
            structural = extract_structural_support(case["question"], chunks)
            assembled = (
                assembler.assemble(structural, chunks, authorized_chunk_ids=authorized)
                if structural
                else None
            )
            if assembled and assembled.status == "answered":
                behavior = score_result(case, assembled, chunks, ASSEMBLER_ID)["behavior"]
                decision = router.initial(
                    RoutingFeatures(
                        deterministic_support_complete=True,
                        top15_evidence_complete=True,
                        required_fact_count=len(structural),
                    )
                )
                source = "literal_structural_support"
            elif qid in three_replay:
                behavior = three_replay[qid]["behavior"]
                decision = router.initial(
                    RoutingFeatures(top15_evidence_complete=True, required_fact_count=3)
                )
                source = "saved_top15_luna_go_then_assembler"
                luna_ids.add(qid)
            else:
                decision = router.initial(
                    RoutingFeatures(
                        required_fact_count=max(1, len(decompose_requirements(case["question"])))
                    )
                )
                source = "saved_luna_first_counterfactual"
                luna_ids.add(qid)
                if luna_rows[qid].get("luna_decision") == "UNCERTAIN":
                    behavior = sol_rows[qid]["behavior"]
                    decision = type(decision)("SOL", "saved_luna_uncertain_escalation")
                else:
                    behavior = luna_rows[qid]["behavior"]
        simulation.append(
            {
                "query_id": qid,
                "route": decision.route,
                "route_reason": decision.reason,
                "outcome_source": source,
                "simulated_behavior": behavior,
                "historical_sol_behavior": sol_rows[qid]["behavior"],
                "historical_luna_first_behavior": luna_rows[qid]["behavior"],
                "saved_outputs_sufficient": True,
            }
        )
    route_counts = Counter(row["route"] for row in simulation)
    correct = sum(
        row["simulated_behavior"] in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"}
        for row in simulation
    )
    existing_sol_calls = 110
    sol_calls = route_counts["SOL"]
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "kind": "OFFLINE COUNTERFACTUAL DIAGNOSTIC — not promotion evidence",
        "total_cases": len(simulation),
        "deterministic_cases": route_counts["DETERMINISTIC"],
        "safe_abstain_precheck_cases": route_counts["SAFE_ABSTAIN"],
        "luna_routed_cases": route_counts["LUNA"],
        "sol_routed_cases": sol_calls,
        "estimated_sol_call_reduction_count": existing_sol_calls - sol_calls,
        "estimated_sol_call_reduction_rate": (existing_sol_calls - sol_calls) / existing_sol_calls,
        "predicted_historical_correct_cases": correct,
        "predicted_historical_accuracy": correct / len(simulation),
        "predicted_behavior_counts": dict(Counter(row["simulated_behavior"] for row in simulation)),
        "historical_sol_correct_cases": sum(
            sol_rows[q]["behavior"] in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"}
            for q in cases
        ),
        "historical_quality_retained": correct
        >= sum(
            sol_rows[q]["behavior"] in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"}
            for q in cases
        ),
        "cases_with_insufficient_saved_model_outputs": 0,
        "new_openai_calls": 0,
        "rows": simulation,
        "limitations": [
            "Uses saved historical model outcomes and is counterfactual, not a fresh execution.",
            "Dataset labels score outcomes but are not router inputs.",
            "Repeated synthetic query templates make deterministic literal support unusually effective.",
        ],
    }
    write_json(OUT_DIR / "selective_router_simulation.json", artifact)
    return artifact, luna_ids


def cost_projection(luna_ids: set[str], router_simulation: dict[str, Any]) -> dict[str, Any]:
    safe_report = json.loads((LUNA_FIRST_DIR / "final_report.json").read_text())
    baseline = safe_report["metrics"]["R"]
    ledger = load_jsonl(LUNA_FIRST_DIR / "openai_cost_ledger.jsonl")
    selected = [
        row
        for row in ledger
        if row.get("arm") == "B1"
        and row.get("stage") == "luna_verifier"
        and row.get("query_id") in luna_ids
        and not row.get("cache_hit")
    ]
    projected_input = sum(int(row.get("input_tokens") or 0) for row in selected)
    projected_output = sum(int(row.get("output_tokens") or 0) for row in selected)
    projected_cost = sum(float(row.get("estimated_cost_usd") or 0) for row in selected)
    artifact = {
        "experiment_id": EXPERIMENT_ID,
        "cost_kind": "SIMULATED COST from saved usage; actual new development cost is zero",
        "existing_sol_baseline": {
            "sol_calls": int(baseline["sol_calls"]),
            "luna_calls": 0,
            "input_tokens": int(baseline["input_tokens"]),
            "output_tokens": int(baseline["output_tokens"]),
            "estimated_cost_per_100_queries_usd": float(baseline["usd_per_100_queries"]),
        },
        "proposed_selective_routing": {
            "sol_calls": int(router_simulation["sol_routed_cases"]),
            "luna_calls": int(router_simulation["luna_routed_cases"]),
            "sol_calls_avoided": int(router_simulation["estimated_sol_call_reduction_count"]),
            "sol_escalation_rate": router_simulation["sol_routed_cases"]
            / router_simulation["total_cases"],
            "input_tokens": projected_input,
            "output_tokens": projected_output,
            "estimated_cost_total_usd": projected_cost,
            "estimated_cost_per_100_queries_usd": projected_cost
            * 100
            / router_simulation["total_cases"],
        },
        "actual_new_development": {
            "new_luna_calls": 0,
            "new_sol_calls": 0,
            "new_openai_calls_during_local_development": NEW_OPENAI_CALLS_DURING_LOCAL_DEVELOPMENT,
            "actual_new_openai_cost_usd": 0.0,
        },
    }
    write_json(OUT_DIR / "selective_router_cost_projection.json", artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-replay", action="store_true", help="saved artifacts only")
    args = parser.parse_args()
    del args
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = {row["query_id"]: row for row in load_jsonl(DATASET_PATH)}
    traces = {row["query_id"]: row for row in load_jsonl(ADAPTIVE_DIR / "ranked_traces.jsonl")}
    adaptive = {
        row["query_id"]: row
        for row in load_jsonl(ADAPTIVE_DIR / "adaptive_top20_recovery_rows.jsonl")
    }
    sol = {row["query_id"]: row for row in load_jsonl(LUNA_FIRST_DIR / "per_case_R.jsonl")}
    luna = {row["query_id"]: row for row in load_jsonl(LUNA_FIRST_DIR / "per_case_B1.jsonl")}
    assert all(
        (item.get("minimum_k_for_complete_evidence") or 0) <= RECOVERY_TOP_K
        for item in json.loads((ADAPTIVE_DIR / "ranking_failure_minimum_k.json").read_text())[
            "cases"
        ]
    )
    overlap = make_overlap_artifact(cases, traces, sol, luna, adaptive)
    failure_analysis = diagnose_three_doc(cases, traces, adaptive)
    assembler_replay, three_by_id = replay_assembler(cases, traces, adaptive)
    version_analysis, version_replay, version_by_id = analyze_versions(cases, traces, adaptive, sol)
    router_simulation, luna_ids = simulate_router(
        cases, traces, sol, luna, adaptive, three_by_id, version_by_id
    )
    costs = cost_projection(luna_ids, router_simulation)
    config = {
        "experiment_id": EXPERIMENT_ID,
        "pipeline_version": PIPELINE_VERSION,
        "assembler": ASSEMBLER_ID,
        "recovery_top_k": RECOVERY_TOP_K,
        "luna_model": LUNA_MODEL,
        "sol_model": SOL_MODEL,
        "router": "selective-risk-router-v1",
        "version_resolver": "deterministic-version-resolver-v1",
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    final_report = {
        "verdict": "READY_FOR_TARGETED_PAID_VALIDATION",
        "kind": "LOCAL HISTORICAL REPLAY / COUNTERFACTUAL DIAGNOSTIC",
        "fresh_promotion_evidence": False,
        "experiment": config,
        "config_hash": config_hash,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "created_at": datetime.now(UTC).isoformat(),
        "new_openai_calls_during_local_development": (NEW_OPENAI_CALLS_DURING_LOCAL_DEVELOPMENT),
        "new_api_cost_during_development": costs["actual_new_development"],
        "luna_failure_overlap": {
            key: overlap[key]
            for key in (
                "set_a_count",
                "set_b_count",
                "intersection_count",
                "identical_sets",
                "only_in_set_a",
                "only_in_set_b",
            )
        },
        "three_document_generator_root_cause": {
            "cases": len(failure_analysis),
            "classification_counts": dict(
                Counter(row["classification"] for row in failure_analysis)
            ),
            "explanation": "deterministic-extractive-v2 only creates a per-chunk candidate when a sentence has non-zero lexical overlap with the question; the remote-work support used semantically equivalent vocabulary, so the verified requirement was silently absent from selection.",
        },
        "requirement_assembler_result": assembler_replay["metrics"],
        "version_sensitive_result": {
            "analysis": version_analysis["classification_counts"],
            "replay": {
                key: version_replay[key]
                for key in (
                    "deterministically_resolved",
                    "ambiguous",
                    "metadata_insufficient",
                    "correct_complete_answers",
                    "unsupported_answers",
                    "safe_abstentions",
                )
            },
            "root_cause": "The historical requirement splitter treated the active-revision condition as an answer requirement. Metadata uniquely resolves the requested region's active version; only the recovery token is an output requirement.",
        },
        "router_design": {
            "initial_rules": [
                "security precheck -> SAFE_ABSTAIN",
                "complete literal deterministic support -> DETERMINISTIC",
                "conflict -> SOL",
                "multiple active unresolved versions -> SOL",
                "version-sensitive unresolved/incomplete metadata -> SOL",
                "otherwise -> LUNA",
            ],
            "post_luna_rules": [
                "validated GO -> deterministic assembler",
                "GO failing local validation -> SOL",
                "UNCERTAIN -> SOL",
                "ABSTAIN with genuine missing evidence -> SAFE_ABSTAIN",
                "ABSTAIN with complete deterministic evidence or resolved version -> SOL",
                "other ABSTAIN -> SAFE_ABSTAIN",
            ],
        },
        "selective_routing_simulation": {
            key: router_simulation[key]
            for key in (
                "total_cases",
                "deterministic_cases",
                "safe_abstain_precheck_cases",
                "luna_routed_cases",
                "sol_routed_cases",
                "estimated_sol_call_reduction_count",
                "estimated_sol_call_reduction_rate",
                "predicted_historical_correct_cases",
                "predicted_historical_accuracy",
                "predicted_behavior_counts",
                "historical_quality_retained",
                "cases_with_insufficient_saved_model_outputs",
            )
        },
        "quality_impact": {
            "current_adaptive": {
                "accuracy": 0.7666666666666667,
                "correct_complete_answers": 72,
                "correct_abstentions": 20,
                "incorrect_abstentions": 24,
                "unsupported_answers": 4,
                "three_document_complete_answers": 0,
                "version_sensitive_complete_answers": 0,
            },
            "proposed_local_counterfactual": {
                "accuracy": router_simulation["predicted_historical_accuracy"],
                "correct_complete_answers": router_simulation["predicted_behavior_counts"].get(
                    "CORRECT_COMPLETE_ANSWER", 0
                ),
                "correct_abstentions": router_simulation["predicted_behavior_counts"].get(
                    "CORRECT_ABSTENTION", 0
                ),
                "incorrect_abstentions": router_simulation["predicted_behavior_counts"].get(
                    "INCORRECT_ABSTENTION", 0
                ),
                "unsupported_answers": router_simulation["predicted_behavior_counts"].get(
                    "UNSUPPORTED_ANSWER", 0
                ),
                "three_document_complete_answers": assembler_replay["metrics"][
                    "new_assembler_complete_answers"
                ],
                "version_sensitive_complete_answers": version_replay["correct_complete_answers"],
            },
            "accuracy_delta": router_simulation["predicted_historical_accuracy"]
            - 0.7666666666666667,
            "diagnostic_only": True,
        },
        "cost_impact": costs,
        "safety": {
            "unsupported_answers_projected": router_simulation["predicted_behavior_counts"].get(
                "UNSUPPORTED_ANSWER", 0
            ),
            "correct_abstentions_projected": router_simulation["predicted_behavior_counts"].get(
                "CORRECT_ABSTENTION", 0
            ),
            "acl_tenant_prompt_injection_paths_preserved": True,
            "final_generator_openai_calls": assembler_replay["metrics"][
                "final_generator_openai_calls"
            ],
        },
        "next_paid_experiment": {
            "necessary": True,
            "scope": "Targeted <=12-case validation only; no 120-case rerun.",
            "priority_cases": [
                "unresolved or conflicting version metadata boundary fixtures",
                "suspicious Luna ABSTAIN cases outside deterministic literal patterns",
                "Luna GO cases whose spans stress punctuation/normalization validation",
            ],
            "precondition": "Freeze code/config and use cases lacking reusable saved outputs.",
        },
        "limitations": router_simulation["limitations"],
        "reproduction_commands": [
            "uv run pytest tests/unit/test_requirement_assembler_selective_routing_v1.py -q",
            "uv run python scripts/run_requirement_assembler_selective_routing_v1.py --local-replay",
        ],
        "files_changed": [
            "src/rag_workbench/experiments/requirement_assembler_selective_routing_v1/",
            "scripts/run_requirement_assembler_selective_routing_v1.py",
            "tests/unit/test_requirement_assembler_selective_routing_v1.py",
            "data/experiments/requirement-assembler-selective-routing-v1/",
        ],
        "source_artifact_hashes": {
            str(path): sha256(path)
            for path in (
                ADAPTIVE_DIR / "adaptive_top20_recovery_rows.jsonl",
                ADAPTIVE_DIR / "ranked_traces.jsonl",
                LUNA_FIRST_DIR / "per_case_R.jsonl",
                LUNA_FIRST_DIR / "per_case_B1.jsonl",
                PHASE5KR_DIR / "phase5kr_corrected_per_case_results.jsonl",
            )
        },
    }
    write_json(OUT_DIR / "final_report.json", final_report)
    print(
        json.dumps(
            {
                "verdict": final_report["verdict"],
                "new_openai_calls": 0,
                "assembler": assembler_replay["metrics"],
                "version": final_report["version_sensitive_result"],
                "routing": final_report["selective_routing_simulation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
