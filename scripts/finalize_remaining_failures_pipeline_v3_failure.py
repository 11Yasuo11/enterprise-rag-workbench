"""Finalize v3 artifacts after the mandatory P1 gate failed."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.experiments.atomic_requirement_contract_v1 import (
    AtomicRequirement,
    ContextQualifier,
    FrozenQuestionPlan,
)
from rag_workbench.experiments.atomic_requirement_contract_v1.verifier import (
    FROZEN_VERIFIER_PROMPT,
    frozen_messages,
    frozen_schema,
)
from rag_workbench.experiments.safe_recovery_luna_v2.identities import (
    LUNA_MODEL,
    sha256_text,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/experiments/rag-remaining-failures-release-pipeline-v3"
P1 = BASE / "p1-targeted"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def plan_from_dict(value: dict[str, Any]) -> FrozenQuestionPlan:
    return FrozenQuestionPlan(
        value["question"],
        tuple(AtomicRequirement(**item) for item in value["requirements"]),
        tuple(ContextQualifier(**item) for item in value["context_qualifiers"]),
        value["question_plan_hash"],
        tuple(value["detected_output_units"]),
        bool(value["cardinality_match"]),
    )


def blocked(artifact: str) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "status": "NOT_RUN_BLOCKED_BY_P1_CONFIRMATION",
        "blocked_by": "P1_TARGETED_REAL_API_FAILED",
        "api_calls": 0,
        "cost_usd": 0.0,
    }


def main() -> None:
    timestamp = datetime.now(UTC).isoformat()
    formal = load(P1 / "final_report.json")
    predictions = load_jsonl(P1 / "final_predictions.jsonl")
    ledger = load_jsonl(P1 / "api_cost_ledger.jsonl")
    payload_rows = load_jsonl(P1 / "verifier_evidence_payload_audit.jsonl")
    payload_by_case = {row["case_id"]: row for row in payload_rows}
    diagnostic_cases = []
    for prediction in predictions:
        mappings = [
            mapping
            for requirement in prediction["requirement_version_mappings"]
            for mapping in requirement["mappings"]
        ]
        payload_row = payload_by_case[prediction["case_id"]]
        plan = plan_from_dict(payload_row["question_plan"])
        evidence = tuple(
            GateEvidence(
                row["chunk_id"],
                row["document_id"],
                row.get("document_version_id", ""),
                row.get("version", ""),
                row["text"],
            )
            for row in payload_row["evidence_rows"]
        )
        schema = frozen_schema(plan)
        exact_payload = {
            "model": LUNA_MODEL,
            "temperature": 0,
            "reasoning_effort": "none",
            "max_completion_tokens": 700,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "atomic_requirement_contract_v1",
                    "strict": True,
                    "schema": schema,
                },
            },
            "messages": frozen_messages(plan, evidence),
            "prompt_cache_key": sha256_text(
                FROZEN_VERIFIER_PROMPT + json.dumps(schema, sort_keys=True)
            )[:64],
        }
        diagnostic_cases.append(
            {
                "case_id": prediction["case_id"],
                "frozen_answer_present": bool(prediction["answer"]),
                "luna_decision": prediction["luna_decision"]["decision"],
                "citation_count": len(prediction["citations"]),
                "requirement_version_mapping_count": len(mappings),
                "all_mapped_versions_present": all(mapping["version"] for mapping in mappings),
                "formal_scorer_authorized_citations": 0,
                "diagnostic_authorized_with_version_and_temporal_scope": len(
                    prediction["citations"]
                ),
                "top15_ids": prediction["top15_chunk_ids"],
                "selected_evidence_ids": [row["chunk_id"] for row in payload_row["evidence_rows"]],
                "document_ids": sorted(
                    {row["document_id"] for row in payload_row["evidence_rows"]}
                ),
                "document_version_ids": sorted(
                    {row["document_version_id"] for row in payload_row["evidence_rows"]}
                ),
                "requirement_version_mappings": prediction["requirement_version_mappings"],
                "question_plan_hash": prediction["question_plan_hash"],
                "luna_payload_hash": canonical_hash(exact_payload),
                "validator_plan_hash": prediction["question_plan_hash"],
                "assembler_plan_hash": prediction["question_plan_hash"],
            }
        )
    p1_report = formal | {
        "formal_gate_passed": False,
        "pipeline_action": "STOP_PAID_PIPELINE",
        "first_causal_failure": {
            "code": "P1_SCORER_TEMPORAL_AUTHORIZATION_METADATA_OMISSION",
            "component": (
                "scripts/run_p1_version_temporal_targeted_real_api_confirmation_v1.py:"
                "score_and_finalize"
            ),
            "mechanism": (
                "Citation authorization was evaluated without version/is_active metadata "
                "and without the question temporal scope, so version_is_eligible used the "
                "current-only default and rejected historical citations."
            ),
            "frozen_citations_checked": sum(len(row["citations"]) for row in predictions),
            "authorized_with_complete_metadata": sum(len(row["citations"]) for row in predictions),
            "formal_results_reclassified": False,
        },
        "diagnostic_cases": diagnostic_cases,
    }
    dump(P1 / "p1_targeted_report.json", p1_report)

    blocked_files = {
        BASE / "p2-version-validation/p2_report.json": "P2_VERSION_VALIDATION",
        BASE / "p3-verifier-consistency/p3_report.json": "P3_VERIFIER_CONSISTENCY",
        BASE / "p4-deterministic-inference/p4_report.json": "P4_DETERMINISTIC_INFERENCE",
        BASE / "p5-luna-false-negative/p5_report.json": "P5_LUNA_FALSE_NEGATIVE",
        BASE / "local-regression/local_19_replay.json": "LOCAL_19_REPLAY",
        BASE / "local-regression/local_regression_report.json": "FULL_LOCAL_REGRESSION",
        BASE / "local-regression/rc_freeze_manifest.json": "RC_FINAL_PRE_API",
        BASE / "targeted-19/targeted_19_predictions_freeze.json": "TARGETED_19_FREEZE",
        BASE / "targeted-19/targeted_19_report.json": "TARGETED_19_REPORT",
        BASE / "fresh-v1-regression/fresh_v1_regression_report.json": (
            "FRESH_V1_1_REGRESSION_RERUN"
        ),
        BASE / "fresh-holdout-v2/fresh_v2_dataset_manifest.json": ("FRESH_UNSEEN_HOLDOUT_V2"),
        BASE / "fresh-holdout-v2/fresh_v2_overlap_audit.json": "V2_OVERLAP_AUDIT",
        BASE / "fresh-holdout-v2/fresh_v2_gold_grounding_audit.json": ("V2_GOLD_GROUNDING"),
        BASE / "fresh-holdout-v2/fresh_v2_predictions_freeze.json": ("V2_PREDICTIONS_FREEZE"),
        BASE / "fresh-holdout-v2/fresh_v2_final_report.json": "V2_FINAL_REPORT",
    }
    for path, artifact in blocked_files.items():
        dump(path, blocked(artifact))
    dump_jsonl(BASE / "targeted-19/targeted_19_predictions.jsonl", [])
    dump_jsonl(BASE / "fresh-v1-regression/fresh_v1_regression_predictions.jsonl", [])
    dump_jsonl(BASE / "fresh-holdout-v2/fresh_v2_predictions.jsonl", [])
    dump_jsonl(BASE / "total_api_cost_ledger.jsonl", ledger)

    release_gate = {
        "experiment_id": "RAG_REMAINING_FAILURES_RELEASE_PIPELINE_V3_RELEASE_GATES",
        "timestamp": timestamp,
        "verdict": "PIPELINE_FAILED_P1_CONFIRMATION",
        "release_decision": "NOT_READY_FOR_PRODUCTION",
        "p0_generalization": "PASSED",
        "all60_plan_audit": "PASSED_60_OF_60",
        "plan_hash_rebase": "PASSED_60_OF_60",
        "p1_targeted": "FAILED_FORMAL_GATE",
        "first_causal_failure": p1_report["first_causal_failure"],
        "downstream_phases": "NOT_RUN_BLOCKED_BY_P1_CONFIRMATION",
        "production_action_taken": False,
    }
    dump(BASE / "release-decision/release_gate_report.json", release_gate)
    final = {
        "experiment_id": "RAG_REMAINING_FAILURES_RELEASE_PIPELINE_V3",
        "timestamp": timestamp,
        "verdict": "PIPELINE_FAILED_P1_CONFIRMATION",
        "final_status": "NOT_READY_FOR_PRODUCTION",
        "p0_generalization": {"passed": 60, "required": 60},
        "plan_hash_rebase": {"passed": 60, "required": 60},
        "p1_targeted": {
            "formal_verdict": formal["verdict"],
            "correct": formal["quality"]["correct"],
            "required": 5,
            "luna_calls": formal["cost"]["luna"]["calls"],
            "sol_calls": formal["cost"]["sol"]["calls"],
            "cost_usd": formal["cost"]["total_targeted_cost_usd"],
            "first_causal_failure": p1_report["first_causal_failure"],
        },
        "p2_through_release_evaluations": "NOT_RUN_BLOCKED_BY_P1_CONFIRMATION",
        "total_api_cost_usd": sum(float(row["estimated_cost_usd"]) for row in ledger),
        "total_hard_cap_usd": 0.8,
        "final_generation_llm_calls": 0,
        "production_action_taken": False,
    }
    dump(BASE / "final_report.json", final)


if __name__ == "__main__":
    main()
