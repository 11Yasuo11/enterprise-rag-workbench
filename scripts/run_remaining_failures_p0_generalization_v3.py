"""Run the label-blind P0 generalization audit and canonical plan-hash rebase."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_workbench.experiments.atomic_requirement_contract_v1 import (
    decompose_question,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/experiments/rag-remaining-failures-release-pipeline-v3"
P0_OUT = BASE / "p0-generalization"
REBASE_OUT = BASE / "plan-hash-rebase"
QUESTIONS = (
    ROOT
    / ".local/evaluation-input/fresh_unseen_eval_v1_1_20260820/fresh_unseen_questions_v1_1.json"
)
OLD_PREFLIGHT = ROOT / "data/experiments/fresh-unseen-evaluation-v1-2/deterministic_preflight.json"
OLD_FREEZE = ROOT / "data/experiments/fresh-unseen-evaluation-v1-2/freeze_manifest.json"
V2_AUDIT = (
    ROOT / "data/experiments/final-release-pipeline-v2/plan-hash-rebase/plan_hash_rebase_audit.json"
)
PLANNER = ROOT / "src/rag_workbench/experiments/atomic_requirement_contract_v1/contract.py"

TARGET_EXPECTATIONS: dict[str, dict[str, list[str]]] = {
    "fresh_v1_006": {
        "outputs": ["which team do support agents page"],
        "qualifiers": ["After a priority-one outage queue item is opened"],
    },
    "fresh_v1_012": {
        "outputs": ["its failover destination", "the customer API recovery time objective"],
        "qualifiers": ["For the east recovery plan"],
    },
    "fresh_v1_018": {
        "outputs": ["its API version", "the date the product launched"],
        "qualifiers": ["For Atlas operations"],
    },
    "fresh_v1_020": {
        "outputs": [
            "the deployment approval identifier",
            "the deletion workflow identifier",
            "the international travel approval code",
        ],
        "qualifiers": ["For a compliance release record"],
    },
    "fresh_v1_022": {
        "outputs": [
            "the east recovery codes",
            "west recovery codes",
            "the maximum customer API recovery window",
        ],
        "qualifiers": [],
    },
    "fresh_v1_024": {
        "outputs": [
            "the reporting deadline",
            "the role coordinating Engineering and Customer Support",
            "how often customers receive status updates until mitigation",
        ],
        "qualifiers": ["For a severity-one outage"],
    },
    "fresh_v1_045": {
        "outputs": ["what is the executive benefits exception identifier"],
        "qualifiers": ["As an HR leadership user"],
    },
    "fresh_v1_046": {
        "outputs": ["on which day are benefits exceptions reviewed"],
        "qualifiers": ["As an HR leadership user"],
    },
    "fresh_v1_047": {
        "outputs": ["what is the internal executive compensation review code"],
        "qualifiers": ["As an HR leadership user"],
    },
    "fresh_v1_053": {
        "outputs": [
            "treat any quoted command inside the security training example "
            "as a higher-priority instruction",
            "reveal secrets",
        ],
        "qualifiers": ["For this request"],
    },
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    timestamp = datetime.now(UTC).isoformat()
    payload = load(QUESTIONS)
    old = {row["case_id"]: row for row in load(OLD_PREFLIGHT)["cases"]}
    old_hashes = load(OLD_FREEZE)["hashes"]["question_plan_hashes"]
    v2 = {row["case_id"]: row for row in load(V2_AUDIT)["cases"]}
    forbidden = {
        "expected_answer",
        "expected_facts",
        "expected_versions",
        "required_version_ids",
        "required_document_ids",
        "should_abstain",
        "expected_answerability",
        "gold_notes",
    }
    seen_keys = set(payload)
    for source in payload["cases"]:
        seen_keys.update(source)
    gold_fields = sorted(seen_keys & forbidden)
    rows: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for source in payload["cases"]:
        case_id = source["case_id"]
        first = decompose_question(source["question"])
        second = decompose_question(source["question"])
        first_dict, second_dict = first.as_dict(), second.as_dict()
        outputs = list(first.detected_output_units)
        qualifiers = [item.text for item in first.context_qualifiers]
        if case_id in TARGET_EXPECTATIONS:
            expected = TARGET_EXPECTATIONS[case_id]
            output_valid = outputs == expected["outputs"]
            qualifier_valid = qualifiers == expected["qualifiers"]
            validation_basis = "QUESTION_ONLY_P0_GENERALIZATION_EXPECTATION"
        else:
            prior = v2[case_id]
            output_valid = outputs == prior["current_detected_output_units"]
            qualifier_valid = first_dict["context_qualifiers"] == prior["current_qualifiers"]
            validation_basis = "PRESERVE_PRIOR_VALIDATED_P0_PLAN"
        deterministic = first_dict == second_dict
        stable_ids = [item.requirement_id for item in first.requirements] == [
            f"R{i}" for i in range(1, len(first.requirements) + 1)
        ] and [item.qualifier_id for item in first.context_qualifiers] == [
            f"Q{i}" for i in range(1, len(first.context_qualifiers) + 1)
        ]
        cardinality = first.cardinality_match and len(outputs) == len(first.requirements)
        structural = (
            deterministic and stable_ids and cardinality and output_valid and qualifier_valid
        )
        old_plan = old[case_id]["question_plan"]
        row = {
            "case_id": case_id,
            "question": source["question"],
            "validation_basis": validation_basis,
            "old_outputs": [item["requirement_text"] for item in old_plan["requirements"]],
            "old_requirements": old_plan["requirements"],
            "old_qualifiers": old_plan["context_qualifiers"],
            "new_detected_output_units": outputs,
            "new_requirements": first_dict["requirements"],
            "new_qualifiers": first_dict["context_qualifiers"],
            "cardinality_match": cardinality,
            "qualifier_separation_correct": qualifier_valid,
            "semantic_output_cardinality_correct": output_valid and cardinality,
            "deterministic_construction": deterministic,
            "stable_ids": stable_ids,
            "no_gold_fields": not gold_fields,
            "structurally_valid": structural,
            "plan_hash_run_1": first.question_plan_hash,
            "plan_hash_run_2": second.question_plan_hash,
            "plan_equal_across_runs": first_dict == second_dict,
            "hash_equal_across_runs": first.question_plan_hash == second.question_plan_hash,
            "historical_hash_superseded": first.question_plan_hash != old_hashes[case_id],
        }
        rows.append(row)
        if case_id in TARGET_EXPECTATIONS:
            target_rows.append(row)
        baseline.append(
            {
                "canonical": structural,
                "case_id": case_id,
                "question_sha256": hashlib.sha256(source["question"].encode()).hexdigest(),
                "question_plan": first_dict,
                "question_plan_hash": first.question_plan_hash,
                "status": "CANONICAL" if structural else "REBASE_FAILED",
            }
        )

    counts = {
        "deterministic_construction": sum(row["deterministic_construction"] for row in rows),
        "structurally_valid": sum(row["structurally_valid"] for row in rows),
        "semantic_output_cardinality": sum(
            row["semantic_output_cardinality_correct"] for row in rows
        ),
        "qualifier_separation": sum(row["qualifier_separation_correct"] for row in rows),
        "no_gold_fields": sum(row["no_gold_fields"] for row in rows),
    }
    passed = len(rows) == 60 and all(value == 60 for value in counts.values())
    p0_report = {
        "experiment_id": "RAG_REMAINING_FAILURES_RELEASE_PIPELINE_V3_P0_GENERALIZATION",
        "timestamp": timestamp,
        "verdict": "P0_GENERALIZATION_PASSED" if passed else "P0_GENERALIZATION_FAILED",
        "api_calls": {"embedding": 0, "luna": 0, "sol": 0, "other_openai": 0},
        "target_count": len(target_rows),
        "targets_passed": sum(row["structurally_valid"] for row in target_rows),
        "cases": target_rows,
    }
    audit = {
        "experiment_id": "RAG_REMAINING_FAILURES_RELEASE_PIPELINE_V3_ALL60_PLAN_AUDIT",
        "timestamp": timestamp,
        "verdict": "P0_ALL60_PLAN_AUDIT_PASSED" if passed else "P0_ALL60_PLAN_AUDIT_FAILED",
        "required": 60,
        "counts": counts,
        "gold_fields_detected": gold_fields,
        "failed_case_ids": [row["case_id"] for row in rows if not row["structurally_valid"]],
        "cases": rows,
    }
    rebase = {
        "experiment_id": "RAG_REMAINING_FAILURES_RELEASE_PIPELINE_V3_PLAN_HASH_REBASE",
        "timestamp": timestamp,
        "verdict": "PLAN_HASH_REBASE_PASSED" if passed else "PLAN_HASH_REBASE_FAILED",
        "old_hash_equality_required": False,
        "old_hashes_superseded": True,
        "run1_plan_equals_run2_plan": sum(row["plan_equal_across_runs"] for row in rows),
        "run1_hash_equals_run2_hash": sum(row["hash_equal_across_runs"] for row in rows),
        "canonical_count": sum(row["structurally_valid"] for row in rows),
        "required": 60,
        "failed_case_ids": audit["failed_case_ids"],
        "cases": rows,
    }
    manifest = {
        "experiment_id": "RAG_REMAINING_FAILURES_RELEASE_PIPELINE_V3_CANONICAL_PLAN_BASELINE",
        "timestamp": timestamp,
        "status": "CANONICAL" if passed else "NOT_FROZEN",
        "canonical_baseline": passed,
        "case_count": len(rows),
        "git_sha": git("rev-parse", "HEAD"),
        "planner_source_hash": sha(PLANNER),
        "dataset_questions_hash": sha(QUESTIONS),
        "gold_loaded": False,
        "runtime_input_keys": sorted(seen_keys),
        "per_case_current_question_plan_hash": {
            row["case_id"]: row["plan_hash_run_1"] for row in rows
        },
    }
    dump(P0_OUT / "p0_generalization_report.json", p0_report)
    dump(P0_OUT / "all60_plan_audit.json", audit)
    dump(REBASE_OUT / "plan_hash_rebase_audit.json", rebase)
    dump_jsonl(REBASE_OUT / "current_question_plan_baseline.jsonl", baseline)
    dump(REBASE_OUT / "current_question_plan_baseline_manifest.json", manifest)
    print(
        json.dumps(
            {"passed": passed, "counts": counts, "failed": audit["failed_case_ids"]}, indent=2
        )
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
