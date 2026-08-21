# ruff: noqa: E501
"""Question-only current-plan rebase audit for RAG_FINAL_RELEASE_PIPELINE_V2."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_workbench.experiments.atomic_requirement_contract_v1 import decompose_question

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/experiments/final-release-pipeline-v2"
OUT = BASE / "plan-hash-rebase"
QUESTIONS = ROOT / ".local/evaluation-input/fresh_unseen_eval_v1_1_20260820/fresh_unseen_questions_v1_1.json"
OLD_PREFLIGHT = ROOT / "data/experiments/fresh-unseen-evaluation-v1-2/deterministic_preflight.json"
OLD_FREEZE = ROOT / "data/experiments/fresh-unseen-evaluation-v1-2/freeze_manifest.json"

# Question-only semantic inspection. These are general P0 contract violations,
# not benchmark gold facts or expected answers.
STRUCTURAL_ISSUES: dict[str, list[str]] = {
    "fresh_v1_006": ["CONTEXT_CLAUSE_PROMOTED_TO_REQUIREMENT"],
    "fresh_v1_012": ["FRAMING_PROMOTED_TO_REQUIREMENT"],
    "fresh_v1_018": ["FRAMING_PROMOTED_TO_REQUIREMENT"],
    "fresh_v1_020": ["FRAMING_PROMOTED_TO_REQUIREMENT"],
    "fresh_v1_022": ["COORDINATED_OUTPUT_MALFORMED", "R1_DROPS_SHARED_HEAD_NOUN"],
    "fresh_v1_024": ["INDEPENDENT_OUTPUTS_MERGED", "REQUESTED_OUTPUT_DROPPED"],
    "fresh_v1_045": ["PRINCIPAL_FRAMING_PROMOTED_TO_REQUIREMENT"],
    "fresh_v1_046": ["PRINCIPAL_FRAMING_PROMOTED_TO_REQUIREMENT"],
    "fresh_v1_047": ["PRINCIPAL_FRAMING_PROMOTED_TO_REQUIREMENT"],
    "fresh_v1_053": ["REQUEST_FRAMING_PROMOTED_TO_REQUIREMENT"],
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str, binary: bool = False):
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=not binary)
    return result.stdout


def dirty_hash() -> str:
    digest = hashlib.sha256()
    digest.update(git_output("diff", "--binary", "HEAD", "--", binary=True))
    untracked = git_output("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    for relative in sorted(item for item in untracked if item):
        path = ROOT / relative
        if path.is_file():
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    questions_payload = load_json(QUESTIONS)
    allowed_top_keys = {"dataset_id", "version", "created_at", "case_count", "cases"}
    forbidden = {
        "expected_answer", "expected_facts", "expected_versions", "required_version_ids",
        "required_document_ids", "should_abstain", "expected_answerability", "gold_notes",
    }
    seen_keys = set(questions_payload) | {key for row in questions_payload["cases"] for key in row}
    gold_fields = sorted(seen_keys & forbidden)
    old_cases = {row["case_id"]: row for row in load_json(OLD_PREFLIGHT)["cases"]}
    old_hashes = load_json(OLD_FREEZE)["hashes"]["question_plan_hashes"]
    audit_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for source in questions_payload["cases"]:
        case_id = source["case_id"]
        first = decompose_question(source["question"])
        second = decompose_question(source["question"])
        first_dict = first.as_dict()
        second_dict = second.as_dict()
        old_plan = old_cases[case_id]["question_plan"]
        old_requirements = old_plan["requirements"]
        old_qualifiers = old_plan["context_qualifiers"]
        semantic_changed = (
            old_requirements != first_dict["requirements"]
            or old_qualifiers != first_dict["context_qualifiers"]
        )
        deterministic = first_dict == second_dict and first.question_plan_hash == second.question_plan_hash
        ids_stable = [item.requirement_id for item in first.requirements] == [
            f"R{i}" for i in range(1, len(first.requirements) + 1)
        ] and [item.qualifier_id for item in first.context_qualifiers] == [
            f"Q{i}" for i in range(1, len(first.context_qualifiers) + 1)
        ]
        issue_codes = STRUCTURAL_ISSUES.get(case_id, [])
        structurally_valid = (
            first.cardinality_match
            and len(first.detected_output_units) == len(first.requirements)
            and ids_stable
            and not issue_codes
        )
        change_reason = (
            "P0_SEMANTIC_DECOMPOSITION_OR_QUALIFIER_CHANGE"
            if semantic_changed
            else "P0_PLAN_SERIALIZATION_ADDED_OUTPUT_UNITS_AND_CARDINALITY"
        )
        audit_rows.append(
            {
                "case_id": case_id,
                "question": source["question"],
                "old_fresh_v1_2_hash": old_hashes[case_id],
                "current_hash_run_1": first.question_plan_hash,
                "current_hash_run_2": second.question_plan_hash,
                "deterministically_stable": deterministic,
                "hash_changed_from_old_baseline": first.question_plan_hash != old_hashes[case_id],
                "change_expected_due_to_p0": True,
                "change_reason": change_reason,
                "semantic_plan_changed": semantic_changed,
                "old_requirements": old_requirements,
                "old_qualifiers": old_qualifiers,
                "current_detected_output_units": list(first.detected_output_units),
                "current_requirements": first_dict["requirements"],
                "current_qualifiers": first_dict["context_qualifiers"],
                "stable_requirement_ids": ids_stable,
                "no_gold_fields_involved": not gold_fields,
                "structurally_valid_under_p0": structurally_valid,
                "semantic_issue_codes": issue_codes,
            }
        )
        baseline_rows.append(
            {
                "case_id": case_id,
                "question_sha256": hashlib.sha256(source["question"].encode()).hexdigest(),
                "question_plan": first_dict,
                "question_plan_hash": first.question_plan_hash,
                "canonical": False,
                "status": "DRAFT_REBASE_FAILED" if not structurally_valid else "DRAFT_PENDING_GLOBAL_GATE",
            }
        )
    stable_count = sum(row["deterministically_stable"] for row in audit_rows)
    valid_count = sum(row["structurally_valid_under_p0"] for row in audit_rows)
    passed = stable_count == valid_count == 60 and not gold_fields
    manifest = {
        "experiment_id": "RAG_FINAL_RELEASE_PIPELINE_V2_PLAN_HASH_REBASE",
        "timestamp": now(),
        "status": "CANONICAL" if passed else "NOT_FROZEN_PLAN_HASH_REBASE_FAILED",
        "canonical_baseline": passed,
        "candidate_git_sha": git_output("rev-parse", "HEAD").strip(),
        "dirty_worktree_hash": dirty_hash(),
        "planner_source_hash": sha256_path(ROOT / "src/rag_workbench/experiments/atomic_requirement_contract_v1/contract.py"),
        "p0_source_hash": sha256_path(ROOT / "src/rag_workbench/experiments/atomic_requirement_contract_v1/contract.py"),
        "p1_source_hash": sha256_path(ROOT / "src/rag_workbench/retrieval/temporal.py"),
        "dataset_questions_hash": sha256_path(QUESTIONS),
        "parent_fresh_v1_2_freeze_hash": sha256_path(OLD_FREEZE),
        "per_case_current_question_plan_hash": {row["case_id"]: row["current_hash_run_1"] for row in audit_rows},
        "deterministic_stability_count": stable_count,
        "structurally_valid_count": valid_count,
        "case_count": len(audit_rows),
        "gold_loaded": False,
        "runtime_input_keys": sorted(seen_keys),
        "allowed_top_level_question_keys": sorted(allowed_top_keys),
    }
    audit = {
        "experiment_id": "RAG_FINAL_RELEASE_PIPELINE_V2_PLAN_HASH_REBASE",
        "timestamp": now(),
        "verdict": "PLAN_HASH_REBASE_PASSED" if passed else "PLAN_HASH_REBASE_FAILED",
        "old_hash_equality_required": False,
        "old_hashes_superseded_by_p0": True,
        "deterministic_stability": {"passed": stable_count, "required": 60},
        "current_plans_structurally_valid": {"passed": valid_count, "required": 60},
        "atomic_requirement_contract_preserved": stable_count == 60,
        "no_downstream_redecomposition": True,
        "gold_fields_involved": bool(gold_fields),
        "gold_fields_detected": gold_fields,
        "p0_cardinality_preserved": valid_count == 60,
        "p0_qualifier_separation_preserved": not any(
            "FRAMING_PROMOTED_TO_REQUIREMENT" in issue
            or "CONTEXT_CLAUSE_PROMOTED_TO_REQUIREMENT" in issue
            or "PRINCIPAL_FRAMING_PROMOTED_TO_REQUIREMENT" in issue
            or "REQUEST_FRAMING_PROMOTED_TO_REQUIREMENT" in issue
            for row in audit_rows for issue in row["semantic_issue_codes"]
        ),
        "failed_case_ids": [row["case_id"] for row in audit_rows if not row["structurally_valid_under_p0"]],
        "first_causal_failure": next(
            (
                {"case_id": row["case_id"], "issue_codes": row["semantic_issue_codes"]}
                for row in audit_rows if not row["structurally_valid_under_p0"]
            ),
            None,
        ),
        "cases": audit_rows,
    }
    write_json(OUT / "plan_hash_rebase_audit.json", audit)
    write_jsonl(OUT / "current_question_plan_baseline.jsonl", baseline_rows)
    write_json(OUT / "current_question_plan_baseline_manifest.json", manifest)
    print(json.dumps({"verdict": audit["verdict"], "stable": stable_count, "valid": valid_count, "failed": audit["failed_case_ids"]}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
