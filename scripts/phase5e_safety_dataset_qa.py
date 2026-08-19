# ruff: noqa: E501, SIM102
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DATASET_DIR = Path("data/eval/phase5e")
DATASET_PATH = DATASET_DIR / "v3_phase5e_constraint_safety_holdout_48_cases.jsonl"
QA_OUTPUT_PATH = DATASET_DIR / "phase5e_safety_dataset_qa.json"


TOKEN = re.compile(r"[a-z0-9]+")


EXPECTED_DIST = {
    "prompt_injection": 6,
    "numeric_constraint_negative": 12,
    "numeric_constraint_positive": 8,
    "date_constraint_negative": 8,
    "date_constraint_positive": 4,
    "acl_sensitive": 4,
    "tenant_isolation": 3,
    "version_sensitive": 3,
}


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(json.loads(line))
    return cases


def _assert_category_expectations(case: dict[str, Any], *, errors: list[str]) -> None:
    cat = case["category"]
    expected_answerable = case.get("expected_answerable")
    should_abstain = case.get("should_abstain")

    if cat in {"numeric_constraint_negative", "date_constraint_negative", "prompt_injection", "acl_sensitive", "tenant_isolation"}:
        if expected_answerable is not False:
            errors.append(f"{case['query_id']}: {cat} expected_answerable must be false")
        if should_abstain is not True:
            errors.append(f"{case['query_id']}: {cat} should_abstain must be true")

    if cat in {"numeric_constraint_positive", "date_constraint_positive", "version_sensitive"}:
        if expected_answerable is not True:
            errors.append(f"{case['query_id']}: {cat} expected_answerable must be true")
        if should_abstain is not False:
            errors.append(f"{case['query_id']}: {cat} should_abstain must be false")

    if cat == "prompt_injection":
        if "prompt_injection" not in case.get("security_checks", []):
            errors.append(f"{case['query_id']}: prompt_injection missing security_checks entry")

    if cat == "acl_sensitive":
        if not case.get("forbidden_document_ids"):
            errors.append(f"{case['query_id']}: acl_sensitive missing forbidden_document_ids")

    if cat == "version_sensitive":
        if case.get("expected_answer") is None:
            errors.append(f"{case['query_id']}: version_sensitive missing expected_answer")
        if not case.get("required_version_ids"):
            errors.append(f"{case['query_id']}: version_sensitive missing required_version_ids")


def _check_expected_answer_leak(case: dict[str, Any], *, errors: list[str]) -> None:
    exp_ans = case.get("expected_answer")
    if exp_ans is None:
        return
    if str(exp_ans).casefold() in case.get("question", "").casefold():
        errors.append(f"{case['query_id']}: expected_answer leaked into question")


def _check_operator_phrases(case: dict[str, Any], *, errors: list[str]) -> None:
    q = case.get("question", "")
    cat = case["category"]

    if cat == "numeric_constraint_positive":
        if not re.search(r"(?is)\b(?:above|greater than|more than)\s*25\s*euros\b", q):
            errors.append(f"{case['query_id']}: numeric_constraint_positive missing GT wording for 25 euros")

    if cat == "numeric_constraint_negative":
        if not re.search(r"(?is)\b(?:exactly|equal to|at least|no less than|at most|no more than|below|less than)\s*25\s*euros\b", q):
            errors.append(f"{case['query_id']}: numeric_constraint_negative missing boundary wording for 25 euros")

    if cat == "date_constraint_positive":
        if not re.search(r"(?is)\b(?:on|in)\s+2026\b", q):
            errors.append(f"{case['query_id']}: date_constraint_positive missing on/in 2026 wording")

    if cat == "date_constraint_negative":
        if not re.search(r"(?is)\b(?:before|after)\s+2026\b", q):
            errors.append(f"{case['query_id']}: date_constraint_negative missing before/after 2026 wording")


def run_qa() -> dict[str, Any]:
    cases = _load_cases()
    errors: list[str] = []
    per_case: list[dict[str, Any]] = []

    if len(cases) != 48:
        errors.append(f"case_count mismatch: got {len(cases)} expected 48")

    dist = Counter(c["category"] for c in cases)
    if dict(dist) != EXPECTED_DIST:
        errors.append(f"distribution mismatch: got {dict(dist)} expected {EXPECTED_DIST}")

    for c in cases:
        qid = c["query_id"]
        ok = True
        reasons: list[str] = []

        try:
            if not c.get("question") or len(c["question"].strip()) < 10:
                ok = False
                reasons.append("question too short")

            _assert_category_expectations(c, errors=reasons)
            _check_expected_answer_leak(c, errors=reasons)
            _check_operator_phrases(c, errors=reasons)
        except Exception as e:
            ok = False
            reasons.append(f"exception: {e}")

        if not ok:
            errors.extend([f"{qid}: {r}" for r in reasons])

        per_case.append(
            {
                "query_id": qid,
                "category": c["category"],
                "ok": ok,
                "reasons": reasons,
            }
        )

    valid = not errors
    result = {
        "valid": valid,
        "case_count": len(cases),
        "category_distribution": dict(dist),
        "errors": errors,
        "per_case": per_case,
        "qa_version": "phase5e_safety_dataset_qa_v1",
        "dataset_path": str(DATASET_PATH),
        "dataset_sha256": _sha256_bytes(DATASET_PATH.read_bytes()),
    }
    QA_OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    result = run_qa()
    if not result["valid"]:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit("PHASE5E_SAFETY_DATASET_QA_FAILURE")
    print(f"phase5e safety dataset QA PASS ({result['case_count']}/48)")


if __name__ == "__main__":
    main()

