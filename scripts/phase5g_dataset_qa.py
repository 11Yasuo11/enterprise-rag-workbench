# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DATASET_DIR = Path("data/eval/phase5g")
DATASET_PATH = DATASET_DIR / "v3_phase5g_corrected_safety_diagnostic_48_cases.jsonl"
QA_OUTPUT_PATH = DATASET_DIR / "phase5g_dataset_qa.json"


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
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _check_case(c: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    qid = c["query_id"]
    cat = c["category"]
    q = c.get("question", "")

    if not q or len(q.strip()) < 12:
        errs.append("question too short")
    if c.get("expected_answerable") != (not c.get("should_abstain")):
        errs.append("expected_answerable/should_abstain mismatch")
    if (
        c.get("expected_answer") is not None
        and str(c["expected_answer"]).casefold() in q.casefold()
    ):
        errs.append("expected_answer leaked into question")

    # category semantics
    if cat in {
        "numeric_constraint_negative",
        "date_constraint_negative",
        "prompt_injection",
        "acl_sensitive",
        "tenant_isolation",
    }:
        if c.get("expected_answerable") is not False:
            errs.append(f"{cat} must be expected_answerable=false")
        if c.get("should_abstain") is not True:
            errs.append(f"{cat} must be should_abstain=true")

    if cat in {"numeric_constraint_positive", "date_constraint_positive", "version_sensitive"}:
        if c.get("expected_answerable") is not True:
            errs.append(f"{cat} must be expected_answerable=true")
        if c.get("should_abstain") is not False:
            errs.append(f"{cat} must be should_abstain=false")
        if not c.get("required_facts"):
            errs.append(f"{cat} missing required_facts")
        if not c.get("required_chunk_ids"):
            errs.append(f"{cat} missing required_chunk_ids")
        if not c.get("required_document_ids"):
            errs.append(f"{cat} missing required_document_ids")

    if cat == "prompt_injection" and "prompt_injection" not in c.get("security_checks", []):
        errs.append("prompt_injection missing security_checks")

    if cat == "acl_sensitive":
        forbidden = c.get("forbidden_document_ids", [])
        if not forbidden:
            errs.append("acl_sensitive missing forbidden_document_ids")
        if not re.search(r"(?is)\bhr\b", q):
            errs.append("acl_sensitive question should reference HR-restricted context")

    if cat == "tenant_isolation":
        principal = c.get("principal", {})
        if principal.get("tenant_id") == "acmeai":
            errs.append("tenant_isolation principal should be cross-tenant")

    if cat == "version_sensitive" and "version" not in q.casefold():
        errs.append("version_sensitive question should explicitly mention version")

    if cat == "numeric_constraint_positive" and not re.search(r"(?is)\bthreshold\b", q):
        errs.append("numeric_constraint_positive should ask threshold relation")

    if cat == "numeric_constraint_negative" and not re.search(
        r"(?is)\b(?:exactly|equal to|at least|no less than|at most|no more than|below|less than)\s*25\s*euros\b",
        q,
    ):
        errs.append("numeric_constraint_negative missing explicit boundary expression")

    if cat == "date_constraint_positive" and not re.search(r"(?is)\bon\s+2026\b", q):
        errs.append("date_constraint_positive missing explicit ON 2026 wording")

    if cat == "date_constraint_negative" and not re.search(
        r"(?is)\b(?:before|after)\s+2026\b",
        q,
    ):
        errs.append("date_constraint_negative missing before/after 2026 wording")

    # minimal structure
    if not isinstance(c.get("required_facts", []), list):
        errs.append("required_facts must be list")
    if not isinstance(c.get("required_chunk_ids", []), list):
        errs.append("required_chunk_ids must be list")
    if not isinstance(c.get("required_document_ids", []), list):
        errs.append("required_document_ids must be list")
    if not isinstance(c.get("required_version_ids", {}), dict):
        errs.append("required_version_ids must be dict")

    if errs:
        errs = [f"{qid}: {e}" for e in errs]
    return errs


def run_qa() -> dict[str, Any]:
    cases = _load_cases()
    errors: list[str] = []
    if len(cases) != 48:
        errors.append(f"case_count mismatch: got {len(cases)} expected 48")
    dist = Counter(c["category"] for c in cases)
    if dict(dist) != EXPECTED_DIST:
        errors.append(f"distribution mismatch: got {dict(dist)} expected {EXPECTED_DIST}")

    per_case = []
    for c in cases:
        errs = _check_case(c)
        ok = not errs
        errors.extend(errs)
        per_case.append(
            {"query_id": c["query_id"], "category": c["category"], "ok": ok, "errors": errs}
        )

    out = {
        "valid": not errors,
        "case_count": len(cases),
        "category_distribution": dict(dist),
        "errors": errors,
        "per_case": per_case,
        "dataset_path": str(DATASET_PATH),
        "dataset_sha256": _sha256_bytes(DATASET_PATH.read_bytes()),
        "qa_version": "phase5g_dataset_qa_v1",
    }
    QA_OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main() -> None:
    out = run_qa()
    if not out["valid"]:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        raise SystemExit("PHASE5G_DATASET_QA_FAILURE")
    print(f"phase5g dataset QA PASS ({out['case_count']}/48)")


if __name__ == "__main__":
    main()
