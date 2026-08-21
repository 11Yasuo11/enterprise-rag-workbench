from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DATASET_DIR = Path("data/eval/phase5d")
DATASET_PATH = DATASET_DIR / "v3_phase5d_safety_holdout_40_cases.jsonl"
QA_OUTPUT_PATH = DATASET_DIR / "phase5d_safety_dataset_qa.json"

CORPUS_DIR = Path("data/synthetic_company")
FORBIDDEN_QUERY_IDS = {"clean_none_003", "clean_inj_003"}

TOKEN = re.compile(r"[a-z0-9]+")


def _load_corpus_doc_index() -> dict[str, dict[str, Any]]:
    """
    Minimal corpus frontmatter extraction for QA checks.
    """
    docs: dict[str, dict[str, Any]] = {}
    for p in sorted(CORPUS_DIR.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        if not text.startswith("---"):
            continue
        parts = text.split("---")
        if len(parts) < 3:
            continue
        fm = parts[1]
        doc_id_m = re.search(r"document_id:\s*(.+)", fm)
        version_m = re.search(r'version:\s*"?([^"\n]+)"?', fm)
        doc_id = doc_id_m.group(1).strip() if doc_id_m else p.stem
        version = version_m.group(1).strip() if version_m else ""
        docs[doc_id] = {"version": version, "file": p.name, "text": text}
    return docs


def _load_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(json.loads(line))
    return cases


def _case_fingerprint(c: dict[str, Any]) -> str:
    # For debugging only
    payload = {
        k: c.get(k)
        for k in ("query_id", "category", "question", "expected_answerable", "should_abstain")
    }
    b = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:12]


def run_qa() -> dict[str, Any]:
    corpus = _load_corpus_doc_index()
    cases = _load_cases()

    errors: list[str] = []
    per_case: list[dict[str, Any]] = []

    dist = Counter(c["category"] for c in cases)

    if len(cases) != 40:
        errors.append(f"case_count mismatch: got {len(cases)}")

    for c in cases:
        qid = c["query_id"]
        cat = c["category"]
        ok = True
        reasons: list[str] = []

        if qid in FORBIDDEN_QUERY_IDS:
            ok = False
            reasons.append("forbidden query_id used")

        q = c.get("question", "")
        if not q or len(q.strip()) < 10:
            ok = False
            reasons.append("question too short")

        expected_answerable = c.get("expected_answerable")
        should_abstain = c.get("should_abstain")
        if expected_answerable is None or should_abstain is None:
            ok = False
            reasons.append("missing expected_answerable/should_abstain")
        else:
            if should_abstain != (not expected_answerable):
                ok = False
                reasons.append("expected_answerable/should_abstain mismatch")

        # Category-specific checks
        if cat in {"prompt_injection", "acl_sensitive", "tenant_isolation", "partial_no_answer"}:
            if expected_answerable is not False:
                ok = False
                reasons.append(f"{cat} must be expected_answerable=false")
            if should_abstain is not True:
                ok = False
                reasons.append(f"{cat} must be should_abstain=true")

        if cat == "prompt_injection":
            sec = c.get("security_checks", [])
            if "prompt_injection" not in sec:
                ok = False
                reasons.append("prompt_injection missing security_checks entry")

        if cat == "acl_sensitive":
            forbidden = c.get("forbidden_document_ids", [])
            if not forbidden:
                ok = False
                reasons.append("acl_sensitive missing forbidden_document_ids")
            else:
                for did in forbidden:
                    if did not in corpus:
                        ok = False
                        reasons.append(f"acl forbidden doc_id unknown: {did}")

        if cat == "version_region":
            if expected_answerable is not True:
                ok = False
                reasons.append("version_region must be expected_answerable=true")
            if c.get("expected_answer") is None:
                ok = False
                reasons.append("version_region expected_answer missing")
            if not c.get("required_version_ids"):
                ok = False
                reasons.append("version_region missing required_version_ids")
            else:
                for doc_id, _v in c["required_version_ids"].items():
                    if doc_id not in corpus:
                        ok = False
                        reasons.append(f"version doc_id unknown: {doc_id}")

        if c.get("expected_answer") is not None:
            ans = str(c["expected_answer"]).casefold()
            if ans and ans in c["question"].casefold():
                ok = False
                reasons.append("expected_answer leaked into question")

        per_case.append(
            {
                "query_id": qid,
                "category": cat,
                "ok": ok,
                "fingerprint": _case_fingerprint(c),
                "reasons": reasons,
            }
        )
        if not ok:
            errors.append(f"{qid}: {', '.join(reasons)}")

    # Distribution check
    expected_dist = {
        "prompt_injection": 20,
        "partial_no_answer": 8,
        "acl_sensitive": 4,
        "tenant_isolation": 4,
        "version_region": 4,
    }
    if dist != expected_dist:
        errors.append(f"distribution mismatch: got {dict(dist)} expected {expected_dist}")

    valid = not errors
    result = {
        "valid": valid,
        "case_count": len(cases),
        "category_distribution": dict(dist),
        "errors": errors,
        "per_case": per_case,
        "qa_version": "phase5d_safety_dataset_qa_v1",
        "dataset_path": str(DATASET_PATH),
        "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
    }
    QA_OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    result = run_qa()
    if not result["valid"]:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit("PHASE5D_SAFETY_DATASET_QA_FAILURE")
    print(f"phase5d safety dataset QA PASS ({result['case_count']}/40)")


if __name__ == "__main__":
    main()
