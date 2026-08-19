# ruff: noqa: E501
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATASET_PATH = Path("data/eval/phase5i/v3_phase5i_version_ranking_holdout_40_cases.jsonl")
OUT_PATH = Path("data/eval/phase5i/phase5i_ranking_holdout_qa.json")
EXPECTED = {
    "version_sensitive": 16,
    "region_sensitive": 8,
    "version_region": 8,
    "ordinary_qa": 8,
}


def main() -> None:
    rows = [json.loads(x) for x in DATASET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    errors: list[str] = []
    if len(rows) != 40:
        errors.append(f"case_count expected 40 got {len(rows)}")
    dist = Counter(r["category"] for r in rows)
    if dict(dist) != EXPECTED:
        errors.append(f"distribution mismatch: {dict(dist)} != {EXPECTED}")
    for r in rows:
        if len(r["question"].strip()) < 12:
            errors.append(f"{r['query_id']}: question too short")
        if not r.get("required_document_ids"):
            errors.append(f"{r['query_id']}: required_document_ids empty")
        if not r.get("required_facts"):
            errors.append(f"{r['query_id']}: required_facts empty")
    out = {"valid": not errors, "count": len(rows), "distribution": dict(dist), "errors": errors}
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        raise SystemExit("PHASE5I_HOLDOUT_QA_FAIL")
    print("phase5i holdout QA PASS (40/40)")


if __name__ == "__main__":
    main()

