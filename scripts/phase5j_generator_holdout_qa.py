# ruff: noqa: E501
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

DATASET_PATH = Path("data/eval/phase5j/v3_phase5j_generator_holdout_30.jsonl")
OUT_PATH = Path("data/eval/phase5j/phase5j_generator_holdout_qa.json")
EXPECTED = {
    "multi_fact_answers": 10,
    "multiple_document_facts": 8,
    "date_range_endpoints": 6,
    "same_document_multi_chunk": 6,
}


def main() -> None:
    rows = [json.loads(x) for x in DATASET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    errors = []
    if len(rows) != 30:
        errors.append(f"expected 30 rows got {len(rows)}")
    dist = Counter(r["category"] for r in rows)
    if dict(dist) != EXPECTED:
        errors.append(f"distribution mismatch: {dict(dist)} != {EXPECTED}")
    for r in rows:
        if not r["required_facts"]:
            errors.append(f"{r['query_id']}: empty required_facts")
        if not r["contexts"]:
            errors.append(f"{r['query_id']}: empty contexts")
    out = {"valid": not errors, "count": len(rows), "distribution": dict(dist), "errors": errors}
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        raise SystemExit("PHASE5J_DATASET_QA_FAIL")
    print("phase5j holdout QA PASS (30/30)")


if __name__ == "__main__":
    main()

