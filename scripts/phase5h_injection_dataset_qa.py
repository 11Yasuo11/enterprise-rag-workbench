# ruff: noqa: E501
from __future__ import annotations

import json
import re
from pathlib import Path

DATASET_PATH = Path("data/eval/phase5h/v3_phase5h_injection_holdout_30_cases.jsonl")
QA_PATH = Path("data/eval/phase5h/phase5h_injection_holdout_qa.json")


def main() -> None:
    rows = [json.loads(x) for x in DATASET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    errors: list[str] = []
    if len(rows) != 30:
        errors.append(f"expected 30 rows got {len(rows)}")
    attacks = [r for r in rows if r["attack"]]
    benign = [r for r in rows if not r["attack"]]
    if len(attacks) != 20:
        errors.append(f"expected 20 attacks got {len(attacks)}")
    if len(benign) != 10:
        errors.append(f"expected 10 benign got {len(benign)}")

    for r in rows:
        q = r["question"]
        if len(q.strip()) < 16:
            errors.append(f"{r['query_id']}: question too short")
        if r["attack"]:
            if not re.search(r"(?is)\b(ignore|override|bypass|comply|follow|obey|execute|higher[- ]priority|authoritative)\b", q):
                errors.append(f"{r['query_id']}: weak attack intent signal")
        else:
            if not re.search(r"(?is)\b(explain|analyze|describe|compare|summarize|what does|why)\b", q):
                errors.append(f"{r['query_id']}: weak benign-analysis framing")

    out = {"valid": not errors, "count": len(rows), "errors": errors}
    QA_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        raise SystemExit("PHASE5H_HOLDOUT_QA_FAIL")
    print("phase5h holdout QA PASS (30/30)")


if __name__ == "__main__":
    main()

