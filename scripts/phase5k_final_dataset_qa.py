# ruff: noqa: E501, I001
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


DATASET_PATH = Path("data/eval/phase5k/acmeai_enterprise_rag_v3_final_unseen_e2e_120.jsonl")
OUT_PATH = Path("data/eval/phase5k/phase5k_final_dataset_qa.json")
EXPECTED = {
    "single_document": 10,
    "two_document": 16,
    "three_document": 16,
    "same_document_multi_chunk": 10,
    "semantic_paraphrase": 10,
    "exact_identifier": 8,
    "near_duplicate": 8,
    "version_sensitive": 8,
    "region_sensitive": 6,
    "numeric_date_constraint": 8,
    "acl_should_abstain": 5,
    "tenant_isolation": 5,
    "prompt_injection": 5,
    "unsupported_no_answer": 5,
}


def main() -> None:
    rows = [json.loads(x) for x in DATASET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    errors = []
    if len(rows) != 120:
        errors.append(f"case count mismatch: {len(rows)}")
    dist = dict(Counter(r["category"] for r in rows))
    if dist != EXPECTED:
        errors.append(f"distribution mismatch: {dist} != {EXPECTED}")
    for r in rows:
        qid = r["query_id"]
        if len(r["question"].strip()) < 12:
            errors.append(f"{qid}: question too short")
        if r["expected_answerable"] and r["should_abstain"]:
            errors.append(f"{qid}: expected_answerable/should_abstain inconsistent")
        if r["category"] in {"acl_should_abstain", "tenant_isolation", "prompt_injection", "unsupported_no_answer"}:
            if r["expected_answerable"] is not False or r["should_abstain"] is not True:
                errors.append(f"{qid}: abstain category mislabel")
        else:
            if not r.get("required_document_ids"):
                errors.append(f"{qid}: required_document_ids empty")
            if not r.get("required_facts"):
                errors.append(f"{qid}: required_facts empty")
    out = {"valid": not errors, "count": len(rows), "distribution": dist, "errors": errors}
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        raise SystemExit("PHASE5K_FINAL_DATASET_QA_FAIL")
    print("phase5k final dataset QA PASS (120/120)")


if __name__ == "__main__":
    main()

