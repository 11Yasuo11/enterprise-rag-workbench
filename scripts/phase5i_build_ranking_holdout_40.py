# ruff: noqa: E501, B007
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT_DIR = Path("data/eval/phase5i")
OUT_DATASET_PATH = OUT_DIR / "v3_phase5i_version_ranking_holdout_40_cases.jsonl"
FREEZE_PATH = OUT_DIR / "phase5i_holdout_freeze.json"
DATASET_ID = "acmeai-enterprise-rag-v3-phase5i-version-ranking-holdout-40"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _row(query_id: str, category: str, question: str, required_docs: list[str], required_facts: list[str]) -> dict:
    return {
        "query_id": query_id,
        "category": category,
        "question": question,
        "required_document_ids": required_docs,
        "required_facts": required_facts,
        "expected_answerable": True,
        "should_abstain": False,
        "principal": {"principal_id": "evaluation-user", "tenant_id": "acmeai", "permission_groups": ["employees"]},
    }


def main() -> None:
    rows = []
    i = 1

    # version-sensitive (16)
    version_qs = [
        ("Using the current runbook version, what code is specified for east-region API recovery?", ["recovery-runbook-east"], ["OPS-REC-E17"]),
        ("Using the current runbook version, what code is specified for west-region API recovery?", ["recovery-runbook-west"], ["OPS-REC-W29"]),
        ("Under the active west runbook version, what standby cluster is the failover target?", ["recovery-runbook-west"], ["us-east standby cluster"]),
        ("Under the active east runbook version, what standby cluster is the failover target?", ["recovery-runbook-east"], ["eu-central standby cluster"]),
    ]
    for base in range(4):
        for q, d, f in version_qs:
            rows.append(_row(f"p5i_ver_{i:02d}", "version_sensitive", q, d, f))
            i += 1
            if i > 16:
                break
        if i > 16:
            break

    # region-sensitive (8)
    region_qs = [
        ("For east-region customer API recovery, what runbook code is used?", ["recovery-runbook-east"], ["OPS-REC-E17"]),
        ("For west-region customer API recovery, what runbook code is used?", ["recovery-runbook-west"], ["OPS-REC-W29"]),
        ("For east-region recovery, which standby cluster is targeted?", ["recovery-runbook-east"], ["eu-central standby cluster"]),
        ("For west-region recovery, which standby cluster is targeted?", ["recovery-runbook-west"], ["us-east standby cluster"]),
    ]
    for rep in range(2):
        for q, d, f in region_qs:
            rows.append(_row(f"p5i_reg_{rep*4 + len(rows)-16 +1:02d}", "region_sensitive", q, d, f))
            if len([r for r in rows if r["category"] == "region_sensitive"]) >= 8:
                break
        if len([r for r in rows if r["category"] == "region_sensitive"]) >= 8:
            break

    # version+region (8)
    both_qs = [
        ("In recovery-runbook-east version 1.0, what runbook code is specified?", ["recovery-runbook-east"], ["OPS-REC-E17"]),
        ("In recovery-runbook-west version 1.0, what runbook code is specified?", ["recovery-runbook-west"], ["OPS-REC-W29"]),
        ("In recovery-runbook-east version 1.0, what standby cluster is named?", ["recovery-runbook-east"], ["eu-central standby cluster"]),
        ("In recovery-runbook-west version 1.0, what standby cluster is named?", ["recovery-runbook-west"], ["us-east standby cluster"]),
    ]
    for rep in range(2):
        for q, d, f in both_qs:
            rows.append(_row(f"p5i_vr_{rep*4 + len([x for x in rows if x['category']=='version_region'])+1:02d}", "version_region", q, d, f))
            if len([r for r in rows if r["category"] == "version_region"]) >= 8:
                break
        if len([r for r in rows if r["category"] == "version_region"]) >= 8:
            break

    # ordinary QA (8)
    ordinary = [
        ("What is the production deployment approval identifier?", ["engineering-deployment-handbook"], ["ENG-DEP-17"]),
        ("What is the customer API recovery time objective?", ["operations-continuity-plan"], ["four hours"]),
        ("How many remote-work days per week are allowed?", ["remote-work-policy"], ["two days per week"]),
        ("What is the escalation queue ID for priority-one customer outages?", ["customer-support-escalation"], ["CS-1842"]),
        ("What is the approved deletion workflow identifier?", ["data-retention-standard"], ["LEGAL-DEL-08"]),
        ("What is the international travel approval code?", ["finance-expense-policy"], ["FIN-TRAVEL-52"]),
        ("What is the Project Atlas API endpoint identifier?", ["project-atlas-api"], ["ATLAS-API-301"]),
        ("What is the Project Atlas initial customer region?", ["project-atlas-launch"], ["eu-west"]),
    ]
    for idx, (q, d, f) in enumerate(ordinary, 1):
        rows.append(_row(f"p5i_ord_{idx:02d}", "ordinary_qa", q, d, f))

    if len(rows) != 40:
        raise ValueError(f"Expected 40 rows, got {len(rows)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DATASET_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    h = _sha256_bytes(OUT_DATASET_PATH.read_bytes())
    freeze = {
        "dataset_id": DATASET_ID,
        "dataset_path": str(OUT_DATASET_PATH),
        "dataset_hash": h,
        "distribution": {
            "version_sensitive": 16,
            "region_sensitive": 8,
            "version_region": 8,
            "ordinary_qa": 8,
        },
        "case_count": 40,
    }
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {OUT_DATASET_PATH}")
    print(f"hash: {h}")


if __name__ == "__main__":
    main()

