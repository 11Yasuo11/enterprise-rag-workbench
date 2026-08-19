# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUT_DIR = Path("data/eval/phase5j")
DATASET_PATH = OUT_DIR / "v3_phase5j_generator_holdout_30.jsonl"
FREEZE_PATH = OUT_DIR / "phase5j_generator_holdout_freeze.json"
DATASET_ID = "acmeai-enterprise-rag-v3-phase5j-generator-holdout-30"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _mk(query_id: str, category: str, question: str, required_facts: list[str], contexts: list[dict]) -> dict:
    return {
        "query_id": query_id,
        "category": category,
        "question": question,
        "required_facts": required_facts,
        "contexts": contexts,
    }


def main() -> None:
    rows = []
    n = 1

    def c(chunk_id: str, label: str, text: str) -> dict:
        return {"chunk_id": chunk_id, "citation_label": label, "text": text}

    # multi-fact answers (10)
    for i in range(10):
        rows.append(
            _mk(
                f"p5j_mf_{i+1:02d}",
                "multi_fact_answers",
                "Provide the deployment approval identifier and API recovery time objective.",
                ["ENG-DEP-17", "four hours"],
                [
                    c("eng1", "C1", "The production deployment approval identifier is ENG-DEP-17."),
                    c("ops1", "C2", "The recovery time objective for the customer API is four hours."),
                ],
            )
        )
        n += 1

    # multi-document facts (8)
    for i in range(8):
        rows.append(
            _mk(
                f"p5j_md_{i+1:02d}",
                "multiple_document_facts",
                "Give the runbook code and incident reporting window.",
                ["OPS-REC-W29", "15 minutes"],
                [
                    c("west1", "C1", "The west-region customer API recovery sequence uses runbook code OPS-REC-W29."),
                    c("sec1", "C2", "Suspected severity-one incidents must be reported within 15 minutes of discovery."),
                ],
            )
        )
        n += 1

    # date/range endpoints (6) -- includes p5e_date_pos_04 style failure mode
    endpoint_context = [
        c("api1", "C1", "Project Atlas uses API version v3. Its production endpoint identifier is ATLAS-API-301."),
        c("launch1", "C2", "Project Atlas launched on April 12, 2026. The initial customer region is eu-west."),
    ]
    for i in range(6):
        rows.append(
            _mk(
                f"p5j_dr_{i+1:02d}",
                "date_range_endpoints",
                "Using the launch and API references, provide the production endpoint identifier and initial customer region.",
                ["ATLAS-API-301", "eu-west"],
                endpoint_context,
            )
        )
        n += 1

    # same-document multi-chunk (6)
    same_doc_context = [
        c("fin1", "C1", "Expense reports are due within ten business days."),
        c("fin2", "C2", "The international travel approval code is FIN-TRAVEL-52."),
    ]
    for i in range(6):
        rows.append(
            _mk(
                f"p5j_sc_{i+1:02d}",
                "same_document_multi_chunk",
                "Provide both the expense report deadline and the travel approval code.",
                ["ten business days", "FIN-TRAVEL-52"],
                same_doc_context,
            )
        )

    if len(rows) != 30:
        raise ValueError(f"expected 30 rows, got {len(rows)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    h = _sha256_bytes(DATASET_PATH.read_bytes())
    FREEZE_PATH.write_text(
        json.dumps(
            {
                "dataset_id": DATASET_ID,
                "dataset_path": str(DATASET_PATH),
                "dataset_hash": h,
                "case_count": 30,
                "distribution": {
                    "multi_fact_answers": 10,
                    "multiple_document_facts": 8,
                    "date_range_endpoints": 6,
                    "same_document_multi_chunk": 6,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {DATASET_PATH}")
    print(f"hash: {h}")


if __name__ == "__main__":
    main()

