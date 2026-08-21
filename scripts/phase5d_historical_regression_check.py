from __future__ import annotations

import json
from pathlib import Path

from rag_workbench.safety.answerability_constraint_guard_v1 import (
    should_abstain_due_to_answerability_constraint,
)
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2

PHASE5C_DATASET_PATH = Path("data/eval/phase5c/v3_clean_120_cases.jsonl")
RETRIEVAL_TRACE_PATH = Path(
    "data/experiments/v3-phase5d-safe-generator-candidate/phase5c_safety_failures_retrieval_trace.jsonl"
)


def _load_phase5c_case(query_id: str) -> dict:
    for line in PHASE5C_DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if case["query_id"] == query_id:
            return case
    raise KeyError(f"Missing query_id={query_id} in {PHASE5C_DATASET_PATH}")


def _load_trace_item(query_id: str) -> dict:
    items = json.loads(RETRIEVAL_TRACE_PATH.read_text(encoding="utf-8"))
    for item in items:
        if item["query_id"] == query_id:
            return item
    raise KeyError(f"Missing query_id={query_id} in {RETRIEVAL_TRACE_PATH}")


def _evidence_texts_from_dense_candidates(trace_item: dict) -> list[str]:
    dense = trace_item.get("dense_candidates", [])
    return [c.get("text") or c.get("text_preview") or "" for c in dense]


def main() -> None:
    cases = ["clean_none_003", "clean_inj_003"]

    for qid in cases:
        case = _load_phase5c_case(qid)
        trace = _load_trace_item(qid)
        dense_texts = _evidence_texts_from_dense_candidates(trace)

        if qid == "clean_inj_003":
            triggered = is_question_injection_v2(case["question"])
            print(f"{qid}: QUESTION_INJECTION_GUARD_V2 triggered = {triggered}")
            if triggered is not True:
                raise SystemExit(1)
        elif qid == "clean_none_003":
            triggered = should_abstain_due_to_answerability_constraint(
                question=case["question"], evidence_texts=dense_texts
            )
            print(
                f"{qid}: ANSWERABILITY_CONSTRAINT_GUARD_V1 triggered = {triggered}"
            )
            if triggered is not True:
                raise SystemExit(1)
        else:
            raise AssertionError("Unexpected qid")

    print("Historical SAFE-ABSTAIN guard regression: PASS")


if __name__ == "__main__":
    main()

