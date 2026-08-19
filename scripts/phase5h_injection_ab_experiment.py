# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

from rag_workbench.experiments.v2_quality_recovery import stable_hash
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.safety.question_injection_guard_v3 import is_question_injection_v3
from rag_workbench.safety.safety_identities import (
    QUESTION_INJECTION_GUARD_V2_CONFIG,
    QUESTION_INJECTION_GUARD_V2_HASH,
    QUESTION_INJECTION_GUARD_V3_CONFIG,
    QUESTION_INJECTION_GUARD_V3_HASH,
)

DATASET_PATH = Path("data/eval/phase5h/v3_phase5h_injection_holdout_30_cases.jsonl")
OUT_DIR = Path("data/experiments/v3-phase5h-injection-detection")
OUT_REPORT = OUT_DIR / "phase5h_injection_ab_report.json"


def _load():
    return [json.loads(x) for x in DATASET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]


def _eval(rows, fn):
    out = []
    for r in rows:
        trig = bool(fn(r["question"]))
        attack = bool(r["attack"])
        if attack:
            behavior = "CORRECT_ABSTENTION" if trig else "UNSAFE_MISS"
        else:
            behavior = "CORRECT_ALLOW" if not trig else "FALSE_POSITIVE_BLOCK"
        out.append({"query_id": r["query_id"], "attack": attack, "triggered": trig, "behavior": behavior})
    return out


def _metrics(rows):
    attacks = [r for r in rows if r["attack"]]
    benign = [r for r in rows if not r["attack"]]
    attack_safe = sum(1 for r in attacks if r["behavior"] == "CORRECT_ABSTENTION")
    misses = [r["query_id"] for r in rows if r["behavior"] == "UNSAFE_MISS"]
    fps = [r["query_id"] for r in rows if r["behavior"] == "FALSE_POSITIVE_BLOCK"]
    return {
        "attack_safety": {"correct": attack_safe, "total": len(attacks)},
        "benign_false_positives": {"count": len(fps), "total": len(benign), "query_ids": fps},
        "unsupported_answers": len(misses),
        "unsupported_answer_query_ids": misses,
    }


def main() -> None:
    rows = _load()
    h0_rows = _eval(rows, is_question_injection_v2)
    h1_rows = _eval(rows, is_question_injection_v3)
    report = {
        "phase": "V3_PHASE5H_INJECTION_DETECTION_COMPLETION",
        "candidate_identity": "QUESTION_INJECTION_GUARD_V3",
        "candidate_hash": QUESTION_INJECTION_GUARD_V3_HASH,
        "baseline_identity": "QUESTION_INJECTION_GUARD_V2",
        "baseline_hash": QUESTION_INJECTION_GUARD_V2_HASH,
        "config_hashes": {
            "QUESTION_INJECTION_GUARD_V2_CONFIG_HASH": stable_hash(QUESTION_INJECTION_GUARD_V2_CONFIG),
            "QUESTION_INJECTION_GUARD_V3_CONFIG_HASH": stable_hash(QUESTION_INJECTION_GUARD_V3_CONFIG),
        },
        "metrics": {
            "H0": _metrics(h0_rows),
            "H1": _metrics(h1_rows),
        },
        "rows_by_arm": {"H0": h0_rows, "H1": h1_rows},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {OUT_REPORT}")


if __name__ == "__main__":
    main()

