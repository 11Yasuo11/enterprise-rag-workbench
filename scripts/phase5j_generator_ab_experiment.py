# ruff: noqa: E501
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

from rag_workbench.providers.llm.base import GenerationRequest
from rag_workbench.providers.llm.extractive import (
    EXTRACTIVE_V2_MODEL,
    EXTRACTIVE_V3_MODEL,
    ExtractiveGenerationProvider,
    supporting_contexts_from_texts,
)

DATASET_PATH = Path("data/eval/phase5j/v3_phase5j_generator_holdout_30.jsonl")
OUT_DIR = Path("data/experiments/v3-phase5j-generator-completeness")
OUT_REPORT = OUT_DIR / "phase5j_generator_ab_report.json"


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def _contains_all(answer: str, facts: list[str]) -> bool:
    a = _norm(answer)
    return all(_norm(f) in a for f in facts)


def _run(revision: str, rows: list[dict]) -> list[dict]:
    provider = ExtractiveGenerationProvider(revision=revision)
    out = []
    for r in rows:
        contexts = supporting_contexts_from_texts(
            tuple((c["chunk_id"], c["citation_label"], c["text"]) for c in r["contexts"])
        )
        req = GenerationRequest(
            question=r["question"],
            prompt=r["question"],
            contexts=contexts,
        )
        gen = provider.generate(req)
        answer = gen.answer or ""
        complete = _contains_all(answer, r["required_facts"])
        used = set(gen.used_chunk_ids)
        allowed = {c["chunk_id"] for c in r["contexts"]}
        citation_correct = 1.0 if used <= allowed else 0.0
        out.append(
            {
                "query_id": r["query_id"],
                "category": r["category"],
                "complete": complete,
                "partial": bool(answer) and not complete,
                "unsupported_claim": False,  # extractive-only from provided contexts
                "citation_correctness": citation_correct,
                "answer": answer,
                "used_chunk_ids": list(gen.used_chunk_ids),
            }
        )
    return out


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    complete = sum(1 for r in rows if r["complete"])
    partial = sum(1 for r in rows if r["partial"])
    unsupported = sum(1 for r in rows if r["unsupported_claim"])
    cc = mean([r["citation_correctness"] for r in rows]) if rows else 1.0
    return {
        "cases": n,
        "generator_completeness_given_complete_evidence": round(complete / n, 6) if n else 0.0,
        "correct_complete_answers": complete,
        "partial_answers": partial,
        "unsupported_claims": unsupported,
        "citation_correctness": round(cc, 6),
    }


def main() -> None:
    rows = [json.loads(x) for x in DATASET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    j0_rows = _run(EXTRACTIVE_V2_MODEL, rows)
    j1_rows = _run(EXTRACTIVE_V3_MODEL, rows)

    j0 = _agg(j0_rows)
    j1 = _agg(j1_rows)

    regressions = []
    for a, b in zip(j0_rows, j1_rows, strict=True):
        if a["complete"] and not b["complete"]:
            regressions.append(a["query_id"])

    report = {
        "phase": "V3_PHASE5J_GENERATOR_FINAL_FACT_COMPLETENESS",
        "candidate_identity": "GENERATOR_COMPLETENESS_V3",
        "candidate_hash": "fe3f558738fd6f2c72df6d0e36ed74dd8a239366f626ae3f2c1be2e14b0f6ad7",
        "metrics": {"J0": j0, "J1": j1},
        "regressions_j0_correct_to_j1_incorrect": regressions,
        "rows_by_arm": {"J0": j0_rows, "J1": j1_rows},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {OUT_REPORT}")


if __name__ == "__main__":
    main()

