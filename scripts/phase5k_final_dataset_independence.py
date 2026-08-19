# ruff: noqa: E501, I001
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path


DATASET_PATH = Path("data/eval/phase5k/acmeai_enterprise_rag_v3_final_unseen_e2e_120.jsonl")
OUT_PATH = Path("data/eval/phase5k/phase5k_final_dataset_independence.json")
THRESHOLD = 0.50


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _collect_prior_questions(root: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in {".json", ".jsonl"}:
            continue
        if "phase5k/acmeai_enterprise_rag_v3_final_unseen_e2e_120.jsonl" in str(p).replace("\\", "/"):
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if p.suffix == ".jsonl":
            for line in txt.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    q = obj.get("question")
                    if isinstance(q, str) and q.strip():
                        out.append((str(p), _norm(q)))
        else:
            try:
                obj = json.loads(txt)
            except Exception:
                continue
            stack = [obj]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    q = cur.get("question")
                    if isinstance(q, str) and q.strip():
                        out.append((str(p), _norm(q)))
                    stack.extend(cur.values())
                elif isinstance(cur, list):
                    stack.extend(cur)
    return out


def main() -> None:
    rows = [json.loads(x) for x in DATASET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    prior = _collect_prior_questions(Path("."))
    prior_q = [q for _, q in prior]
    worst = []
    failures = []
    for r in rows:
        q = _norm(r["question"])
        best = 0.0
        best_src = ""
        for idx, p in enumerate(prior_q):
            sim = SequenceMatcher(a=q, b=p).ratio()
            if sim > best:
                best = sim
                best_src = prior[idx][0]
        worst.append(best)
        item = {"query_id": r["query_id"], "max_overlap": round(best, 6), "closest_source": best_src}
        if best >= THRESHOLD:
            failures.append(item)
    report = {
        "threshold": THRESHOLD,
        "max_overlap": max(worst) if worst else 0.0,
        "status": "PASS" if not failures else "FAIL",
        "failing_cases": failures,
        "prior_question_count": len(prior),
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        raise SystemExit("PHASE5K_FINAL_DATASET_INDEPENDENCE_FAIL")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

