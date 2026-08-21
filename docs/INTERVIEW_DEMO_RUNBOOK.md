# Interview Demo Runbook

How to walk an interviewer through Enterprise RAG Workbench in 3–10 minutes without drowning in experiment dumps.

**Current story:** Canonical production serving (Web + Eval) + V14 quality/safety regression.  
**Do not claim:** deployed to production · V2 60/60 is unseen generalization · V3 beat V2 · final-generation LLM writes answers.

---

## 3-minute demo

1. **README** — What / Why / Architecture diagram / Results (label V2 as regression).
2. **Canonical path** — `POST /rag/query` and `POST /eval/run` both call `CanonicalRagRuntime`.
3. **Why it’s interesting** — Hybrid + CE Top-15 + atomic requirements + deterministic support + Luna/Sol + fail-closed safety.

One-liner: “I built an evaluation loop that turns retrieval failures into general architecture fixes, then wired the validated stack into one production runtime.”

---

## 5-minute demo

| Min | Show | Path |
|---:|---|---|
| 0–1 | Problem + architecture | `README.md` |
| 1–2 | Serving = eval | `docs/CURRENT_SERVING_RAG_ARCHITECTURE.md` |
| 2–3 | Runtime orchestration | `src/rag_workbench/runtime/canonical_runtime.py` |
| 3–4 | Packet / year-token + injection | `…/atomic_requirement_contract_v1/verifier.py` + V14 report |
| 4–5 | Results honesty | V14 60/60 regression vs historical V2/V3 holdouts |

---

## 10-minute demo

Follow the 5-minute path, then:

| Min | Show | Path |
|---:|---|---|
| 5–6 | Config invariants 20/20/60/30/15 | `src/rag_workbench/runtime/config.py` |
| 6–7 | Hybrid + CE | `runtime/retrieval.py`, `reranking/cross_encoder.py` |
| 7–8 | Deterministic support / router | `deterministic_support.py`, assembler router |
| 8–9 | Safety | injection guard, ACL filters, abstention |
| 9–10 | Eval-driven story | failure → fix → regression → holdout |

---

## Questions to answer quickly

1. What does it do? — Grounded enterprise RAG with auth + versioning + abstention.
2. What’s interesting? — Requirements + evidence packets + deterministic path before LLMs.
3. Architecture? — See README Mermaid.
4. Why hybrid? — Lexical anchors (years, IDs, phrases) dense alone misses.
5. Why CE? — Re-rank the fused pool so required spans reach the working set.
6. Why Top-15? — Internal evidence pool for planning/support; not a chat “Top-5 dump.”
7. Luna/Sol? — Verifiers only; Sol escalates; assembler writes the answer.
8. Hallucinations? — Deterministic support, citation checks, completeness gate, fail-closed.
9. Versioning? — Temporal plan + deterministic version resolver before packets.
10. How measured? — Frozen datasets, failure census, targeted fix, full regression.
11. What’s validated? — V14 regression 60/60 + production finalization gates (local).
12. Deployed? — **No.** Ready for deployment review only.
