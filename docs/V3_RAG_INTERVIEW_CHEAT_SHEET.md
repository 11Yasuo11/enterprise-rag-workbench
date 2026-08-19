# V3 RAG Interview Cheat Sheet

**Status:** `V3_RESEARCH_FULLY_CLOSED_AFTER_CORRECTED_SCORING` · Production: **V2 (`main` / v2.0.0)** · V3 **not promoted**

---

## Architecture (retained V2 path)

```
Query → ACL/tenant/active-version → text-embedding-3-small
→ Dense Top-20 + BM25 Top-20 → RRF k=60 → Cross-Encoder → Top-5
→ GPT-5.6 Sol Evidence Sufficiency Judge → Generate→Verify / Safety
→ GENERATOR_COMPLETENESS_V2 → Answer + Citations | Safe Abstention
```

**Key paths:** `src/rag_workbench/generation/generator.py` (RagService) · `retrieval/hybrid.py` · `reranking/cross_encoder.py` · `answerability/` · `safety/` · `providers/llm/extractive.py`

---

## Final Authoritative Metrics (Phase-5KR only)

| Metric | Value |
|---|---:|
| Strict E2E Accuracy | **74.17%** |
| Precision | 94.52% |
| Recall | 71.88% |
| F1 | 81.66% |
| Correct complete answers | 69 |
| Correct abstentions | 20 |
| Incorrect abstentions | 27 |
| Unsupported answers | 4 |
| Citation validity | 1.00 |
| Citation correctness | 0.945 |
| Generator completeness (given complete evidence) | 89.61% |

**All three final arms tied:** FINAL_R (stable V2) = FINAL_A (current V3) = FINAL_B (final V3) = **74.17%**

**NOT final:** 16.67% (invalid Phase-K runner + broken Phase-5KC scorer)

---

## 3 Biggest Wins

1. **GENERATOR_COMPLETENESS_V2** — Historical controlled +13.33 pp (61.67% → 75.00%), 16 rescues, 0 regressions (Phase-5C diagnostic dataset; not final benchmark).
2. **Evaluation integrity recovery** — Phase-5KX/5KC/5KR found runner + scorer bugs; authoritative score is 74.17%, not 16.67%.
3. **Constraint Guard V2** — Fixed V1 boundary semantics (GT ≠ GE for policy scope); useful on diagnostic holdouts.

---

## 3 Biggest Failures

1. **Prompt injection generalization** — Phase-5H targeted holdout: V2 **14/20**, V3 **11/20** (6 unsafe failures); cumulative blocker.
2. **Three-document QA** — Final category **0%** (0/N); ranking/evidence-set coverage still fails multi-doc synthesis.
3. **Version-sensitive QA** — Final category **0%**; version-aware ranking showed no fresh holdout gain (baseline saturated).

---

## 3 Bugs Found (evaluation — mostly fixed)

| Bug | Effect | Status |
|---|---|---|
| **A — Invalid Phase-K runner** | Hashing embedding, Judge/recovery/generator bypass → false 16.67% | Fixed in Phase-5KC path; historical artifact preserved |
| **B — Citation scorer contract** | `expected_answer=None` → 73 false UNSUPPORTED | Fixed: `FINAL_E2E_SCORER_V2` (Phase-5KR) |
| **C — Required vs authorized IDs** | 6 apparent unauthorized → **0 true** | Fixed in Phase-5F evaluation logic |

---

## Why V3 Was Rejected

| Gate | Result |
|---|---|
| **Quality** | No improvement over stable V2 on final 120-case benchmark (all 74.17%) |
| **Security** | Phase-5H prompt-injection holdout 14/20 (6 unsafe) — hard release gate fails |

**Verdicts:** `V3_QUALITY_IMPROVEMENT_NOT_CONFIRMED` · `V3_RELEASE_PROMOTION_REJECTED`

---

## 5 Likely Interview Questions

1. **Why didn't V3 beat V2?** — Final unseen benchmark: identical 74.17%; component wins (generator) did not compound into net E2E gain; security holdout failed.
2. **What was the 16.67% result?** — Invalid runner (no hosted embedding/Judge) plus broken scorer; **not** real RAG quality. Authoritative: 74.17%.
3. **What does the Judge do?** — Evidence *sufficiency* (can validated Top-5 support a complete answer?), not relevance ranking.
4. **Biggest debugging discovery?** — Scorer passed `expected_answer=None` while dataset uses `required_facts`; zeroed citation correctness for all answered cases.
5. **What would you build next?** — P0 semantic prompt-injection defense; P1 three-doc evidence coverage; P2 version-sensitive retrieval.

---

## 30-Second Answer

I built and evaluated an enterprise RAG workbench with hybrid retrieval, Cross-Encoder reranking, a GPT-5.6 Sol evidence-sufficiency Judge, deterministic cited generation, and ACL/tenant/version safety controls. I ran controlled V3 experiments and discovered both real model limits and serious evaluation infrastructure bugs. After corrected execution and scoring, the authoritative final benchmark was **74.17%** strict accuracy — V3 did not beat stable V2 and failed a targeted prompt-injection safety holdout (**14/20**), so I did not promote it. Production remains V2.

---

*Full detail: [`V3_RAG_RESEARCH_POSTMORTEM_AND_INTERVIEW_GUIDE.md`](V3_RAG_RESEARCH_POSTMORTEM_AND_INTERVIEW_GUIDE.md)*
