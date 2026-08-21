# Interview Cheat Sheet

One screen / one page.

## ARCHITECTURE

```text
ACL / Tenant / Region
→ Prompt-Injection Precheck
→ Temporal + Question Plan
→ Dense20 + BM25 20
→ RRF k=60
→ Candidate ≤30
→ Cross-Encoder → Top-15
→ Version / Auth validation
→ Atomic Requirements
→ CanonicalEvidenceMapping + Packets
→ Deterministic Support
   ├─ grounded → Deterministic
   └─ ambiguous → Luna → Sol (escalation only)
→ Local Validation
→ Universal Completeness Gate
→ deterministic-requirement-assembler-v3
→ Answer + Citations  OR  Safe Abstention
```

Serving: `POST /rag/query` and `POST /eval/run` → **same** `CanonicalRagRuntime`.
Final-generation LLM: **disabled**.

## LOCAL vs PRODUCTION

| | Local demo | Production config |
|---|---|---|
| Embeddings | hashing (safe/offline) | `text-embedding-3-small` |
| External calls | off by default | enabled only when reviewed |
| Cost | $0 | paid providers when used |

## RESULT (V14 / V2 regression)

| Metric | Value |
|---|---:|
| Cases | **60 / 60** |
| Unsupported | **0** |
| Incorrect abstention | **0** |
| Injection 055/056/057 | **3 / 3** |
| Citation metrics | **100%** |
| Requirement completeness | **100%** |
| Version correctness | **100%** |

**IMPORTANT DISCLAIMER:** Regression result after iterative development — **not** unseen 100% generalization / production accuracy.

## STATUS

Production-serving architecture **integrated and validated**.
**Not externally deployed.** Ready for deployment review.

## DO NOT SAY

- “100% on unseen questions”
- “Deployed to production”
- “V3 beat V2”
- “An LLM writes the final answer”
- “Hashing embeddings are the production embedding model”
