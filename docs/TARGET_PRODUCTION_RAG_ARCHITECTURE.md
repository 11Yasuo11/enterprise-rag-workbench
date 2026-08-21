# Target Production RAG Architecture

**Status:** Achieved locally via `CanonicalRagRuntime` (see Current Serving + Production Architecture docs).  
**Deployment:** Not deployed externally. Preferred status wording: *Production-serving architecture integrated and validated locally; ready for deployment review.*

Validated retrieval stack (executing):

`dense_top_k=20`, `bm25_top_k=20`, `rrf_k=60`, `fused_candidate_cap=30`, `cross_encoder_top_k=15`.

Historical V12 freeze prose that mentions 50/50/100 is **not** the current executing config.

---

## 1. One runtime principle

```
Evaluation  ──┐
              ├──→ CanonicalRagRuntime.query(...)
Web Serving ──┘
```

`POST /rag/query` and `POST /eval/run` both use `CanonicalRagRuntime`. Evaluation may add scoring/tracing only.

Package:

```
src/rag_workbench/runtime/
  config.py
  canonical_runtime.py
  retrieval.py
  guards.py
  identity_reranker.py   # local/hashing only
  types.py
```

---

## 2. Query path

```
User Query
↓
ACL / Tenant / Region
↓
Temporal + Question Planning
↓
Dense Top-20 + BM25 Top-20
↓
RRF k=60
↓
Candidate ≤30
↓
Cross-Encoder → Top-15
↓
Version / Authorization Validation
↓
Atomic Requirement Plan
↓
Canonical Evidence Mapping → Requirement Packets
↓
Deterministic Support?
   ├─ yes → deterministic assembler-v3
   └─ no  → Luna → Sol (escalation only)
↓
Local Validation + Completeness Gate
↓
Answer + Citations OR Safe Abstention
```

Final-generation LLM: **disabled**.

---

## 3. Production embedding

- Model: `text-embedding-3-small` (stored dim 64)
- Frozen semantic `index_identity`: `e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466`
- Provider labels `openai` and `openai-compatible` are treated as the same semantic space for retrieval
- Hashing embeddings allowed only when explicitly configured for local/test

---

## 4. Related

- [CURRENT_SERVING_RAG_ARCHITECTURE.md](CURRENT_SERVING_RAG_ARCHITECTURE.md)
- [PRODUCTION_RAG_ARCHITECTURE.md](PRODUCTION_RAG_ARCHITECTURE.md)
- [PRODUCTION_RAG_PROMOTION_AUDIT.md](PRODUCTION_RAG_PROMOTION_AUDIT.md)
