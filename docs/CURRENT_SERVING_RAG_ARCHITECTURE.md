# Current Serving RAG Architecture

**Audit date:** 2026-08-21 (updated after V14 + production finalization consolidation)  
**Branch / HEAD:** `chore/repository-hygiene-cleanup`  
**Scope:** Real Web → API request path

Production-serving architecture is **integrated and validated locally**; ready for deployment review. This is **not** an external production deployment.

**Quality logic:** CanonicalRagRuntime imports the V14-validated packet/year-token + V13 injection-guard modules (`atomic_requirement_contract_v1.verifier`). Latest V2 regression artifact: `data/experiments/rag-release-pipeline-v14/` (60/60; regression set, not an unseen claim).

---

## 1. Browser → API call graph

```
Browser (apps/web)
  └─ apps/web/app/chat/page.tsx
       └─ apiRequest("/rag/query", { query })
            └─ apps/web/lib/api.ts  →  NEXT_PUBLIC_API_URL
                 └─ POST /rag/query
                      └─ src/rag_workbench/api/app.py :: rag_query()
                           └─ CanonicalRagRuntime.query(...)
```

| UI | Endpoint | Backend |
|---|---|---|
| Chat | `POST /rag/query` | `CanonicalRagRuntime.query` |
| Evaluations | `POST /eval/run` | `EvaluationRunner(CanonicalRagRuntime)` |
| Retrieval debug | `POST /retrieve` | dense `Retriever` only (debug) |
| Documents | `GET/POST /documents*` | `IngestionPipeline` |

---

## 2. Canonical stack (executing)

`Dense20 + BM25 20 → RRF(k=60) → union≤30 → Cross-Encoder → Top-15 → Question Plan → Deterministic | Luna → Sol(escalation) → Assembler-v3`

| Knob | Value |
|---|---:|
| dense_top_k | 20 |
| bm25_top_k | 20 |
| rrf_k | 60 |
| fused_candidate_cap | 30 |
| cross_encoder_top_k | 15 |
| embedding_model (production) | `text-embedding-3-small` |
| semantic index_identity | `e598d9fb…` (frozen) |
| final_generation_llm | disabled |

**Note on historical freeze text:** some V12 research artifacts mention dense(50)/BM25(50)/union(100). Those strings are historical. The **executing** validated configuration is **20 / 20 / 60 / 30 / 15** as enforced by `ProductionRagConfig`.

---

## 3. Legacy

`RagService` (dense Top-5 + extractive) remains in-tree for historical tests only. It is **not** used by Web `/rag/query` or API `/eval/run`.

---

## 4. Related docs

- [PRODUCTION_RAG_ARCHITECTURE.md](PRODUCTION_RAG_ARCHITECTURE.md)
- [PRODUCTION_RAG_PROMOTION_AUDIT.md](PRODUCTION_RAG_PROMOTION_AUDIT.md)
- [TARGET_PRODUCTION_RAG_ARCHITECTURE.md](TARGET_PRODUCTION_RAG_ARCHITECTURE.md)
- Finalization artifacts: `data/experiments/production-finalization/`
