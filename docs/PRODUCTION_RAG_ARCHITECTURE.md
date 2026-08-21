# Production RAG Architecture

**Status:** Integrated into Web serving **and** API evaluation via `CanonicalRagRuntime`  
**Date:** 2026-08-21  
**Quality baseline:** V14-validated requirement-packet / injection logic (see `data/experiments/rag-release-pipeline-v14/`)  
**Do not deploy from this document alone — stop at deployment review.**

Evaluation and Web Serving use the **same** canonical runtime:

```
POST /eval/run  ──┐
                  ├──→ CanonicalRagRuntime.query(...)
POST /rag/query ──┘
```

Legacy `RagService` (dense Top-5 + extractive) remains in-tree for historical tests only and is **not** used by `POST /rag/query` or `POST /eval/run`.

Production-serving architecture integrated and validated locally; ready for deployment review.

---

## 1. Ingestion architecture

```
Docs (Markdown / PDF / …)
  → loaders → CanonicalDocument
  → metadata (tenant, visibility/ACL, region, version, is_active, effective_*)
  → FixedTokenChunker
  → Embedding provider
  → pgvector Chunk rows  +  BM25 over the SAME chunks
```

Stable `chunk_id` identities are shared by dense and lexical indexes. There is no second chunk set for BM25.

Production embedding expectation: `text-embedding-3-small`. Local/test may use hashing only when `allow_hashing_embeddings=true`. Silent hashing fallback in production mode is rejected.

---

## 2. Query architecture

```
User Query
↓
Auth / Tenant / Region (Principal + ACL)
↓
Question + Temporal Plan
↓
Dense20 + BM25 20
↓
RRF (k=60)
↓
Candidate ≤30
↓
Cross-Encoder (ms-marco-MiniLM-L6-v2)
↓
Top-15 Internal Pool
↓
Version / Auth Validation
↓
Atomic Requirements (Frozen Question Plan)
↓
Canonical Evidence Mapping
↓
Requirement Packets
↓
Deterministic Support?
   ├─ YES → Deterministic Verified Answer
   └─ NO  → Luna (Evidence Verifier)
                 ↓
           ambiguous / suspicious abstention?
                 ↓
                Sol (escalation only)
↓
Local Validation
↓
Completeness Gate
↓
deterministic-requirement-assembler-v3
↓
Answer + Citations
OR Safe Abstention
```

Final-generation LLM calls: **0**.

---

## 3. Exact Top-K meanings

| Knob | Value | Meaning |
|---|---:|---|
| `dense_top_k` | 20 | Dense branch candidate depth |
| `bm25_top_k` | 20 | BM25 branch candidate depth |
| `rrf_k` | 60 | Reciprocal rank fusion constant |
| `fused_candidate_cap` | 30 | Max union candidates entering CE |
| `cross_encoder_top_k` | 15 | Internal evidence pool after CE |

These match the executing validated research path (`retrieve_trace` / hybrid_reranker_benchmark constants). Top-15 is **not** “send 15 chunks blindly to Luna”; Luna receives requirement-scoped packets.

**Historical note:** Some frozen V12 manifests still contain prose like `dense(50)+BM25(50)→union(100)`. That text is immutable historical artifact language. The **current canonical executing configuration** is **20 / 20 / 60 / 30 / 15** only.

---

## 4. Authorization

Enforced inside the runtime (not the UI):

- tenant match
- ACL / visibility
- region (via document metadata on version candidates)
- document / version identity on mappings and citations

Unauthorized evidence cannot become requirement evidence, verifier input, assembler input, or citations.

---

## 5. Temporal / version handling

Modes: `CURRENT_ONLY`, `HISTORICAL_ONLY`, `CROSS_VERSION`, `UNSPECIFIED_CURRENT_DEFAULT`.

Explicit multi-version requests resolve to `VERSION_SET_RESOLVED` (not forced ambiguity). Single resolved version uses `VERSION_RESOLVED`. Version identity is preserved through retrieval → mapping → packets → validation → assembly → citations.

---

## 6. Deterministic support

When every atomic requirement is completely and uniquely grounded with authorized, version-valid, literal, cited evidence, the runtime routes to `DETERMINISTIC` and skips Luna/Sol.

---

## 7. Luna role

Luna is an **Evidence Verifier** (`FrozenEvidenceVerifier`), not a free-form answer generator. It evaluates the frozen Question Plan against scoped packets.

---

## 8. Sol role

Sol is **escalation-only** (suspicious Luna abstention, uncertainty, unresolved conflict). Request-scoped guards bound Luna/Sol calls per user request (no cross-request leakage; no evaluation-stage global caps inside serving).

---

## 9. Completeness gate / assembler

Assembler id: `deterministic-requirement-assembler-v3`.

Invariant (conceptual):

required requirements = verified = assembled = cited

Cross-version: required version-scoped units = assembled version-scoped units. Fail closed if incomplete.

---

## 10. Abstention

Insufficient support → structured safe abstention (`status=abstain`). Prompt-injection questions abstain via `question_injection_guard_v2` before retrieval generation.

---

## 11. Evaluation–serving parity

Parity harness compares research `retrieve_trace` depths and serving `ProductionRagConfig` / `retrieve_evidence_pool`. Artifact:

`data/experiments/production-serving-promotion/research_serving_parity.json`

---

## 12. Web API

```
Browser → apps/web/app/chat → POST /rag/query
  → CanonicalRagRuntime.query(...)
  → { status, answer, citations, requirements, route, request_id }
```

Debug traces require admin debug mode.

---

## Related docs

- [PRODUCTION_RAG_PROMOTION_AUDIT.md](PRODUCTION_RAG_PROMOTION_AUDIT.md)
- [CURRENT_SERVING_RAG_ARCHITECTURE.md](CURRENT_SERVING_RAG_ARCHITECTURE.md) (pre-promotion baseline)
- [TARGET_PRODUCTION_RAG_ARCHITECTURE.md](TARGET_PRODUCTION_RAG_ARCHITECTURE.md)
