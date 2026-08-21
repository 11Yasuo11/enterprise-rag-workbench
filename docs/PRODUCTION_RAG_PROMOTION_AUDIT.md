# Production RAG Promotion Audit

**Audit date:** 2026-08-21  
**HEAD:** `6e9741f` (`chore/repository-hygiene-cleanup`) + uncommitted finalization work  
**Latest validated research orchestration:** RAG release pipeline **V13** (builds on V4–V12; no V14)  
**Promotion principle:** one `CanonicalRagRuntime` shared by Web serving and evaluation  
**Finalization:** see `data/experiments/production-finalization/` — semantic corpus validated, `/eval/run` migrated, real CE + provider smokes recorded.

---

## 0. Post-finalization status

| Item | Status |
|---|---|
| Web `/rag/query` | `CanonicalRagRuntime` |
| Eval `/eval/run` | `CanonicalRagRuntime` (migrated) |
| Semantic index `e598d9fb…` | Present; 24/24 chunks match BM25 identity set |
| Real Cross-Encoder | Loaded in production mode |
| Real OpenAI embedding / Luna | Smoke recorded |
| Sol | Escalation-only; not manufactured when router does not trigger |
| Deployed externally | **No** — ready for deployment review only |

**Historical Top-K note:** V12 freeze prose may say dense50/BM25 50/union 100. Executing config remains **20/20/60/30/15**.

## 1. Component matrix

| Component | Current Web Serving | Latest Research | Latest Validated | Target Serving | Action |
|---|---|---|---|---|---|
| Document parsing | `ingestion/loaders` → `CanonicalDocument` | same loaders | same | same | **Reuse** |
| Chunking | `FixedTokenChunker` (180/30 default) | same strategy; research index identity frozen | frozen `INDEX_IDENTITY` for semantic index | shared chunker + explicit index identity | **Reuse + config** |
| Metadata | tenant, visibility, version, is_active, effective_at | + region in metadata; version resolver | V12 versioning contract | require tenant/ACL/region/version fields for evidence | **Enrich / enforce** |
| Embeddings | **hashing / local-hashing-64** default | **text-embedding-3-small** (dim 64 in research index) | V12 freeze: `text-embedding-3-small` | `RAG_EMBEDDING_PROVIDER=openai` + model; hashing only for local/test | **Promote + fail-closed** |
| Vector index | pgvector `Chunk.embedding` | same table, research identity | `INDEX_IDENTITY` in luna identities | same chunks for dense+BM25 | **Reuse** |
| BM25 | **not called** | `BM25Retriever` on same chunks | used in `retrieve_trace` | required | **Wire into runtime** |
| Dense retrieval | Top-**5**, threshold 0.2 | Top-**20**, threshold 0.28 | `DENSE_DEPTH=20` in hybrid_reranker_benchmark | dense_top_k=20 | **Promote** |
| RRF | **not called** | `reciprocal_rank_fusion`, k=60 | k=60, union capped | rrf_k=60, fused_candidate_cap=30 | **Promote** |
| Cross-Encoder | **not called** | `ms-marco-MiniLM-L6-v2` | same model; pool Top-15 | CE required; top_k=15 internal pool | **Promote** |
| Top-K knobs | ambiguous `top_k=5` | explicit depths in research | see §2 | explicit ProductionRagConfig | **Replace** |
| ACL | dense ACL filter | ACL before CE/packets | safety gates pass | hard runtime invariant | **Keep + strengthen** |
| Tenant | principal.tenant_id | same | same | same | **Keep** |
| Region | metadata only | citation/version checks | V12 safety | enforce on evidence mapping | **Promote checks** |
| Temporal scope | `plan_temporal_scope` on dense | same planner | CURRENT/HISTORICAL/CROSS/UNSPECIFIED | same planner in runtime | **Promote** |
| Version resolver | active-version filter only | `DeterministicVersionResolver` + VERSION_SET_RESOLVED | V12 contract | promote resolver | **Promote** |
| Question Plan | none | `decompose_question` / `FrozenQuestionPlan` | frozen baselines V4–V12 | same planner | **Promote** |
| Atomic Requirement Contract | none | `atomic_requirement_contract_v1` | used by V4–V12 | same | **Promote** |
| CanonicalEvidenceMapping | none | `evidence_mapping.py` | schema hashed in V12 freeze | same | **Promote** |
| Requirement evidence packets | none | packets in V4 runner / adaptive recovery | validated in release scripts | same | **Promote** |
| Deterministic support | none | `deterministic_support_complete` | V8+ tests + V12 | DETERMINISTIC_VERIFIED_ANSWER path | **Promote** |
| Luna verification | none (optional different answerability gate) | `FrozenEvidenceVerifier` / Luna | V4–V12 | Evidence Verifier only | **Promote** |
| Sol escalation | optional judge gate (not selective) | SelectiveRiskRouter after Luna | V4–V12 | escalation-only | **Promote** |
| Router | none | `SelectiveRiskRouter` | V4–V12 | same | **Promote** |
| Local validation | gate validation if enabled | `validate_verifier_result` + auth | V4–V12 | always | **Promote** |
| Completeness gate | none | universal completeness + assembler checks | V11 cardinality + V12 | universal gate | **Promote** |
| Assembler | extractive / OpenAI LLM | `deterministic-requirement-assembler-v3` | assembler-v3 | final gen LLM disabled | **Promote** |
| Citations | extractive citation builder | assembler + auth | citation_authorization | authorized citations only | **Promote** |
| Safe abstention | gate fail / empty context | SAFE_ABSTAIN routes | safety reports | structured abstain | **Promote** |
| Prompt-injection | **not on serving path** | `question_injection_guard_v2` | used in research pipelines | precheck in runtime | **Promote** |
| Model-call / cost guard | process-level judge limits | `StageResourceGuard` | V9 stage guard | **request-scoped** guards | **Promote (adapt)** |
| Logging / tracing | RagRun latencies | research ledgers | experiment artifacts | structured production trace | **Add** |

---

## 2. Validated retrieval Top-K (do not guess)

**Actual code path used by V4–V12 offline/online evaluation** (`safe_recovery_luna_v2.pipeline.retrieve_trace` + constants from `hybrid_reranker_benchmark`):

| Parameter | Validated value | Symbol |
|---|---:|---|
| Dense depth | **20** | `DENSE_DEPTH` |
| BM25 depth | **20** | `BM25_DEPTH` |
| RRF k | **60** | `RRF_K` |
| Fused candidate cap | **30** | `UNION_LIMIT` |
| Cross-Encoder model | `cross-encoder/ms-marco-MiniLM-L6-v2` | `MODEL_ID` |
| Internal evidence pool | **15** | `TOP_K = 15` in V4 runner (`top20[:15]`) |

**Documentation drift:** V12 freeze manifest text says `dense(50)+BM25(50)→RRF union=100→CE→Top-15`. That string does **not** match the executing `retrieve_trace` constants (20/20/30). **Promotion uses the executing validated values (20/20/60/30/15).**

User brief suggested the same 20/20/60/30/15 stack; no override needed.

---

## 3. Research ↔ serving skew (explicit)

1. **Two inference stacks:** Web uses `RagService`+dense Top-5; evaluation uses hybrid+CE+plan+Luna/Sol+assembler.
2. **Embedding skew:** serving defaults hashing; research uses `text-embedding-3-small` on frozen index identity.
3. **Top-K ambiguity:** serving `top_k=5` vs research multi-stage depths.
4. **No BM25/RRF/CE** on Web path.
5. **No frozen Question Plan** on Web path → no atomic completeness.
6. **No CanonicalEvidenceMapping / packets** on Web path.
7. **Luna/Sol roles inverted/absent:** serving optional answerability gate ≠ Luna Evidence Verifier + selective Sol.
8. **Final answer generation:** serving may call extractive/OpenAI; research assembler has **0** final-generation LLM calls.
9. **Injection guard** not applied on `/rag/query`.
10. **Version-set resolution** (cross-version) exists only in research versioning module.
11. **Observability:** serving RagRun ≠ research stage/request counters.
12. **Eval via API** (`POST /eval/run`) still calls `_rag_service` → **same legacy skew** for API-driven eval.

---

## 4. What should be promoted (canonical)

Promote by **composition into one runtime**, not by copying into a second serving fork:

| Module | Path | Status after promotion |
|---|---|---|
| Hybrid + RRF | `retrieval/hybrid.py`, `retrieval/bm25.py` | canonical retrieval |
| Cross-Encoder | `reranking/cross_encoder.py` | canonical rerank |
| Temporal | `retrieval/temporal.py` | canonical |
| Atomic contract | `experiments/atomic_requirement_contract_v1/*` | canonical (re-exported via runtime) |
| Versioning + router + assembler | `experiments/requirement_assembler_selective_routing_v1/*` | canonical |
| Luna/Sol verifier | `experiments/atomic_requirement_contract_v1/verifier.py` | canonical |
| Stage/request guard pattern | `evaluation/stage_guard.py` | adapted to per-request guards |
| Injection guard | `safety/question_injection_guard_v2.py` | canonical precheck |
| Citation auth | `evaluation/citation_authorization.py` | canonical |

`RagService` remains available as **legacy** for historical tests until callers migrate; active Web `/rag/query` must use `CanonicalRagRuntime`.

---

## 5. Classification summary

| Layer | Label |
|---|---|
| `RagService` dense Top-5 path | **legacy serving** |
| Hybrid/CE/plan/Luna/Sol/assembler stack | **canonical (to promote)** |
| V1–V3 phase experiment scripts | **historical** |
| Ad-hoc experiment arms / rejected candidates | **experimental** |
| V12 freeze artifacts | **historical evidence (do not rewrite)** |

---

## 6. Promotion gates (this engineering phase)

1. Audit docs complete (this file + CURRENT + TARGET).
2. Single `CanonicalRagRuntime.query` used by Web + parity harness.
3. Zero-API unit/integration tests green; Ruff; `git diff --check`.
4. Research–serving parity artifact written.
5. Web/API e2e artifact written.
6. Optional real-API smoke ≤ $0.10.
7. Stop at `READY_FOR_PRODUCTION_DEPLOYMENT_REVIEW` — **no deploy**.
