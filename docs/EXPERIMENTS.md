# Experiments

Measured chronology. Official v1 and v2 architecture rows were not rewritten
after these results. Full traces live in [BENCHMARK.md](../BENCHMARK.md).

Controlled experiments change one independent variable on a sealed dataset.
Final V1 vs V2 numbers use different unseen datasets and are descriptive only.

## 1. Hashing baseline

Local hashing embeddings, deterministic extractive generation, `acmeai-eval-v1`
(100 cases). Recall@5 0.740000, abstention F1 0.666667, ACL safety 1.000000.
This is the offline engineering baseline, not the semantic quality path.

## 2. Hashing → Semantic embeddings — CONTROLLED

| | |
|---|---|
| Hypothesis | Semantic embeddings improve retrieval over local hashing without changing chunking, Top-K, threshold, or generator |
| Independent variable | Embedding provider / model |
| Baseline | `HashingEmbeddingProvider` / `local-hashing-64` |
| Candidate | `text-embedding-3-small`, 64 dimensions |
| Dataset | `acmeai-eval-v1` (100 cases, same corpus) |
| Result | Recall@5 0.740000 → 0.800000; MRR +0.060500; nDCG +0.073905; multi-document Recall@5 0.750000 → 1.000000; ACL 1.000000 |
| Decision | Adopt semantic embeddings for quality work |

## 3. Semantic threshold calibration

Preserved `semantic-baseline-v1`. Calibrated the cosine threshold on cached
query embeddings. Selected 0.28 for later dense retrieval. No document
re-embedding.

## 4. Evidence Sufficiency Gate

Provider-independent gate with schema validation and fail-closed supporting IDs.
Structural abstention became a first-class outcome. Semantic answer-correctness
judging remains unimplemented.

## 5. Qwen3-8B vs GPT-5.6 Luna — CONTROLLED (judge provider)

Local Qwen versus hosted Luna on the same retrieval traces. Luna became the
hosted sufficiency judge for subsequent work. Qwen remains available as a local
adapter and was not the v1/v2 production Judge.

## 6. Evidence Coverage v2 — REJECTED

| | |
|---|---|
| Hypothesis | Requirement-level coverage judging recovers sufficiency false negatives |
| Independent variable | Judge prompt / schema (`evidence-sufficiency-v1` vs `evidence-coverage-v2`) |
| Baseline | Sufficiency v1 + Luna |
| Candidate | Coverage v2 + Luna |
| Dataset | `acmeai-evidence-judge-e2e-eval-v1` (60 cases) |
| Result | Correct answers 37 → 33; incorrect abstentions 15 → 19; five regressions, one rescue |
| Decision | Reject Coverage v2; keep Evidence Sufficiency v1 |

## 7. Dense → Cross-Encoder — CONTROLLED

| | |
|---|---|
| Hypothesis | Local Cross-Encoder ordering improves Top-5 evidence coverage of the same Dense Top-20 |
| Independent variable | Final ranking of authorized Dense Top-20 |
| Baseline | Dense cosine order, Top-5 |
| Candidate | `cross-encoder/ms-marco-MiniLM-L6-v2` descending logits, Top-5 |
| Dataset | `acmeai-reranking-eval-v1` (60 cases) |
| Result | All Required Evidence Coverage@5 0.777778 → 0.972222; three-document Coverage@5 +0.454545 |
| Decision | Adopt `DENSE_CROSS_ENCODER_RERANK` |

A later end-to-end confirmation on a new sealed set kept the reranker:
coverage@5 0.769231 → 0.884615, eight additional correct answers, unsupported
answers unchanged at 0.

## 8. Luna → Sol Judge — CONTROLLED

| | |
|---|---|
| Hypothesis | Sol reduces retrieval-complete sufficiency false negatives without increasing unsupported answers |
| Independent variable | Judge model identity |
| Baseline | `gpt-5.6-luna` / `evidence-sufficiency-v1` |
| Candidate | `gpt-5.6-sol` / identical prompt, schema, and request settings |
| Dataset | `acmeai-sol-judge-e2e-eval-v1` (60 cases) |
| Result | Correct answers 28 → 36; F1 0.700000 → 0.818182; retrieval-complete recall 0.651163 → 0.837209; unsupported 0 → 0 |
| Decision | Adopt `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1` |

## 9. Dense vs BM25 vs Hybrid, then Hybrid+Cross-Encoder

The first retrieval-only Hybrid calibration missed a +0.10 coverage materiality
bar (gain +0.083333) and locked Dense before holdout.

A later Dense+CE vs Hybrid+CE calibration also missed its bar (coverage@5
+0.078947). That experiment's sealed holdout then showed a larger Hybrid effect
(coverage@5 0.722222 → 1.000000) and was not allowed to rewrite the original
rule.

Final v1 replication on a new 80-case dataset, with a new pre-registered rule:

| Metric | Dense+CE | Hybrid+CE |
|---|---:|---:|
| Coverage@5 | 0.797297 | 0.932432 |
| Three-document coverage@5 | 0.650000 | 0.875000 |
| Correct answers | 34 | 41 |
| Unsupported answers | 0 | 0 |

Decision: adopt `HYBRID_CROSS_ENCODER_RERANK` as the v1 retriever.

## 10. Enterprise RAG Workbench v1 final

Frozen architecture hash
`e76aa8834d995c03d47d6d80c9917cc9e16e4f31b24dcd742250329239a23c85`.
Dataset `acmeai-enterprise-rag-v1-final-eval` (80 unseen cases).

| Metric | Value |
|---|---:|
| Correct answers | 24 |
| Correct abstentions | 11 |
| Incorrect abstentions | 45 |
| Unsupported answers | 0 |
| F1 | 0.516129 |
| Candidate-pool coverage | 1.000000 |
| Top-5 coverage | 0.753623 |

V1 is immutable. Historical rows, datasets, metrics, Judge cache, and this
hash were not rewritten.

## 11. V2 research ranking / Judge candidates — REJECTED

V2 research used architecture `enterprise-rag-workbench-v2-research`. Frozen v1
was not modified. Selected ranking stayed `POINTWISE_CROSS_ENCODER_TOP5`.
Selected Judge stayed `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`.

### 11a. Hard one-chunk-per-document diversity — REJECTED

| | |
|---|---|
| Hypothesis | Same-document Cross-Encoder crowding is the three-document coverage bottleneck; keep one chunk per document |
| Baseline | `POINTWISE_CROSS_ENCODER_TOP5` |
| Independent variable | Final Top-5 set construction |
| Dataset | `acmeai-v2-document-diversity-eval-v1` (80 cases) |
| Metrics | Three-document Coverage@5 +0.055556 (bar +0.15); all-required Coverage@5 +0.026316 (bar +0.08); same-document multi-chunk coverage 1.000000 → 0.000000 |
| Result | 2 crowding rescues vs 7 A-complete → B-incomplete regressions |
| Decision | Reject. Pointwise Top-5 remains selected |
| Failure analysis | Cap=1 destroyed legitimate two-chunk same-document evidence. Remaining three-document misses sat at Cross-Encoder ranks 7–8 |

### 11b. Max-two-chunks-per-document — REJECTED

| | |
|---|---|
| Hypothesis | A max-2 cap removes pathological crowding while keeping two-chunk same-document evidence |
| Baseline | `POINTWISE_CROSS_ENCODER_TOP5` |
| Independent variable | Final Top-5 occupancy cap |
| Dataset | `acmeai-v2-soft-document-cap-eval-v1` (80 cases) |
| Metrics | Coverage@5, three-document Coverage@5, required recall: all Δ 0.000000 |
| Result | Control A never placed 3+ chunks from one document, so B selected identical Top-5 sets. Rescues 0, regressions 0 |
| Decision | Reject. Ranking research frozen for the v2 cycle |
| Failure analysis | Remaining pool-complete / Top-5-incomplete misses were Cross-Encoder ranks 6–10+, not occupancy |

### 11c. Evidence Sufficiency prompt v2 — REJECTED

| | |
|---|---|
| Hypothesis | Prompt-only `evidence-sufficiency-v2` recovers Sol false negatives |
| Baseline | `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1` |
| Independent variable | Prompt text / version |
| Dataset | `acmeai-v2-sufficiency-fn-eval-v1` (80 cases) |
| Metrics | Retrieval-complete Judge recall 0.741379 → 0.741379; precision 1.000000 both arms |
| Result | 3 FN rescues and 3 TP→FN regressions; recall gain 0.000000 (bar +0.12 or ≥8 rescues) |
| Decision | Reject. Keep `evidence-sufficiency-v1` |
| Failure analysis | `EVIDENCE_GATE_FALSE_NEGATIVE` remained the largest measured problem after frozen pointwise Top-5 |

Phase 4 reliability work added `deterministic-extractive-v1.1`, bounded
transport retry, and typed `GENERATION_FAILURE` without changing retrieval or
Judge quality policy.

## 12. Enterprise RAG Workbench v2 final

Architecture hash
`964a639905502159e9e66d1b3b1bbfe7013a7f6e8cbc73a33d8412a91419a5da`.
Dataset `acmeai-enterprise-rag-v2-final-eval` (100 unseen cases). One-shot.

| Metric | Value |
|---|---:|
| Accuracy | 0.690000 |
| Precision | 1.000000 |
| Recall | 0.659341 |
| F1 | 0.794702 |
| Correct answers | 60 |
| Correct abstentions | 9 |
| Incorrect abstentions | 31 |
| Unsupported answers | 0 |
| Top-5 evidence coverage | 0.857143 |
| Candidate-pool coverage | 0.989011 |

Primary failures: Judge FN 17, Cross-Encoder failed to promote 7, Cross-Encoder
demoted required evidence 5, candidate-generation miss 2.

The final V1 and V2 benchmarks used different unseen datasets. The comparison
is descriptive, not a controlled paired A/B experiment.

## 13. Post-v2 quality A/B research

Research artifacts only. They do not replace frozen v1 or v2:

- `src/rag_workbench/experiments/v2_quality_ab_core.py`
- `src/rag_workbench/experiments/v2_quality_ab.py`
- `src/rag_workbench/experiments/v2_quality_ab_report.py`
- `scripts/run_v2_quality_ab.py`
- `migrations/versions/0021_v2_quality_ab.py` (additive lock table)
- `data/experiments/v2-quality-ab/`

Accepted changes: none. Final recommendation: keep current v2.

New embeddings = 0. New Sol calls = 0. New external reranker calls = 0.

| Method | Status | Notes |
|---|---|---|
| Corpus scale stress | MEASURED — future V3 concern | SYNTHETIC / STRESS TEST. Gold Recall@20 0.994505; 10k 0.822511 / 27 misses; 500k 0.826840, dense p95 81.99 ms, BM25 p95 1426.34 ms |
| Lexical rewrite | REJECTED PROXY | Deterministic synonym rewrite, not an LLM rewriter |
| Multi-query BM25 | REJECTED PROXY | Local multi-query expansion |
| Template HyDE | REJECTED PROXY | Template hypothetical document; embedding HyDE was NOT_EXECUTED |
| Contextual sparse lite | REJECTED PROXY | Identifier-weighted TF-IDF, not neural SPLADE |
| 3-way sparse fusion | REJECTED PROXY | Dense + BM25 + lite sparse |
| Sentence MaxSim | REJECTED PROXY | Local sentence vectors, not ColBERT weights |
| Query decomposition | REJECTED PROXY | Applied only to two- and three-document questions |
| LLM rewrite | NOT_EXECUTED | Paid query-generation ceiling not authorized |
| Embedding-based HyDE | NOT_EXECUTED | Query-embedding ceiling already consumed |
| Neural SPLADE | NOT_EXECUTED | Model download / infrastructure not authorized |
| Real ColBERT | NOT_EXECUTED | Real late-interaction weights not executed |
| GraphRAG | NOT NEEDED BY CURRENT FAILURE DATA | Official v2 losses are Judge FN and ranking, not graph retrieval. Isolated census was INCONCLUSIVE on multi-hop volume; no GraphRAG implementation was run |

`REJECTED PROXY` means a cheap stand-in was measured and missed the frozen
materiality bar. `NOT_EXECUTED` means the full method was not empirically
tested.

V3 topic from the stress test: `RETRIEVAL_SCALABILITY_UNDER_HARD_NEGATIVES`.

Generate → Verify recovery code under `src/rag_workbench/recovery/` and migration
`0022` is additive research scaffolding. It is not on the official v2 query path,
was not executed in this release-hardening phase, and is `FUTURE_RESEARCH`.

## 14. V3 Phase 2 — recovery safety research

Research artifacts only. Official v2 remains immutable. Architecture
`enterprise-rag-workbench-v3-research`, `production=false`.

Independent variable order is frozen in
`src/rag_workbench/experiments/v3_phase2_safety.py` as `SELECTION_POLICY`.
The Phase-1 Generate→Verify baseline (`generate-verify-draft-v1` /
`generate-verify-claim-verifier-v1`) is not retuned.

Experiment 1 adds a deterministic untrusted-instruction boundary after claim
verification and completeness. It uses document trust metadata, sentence
speech-act structure, and question illocution. It is not a naive keyword
blocker.

V3 files live under `data/v3_research_corpus/` and are ingested only for corpus
version `acmeai-v0.1-v3-research-extension`. They are not mixed into
`data/synthetic_company`, so frozen V1/V2 hashing and index identity stay intact.

Safety validation dataset `acmeai-v3-recovery-safety-validation-v1` (60 cases)
is a candidate-selection set, not the final unseen V3 benchmark.
