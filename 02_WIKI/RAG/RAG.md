# RAG

Enterprise RAG Workbench の Knowledge Map。まずここから辿る。

> **Production Source of Truth:** [[RAG - Production Architecture V2]] → リポジトリ正本は [`README.md`](../../README.md)

---

## Start Here

1. [[RAG - Overview]] — RAG 全体像
2. [[RAG - Production Architecture V2]] — 現在の Production（V2）
3. [[RAG - Management Rules]] — 今後のファイル管理ルール
4. 実験 chronology → [`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md)
5. 完全な数値ログ → [`BENCHMARK.md`](../../BENCHMARK.md)

---

## Core Concepts

| Topic | Canonical Note |
|---|---|
| Overview | [[RAG - Overview]] |
| Embeddings | [[RAG - Embeddings]] |
| Chunking | [[RAG - Chunking]] |
| Dense Retrieval | [[RAG - Dense Retrieval]] |
| BM25 / Sparse | [[RAG - BM25 Sparse Retrieval]] |
| Hybrid Search | [[RAG - Hybrid Retrieval]] |
| RRF | [[RAG - RRF]] |
| Reranking | [[RAG - Cross Encoder Reranking]] |
| Evidence Gate | [[RAG - Evidence Sufficiency Gate]] |
| Abstention | [[RAG - Abstention Strategy]] |
| Evaluation | [[RAG - Evaluation Metrics]] |
| Prompt Injection | [[RAG - Prompt Injection Defense]] |
| Agentic RAG | [[RAG - Agentic RAG]] |
| Graph RAG | [[RAG - Graph RAG]] |

---

## Retrieval

- [[RAG - Dense Retrieval]]
- [[RAG - BM25 Sparse Retrieval]]
- [[RAG - Hybrid Retrieval]]
- [[RAG - RRF]]
- [[RAG - Retrieval Pipeline]]

---

## Reranking

- [[RAG - Cross Encoder Reranking]]
- [[RAG - Reranking Pipeline]]

---

## Generation

- [[RAG - Answer Generation]]
- [[RAG - Evidence Sufficiency Gate]]
- [[RAG - Abstention Strategy]]

---

## Evaluation

- [[RAG - Evaluation Metrics]]
- [[RAG - Evaluation Pipeline]]
- 完全ログ: [`BENCHMARK.md`](../../BENCHMARK.md)
- 決定理由: [`docs/ENGINEERING_DECISIONS.md`](../../docs/ENGINEERING_DECISIONS.md)

---

## Safety

- [[RAG - Prompt Injection Defense]]
- [[RAG - Safety Pipeline]]
- [[RAG - Abstention Strategy]]

---

## Agentic RAG

- [[RAG - Agentic RAG]] — Future research（未実装）

---

## Graph RAG

- [[RAG - Graph RAG]] — V2 失敗データ上は Not Needed / Inconclusive

---

## Current Production Architecture

**Version:** V2 (`enterprise-rag-workbench-v2`)

| Component | Identity |
|---|---|
| Architecture hash | `964a639905502159e9e66d1b3b1bbfe7013a7f6e8cbc73a33d8412a91419a5da` |
| Retriever | `HYBRID_CROSS_ENCODER_RERANK` → `POINTWISE_CROSS_ENCODER_TOP5` |
| Judge | `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1` |
| Generator | `deterministic-extractive-v1.1` |
| Final dataset | `acmeai-enterprise-rag-v2-final-eval` (100 cases) |

詳細: [[RAG - Production Architecture V2]]

---

## Research

### V1

- [[RAG V1 - Final Evaluation]] — Immutable frozen record

### V2

- [[RAG V2 - Research Overview]]
- [[RAG V2 - Phase 1 Document Diversity]] — REJECTED
- [[RAG V2 - Phase 2 Soft Document Cap]] — REJECTED
- [[RAG V2 - Phase 3 Sufficiency FN Reduction]] — REJECTED
- [[RAG V2 - Phase 4 Reliability]] — Accepted (operational only)
- [[RAG V2 - Final Evaluation]] — Frozen final benchmark
- [[RAG V2 - Failure Analysis]]
- [[RAG V2 - Post-V2 Quality AB]] — No architecture change accepted

### V3

- [[RAG V3 - Generate Verify Recovery]] — `FUTURE_RESEARCH`

監査一覧: [[RAG - Inventory Audit]]

---

## Known Limitations

V2 公式 benchmark 上の未解決問題（[`README.md`](../../README.md) より）:

- 17 retrieval-complete Evidence Gate false negatives
- 12 ranking-stage failures (7 failed to promote, 5 demoted)
- 2 candidate-generation misses
- Near-duplicate weakness (preferred-source Top-5 success 0.625000)
- Three-document ranking loss (Top-5 coverage 0.666667)
- Large-corpus retrieval degradation (Recall@20 ≈ 0.82 at 10k–500k hard negatives)

詳細: [[RAG V2 - Failure Analysis]]

---

## Interview / Study

- [[RAG - Interview Quick Reference]]

---

## Related

- [`README.md`](../../README.md) — Production Source of Truth
- [`BENCHMARK.md`](../../BENCHMARK.md) — Full measured research log
- [`docs/ENGINEERING_DECISIONS.md`](../../docs/ENGINEERING_DECISIONS.md) — Why decisions were made
- [`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md) — Experiment chronology
- [[RAG - Management Rules]]
