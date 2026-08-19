# RAG - Dense Retrieval

Dense retrieval は embedding ベクトルのコサイン類似度で chunk を順位付けする手法。

## Measured（semantic embeddings）

`text-embedding-3-small`（64 次元）を `acmeai-eval-v1`（100 cases）で評価:

| Metric | Hashing → Semantic |
|---|---:|
| Recall@5 | 0.740000 → 0.800000 |
| MRR | +0.060500 |
| nDCG | +0.073905 |
| Multi-document Recall@5 | 0.750000 → 1.000000 |

Decision: **Accepted** for quality work. Hashing remains CI/offline default.

## Configuration（V2）

- Provider: `openai-compatible`
- Model: `text-embedding-3-small`
- Top-20 before reranking
- Score threshold: 0.28（calibrated）

## Related

- [[RAG - Embeddings]]
- [[RAG - Hybrid Retrieval]]
- [[RAG - Retrieval Pipeline]]
- [`BENCHMARK.md`](../../../BENCHMARK.md) — Hashing vs Semantic Embeddings
