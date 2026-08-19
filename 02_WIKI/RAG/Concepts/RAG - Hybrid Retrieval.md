# RAG - Hybrid Retrieval

Dense Top-20 + BM25 Top-20 を RRF で融合し、Cross-Encoder へ渡す candidate generation path。

## Production Identity

`HYBRID_CROSS_ENCODER_RERANK` — adopted by pre-registered 80-case replication.

| Metric | Dense+CE | Hybrid+CE |
|---|---:|---:|
| Coverage@5 | 0.797297 | 0.932432 |
| Three-document coverage@5 | 0.650000 | 0.875000 |
| Correct answers | 34 | 41 |
| Unsupported answers | 0 | 0 |

## Why Hybrid

Dense retrieval can miss identifier-heavy evidence. BM25 complements dense. RRF combines ranks without treating BM25 scores as cosine similarities.

Earlier calibration missed +0.10 materiality bar; later sealed replication met pre-registered rule.

## Related

- [[RAG - Dense Retrieval]]
- [[RAG - BM25 Sparse Retrieval]]
- [[RAG - RRF]]
- [[RAG - Cross Encoder Reranking]]
- [[RAG - Retrieval Pipeline]]
- [`BENCHMARK.md`](../../../BENCHMARK.md) — Final Hybrid+Cross-Encoder Replication
