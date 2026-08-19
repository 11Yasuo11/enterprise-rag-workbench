# RAG - Retrieval Pipeline

Authorized scope 内で Dense + BM25 → RRF により candidate pool を構築する。

## Stages

1. **Scope** — Tenant, ACL, active-version SQL predicates before ranking
2. **Dense** — `text-embedding-3-small`, Top-20, threshold 0.28
3. **BM25** — Okapi v1, Top-20, same scope
4. **RRF** — k=60, chunk_id union ≤ 30
5. **Output** — Candidate pool → Cross-Encoder

## Production

`HYBRID_CROSS_ENCODER_RERANK`

## Measured Funnel（V2 final）

- Candidate Pool All Required Coverage: 0.989011
- Top-5 All Required Coverage: 0.857143

2 candidate-generation misses remain.

## Related

- [[RAG - Dense Retrieval]]
- [[RAG - BM25 Sparse Retrieval]]
- [[RAG - Hybrid Retrieval]]
- [[RAG - RRF]]
- [[RAG - Production Architecture V2]]
