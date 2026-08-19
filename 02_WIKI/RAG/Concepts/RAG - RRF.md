# RAG - RRF

Reciprocal Rank Fusion — Dense と BM25 の incomparable scores を rank ベースで融合する。

## Configuration

- `k = 60`（fixed）
- `chunk_id` union ≤ 30
- 0.28 dense threshold は BM25 / RRF scores に適用しない

## Why RRF

Dense cosine scores と BM25 scores は直接比較できない。RRF は rank のみを使い、スコアスケールの違いを回避する。

## Related

- [[RAG - Hybrid Retrieval]]
- [[RAG - Dense Retrieval]]
- [[RAG - BM25 Sparse Retrieval]]
- [`docs/ENGINEERING_DECISIONS.md`](../../../docs/ENGINEERING_DECISIONS.md) — Why RRF
