# RAG - Embeddings

Chunk と query をベクトル空間に写像し、類似度検索の基盤とする。

## Providers

| Provider | Model | Role |
|---|---|---|
| Hashing | `local-hashing-64` | Offline / CI baseline |
| OpenAI-compatible | `text-embedding-3-small` (64 dim) | Quality work / Production |

## Measured Impact

Semantic embeddings on `acmeai-eval-v1`: Recall@5 0.740000 → 0.800000. Hashing remains for deterministic CI.

## Related

- [[RAG - Dense Retrieval]]
- [[RAG - Chunking]]
- [`BENCHMARK.md`](../../../BENCHMARK.md) — Hashing vs Semantic Embeddings
