# RAG - Cross Encoder Reranking

Candidate pool（authorized union）を Cross-Encoder で再順位付けし、final Top-5 を決定する。

## Production

- Model: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Revision: `233902d25c440f23af6f7d6e94d2946bac0bee0a`
- Strategy: `POINTWISE_CROSS_ENCODER_TOP5`
- Local inference only（external reranker API 不使用）

## Measured

On `acmeai-reranking-eval-v1` (60 cases):

| Metric | Dense → Cross-Encoder |
|---|---:|
| All Required Evidence Coverage@5 | 0.777778 → 0.972222 |
| Three-document Coverage@5 | +0.454545 |

Decision: **Accepted** `DENSE_CROSS_ENCODER_RERANK` → later hybrid path.

## V2 Remaining Failures

- Cross-Encoder failed to promote: 7
- Cross-Encoder demoted required evidence: 5

## Related

- [[RAG - Reranking Pipeline]]
- [[RAG - Hybrid Retrieval]]
- [[RAG V2 - Failure Analysis]]
- [`BENCHMARK.md`](../../../BENCHMARK.md) — Dense Cross-Encoder Reranking Benchmark
