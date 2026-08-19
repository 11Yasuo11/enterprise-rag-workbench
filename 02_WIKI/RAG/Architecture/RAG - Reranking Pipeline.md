# RAG - Reranking Pipeline

RRF 後の candidate pool を Cross-Encoder で final Top-5 に絞る。

## Stages

1. Input: authorized hybrid union（Dense Top-20 + BM25 Top-20 via RRF）
2. Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L6-v2` descending logits
3. Output: `POINTWISE_CROSS_ENCODER_TOP5`

## V2 Failures

| Failure Mode | Count |
|---|---:|
| Cross-Encoder failed to promote | 7 |
| Cross-Encoder demoted required evidence | 5 |

## Rejected Ranking Research

- Hard one-chunk-per-document — REJECTED（[[RAG V2 - Phase 1 Document Diversity]]）
- Max-two-chunks-per-document — REJECTED（[[RAG V2 - Phase 2 Soft Document Cap]]）

Ranking research **FROZEN_FOR_CURRENT_V2_CYCLE**.

## Related

- [[RAG - Cross Encoder Reranking]]
- [[RAG - Retrieval Pipeline]]
- [[RAG V2 - Failure Analysis]]
