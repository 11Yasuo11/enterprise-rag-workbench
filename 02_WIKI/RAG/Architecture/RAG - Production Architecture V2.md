# RAG - Production Architecture V2

**Production Source of Truth** — 数値・Architecture identity の正本は [`README.md`](../../../README.md)。

## Identity

| Field | Value |
|---|---|
| Architecture | `enterprise-rag-workbench-v2` |
| Parent | `enterprise-rag-workbench-v1` |
| Architecture hash | `964a639905502159e9e66d1b3b1bbfe7013a7f6e8cbc73a33d8412a91419a5da` |
| Final dataset | `acmeai-enterprise-rag-v2-final-eval` (100 unseen cases) |

## Request Path

```text
Query
  → Authorized tenant / ACL / active-version scope
  → text-embedding-3-small
  → Dense Top-20 + BM25 Top-20
  → RRF k=60
  → Cross-Encoder (ms-marco-MiniLM-L6-v2)
  → POINTWISE_CROSS_ENCODER_TOP5
  → GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1
  → Validated supporting chunks
  → deterministic-extractive-v1.1
  → Answer + Citations  or  Safe Abstention
```

## Component Identities

| Stage | Identity |
|---|---|
| Retriever | `HYBRID_CROSS_ENCODER_RERANK` |
| Ranking | `POINTWISE_CROSS_ENCODER_TOP5` |
| Judge | `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1` |
| Generator | `deterministic-extractive-v1.1` |

## V2 Final Metrics

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

## Pipeline Notes

- [[RAG - Retrieval Pipeline]]
- [[RAG - Reranking Pipeline]]
- [[RAG - Answer Generation]]
- [[RAG - Evaluation Pipeline]]
- [[RAG - Safety Pipeline]]

## Related

- [[RAG]]
- [[RAG V1 - Final Evaluation]] — Immutable parent
- [[RAG V2 - Final Evaluation]] — Frozen benchmark detail
- [`docs/ENGINEERING_DECISIONS.md`](../../../docs/ENGINEERING_DECISIONS.md)
