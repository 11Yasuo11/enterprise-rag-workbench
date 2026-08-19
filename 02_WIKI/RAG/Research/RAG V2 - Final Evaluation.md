# RAG V2 - Final Evaluation

Enterprise RAG Workbench v2 — **frozen final end-to-end benchmark**.

## Identity

| Field | Value |
|---|---|
| Architecture hash | `964a639905502159e9e66d1b3b1bbfe7013a7f6e8cbc73a33d8412a91419a5da` |
| Dataset | `acmeai-enterprise-rag-v2-final-eval` (100 unseen cases) |
| One-shot | Yes |

## Final Metrics

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
| Three-document Top-5 coverage | 0.666667 |

## V1 vs V2（Descriptive Only）

Different unseen datasets — **not** a controlled paired A/B estimate.

| Metric | V1 (80) | V2 (100) |
|---|---:|---:|
| F1 | 0.516129 | 0.794702 |
| Precision | 1.000000 | 1.000000 |
| Incorrect abstentions | 45 | 31 |
| Unsupported answers | 0 | 0 |

## Related

- [[RAG - Production Architecture V2]]
- [[RAG V2 - Failure Analysis]]
- [`BENCHMARK.md`](../../../BENCHMARK.md) — V2 Final section
- [`README.md`](../../../README.md)
