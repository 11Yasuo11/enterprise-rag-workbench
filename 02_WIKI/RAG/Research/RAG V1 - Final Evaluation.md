# RAG V1 - Final Evaluation

Enterprise RAG Workbench v1 — **immutable frozen record**.

## Identity

| Field | Value |
|---|---|
| Architecture | `enterprise-rag-workbench-v1` |
| Architecture hash | `e76aa8834d995c03d47d6d80c9917cc9e16e4f31b24dcd742250329239a23c85` |
| Retriever | `HYBRID_CROSS_ENCODER_RERANK` |
| Judge | `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1` |
| Dataset | `acmeai-enterprise-rag-v1-final-eval` (80 unseen cases) |

## Final Metrics

| Metric | Value |
|---|---:|
| Correct answers | 24 |
| Correct abstentions | 11 |
| Incorrect abstentions | 45 |
| Unsupported answers | 0 |
| F1 | 0.516129 |
| Candidate-pool coverage | 1.000000 |
| Top-5 coverage | 0.753623 |

V1 historical rows, datasets, metrics, and hash were **not rewritten**.

## Related

- [[RAG V2 - Final Evaluation]] — Descriptive comparison only（different datasets）
- [[RAG - Production Architecture V2]]
- [`BENCHMARK.md`](../../../BENCHMARK.md) — Enterprise RAG Workbench v1 section
