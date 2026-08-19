# RAG - Evaluation Metrics

RAG Workbench で使用する retrieval / end-to-end 評価指標。

## Retrieval Metrics

| Metric | Description |
|---|---|
| Recall@K / Hit@K | Required evidence in Top-K |
| MRR | Mean Reciprocal Rank |
| nDCG | Normalized Discounted Cumulative Gain |
| Coverage@5 | All required evidence in Top-5 |

## End-to-End Metrics

| Metric | V2 Final |
|---|---:|
| Accuracy | 0.690000 |
| Precision | 1.000000 |
| Recall | 0.659341 |
| F1 | 0.794702 |

## Safety Metrics

- ACL safety, tenant isolation, version correctness: 1.000000
- Citation validity / correctness: 100%
- Prompt-injection boundary: 100% (4/4)

## Evaluation Principles

- Frozen datasets before first inference
- One independent variable per controlled experiment
- Final benchmarks are one-shot
- Official metrics are versioned（V2 numbers are historical frozen record）

## Related

- [[RAG - Evaluation Pipeline]]
- [`BENCHMARK.md`](../../../BENCHMARK.md)
- [`docs/EXPERIMENTS.md`](../../../docs/EXPERIMENTS.md)
