# RAG - Evaluation Pipeline

Frozen dataset 上で one-shot benchmark を実行し、結果を永続化する。

## Principles

- Datasets frozen before first inference
- Hidden ground truth never reaches retrieval / Judge / generator
- One independent variable per controlled experiment
- Final benchmarks executed once on unseen datasets
- Token-set overlap guard（0.5 ceiling）against paraphrased prior failures

## Artifacts

| Location | Content |
|---|---|
| `data/eval/` | Frozen evaluation datasets |
| `data/experiments/` | Persisted research artifacts |
| `BENCHMARK.md` | Full measured log |
| `docs/EXPERIMENTS.md` | Chronology |
| `migrations/` | Additive schema history |

## Related

- [[RAG - Evaluation Metrics]]
- [`BENCHMARK.md`](../../../BENCHMARK.md)
- [`docs/EXPERIMENTS.md`](../../../docs/EXPERIMENTS.md)
