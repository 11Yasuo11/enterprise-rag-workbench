# RAG V2 - Post-V2 Quality AB

Post-v2 controlled A/B research. **Research artifacts only — no architecture change accepted.**

Final recommendation: keep current v2.

## Methods Tested

| Method | Status |
|---|---|
| Corpus scale stress | MEASURED — future V3 concern |
| Lexical rewrite | REJECTED PROXY |
| Multi-query BM25 | REJECTED PROXY |
| Template HyDE | REJECTED PROXY |
| Contextual sparse lite | REJECTED PROXY |
| 3-way sparse fusion | REJECTED PROXY |
| Sentence MaxSim | REJECTED PROXY |
| Query decomposition | REJECTED PROXY |
| LLM rewrite | NOT_EXECUTED |
| Embedding-based HyDE | NOT_EXECUTED |
| Neural SPLADE | NOT_EXECUTED |
| Real ColBERT | NOT_EXECUTED |
| GraphRAG | NOT NEEDED / INCONCLUSIVE |

## Corpus Scale Stress

| Scale | Recall@20 |
|---|---:|
| Current research corpus | 0.994505 |
| 10k synthetic hard negatives | 0.822511 |
| 500k synthetic hard negatives | 0.826840 |

V3 topic: `RETRIEVAL_SCALABILITY_UNDER_HARD_NEGATIVES`.

## Artifacts

- `data/experiments/v2-quality-ab/`
- `src/rag_workbench/experiments/v2_quality_ab*.py`

## Related

- [[RAG - Graph RAG]]
- [[RAG V3 - Generate Verify Recovery]]
- [`docs/EXPERIMENTS.md`](../../../docs/EXPERIMENTS.md) — Section 13
