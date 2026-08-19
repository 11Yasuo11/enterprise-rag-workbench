# RAG V2 - Failure Analysis

Official V2 failed-answerable census（31 incorrect abstentions, 0 unsupported answers）.

## Root Cause Breakdown

```text
Evidence Gate False Negative          17
Cross-Encoder Failed to Promote        7
Cross-Encoder Demoted Evidence         5
Candidate Generation Miss              2
```

## Retrieval Funnel

```text
Candidate Pool All Required Coverage     98.90%
        ↓
Top-5 All Required Coverage              85.71%
        ↓
Retrieval-complete Judge Recall          77.33%
        ↓
Final answerable correct-answer rate     65.93%
```

Stage counts: 91 answerable → 90 candidate-complete → 75 Top-5 complete → 58 Judge-approved → 60 correct final answers.

## Known Limitations

- 17 retrieval-complete Evidence Gate false negatives
- 12 ranking-stage failures
- 2 candidate-generation misses
- Near-duplicate weakness（preferred-source Top-5 success 0.625000）
- Three-document ranking loss
- Large-corpus retrieval degradation（Recall@20 ≈ 0.82 at 10k–500k hard negatives）

## Related

- [[RAG V2 - Final Evaluation]]
- [[RAG - Evidence Sufficiency Gate]]
- [[RAG - Cross Encoder Reranking]]
- [`README.md`](../../../README.md) — Failure analysis section
