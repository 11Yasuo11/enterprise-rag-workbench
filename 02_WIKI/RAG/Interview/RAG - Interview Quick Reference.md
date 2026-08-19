# RAG - Interview Quick Reference

面接・復習用クイックリファレンス。**数値の正本は Research Notes / BENCHMARK へリンク。**

## Elevator Pitch

Enterprise RAG evaluation workbench. Measured quality improvements through controlled one-variable experiments. V2 achieves 100% precision, 0 unsupported answers, but 31 incorrect abstentions on final benchmark.

## Architecture（30 sec）

Dense + BM25 → RRF k=60 → Cross-Encoder Top-5 → Sol Evidence Sufficiency Judge → Extractive generation.

## Key Decisions to Explain

| Decision | Why |
|---|---|
| Semantic over hashing | Recall@5 0.74→0.80 on sealed dataset |
| Hybrid over dense-only | Coverage@5 0.797→0.932 on 80-case replication |
| Cross-Encoder reranking | Coverage@5 0.778→0.972 on reranking dataset |
| Sol over Luna Judge | F1 0.70→0.82, 10 FN rescues, 0 unsupported increase |
| Reject Coverage v2 | 5 regressions vs 1 rescue |
| Fail-closed abstention | Precision 1.0, prefer incorrect abstention over unsupported |

## Metrics to Know

V2 final: Accuracy 69%, Precision 100%, Recall 65.9%, F1 79.5%.

Main bottleneck: 17 Evidence Gate false negatives（retrieval-complete but Judge rejected）.

## Common Interview Questions

**Q: Why not use an LLM for generation?**
A: Frozen V2 uses deterministic extractive to isolate retrieval/Judge quality from hallucination.

**Q: Why BM25 if dense works?**
A: Identifier-heavy queries; hybrid replication showed +0.135 coverage@5.

**Q: What's next (V3)?**
A: Generate→Verify for Judge FN, claim-level verification, large-corpus retrieval.

## Deep Dives

- [[RAG - Hybrid Retrieval]]
- [[RAG - Cross Encoder Reranking]]
- [[RAG - Evidence Sufficiency Gate]]
- [[RAG V2 - Failure Analysis]]
- [[RAG V2 - Final Evaluation]]

## Related

- [[RAG]]
- [`README.md`](../../../README.md)
