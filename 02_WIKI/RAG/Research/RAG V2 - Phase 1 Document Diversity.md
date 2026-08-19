# RAG V2 - Phase 1 Document Diversity

Hard one-chunk-per-document Top-5 diversification experiment.

## Hypothesis

Same-document Cross-Encoder crowding is the three-document coverage bottleneck.

## Result — REJECTED

| Metric | Delta | Bar |
|---|---:|---:|
| Three-document Coverage@5 | +0.055556 | +0.15 |
| All-required Coverage@5 | +0.026316 | +0.08 |
| Same-document multi-chunk coverage | 1.000000 → 0.000000 | — |

2 crowding rescues vs 7 A-complete → B-incomplete regressions.

Cap=1 destroyed legitimate two-chunk same-document evidence.

## Decision

Reject. `POINTWISE_CROSS_ENCODER_TOP5` remains selected.

## Related

- [[RAG V2 - Research Overview]]
- [[RAG - Cross Encoder Reranking]]
- [`BENCHMARK.md`](../../../BENCHMARK.md) — V2 Phase 1 section
