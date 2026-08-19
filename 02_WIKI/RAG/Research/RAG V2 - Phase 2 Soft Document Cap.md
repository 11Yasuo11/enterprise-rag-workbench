# RAG V2 - Phase 2 Soft Document Cap

Max-two-chunks-per-document Top-5 occupancy cap experiment.

## Hypothesis

Max-2 cap removes pathological crowding while keeping two-chunk same-document evidence.

## Result — REJECTED

| Metric | Delta |
|---|---:|
| Coverage@5 | 0.000000 |
| Three-document Coverage@5 | 0.000000 |
| Required recall | 0.000000 |

Control A never placed 3+ chunks from one document → B selected identical Top-5 sets. Rescues 0, regressions 0.

## Decision

Reject. **Ranking research frozen for the v2 cycle.**

## Related

- [[RAG V2 - Research Overview]]
- [[RAG V2 - Phase 1 Document Diversity]]
- [`BENCHMARK.md`](../../../BENCHMARK.md) — V2 Phase 2 section
