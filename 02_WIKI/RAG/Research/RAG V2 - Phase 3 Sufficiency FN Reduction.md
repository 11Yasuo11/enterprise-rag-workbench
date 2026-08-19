# RAG V2 - Phase 3 Sufficiency FN Reduction

Evidence Sufficiency prompt v2 experiment.

## Hypothesis

Prompt-only `evidence-sufficiency-v2` recovers Sol false negatives.

## Result — REJECTED

| Metric | v1 → v2 |
|---|---:|
| Retrieval-complete Judge recall | 0.741379 → 0.741379 |
| Precision | 1.000000 both |

3 FN rescues and 3 TP→FN regressions. Recall gain 0.000000（bar +0.12 or ≥8 rescues）.

## Decision

Reject. Keep `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`. **Judge research frozen for v2 cycle.**

`EVIDENCE_GATE_FALSE_NEGATIVE` remains largest measured problem.

## Related

- [[RAG - Evidence Sufficiency Gate]]
- [[RAG V2 - Research Overview]]
- [`BENCHMARK.md`](../../../BENCHMARK.md) — V2 Phase 3 section
