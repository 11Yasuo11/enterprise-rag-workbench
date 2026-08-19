# RAG V2 - Phase 4 Reliability

Generation and provider reliability hardening. **Operational only — no quality policy change.**

## Changes

- `deterministic-extractive-v1.1`
- Bounded transport retry（max 2 physical attempts）
- Typed `GENERATION_FAILURE`
- Support content-identity validation
- Logical vs physical request accounting

## Unchanged

- Retrieval, ranking, Judge, answerability quality policy
- `POINTWISE_CROSS_ENCODER_TOP5`
- `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`

## Related

- [[RAG - Answer Generation]]
- [[RAG V2 - Research Overview]]
- [`BENCHMARK.md`](../../../BENCHMARK.md) — V2 Phase 4 section
