# RAG V2 - Research Overview

V2 research cycle の全体像。Architecture `enterprise-rag-workbench-v2-research`。

## Phases

| Phase | Topic | Decision |
|---|---|---|
| Phase 1 | Document-diversified Top-5 ranking | REJECTED |
| Phase 2 | Soft document-cap Top-5 ranking | REJECTED |
| Phase 3 | Evidence Sufficiency FN reduction | REJECTED |
| Phase 4 | Generation & provider reliability | Accepted（operational only） |
| Final | Frozen end-to-end benchmark | V2 release |

## Frozen Selections

- Ranking: `POINTWISE_CROSS_ENCODER_TOP5`
- Judge: `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`

## Related

- [[RAG V2 - Phase 1 Document Diversity]]
- [[RAG V2 - Phase 2 Soft Document Cap]]
- [[RAG V2 - Phase 3 Sufficiency FN Reduction]]
- [[RAG V2 - Phase 4 Reliability]]
- [[RAG V2 - Final Evaluation]]
- [[RAG V2 - Failure Analysis]]
- [[RAG V2 - Post-V2 Quality AB]]
- [`docs/EXPERIMENTS.md`](../../../docs/EXPERIMENTS.md)
