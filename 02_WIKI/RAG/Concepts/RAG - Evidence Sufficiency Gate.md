# RAG - Evidence Sufficiency Gate

Top-5 evidence が質問に答えうるかを判定する Judge。Fail-closed supporting-ID checks 付き。

## Production Identity

`GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`

- Model: `gpt-5.6-sol`
- Prompt: `evidence-sufficiency-v1`
- Temperature: 0

## Measured（Luna → Sol）

On `acmeai-sol-judge-e2e-eval-v1` (60 cases):

| Metric | Luna → Sol |
|---|---:|
| Correct answers | 28 → 36 |
| F1 | 0.700000 → 0.818182 |
| Retrieval-complete recall | +0.186047 |
| Unsupported answers | 0 → 0 |

## Rejected Candidates

- **Evidence Coverage v2** — Correct 37→33, incorrect abstentions 15→19
- **Evidence Sufficiency prompt v2** — Recall unchanged 0.741379

## V2 Remaining Issue

17 retrieval-complete Evidence Gate false negatives（largest measured bottleneck）

## Related

- [[RAG - Abstention Strategy]]
- [[RAG V2 - Phase 3 Sufficiency FN Reduction]]
- [[RAG V2 - Failure Analysis]]
- [`docs/ENGINEERING_DECISIONS.md`](../../../docs/ENGINEERING_DECISIONS.md)
