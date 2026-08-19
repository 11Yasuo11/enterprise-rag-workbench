# RAG - Abstention Strategy

Evidence が不十分な場合、回答せず safe abstention する fail-closed 設計。

## Outcome Types

| Outcome | Meaning |
|---|---|
| Correct answer | Sufficient evidence + valid generation |
| Correct abstention | Should abstain + did abstain |
| Incorrect abstention | Should answer + abstained |
| Unsupported answer | Answered without sufficient evidence |

## V2 Final Benchmark

| Outcome | Count |
|---|---:|
| Correct answers | 60 |
| Correct abstentions | 9 |
| Incorrect abstentions | 31 |
| Unsupported answers | 0 |

Precision 1.000000 — system prefers incorrect abstention over unsupported answer.

## Related

- [[RAG - Evidence Sufficiency Gate]]
- [[RAG - Safety Pipeline]]
- [[RAG V2 - Failure Analysis]]
