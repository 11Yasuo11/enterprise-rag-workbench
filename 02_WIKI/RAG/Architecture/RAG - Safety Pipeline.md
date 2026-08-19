# RAG - Safety Pipeline

Retrieval 前後で tenant / ACL / version / injection 境界を enforce する。

## Pre-Retrieval

- Tenant filtering（SQL predicate）
- ACL filtering（permission groups）
- Active-version filtering

## Post-Retrieval

- Judge supporting IDs must be in authorized Top-5
- Content identity validation
- Citation validation
- Prompt-injection boundary（retrieved text = untrusted DATA）

## V2 Final Safety

| Control | Result |
|---|---:|
| ACL safety | 1.000000 |
| Tenant isolation | 1.000000 |
| Version correctness | 1.000000 |
| Citation validity | 100% |
| Prompt-injection boundary | 100% (4/4) |
| Unsupported answers | 0 |

## Related

- [[RAG - Prompt Injection Defense]]
- [[RAG - Abstention Strategy]]
- [[RAG - Production Architecture V2]]
