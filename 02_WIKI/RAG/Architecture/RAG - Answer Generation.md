# RAG - Answer Generation

Validated supporting chunks から citation-safe answer を生成する。

## Production

`deterministic-extractive-v1.1`

- May only emit text from validated supporting chunks
- Verbatim supporting-text fallback when extractive synthesis cannot form citation-safe answer
- Typed `GENERATION_FAILURE` on exceptions
- Does **not** claim semantic answer correctness

## Reliability（Phase 4）

- Bounded transport retry（TIMEOUT, CONNECTION_ERROR, RATE_LIMIT, PROVIDER_5XX only）
- At most 2 physical attempts
- Quality decisions are never retried

## Related

- [[RAG - Evidence Sufficiency Gate]]
- [[RAG - Abstention Strategy]]
- [[RAG V2 - Phase 4 Reliability]]
- [[RAG - Production Architecture V2]]
