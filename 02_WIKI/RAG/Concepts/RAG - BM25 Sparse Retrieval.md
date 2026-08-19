# RAG - BM25 Sparse Retrieval

BM25 Okapi による lexical（疎）検索。identifier-heavy や lexically exact な evidence を補完する。

## Implementation

- Okapi BM25: `k1=1.2`, `b=0.75`
- Tokenization: Unicode NFKC, casefold, identifier tokens（例: `POL-2026-004`, `API-V3`）
- No stemming, synonym expansion, or learned weighting
- Tenant / ACL / active-version filters **before** scoring

BM25 は learned sparse model（SPLADE）ではない。

## Related

- [[RAG - Hybrid Retrieval]]
- [[RAG - RRF]]
- [[RAG - Dense Retrieval]]
- [`docs/ENGINEERING_DECISIONS.md`](../../../docs/ENGINEERING_DECISIONS.md) — Why BM25
