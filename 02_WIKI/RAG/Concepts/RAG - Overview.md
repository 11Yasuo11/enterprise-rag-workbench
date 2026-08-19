# RAG - Overview

Retrieval-Augmented Generation（RAG）は、外部知識を検索して LLM の回答根拠とするアーキテクチャ。

本 Vault の RAG Knowledge は **Enterprise RAG Workbench**（評価・研究システム）に基づく。

## Canonical

- Production: [[RAG - Production Architecture V2]]
- Hub: [[RAG]]

## Pipeline（V2 公式）

```text
Query → ACL/tenant/version scope
      → Dense + BM25 → RRF
      → Cross-Encoder → Top-5
      → Evidence Sufficiency Judge
      → Extractive Generation → Answer or Abstention
```

## Related

- [[RAG - Dense Retrieval]]
- [[RAG - Hybrid Retrieval]]
- [[RAG - Cross Encoder Reranking]]
- [[RAG - Evidence Sufficiency Gate]]
- [[RAG - Evaluation Metrics]]
- [`README.md`](../../../README.md)
