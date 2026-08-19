# RAG - Chunking

Document を retrieval 単位の chunk に分割する前処理。

## V2 Configuration

- Chunk size: 180
- Overlap: 30
- Semantic index: 64-dimensional vectors

Chunking strategy は hashing → semantic 比較実験では **独立変数に含めず固定**。

## Related

- [[RAG - Embeddings]]
- [[RAG - Dense Retrieval]]
