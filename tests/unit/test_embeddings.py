from rag_workbench.providers.embeddings import HashingEmbeddingProvider


def test_hashing_embeddings_are_deterministic_and_normalized() -> None:
    provider = HashingEmbeddingProvider(64)
    first = provider.embed_query("exact identifier CS-1842")
    second = provider.embed_documents(["exact identifier CS-1842"])[0]
    assert first == second
    assert round(sum(value * value for value in first), 8) == 1.0
