from rag_workbench.providers.embeddings import EmbeddingProvider


def validate_embedding_space(
    provider: EmbeddingProvider, model: str, version: str, dimension: int
) -> None:
    expected = (provider.model_name, provider.version, provider.dimension)
    actual = (model, version, dimension)
    if expected != actual:
        raise ValueError(f"Embedding space mismatch: query={expected!r}, chunk={actual!r}")
