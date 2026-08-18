from rag_workbench.retrieval.query_embedding_cache import (
    normalize_query_text,
    query_embedding_cache_key,
)


def key(**overrides) -> str:
    identity = {
        "query": " What   is the policy? ",
        "provider": "openai-compatible",
        "model": "text-embedding-3-small",
        "version": "1",
        "dimension": 64,
    }
    identity.update(overrides)
    return query_embedding_cache_key(**identity)


def test_query_embedding_cache_key_is_deterministic_and_normalizes_whitespace() -> None:
    assert normalize_query_text(" What   is the policy? ") == "What is the policy?"
    assert key() == key(query="What is the policy?")


def test_query_embedding_cache_key_isolates_provider_model_version_and_dimension() -> None:
    baseline = key()
    assert key(provider="another-provider") != baseline
    assert key(model="another-model") != baseline
    assert key(version="2") != baseline
    assert key(dimension=128) != baseline
