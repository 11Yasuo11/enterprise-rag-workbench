from typing import Any

import httpx
import pytest

from rag_workbench.providers.embeddings import OpenAICompatibleEmbeddingProvider


def test_openai_compatible_embeddings_are_batched_dimension_checked_and_metered() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(__import__("json").loads(request.content))
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0] * 64},
                    {"index": 0, "embedding": [1.0] * 64},
                ],
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleEmbeddingProvider(
        api_key="test",
        model="text-embedding-3-small",
        dimension=64,
        client=client,
    )
    vectors = provider.embed_documents(["first", "second"])
    assert vectors[0] == [1.0] * 64
    assert captured["dimensions"] == 64
    assert captured["input"] == ["first", "second"]
    assert provider.usage.input_tokens == 7


def test_openai_compatible_embeddings_reject_wrong_dimension() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
        )
    )
    provider = OpenAICompatibleEmbeddingProvider(
        api_key="test", model="test", dimension=64, client=client
    )
    with pytest.raises(ValueError, match="incompatible vector dimension"):
        provider.embed_query("query")


def test_document_and_query_embeddings_share_identity_and_dimension() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "data": [
                        {"index": index, "embedding": [float(index + 1)] * 64}
                        for index, _ in enumerate(
                            __import__("json").loads(request.content)["input"]
                        )
                    ]
                },
            )
        )
    )
    provider = OpenAICompatibleEmbeddingProvider(
        api_key="test", model="text-embedding-3-small", dimension=64, client=client
    )
    documents = provider.embed_documents(["one", "two"])
    query = provider.embed_query("query")
    assert provider.provider_name == "openai-compatible"
    assert provider.model_name == "text-embedding-3-small"
    assert provider.version == "1"
    assert all(len(vector) == len(query) == provider.dimension for vector in documents)
