from typing import Any

import httpx

from rag_workbench.providers.embeddings.base import EmbeddingUsage


class OpenAICompatibleEmbeddingProvider:
    """Vendor-neutral adapter for the OpenAI-compatible ``POST /embeddings`` contract."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimension: int,
        base_url: str = "https://api.openai.com/v1",
        version: str = "1",
        provider_name: str = "openai-compatible",
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY is required for external embeddings")
        if dimension < 1:
            raise ValueError("embedding dimension must be positive")
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._base_url = base_url.rstrip("/")
        self._version = version
        self._provider_name = provider_name
        self._timeout = timeout
        self._client = client
        self._input_tokens = 0
        self._external_calls = 0

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def version(self) -> str:
        return self._version

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_external(self) -> bool:
        return True

    @property
    def usage(self) -> EmbeddingUsage:
        return EmbeddingUsage(
            input_tokens=self._input_tokens, external_calls=self._external_calls
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("embedding inputs cannot be empty")
        payload = self._request(texts)
        data = sorted(payload.get("data", []), key=lambda item: item["index"])
        vectors = [item["embedding"] for item in data]
        if len(vectors) != len(texts):
            raise ValueError("embedding provider returned an unexpected vector count")
        if any(len(vector) != self.dimension for vector in vectors):
            raise ValueError("embedding provider returned an incompatible vector dimension")
        usage = payload.get("usage") or {}
        self._input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _request(self, texts: list[str]) -> dict[str, Any]:
        self._external_calls += 1
        client = self._client or httpx.Client(timeout=self._timeout)
        should_close = self._client is None
        try:
            response = client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "input": texts,
                    "model": self.model_name,
                    "dimensions": self.dimension,
                    "encoding_format": "float",
                },
            )
            response.raise_for_status()
            return response.json()
        finally:
            if should_close:
                client.close()
