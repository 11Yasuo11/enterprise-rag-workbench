import hashlib
import math
import re

from rag_workbench.providers.embeddings.base import EmbeddingUsage

TOKEN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.IGNORECASE)


class HashingEmbeddingProvider:
    """Deterministic baseline for zero-cost local evaluation, not production recall."""

    def __init__(self, dimension: int = 64) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self._dimension = dimension

    @property
    def provider_name(self) -> str:
        return "hashing"

    @property
    def model_name(self) -> str:
        return f"local-hashing-{self.dimension}"

    @property
    def version(self) -> str:
        return "1"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_external(self) -> bool:
        return False

    @property
    def usage(self) -> EmbeddingUsage:
        return EmbeddingUsage()

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = TOKEN.findall(text.lower())
        features = tokens + [
            f"{token[i : i + 3]}" for token in tokens for i in range(len(token) - 2)
        ]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
