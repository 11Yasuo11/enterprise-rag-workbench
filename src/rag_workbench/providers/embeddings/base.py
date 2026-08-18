from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmbeddingUsage:
    input_tokens: int = 0
    external_calls: int = 0


class EmbeddingProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def is_external(self) -> bool: ...

    @property
    def usage(self) -> EmbeddingUsage: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
