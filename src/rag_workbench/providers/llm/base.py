from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class GenerationContext:
    chunk_id: str
    citation_label: str
    text: str


@dataclass(frozen=True)
class GenerationRequest:
    question: str
    prompt: str
    contexts: tuple[GenerationContext, ...]


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    used_chunk_ids: tuple[str, ...]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...
