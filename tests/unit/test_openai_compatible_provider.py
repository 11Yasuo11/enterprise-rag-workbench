from typing import Any

from rag_workbench.providers.llm.base import GenerationContext, GenerationRequest
from rag_workbench.providers.llm.openai_compatible import OpenAICompatibleLLMProvider


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"answer":"Supported answer [C1]",'
                            '"used_chunk_ids":["real-chunk","invented-chunk"]}'
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }


def test_openai_compatible_provider_filters_unretrieved_citation_ids(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag_workbench.providers.llm.openai_compatible.httpx.post",
        lambda *args, **kwargs: FakeResponse(),
    )
    provider = OpenAICompatibleLLMProvider("test-key", "test-model", "https://example.test/v1")
    result = provider.generate(
        GenerationRequest(
            question="Question?",
            prompt="System policy and context",
            contexts=(GenerationContext("real-chunk", "C1", "Evidence"),),
        )
    )
    assert result.answer == "Supported answer [C1]"
    assert result.used_chunk_ids == ("real-chunk",)
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 4
