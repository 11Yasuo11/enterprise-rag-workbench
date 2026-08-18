import json

import httpx

from rag_workbench.providers.llm.base import GenerationRequest, GenerationResult


class OpenAICompatibleLLMProvider:
    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 45.0) -> None:
        if not api_key:
            raise ValueError("An API key is required for the OpenAI-compatible provider")
        self.api_key = api_key
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "openai-compatible"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, request: GenerationRequest) -> GenerationResult:
        allowed_ids = {context.chunk_id for context in request.contexts}
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": request.prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {request.question}\nReturn JSON with answer and "
                            "used_chunk_ids. "
                            "Only use chunk IDs present in the context."
                        ),
                    },
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        parsed = json.loads(payload["choices"][0]["message"]["content"])
        used_ids = tuple(item for item in parsed.get("used_chunk_ids", []) if item in allowed_ids)
        usage = payload.get("usage", {})
        return GenerationResult(
            answer=str(parsed.get("answer", "")).strip(),
            used_chunk_ids=used_ids,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
