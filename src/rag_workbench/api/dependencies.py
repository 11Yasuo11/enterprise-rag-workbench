from functools import lru_cache

from sqlalchemy.orm import Session

from rag_workbench.answerability.cache import CachedAnswerabilityGate
from rag_workbench.answerability.openai_compatible import OpenAICompatibleAnswerabilityGate
from rag_workbench.answerability.planning import (
    ExternalJudgeCallLimitGate,
    LocalJudgeCallLimitGate,
)
from rag_workbench.answerability.qwen import QwenAnswerabilityGate
from rag_workbench.config import get_settings
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.providers.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from rag_workbench.providers.llm import ExtractiveGenerationProvider, OpenAICompatibleLLMProvider
from rag_workbench.reranking.cross_encoder import CrossEncoderReranker
from rag_workbench.runtime.identity_reranker import IdentityReranker


@lru_cache
def embedding_provider() -> EmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingProvider(settings.embedding_dimension)
    if settings.embedding_provider in {"openai", "openai-compatible"}:
        if not settings.allow_external_calls:
            raise ValueError("ALLOW_EXTERNAL_CALLS=true is required for external embeddings")
        return OpenAICompatibleEmbeddingProvider(
            api_key=settings.embedding_api_key or "",
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
            base_url=settings.embedding_base_url,
            version=settings.embedding_version,
            provider_name=settings.embedding_provider,
        )
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


@lru_cache
def production_reranker():
    """Process-scoped reranker. Hashing/local may use identity; production requires CE."""
    settings = get_settings()
    if settings.embedding_provider == "hashing":
        return IdentityReranker()
    return CrossEncoderReranker(device="cpu")


@lru_cache
def llm_provider() -> ExtractiveGenerationProvider | OpenAICompatibleLLMProvider:
    settings = get_settings()
    if settings.llm_provider == "extractive":
        return ExtractiveGenerationProvider()
    if settings.llm_provider == "openai":
        return OpenAICompatibleLLMProvider(
            api_key=settings.openai_api_key or "",
            model=settings.openai_model,
            base_url=settings.openai_base_url,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def context_builder() -> ContextBuilder:
    return ContextBuilder(get_settings().context_budget)


def answerability_gate(session: Session):
    settings = get_settings()
    if not settings.answerability_gate_provider and not settings.answerability_gate_model:
        return None
    if not settings.answerability_gate_provider or not settings.answerability_gate_model:
        raise ValueError(
            "Both ANSWERABILITY_GATE_PROVIDER and ANSWERABILITY_GATE_MODEL are required"
        )
    if settings.answerability_gate_provider not in {"openai", "qwen"}:
        raise ValueError("Unsupported ANSWERABILITY_GATE_PROVIDER")
    if settings.answerability_gate_provider == "openai":
        if not settings.allow_external_judge_calls:
            raise ValueError("ALLOW_EXTERNAL_JUDGE_CALLS=true is required")
        if settings.max_external_judge_calls < 1:
            raise ValueError("MAX_EXTERNAL_JUDGE_CALLS must be a positive reviewed limit")
        delegate = OpenAICompatibleAnswerabilityGate(
            api_key=settings.effective_judge_api_key or "",
            model=settings.answerability_gate_model,
            base_url=settings.judge_base_url,
            gate_version=settings.answerability_gate_version,
            prompt_version=settings.answerability_gate_prompt_version,
            provider_name="openai",
        )
        limited = ExternalJudgeCallLimitGate(delegate, settings.max_external_judge_calls)
    else:
        delegate = QwenAnswerabilityGate(
            api_key=settings.local_judge_api_key,
            model=settings.answerability_gate_model,
            base_url=settings.local_judge_base_url,
            gate_version=settings.answerability_gate_version,
            prompt_version=settings.answerability_gate_prompt_version,
        )
        limited = LocalJudgeCallLimitGate(delegate, settings.max_local_judge_calls)
    return CachedAnswerabilityGate(session, limited)
