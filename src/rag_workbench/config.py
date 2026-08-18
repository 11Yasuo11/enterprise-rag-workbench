from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag_workbench"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    embedding_provider: str = "hashing"
    embedding_model: str = "local-hashing-64"
    embedding_version: str = "1"
    embedding_dimension: int = Field(default=64, ge=8)
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.openai.com/v1"
    allow_external_calls: bool = False
    max_external_embedding_calls: int = Field(default=0, ge=0)
    answerability_gate_provider: str | None = None
    answerability_gate_model: str | None = None
    answerability_gate_version: str = "1"
    answerability_gate_prompt_version: str = "evidence-sufficiency-v1"
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_api_key: str | None = None
    judge_base_url: str = "https://api.openai.com/v1"
    allow_external_judge_calls: bool = False
    max_external_judge_calls: int = Field(default=92, ge=0)
    local_judge_provider: str = "qwen"
    local_judge_model: str = "qwen3:8b"
    local_judge_base_url: str = "http://localhost:11434/v1"
    local_judge_api_key: str = "ollama"
    max_local_judge_calls: int = Field(default=92, ge=0)
    corpus_version: str = "acmeai-v0.1"
    evaluation_dataset_version: str = "acmeai-initial-v1"
    llm_provider: str = "extractive"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-5-mini"
    rag_admin_debug: bool = True
    default_tenant_id: str = "acmeai"
    default_principal_id: str = "local-developer"
    default_permission_groups: str = "employees"
    chunk_size: int = 180
    chunk_overlap: int = 30
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.2
    context_budget: int = 1200

    @property
    def effective_judge_api_key(self) -> str | None:
        return self.judge_api_key or self.embedding_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
