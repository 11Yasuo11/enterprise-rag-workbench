import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def stable_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)

    @property
    def config_hash(self) -> str:
        serialized = json.dumps(
            self.stable_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class CorpusDatasetIdentity(FrozenConfig):
    corpus_version: str
    evaluation_dataset_version: str


class IngestionConfig(FrozenConfig):
    chunk_strategy: Literal["fixed_token"] = "fixed_token"
    chunk_size: int = Field(default=180, gt=0)
    chunk_overlap: int = Field(default=30, ge=0)
    embedding_provider: str = "hashing"
    embedding_model: str = "local-hashing-64"
    embedding_dimension: int = Field(default=64, gt=0)
    embedding_version: str = "1"

    @model_validator(mode="after")
    def validate_overlap(self) -> "IngestionConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class RetrievalConfig(FrozenConfig):
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float | None = Field(default=0.2, ge=-1, le=1)
    distance_metric: Literal["cosine"] = "cosine"


class GenerationConfig(FrozenConfig):
    llm_provider: str = "extractive"
    llm_model: str = "deterministic-extractive-v1"
    prompt_version: str = "baseline-v1"
    temperature: float = Field(default=0.0, ge=0)
    context_budget: int = Field(default=1200, gt=0)


class AnswerabilityGateConfig(FrozenConfig):
    gate_type: Literal["semantic_judge"] = "semantic_judge"
    judge_provider: str
    judge_model: str
    judge_version: str = "1"
    prompt_version: str = "evidence-sufficiency-v1"
    supporting_context_only: bool = False


class ExperimentConfig(FrozenConfig):
    name: str
    identity: CorpusDatasetIdentity
    ingestion: IngestionConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    answerability_gate: AnswerabilityGateConfig | None = None

    @property
    def ingestion_config_hash(self) -> str:
        return self.ingestion.config_hash

    @property
    def retrieval_config_hash(self) -> str:
        return self.retrieval.config_hash

    @property
    def generation_config_hash(self) -> str:
        return self.generation.config_hash

    @property
    def gate_config_hash(self) -> str | None:
        return self.answerability_gate.config_hash if self.answerability_gate else None

    @property
    def experiment_config_hash(self) -> str:
        payload = self.stable_payload()
        payload.pop("name", None)
        if payload.get("answerability_gate") is None:
            payload.pop("answerability_gate", None)
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def index_identity(self) -> str:
        payload = {
            "corpus_version": self.identity.corpus_version,
            "ingestion": self.ingestion.stable_payload(),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def index_fingerprint(self) -> str:
        return self.index_identity[:16]

    def requires_reindex(self, previous: "ExperimentConfig") -> bool:
        return self.index_identity != previous.index_identity


# Backward-compatible name retained for callers of the original baseline API.
DatasetIdentity = CorpusDatasetIdentity


def load_experiment_config(path: Path) -> ExperimentConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(payload)


def index_identity_for(corpus_version: str, ingestion: IngestionConfig) -> str:
    payload = {"corpus_version": corpus_version, "ingestion": ingestion.stable_payload()}
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
