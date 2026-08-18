from __future__ import annotations

from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_workbench.evaluation.datasets import EvaluationPrincipal


class RetrievalGroundTruthCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    question: str = Field(min_length=1)
    required_document_ids: tuple[str, ...] = ()
    required_chunk_ids: tuple[str, ...] = ()
    required_version_ids: dict[str, str] = Field(default_factory=dict)
    forbidden_document_ids: tuple[str, ...] = ()
    expected_access_behavior: str
    expected_answerability: bool
    category: str
    principal: EvaluationPrincipal = Field(default_factory=EvaluationPrincipal)

    @model_validator(mode="after")
    def validate_ground_truth(self) -> RetrievalGroundTruthCase:
        if self.expected_answerability and not (
            self.required_document_ids or self.required_chunk_ids
        ):
            raise ValueError("answerable retrieval cases require evidence")
        if set(self.required_document_ids) & set(self.forbidden_document_ids):
            raise ValueError("required and forbidden documents must be disjoint")
        return self


class RetrievalGroundTruthDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    corpus_version: str
    cases: tuple[RetrievalGroundTruthCase, ...]

    @model_validator(mode="after")
    def validate_cases(self) -> RetrievalGroundTruthDataset:
        identifiers = [case.case_id for case in self.cases]
        duplicates = [item for item, count in Counter(identifiers).items() if count > 1]
        if duplicates:
            raise ValueError(f"duplicate case IDs: {', '.join(sorted(duplicates))}")
        return self


def load_retrieval_ground_truth(path: Path) -> RetrievalGroundTruthDataset:
    return RetrievalGroundTruthDataset.model_validate_json(path.read_text(encoding="utf-8"))
