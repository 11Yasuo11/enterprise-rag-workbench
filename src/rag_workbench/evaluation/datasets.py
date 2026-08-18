from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_workbench.ingestion.corpus_roots import corpus_roots_for_version
from rag_workbench.ingestion.loaders import load_document

EvaluationCategory = Literal[
    "single_document",
    "multi_document",
    "exact_identifier",
    "versioning",
    "access_control",
    "abstention",
    "prompt_injection",
    "duplicate",
    "multidoc_two",
    "multidoc_three",
    "partial_evidence",
    "multidoc_version",
    "multidoc_acl",
    "multidoc_near_duplicate",
    "multidoc_no_answer",
    "near_duplicate",
    "version_region",
    "semantic_paraphrase",
    "acl_sensitive",
    "partial_no_answer",
]


class EvaluationPrincipal(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    principal_id: str = "evaluation-user"
    tenant_id: str = "acmeai"
    permission_groups: tuple[str, ...] = ("employees",)


class EvidenceReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    quote: str | None = None


class EvaluationCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    question: str = Field(min_length=1)
    expected_answer: str | None = None
    expected_document_ids: tuple[str, ...] = ()
    expected_chunk_ids: tuple[str, ...] = ()
    forbidden_document_ids: tuple[str, ...] = ()
    expected_versions: dict[str, str] = Field(default_factory=dict)
    evidence: tuple[EvidenceReference, ...] = ()
    required_fact_ids: tuple[str, ...] = ()
    unavailable_required_fact_ids: tuple[str, ...] = ()
    expected_facts: tuple[str, ...] = ()
    security_checks: tuple[Literal["acl", "version", "prompt_injection"], ...] = ()
    should_abstain: bool = False
    category: EvaluationCategory
    principal: EvaluationPrincipal = Field(default_factory=EvaluationPrincipal)

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_principal(cls, value: object) -> object:
        if isinstance(value, dict) and "principal" not in value and "permission_groups" in value:
            upgraded = dict(value)
            upgraded["principal"] = {"permission_groups": upgraded.pop("permission_groups")}
            return upgraded
        return value

    @property
    def permission_groups(self) -> tuple[str, ...]:
        return self.principal.permission_groups


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_version: str
    corpus_version: str = "acmeai-v0.1"
    cases: tuple[EvaluationCase, ...]

    def __iter__(self) -> Iterator[EvaluationCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> EvaluationCase:
        return self.cases[index]


class DatasetValidationResult(BaseModel):
    valid: bool
    case_count: int
    category_distribution: dict[str, int]
    errors: tuple[str, ...] = ()


def load_evaluation_dataset(path: Path, *, validate: bool = True) -> EvaluationDataset:
    dataset = EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))
    if validate:
        data_root = path.parent.parent
        result = validate_evaluation_dataset(
            dataset,
            corpus_roots_for_version(dataset.corpus_version, data_root=data_root),
        )
        if not result.valid:
            raise ValueError("Invalid evaluation dataset: " + "; ".join(result.errors))
    return dataset


def validate_evaluation_dataset(
    dataset: EvaluationDataset, corpus_root: Path | Sequence[Path]
) -> DatasetValidationResult:
    errors: list[str] = []
    ids = [case.case_id for case in dataset.cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        errors.append(f"duplicate evaluation IDs: {', '.join(duplicates)}")

    roots = [corpus_root] if isinstance(corpus_root, Path) else list(corpus_root)
    corpus: dict[str, object] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".markdown", ".pdf"}:
                continue
            if "eval" in path.parts:
                errors.append(f"evaluation data appears inside corpus root: {path}")
                continue
            document = load_document(path)
            corpus[document.document_id] = document

    for case in dataset.cases:
        referenced = set(case.expected_document_ids) | set(case.forbidden_document_ids)
        missing = sorted(referenced - corpus.keys())
        if missing:
            errors.append(f"{case.case_id}: unknown documents: {', '.join(missing)}")
        if (
            not case.should_abstain
            and not case.expected_document_ids
            and not case.expected_chunk_ids
        ):
            errors.append(f"{case.case_id}: answerable case has no retrieval ground truth")
        if case.should_abstain and case.expected_answer is not None:
            errors.append(f"{case.case_id}: abstention case must not define expected_answer")
        if (
            not case.should_abstain
            and len(case.expected_document_ids) < 2
            and case.category.startswith("multidoc_")
        ):
            errors.append(f"{case.case_id}: answerable multi-document case needs two sources")
        if case.unavailable_required_fact_ids and not case.should_abstain:
            errors.append(f"{case.case_id}: unavailable required facts require abstention")
        if len(case.required_fact_ids) != len(set(case.required_fact_ids)):
            errors.append(f"{case.case_id}: duplicate required fact IDs")
        if not case.principal.tenant_id or not case.principal.principal_id:
            errors.append(f"{case.case_id}: invalid principal identity")
        for evidence in case.evidence:
            if evidence.document_id not in case.expected_document_ids:
                errors.append(f"{case.case_id}: evidence document is not expected")
                continue
            document = corpus.get(evidence.document_id)
            if document is not None and evidence.quote and evidence.quote not in document.content:
                errors.append(f"{case.case_id}: evidence quote not found in {evidence.document_id}")
        for document_id, expected_version in case.expected_versions.items():
            document = corpus.get(document_id)
            if document is None:
                continue
            if document.version != expected_version:
                # Historical versions share a logical ID. At least one matching source
                # file must exist.
                versions = [
                    load_document(path).version
                    for path in corpus_root.glob("*")
                    if path.is_file()
                    and path.suffix.lower() in {".md", ".markdown", ".pdf"}
                    and load_document(path).document_id == document_id
                ]
                if expected_version not in versions:
                    errors.append(f"{case.case_id}: unknown expected version {expected_version}")

    distribution: dict[str, int] = {}
    for case in dataset.cases:
        distribution[case.category] = distribution.get(case.category, 0) + 1
    return DatasetValidationResult(
        valid=not errors,
        case_count=len(dataset),
        category_distribution=dict(sorted(distribution.items())),
        errors=tuple(errors),
    )
