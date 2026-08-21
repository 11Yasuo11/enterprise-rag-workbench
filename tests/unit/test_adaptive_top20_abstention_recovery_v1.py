from rag_workbench.answerability.base import GateEvidence
from rag_workbench.experiments.adaptive_top20_abstention_recovery_v1.requirements import (
    decompose_requirements,
)
from rag_workbench.experiments.adaptive_top20_abstention_recovery_v1.validation import validate_go
from rag_workbench.experiments.adaptive_top20_abstention_recovery_v1.verifier import (
    RecoveryDecision,
    RequirementSupport,
)
from rag_workbench.security.permissions import Principal


class Chunk:
    def __init__(self, cid: str, text: str) -> None:
        self.id, self.text = cid, text


class Document:
    id = "doc-fk"
    tenant_id = "acmeai"


class Version:
    id = "v1"
    version = "1"
    is_active = True


class Session:
    def __init__(self) -> None:
        self.scalar_mode = False

    def execute(self, _stmt):
        self.scalar_mode = False
        return self

    def all(self):
        if self.scalar_mode:
            return ["c1"]
        return [(Chunk("c1", "Atlas is owned by Alice."), Document(), Version())]

    def scalars(self, _stmt):
        self.scalar_mode = True
        return self


def decision(**overrides):
    values = dict(
        decision="GO",
        requirements=[
            RequirementSupport(
                requirement_id="R1",
                requirement="Atlas owner",
                supported=True,
                chunk_id="c1",
                document_id="doc-a",
                cross_encoder_rank=1,
                supporting_span="owned by Alice",
            )
        ],
        required_count=1,
        supported_count=1,
        all_supported=True,
        distinct_documents_used=1,
        conflict=False,
        tenant_valid=True,
        acl_valid=True,
        version_valid=True,
        region_valid=True,
    )
    values.update(overrides)
    return RecoveryDecision(**values)


def test_three_part_question_is_atomic() -> None:
    q = (
        "Provide deployment approval ID, API recovery time objective, "
        "and remote-work weekly allowance."
    )
    assert decompose_requirements(q) == (
        "deployment approval ID",
        "API recovery time objective",
        "remote-work weekly allowance",
    )


def test_literal_span_and_rank_are_verified() -> None:
    chunks = (GateEvidence("c1", "doc-a", "v1", "1", "Atlas is owned by Alice.", "idx"),)
    assert (
        validate_go(
            decision(),
            ("Atlas owner",),
            chunks,
            session=Session(),
            principal=Principal("u", "acmeai", frozenset({"employees"})),
        )
        is None
    )
    assert (
        validate_go(
            decision(
                requirements=[
                    RequirementSupport(
                        requirement_id="R1",
                        requirement="Atlas owner",
                        supported=True,
                        chunk_id="c1",
                        document_id="doc-a",
                        cross_encoder_rank=2,
                        supporting_span="owned by Alice",
                    )
                ]
            ),
            ("Atlas owner",),
            chunks,
            session=Session(),
            principal=Principal("u", "acmeai", frozenset({"employees"})),
        )
        == "RANK_OR_DOCUMENT_MISMATCH"
    )


def test_partial_coverage_cannot_validate() -> None:
    chunks = (GateEvidence("c1", "doc-a", "v1", "1", "Atlas is owned by Alice.", "idx"),)
    result = decision(supported_count=0, all_supported=False)
    assert (
        validate_go(
            result,
            ("Atlas owner",),
            chunks,
            session=Session(),
            principal=Principal("u", "acmeai", frozenset({"employees"})),
        )
        == "GO_INVARIANT_FAILED"
    )
