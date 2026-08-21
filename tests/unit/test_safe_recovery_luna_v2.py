# ruff: noqa: E501
from rag_workbench.answerability.base import GateEvidence
from rag_workbench.experiments.safe_recovery_luna_v2.validation import validate_luna_go
from rag_workbench.experiments.safe_recovery_luna_v2.verifier import (
    LunaRequirement,
    LunaVerifierResult,
)
from rag_workbench.security.permissions import Principal


class _Chunk:
    def __init__(self, chunk_id: str, text: str) -> None:
        self.id = chunk_id
        self.text = text
        self.document_version_id = "v"
        self.index_identity = "idx"


class _Document:
    def __init__(self, tenant: str) -> None:
        self.tenant_id = tenant
        self.id = "d"


class _Version:
    def __init__(self, active: bool) -> None:
        self.is_active = active
        self.id = "v"
        self.version = "1"


class _FakeSession:
    def __init__(self, rows: list[tuple]) -> None:
        self.rows = rows
        self._authorized = [row[0].id for row in rows]
        self._scalar = False

    def execute(self, _stmt: object) -> "_FakeSession":
        self._scalar = False
        return self

    def all(self) -> list:
        if self._scalar:
            return list(self._authorized)
        return list(self.rows)

    def scalars(self, _stmt: object) -> "_FakeSession":
        self._scalar = True
        return self


def _go(span: str, chunk_id: str = "c1") -> LunaVerifierResult:
    return LunaVerifierResult(
        decision="GO",
        requirements=[
            LunaRequirement(
                requirement="id",
                supported=True,
                chunk_id=chunk_id,
                supporting_span=span,
            )
        ],
        all_supported=True,
        conflict=False,
        version_valid=True,
        region_valid=True,
    )


def test_span_must_exist_literally() -> None:
    chunks = (GateEvidence("c1", "doc", "v", "1", "The token is ENG-DEP-17.", "idx"),)
    session = _FakeSession([(_Chunk("c1", "The token is ENG-DEP-17."), _Document("acmeai"), _Version(True))])
    principal = Principal("u", "acmeai", frozenset({"employees"}))
    assert validate_luna_go(_go("ENG-DEP-17"), chunks, session=session, principal=principal) is None
    assert (
        validate_luna_go(_go("ENG-DEP-99"), chunks, session=session, principal=principal)
        == "SPAN_NOT_IN_CHUNK"
    )


def test_fabricated_chunk_id_rejected() -> None:
    chunks = (GateEvidence("c1", "doc", "v", "1", "hello world", "idx"),)
    session = _FakeSession([(_Chunk("c1", "hello world"), _Document("acmeai"), _Version(True))])
    principal = Principal("u", "acmeai", frozenset({"employees"}))
    assert (
        validate_luna_go(_go("hello", chunk_id="invented"), chunks, session=session, principal=principal)
        == "FABRICATED_CHUNK_ID"
    )


def test_go_without_span_is_safe_abstention() -> None:
    result = LunaVerifierResult(
        decision="GO",
        requirements=[LunaRequirement(requirement="id", supported=True, chunk_id="c1", supporting_span=None)],
        all_supported=True,
        conflict=False,
        version_valid=True,
        region_valid=True,
    )
    chunks = (GateEvidence("c1", "doc", "v", "1", "hello", "idx"),)
    session = _FakeSession([(_Chunk("c1", "hello"), _Document("acmeai"), _Version(True))])
    principal = Principal("u", "acmeai", frozenset({"employees"}))
    assert validate_luna_go(result, chunks, session=session, principal=principal) == "MISSING_SUPPORTING_SPAN"
