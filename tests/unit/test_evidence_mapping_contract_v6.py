from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.experiments.atomic_requirement_contract_v1 import (
    EvidenceMappingSchemaError,
    assemble_frozen_plan,
    canonical_mappings_to_validation,
    decompose_question,
    normalize_evidence_mapping,
)
from rag_workbench.experiments.universal_requirement_completeness_v1 import (
    UniversalEvidenceChunk,
)


def evidence() -> tuple[GateEvidence, ...]:
    return (
        GateEvidence("c1", "doc-a", "dv-a1", "1", "Alpha owns Atlas. Beta maintains it."),
        GateEvidence("c2", "doc-b", "dv-b2", "2", "Gamma approves Atlas."),
    )


def normalize(value: object, *, authorized: frozenset[str] = frozenset({"c1", "c2"})):
    chunks = evidence()
    return normalize_evidence_mapping(
        value,
        evidence_by_chunk={item.chunk_id: item for item in chunks},
        authorized_chunk_ids=authorized,
        selected_versions_by_document={
            "doc-a": frozenset({"dv-a1"}),
            "doc-b": frozenset({"dv-b2"}),
        },
        source_component="contract_test",
    )


def base(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "requirement_id": "R1",
        "document_id": "doc-a",
        "document_version_id": "dv-a1",
        "version": "1",
        "chunk_id": "c1",
        "supporting_span": "Alpha owns Atlas.",
    }
    value.update(changes)
    return value


def test_normalizes_single_supporting_span() -> None:
    assert normalize(base()).supporting_spans == ("Alpha owns Atlas.",)


def test_normalizes_one_plural_supporting_span() -> None:
    value = base(supporting_spans=["Alpha owns Atlas."])
    value.pop("supporting_span")
    assert normalize(value).supporting_spans == ("Alpha owns Atlas.",)


def test_preserves_multiple_supporting_spans_without_singular_truncation() -> None:
    value = base(supporting_spans=["Alpha owns Atlas.", "Beta maintains it."])
    value.pop("supporting_span")
    mapping = normalize(value)
    assert mapping.supporting_spans == ("Alpha owns Atlas.", "Beta maintains it.")
    with pytest.raises(EvidenceMappingSchemaError, match="SINGULAR_SPAN_CARDINALITY_REQUIRED"):
        _ = mapping.supporting_span


def test_preserves_multiple_authorized_citations() -> None:
    assert normalize(base(citations=["c1", "c2"])).citations == ("c1", "c2")


def test_cross_version_mapping_retains_explicit_identity() -> None:
    assert normalize(base()).document_version_id == "dv-a1"
    other = normalize(
        base(
            document_id="doc-b",
            document_version_id="dv-b2",
            version="2",
            chunk_id="c2",
            supporting_span="Gamma approves Atlas.",
        )
    )
    assert other.version == "2"


def test_multi_document_mappings_remain_distinct() -> None:
    first = normalize(base())
    second = normalize(
        base(
            requirement_id="R2",
            document_id="doc-b",
            document_version_id="dv-b2",
            version="2",
            chunk_id="c2",
            supporting_span="Gamma approves Atlas.",
        )
    )
    assert {(first.document_id, first.chunk_id), (second.document_id, second.chunk_id)} == {
        ("doc-a", "c1"),
        ("doc-b", "c2"),
    }


def test_unauthorized_mapping_is_rejected() -> None:
    with pytest.raises(EvidenceMappingSchemaError, match="UNAUTHORIZED_MAPPING"):
        normalize(base(), authorized=frozenset())


def test_wrong_version_mapping_is_rejected() -> None:
    with pytest.raises(EvidenceMappingSchemaError, match="DOCUMENT_VERSION_IDENTITY_MISMATCH"):
        normalize(base(document_version_id="dv-wrong"))


@pytest.mark.parametrize("missing", ["requirement_id", "document_id", "chunk_id"])
def test_malformed_mapping_fails_closed_with_structured_error(missing: str) -> None:
    value = base()
    value.pop(missing)
    with pytest.raises(EvidenceMappingSchemaError) as raised:
        normalize(value)
    assert raised.value.code == "INVALID_EVIDENCE_MAPPING_SCHEMA"
    assert raised.value.missing_field == missing
    assert raised.value.source_component == "contract_test"


def test_validation_adapter_persists_structured_schema_error() -> None:
    plan = decompose_question("Who owns Atlas?")
    validation = canonical_mappings_to_validation(
        plan,
        (SimpleNamespace(requirement_id="R1", document_id="doc-a", supporting_span="x"),),
        evidence(),
        authorized_chunk_ids=frozenset({"c1", "c2"}),
        selected_versions_by_document={"doc-a": frozenset({"dv-a1"})},
    )
    assert not validation.valid
    assert validation.failure_code == "INVALID_EVIDENCE_MAPPING_SCHEMA"
    assert validation.schema_error == {
        "code": "INVALID_EVIDENCE_MAPPING_SCHEMA",
        "source_component": "deterministic_validation",
        "reason": "MISSING_REQUIRED_IDENTITY",
        "missing_field": "chunk_id",
        "requirement_id": "R1",
        "mapping_shape": {
            "requirement_id": "str",
            "document_id": "str",
            "supporting_span": "str",
        },
    }


def test_assembler_preserves_every_validated_span_and_required_citation() -> None:
    plan = decompose_question("Who owns Atlas?")
    value = SimpleNamespace(
        requirement_id="R1",
        document_id="doc-a",
        chunk_id="c1",
        supporting_spans=["Alpha owns Atlas.", "Beta maintains it."],
    )
    validation = canonical_mappings_to_validation(
        plan,
        (value,),
        evidence(),
        authorized_chunk_ids=frozenset({"c1", "c2"}),
        selected_versions_by_document={"doc-a": frozenset({"dv-a1"})},
    )
    assert validation.valid
    assert [item.supporting_span for item in validation.requirements[0].mappings] == [
        "Alpha owns Atlas.",
        "Beta maintains it.",
    ]
    assembly = assemble_frozen_plan(
        plan,
        validation,
        (UniversalEvidenceChunk("c1", "doc-a", evidence()[0].text, "dv-a1"),),
        authorized_chunk_ids=frozenset({"c1"}),
        selected_versions_by_document={"doc-a": frozenset({"dv-a1"})},
    )
    assert assembly.status == "answered"
    assert assembly.citations == ("c1",)
    assert "Alpha owns Atlas." in (assembly.answer or "")
    assert "Beta maintains it." in (assembly.answer or "")


def test_frozen_fresh_005_plural_shape_is_accepted_and_enriched() -> None:
    value = SimpleNamespace(
        requirement_id="R1",
        requirement="What identifier names the Atlas production API endpoint?",
        document_id="doc-a",
        chunk_id="c1",
        supporting_spans=["Alpha owns Atlas."],
    )
    mapping = normalize(value)
    assert mapping.document_version_id == "dv-a1"
    assert mapping.version == "1"
    assert mapping.supporting_spans == ("Alpha owns Atlas.",)
