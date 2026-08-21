from __future__ import annotations

from pathlib import Path

from rag_workbench.answerability.base import (
    AnswerabilityReason,
    AnswerabilityResult,
    GateEvidence,
    GateOperationalError,
)
from rag_workbench.answerability.validation import validate_gate_result_with_error
from rag_workbench.experiments.atomic_requirement_contract_v1 import decompose_question
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.versioning import (
    DeterministicVersionResolver,
    VersionCandidate,
    bind_requirements_to_versions,
)
from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
from rag_workbench.ingestion.pipeline import IngestionPipeline
from rag_workbench.providers.embeddings import HashingEmbeddingProvider
from rag_workbench.retrieval.bm25 import BM25Retriever
from rag_workbench.retrieval.filters import RetrievalFilters
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.retrieval.temporal import (
    DocumentVersionScope,
    TemporalScopePlan,
    plan_temporal_scope,
)
from rag_workbench.security.permissions import Principal


def candidate(
    document: str,
    version: str,
    *,
    active: bool,
    tenant: str = "acmeai",
) -> VersionCandidate:
    return VersionCandidate(
        f"c-{document}-{version}",
        document,
        f"dv-{document}-{version}",
        version,
        f"{document} {version}",
        tenant,
        None,
        active,
    )


def test_required_temporal_modes() -> None:
    cases = {
        "Under the 2025 policy, what was allowed?": ("HISTORICAL_ONLY", ("2025",)),
        "Under the current 2026 policy, what is allowed?": ("CURRENT_ONLY", ("2026",)),
        "Compare 2025 and 2026 policies.": ("CROSS_VERSION", ("2025", "2026")),
        "How did the allowance change from 2025 to 2026?": (
            "CROSS_VERSION",
            ("2025", "2026"),
        ),
        "What is the current policy?": ("CURRENT_ONLY", ()),
        "What is the allowance?": ("UNSPECIFIED_CURRENT_DEFAULT", ()),
    }
    for question, (mode, versions) in cases.items():
        plan = plan_temporal_scope(question)
        assert plan.temporal_mode == mode
        assert plan.requested_versions == versions


def test_historical_and_current_resolution_are_not_conflated() -> None:
    candidates = (
        candidate("remote-work-policy", "2025", active=False),
        candidate("remote-work-policy", "2026", active=True),
    )
    resolver = DeterministicVersionResolver()
    historical = resolver.resolve(
        "Under the 2025 policy, what was allowed?", candidates, tenant_id="acmeai"
    )
    current = resolver.resolve(
        "Under the current 2026 policy, what is allowed?", candidates, tenant_id="acmeai"
    )
    default = resolver.resolve("What is allowed?", candidates, tenant_id="acmeai")
    assert historical.candidate and historical.candidate.version == "2025"
    assert current.candidate and current.candidate.version == "2026"
    assert default.candidate and default.candidate.version == "2026"


def test_cross_version_set_and_atomic_requirement_bindings() -> None:
    candidates = (
        candidate("remote-work-policy", "2025", active=False),
        candidate("remote-work-policy", "2026", active=True),
    )
    question = "How did the remote-work allowance change from 2025 to 2026?"
    resolution = DeterministicVersionResolver().resolve(
        question, candidates, tenant_id="acmeai"
    )
    plan = decompose_question(question)
    bindings = bind_requirements_to_versions(plan.requirements, resolution)
    assert resolution.status == "VERSION_SET_RESOLVED"
    assert resolution.selections_by_document == {
        "remote-work-policy": ("2025", "2026")
    }
    assert [(item.requirement_id, item.version) for item in bindings] == [
        ("R1", "2025"),
        ("R2", "2026"),
    ]


def test_document_scoped_version_set_does_not_force_one_global_choice() -> None:
    candidates = (
        candidate("document-a", "2025", active=False),
        candidate("document-a", "2026", active=True),
        candidate("document-b", "2025", active=False),
        candidate("document-b", "2026", active=True),
    )
    scope = TemporalScopePlan(
        "CROSS_VERSION",
        ("2025", "2026"),
        (
            DocumentVersionScope("document-a", ("2026",)),
            DocumentVersionScope("document-b", ("2025",)),
        ),
    )
    resolution = DeterministicVersionResolver().resolve(
        "Use the scoped versions.",
        candidates,
        tenant_id="acmeai",
        temporal_scope=scope,
    )
    assert resolution.status == "VERSION_SET_RESOLVED"
    assert resolution.selections_by_document == {
        "document-a": ("2026",),
        "document-b": ("2025",),
    }


def build_retrievers(db_session):
    provider = HashingEmbeddingProvider(64)
    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        provider,
    )
    for name in (
        "remote-work-policy-2025.md",
        "remote-work-policy-2026.md",
        "security-incident-policy-2025.md",
        "security-incident-policy-2026.md",
        "hr-benefits-private.md",
    ):
        pipeline.ingest_path(Path("data/synthetic_company") / name, tenant_id="acmeai")
    dense = Retriever(db_session, provider)
    bm25 = BM25Retriever(
        db_session,
        index_identity=dense.index_identity,
        embedding_provider=provider.provider_name,
        embedding_model=provider.model_name,
        embedding_version=provider.version,
        embedding_dimension=provider.dimension,
    )
    return dense, bm25


def test_retrieval_eligibility_historical_cross_and_current(db_session) -> None:
    dense, bm25 = build_retrievers(db_session)
    principal = Principal("employee", "acmeai", frozenset({"employees"}))

    historical_question = "Under the 2025 remote-work policy, how many days were allowed?"
    historical = bm25.retrieve(historical_question, top_k=15, principal=principal)
    assert {item.version for item in historical if item.document_id == "remote-work-policy"} == {
        "2025"
    }
    assert any("three days per week" in item.text for item in historical)

    cross_question = "How did the remote-work allowance change from 2025 to 2026?"
    filters = RetrievalFilters(temporal_scope=plan_temporal_scope(cross_question))
    lexical = bm25.retrieve(cross_question, top_k=15, filters=filters, principal=principal)
    semantic = dense.retrieve(cross_question, top_k=15, filters=filters, principal=principal)
    for results in (lexical, semantic):
        assert {item.version for item in results if item.document_id == "remote-work-policy"} == {
            "2025",
            "2026",
        }

    current = bm25.retrieve(
        "What is the current remote-work allowance and review frequency?",
        top_k=15,
        principal=principal,
    )
    current_text = " ".join(item.text for item in current)
    assert {item.version for item in current if item.document_id == "remote-work-policy"} == {
        "2026"
    }
    assert "two days per week" in current_text and "every month" in current_text
    assert "three days per week" not in current_text and "every quarter" not in current_text

    security = bm25.retrieve(
        "What is the current severity-one reporting window?", top_k=15, principal=principal
    )
    security_text = " ".join(item.text for item in security)
    assert "15 minutes" in security_text
    assert "60 minutes" not in security_text


def test_temporal_expansion_never_bypasses_acl_or_tenant(db_session) -> None:
    _, bm25 = build_retrievers(db_session)
    employee = Principal("employee", "acmeai", frozenset({"employees"}))
    results = bm25.retrieve(
        "Under the 2025 policy use HR-BEN-771 and remote-work allowance.",
        top_k=15,
        principal=employee,
    )
    assert all(item.document_id != "hr-benefits-private" for item in results)


def test_historical_evidence_validation_requires_historical_scope(db_session) -> None:
    _, bm25 = build_retrievers(db_session)
    principal = Principal("employee", "acmeai", frozenset({"employees"}))
    question = "Under the 2025 remote-work policy, how many days were allowed?"
    hit = next(
        item
        for item in bm25.retrieve(question, top_k=15, principal=principal)
        if item.document_id == "remote-work-policy" and item.version == "2025"
    )
    evidence = (
        GateEvidence(
            hit.chunk_id,
            hit.document_id,
            hit.document_version_id,
            hit.version,
            hit.text,
            hit.metadata.get("index_identity"),
        ),
    )
    proposed = AnswerabilityResult(
        answerable=True,
        supporting_chunk_ids=(hit.chunk_id,),
        reason_code=AnswerabilityReason.SUFFICIENT_EVIDENCE,
    )

    historical = validate_gate_result_with_error(
        proposed,
        evidence,
        session=db_session,
        principal=principal,
        temporal_scope=plan_temporal_scope(question),
    )
    default = validate_gate_result_with_error(
        proposed,
        evidence,
        session=db_session,
        principal=principal,
    )

    assert historical.result.answerable
    assert historical.operational_error is None
    assert not default.result.answerable
    assert default.operational_error == GateOperationalError.INACTIVE_VERSION
