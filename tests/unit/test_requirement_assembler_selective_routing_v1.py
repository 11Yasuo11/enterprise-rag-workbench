from datetime import UTC, datetime, timedelta

from rag_workbench.experiments.requirement_assembler_selective_routing_v1.assembler import (
    DeterministicRequirementAssemblerV3,
    EvidenceChunk,
    VerifiedRequirement,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.router import (
    PostLunaFeatures,
    RoutingFeatures,
    SelectiveRiskRouter,
)
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.versioning import (
    DeterministicVersionResolver,
    VersionCandidate,
)


def chunks() -> tuple[EvidenceChunk, ...]:
    return (
        EvidenceChunk("c1", "d1", "Owner is Alice."),
        EvidenceChunk("c2", "d2", "Version is v3."),
        EvidenceChunk("c3", "d3", "Deadline is Friday."),
    )


def requirement(index: int, chunk_id: str, document_id: str, span: str) -> VerifiedRequirement:
    return VerifiedRequirement(f"R{index}", f"requirement {index}", chunk_id, document_id, (span,))


def assemble(*requirements: VerifiedRequirement, authorized=frozenset({"c1", "c2", "c3"})):
    result = DeterministicRequirementAssemblerV3().assemble(
        tuple(requirements), chunks(), authorized_chunk_ids=authorized
    )
    assert result.final_generator_openai_calls == 0
    return result


def test_assembler_preserves_one_requirement() -> None:
    result = assemble(requirement(1, "c1", "d1", "Owner is Alice."))
    assert result.status == "answered"
    assert len(result.requirements) == 1
    assert result.final_generator_openai_calls == 0


def test_assembler_preserves_two_requirements_and_citations() -> None:
    result = assemble(
        requirement(1, "c1", "d1", "Owner is Alice."),
        requirement(2, "c2", "d2", "Version is v3."),
    )
    assert len(result.requirements) == 2
    assert "[C1]" in result.answer and "[C2]" in result.answer
    assert result.citations == ("c1", "c2")


def test_assembler_preserves_three_documents() -> None:
    result = assemble(
        requirement(1, "c1", "d1", "Owner is Alice."),
        requirement(2, "c2", "d2", "Version is v3."),
        requirement(3, "c3", "d3", "Deadline is Friday."),
    )
    assert result.status == "answered"
    assert len(result.requirements) == 3
    assert all(
        span in result.answer
        for span in ("Owner is Alice.", "Version is v3.", "Deadline is Friday.")
    )
    assert result.final_generator_openai_calls == 0


def test_assembler_rejects_missing_span() -> None:
    result = assemble(VerifiedRequirement("R1", "owner", "c1", "d1", ()))
    assert result.failure_code == "MISSING_SUPPORTING_SPAN"


def test_assembler_rejects_missing_requirement_id() -> None:
    result = assemble(
        requirement(1, "c1", "d1", "Owner is Alice."),
        requirement(3, "c3", "d3", "Deadline is Friday."),
    )
    assert result.failure_code == "INVALID_REQUIREMENT_IDS"


def test_assembler_rejects_invalid_or_unauthorized_chunk() -> None:
    invalid = assemble(requirement(1, "missing", "d1", "Owner is Alice."))
    unauthorized = assemble(requirement(1, "c1", "d1", "Owner is Alice."), authorized=frozenset())
    assert invalid.failure_code == "INVALID_CHUNK"
    assert unauthorized.failure_code == "UNAUTHORIZED_CHUNK"


def test_duplicate_evidence_does_not_drop_requirement() -> None:
    result = assemble(
        requirement(1, "c1", "d1", "Owner is Alice."),
        requirement(2, "c1", "d1", "Owner is Alice."),
    )
    assert len(result.requirements) == 2
    assert result.answer.count("Owner is Alice.") == 2
    assert result.answer.count("[C1]") == 2


def candidate(**overrides) -> VersionCandidate:
    values = dict(
        chunk_id="c1",
        document_id="runbook-east",
        document_version_id="dv1",
        version="2.0",
        text="active",
        tenant_id="acmeai",
        region="east",
        is_active=True,
    )
    values.update(overrides)
    return VersionCandidate(**values)


def resolve(question: str, candidates: tuple[VersionCandidate, ...], **kwargs):
    return DeterministicVersionResolver().resolve(
        question, candidates, tenant_id="acmeai", **kwargs
    )


def test_resolver_prefers_one_active_over_superseded() -> None:
    old = candidate(chunk_id="old", version="1.0", is_active=False, superseded_by="dv1")
    assert (
        resolve("Use the current active revision.", (old, candidate())).candidate.version == "2.0"
    )


def test_resolver_honors_explicit_historical_version() -> None:
    old = candidate(chunk_id="old", version="1.0", is_active=False)
    assert resolve("Use revision 1.0.", (old, candidate())).candidate.version == "1.0"


def test_resolver_latest_current_and_effective_date() -> None:
    now = datetime(2026, 8, 20, tzinfo=UTC)
    future = candidate(
        chunk_id="future", version="3.0", effective_from=now + timedelta(days=1), is_active=False
    )
    result = resolve(
        "Use the current version.", (candidate(effective_from=now), future), query_time=now
    )
    assert result.status == "VERSION_RESOLVED" and result.candidate.version == "2.0"


def test_resolver_rejects_two_active_versions() -> None:
    other = candidate(chunk_id="c2", document_version_id="dv2", version="2.1")
    assert (
        resolve("Use the latest active revision.", (candidate(), other)).status
        == "VERSION_AMBIGUOUS"
    )


def test_resolver_reports_missing_active_metadata() -> None:
    unknown = candidate(is_active=None)
    assert resolve("Use the current active revision.", (unknown,)).status == "METADATA_INSUFFICIENT"


def test_resolver_filters_region_and_tenant() -> None:
    west = candidate(chunk_id="w", document_id="runbook-west", region="west")
    foreign = candidate(chunk_id="f", tenant_id="other")
    result = resolve("Use the current east-region revision.", (candidate(), west, foreign))
    assert result.candidate.chunk_id == "c1"


def test_router_initial_paths() -> None:
    router = SelectiveRiskRouter()
    assert router.initial(RoutingFeatures()).route == "LUNA"
    assert (
        router.initial(RoutingFeatures(deterministic_support_complete=True)).route
        == "DETERMINISTIC"
    )
    assert router.initial(RoutingFeatures(version_sensitive=True)).route == "SOL"
    assert router.initial(RoutingFeatures(conflicting_evidence=True)).route == "SOL"
    assert (
        router.initial(RoutingFeatures(security_precheck_requires_abstention=True)).route
        == "SAFE_ABSTAIN"
    )


def test_router_post_luna_paths() -> None:
    router = SelectiveRiskRouter()
    assert (
        router.after_luna(PostLunaFeatures("GO", deterministic_validation_pass=True)).route
        == "DETERMINISTIC"
    )
    assert router.after_luna(PostLunaFeatures("UNCERTAIN")).route == "SOL"
    assert (
        router.after_luna(PostLunaFeatures("ABSTAIN", genuine_required_evidence_absence=True)).route
        == "SAFE_ABSTAIN"
    )
    assert (
        router.after_luna(PostLunaFeatures("ABSTAIN", deterministic_evidence_complete=True)).route
        == "SOL"
    )
    suspicious = router.after_luna(
        PostLunaFeatures("ABSTAIN", authorized_candidate_evidence_complete=True)
    )
    assert suspicious.route == "SOL"
    assert suspicious.reason == "suspicious_luna_abstention"
