# ruff: noqa: E501
import hashlib
import inspect
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from rag_workbench.answerability.base import GateEvidence, GateTiming
from rag_workbench.answerability.cache import gate_cache_key
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    evidence_sufficiency_schema_identity,
    evidence_sufficiency_template_hash,
)
from rag_workbench.db.models import (
    Chunk,
    Document,
    DocumentPermission,
    DocumentVersion,
    ResearchArchitectureRecord,
    RetrievalArchitectureRecord,
    V2FinalBenchmarkRecord,
    V3Phase1ExperimentRecord,
)
from rag_workbench.experiments.hybrid_reranker_replication import RELEASE_ARCHITECTURE_ID
from rag_workbench.experiments.reranker_e2e_benchmark import (
    CORPUS_IDENTITY,
    SEMANTIC_INDEX_IDENTITY,
)
from rag_workbench.experiments.v2_final_benchmark import (
    DATASET_HASH as V2_FINAL_DATASET_HASH,
)
from rag_workbench.experiments.v2_final_benchmark import (
    DATASET_ID as V2_FINAL_DATASET_ID,
)
from rag_workbench.experiments.v2_final_benchmark import (
    HIDDEN_GROUND_TRUTH_FIELDS,
    V2_ARCHITECTURE_ID,
    V2FinalBenchmark,
    v2_architecture_configuration,
)
from rag_workbench.experiments.v2_final_benchmark_report import V2_FINAL_BENCHMARK_HEADING
from rag_workbench.experiments.v2_quality_ab_report import QUALITY_AB_BENCHMARK_HEADING
from rag_workbench.experiments.v2_quality_recovery import (
    FROZEN_V1_ARCHITECTURE_HASH,
    V1_BENCHMARK_HEADING,
    V2_RESEARCH_ARCHITECTURE_ID,
)
from rag_workbench.experiments.v2_sufficiency_fn import (
    CONTROL_JUDGE,
    CONTROL_MODE,
    FROZEN_V1_TEMPLATE_HASH,
    SCHEMA_IDENTITY,
    SOL_MODEL,
)
from rag_workbench.experiments.v3_generate_verify import (
    CANDIDATE_STRATEGY,
    CONTROL_STRATEGY,
    GO_POLICY,
    LOCK_ID,
    SELECTION_POLICY,
    V3_ARCHITECTURE_ID,
    V3GenerateVerifyBenchmark,
    apply_selection_policy,
    end_to_end_metrics,
    evaluator_supported,
    v3_candidate_configuration,
    v3_control_configuration,
)
from rag_workbench.experiments.v3_generate_verify_cases import (
    CASES,
    DATASET_ID,
    EXPECTED_DISTRIBUTION,
    OVERLAP_CEILING,
    dataset_overlap_report,
)
from rag_workbench.recovery.contracts import (
    CANNOT_DRAFT,
    CLAIM_VERIFIER_PROMPT_VERSION,
    COMPLETENESS_VERIFIER_PROMPT_VERSION,
    PRIMARY_JUDGE_STAGE,
    RECOVERY_DRAFT_PROMPT_VERSION,
    RECOVERY_DRAFT_STAGE,
    STAGE_CLAIM_VERIFIER,
    STAGE_COMPLETENESS_VERIFIER,
    AtomicClaim,
    ClaimVerification,
    DraftCitation,
    RecoveryDraft,
    RecoveryVerification,
    build_claim_verifier_messages,
    build_recovery_draft_messages,
    hosted_recovery_request_settings,
    recovery_draft_schema_identity,
    recovery_verifier_schema_identity,
)
from rag_workbench.recovery.runtime import (
    evaluate_recovery,
    recovery_cache_key,
    verification_passes,
)
from rag_workbench.security.permissions import Principal


def evidence() -> tuple[GateEvidence, ...]:
    return (
        GateEvidence(
            "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "engineering-deployment-handbook",
            "version-1",
            "2026.1",
            "The production deployment approval identifier is ENG-DEP-17.",
            SEMANTIC_INDEX_IDENTITY,
        ),
        GateEvidence(
            "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "operations-continuity-plan",
            "version-2",
            "4.0",
            "The recovery time objective for the customer API is four hours.",
            SEMANTIC_INDEX_IDENTITY,
        ),
    )


class FakeCache:
    def __init__(self, draft: dict, verification: dict | None = None) -> None:
        self.draft = draft
        self.verification = verification or {}
        self.calls: list[str] = []
        self.last_timing = GateTiming(logical_request_id="draft-1")

    def complete(self, question, chunks, *, stage, messages, prompt_version, schema_identity):
        del question, chunks, messages, prompt_version, schema_identity
        self.calls.append(stage)
        if stage == RECOVERY_DRAFT_STAGE:
            self.last_timing = GateTiming(logical_request_id="draft-1", cache_hit=False)
            return self.draft
        self.last_timing = GateTiming(logical_request_id="verifier-1", cache_hit=False)
        return self.verification


def _seed_chunk(db_session, evidence_item: GateEvidence, *, active: bool = True) -> None:
    document = Document(
        id=evidence_item.chunk_id,
        document_id=evidence_item.document_id,
        tenant_id="acmeai",
        title=evidence_item.document_id,
        source=f"{evidence_item.document_id}.md",
        source_type="markdown",
        visibility="public",
    )
    version = DocumentVersion(
        id=evidence_item.document_version_id,
        document_fk=document.id,
        version=evidence_item.version,
        content=evidence_item.text,
        content_hash=hashlib.sha256(evidence_item.text.encode()).hexdigest(),
        is_active=active,
    )
    chunk = Chunk(
        id=evidence_item.chunk_id,
        document_fk=document.id,
        document_version_id=version.id,
        chunk_index=0,
        text=evidence_item.text,
        token_count=8,
        embedding=[0.0] * 64,
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_version="1",
        embedding_dimension=64,
        index_identity=SEMANTIC_INDEX_IDENTITY,
    )
    db_session.add_all(
        [
            document,
            version,
            DocumentPermission(document_fk=document.id, permission_group="employees"),
            chunk,
        ]
    )
    db_session.flush()


def test_v2_immutability_identities() -> None:
    configuration = v2_architecture_configuration()
    assert configuration["architecture_id"] == V2_ARCHITECTURE_ID
    assert configuration["selected_retriever"] == CONTROL_MODE
    assert configuration["judge"]["id"] == CONTROL_JUDGE
    assert configuration["judge"]["prompt"] == EVIDENCE_GATE_PROMPT_VERSION
    assert configuration["judge"]["prompt_hash"] == FROZEN_V1_TEMPLATE_HASH
    assert configuration["judge"]["schema_identity"] == SCHEMA_IDENTITY
    assert (
        evidence_sufficiency_template_hash(EVIDENCE_GATE_PROMPT_VERSION) == FROZEN_V1_TEMPLATE_HASH
    )
    assert evidence_sufficiency_schema_identity() == SCHEMA_IDENTITY
    assert v3_control_configuration()["parent_architecture_id"] == V2_ARCHITECTURE_ID
    assert v3_control_configuration()["judge"]["prompt"] == EVIDENCE_GATE_PROMPT_VERSION
    assert v3_candidate_configuration()["recovery"]["stage_completeness"] == STAGE_COMPLETENESS_VERIFIER
    text = Path("BENCHMARK.md").read_text()
    assert text.count(V1_BENCHMARK_HEADING) == 1
    assert text.count(V2_FINAL_BENCHMARK_HEADING) == 1
    assert text.count(QUALITY_AB_BENCHMARK_HEADING) == 1
    assert FROZEN_V1_ARCHITECTURE_HASH in text
    assert V3_ARCHITECTURE_ID != V2_ARCHITECTURE_ID
    assert V3_ARCHITECTURE_ID != V2_RESEARCH_ARCHITECTURE_ID


def test_recovery_only_after_primary_negative(db_session) -> None:
    chunks = evidence()
    for item in chunks:
        _seed_chunk(db_session, item)
    draft = {
        "status": "DRAFTED",
        "candidate_answer": "ENG-DEP-17",
        "atomic_claims": [
            {
                "claim_id": "c1",
                "text": "ENG-DEP-17",
                "citation_ids": [chunks[0].chunk_id],
            }
        ],
        "citations": [{"chunk_id": chunks[0].chunk_id}],
    }
    verification = {
        "claim_results": [
            {
                "claim_id": "c1",
                "state": "SUPPORTED",
                "supporting_chunk_ids": [chunks[0].chunk_id],
            }
        ],
        "completeness": "COMPLETE",
    }
    cache = FakeCache(draft, verification)
    principal = Principal("u", "acmeai", frozenset({"employees"}))
    skipped = evaluate_recovery(
        session=db_session,
        principal=principal,
        question="q",
        chunks=chunks,
        cache=cache,
        primary_answerable=True,
        primary_schema_valid=True,
    )
    assert skipped.triggered is False
    assert cache.calls == []
    recovered = evaluate_recovery(
        session=db_session,
        principal=principal,
        question="q",
        chunks=chunks,
        cache=cache,
        primary_answerable=False,
        primary_schema_valid=True,
    )
    assert recovered.triggered is True
    assert cache.calls == [RECOVERY_DRAFT_STAGE, STAGE_CLAIM_VERIFIER]
    invalid = evaluate_recovery(
        session=db_session,
        principal=principal,
        question="q",
        chunks=chunks,
        cache=cache,
        primary_answerable=False,
        primary_schema_valid=False,
    )
    assert invalid.triggered is False


def test_draft_evidence_only_contract() -> None:
    messages = build_recovery_draft_messages("user question only", evidence())
    blob = " ".join(item["content"] for item in messages)
    source = inspect.getsource(build_recovery_draft_messages)
    for field in HIDDEN_GROUND_TRUTH_FIELDS:
        assert field not in blob
        assert field not in source
    assert "expected_answer" not in blob
    assert "required_chunk_ids" not in blob
    assert "category" not in blob
    assert "preferred_source_id" not in blob
    assert "untrusted_retrieved_evidence" in blob
    assert "parametric" in messages[0]["content"]
    assert CANNOT_DRAFT in messages[0]["content"]


def test_atomic_claim_schema_and_support_validation() -> None:
    verification = RecoveryVerification(
        claim_results=(
            ClaimVerification(
                claim_id="c1",
                state="SUPPORTED",
                supporting_chunk_ids=("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",),
            ),
        ),
        completeness="COMPLETE",
    )
    assert verification_passes(verification)
    incomplete = verification.model_copy(update={"completeness": "INCOMPLETE"})
    assert verification_passes(incomplete) is False
    contradicted = verification.model_copy(
        update={
            "claim_results": (
                ClaimVerification(
                    claim_id="c1",
                    state="CONTRADICTED",
                    supporting_chunk_ids=(),
                ),
            )
        }
    )
    assert verification_passes(contradicted) is False
    assert recovery_draft_schema_identity() != recovery_verifier_schema_identity()
    assert RECOVERY_DRAFT_PROMPT_VERSION != CLAIM_VERIFIER_PROMPT_VERSION
    assert PRIMARY_JUDGE_STAGE != RECOVERY_DRAFT_STAGE != STAGE_CLAIM_VERIFIER


def test_invalid_citation_and_unauthorized_and_inactive_fail_closed(db_session) -> None:
    chunks = evidence()
    _seed_chunk(db_session, chunks[0])
    _seed_chunk(db_session, chunks[1], active=False)
    principal = Principal("u", "acmeai", frozenset({"employees"}))
    invalid_draft = {
        "status": "DRAFTED",
        "candidate_answer": "secret",
        "atomic_claims": [
            {
                "claim_id": "c1",
                "text": "secret",
                "citation_ids": ["missing-id"],
            }
        ],
        "citations": [{"chunk_id": "missing-id"}],
    }
    outcome = evaluate_recovery(
        session=db_session,
        principal=principal,
        question="q",
        chunks=chunks,
        cache=FakeCache(invalid_draft),
        primary_answerable=False,
        primary_schema_valid=True,
    )
    assert outcome.answered is False
    assert outcome.typed_failure == "UNAUTHORIZED_OR_INVALID_CITATION"
    inactive_draft = {
        "status": "DRAFTED",
        "candidate_answer": "four hours",
        "atomic_claims": [
            {
                "claim_id": "c1",
                "text": "four hours",
                "citation_ids": [chunks[1].chunk_id],
            }
        ],
        "citations": [{"chunk_id": chunks[1].chunk_id}],
    }
    inactive_verification = {
        "claim_results": [
            {
                "claim_id": "c1",
                "state": "SUPPORTED",
                "supporting_chunk_ids": [chunks[1].chunk_id],
            }
        ],
        "completeness": "COMPLETE",
    }
    inactive = evaluate_recovery(
        session=db_session,
        principal=principal,
        question="q",
        chunks=chunks,
        cache=FakeCache(inactive_draft, inactive_verification),
        primary_answerable=False,
        primary_schema_valid=True,
    )
    assert inactive.answered is False
    assert inactive.typed_failure == "INACTIVE_VERSION"


def test_false_positive_and_valid_rescue_metrics() -> None:
    metrics = end_to_end_metrics(
        [
            {"behavior": "INCORRECT_ABSTENTION", "expected_answerability": True},
            {"behavior": "CORRECT_ABSTENTION", "expected_answerability": False},
            {"behavior": "CORRECT_ANSWER", "expected_answerability": True},
        ]
    )
    assert metrics["incorrect_abstentions"] == 1
    assert metrics["correct_abstentions"] == 1
    assert metrics["correct_answers"] == 1
    policy = apply_selection_policy(
        control={
            "answerable_case_correct_answer_rate": 0.65,
            "unsupported_answers": 0,
            "precision": 1.0,
        },
        candidate={
            "answerable_case_correct_answer_rate": 0.76,
            "unsupported_answers": 0,
            "precision": 1.0,
        },
        rescues=8,
        regressions=0,
        unsupported_from_correct_abstention=0,
        security={
            "acl_safety": 1.0,
            "tenant_isolation": 1.0,
            "version_correctness": 1.0,
            "prompt_injection_boundary": 1.0,
            "invalid_supporting_ids": 0,
            "unauthorized_supporting_ids": 0,
        },
        citations={"validity": 1.0},
        false_positive_recoveries=0,
    )
    assert policy["selected_strategy"] == CANDIDATE_STRATEGY
    blocked = apply_selection_policy(
        control={
            "answerable_case_correct_answer_rate": 0.65,
            "unsupported_answers": 0,
            "precision": 1.0,
        },
        candidate={
            "answerable_case_correct_answer_rate": 0.66,
            "unsupported_answers": 1,
            "precision": 0.98,
        },
        rescues=2,
        regressions=1,
        unsupported_from_correct_abstention=1,
        security={
            "acl_safety": 1.0,
            "tenant_isolation": 1.0,
            "version_correctness": 1.0,
            "prompt_injection_boundary": 1.0,
            "invalid_supporting_ids": 0,
            "unauthorized_supporting_ids": 0,
        },
        citations={"validity": 1.0},
        false_positive_recoveries=1,
    )
    assert blocked["selected_strategy"] == CONTROL_STRATEGY
    assert SELECTION_POLICY["control"] == CONTROL_STRATEGY
    assert GO_POLICY["historical_judge_fn_rescues_min"] == 6
    assert GO_POLICY["historical_should_abstain_false_positive_recoveries"] == 0
    assert GO_POLICY["unsupported_recovered_answers"] == 0
    assert GO_POLICY["version_violations"] == 0


def test_go_policy_and_failure_taxonomy() -> None:
    from rag_workbench.experiments.v3_phase1_rollup import (
        NO_GO,
        classify_recovery_failure,
        diagnostic_is_go,
        diagnostic_rollup,
    )

    assert diagnostic_is_go("GO") is True
    assert diagnostic_is_go(NO_GO) is False
    assert classify_recovery_failure("CANNOT_DRAFT_SUPPORTED_ANSWER") == "DRAFT_CANNOT_ANSWER"
    assert classify_recovery_failure("INACTIVE_VERSION") == "VERSION_FAILURE"
    assert classify_recovery_failure("COMPLETENESS_FAILURE") == "COMPLETENESS_FAILURE"
    rollup = diagnostic_rollup(
        {
            "go_nogo": "NO_GO_FOR_UNSEEN_EXPERIMENT",
            "cases": [
                {
                    "cohort": "FN",
                    "category": "near_duplicate",
                    "valid_rescue": True,
                    "answered": True,
                    "draft_success": True,
                    "verification_pass": True,
                    "recovery_triggered": True,
                    "behavior": "CORRECT_ANSWER",
                    "completeness_state": "COMPLETE",
                    "typed_failure": None,
                },
                {
                    "cohort": "SAFETY",
                    "category": "prompt_injection",
                    "false_positive_recovery": True,
                    "answered": True,
                    "draft_success": True,
                    "verification_pass": True,
                    "recovery_triggered": True,
                    "behavior": "UNSUPPORTED_ANSWER",
                    "completeness_state": "COMPLETE",
                    "typed_failure": None,
                },
            ],
        }
    )
    assert rollup["go_nogo"] == NO_GO
    assert rollup["valid_rescue_count"] == 1
    assert rollup["safety_control_false_positives"] == 1
    assert rollup["unsupported_recovery_count"] == 1
    assert rollup["near_duplicate_rescues"] == 1
    assert rollup["valid_rescue_count"] >= 6 or rollup["safety_control_false_positives"] > 0


def test_diagnostic_go_policy_includes_safety_ceilings() -> None:
    from rag_workbench.experiments.v3_phase1_rollup import DIAGNOSTIC_GO_POLICY, NO_GO

    assert DIAGNOSTIC_GO_POLICY["historical_judge_fn_rescues_min"] == 6
    assert DIAGNOSTIC_GO_POLICY["historical_should_abstain_false_positive_recoveries"] == 0
    assert DIAGNOSTIC_GO_POLICY["unsupported_recovered_answers"] == 0
    assert DIAGNOSTIC_GO_POLICY["version_violations"] == 0
    assert NO_GO == "NO_GO_FOR_UNSEEN_V3_PHASE1"


def test_cache_stage_separation_and_no_quality_retry() -> None:
    chunks = evidence()
    draft_key, draft_identity = recovery_cache_key(
        "q",
        chunks,
        stage=RECOVERY_DRAFT_STAGE,
        model=SOL_MODEL,
        prompt_version=RECOVERY_DRAFT_PROMPT_VERSION,
        prompt_hash="draft",
        schema_identity=recovery_draft_schema_identity(),
    )
    verifier_key, verifier_identity = recovery_cache_key(
        "q",
        chunks,
        stage=STAGE_CLAIM_VERIFIER,
        model=SOL_MODEL,
        prompt_version=CLAIM_VERIFIER_PROMPT_VERSION,
        prompt_hash="verifier",
        schema_identity=recovery_verifier_schema_identity(),
    )
    judge_key, _ = gate_cache_key(
        "q",
        chunks,
        provider="openai",
        model=SOL_MODEL,
        gate_version="1",
        prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
    )
    completeness_key, completeness_identity = recovery_cache_key(
        "q",
        chunks,
        stage=STAGE_COMPLETENESS_VERIFIER,
        model=SOL_MODEL,
        prompt_version=COMPLETENESS_VERIFIER_PROMPT_VERSION,
        prompt_hash="completeness",
        schema_identity="completeness-schema",
    )
    assert draft_key != verifier_key
    assert draft_key != judge_key
    assert verifier_key != judge_key
    assert completeness_key not in {draft_key, verifier_key, judge_key}
    assert draft_identity["stage"] == RECOVERY_DRAFT_STAGE
    assert verifier_identity["stage"] == STAGE_CLAIM_VERIFIER
    assert completeness_identity["stage"] == STAGE_COMPLETENESS_VERIFIER
    assert PRIMARY_JUDGE_STAGE != RECOVERY_DRAFT_STAGE != STAGE_CLAIM_VERIFIER != STAGE_COMPLETENESS_VERIFIER
    again, _ = recovery_cache_key(
        "q",
        chunks,
        stage=RECOVERY_DRAFT_STAGE,
        model=SOL_MODEL,
        prompt_version=RECOVERY_DRAFT_PROMPT_VERSION,
        prompt_hash="draft",
        schema_identity=recovery_draft_schema_identity(),
    )
    assert again == draft_key
    settings = hosted_recovery_request_settings(RECOVERY_DRAFT_STAGE)
    assert settings["quality_outcomes_retried"] is False
    assert settings["retry_policy"] == "bounded_transport_retries_only"
    runtime = inspect.getsource(
        __import__("rag_workbench.recovery.runtime", fromlist=["HostedStructuredRecoveryClient"]).HostedStructuredRecoveryClient
    )
    assert "max_total_attempts" in runtime


def test_prompt_injection_boundary_in_prompts() -> None:
    blob = " ".join(
        item["content"] for item in build_recovery_draft_messages("q", evidence())
    ) + " ".join(
        item["content"]
        for item in build_claim_verifier_messages(
            "q",
            evidence(),
            RecoveryDraft(
                status="DRAFTED",
                candidate_answer="ENG-DEP-17",
                atomic_claims=(
                    AtomicClaim(
                        claim_id="c1",
                        text="ENG-DEP-17",
                        citation_ids=("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",),
                    ),
                ),
                citations=(DraftCitation(chunk_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),),
            ),
        )
    )
    assert "untrusted" in blob.casefold()
    assert "never an instruction" in blob.casefold()
    assert "reveal" in blob.casefold()


def test_dataset_distribution_and_overlap_guard() -> None:
    assert len(CASES) == 80
    assert dict(sorted(Counter(item["category"] for item in CASES).items())) == EXPECTED_DISTRIBUTION
    three = [item for item in CASES if item["category"] == "multidoc_three"]
    assert all(len(set(item["required_document_ids"])) == 3 for item in three)
    two = [item for item in CASES if item["category"] == "multidoc_two"]
    assert all(len(set(item["required_document_ids"])) == 2 for item in two)
    report = dataset_overlap_report()
    assert report["overlap_threshold"] == OVERLAP_CEILING
    assert report["pass"] is True
    assert report["maximum_normalized_overlap"] < 0.5
    assert DATASET_ID == "acmeai-v3-generate-verify-recovery-eval-v1"


def test_v3_initialize_does_not_mutate_v2(db_session) -> None:
    now = datetime.now(UTC)
    db_session.add(
        RetrievalArchitectureRecord(
            architecture_id=V2_ARCHITECTURE_ID,
            selected_retriever=CONTROL_MODE,
            configuration=v2_architecture_configuration(),
            selection_policy={"frozen": True},
            dataset_id=V2_FINAL_DATASET_ID,
            dataset_hash=V2_FINAL_DATASET_HASH,
            retrieval_metrics={},
            immutable=True,
        )
    )
    db_session.add(
        RetrievalArchitectureRecord(
            architecture_id=RELEASE_ARCHITECTURE_ID,
            selected_retriever="HYBRID_CROSS_ENCODER_RERANK",
            configuration={"architecture_hash": FROZEN_V1_ARCHITECTURE_HASH},
            selection_policy={"frozen": True},
            dataset_id="acmeai-enterprise-rag-v1-final-eval",
            dataset_hash="x",
            retrieval_metrics={},
            immutable=True,
        )
    )
    db_session.add(
        ResearchArchitectureRecord(
            architecture_id=V2_RESEARCH_ARCHITECTURE_ID,
            parent_architecture_id=RELEASE_ARCHITECTURE_ID,
            selected_retriever=CONTROL_MODE,
            research_status="ACTIVE",
            production_status=False,
            control_configuration={},
            control_equivalence_hash="x",
            diagnosis_dataset_id="acmeai-enterprise-rag-v1-final-eval",
            diagnosis_dataset_hash="x",
            security_guardrails={},
            v1_preservation={},
            failure_census={},
            ranking_diagnostic={},
            judge_false_negative_diagnostic={},
            operational_diagnostic={},
            primary_bottleneck="x",
            recommended_ranking_intervention="x",
            selected_v2_ranking=CONTROL_MODE,
            selected_v2_judge=CONTROL_JUDGE,
            final_v2_architecture_id=V2_ARCHITECTURE_ID,
            immutable=True,
        )
    )
    db_session.add(
        V2FinalBenchmarkRecord(
            dataset_id=V2_FINAL_DATASET_ID,
            architecture_id=V2_ARCHITECTURE_ID,
            parent_architecture_id=RELEASE_ARCHITECTURE_ID,
            architecture_hash=v2_architecture_configuration()["architecture_hash"],
            architecture_configuration=v2_architecture_configuration(),
            dataset_hash=V2_FINAL_DATASET_HASH,
            case_ids=["fv2_single_01"],
            category_distribution={},
            generation_method="manual",
            maximum_prior_overlap=0.1,
            overlap_report={"pass": True},
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=CORPUS_IDENTITY,
            case_results=[{"case_id": "fv2_single_01", "root_cause": None}],
            retrieval_traces=[{"case_id": "fv2_single_01", "final_top5": []}],
            completed_at=now,
        )
    )
    db_session.commit()
    try:
        V3GenerateVerifyBenchmark(db_session).initialize()
    except ValueError as exc:
        assert "frozen v1" in str(exc) or "v2" in str(exc)
    v2 = db_session.get(RetrievalArchitectureRecord, V2_ARCHITECTURE_ID)
    assert v2 is not None
    assert v2.immutable is True
    assert v2.selected_retriever == CONTROL_MODE
    lock = db_session.get(V3Phase1ExperimentRecord, LOCK_ID)
    if lock is not None:
        assert lock.production_status is False
        assert lock.parent_architecture_id == V2_ARCHITECTURE_ID


def test_evaluator_supported_requires_facts_and_citations() -> None:
    case = SimpleNamespace(
        expected_facts=("ENG-DEP-17",),
        required_chunk_markers=("ENG-DEP-17",),
    )
    top5 = [
        {
            "chunk_id": "c1",
            "text": "The production deployment approval identifier is ENG-DEP-17.",
        }
    ]
    assert evaluator_supported(case, "ENG-DEP-17 is required.", ("c1",), top5) is True
    assert evaluator_supported(case, "unrelated", ("c1",), top5) is False
    assert evaluator_supported(case, "ENG-DEP-17", ("missing",), top5) is False


def test_no_recovery_after_positive_in_benchmark_source() -> None:
    source = inspect.getsource(V3GenerateVerifyBenchmark._run_unseen)
    assert "primary_answerable=result.answerable" in source
    assert "evaluate_recovery" in source
    assert "schema_valid_negative" in source
    generate = inspect.getsource(V3GenerateVerifyBenchmark._generate_control)
    assert "ExtractiveGenerationProvider" in generate


def test_v2_final_execute_path_unchanged() -> None:
    source = inspect.getsource(V2FinalBenchmark.execute)
    assert "evaluate_recovery" not in source
    assert "V3" not in source
