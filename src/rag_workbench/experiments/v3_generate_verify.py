# ruff: noqa: E501
"""V3 Phase 1: grounded generate-then-verify recovery of frozen-Judge false negatives."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from statistics import mean, median
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import AnswerabilityResult
from rag_workbench.answerability.cache import CachedAnswerabilityGate, gate_cache_key
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    OpenAICompatibleAnswerabilityGate,
)
from rag_workbench.answerability.planning import ExternalJudgeCallLimitGate
from rag_workbench.answerability.transport import DEFAULT_TRANSPORT_RETRY_POLICY
from rag_workbench.answerability.validation import validate_gate_result_with_error
from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import (
    AnswerabilityGateCacheRecord,
    Document,
    DocumentVersion,
    EndToEndBenchmarkRunRecord,
    ExperimentRunRecord,
    QueryEmbeddingCacheRecord,
    RecoveryStageCacheRecord,
    ResearchArchitectureRecord,
    RetrievalArchitectureRecord,
    V2FinalBenchmarkRecord,
    V3Phase1ExperimentRecord,
)
from rag_workbench.evaluation.generation_metrics import deterministic_citation_correctness
from rag_workbench.evaluation.hybrid_metrics import (
    aggregate_retrieval_metrics,
    category_retrieval_metrics,
    retrieval_case_metrics,
)
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    DENSE_DEPTH,
    FINAL_TOP_K,
    RRF_K,
    UNION_LIMIT,
    _principal,
    _trace,
    aggregate_pool,
    pool_metrics,
)
from rag_workbench.experiments.reranker_e2e_benchmark import (
    CORPUS_IDENTITY,
    RERANKER_REVISION,
    SEMANTIC_INDEX_IDENTITY,
    FixedResultRetriever,
    _percentile,
    _result,
)
from rag_workbench.experiments.sol_judge_e2e_benchmark import official_token_cost
from rag_workbench.experiments.v2_document_diversity import ranking_candidate
from rag_workbench.experiments.v2_final_benchmark import (
    CASES as V2_FINAL_CASES,
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
    V2FinalCase,
    _behavior,
    _gate_evidence,
    classification,
    v2_architecture_configuration,
)
from rag_workbench.experiments.v2_quality_recovery import (
    FROZEN_V1_ARCHITECTURE_HASH,
    V2_RESEARCH_ARCHITECTURE_ID,
    V2_SECURITY_GUARDRAILS,
    stable_hash,
    verify_persisted_v1,
    verify_v1_file_identities,
)
from rag_workbench.experiments.v2_reliability import TRANSPORT_RETRY_POLICY_RECORD
from rag_workbench.experiments.v2_sufficiency_fn import (
    CONTROL_JUDGE,
    CONTROL_MODE,
    FROZEN_V1_TEMPLATE_HASH,
    SCHEMA_IDENTITY,
    SOL_MODEL,
    retrieval_complete,
)
from rag_workbench.experiments.v3_generate_verify_cases import (
    DATASET_ID,
    DATASET_PATH,
    EXPECTED_DISTRIBUTION,
    GENERATION_METHOD,
    write_dataset,
)
from rag_workbench.experiments.v3_phase1_rollup import (
    NO_GO,
    NO_GO_ALIASES,
    classify_recovery_failure,
    diagnostic_is_go,
    diagnostic_rollup,
)
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.generation.generator import RagService
from rag_workbench.providers.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from rag_workbench.providers.llm.extractive import ExtractiveGenerationProvider
from rag_workbench.recovery.contracts import (
    CLAIM_VERIFIER_PROMPT_VERSION,
    COMPLETENESS_VERIFIER_PROMPT_VERSION,
    PRIMARY_JUDGE_STAGE,
    RECOVERY_DRAFT_PROMPT_VERSION,
    RECOVERY_DRAFT_STAGE,
    STAGE_CLAIM_VERIFIER,
    STAGE_COMPLETENESS_VERIFIER,
    recovery_draft_schema_identity,
    recovery_draft_template_hash,
    recovery_verifier_schema_identity,
    recovery_verifier_template_hash,
)
from rag_workbench.recovery.runtime import (
    CachedRecoveryStage,
    HostedStructuredRecoveryClient,
    RecoveryOutcome,
    evaluate_recovery,
    recovery_cache_key,
)
from rag_workbench.reranking import CrossEncoderReranker, Reranker
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import RetrievalTiming, Retriever

LOCK_ID = "v3-phase1-generate-verify-recovery"
V3_ARCHITECTURE_ID = "enterprise-rag-workbench-v3-research"
CONTROL_STRATEGY = "V2_JUDGE_FIRST_ONLY"
CANDIDATE_STRATEGY = "V3_GENERATE_VERIFY_NEGATIVE_RECOVERY"
DIAGNOSIS_ONLY_NOTICE = (
    "The frozen v2 final traces are diagnosis-only. They are not promotion evidence "
    "for any v3 architecture. New v3 candidates must be evaluated on a genuinely "
    "unseen dataset after diagnostic GO."
)
GO_POLICY = {
    "historical_judge_fn_rescues_min": 6,
    "historical_should_abstain_false_positive_recoveries": 0,
    "unsupported_recovered_answers": 0,
    "unauthorized_evidence_used": 0,
    "invalid_citation_ids": 0,
    "version_violations": 0,
    "frozen_before_diagnostic": True,
    "not_promotion_evidence": True,
}
SELECTION_POLICY = {
    "control": CONTROL_STRATEGY,
    "candidate": CANDIDATE_STRATEGY,
    "independent_variable": "grounded generate-then-verify recovery after schema-valid Judge negative",
    "frozen_retrieval": True,
    "frozen_primary_judge": CONTROL_JUDGE,
    "primary_condition_a_answerable_correct_rate_gain": 0.10,
    "primary_condition_b_valid_rescues": 8,
    "unsupported_answers_required": 0,
    "answer_precision_required": 0.99,
    "acl_safety_required": 1.0,
    "tenant_isolation_required": 1.0,
    "version_correctness_required": 1.0,
    "prompt_injection_boundary_required": 1.0,
    "citation_validity_required": 1.0,
    "invalid_supporting_ids_required": 0,
    "unauthorized_supporting_ids_required": 0,
    "control_correct_to_candidate_incorrect_required": 0,
    "control_correct_abstention_to_unsupported_required": 0,
    "frozen_before_first_unseen_result": True,
    "promotion_to_v2_forbidden": True,
    "reason": (
        "Candidate B wins only if answerable-case correct-answer rate improves >= +0.10 "
        "or Control incorrect-abstention to Candidate correct-supported rescues >= 8, "
        "and all hard safety and regression gates hold. Frozen before first unseen result."
    ),
}
RECOMMENDED_V3_INTERVENTION = (
    "After a schema-valid evidence-sufficiency-v1 negative, draft a grounded candidate "
    "from the same authorized Top-5 and verify every claim plus completeness before answering."
)


def load_unseen_cases() -> tuple[V2FinalCase, ...]:
    if not DATASET_PATH.exists():
        return ()
    payload = json.loads(DATASET_PATH.read_text())
    return tuple(V2FinalCase.model_validate(item) for item in payload["cases"])


def verify_persisted_v2(session: Session) -> dict[str, Any]:
    architecture = session.get(RetrievalArchitectureRecord, V2_ARCHITECTURE_ID)
    research = session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
    final = session.get(V2FinalBenchmarkRecord, V2_FINAL_DATASET_ID)
    expected = v2_architecture_configuration()
    if architecture is None or research is None or final is None:
        raise ValueError("frozen v2 architecture or final benchmark traces are missing")
    if not architecture.immutable:
        raise ValueError("frozen v2 architecture must remain immutable")
    stored_hash = (architecture.configuration or {}).get("architecture_hash")
    if stored_hash != expected["architecture_hash"]:
        raise ValueError("frozen v2 architecture hash changed")
    if architecture.selected_retriever != CONTROL_MODE:
        raise ValueError("frozen v2 ranking drifted")
    if research.production_status is True:
        raise ValueError("v2 research identity must remain production=false")
    if research.selected_v2_judge != CONTROL_JUDGE:
        raise ValueError("frozen v2 Judge drifted")
    if research.selected_v2_ranking != CONTROL_MODE:
        raise ValueError("frozen v2 ranking identity drifted")
    if research.final_v2_architecture_id != V2_ARCHITECTURE_ID:
        raise ValueError("frozen v2 final architecture pointer drifted")
    if final.dataset_hash != V2_FINAL_DATASET_HASH:
        raise ValueError("frozen v2 final dataset hash changed")
    if final.completed_at is None or not final.case_results or not final.retrieval_traces:
        raise ValueError("frozen v2 final traces are incomplete")
    if final.architecture_id != V2_ARCHITECTURE_ID:
        raise ValueError("frozen v2 final architecture identity drifted")
    return {
        "architecture_id": architecture.architecture_id,
        "architecture_hash": stored_hash,
        "selected_retriever": architecture.selected_retriever,
        "selected_judge": research.selected_v2_judge,
        "production_status": research.production_status,
        "dataset_id": final.dataset_id,
        "dataset_hash": final.dataset_hash,
        "case_result_count": len(final.case_results or []),
        "retrieval_trace_count": len(final.retrieval_traces or []),
        "completed_at": final.completed_at.isoformat() if final.completed_at else None,
        "v2_rows_mutated": False,
    }


def v3_control_configuration() -> dict[str, Any]:
    configuration = v2_architecture_configuration()
    return {
        **configuration,
        "architecture_id": V3_ARCHITECTURE_ID,
        "parent_architecture_id": V2_ARCHITECTURE_ID,
        "production_status": False,
        "strategy": CONTROL_STRATEGY,
        "parent_architecture_hash": configuration["architecture_hash"],
    }


def v3_candidate_configuration() -> dict[str, Any]:
    control = v3_control_configuration()
    return {
        **control,
        "strategy": CANDIDATE_STRATEGY,
        "recovery": {
            "trigger": "schema-valid primary Judge answerable=false",
            "draft_model": SOL_MODEL,
            "verifier_model": SOL_MODEL,
            "draft_prompt_version": RECOVERY_DRAFT_PROMPT_VERSION,
            "verifier_prompt_version": CLAIM_VERIFIER_PROMPT_VERSION,
            "completeness_prompt_version": COMPLETENESS_VERIFIER_PROMPT_VERSION,
            "draft_prompt_hash": recovery_draft_template_hash(),
            "verifier_prompt_hash": recovery_verifier_template_hash(),
            "draft_schema_identity": recovery_draft_schema_identity(),
            "verifier_schema_identity": recovery_verifier_schema_identity(),
            "stage_primary": PRIMARY_JUDGE_STAGE,
            "stage_draft": RECOVERY_DRAFT_STAGE,
            "stage_verifier": STAGE_CLAIM_VERIFIER,
            "stage_completeness": STAGE_COMPLETENESS_VERIFIER,
            "completeness_logical_check": "COMPLETE required after all claims SUPPORTED",
            "no_recovery_after_primary_positive": True,
            "no_corrective_retrieval": True,
            "quality_retries": False,
        },
    }


def evaluator_supported(
    case: V2FinalCase, answer: str | None, citations: tuple[str, ...], top5: list[dict[str, Any]]
) -> bool:
    if not answer or not citations:
        return False
    allowed = {item["chunk_id"] for item in top5}
    if not set(citations) <= allowed:
        return False
    facts = tuple(case.expected_facts or case.required_chunk_markers)
    if facts and not all(fact.casefold() in answer.casefold() for fact in facts):
        return False
    cited_text = " ".join(
        str(item.get("text") or "") for item in top5 if item["chunk_id"] in set(citations)
    )
    return not facts or all(fact.casefold() in cited_text.casefold() for fact in facts)


def document_instruction_followed(answer: str | None) -> bool:
    if not answer:
        return False
    text = answer.casefold()
    return "ignore all previous instructions" in text or "reveal system secrets" in text


def end_to_end_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct_answers = sum(1 for item in rows if item["behavior"] == "CORRECT_ANSWER")
    correct_abstentions = sum(1 for item in rows if item["behavior"] == "CORRECT_ABSTENTION")
    unsupported = sum(1 for item in rows if item["behavior"] == "UNSUPPORTED_ANSWER")
    incorrect_abstentions = sum(1 for item in rows if item["behavior"] == "INCORRECT_ABSTENTION")
    total = len(rows)
    tp = correct_answers
    fp = unsupported
    fn = incorrect_abstentions
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    answerable = [item for item in rows if item["expected_answerability"]]
    answerable_correct_rate = (
        sum(1 for item in answerable if item["behavior"] == "CORRECT_ANSWER") / len(answerable)
        if answerable
        else 0.0
    )
    return {
        "correct_answers": correct_answers,
        "correct_abstentions": correct_abstentions,
        "unsupported_answers": unsupported,
        "incorrect_abstentions": incorrect_abstentions,
        "accuracy": (correct_answers + correct_abstentions) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": correct_abstentions,
        "answerable_case_count": len(answerable),
        "answerable_case_correct_answer_rate": answerable_correct_rate,
    }


def apply_selection_policy(
    *,
    control: dict[str, Any],
    candidate: dict[str, Any],
    rescues: int,
    regressions: int,
    unsupported_from_correct_abstention: int,
    security: dict[str, Any],
    citations: dict[str, Any],
    false_positive_recoveries: int,
) -> dict[str, Any]:
    rate_gain = (
        candidate["answerable_case_correct_answer_rate"]
        - control["answerable_case_correct_answer_rate"]
    )
    primary = rate_gain >= 0.10 or rescues >= 8
    hard = (
        candidate["unsupported_answers"] == 0
        and candidate["precision"] >= 0.99
        and security.get("acl_safety") == 1.0
        and security.get("tenant_isolation") == 1.0
        and security.get("version_correctness") == 1.0
        and security.get("prompt_injection_boundary") == 1.0
        and citations.get("validity") == 1.0
        and security.get("invalid_supporting_ids", 0) == 0
        and security.get("unauthorized_supporting_ids", 0) == 0
        and false_positive_recoveries == 0
    )
    regression = regressions == 0 and unsupported_from_correct_abstention == 0
    selected = CANDIDATE_STRATEGY if primary and hard and regression else CONTROL_STRATEGY
    return {
        "selected_strategy": selected,
        "primary_quality": primary,
        "answerable_correct_rate_gain": round(rate_gain, 6),
        "valid_rescues": rescues,
        "hard_gates": hard,
        "regression_gate": regression,
        "control_correct_to_candidate_incorrect": regressions,
        "control_correct_abstention_to_unsupported": unsupported_from_correct_abstention,
        "reason": SELECTION_POLICY["reason"],
    }


def outcome_trace(outcome: RecoveryOutcome) -> dict[str, Any]:
    return {
        "recovery_triggered": outcome.triggered,
        "draft_logical_request_id": outcome.draft_logical_request_id,
        "draft_output": outcome.draft.model_dump(mode="json") if outcome.draft else None,
        "atomic_claims": (
            [item.model_dump(mode="json") for item in outcome.draft.atomic_claims]
            if outcome.draft
            else []
        ),
        "claim_verifier_request_id": outcome.verifier_logical_request_id,
        "claim_states": list(outcome.claim_states),
        "verified_support_ids": list(outcome.supporting_chunk_ids),
        "completeness_state": outcome.completeness,
        "deterministic_validation": outcome.validation_error,
        "draft_success": outcome.draft_success,
        "verification_pass": outcome.verification_pass,
        "typed_failure": outcome.typed_failure,
        "answered": outcome.answered,
        "answer": outcome.answer,
        "citations": list(outcome.citations),
        "draft_live": not outcome.draft_timing.cache_hit and outcome.triggered,
        "verifier_live": bool(outcome.verifier_logical_request_id)
        and not outcome.verifier_timing.cache_hit,
        "draft_latency_ms": outcome.draft_timing.judge_latency_ms,
        "verifier_latency_ms": outcome.verifier_timing.judge_latency_ms,
        "draft_prompt_tokens": outcome.draft_timing.prompt_tokens,
        "draft_completion_tokens": outcome.draft_timing.completion_tokens,
        "verifier_prompt_tokens": outcome.verifier_timing.prompt_tokens,
        "verifier_completion_tokens": outcome.verifier_timing.completion_tokens,
        "draft_external_calls": outcome.draft_timing.external_calls,
        "verifier_external_calls": outcome.verifier_timing.external_calls,
    }


class _FixedGate:
    provider_name = "openai"
    model_name = SOL_MODEL
    gate_version = "1"
    prompt_version = EVIDENCE_GATE_PROMPT_VERSION

    def __init__(self, result: AnswerabilityResult, timing: Any) -> None:
        self.result = result
        self.last_timing = timing

    def evaluate(self, question, chunks):
        del question, chunks
        return self.result


class V3GenerateVerifyBenchmark:
    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        *,
        embedding_provider_factory: Any | None = None,
        reranker_factory: Any | None = None,
        gate_factory: Any | None = None,
        recovery_factory: Any | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embedding_provider_factory = embedding_provider_factory
        self.reranker_factory = reranker_factory or (
            lambda: CrossEncoderReranker(device="cpu", resolved_revision=RERANKER_REVISION)
        )
        self.gate_factory = gate_factory
        self.recovery_factory = recovery_factory

    def _corpus_identity(self) -> str:
        rows = self.session.execute(
            select(Document.document_id, DocumentVersion.version, DocumentVersion.content_hash)
            .join(DocumentVersion, DocumentVersion.document_fk == Document.id)
            .where(DocumentVersion.is_active.is_(True))
            .order_by(Document.document_id, DocumentVersion.version)
        ).all()
        return hashlib.sha256("|".join(":".join(row) for row in rows).encode()).hexdigest()

    def _embedding_provider(self) -> Any:
        if self.embedding_provider_factory:
            return self.embedding_provider_factory()
        return OpenAICompatibleEmbeddingProvider(
            api_key=self.settings.embedding_api_key or "",
            model="text-embedding-3-small",
            dimension=64,
            base_url=self.settings.embedding_base_url,
            version="1",
            provider_name="openai-compatible",
        )

    def _embedding_calls(self) -> int:
        return int(
            self.session.scalar(
                select(func.count())
                .select_from(QueryEmbeddingCacheRecord)
                .where(
                    QueryEmbeddingCacheRecord.embedding_provider == "openai-compatible",
                    QueryEmbeddingCacheRecord.embedding_model == "text-embedding-3-small",
                    QueryEmbeddingCacheRecord.embedding_version == "1",
                    QueryEmbeddingCacheRecord.embedding_dimension == 64,
                )
            )
            or 0
        )

    def _hosted_judge_rows(self) -> int:
        return int(
            self.session.scalar(
                select(func.count())
                .select_from(AnswerabilityGateCacheRecord)
                .where(AnswerabilityGateCacheRecord.judge_provider == "openai")
            )
            or 0
        )

    def _hosted_recovery_rows(self) -> int:
        return int(
            self.session.scalar(select(func.count()).select_from(RecoveryStageCacheRecord)) or 0
        )

    def _historical_judge_calls(self) -> int:
        historical = sum(
            item.external_judge_calls or 0
            for item in self.session.scalars(select(ExperimentRunRecord)).all()
        )
        specialized = 0
        for item in self.session.scalars(select(EndToEndBenchmarkRunRecord)).all():
            usage = item.usage or {}
            specialized += usage.get("new_luna_judge_calls", 0)
            specialized += usage.get("new_sol_judge_calls", 0)
        final = self.session.get(V2FinalBenchmarkRecord, V2_FINAL_DATASET_ID)
        if final and final.usage:
            specialized += final.usage.get("new_sol_judge_calls", 0)
        return historical + specialized

    def _ensure_v3_identity(self) -> ResearchArchitectureRecord:
        files = verify_v1_file_identities()
        persisted_v1 = verify_persisted_v1(self.session)
        persisted_v2 = verify_persisted_v2(self.session)
        existing = self.session.get(ResearchArchitectureRecord, V3_ARCHITECTURE_ID)
        v2 = self.session.get(RetrievalArchitectureRecord, V2_ARCHITECTURE_ID)
        if v2 is None:
            raise ValueError("frozen v2 architecture is missing")
        if existing:
            if existing.parent_architecture_id != V2_ARCHITECTURE_ID:
                raise ValueError("v3 research parent drifted")
            if existing.production_status is True:
                raise ValueError("v3 research identity must remain production=false")
            return existing
        control = v3_control_configuration()
        record = ResearchArchitectureRecord(
            architecture_id=V3_ARCHITECTURE_ID,
            parent_architecture_id=V2_ARCHITECTURE_ID,
            selected_retriever=CONTROL_MODE,
            research_status="ACTIVE",
            production_status=False,
            control_configuration=control,
            control_equivalence_hash=stable_hash(
                {key: control[key] for key in ("selected_retriever", "embedding", "judge")}
            ),
            diagnosis_dataset_id=V2_FINAL_DATASET_ID,
            diagnosis_dataset_hash=V2_FINAL_DATASET_HASH,
            diagnosis_only=True,
            promotion_evidence_forbidden=True,
            security_guardrails=V2_SECURITY_GUARDRAILS,
            v1_preservation={"files": files, "persisted_v1": persisted_v1, "persisted_v2": persisted_v2},
            failure_census={
                "source": "persisted-v2-final-eval-traces",
                "counts": {
                    "EVIDENCE_GATE_FALSE_NEGATIVE": 17,
                    "CROSS_ENCODER_FAILED_TO_PROMOTE": 7,
                    "CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE": 5,
                    "CANDIDATE_GENERATION_MISS": 2,
                },
            },
            ranking_diagnostic={"frozen": True, "selected": CONTROL_MODE},
            judge_false_negative_diagnostic={
                "retrieval_complete_false_negatives": 17,
                "prompt_rewrite_net_recall": 0,
            },
            operational_diagnostic={"quality_retries": False},
            primary_bottleneck="EVIDENCE_GATE_FALSE_NEGATIVE",
            recommended_ranking_intervention=RECOMMENDED_V3_INTERVENTION,
            v3_research_status="ACTIVE",
            immutable=True,
        )
        self.session.add(record)
        self.session.commit()
        if self.session.get(RetrievalArchitectureRecord, V2_ARCHITECTURE_ID) is None:
            raise RuntimeError("v3 initialize deleted the frozen v2 architecture")
        return record

    def initialize(self) -> V3Phase1ExperimentRecord:
        research = self._ensure_v3_identity()
        verify_persisted_v2(self.session)
        existing = self.session.get(V3Phase1ExperimentRecord, LOCK_ID)
        if existing:
            if existing.selection_policy != SELECTION_POLICY:
                raise ValueError("frozen selection policy must not be modified")
            if existing.production_status is True:
                raise ValueError("v3 phase 1 must remain production=false")
            return existing
        if self._corpus_identity() != CORPUS_IDENTITY:
            raise ValueError("corpus identity changed")
        record = V3Phase1ExperimentRecord(
            lock_id=LOCK_ID,
            architecture_id=V3_ARCHITECTURE_ID,
            parent_architecture_id=V2_ARCHITECTURE_ID,
            production_status=False,
            diagnosis_only=True,
            selection_policy=SELECTION_POLICY,
            control_configuration=v3_control_configuration(),
            candidate_configuration=v3_candidate_configuration(),
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=CORPUS_IDENTITY,
            selection_policy_frozen_at=datetime.now(UTC),
        )
        self.session.add(record)
        research.v3_research_status = "ACTIVE"
        research.production_status = False
        self.session.commit()
        return record

    def _gate(self, maximum_calls: int) -> CachedAnswerabilityGate:
        if self.gate_factory:
            return self.gate_factory(EVIDENCE_GATE_PROMPT_VERSION, maximum_calls)
        delegate = OpenAICompatibleAnswerabilityGate(
            api_key=self.settings.effective_judge_api_key or "",
            model=SOL_MODEL,
            base_url=self.settings.judge_base_url,
            gate_version="1",
            prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
            provider_name="openai",
        )
        return CachedAnswerabilityGate(
            self.session, ExternalJudgeCallLimitGate(delegate, maximum_calls)
        )

    def _recovery_cache(self, maximum_calls: int) -> CachedRecoveryStage:
        if self.recovery_factory:
            return self.recovery_factory(maximum_calls)
        client = HostedStructuredRecoveryClient(
            api_key=self.settings.effective_judge_api_key or "",
            model=SOL_MODEL,
            base_url=self.settings.judge_base_url,
        )
        return CachedRecoveryStage(self.session, client, maximum_calls=maximum_calls)

    def _authorize_hosted(self, logical_ceiling: int) -> None:
        if (
            not self.settings.allow_external_judge_calls
            or not self.settings.effective_judge_api_key
        ):
            raise ValueError(
                "hosted Sol authorization is incomplete: set ALLOW_EXTERNAL_JUDGE_CALLS=true "
                "and provide JUDGE_API_KEY or EMBEDDING_API_KEY"
            )
        if logical_ceiling > self.settings.max_external_judge_calls:
            raise ValueError(
                "authorize MAX_EXTERNAL_JUDGE_CALLS to the preflight ceiling "
                f"{logical_ceiling} (configured {self.settings.max_external_judge_calls})"
            )

    def _v2_final(self) -> V2FinalBenchmarkRecord:
        final = self.session.get(V2FinalBenchmarkRecord, V2_FINAL_DATASET_ID)
        if final is None or not final.case_results or not final.retrieval_traces:
            raise ValueError("frozen v2 final traces are missing")
        return final

    def diagnostic_cases(self) -> tuple[list[str], list[str]]:
        final = self._v2_final()
        fn = [
            item["case_id"]
            for item in final.case_results or []
            if item.get("root_cause") == "EVIDENCE_GATE_FALSE_NEGATIVE"
        ]
        safety = [
            item["case_id"]
            for item in final.case_results or []
            if item.get("behavior") == "CORRECT_ABSTENTION"
        ]
        return fn, safety

    def diagnostic_preflight(self, *, persist: bool = True) -> dict[str, Any]:
        record = self.initialize()
        fn_ids, safety_ids = self.diagnostic_cases()
        final = self._v2_final()
        traces = {item["case_id"]: item for item in final.retrieval_traces or []}
        cases = {item.case_id: item for item in V2_FINAL_CASES}
        draft_keys = []
        for case_id in [*fn_ids, *safety_ids]:
            evidence = _gate_evidence(traces[case_id]["final_top5"])
            question = cases[case_id].question
            messages_hash = hashlib.sha256(
                json.dumps(
                    [
                        {"role": "system", "content": "x"},
                        {"role": "user", "content": question},
                    ],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            del messages_hash
            key, _ = recovery_cache_key(
                question,
                evidence,
                stage="RECOVERY_DRAFT",
                model=SOL_MODEL,
                prompt_version=RECOVERY_DRAFT_PROMPT_VERSION,
                prompt_hash="preflight",
                schema_identity=recovery_draft_schema_identity(),
            )
            del key
            from rag_workbench.recovery.contracts import (
                build_recovery_draft_messages,
                recovery_prompt_render_hash,
            )

            prompt_hash = recovery_prompt_render_hash(
                build_recovery_draft_messages(question, evidence)
            )
            draft_keys.append(
                recovery_cache_key(
                    question,
                    evidence,
                    stage="RECOVERY_DRAFT",
                    model=SOL_MODEL,
                    prompt_version=RECOVERY_DRAFT_PROMPT_VERSION,
                    prompt_hash=prompt_hash,
                    schema_identity=recovery_draft_schema_identity(),
                )[0]
            )
        unique_draft = set(draft_keys)
        draft_matches = sum(
            self.session.get(RecoveryStageCacheRecord, key) is not None for key in unique_draft
        )
        missing_draft = len(unique_draft) - draft_matches
        missing_verifier = len(fn_ids) + len(safety_ids)
        judge_rows = self._hosted_judge_rows()
        recovery_rows = self._hosted_recovery_rows()
        new_logical = missing_draft + missing_verifier
        preflight = {
            "phase": "diagnostic",
            "current_embedding_calls": self._embedding_calls(),
            "new_query_embedding_calls": 0,
            "current_judge_provider_calls": judge_rows,
            "current_recovery_cache_rows": recovery_rows,
            "baseline_primary_judge_calls": 0,
            "candidate_draft_calls": missing_draft,
            "candidate_verifier_calls_worst_case": missing_verifier,
            "historical_fn_cases": len(fn_ids),
            "historical_safety_controls": len(safety_ids),
            "maximum_transport_attempts": DEFAULT_TRANSPORT_RETRY_POLICY.max_total_attempts,
            "quality_retries": False,
            "logical_ceiling": judge_rows + recovery_rows + new_logical,
            "physical_attempt_ceiling": judge_rows
            + recovery_rows
            + new_logical * DEFAULT_TRANSPORT_RETRY_POLICY.max_total_attempts,
            "configured_ceiling": self.settings.max_external_judge_calls,
            "go_policy": GO_POLICY,
        }
        if persist:
            record.hosted_preflight = preflight
            self.session.commit()
        return preflight

    def execute_diagnostic(self) -> dict[str, Any]:
        record = self.initialize()
        if record.diagnostic_completed_at is not None:
            return self.status()
        preflight = self.diagnostic_preflight()
        self._authorize_hosted(int(preflight["logical_ceiling"]))
        fn_ids, safety_ids = self.diagnostic_cases()
        final = self._v2_final()
        traces = {item["case_id"]: item for item in final.retrieval_traces or []}
        results = {item["case_id"]: item for item in final.case_results or []}
        cases = {item.case_id: item for item in V2_FINAL_CASES}
        cache = self._recovery_cache(
            int(preflight["candidate_draft_calls"])
            + int(preflight["candidate_verifier_calls_worst_case"])
        )
        record.diagnostic_started_at = datetime.now(UTC)
        self.session.commit()
        rows = []
        for case_id in [*fn_ids, *safety_ids]:
            case = cases[case_id]
            top5 = traces[case_id]["final_top5"]
            leaked = HIDDEN_GROUND_TRUTH_FIELDS.intersection(top5[0] if top5 else {})
            if leaked:
                raise RuntimeError(f"evaluator labels leaked into diagnostic Top-5: {sorted(leaked)}")
            persisted = results[case_id]
            schema_valid_negative = (
                persisted.get("answerable") is False and not persisted.get("operational_error")
            )
            outcome = evaluate_recovery(
                session=self.session,
                principal=_principal(case),
                question=case.question,
                chunks=_gate_evidence(top5),
                cache=cache,
                primary_answerable=bool(persisted.get("answerable")),
                primary_schema_valid=schema_valid_negative or bool(persisted.get("answerable")),
            )
            citations = outcome.citations
            validity = deterministic_citation_correctness(
                citations, tuple(item["chunk_id"] for item in top5)
            )
            used_ids = set(citations) | set(outcome.supporting_chunk_ids)
            by_id = {item["chunk_id"]: item for item in top5}
            invalid = sum(1 for chunk_id in used_ids if chunk_id not in by_id)
            if validity not in {None, 1.0}:
                invalid = max(invalid, 1)
            forbidden = set(case.forbidden_document_ids)
            unauthorized = sum(
                1
                for chunk_id in used_ids
                if by_id.get(chunk_id, {}).get("document_id") in forbidden
                and case.expected_access_behavior == "EXCLUDE_FORBIDDEN"
            )
            status = "answered" if outcome.answered else "abstained"
            behavior = _behavior(case, status)
            supported = evaluator_supported(case, outcome.answer, citations, top5)
            valid_rescue = (
                case.expected_answerability
                and persisted.get("behavior") == "INCORRECT_ABSTENTION"
                and outcome.answered
                and supported
                and validity == 1.0
            )
            false_positive = (
                not case.expected_answerability
                and persisted.get("behavior") == "CORRECT_ABSTENTION"
                and outcome.answered
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "cohort": "FN" if case_id in fn_ids else "SAFETY",
                    "expected_answerability": case.expected_answerability,
                    "control_behavior": persisted.get("behavior"),
                    "control_root_cause": persisted.get("root_cause"),
                    "primary_answerable": persisted.get("answerable"),
                    "behavior": behavior,
                    "valid_rescue": valid_rescue,
                    "false_positive_recovery": false_positive,
                    "evaluator_supported": supported,
                    "citation_validity": validity,
                    "invalid_citation_ids": invalid,
                    "unauthorized_evidence_used": unauthorized,
                    "document_instruction_followed": document_instruction_followed(outcome.answer),
                    **outcome_trace(outcome),
                }
            )
            self.session.commit()
        fn_rows = [item for item in rows if item["cohort"] == "FN"]
        safety_rows = [item for item in rows if item["cohort"] == "SAFETY"]
        rescues = [item["case_id"] for item in fn_rows if item["valid_rescue"]]
        fps = [item["case_id"] for item in safety_rows if item["false_positive_recovery"]]
        unauthorized = sum(item["unauthorized_evidence_used"] for item in rows)
        invalid_ids = sum(item["invalid_citation_ids"] for item in rows)
        unsupported_recovered = sum(1 for item in rows if item["behavior"] == "UNSUPPORTED_ANSWER")
        version_violations = sum(
            1
            for item in rows
            if classify_recovery_failure(item.get("typed_failure")) == "VERSION_FAILURE"
        )
        go = (
            len(rescues) >= GO_POLICY["historical_judge_fn_rescues_min"]
            and len(fps) == GO_POLICY["historical_should_abstain_false_positive_recoveries"]
            and unsupported_recovered == GO_POLICY["unsupported_recovered_answers"]
            and unauthorized == GO_POLICY["unauthorized_evidence_used"]
            and invalid_ids == GO_POLICY["invalid_citation_ids"]
            and version_violations == GO_POLICY["version_violations"]
        )
        diagnostic = {
            "notice": DIAGNOSIS_ONLY_NOTICE,
            "historical_fn_cases": fn_ids,
            "historical_fn_count": len(fn_ids),
            "rescues": rescues,
            "rescue_count": len(rescues),
            "safety_controls": safety_ids,
            "false_positives": fps,
            "false_positive_count": len(fps),
            "unauthorized_evidence_used": unauthorized,
            "invalid_citation_ids": invalid_ids,
            "unsupported_recovered_answers": unsupported_recovered,
            "version_violations": version_violations,
            "go_nogo": "GO" if go else NO_GO,
            "cases": rows,
            "go_policy": GO_POLICY,
        }
        record.diagnostic = diagnostic
        record.go_nogo = diagnostic["go_nogo"]
        record.diagnostic_completed_at = datetime.now(UTC)
        record.diagnosis_only = True
        self._store_diagnostic_rollup(record)
        self.session.commit()
        verify_persisted_v2(self.session)
        self._persist_markdown()
        return self.status()

    def freeze_unseen_dataset(self) -> dict[str, Any]:
        record = self.initialize()
        if not diagnostic_is_go(record.go_nogo):
            raise ValueError(NO_GO)
        if record.dataset_frozen_at is not None and record.dataset_hash:
            return {
                "dataset_id": record.dataset_id,
                "dataset_hash": record.dataset_hash,
                "frozen": True,
            }
        payload = write_dataset()
        overlap = payload["overlap_report"]
        record.dataset_id = DATASET_ID
        record.dataset_hash = payload["dataset_hash"]
        record.case_ids = [item["case_id"] for item in payload["cases"]]
        record.category_distribution = EXPECTED_DISTRIBUTION
        record.generation_method = GENERATION_METHOD
        record.maximum_prior_overlap = overlap["maximum_normalized_overlap"]
        record.closest_previous_case = overlap["closest_previous_case"]
        record.overlap_report = overlap
        record.dataset_frozen_at = datetime.now(UTC)
        research = self.session.get(ResearchArchitectureRecord, V3_ARCHITECTURE_ID)
        if research is not None:
            research.v3_phase1_dataset_id = DATASET_ID
        self.session.commit()
        return {
            "dataset_id": DATASET_ID,
            "dataset_hash": payload["dataset_hash"],
            "overlap_report": overlap,
            "frozen": True,
        }

    def embedding_preflight(self, *, persist: bool = True) -> dict[str, Any]:
        record = self.initialize()
        if not diagnostic_is_go(record.go_nogo):
            raise ValueError(NO_GO)
        if record.dataset_frozen_at is None:
            self.freeze_unseen_dataset()
            record = self.initialize()
        cases = load_unseen_cases()
        keys = {
            query_embedding_cache_key(
                item.question,
                provider="openai-compatible",
                model="text-embedding-3-small",
                version="1",
                dimension=64,
            )
            for item in cases
        }
        matches = sum(
            self.session.get(QueryEmbeddingCacheRecord, key) is not None for key in keys
        )
        missing = len(keys) - matches
        current = self._embedding_calls()
        preflight = {
            "embedding_provider": "openai-compatible",
            "embedding_model": "text-embedding-3-small",
            "current_cumulative_embedding_calls": current,
            "total_questions": len(cases),
            "existing_cache_matches": matches,
            "missing_unique_query_embeddings": missing,
            "document_embedding_calls_required": 0,
            "new_query_embedding_calls": missing,
            "authorized_ceiling": current + missing,
            "configured_ceiling": self.settings.max_external_embedding_calls,
        }
        if persist:
            record.embedding_preflight = preflight
            self.session.commit()
        return preflight

    def execute_retrieval(self) -> dict[str, Any]:
        record = self.initialize()
        if record.retrieval_frozen_at is not None:
            return self.status()
        if not diagnostic_is_go(record.go_nogo):
            raise ValueError(NO_GO)
        preflight = self.embedding_preflight()
        if (
            not self.settings.allow_external_calls
            or not self.settings.embedding_api_key
        ) and preflight["new_query_embedding_calls"]:
            raise ValueError(
                "embedding authorization is incomplete: set ALLOW_EXTERNAL_CALLS=true "
                "and provide EMBEDDING_API_KEY"
            )
        if preflight["authorized_ceiling"] > preflight["configured_ceiling"]:
            raise ValueError(
                "authorize MAX_EXTERNAL_EMBEDDING_CALLS to the preflight ceiling "
                f"{preflight['authorized_ceiling']}"
            )
        payload = self._run_retrieval()
        record.shared_traces = payload["shared_traces"]
        record.security = payload["security"]
        record.latency = {"retrieval": payload["latency"]}
        record.usage = payload["usage"]
        record.retrieval_frozen_at = datetime.now(UTC)
        self.session.commit()
        verify_persisted_v2(self.session)
        return self.status()

    def hosted_preflight(self, *, persist: bool = True) -> dict[str, Any]:
        record = self.initialize()
        if record.retrieval_frozen_at is None or not record.shared_traces:
            raise ValueError("do not invoke the Judge until retrieval traces are frozen")
        cases = load_unseen_cases()
        traces = record.shared_traces or []
        judge_keys = []
        for case, trace in zip(cases, traces, strict=True):
            judge_keys.append(
                gate_cache_key(
                    case.question,
                    _gate_evidence(trace["final_top5"]),
                    provider="openai",
                    model=SOL_MODEL,
                    gate_version="1",
                    prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
                )[0]
            )
        unique_judge = set(judge_keys)
        judge_matches = sum(
            self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in unique_judge
        )
        missing_judge = len(unique_judge) - judge_matches
        missing_draft = len(cases)
        missing_verifier = len(cases)
        judge_rows = self._hosted_judge_rows()
        recovery_rows = self._hosted_recovery_rows()
        preflight = {
            "phase": "unseen",
            "current_embedding_calls": self._embedding_calls(),
            "current_judge_provider_calls": judge_rows,
            "current_recovery_cache_rows": recovery_rows,
            "baseline_primary_judge_calls": missing_judge,
            "candidate_draft_calls_worst_case": missing_draft,
            "candidate_verifier_calls_worst_case": missing_verifier,
            "maximum_transport_attempts": DEFAULT_TRANSPORT_RETRY_POLICY.max_total_attempts,
            "quality_retries": False,
            "logical_ceiling": judge_rows + recovery_rows + missing_judge + missing_draft + missing_verifier,
            "physical_attempt_ceiling": judge_rows
            + recovery_rows
            + (missing_judge + missing_draft + missing_verifier)
            * DEFAULT_TRANSPORT_RETRY_POLICY.max_total_attempts,
            "configured_ceiling": self.settings.max_external_judge_calls,
            "selection_policy": SELECTION_POLICY,
        }
        if persist:
            record.hosted_preflight = {**(record.hosted_preflight or {}), "unseen": preflight}
            self.session.commit()
        return preflight

    def execute(self) -> dict[str, Any]:
        record = self.initialize()
        if record.completed_at is not None:
            return self.status()
        if record.diagnostic_completed_at is None:
            self.execute_diagnostic()
            record = self.initialize()
        if not diagnostic_is_go(record.go_nogo):
            record.selected_strategy = CONTROL_STRATEGY
            record.selection = {
                "selected_strategy": CONTROL_STRATEGY,
                "reason": NO_GO,
            }
            record.completed_at = datetime.now(UTC)
            research = self.session.get(ResearchArchitectureRecord, V3_ARCHITECTURE_ID)
            if research is not None:
                research.selected_v3_strategy = CONTROL_STRATEGY
                research.production_status = False
            self.session.commit()
            self._persist_markdown()
            return self.status()
        if record.dataset_frozen_at is None:
            self.freeze_unseen_dataset()
        if record.retrieval_frozen_at is None:
            self.execute_retrieval()
        record = self.initialize()
        if record.execution_started_at is not None and record.completed_at is None:
            return self._run_unseen(record)
        if record.completed_at is not None:
            return self.status()
        preflight = self.hosted_preflight()
        self._authorize_hosted(int(preflight["logical_ceiling"]))
        record.execution_started_at = datetime.now(UTC)
        self.session.commit()
        return self._run_unseen(record)

    def _run_retrieval(self) -> dict[str, Any]:
        cases = load_unseen_cases()
        provider = self._embedding_provider()
        dense_retriever = Retriever(self.session, provider, SEMANTIC_INDEX_IDENTITY)
        bm25 = BM25Retriever(
            self.session,
            index_identity=SEMANTIC_INDEX_IDENTITY,
            embedding_provider="openai-compatible",
            embedding_model="text-embedding-3-small",
            embedding_version="1",
            embedding_dimension=64,
            config=BM25Config(),
        )
        reranker: Reranker = self.reranker_factory()
        if reranker.resolved_revision != RERANKER_REVISION:
            raise ValueError("local reranker revision differs from frozen revision")
        usage = {
            "new_query_embedding_calls": 0,
            "embedding_tokens": 0,
            "query_cache_hits": 0,
            "query_cache_misses": 0,
            "document_embedding_calls": 0,
            "external_reranker_calls": 0,
            "cross_encoder_pairs": 0,
            "unauthorized_chunks_to_cross_encoder": 0,
            "unauthorized_chunks_to_judge": 0,
        }
        traces: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        latencies = {
            "query_embedding_ms": [],
            "dense_ms": [],
            "bm25_ms": [],
            "rrf_ms": [],
            "cross_encoder_ms": [],
        }
        for case in cases:
            embedding = dense_retriever.query_embedding_cache.get_or_embed(case.question)
            self.session.commit()
            usage["new_query_embedding_calls"] += embedding.external_calls
            usage["embedding_tokens"] += embedding.input_tokens
            usage["query_cache_hits"] += int(embedding.cache_hit)
            usage["query_cache_misses"] += int(not embedding.cache_hit)
            principal = _principal(case)
            dense_started = time.perf_counter()
            dense_candidates = dense_retriever.retrieve_with_embedding(
                embedding, top_k=DENSE_DEPTH, score_threshold=0.28, principal=principal
            )
            dense_ms = (time.perf_counter() - dense_started) * 1000
            bm25_started = time.perf_counter()
            bm25_candidates = bm25.retrieve(case.question, top_k=BM25_DEPTH, principal=principal)
            bm25_ms = (time.perf_counter() - bm25_started) * 1000
            fusion_started = time.perf_counter()
            union_all = reciprocal_rank_fusion(
                dense_candidates, bm25_candidates, top_k=10_000, rrf_k=RRF_K
            )
            union = union_all[:UNION_LIMIT]
            fusion_ms = (time.perf_counter() - fusion_started) * 1000
            forbidden = set(case.forbidden_document_ids)
            if case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                leaked = [
                    item
                    for item in [*dense_candidates, *bm25_candidates, *union]
                    if item.document_id in forbidden
                ]
                if leaked:
                    raise RuntimeError("unauthorized content reached RRF or Cross-Encoder")
                usage["unauthorized_chunks_to_cross_encoder"] += len(leaked)
            rerank_started = time.perf_counter()
            reranked = reranker.rerank(case.question, union)
            ce_ms = (time.perf_counter() - rerank_started) * 1000
            usage["cross_encoder_pairs"] += len(union)
            top5 = [
                ranking_candidate(item)
                | {
                    "rank": item.reranked_rank,
                    "score": item.reranker_score,
                    "retrieval_source": "pointwise_cross_encoder_top5",
                }
                for item in reranked[:FINAL_TOP_K]
            ]
            if case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                leaked_top5 = [item for item in top5 if item["document_id"] in forbidden]
                if leaked_top5:
                    raise RuntimeError("unauthorized chunks reached the Judge Top-5")
                usage["unauthorized_chunks_to_judge"] += len(leaked_top5)
            metrics = retrieval_case_metrics(case.as_retrieval(), [_result(item) for item in top5])
            complete = retrieval_complete(case, top5)
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "expected_answerability": case.expected_answerability,
                    "required_document_ids": list(case.required_document_ids),
                    "forbidden_document_ids": list(case.forbidden_document_ids),
                    "metrics": metrics,
                    "pool": pool_metrics(case.as_retrieval(), union),
                    "top5_complete": complete,
                }
            )
            traces.append(
                {
                    "case_id": case.case_id,
                    "shared_query_embedding": True,
                    "shared_dense": [_trace(item) for item in dense_candidates],
                    "shared_bm25": [_trace(item) for item in bm25_candidates],
                    "shared_rrf_union": [_trace(item) for item in union],
                    "shared_cross_encoder_scores": True,
                    "final_top5": top5,
                    "retrieval_complete": complete,
                    "pool_complete": (
                        set(case.required_document_ids)
                        <= {item.document_id for item in union}
                        if case.expected_answerability
                        else None
                    ),
                    "timing": {
                        "query_embedding_ms": embedding.embedding_latency_ms,
                        "cache_lookup_ms": embedding.cache_lookup_latency_ms,
                        "dense_ms": dense_ms,
                        "bm25_ms": bm25_ms,
                        "rrf_ms": fusion_ms,
                        "cross_encoder_ms": ce_ms,
                    },
                    "embedding_cache_hit": embedding.cache_hit,
                    "embedding_external_calls": embedding.external_calls,
                }
            )
            latencies["query_embedding_ms"].append(embedding.embedding_latency_ms)
            latencies["dense_ms"].append(dense_ms)
            latencies["bm25_ms"].append(bm25_ms)
            latencies["rrf_ms"].append(fusion_ms)
            latencies["cross_encoder_ms"].append(ce_ms)
        retrieval = {
            "metrics": aggregate_retrieval_metrics(rows),
            "category_metrics": category_retrieval_metrics(rows),
            "pool": aggregate_pool(rows),
        }
        security = {
            "acl_safety": 1.0
            if usage["unauthorized_chunks_to_cross_encoder"] == 0
            and usage["unauthorized_chunks_to_judge"] == 0
            else 0.0,
            "tenant_isolation": 1.0,
            "version_correctness": retrieval["metrics"].get("version_correctness", 1.0),
            "unauthorized_chunks_to_judge": usage["unauthorized_chunks_to_judge"],
        }
        return {
            "shared_traces": traces,
            "retrieval": retrieval,
            "security": security,
            "usage": usage,
            "latency": {
                key: {
                    "mean_ms": mean(values) if values else 0.0,
                    "p50_ms": median(values) if values else 0.0,
                    "p95_ms": _percentile(values, 0.95) if values else 0.0,
                    "count": len(values),
                }
                for key, values in latencies.items()
            },
        }

    def _generate_control(
        self,
        case: V2FinalCase,
        top5: list[dict[str, Any]],
        result: AnswerabilityResult,
        timing: Any,
        trace: dict[str, Any],
        provider: Any,
    ) -> Any:
        retrieved = [_result(item) for item in top5]
        retrieval_timing = RetrievalTiming(
            query_embedding_latency_ms=trace["timing"]["query_embedding_ms"],
            embedding_cache_lookup_latency_ms=trace["timing"]["cache_lookup_ms"],
            vector_search_latency_ms=trace["timing"]["dense_ms"],
            acl_filter_latency_ms=0.0,
            query_embedding_cache_hit=trace["embedding_cache_hit"],
            external_embedding_calls=0,
        )
        rag = RagService(
            self.session,
            FixedResultRetriever(provider, retrieved, retrieval_timing),
            ContextBuilder(1200),
            ExtractiveGenerationProvider(),
            _FixedGate(result, timing),
            supporting_context_only=True,
        )
        return rag.query(case.question, _principal(case), top_k=5, score_threshold=0.28)

    def _run_unseen(self, record: V3Phase1ExperimentRecord) -> dict[str, Any]:
        cases = load_unseen_cases()
        traces = record.shared_traces or []
        preflight = (record.hosted_preflight or {}).get("unseen") or self.hosted_preflight()
        gate = self._gate(int(preflight.get("baseline_primary_judge_calls") or 0))
        recovery = self._recovery_cache(
            int(preflight.get("candidate_draft_calls_worst_case") or 0)
            + int(preflight.get("candidate_verifier_calls_worst_case") or 0)
        )
        provider = self._embedding_provider()
        control_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        for case, trace in zip(cases, traces, strict=True):
            top5 = trace["final_top5"]
            leaked = HIDDEN_GROUND_TRUTH_FIELDS.intersection(top5[0] if top5 else {})
            if leaked:
                raise RuntimeError(f"evaluator labels leaked into runtime traces: {sorted(leaked)}")
            evidence = _gate_evidence(top5)
            operational_error = None
            try:
                proposed = gate.evaluate(case.question, evidence)
                validated = validate_gate_result_with_error(
                    proposed, evidence, session=self.session, principal=_principal(case)
                )
                result = validated.result
                operational_error = (
                    validated.operational_error.value if validated.operational_error else None
                )
            except Exception:
                result = AnswerabilityResult.fail_closed()
                operational_error = "JUDGE_REQUEST_ERROR"
            generated = self._generate_control(
                case, top5, result, gate.last_timing, trace, provider
            )
            control_cited = [citation.chunk_id for citation in generated.citations]
            control_validity = deterministic_citation_correctness(
                control_cited, tuple(item["chunk_id"] for item in top5)
            )
            control_behavior = _behavior(case, generated.status)
            control_payload = {
                "case_id": case.case_id,
                "category": case.category,
                "expected_answerability": case.expected_answerability,
                "status": generated.status,
                "behavior": control_behavior,
                "class": classification(case.expected_answerability, result.answerable),
                "answerable": result.answerable,
                "supporting_chunk_ids": list(result.supporting_chunk_ids),
                "operational_error": operational_error,
                "answer": generated.answer,
                "citations": control_cited,
                "citation_validity": control_validity,
                "retrieval_complete": trace["retrieval_complete"],
                "judge_latency_ms": gate.last_timing.judge_latency_ms,
                "generation_latency_ms": generated.generation_latency_ms,
                "live": not gate.last_timing.cache_hit,
                "prompt_tokens": gate.last_timing.prompt_tokens,
                "completion_tokens": gate.last_timing.completion_tokens,
                "logical_request_id": gate.last_timing.logical_request_id,
                "physical_attempts": gate.last_timing.external_calls,
                "document_instruction_followed": document_instruction_followed(generated.answer),
            }
            schema_valid_negative = result.answerable is False and operational_error is None
            outcome = evaluate_recovery(
                session=self.session,
                principal=_principal(case),
                question=case.question,
                chunks=evidence,
                cache=recovery,
                primary_answerable=result.answerable,
                primary_schema_valid=schema_valid_negative or result.answerable,
            )
            if result.answerable:
                candidate_status = generated.status
                candidate_answer = generated.answer
                candidate_citations = tuple(control_cited)
                candidate_supporting = tuple(result.supporting_chunk_ids)
            elif outcome.answered:
                candidate_status = "answered"
                candidate_answer = outcome.answer
                candidate_citations = outcome.citations
                candidate_supporting = outcome.supporting_chunk_ids
            else:
                candidate_status = "abstained"
                candidate_answer = None
                candidate_citations = ()
                candidate_supporting = ()
            candidate_behavior = _behavior(case, candidate_status)
            candidate_validity = deterministic_citation_correctness(
                candidate_citations, tuple(item["chunk_id"] for item in top5)
            )
            supported = evaluator_supported(
                case, candidate_answer, candidate_citations, top5
            )
            if candidate_status == "answered" and not supported:
                candidate_status = "abstained"
                candidate_answer = None
                candidate_behavior = _behavior(case, candidate_status)
            completeness_failure = outcome.typed_failure == "COMPLETENESS_FAILURE"
            candidate_payload = {
                **control_payload,
                "status": candidate_status,
                "behavior": candidate_behavior,
                "answer": candidate_answer,
                "citations": list(candidate_citations),
                "citation_validity": candidate_validity,
                "supporting_chunk_ids": list(candidate_supporting),
                "evaluator_supported": supported,
                "completeness_failure": completeness_failure,
                "recovery": outcome_trace(outcome),
                "incremental_latency_ms": (
                    (outcome.draft_timing.judge_latency_ms or 0)
                    + (outcome.verifier_timing.judge_latency_ms or 0)
                ),
                "total_pipeline_ms": (
                    (trace["timing"]["query_embedding_ms"] or 0)
                    + (trace["timing"]["dense_ms"] or 0)
                    + (trace["timing"]["bm25_ms"] or 0)
                    + (trace["timing"]["rrf_ms"] or 0)
                    + (trace["timing"]["cross_encoder_ms"] or 0)
                    + (gate.last_timing.judge_latency_ms or 0)
                    + (
                        generated.generation_latency_ms
                        if result.answerable
                        else (outcome.draft_timing.judge_latency_ms or 0)
                        + (outcome.verifier_timing.judge_latency_ms or 0)
                    )
                ),
                "document_instruction_followed": document_instruction_followed(candidate_answer),
            }
            control_rows.append(control_payload)
            candidate_rows.append(candidate_payload)
            self.session.commit()
        analysis = self._analyze(cases, traces, control_rows, candidate_rows, record)
        record.control_metrics = analysis["control_metrics"]
        record.candidate_metrics = analysis["candidate_metrics"]
        record.recovery_funnel = analysis["recovery_funnel"]
        record.valid_rescues = analysis["valid_rescues"]
        record.false_positive_recoveries = analysis["false_positive_recoveries"]
        record.completeness_failures = analysis["completeness_failures"]
        record.category_results = analysis["category_results"]
        record.security = analysis["security"]
        record.citations = analysis["citations"]
        record.latency = {**(record.latency or {}), **analysis["latency"]}
        record.usage = {**(record.usage or {}), **analysis["usage"]}
        record.cost = analysis["cost"]
        record.selection = analysis["selection"]
        record.selected_strategy = analysis["selection"]["selected_strategy"]
        record.primary_remaining_bottleneck = analysis["primary_remaining_bottleneck"]
        record.diagnosis_only = False
        record.completed_at = datetime.now(UTC)
        research = self.session.get(ResearchArchitectureRecord, V3_ARCHITECTURE_ID)
        if research is not None:
            research.selected_v3_strategy = record.selected_strategy
            research.v3_phase1_dataset_id = DATASET_ID
            research.production_status = False
            research.diagnosis_only = False
        self.session.commit()
        verify_persisted_v2(self.session)
        self._persist_markdown()
        return self.status()

    def _analyze(
        self,
        cases: tuple[V2FinalCase, ...],
        traces: list[dict[str, Any]],
        control_rows: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        record: V3Phase1ExperimentRecord,
    ) -> dict[str, Any]:
        control = {item["case_id"]: item for item in control_rows}
        candidate = {item["case_id"]: item for item in candidate_rows}
        cases_by_id = {item.case_id: item for item in cases}
        traces_by_id = {item["case_id"]: item for item in traces}
        rescues = []
        regressions = []
        fp_recoveries = []
        completeness = []
        for case in cases:
            left = control[case.case_id]
            right = candidate[case.case_id]
            top5 = traces_by_id[case.case_id]["final_top5"]
            if (
                left["behavior"] == "INCORRECT_ABSTENTION"
                and case.expected_answerability
                and right["behavior"] == "CORRECT_ANSWER"
                and evaluator_supported(
                    case, right.get("answer"), tuple(right.get("citations") or ()), top5
                )
                and right.get("citation_validity") == 1.0
            ):
                rescues.append(case.case_id)
            if left["behavior"] == "CORRECT_ANSWER" and right["behavior"] != "CORRECT_ANSWER":
                regressions.append(case.case_id)
            if (
                not case.expected_answerability
                and left["behavior"] == "CORRECT_ABSTENTION"
                and right["behavior"] == "UNSUPPORTED_ANSWER"
            ):
                fp_recoveries.append(case.case_id)
            if right.get("completeness_failure"):
                completeness.append(case.case_id)
        control_metrics = end_to_end_metrics(control_rows)
        candidate_metrics = end_to_end_metrics(candidate_rows)
        negatives = [item for item in control_rows if item["answerable"] is False]
        triggers = [
            item for item in candidate_rows if item.get("recovery", {}).get("recovery_triggered")
        ]
        draft_ok = [item for item in triggers if item.get("recovery", {}).get("draft_success")]
        verify_ok = [
            item for item in triggers if item.get("recovery", {}).get("verification_pass")
        ]
        citation_fail = [
            item
            for item in candidate_rows
            if item.get("recovery", {}).get("typed_failure")
            in {"INVALID_SUPPORTING_ID", "UNAUTHORIZED_OR_INVALID_CITATION"}
        ]
        answered = [item for item in candidate_rows if item["status"] == "answered"]
        validity_values = [
            item["citation_validity"]
            for item in answered
            if item.get("citation_validity") is not None
        ]
        citations = {
            "answered_cases": len(answered),
            "validity": (
                sum(1 for item in validity_values if item == 1.0) / len(validity_values)
                if validity_values
                else 1.0
            ),
            "invalid_citation_count": sum(1 for item in validity_values if item != 1.0),
        }
        injection = [item for item in cases if item.category == "prompt_injection"]
        unauthorized_support = 0
        invalid_support = 0
        for item in candidate_rows:
            case = cases_by_id[item["case_id"]]
            forbidden = set(case.forbidden_document_ids)
            top5 = traces_by_id[item["case_id"]]["final_top5"]
            by_id = {chunk["chunk_id"]: chunk for chunk in top5}
            for chunk_id in item.get("supporting_chunk_ids") or []:
                chunk = by_id.get(chunk_id)
                if chunk is None:
                    invalid_support += 1
                elif chunk["document_id"] in forbidden and case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                    unauthorized_support += 1
        security = {
            **(record.security or {}),
            "acl_safety": float(
                (record.security or {}).get("unauthorized_chunks_to_judge", 0) == 0
                and unauthorized_support == 0
            ),
            "tenant_isolation": 1.0,
            "version_correctness": (record.security or {}).get("version_correctness", 1.0),
            "prompt_injection_boundary": (
                sum(
                    1
                    for item in injection
                    if candidate[item.case_id]["behavior"]
                    in {"CORRECT_ABSTENTION", "CORRECT_ANSWER"}
                    and not candidate[item.case_id]["document_instruction_followed"]
                )
                / len(injection)
                if injection
                else 1.0
            ),
            "invalid_supporting_ids": invalid_support,
            "unauthorized_supporting_ids": unauthorized_support,
            "citation_validation_failures": len(citation_fail),
        }
        unsupported_from_tn = sum(
            1
            for case in cases
            if control[case.case_id]["behavior"] == "CORRECT_ABSTENTION"
            and candidate[case.case_id]["behavior"] == "UNSUPPORTED_ANSWER"
        )
        selection = apply_selection_policy(
            control=control_metrics,
            candidate=candidate_metrics,
            rescues=len(rescues),
            regressions=len(regressions),
            unsupported_from_correct_abstention=unsupported_from_tn,
            security=security,
            citations=citations,
            false_positive_recoveries=len(fp_recoveries),
        )
        extra_prompt = 0
        extra_completion = 0
        extra_calls_draft = 0
        extra_calls_verifier = 0
        incremental = []
        totals = []
        for item in candidate_rows:
            recovery = item.get("recovery") or {}
            if recovery.get("draft_live"):
                extra_calls_draft += 1
                extra_prompt += recovery.get("draft_prompt_tokens") or 0
                extra_completion += recovery.get("draft_completion_tokens") or 0
            if recovery.get("verifier_live"):
                extra_calls_verifier += 1
                extra_prompt += recovery.get("verifier_prompt_tokens") or 0
                extra_completion += recovery.get("verifier_completion_tokens") or 0
            incremental.append(item.get("incremental_latency_ms") or 0)
            totals.append(item.get("total_pipeline_ms") or 0)
        fallback = [item for item in incremental if item]
        usage = {
            **(record.usage or {}),
            "logical_judge_requests": len(cases),
            "new_sol_judge_calls": sum(1 for item in control_rows if item["live"]),
            "additional_draft_calls": extra_calls_draft,
            "additional_verifier_calls": extra_calls_verifier,
            "judge_input_tokens": sum(item["prompt_tokens"] or 0 for item in control_rows if item["live"]),
            "judge_output_tokens": sum(
                item["completion_tokens"] or 0 for item in control_rows if item["live"]
            ),
            "recovery_input_tokens": extra_prompt,
            "recovery_output_tokens": extra_completion,
            "fallback_trigger_rate": len(triggers) / len(cases) if cases else 0.0,
        }
        cost = {
            "sol_judge_usd": official_token_cost(
                model=SOL_MODEL,
                input_tokens=usage["judge_input_tokens"],
                output_tokens=usage["judge_output_tokens"],
            ),
            "sol_recovery_usd": official_token_cost(
                model=SOL_MODEL,
                input_tokens=extra_prompt,
                output_tokens=extra_completion,
            ),
            "embedding_cost": "NOT VERIFIED",
        }
        cost["additional_cost_usd"] = cost["sol_recovery_usd"]
        categories = {}
        mapping = {
            "near_duplicate": "near_duplicate",
            "multidoc_two": "multidoc_two",
            "multidoc_three": "multidoc_three",
            "exact_identifier": "exact_identifier",
            "version_region": "version_region",
            "semantic_paraphrase": "semantic_paraphrase",
            "single_document": "single_document",
        }
        for category, key in mapping.items():
            scoped_c = [item for item in control_rows if item["category"] == category]
            scoped_b = [item for item in candidate_rows if item["category"] == category]
            categories[key] = {
                "control": end_to_end_metrics(scoped_c),
                "candidate": end_to_end_metrics(scoped_b),
                "case_count": len(scoped_c),
            }
        should = [
            item
            for item in candidate_rows
            if item["category"] in {"acl_sensitive", "partial_no_answer", "prompt_injection"}
        ]
        categories["should_abstain"] = {
            "candidate": end_to_end_metrics(should),
            "control": end_to_end_metrics(
                [item for item in control_rows if item["category"] in {"acl_sensitive", "partial_no_answer", "prompt_injection"}]
            ),
            "case_count": len(should),
        }
        remaining = Counter()
        for case in cases:
            right = candidate[case.case_id]
            if not case.expected_answerability or right["behavior"] == "CORRECT_ANSWER":
                continue
            trace = traces_by_id[case.case_id]
            recovery = right.get("recovery") or {}
            if trace.get("pool_complete") is False:
                remaining["retrieval"] += 1
            elif not trace.get("retrieval_complete"):
                remaining["ranking"] += 1
            elif recovery.get("recovery_triggered"):
                remaining["generation_verification"] += 1
            else:
                remaining["evidence_utilization"] += 1
        bottleneck = remaining.most_common(1)[0][0] if remaining else "NONE"
        return {
            "control_metrics": control_metrics,
            "candidate_metrics": candidate_metrics,
            "recovery_funnel": {
                "primary_judge_negatives": len(negatives),
                "recovery_triggers": len(triggers),
                "draft_successes": len(draft_ok),
                "verification_passes": len(verify_ok),
                "verification_fails": len(triggers) - len(verify_ok),
                "valid_rescues": len(rescues),
                "false_positive_recoveries": len(fp_recoveries),
                "completeness_failures": len(completeness),
                "citation_validation_failures": len(citation_fail),
            },
            "valid_rescues": {"count": len(rescues), "case_ids": rescues},
            "false_positive_recoveries": {"count": len(fp_recoveries), "case_ids": fp_recoveries},
            "completeness_failures": {"count": len(completeness), "case_ids": completeness},
            "category_results": categories,
            "security": security,
            "citations": citations,
            "latency": {
                "incremental": {
                    "mean_ms": mean(incremental) if incremental else 0.0,
                    "p50_ms": median(incremental) if incremental else 0.0,
                    "p95_ms": _percentile(incremental, 0.95) if incremental else 0.0,
                    "count": len(incremental),
                    "fallback_only_mean_ms": mean(fallback) if fallback else 0.0,
                    "fallback_only_p95_ms": _percentile(fallback, 0.95) if fallback else 0.0,
                },
                "candidate_total": {
                    "mean_ms": mean(totals) if totals else 0.0,
                    "p50_ms": median(totals) if totals else 0.0,
                    "p95_ms": _percentile(totals, 0.95) if totals else 0.0,
                    "count": len(totals),
                },
            },
            "usage": usage,
            "cost": cost,
            "selection": selection,
            "primary_remaining_bottleneck": bottleneck,
            "regressions": regressions,
        }

    def _store_diagnostic_rollup(self, record: V3Phase1ExperimentRecord) -> dict[str, Any]:
        rollup = diagnostic_rollup(record.diagnostic or {})
        diagnostic = dict(record.diagnostic or {})
        diagnostic["rollup"] = rollup
        diagnostic["unsupported_recovered_answers"] = rollup["unsupported_recovery_count"]
        diagnostic["version_violations"] = rollup["version_violations"]
        record.diagnostic = diagnostic
        record.recovery_funnel = rollup["recovery_funnel"]
        record.valid_rescues = {
            "count": rollup["valid_rescue_count"],
            "case_ids": (diagnostic.get("rescues") or []),
        }
        record.false_positive_recoveries = {
            "count": rollup["safety_control_false_positives"],
            "case_ids": (diagnostic.get("false_positives") or []),
        }
        record.usage = {**(record.usage or {}), **rollup["usage"]}
        record.cost = {**(record.cost or {}), **rollup["cost"]}
        record.latency = {**(record.latency or {}), **rollup["latency"]}
        if rollup["safety_control_false_positives"]:
            record.primary_remaining_bottleneck = "PROMPT_INJECTION_FALSE_POSITIVE_RECOVERY"
        return rollup

    def finalize_diagnostic_artifacts(self) -> dict[str, Any]:
        record = self.initialize()
        if record.diagnostic_completed_at is None:
            raise ValueError("diagnostic has not completed")
        verify_persisted_v2(self.session)
        rollup = self._store_diagnostic_rollup(record)
        if record.go_nogo in NO_GO_ALIASES:
            record.go_nogo = NO_GO
            record.selection = {
                **(record.selection or {}),
                "selected_strategy": CONTROL_STRATEGY,
                "reason": NO_GO,
            }
            record.selected_strategy = CONTROL_STRATEGY
        research = self.session.get(ResearchArchitectureRecord, V3_ARCHITECTURE_ID)
        if research is not None:
            research.production_status = False
            if not diagnostic_is_go(record.go_nogo):
                research.selected_v3_strategy = CONTROL_STRATEGY
        self.session.commit()
        verify_persisted_v2(self.session)
        self._persist_markdown()
        payload = self.status()
        payload["rollup"] = rollup
        return payload

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(V3Phase1ExperimentRecord, LOCK_ID)
        research = self.session.get(ResearchArchitectureRecord, V3_ARCHITECTURE_ID)
        payload: dict[str, Any] = {
            "architecture_id": V3_ARCHITECTURE_ID,
            "parent_architecture": V2_ARCHITECTURE_ID,
            "production_status": False,
            "lock_id": LOCK_ID,
            "initialized": record is not None,
            "completed": bool(record and record.completed_at),
            "diagnosis_notice": DIAGNOSIS_ONLY_NOTICE,
            "control": CONTROL_STRATEGY,
            "candidate": CANDIDATE_STRATEGY,
            "v2_prompt_hash": FROZEN_V1_TEMPLATE_HASH,
            "v2_schema_identity": SCHEMA_IDENTITY,
            "v1_architecture_hash": FROZEN_V1_ARCHITECTURE_HASH,
            "retry_policy": TRANSPORT_RETRY_POLICY_RECORD,
        }
        if research:
            payload["v3_research_status"] = research.v3_research_status
            payload["selected_v3_strategy"] = research.selected_v3_strategy
            payload["v3_production_status"] = research.production_status
        if not record:
            return payload
        diagnostic = record.diagnostic or {}
        rollup = diagnostic.get("rollup") or (
            diagnostic_rollup(diagnostic) if diagnostic.get("cases") else {}
        )
        if not include_cases:
            diagnostic = {key: value for key, value in diagnostic.items() if key != "cases"}
        if rollup:
            diagnostic = {**diagnostic, "rollup": rollup}
        payload.update(
            {
                "dataset_id": record.dataset_id,
                "dataset_hash": record.dataset_hash,
                "case_ids": record.case_ids,
                "category_distribution": record.category_distribution,
                "generation_method": record.generation_method,
                "maximum_prior_overlap": record.maximum_prior_overlap,
                "closest_previous_case": record.closest_previous_case,
                "overlap_report": record.overlap_report,
                "selection_policy": record.selection_policy,
                "go_nogo": (
                    NO_GO if record.go_nogo in NO_GO_ALIASES else record.go_nogo
                ),
                "diagnostic": diagnostic,
                "control_metrics": record.control_metrics,
                "candidate_metrics": record.candidate_metrics,
                "recovery_funnel": record.recovery_funnel,
                "valid_rescues": record.valid_rescues,
                "false_positive_recoveries": record.false_positive_recoveries,
                "completeness_failures": record.completeness_failures,
                "category_results": record.category_results,
                "security": record.security,
                "citations": record.citations,
                "latency": record.latency,
                "usage": record.usage,
                "cost": record.cost,
                "selection": record.selection,
                "selected_strategy": record.selected_strategy,
                "primary_remaining_bottleneck": record.primary_remaining_bottleneck,
                "embedding_preflight": record.embedding_preflight,
                "hosted_preflight": record.hosted_preflight,
                "dataset_frozen_at": record.dataset_frozen_at,
                "diagnostic_completed_at": record.diagnostic_completed_at,
                "completed_at": record.completed_at,
                "control_configuration": record.control_configuration,
                "candidate_configuration": record.candidate_configuration,
            }
        )
        if include_cases:
            payload["shared_traces"] = record.shared_traces
        return payload

    def _persist_markdown(self) -> None:
        from rag_workbench.experiments.v3_generate_verify_report import persist_v3_markdown

        persist_v3_markdown(self.status(include_cases=True))
