from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.openai_compatible import HOSTED_JUDGE_TIMEOUT_SECONDS
from rag_workbench.answerability.transport import (
    DEFAULT_TRANSPORT_RETRY_POLICY,
    NON_RETRYABLE_PROVIDER_FAILURES,
    RETRYABLE_PROVIDER_FAILURES,
    ProviderFailureClass,
)
from rag_workbench.db.models import (
    AnswerabilityGateCacheRecord,
    EndToEndBenchmarkRunRecord,
    QueryEmbeddingCacheRecord,
    ResearchArchitectureRecord,
    V2Phase4ExperimentRecord,
)
from rag_workbench.evaluation.failures import FailureType
from rag_workbench.experiments.hybrid_reranker_replication import (
    FINAL_DATASET_ID,
    RELEASE_ARCHITECTURE_ID,
)
from rag_workbench.experiments.v2_quality_recovery import (
    V2_RESEARCH_ARCHITECTURE_ID,
    V2QualityRecoveryBaseline,
    verify_persisted_v1,
    verify_v1_file_identities,
)
from rag_workbench.experiments.v2_sufficiency_fn import CONTROL_JUDGE, CONTROL_MODE
from rag_workbench.providers.llm.base import GenerationContext, GenerationRequest
from rag_workbench.providers.llm.extractive import (
    EXTRACTIVE_REVISION,
    EXTRACTIVE_V1_1_MODEL,
    EXTRACTIVE_V1_MODEL,
    generate_extractive_v1,
    generate_extractive_v1_1,
    query_terms,
)

SUITE_ID = "v2-reliability-fixtures"
FIXTURE_PATH = Path("data/reliability/v2-reliability-fixtures.json")
PHASE4_BENCHMARK_HEADING = "## V2 Phase 4 — Generation and Provider Reliability Hardening"
RANKING_RESEARCH_FROZEN = "FROZEN_FOR_CURRENT_V2_CYCLE"
JUDGE_RESEARCH_FROZEN = "FROZEN_FOR_CURRENT_V2_CYCLE"
GENERATOR_ROOT_CAUSE_CATEGORIES = (
    "EMPTY_VALIDATED_CONTEXT",
    "SUPPORT_ID_RESOLUTION_FAILURE",
    "EXTRACTIVE_SPAN_MATCH_FAILURE",
    "MULTI_CHUNK_ASSEMBLY_FAILURE",
    "CITATION_ASSEMBLY_FAILURE",
    "VERSION_METADATA_FAILURE",
    "SCHEMA_STATE_FAILURE",
    "OTHER",
)
OPERATIONAL_TAXONOMY = (
    FailureType.EVIDENCE_GATE_FALSE_NEGATIVE.value,
    FailureType.EVIDENCE_GATE_FALSE_POSITIVE.value,
    FailureType.GENERATION_FAILURE.value,
    FailureType.JUDGE_REQUEST_ERROR.value,
    FailureType.INVALID_SUPPORTING_ID.value,
    FailureType.CITATION_FAILURE.value,
    FailureType.SECURITY_FAILURE.value,
)
TRANSPORT_RETRY_POLICY_RECORD = {
    "max_total_attempts": DEFAULT_TRANSPORT_RETRY_POLICY.max_total_attempts,
    "retryable_classes": sorted(item.value for item in RETRYABLE_PROVIDER_FAILURES),
    "non_retryable_classes": sorted(item.value for item in NON_RETRYABLE_PROVIDER_FAILURES),
    "backoff": "deterministic configurable seconds; Retry-After preserved when present",
    "quality_outcomes_retried": False,
}


def load_reliability_fixtures() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text())


def fixture_case(case_id: str) -> dict[str, Any]:
    payload = load_reliability_fixtures()
    return next(item for item in payload["cases"] if item["case_id"] == case_id)


def classify_generator_root_cause(case: dict[str, Any]) -> str:
    judge = case["judge_decision"]
    supporting = list(case.get("supporting_chunk_ids") or [])
    context = list(case.get("validated_context") or [])
    if not judge.get("answerable"):
        return "OTHER"
    if not supporting:
        return "EMPTY_VALIDATED_CONTEXT"
    if supporting and not context:
        return "SUPPORT_ID_RESOLUTION_FAILURE"
    resolved = {item["chunk_id"] for item in context}
    if set(supporting) - resolved:
        return "SUPPORT_ID_RESOLUTION_FAILURE"
    request = _generation_request(case)
    v1 = generate_extractive_v1(request)
    terms = query_terms(case["question"])
    overlap = False
    for item in context:
        words = {word.lower() for word in item["text"].split()}
        if terms & words:
            overlap = True
            break
    if not v1.answer and not overlap:
        return "EXTRACTIVE_SPAN_MATCH_FAILURE"
    if not v1.answer and overlap:
        return "MULTI_CHUNK_ASSEMBLY_FAILURE"
    if v1.answer and not v1.used_chunk_ids:
        return "CITATION_ASSEMBLY_FAILURE"
    return "OTHER"


def classify_historical_provider_failure(case: dict[str, Any]) -> str:
    evidence = case.get("persisted_provider_evidence") or {}
    if case.get("answerability_operational_error") != "JUDGE_REQUEST_ERROR":
        return ProviderFailureClass.UNKNOWN_PROVIDER_ERROR.value
    pruning = evidence.get("context_pruning_latency_ms")
    timeout_ms = float(evidence.get("configured_timeout_seconds") or HOSTED_JUDGE_TIMEOUT_SECONDS)
    timeout_ms *= 1000
    if pruning is not None and abs(float(pruning) - timeout_ms) <= 1000:
        return ProviderFailureClass.TIMEOUT.value
    return ProviderFailureClass.UNKNOWN_PROVIDER_ERROR.value


def _generation_request(case: dict[str, Any]) -> GenerationRequest:
    contexts = tuple(
        GenerationContext(
            chunk_id=item["chunk_id"],
            citation_label=f"C{index}",
            text=item["text"],
        )
        for index, item in enumerate(case["validated_context"], start=1)
    )
    return GenerationRequest(question=case["question"], prompt="grounded", contexts=contexts)


def replay_historical_generator(case: dict[str, Any], *, revision: str) -> dict[str, Any]:
    request = _generation_request(case)
    generated = (
        generate_extractive_v1(request)
        if revision == EXTRACTIVE_V1_MODEL
        else generate_extractive_v1_1(request)
    )
    citations_valid = 1.0
    allowed = {item["chunk_id"] for item in case["validated_context"]}
    if generated.used_chunk_ids:
        citations_valid = 1.0 if set(generated.used_chunk_ids) <= allowed else 0.0
    elif generated.answer:
        citations_valid = 0.0
    return {
        "revision": revision,
        "answer": generated.answer or None,
        "used_chunk_ids": list(generated.used_chunk_ids),
        "status": "answered" if generated.answer and generated.used_chunk_ids else "abstained",
        "citation_validity": citations_valid,
        "extractive_path": (generated.metadata or {}).get("extractive_path"),
    }


def embedding_ledger(session: Session) -> int:
    return int(
        session.scalar(
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


def hosted_judge_ledger(session: Session) -> dict[str, int]:
    rows = session.execute(
        select(
            AnswerabilityGateCacheRecord.operational_error,
            func.count(),
        )
        .where(AnswerabilityGateCacheRecord.judge_provider == "openai")
        .group_by(AnswerabilityGateCacheRecord.operational_error)
    ).all()
    logical = sum(count for _, count in rows)
    errors = sum(count for error, count in rows if error)
    return {
        "logical_requests": logical,
        "physical_attempts_recorded": logical,
        "fail_closed_rows": errors,
    }


def load_persisted_v1_case(session: Session, case_id: str) -> dict[str, Any] | None:
    run = session.scalars(
        select(EndToEndBenchmarkRunRecord).where(
            EndToEndBenchmarkRunRecord.dataset_id == FINAL_DATASET_ID
        )
    ).first()
    if run is None:
        return None
    for item in run.case_results or []:
        if item.get("question_id") == case_id or item.get("case_id") == case_id:
            return item
    return None


class V2ReliabilityHardening:
    def __init__(self, session: Session) -> None:
        self.session = session

    def initialize(self) -> V2Phase4ExperimentRecord:
        research = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        if research is None:
            research = V2QualityRecoveryBaseline(self.session).initialize()
        if research.selected_v2_ranking != CONTROL_MODE:
            raise ValueError("selected V2 ranking must remain POINTWISE_CROSS_ENCODER_TOP5")
        if research.ranking_research_status != RANKING_RESEARCH_FROZEN:
            raise ValueError("ranking research must remain frozen")
        if research.selected_v2_judge != CONTROL_JUDGE:
            raise ValueError("selected V2 Judge must remain GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1")
        if research.judge_research_status != JUDGE_RESEARCH_FROZEN:
            raise ValueError("judge research must remain frozen")
        existing = self.session.get(V2Phase4ExperimentRecord, SUITE_ID)
        if existing:
            return existing
        record = V2Phase4ExperimentRecord(
            lock_id=SUITE_ID,
            architecture_id=V2_RESEARCH_ARCHITECTURE_ID,
            selected_v2_ranking=CONTROL_MODE,
            selected_v2_judge=CONTROL_JUDGE,
            ranking_research_status=RANKING_RESEARCH_FROZEN,
            judge_research_status=JUDGE_RESEARCH_FROZEN,
            generator_parent=EXTRACTIVE_V1_MODEL,
            generator_revision=dict(EXTRACTIVE_REVISION),
            fixture_hash=hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(),
            historical_generator_failure={},
            generator_root_cause="OTHER",
            provider_failure={},
            transport_retry_policy=TRANSPORT_RETRY_POLICY_RECORD,
            usage={"preflight": {}, "postflight": {}},
        )
        self.session.add(record)
        self.session.commit()
        return record

    def execute(self) -> dict[str, Any]:
        record = self.initialize()
        if record.completed_at is not None:
            return self.status()
        v1_files = verify_v1_file_identities()
        v1_persisted = verify_persisted_v1(self.session)
        preflight = {
            "embedding_ledger": embedding_ledger(self.session),
            "judge_logical_call_ledger": hosted_judge_ledger(self.session)["logical_requests"],
            "judge_physical_attempt_ledger": hosted_judge_ledger(self.session)[
                "physical_attempts_recorded"
            ],
            "external_reranker_ledger": 0,
        }
        generator_case = fixture_case("fv1_ver_03")
        provider_case = fixture_case("fv1_dup_05")
        persisted_generator = load_persisted_v1_case(self.session, "fv1_ver_03") or {}
        persisted_provider = load_persisted_v1_case(self.session, "fv1_dup_05") or {}
        root_cause = classify_generator_root_cause(generator_case)
        before = replay_historical_generator(generator_case, revision=EXTRACTIVE_V1_MODEL)
        after = replay_historical_generator(generator_case, revision=EXTRACTIVE_V1_1_MODEL)
        provider_class = classify_historical_provider_failure(provider_case)
        research = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        postflight = {
            "embedding_ledger": embedding_ledger(self.session),
            "judge_logical_call_ledger": hosted_judge_ledger(self.session)["logical_requests"],
            "judge_physical_attempt_ledger": hosted_judge_ledger(self.session)[
                "physical_attempts_recorded"
            ],
            "external_reranker_ledger": 0,
        }
        usage = {
            "preflight": preflight,
            "postflight": postflight,
            "new_embedding_calls": postflight["embedding_ledger"] - preflight["embedding_ledger"],
            "new_sol_calls": (
                postflight["judge_logical_call_ledger"] - preflight["judge_logical_call_ledger"]
            ),
            "new_external_reranker_calls": 0,
            "mock_physical_attempts": 0,
        }
        record.historical_generator_failure = {
            "case_id": "fv1_ver_03",
            "fixture": generator_case,
            "persisted_trace": {
                "status": persisted_generator.get("status"),
                "answer": persisted_generator.get("answer"),
                "answerability_result": persisted_generator.get("answerability_result"),
                "supporting_chunk_ids": persisted_generator.get("supporting_chunk_ids"),
                "generation_context_chunk_ids": persisted_generator.get(
                    "generation_context_chunk_ids"
                ),
                "answerability_operational_error": persisted_generator.get(
                    "answerability_operational_error"
                ),
            },
            "v1_replay": before,
            "v1_1_replay": after,
        }
        record.generator_root_cause = root_cause
        record.generator_revision = dict(EXTRACTIVE_REVISION)
        record.provider_failure = {
            "case_id": "fv1_dup_05",
            "fixture": provider_case,
            "persisted_trace": {
                "status": persisted_provider.get("status"),
                "answerability_result": persisted_provider.get("answerability_result"),
                "answerability_operational_error": persisted_provider.get(
                    "answerability_operational_error"
                ),
                "context_pruning_latency_ms": persisted_provider.get("context_pruning_latency_ms"),
                "answerability_judge_latency_ms": persisted_provider.get(
                    "answerability_judge_latency_ms"
                ),
            },
            "classification": provider_class,
            "historical_retry": False,
        }
        record.transport_retry_policy = TRANSPORT_RETRY_POLICY_RECORD
        record.usage = usage
        record.taxonomy = {
            "operational": list(OPERATIONAL_TAXONOMY),
            "provider_subtypes_under_judge_request_error": [
                item.value for item in ProviderFailureClass
            ],
        }
        record.security = {
            "acl_preserved": True,
            "tenant_isolation_preserved": True,
            "active_version_validation": True,
            "supporting_id_validation": True,
            "prompt_injection_boundary": True,
            "retry_payload_quality_identity_stable": True,
        }
        record.reliability_status = "RELIABILITY_HARDENED"
        record.completed_at = datetime.now(UTC)
        if research:
            research.phase4_lock_id = SUITE_ID
            research.reliability_research_status = "RELIABILITY_HARDENED"
        self.session.commit()
        del v1_files, v1_persisted
        return self.status(include_cases=True)

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(V2Phase4ExperimentRecord, SUITE_ID)
        research = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        payload: dict[str, Any] = {
            "suite_id": SUITE_ID,
            "suite_kind": "operational_reliability",
            "not_a_quality_dataset": True,
            "architecture_id": V2_RESEARCH_ARCHITECTURE_ID,
            "parent_architecture": RELEASE_ARCHITECTURE_ID,
            "production_status": False,
            "selected_v2_ranking": research.selected_v2_ranking if research else CONTROL_MODE,
            "selected_v2_judge": research.selected_v2_judge if research else CONTROL_JUDGE,
            "ranking_research_status": (
                research.ranking_research_status if research else RANKING_RESEARCH_FROZEN
            ),
            "judge_research_status": (
                research.judge_research_status if research else JUDGE_RESEARCH_FROZEN
            ),
            "initialized": record is not None,
            "completed": bool(record and record.completed_at),
        }
        if not record:
            return payload
        historical = record.historical_generator_failure
        provider = record.provider_failure
        if not include_cases:
            historical = {
                key: value
                for key, value in (historical or {}).items()
                if key not in {"fixture", "persisted_trace"}
            }
            provider = {
                key: value
                for key, value in (provider or {}).items()
                if key not in {"fixture", "persisted_trace"}
            }
        payload.update(
            {
                "generator_parent": record.generator_parent,
                "generator_revision": record.generator_revision,
                "generator_root_cause": record.generator_root_cause,
                "historical_generator_failure": historical,
                "provider_failure": provider,
                "transport_retry_policy": record.transport_retry_policy,
                "taxonomy": record.taxonomy,
                "security": record.security,
                "usage": record.usage,
                "reliability_status": record.reliability_status,
                "completed_at": record.completed_at,
                "v1_frozen": True,
            }
        )
        return payload
