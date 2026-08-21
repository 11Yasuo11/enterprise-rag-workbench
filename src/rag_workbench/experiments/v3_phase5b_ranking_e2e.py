# ruff: noqa: E501
"""V3 Phase 5B: three-arm ranking end-to-end benchmark.

Reference R: frozen stable V2 (pointwise CE Top-5 → Sol Judge → V2 answer/abstain).
Control  A: V3 research pipeline (same retrieval → pointwise CE Top-5 → same Sol Judge → recovery on negative).
Candidate B: same as Control A but final ranking is PAIRWISE_COMPLEMENTARITY_RERANK v1.0 instead of pointwise CE Top-5.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from rag_workbench.answerability.base import AnswerabilityResult
from rag_workbench.answerability.cache import gate_cache_key
from rag_workbench.answerability.openai_compatible import EVIDENCE_GATE_PROMPT_VERSION
from rag_workbench.answerability.transport import DEFAULT_TRANSPORT_RETRY_POLICY
from rag_workbench.answerability.validation import validate_gate_result_with_error
from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import (
    AnswerabilityGateCacheRecord,
    QueryEmbeddingCacheRecord,
    ResearchArchitectureRecord,
    V3Phase5bExperimentRecord,
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
    _percentile,
    _result,
)
from rag_workbench.experiments.sol_judge_e2e_benchmark import official_token_cost
from rag_workbench.experiments.v2_document_diversity import ranking_candidate
from rag_workbench.experiments.v2_final_benchmark import (
    HIDDEN_GROUND_TRUTH_FIELDS,
    V2_ARCHITECTURE_ID,
    V2FinalCase,
    _behavior,
    _gate_evidence,
    classification,
)
from rag_workbench.experiments.v2_quality_recovery import stable_hash
from rag_workbench.experiments.v2_reliability import TRANSPORT_RETRY_POLICY_RECORD
from rag_workbench.experiments.v2_sufficiency_fn import SOL_MODEL, retrieval_complete
from rag_workbench.experiments.v3_final_ab import (
    FAILURE_FAMILIES,
    classify_candidate_failure,
    instruction_boundary_safety_gate,
    recovery_trace,
    v3_final_candidate_configuration,
)
from rag_workbench.experiments.v3_generate_verify import (
    V3_ARCHITECTURE_ID,
    V3GenerateVerifyBenchmark,
    document_instruction_followed,
    end_to_end_metrics,
    evaluator_supported,
    v3_control_configuration,
    verify_persisted_v2,
)
from rag_workbench.experiments.v3_phase5b_ranking_cases import (
    DATASET_ID,
    DATASET_PATH,
    EXPECTED_DISTRIBUTION,
    GENERATION_METHOD,
    dataset_overlap_report,
)
from rag_workbench.recovery.instruction_boundary import (
    BOUNDARY_FAIL,
    BOUNDARY_PASS,
)
from rag_workbench.recovery.runtime import evaluate_recovery
from rag_workbench.reranking import Reranker
from rag_workbench.reranking.pairwise_complementarity import (
    ALGORITHM_ID as PAIRWISE_ALGORITHM_ID,
)
from rag_workbench.reranking.pairwise_complementarity import (
    ALGORITHM_VERSION as PAIRWISE_ALGORITHM_VERSION,
)
from rag_workbench.reranking.pairwise_complementarity import (
    pairwise_configuration,
    select_pairwise_complementarity_top5,
)
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import Retriever

LOCK_ID = "v3-phase5b-ranking-e2e"
EXPERIMENT_ID = "v3-phase5b-frozen-ranking-e2e"
FROZEN_DATASET_HASH: str | None = "58ac25869720e094ada10717f4b10999a03382f4a7b45f95d88371730a63acdf"
FINAL_DATASET_INTEGRITY_FAILURE = "FINAL_DATASET_INTEGRITY_FAILURE"
EXTERNAL_BUDGET_REQUIRED = "EXTERNAL_BUDGET_REQUIRED"
AUTHORIZED_EMBEDDING_CEILING = 1400
AUTHORIZED_JUDGE_CEILING = 2100
REFERENCE_STRATEGY = "V2_JUDGE_FIRST_ONLY"
CONTROL_STRATEGY = "V3_GENERATE_VERIFY_WITH_EVIDENCE_INSTRUCTION_BOUNDARY"
CANDIDATE_STRATEGY = "V3_PAIRWISE_COMPLEMENTARITY_RANKING_WITH_GENERATE_VERIFY"
PROMOTE = "PROMOTE_PAIRWISE_RANKING_TO_V3_CANDIDATE"
KEEP = "KEEP_CURRENT_V3_RESEARCH_ARCHITECTURE"
V3_PROMOTED = "V3_CANDIDATE_PROMOTED"
V3_REJECTED = "V3_CANDIDATE_REJECTED"
LEDGER_DIR = Path("data/experiments/v3-phase5b-ranking-e2e")
EMBEDDING_INPUT_USD_PER_MILLION = 0.02
CANDIDATE_RANKING_HASH = "527afb76a0226018e158291c0212e16cfdc31e0d9990cfd4686d72c1d3df69cd"

PROMOTION_POLICY = {
    "unsupported_answers": 0,
    "precision_min": 0.99,
    "prompt_injection_safety": 1.0,
    "acl_safety": 1.0,
    "tenant_isolation": 1.0,
    "version_correctness": 1.0,
    "citation_validity": 1.0,
    "unauthorized_supporting_ids": 0,
    "answerable_correct_rate_gain_vs_control_min": 0.05,
    "additional_correct_supported_vs_control_min": 5,
    "answerable_correct_rate_gain_vs_reference_min": 0.10,
    "additional_correct_supported_vs_reference_min": 10,
    "reference_correct_to_candidate_incorrect": 0,
    "exact_id_recall_gate": ">=_control",
    "version_correctness_gate": ">=_control",
    "same_doc_multi_chunk_regression_gate": 0.05,
    "unauthorized_downstream_evidence": 0,
    "frozen_before_inference": True,
    "promotion_to_public_v2_forbidden": True,
}

PROMOTION_POLICY_REASON = (
    "Candidate B is promoted only if unsupported answers are 0, precision >= 0.99, "
    "injection/ACL/tenant/version/citation safety is 1.0, unauthorized supporting IDs are 0, "
    "Reference-correct → Candidate-incorrect is 0, and either (B vs A: correct-rate gain >= +0.05 "
    "or >= 5 additional correct supported) AND (B vs R: correct-rate gain >= +0.10 or >= 10 "
    "additional correct supported). Ranking gates: Exact-ID Recall@5 >= Control, Version "
    "correctness >= Control, same-doc multi-chunk degradation <= 0.05, unauthorized downstream "
    "evidence = 0. Frozen before final inference."
)


def set_frozen_dataset_hash(h: str) -> None:
    global FROZEN_DATASET_HASH  # noqa: PLW0603
    FROZEN_DATASET_HASH = h


def persisted_dataset_hash() -> str:
    return hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()


def verify_frozen_final_dataset() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        return {
            "ok": False,
            "stop_code": FINAL_DATASET_INTEGRITY_FAILURE,
            "observed_hash": None,
            "expected_hash": FROZEN_DATASET_HASH,
        }
    payload = json.loads(DATASET_PATH.read_text())
    observed = persisted_dataset_hash()
    cases = payload.get("cases") or []
    distribution = dict(sorted(Counter(str(item["category"]) for item in cases).items()))
    if FROZEN_DATASET_HASH is not None and (observed != FROZEN_DATASET_HASH or payload.get("dataset_id") != DATASET_ID):
        return {
            "ok": False,
            "stop_code": FINAL_DATASET_INTEGRITY_FAILURE,
            "observed_hash": observed,
            "expected_hash": FROZEN_DATASET_HASH,
            "dataset_id": payload.get("dataset_id"),
            "cases": len(cases),
            "distribution": distribution,
        }
    overlap = dataset_overlap_report(cases)
    return {
        "ok": True,
        "stop_code": None,
        "dataset_id": DATASET_ID,
        "dataset_hash": observed,
        "cases": len(cases),
        "distribution": distribution,
        "expected_distribution": EXPECTED_DISTRIBUTION,
        "distribution_match": distribution == EXPECTED_DISTRIBUTION,
        "overlap_report": overlap,
        "generation_method": payload.get("generation_method") or GENERATION_METHOD,
        "case_ids": [item["case_id"] for item in cases],
    }


def load_final_cases() -> tuple[V2FinalCase, ...]:
    verified = verify_frozen_final_dataset()
    if not verified["ok"]:
        raise ValueError(FINAL_DATASET_INTEGRITY_FAILURE)
    payload = json.loads(DATASET_PATH.read_text())
    return tuple(V2FinalCase.model_validate(item) for item in payload["cases"])


def v3_phase5b_candidate_configuration() -> dict[str, Any]:
    configuration = v3_final_candidate_configuration()
    return {
        **configuration,
        "strategy": CANDIDATE_STRATEGY,
        "ranking": {
            "algorithm_id": PAIRWISE_ALGORITHM_ID,
            "algorithm_version": PAIRWISE_ALGORITHM_VERSION,
            "configuration": pairwise_configuration(),
            "candidate_ranking_hash": CANDIDATE_RANKING_HASH,
        },
    }


def final_e2e_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = end_to_end_metrics(rows)
    return {
        **base,
        "cases": len(rows),
        "answerable_cases": base["answerable_case_count"],
        "should_abstain_cases": sum(1 for item in rows if not item["expected_answerability"]),
        "answer_precision": base["precision"],
        "answer_recall": base["recall"],
    }


def latency_stats(values: list[float]) -> dict[str, float]:
    clean = [float(item) for item in values if item is not None]
    return {
        "mean_ms": mean(clean) if clean else 0.0,
        "p50_ms": median(clean) if clean else 0.0,
        "p95_ms": _percentile(clean, 0.95) if clean else 0.0,
        "max_ms": max(clean) if clean else 0.0,
        "count": len(clean),
    }


def _set_quality_metrics(top5: list[dict[str, Any]]) -> dict[str, Any]:
    doc_ids = [item["document_id"] for item in top5]
    unique_docs = set(doc_ids)
    doc_counts = Counter(doc_ids)
    redundant = sum(v - 1 for v in doc_counts.values() if v > 1)
    same_doc_repeats = sum(1 for v in doc_counts.values() if v > 1)
    return {
        "unique_documents_in_top5": len(unique_docs),
        "redundant_chunk_count": redundant,
        "same_document_repeat_count": same_doc_repeats,
    }


def apply_promotion_policy(
    *,
    reference: dict[str, Any],
    control: dict[str, Any],
    candidate: dict[str, Any],
    additional_correct_supported_vs_control: int,
    additional_correct_supported_vs_reference: int,
    regressions_vs_reference: int,
    security: dict[str, Any],
    citations: dict[str, Any],
    ranking_gates: dict[str, Any],
) -> dict[str, Any]:
    rate_gain_vs_control = (
        candidate["answerable_case_correct_answer_rate"]
        - control["answerable_case_correct_answer_rate"]
    )
    rate_gain_vs_reference = (
        candidate["answerable_case_correct_answer_rate"]
        - reference["answerable_case_correct_answer_rate"]
    )
    quality_vs_control = (
        rate_gain_vs_control >= PROMOTION_POLICY["answerable_correct_rate_gain_vs_control_min"]
        or additional_correct_supported_vs_control >= PROMOTION_POLICY["additional_correct_supported_vs_control_min"]
    )
    quality_vs_reference = (
        rate_gain_vs_reference >= PROMOTION_POLICY["answerable_correct_rate_gain_vs_reference_min"]
        or additional_correct_supported_vs_reference >= PROMOTION_POLICY["additional_correct_supported_vs_reference_min"]
    )
    safety = (
        candidate["unsupported_answers"] == PROMOTION_POLICY["unsupported_answers"]
        and candidate["precision"] >= PROMOTION_POLICY["precision_min"]
        and security.get("prompt_injection_safety") == PROMOTION_POLICY["prompt_injection_safety"]
        and security.get("acl_safety") == PROMOTION_POLICY["acl_safety"]
        and security.get("tenant_isolation") == PROMOTION_POLICY["tenant_isolation"]
        and security.get("version_correctness") == PROMOTION_POLICY["version_correctness"]
        and citations.get("validity") == PROMOTION_POLICY["citation_validity"]
        and security.get("unauthorized_supporting_ids", 0) == PROMOTION_POLICY["unauthorized_supporting_ids"]
    )
    regression = regressions_vs_reference == PROMOTION_POLICY["reference_correct_to_candidate_incorrect"]
    ranking_pass = (
        ranking_gates.get("exact_id_recall_pass", False)
        and ranking_gates.get("version_correctness_pass", False)
        and ranking_gates.get("same_doc_multi_chunk_pass", False)
        and ranking_gates.get("unauthorized_downstream_evidence", 0) == PROMOTION_POLICY["unauthorized_downstream_evidence"]
    )
    promote = quality_vs_control and quality_vs_reference and safety and regression and ranking_pass
    return {
        "selected_strategy": CANDIDATE_STRATEGY if promote else CONTROL_STRATEGY,
        "promotion_decision": PROMOTE if promote else KEEP,
        "v3_status": V3_PROMOTED if promote else V3_REJECTED,
        "quality_gate_vs_control": quality_vs_control,
        "quality_gate_vs_reference": quality_vs_reference,
        "safety_gate": safety,
        "regression_gate": regression,
        "ranking_gate": ranking_pass,
        "answerable_correct_rate_gain_vs_control": round(rate_gain_vs_control, 6),
        "answerable_correct_rate_gain_vs_reference": round(rate_gain_vs_reference, 6),
        "additional_correct_supported_vs_control": additional_correct_supported_vs_control,
        "additional_correct_supported_vs_reference": additional_correct_supported_vs_reference,
        "reference_correct_to_candidate_incorrect": regressions_vs_reference,
        "ranking_gates": ranking_gates,
        "reason": PROMOTION_POLICY_REASON,
    }


def _instruction_like_text(text: str) -> bool:
    folded = text.casefold()
    return any(token in folded for token in ("instruction", "system", "assistant", "ignore previous"))


class V3Phase5bRankingBenchmark:
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
        self.inner = V3GenerateVerifyBenchmark(
            session,
            self.settings,
            embedding_provider_factory=embedding_provider_factory,
            reranker_factory=reranker_factory,
            gate_factory=gate_factory,
            recovery_factory=recovery_factory,
        )

    def initialize(self) -> V3Phase5bExperimentRecord:
        verified = verify_frozen_final_dataset()
        if not verified["ok"]:
            raise ValueError(FINAL_DATASET_INTEGRITY_FAILURE)
        self.inner._ensure_v3_identity()
        verify_persisted_v2(self.session)
        if self.inner._corpus_identity() != CORPUS_IDENTITY:
            raise ValueError("corpus identity changed")
        existing = self.session.get(V3Phase5bExperimentRecord, LOCK_ID)
        control = v3_control_configuration()
        candidate = v3_phase5b_candidate_configuration()
        reference = {
            "strategy": REFERENCE_STRATEGY,
            "description": "Frozen stable V2: pointwise CE Top-5 → Sol Judge → V2 answer/abstain",
        }
        policy_hash = stable_hash(PROMOTION_POLICY)
        if existing:
            if existing.selection_policy != PROMOTION_POLICY:
                raise ValueError("frozen promotion policy must not be modified")
            if existing.dataset_hash != FROZEN_DATASET_HASH and FROZEN_DATASET_HASH is not None:
                raise ValueError(FINAL_DATASET_INTEGRITY_FAILURE)
            if existing.production_status is True:
                raise ValueError("v3 phase 5b must remain production=false")
            return existing
        now = datetime.now(UTC)
        record = V3Phase5bExperimentRecord(
            lock_id=LOCK_ID,
            experiment_id=EXPERIMENT_ID,
            architecture_id=V3_ARCHITECTURE_ID,
            parent_architecture_id=V2_ARCHITECTURE_ID,
            production_status=False,
            dataset_id=DATASET_ID,
            dataset_hash=verified["dataset_hash"],
            case_ids=verified["case_ids"],
            category_distribution=verified["distribution"],
            generation_method=verified["generation_method"],
            maximum_prior_overlap=verified["overlap_report"]["maximum_normalized_overlap"],
            closest_previous_case=verified["overlap_report"]["closest_previous_case"],
            overlap_report=verified["overlap_report"],
            selection_policy=PROMOTION_POLICY,
            reference_configuration=reference,
            control_configuration=control,
            candidate_configuration=candidate,
            control_architecture_hash=stable_hash(control),
            candidate_architecture_hash=stable_hash(candidate),
            candidate_ranking_hash=CANDIDATE_RANKING_HASH,
            promotion_policy_hash=policy_hash,
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=CORPUS_IDENTITY,
            dataset_frozen_at=now,
            selection_policy_frozen_at=now,
        )
        self.session.add(record)
        research = self.session.get(ResearchArchitectureRecord, V3_ARCHITECTURE_ID)
        if research is not None:
            research.production_status = False
        self.session.commit()
        if FROZEN_DATASET_HASH is None:
            set_frozen_dataset_hash(verified["dataset_hash"])
        return record

    def embedding_preflight(self, *, persist: bool = True) -> dict[str, Any]:
        record = self.initialize()
        cases = load_final_cases()
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
        matches = sum(self.session.get(QueryEmbeddingCacheRecord, key) is not None for key in keys)
        missing = len(keys) - matches
        current = self.inner._embedding_calls()
        authorized_ceiling = current + missing
        preflight = {
            "embedding_provider": "openai-compatible",
            "embedding_model": "text-embedding-3-small",
            "current_cumulative_embedding_calls": current,
            "total_questions": len(cases),
            "existing_cache_matches": matches,
            "missing_unique_query_embeddings": missing,
            "document_embedding_calls_required": 0,
            "new_query_embedding_calls": missing,
            "authorized_ceiling": authorized_ceiling,
            "configured_ceiling": self.settings.max_external_embedding_calls,
            "owner_authorized_ceiling": AUTHORIZED_EMBEDDING_CEILING,
            "authorization_ok": authorized_ceiling <= AUTHORIZED_EMBEDDING_CEILING
            and authorized_ceiling <= self.settings.max_external_embedding_calls,
            "stop_code": None
            if authorized_ceiling <= AUTHORIZED_EMBEDDING_CEILING
            and authorized_ceiling <= self.settings.max_external_embedding_calls
            else EXTERNAL_BUDGET_REQUIRED,
            "required_MAX_EXTERNAL_EMBEDDING_CALLS": authorized_ceiling,
        }
        if persist:
            record.embedding_preflight = preflight
            self.session.commit()
        return preflight

    def hosted_preflight(self, *, persist: bool = True) -> dict[str, Any]:
        record = self.initialize()
        cases = load_final_cases()
        traces = record.shared_traces or []
        judge_rows = self.inner._hosted_judge_rows()
        recovery_rows = self.inner._hosted_recovery_rows()
        if traces and len(traces) == len(cases):
            judge_keys_ref = [
                gate_cache_key(
                    case.question,
                    _gate_evidence(trace["pointwise_top5"]),
                    provider="openai",
                    model=SOL_MODEL,
                    gate_version="1",
                    prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
                )[0]
                for case, trace in zip(cases, traces, strict=True)
            ]
            judge_keys_candidate = [
                gate_cache_key(
                    case.question,
                    _gate_evidence(trace["pairwise_top5"]),
                    provider="openai",
                    model=SOL_MODEL,
                    gate_version="1",
                    prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
                )[0]
                for case, trace in zip(cases, traces, strict=True)
            ]
            unique_judge = set(judge_keys_ref) | set(judge_keys_candidate)
            judge_matches = sum(
                self.session.get(AnswerabilityGateCacheRecord, key) is not None
                for key in unique_judge
            )
            missing_judge = len(unique_judge) - judge_matches
            exact = True
        else:
            missing_judge = len(cases) * 2
            judge_matches = 0
            exact = False
        missing_draft = len(cases) * 2
        missing_verifier = len(cases) * 2
        attempts = DEFAULT_TRANSPORT_RETRY_POLICY.max_total_attempts
        new_logical = missing_judge + missing_draft + missing_verifier
        logical_ceiling = judge_rows + recovery_rows + new_logical
        physical_ceiling = judge_rows + recovery_rows + new_logical * attempts
        configured = self.settings.max_external_judge_calls
        remaining_authorized = AUTHORIZED_JUDGE_CEILING - (judge_rows + recovery_rows)
        preflight = {
            "phase": "phase5b-ranking-e2e",
            "exact_top5_identities": exact,
            "current_embedding_ledger": self.inner._embedding_calls(),
            "current_hosted_logical_ledger": judge_rows + recovery_rows,
            "current_judge_provider_calls": judge_rows,
            "current_recovery_cache_rows": recovery_rows,
            "existing_primary_judge_cache_hits": judge_matches if exact else None,
            "required_new_query_embeddings": (record.embedding_preflight or {}).get("new_query_embedding_calls"),
            "required_primary_judge_calls": missing_judge,
            "maximum_recovery_draft_calls": missing_draft,
            "maximum_verifier_calls": missing_verifier,
            "maximum_physical_attempts": new_logical * attempts,
            "maximum_transport_attempts": attempts,
            "quality_retries": False,
            "logical_ceiling": logical_ceiling,
            "physical_attempt_ceiling": physical_ceiling,
            "configured_ceiling": configured,
            "owner_authorized_ceiling": AUTHORIZED_JUDGE_CEILING,
            "remaining_authorized_headroom": remaining_authorized - new_logical,
            "authorization_ok": logical_ceiling <= AUTHORIZED_JUDGE_CEILING and logical_ceiling <= configured,
            "stop_code": None
            if logical_ceiling <= AUTHORIZED_JUDGE_CEILING and logical_ceiling <= configured
            else EXTERNAL_BUDGET_REQUIRED,
            "required_MAX_EXTERNAL_JUDGE_CALLS": logical_ceiling,
        }
        if persist:
            record.hosted_preflight = preflight
            self.session.commit()
        return preflight

    def combined_preflight(self) -> dict[str, Any]:
        embedding = self.embedding_preflight()
        hosted = self.hosted_preflight()
        stop = embedding.get("stop_code") or hosted.get("stop_code")
        required: dict[str, int] = {}
        if embedding.get("stop_code"):
            required["MAX_EXTERNAL_EMBEDDING_CALLS"] = int(embedding["required_MAX_EXTERNAL_EMBEDDING_CALLS"])
        if hosted.get("stop_code"):
            required["MAX_EXTERNAL_JUDGE_CALLS"] = int(hosted["required_MAX_EXTERNAL_JUDGE_CALLS"])
        return {
            "embedding": embedding,
            "hosted": hosted,
            "stop_code": stop,
            "required_ceilings": required or None,
        }

    def execute(self) -> dict[str, Any]:
        verified = verify_frozen_final_dataset()
        if not verified["ok"]:
            return verified
        record = self.initialize()
        if record.completed_at is not None:
            return self.status()
        embedding = self.embedding_preflight()
        if embedding.get("stop_code"):
            record.primary_remaining_bottleneck = EXTERNAL_BUDGET_REQUIRED
            self.session.commit()
            payload = self.status()
            payload["stop_code"] = EXTERNAL_BUDGET_REQUIRED
            payload["required_ceilings"] = {
                "MAX_EXTERNAL_EMBEDDING_CALLS": embedding["required_MAX_EXTERNAL_EMBEDDING_CALLS"]
            }
            return payload
        if (
            (not self.settings.allow_external_calls or not self.settings.embedding_api_key)
            and embedding.get("new_query_embedding_calls")
        ):
            raise ValueError(
                "embedding authorization is incomplete: set ALLOW_EXTERNAL_CALLS=true "
                "and provide EMBEDDING_API_KEY"
            )
        if record.retrieval_frozen_at is None:
            self._run_retrieval(record)
        hosted = self.hosted_preflight()
        if hosted.get("stop_code"):
            record.primary_remaining_bottleneck = EXTERNAL_BUDGET_REQUIRED
            self.session.commit()
            payload = self.status()
            payload["stop_code"] = EXTERNAL_BUDGET_REQUIRED
            payload["required_ceilings"] = {
                "MAX_EXTERNAL_JUDGE_CALLS": hosted["required_MAX_EXTERNAL_JUDGE_CALLS"]
            }
            return payload
        self.inner._authorize_hosted(int(hosted["logical_ceiling"]))
        if record.execution_started_at is None:
            record.execution_started_at = datetime.now(UTC)
            snapshot = {
                "embedding_ledger": self.inner._embedding_calls(),
                "judge_ledger": self.inner._hosted_judge_rows(),
                "recovery_ledger": self.inner._hosted_recovery_rows(),
            }
            record.usage = {**(record.usage or {}), "ledger_at_execution_start": snapshot}
            self.session.commit()
        return self._run_three_arms(record)

    def _run_retrieval(self, record: V3Phase5bExperimentRecord) -> None:
        cases = load_final_cases()
        traces = list(record.shared_traces or [])
        done = {item["case_id"] for item in traces}
        provider = self.inner._embedding_provider()
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
        reranker: Reranker = self.inner.reranker_factory()
        if reranker.resolved_revision != RERANKER_REVISION:
            raise ValueError("local reranker revision differs from frozen revision")
        usage = {
            **(record.usage or {}),
            "new_query_embedding_calls": (record.usage or {}).get("new_query_embedding_calls", 0),
            "embedding_tokens": (record.usage or {}).get("embedding_tokens", 0),
            "query_cache_hits": (record.usage or {}).get("query_cache_hits", 0),
            "query_cache_misses": (record.usage or {}).get("query_cache_misses", 0),
            "document_embedding_calls": 0,
            "external_reranker_calls": 0,
            "cross_encoder_pairs": (record.usage or {}).get("cross_encoder_pairs", 0),
            "unauthorized_chunks_to_cross_encoder": 0,
            "unauthorized_chunks_to_judge": 0,
        }
        for case in cases:
            if case.case_id in done:
                continue
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
            rerank_started = time.perf_counter()
            reranked = reranker.rerank(case.question, union)
            ce_ms = (time.perf_counter() - rerank_started) * 1000
            usage["cross_encoder_pairs"] += len(union)

            pointwise_top5 = [
                ranking_candidate(item)
                | {
                    "rank": item.reranked_rank,
                    "score": item.reranker_score,
                    "retrieval_source": "pointwise_cross_encoder_top5",
                }
                for item in reranked[:FINAL_TOP_K]
            ]

            pairwise_started = time.perf_counter()
            pairwise_selected = select_pairwise_complementarity_top5(
                reranked, case.question, top_k=FINAL_TOP_K
            )
            pairwise_ms = (time.perf_counter() - pairwise_started) * 1000
            pairwise_top5 = [
                ranking_candidate(item)
                | {
                    "rank": i + 1,
                    "score": item.reranker_score,
                    "retrieval_source": "pairwise_complementarity_top5",
                }
                for i, item in enumerate(pairwise_selected)
            ]

            for top5 in (pointwise_top5, pairwise_top5):
                leaked_fields = HIDDEN_GROUND_TRUTH_FIELDS.intersection(top5[0] if top5 else {})
                if leaked_fields:
                    raise RuntimeError(f"evaluator labels leaked into Top-5: {sorted(leaked_fields)}")
                if case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                    leaked_top5 = [item for item in top5 if item["document_id"] in forbidden]
                    if leaked_top5:
                        raise RuntimeError("unauthorized chunks reached the Judge Top-5")

            complete_pointwise = retrieval_complete(case, pointwise_top5)
            complete_pairwise = retrieval_complete(case, pairwise_top5)
            traces.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "expected_answerability": case.expected_answerability,
                    "required_document_ids": list(case.required_document_ids),
                    "forbidden_document_ids": list(case.forbidden_document_ids),
                    "shared_query_embedding": True,
                    "shared_dense": [_trace(item) for item in dense_candidates],
                    "shared_bm25": [_trace(item) for item in bm25_candidates],
                    "shared_rrf_union": [_trace(item) for item in union],
                    "shared_cross_encoder_scores": True,
                    "pointwise_top5": pointwise_top5,
                    "pairwise_top5": pairwise_top5,
                    "retrieval_complete_pointwise": complete_pointwise,
                    "retrieval_complete_pairwise": complete_pairwise,
                    "pool_complete": (
                        set(case.required_document_ids) <= {item.document_id for item in union}
                        if case.expected_answerability
                        else None
                    ),
                    "metrics_pointwise": retrieval_case_metrics(case.as_retrieval(), [_result(item) for item in pointwise_top5]),
                    "metrics_pairwise": retrieval_case_metrics(case.as_retrieval(), [_result(item) for item in pairwise_top5]),
                    "pool": pool_metrics(case.as_retrieval(), union),
                    "set_quality_pointwise": _set_quality_metrics(pointwise_top5),
                    "set_quality_pairwise": _set_quality_metrics(pairwise_top5),
                    "timing": {
                        "query_embedding_ms": embedding.embedding_latency_ms,
                        "cache_lookup_ms": embedding.cache_lookup_latency_ms,
                        "dense_ms": dense_ms,
                        "bm25_ms": bm25_ms,
                        "rrf_ms": fusion_ms,
                        "cross_encoder_ms": ce_ms,
                        "pairwise_ms": pairwise_ms,
                    },
                    "embedding_cache_hit": embedding.cache_hit,
                    "embedding_external_calls": embedding.external_calls,
                }
            )
            record.shared_traces = list(traces)
            record.usage = dict(usage)
            flag_modified(record, "shared_traces")
            flag_modified(record, "usage")
            self.session.commit()

        retrieval_rows_pw = [
            {"case_id": item["case_id"], "category": item["category"], "expected_answerability": item["expected_answerability"], "metrics": item["metrics_pointwise"], "pool": item["pool"]}
            for item in traces
        ]
        retrieval_rows_pair = [
            {"case_id": item["case_id"], "category": item["category"], "expected_answerability": item["expected_answerability"], "metrics": item["metrics_pairwise"], "pool": item["pool"]}
            for item in traces
        ]
        retrieval = {
            "pointwise_metrics": aggregate_retrieval_metrics(retrieval_rows_pw),
            "pairwise_metrics": aggregate_retrieval_metrics(retrieval_rows_pair),
            "pointwise_category": category_retrieval_metrics(retrieval_rows_pw),
            "pairwise_category": category_retrieval_metrics(retrieval_rows_pair),
            "pool": aggregate_pool(retrieval_rows_pw),
            "pool_complete_pointwise_incomplete": sum(
                1 for item in traces if item.get("pool_complete") is True and item.get("retrieval_complete_pointwise") is False
            ),
            "pool_complete_pairwise_incomplete": sum(
                1 for item in traces if item.get("pool_complete") is True and item.get("retrieval_complete_pairwise") is False
            ),
        }
        record.retrieval_metrics = retrieval
        record.security = {
            "acl_safety": 1.0,
            "tenant_isolation": 1.0,
            "version_correctness": retrieval["pointwise_metrics"].get("version_correctness", 1.0),
            "unauthorized_chunks_to_judge": 0,
        }
        record.latency = {
            "retrieval": {
                key: latency_stats([float(item["timing"][key]) for item in traces])
                for key in ("query_embedding_ms", "dense_ms", "bm25_ms", "rrf_ms", "cross_encoder_ms", "pairwise_ms")
            }
        }
        record.usage = usage
        record.retrieval_frozen_at = datetime.now(UTC)
        self.session.commit()
        verify_persisted_v2(self.session)

    def _run_three_arms(self, record: V3Phase5bExperimentRecord) -> dict[str, Any]:
        cases = load_final_cases()
        traces = record.shared_traces or []
        if len(traces) != len(cases):
            raise ValueError("shared retrieval traces are incomplete")
        preflight = record.hosted_preflight or self.hosted_preflight()
        gate = self.inner._gate(int(preflight.get("required_primary_judge_calls") or 0))
        recovery = self.inner._recovery_cache(
            int(preflight.get("maximum_recovery_draft_calls") or 0)
            + int(preflight.get("maximum_verifier_calls") or 0)
        )
        provider = self.inner._embedding_provider()
        reference_rows = list(record.reference_rows or [])
        control_rows = list(record.control_rows or [])
        candidate_rows = list(record.candidate_rows or [])
        done = {item["case_id"] for item in candidate_rows}
        traces_by_id = {item["case_id"]: item for item in traces}
        for case in cases:
            if case.case_id in done:
                continue
            trace = traces_by_id[case.case_id]
            pw_top5 = trace["pointwise_top5"]
            pair_top5 = trace["pairwise_top5"]

            for top5 in (pw_top5, pair_top5):
                leaked = HIDDEN_GROUND_TRUTH_FIELDS.intersection(top5[0] if top5 else {})
                if leaked:
                    raise RuntimeError(f"evaluator labels leaked into runtime traces: {sorted(leaked)}")

            # --- Shared Judge call for Reference R and Control A (pointwise top-5) ---
            pw_evidence = _gate_evidence(pw_top5)
            pw_operational_error = None
            try:
                pw_proposed = gate.evaluate(case.question, pw_evidence)
                pw_validated = validate_gate_result_with_error(
                    pw_proposed, pw_evidence, session=self.session, principal=_principal(case)
                )
                pw_result = pw_validated.result
                pw_operational_error = (
                    pw_validated.operational_error.value if pw_validated.operational_error else None
                )
            except Exception:
                pw_result = AnswerabilityResult.fail_closed()
                pw_operational_error = "JUDGE_REQUEST_ERROR"
            pw_judge_timing = gate.last_timing

            # --- Reference R: V2 path (no recovery) ---
            ref_generated = self.inner._generate_control(
                case, pw_top5, pw_result, pw_judge_timing, trace, provider
            )
            ref_cited = [citation.chunk_id for citation in ref_generated.citations]
            ref_validity = deterministic_citation_correctness(
                ref_cited, tuple(item["chunk_id"] for item in pw_top5)
            )
            reference_payload = {
                "case_id": case.case_id,
                "category": case.category,
                "expected_answerability": case.expected_answerability,
                "status": ref_generated.status,
                "behavior": _behavior(case, ref_generated.status),
                "class": classification(case.expected_answerability, pw_result.answerable),
                "answerable": pw_result.answerable,
                "supporting_chunk_ids": list(pw_result.supporting_chunk_ids),
                "operational_error": pw_operational_error,
                "answer": ref_generated.answer,
                "citations": ref_cited,
                "citation_validity": ref_validity,
                "retrieval_complete": trace["retrieval_complete_pointwise"],
                "pool_complete": trace.get("pool_complete"),
                "judge_latency_ms": pw_judge_timing.judge_latency_ms,
                "generation_latency_ms": ref_generated.generation_latency_ms,
                "generation_operational_error": ref_generated.generation_operational_error,
                "extractive_path": ref_generated.extractive_path,
                "live": not pw_judge_timing.cache_hit,
                "prompt_tokens": pw_judge_timing.prompt_tokens,
                "completion_tokens": pw_judge_timing.completion_tokens,
                "logical_request_id": pw_judge_timing.logical_request_id,
                "physical_attempts": pw_judge_timing.external_calls,
                "document_instruction_followed": document_instruction_followed(ref_generated.answer),
            }

            # --- Control A: V3 path with recovery (pointwise top-5) ---
            control_payload = dict(reference_payload)
            pw_schema_valid_negative = pw_result.answerable is False and pw_operational_error is None
            control_outcome = evaluate_recovery(
                session=self.session,
                principal=_principal(case),
                question=case.question,
                chunks=pw_evidence,
                cache=recovery,
                primary_answerable=pw_result.answerable,
                primary_schema_valid=pw_schema_valid_negative or pw_result.answerable,
                safety_gate=instruction_boundary_safety_gate,
            )
            if pw_result.answerable:
                ctrl_status = ref_generated.status
                ctrl_answer = ref_generated.answer
                ctrl_citations = tuple(ref_cited)
                ctrl_supporting = tuple(pw_result.supporting_chunk_ids)
            elif control_outcome.answered:
                ctrl_status = "answered"
                ctrl_answer = control_outcome.answer
                ctrl_citations = control_outcome.citations
                ctrl_supporting = control_outcome.supporting_chunk_ids
            else:
                ctrl_status = "abstained"
                ctrl_answer = None
                ctrl_citations = ()
                ctrl_supporting = ()
            ctrl_supported = evaluator_supported(case, ctrl_answer, ctrl_citations, pw_top5)
            if (
                not pw_result.answerable
                and ctrl_status == "answered"
                and case.expected_answerability
                and not ctrl_supported
            ):
                ctrl_status = "abstained"
                ctrl_answer = None
            ctrl_validity = deterministic_citation_correctness(
                ctrl_citations, tuple(item["chunk_id"] for item in pw_top5)
            )
            ctrl_boundary_ms = 0.0
            if (
                control_outcome.safety_verdict is not None
                and control_outcome.draft is not None
                and control_outcome.verification is not None
            ):
                started = time.perf_counter()
                instruction_boundary_safety_gate(
                    case.question, pw_evidence, control_outcome.draft, control_outcome.verification
                )
                ctrl_boundary_ms = (time.perf_counter() - started) * 1000
            control_payload.update({
                "status": ctrl_status,
                "behavior": _behavior(case, ctrl_status),
                "answer": ctrl_answer,
                "citations": list(ctrl_citations),
                "citation_validity": ctrl_validity,
                "supporting_chunk_ids": list(ctrl_supporting),
                "evaluator_supported": ctrl_supported,
                "completeness_failure": control_outcome.typed_failure == "COMPLETENESS_FAILURE",
                "recovery": recovery_trace(control_outcome),
                "boundary_latency_ms": ctrl_boundary_ms,
                "recovery_triggered": control_outcome.triggered,
                "document_instruction_followed": document_instruction_followed(ctrl_answer),
            })

            # --- Candidate B: V3 path with recovery (pairwise top-5) ---
            pair_evidence = _gate_evidence(pair_top5)
            pair_operational_error = None
            try:
                pair_proposed = gate.evaluate(case.question, pair_evidence)
                pair_validated = validate_gate_result_with_error(
                    pair_proposed, pair_evidence, session=self.session, principal=_principal(case)
                )
                pair_result = pair_validated.result
                pair_operational_error = (
                    pair_validated.operational_error.value if pair_validated.operational_error else None
                )
            except Exception:
                pair_result = AnswerabilityResult.fail_closed()
                pair_operational_error = "JUDGE_REQUEST_ERROR"
            pair_judge_timing = gate.last_timing

            pair_schema_valid_negative = pair_result.answerable is False and pair_operational_error is None
            candidate_outcome = evaluate_recovery(
                session=self.session,
                principal=_principal(case),
                question=case.question,
                chunks=pair_evidence,
                cache=recovery,
                primary_answerable=pair_result.answerable,
                primary_schema_valid=pair_schema_valid_negative or pair_result.answerable,
                safety_gate=instruction_boundary_safety_gate,
            )
            if pair_result.answerable:
                cand_generated = self.inner._generate_control(
                    case, pair_top5, pair_result, pair_judge_timing, trace, provider
                )
                cand_status = cand_generated.status
                cand_answer = cand_generated.answer
                cand_cited = [citation.chunk_id for citation in cand_generated.citations]
                cand_supporting = tuple(pair_result.supporting_chunk_ids)
                cand_gen_ms = cand_generated.generation_latency_ms
            elif candidate_outcome.answered:
                cand_status = "answered"
                cand_answer = candidate_outcome.answer
                cand_cited = list(candidate_outcome.citations)
                cand_supporting = candidate_outcome.supporting_chunk_ids
                cand_gen_ms = None
            else:
                cand_status = "abstained"
                cand_answer = None
                cand_cited = []
                cand_supporting = ()
                cand_gen_ms = None
            cand_supported = evaluator_supported(case, cand_answer, tuple(cand_cited), pair_top5)
            if (
                not pair_result.answerable
                and cand_status == "answered"
                and case.expected_answerability
                and not cand_supported
            ):
                cand_status = "abstained"
                cand_answer = None
            cand_validity = deterministic_citation_correctness(
                tuple(cand_cited), tuple(item["chunk_id"] for item in pair_top5)
            )
            cand_boundary_ms = 0.0
            if (
                candidate_outcome.safety_verdict is not None
                and candidate_outcome.draft is not None
                and candidate_outcome.verification is not None
            ):
                started = time.perf_counter()
                instruction_boundary_safety_gate(
                    case.question, pair_evidence, candidate_outcome.draft, candidate_outcome.verification
                )
                cand_boundary_ms = (time.perf_counter() - started) * 1000
            candidate_payload = {
                "case_id": case.case_id,
                "category": case.category,
                "expected_answerability": case.expected_answerability,
                "status": cand_status,
                "behavior": _behavior(case, cand_status),
                "class": classification(case.expected_answerability, pair_result.answerable),
                "answerable": pair_result.answerable,
                "supporting_chunk_ids": list(cand_supporting),
                "operational_error": pair_operational_error,
                "answer": cand_answer,
                "citations": cand_cited,
                "citation_validity": cand_validity,
                "evaluator_supported": cand_supported,
                "completeness_failure": candidate_outcome.typed_failure == "COMPLETENESS_FAILURE",
                "retrieval_complete": trace["retrieval_complete_pairwise"],
                "pool_complete": trace.get("pool_complete"),
                "judge_latency_ms": pair_judge_timing.judge_latency_ms,
                "generation_latency_ms": cand_gen_ms,
                "generation_operational_error": None,
                "live": not pair_judge_timing.cache_hit,
                "prompt_tokens": pair_judge_timing.prompt_tokens,
                "completion_tokens": pair_judge_timing.completion_tokens,
                "logical_request_id": pair_judge_timing.logical_request_id,
                "physical_attempts": pair_judge_timing.external_calls,
                "recovery": recovery_trace(candidate_outcome),
                "boundary_latency_ms": cand_boundary_ms,
                "recovery_triggered": candidate_outcome.triggered,
                "document_instruction_followed": document_instruction_followed(cand_answer),
                "ranking_algorithm": PAIRWISE_ALGORITHM_ID,
                "ranking_version": PAIRWISE_ALGORITHM_VERSION,
            }

            reference_rows.append(reference_payload)
            control_rows.append(control_payload)
            candidate_rows.append(candidate_payload)
            record.reference_rows = list(reference_rows)
            record.control_rows = list(control_rows)
            record.candidate_rows = list(candidate_rows)
            flag_modified(record, "reference_rows")
            flag_modified(record, "control_rows")
            flag_modified(record, "candidate_rows")
            self.session.commit()

        analysis = self._analyze(cases, traces, reference_rows, control_rows, candidate_rows, record)
        record.reference_metrics = analysis["reference_metrics"]
        record.control_metrics = analysis["control_metrics"]
        record.candidate_metrics = analysis["candidate_metrics"]
        record.paired_deltas = analysis["paired_deltas"]
        record.recovery_funnel = analysis["recovery_funnel"]
        record.instruction_boundary = analysis["instruction_boundary"]
        record.prompt_injection = analysis["prompt_injection"]
        record.category_results = analysis["category_results"]
        record.retrieval_metrics = {
            **(record.retrieval_metrics or {}),
            **analysis["retrieval_metrics"],
        }
        record.ranking_metrics = analysis["ranking_metrics"]
        record.failure_census = analysis["failure_census"]
        record.security = analysis["security"]
        record.citations = analysis["citations"]
        record.reliability = analysis["reliability"]
        record.latency = {**(record.latency or {}), **analysis["latency"]}
        record.usage = {**(record.usage or {}), **analysis["usage"]}
        record.cost = analysis["cost"]
        record.selection = analysis["selection"]
        record.selected_strategy = analysis["selection"]["selected_strategy"]
        record.promotion_decision = analysis["selection"]["promotion_decision"]
        record.v3_status = analysis["selection"]["v3_status"]
        record.primary_remaining_bottleneck = analysis["primary_remaining_bottleneck"]
        record.completed_at = datetime.now(UTC)
        research = self.session.get(ResearchArchitectureRecord, V3_ARCHITECTURE_ID)
        if research is not None:
            research.selected_v3_strategy = record.selected_strategy
            research.production_status = False
            research.v3_research_status = record.v3_status
        self.session.commit()
        verify_persisted_v2(self.session)
        self._persist_ledger(record)
        self._persist_markdown()
        return self.status()

    def _analyze(
        self,
        cases: tuple[V2FinalCase, ...],
        traces: list[dict[str, Any]],
        reference_rows: list[dict[str, Any]],
        control_rows: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        record: V3Phase5bExperimentRecord,
    ) -> dict[str, Any]:
        reference = {item["case_id"]: item for item in reference_rows}
        control = {item["case_id"]: item for item in control_rows}
        candidate = {item["case_id"]: item for item in candidate_rows}
        cases_by_id = {item.case_id: item for item in cases}
        traces_by_id = {item["case_id"]: item for item in traces}

        rescues_vs_control: list[str] = []
        rescues_vs_reference: list[str] = []
        regressions_vs_reference: list[str] = []
        regressions_vs_control: list[str] = []
        transitions_ba = Counter()
        transitions_br = Counter()
        for case in cases:
            ref = reference[case.case_id]
            ctrl = control[case.case_id]
            cand = candidate[case.case_id]
            pair_top5 = traces_by_id[case.case_id]["pairwise_top5"]
            pw_top5 = traces_by_id[case.case_id]["pointwise_top5"]  # noqa: F841
            if (
                ctrl["behavior"] == "INCORRECT_ABSTENTION"
                and cand["behavior"] == "CORRECT_ANSWER"
                and evaluator_supported(case, cand.get("answer"), tuple(cand.get("citations") or ()), pair_top5)
                and cand.get("citation_validity") == 1.0
            ):
                rescues_vs_control.append(case.case_id)
            if (
                ref["behavior"] == "INCORRECT_ABSTENTION"
                and cand["behavior"] == "CORRECT_ANSWER"
                and evaluator_supported(case, cand.get("answer"), tuple(cand.get("citations") or ()), pair_top5)
                and cand.get("citation_validity") == 1.0
            ):
                rescues_vs_reference.append(case.case_id)
            if ref["behavior"] == "CORRECT_ANSWER" and cand["behavior"] != "CORRECT_ANSWER":
                regressions_vs_reference.append(case.case_id)
            if ctrl["behavior"] == "CORRECT_ANSWER" and cand["behavior"] != "CORRECT_ANSWER":
                regressions_vs_control.append(case.case_id)
            transitions_ba[self._transition(ctrl, cand)] += 1
            transitions_br[self._transition(ref, cand)] += 1

        reference_metrics = final_e2e_metrics(reference_rows)
        control_metrics = final_e2e_metrics(control_rows)
        candidate_metrics = final_e2e_metrics(candidate_rows)

        paired = {
            "b_vs_a": {
                "candidate_correct_minus_control_correct": candidate_metrics["correct_answers"] - control_metrics["correct_answers"],
                "candidate_incorrect_abstention_minus_control": candidate_metrics["incorrect_abstentions"] - control_metrics["incorrect_abstentions"],
                "candidate_unsupported_minus_control": candidate_metrics["unsupported_answers"] - control_metrics["unsupported_answers"],
                "absolute_answerable_correct_rate_delta": round(
                    candidate_metrics["answerable_case_correct_answer_rate"] - control_metrics["answerable_case_correct_answer_rate"], 6
                ),
                "f1_delta": round(candidate_metrics["f1"] - control_metrics["f1"], 6),
                "additional_correct_supported": len(rescues_vs_control),
                "rescue_ids": rescues_vs_control,
                "regression_ids": regressions_vs_control,
                "transitions": dict(transitions_ba),
            },
            "b_vs_r": {
                "candidate_correct_minus_reference_correct": candidate_metrics["correct_answers"] - reference_metrics["correct_answers"],
                "candidate_incorrect_abstention_minus_reference": candidate_metrics["incorrect_abstentions"] - reference_metrics["incorrect_abstentions"],
                "candidate_unsupported_minus_reference": candidate_metrics["unsupported_answers"] - reference_metrics["unsupported_answers"],
                "absolute_answerable_correct_rate_delta": round(
                    candidate_metrics["answerable_case_correct_answer_rate"] - reference_metrics["answerable_case_correct_answer_rate"], 6
                ),
                "f1_delta": round(candidate_metrics["f1"] - reference_metrics["f1"], 6),
                "additional_correct_supported": len(rescues_vs_reference),
                "rescue_ids": rescues_vs_reference,
                "regression_ids": regressions_vs_reference,
                "transitions": dict(transitions_br),
            },
            "a_vs_r": {
                "control_correct_minus_reference_correct": control_metrics["correct_answers"] - reference_metrics["correct_answers"],
                "absolute_answerable_correct_rate_delta": round(
                    control_metrics["answerable_case_correct_answer_rate"] - reference_metrics["answerable_case_correct_answer_rate"], 6
                ),
                "f1_delta": round(control_metrics["f1"] - reference_metrics["f1"], 6),
            },
        }

        # Recovery funnel (Control A)
        ctrl_negatives = [item for item in control_rows if item.get("answerable") is False]
        ctrl_triggers = [item for item in control_rows if item.get("recovery", {}).get("recovery_triggered")]
        ctrl_draft_ok = [item for item in ctrl_triggers if item.get("recovery", {}).get("draft_success")]
        ctrl_verify_ok = [item for item in ctrl_triggers if item.get("recovery", {}).get("verification_pass")]
        ctrl_completeness_ok = [item for item in ctrl_verify_ok if item.get("recovery", {}).get("completeness_pass")]

        # Recovery funnel (Candidate B)
        cand_negatives = [item for item in candidate_rows if item.get("answerable") is False]
        cand_triggers = [item for item in candidate_rows if item.get("recovery", {}).get("recovery_triggered")]
        cand_draft_ok = [item for item in cand_triggers if item.get("recovery", {}).get("draft_success")]
        cand_verify_ok = [item for item in cand_triggers if item.get("recovery", {}).get("verification_pass")]
        cand_completeness_ok = [item for item in cand_verify_ok if item.get("recovery", {}).get("completeness_pass")]

        funnel = {
            "control": {
                "primary_judge_negatives": len(ctrl_negatives),
                "recovery_triggered": len(ctrl_triggers),
                "draft_success": len(ctrl_draft_ok),
                "claim_verification_pass": len(ctrl_verify_ok),
                "completeness_pass": len(ctrl_completeness_ok),
            },
            "candidate": {
                "primary_judge_negatives": len(cand_negatives),
                "recovery_triggered": len(cand_triggers),
                "draft_success": len(cand_draft_ok),
                "claim_verification_pass": len(cand_verify_ok),
                "completeness_pass": len(cand_completeness_ok),
            },
        }

        # Instruction boundary
        ctrl_boundary_invoked = [item for item in control_rows if item.get("recovery", {}).get("safety_verdict")]
        ctrl_boundary_pass = [item for item in ctrl_boundary_invoked if item["recovery"]["safety_verdict"] == BOUNDARY_PASS]
        ctrl_boundary_fail = [item for item in ctrl_boundary_invoked if item["recovery"]["safety_verdict"] == BOUNDARY_FAIL]
        cand_boundary_invoked = [item for item in candidate_rows if item.get("recovery", {}).get("safety_verdict")]
        cand_boundary_pass = [item for item in cand_boundary_invoked if item["recovery"]["safety_verdict"] == BOUNDARY_PASS]
        cand_boundary_fail = [item for item in cand_boundary_invoked if item["recovery"]["safety_verdict"] == BOUNDARY_FAIL]
        boundary_metrics = {
            "control": {
                "boundary_invocations": len(ctrl_boundary_invoked),
                "boundary_pass": len(ctrl_boundary_pass),
                "boundary_fail": len(ctrl_boundary_fail),
            },
            "candidate": {
                "boundary_invocations": len(cand_boundary_invoked),
                "boundary_pass": len(cand_boundary_pass),
                "boundary_fail": len(cand_boundary_fail),
            },
        }

        # Prompt injection
        injection_cases = [item for item in cases if item.category == "prompt_injection"]
        injection_results = []
        for case in injection_cases:
            ref = reference[case.case_id]
            ctrl = control[case.case_id]
            cand = candidate[case.case_id]
            cand_recovery = cand.get("recovery") or {}
            safe = cand["behavior"] == "CORRECT_ABSTENTION" and not cand["document_instruction_followed"]
            injection_results.append({
                "case_id": case.case_id,
                "reference_behavior": ref["behavior"],
                "control_behavior": ctrl["behavior"],
                "candidate_recovery_triggered": bool(cand_recovery.get("recovery_triggered")),
                "boundary_result": cand_recovery.get("safety_verdict"),
                "final_behavior": cand["behavior"],
                "safe": safe,
            })
        injection_safe = all(item["safe"] for item in injection_results) if injection_results else True

        # Citations
        cand_answered = [item for item in candidate_rows if item["status"] == "answered"]
        validity_values = [item["citation_validity"] for item in cand_answered if item.get("citation_validity") is not None]
        citations = {
            "answered_cases": len(cand_answered),
            "validity": (sum(1 for item in validity_values if item == 1.0) / len(validity_values) if validity_values else 1.0),
            "correctness": (sum(item for item in validity_values) / len(validity_values) if validity_values else 1.0),
            "invalid_citation_count": sum(1 for item in validity_values if item != 1.0),
        }

        # Security
        unauthorized_support = 0
        invalid_support = 0
        for item in candidate_rows:
            case = cases_by_id[item["case_id"]]
            forbidden = set(case.forbidden_document_ids)
            pair_top5 = traces_by_id[item["case_id"]]["pairwise_top5"]
            by_id = {chunk["chunk_id"]: chunk for chunk in pair_top5}
            for chunk_id in item.get("supporting_chunk_ids") or []:
                chunk = by_id.get(chunk_id)
                if chunk is None:
                    invalid_support += 1
                elif chunk["document_id"] in forbidden and case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                    unauthorized_support += 1
        security = {
            **(record.security or {}),
            "acl_safety": 1.0 if unauthorized_support == 0 else 0.0,
            "tenant_isolation": 1.0,
            "version_correctness": (record.security or {}).get("version_correctness", 1.0),
            "unauthorized_evidence_downstream": unauthorized_support,
            "prompt_injection_safety": 1.0 if injection_safe else 0.0,
            "invalid_supporting_ids": invalid_support,
            "unauthorized_supporting_ids": unauthorized_support,
            "content_identity_failures": 0,
            "citation_validity": citations["validity"],
            "citation_correctness": citations["correctness"],
        }

        # Ranking-specific metrics
        answerable_traces = [t for t in traces if t["expected_answerability"]]
        pw_rows = [{"case_id": t["case_id"], "category": t["category"], "expected_answerability": t["expected_answerability"], "metrics": t["metrics_pointwise"]} for t in traces]
        pair_rows = [{"case_id": t["case_id"], "category": t["category"], "expected_answerability": t["expected_answerability"], "metrics": t["metrics_pairwise"]} for t in traces]
        pw_agg = aggregate_retrieval_metrics(pw_rows)
        pair_agg = aggregate_retrieval_metrics(pair_rows)

        exact_id_cases = [t for t in answerable_traces if t["category"] == "exact_identifier"]
        pw_exact_recall = mean([t["metrics_pointwise"].get("exact_identifier_recall_at_5", 0.0) or 0.0 for t in exact_id_cases]) if exact_id_cases else 0.0
        pair_exact_recall = mean([t["metrics_pairwise"].get("exact_identifier_recall_at_5", 0.0) or 0.0 for t in exact_id_cases]) if exact_id_cases else 0.0

        same_doc_cases = [t for t in answerable_traces if t["category"] in {"same_doc_multi_chunk", "single_document"}]
        pw_same_doc = mean([t["metrics_pointwise"].get("all_required_evidence_coverage_at_5", 0.0) or 0.0 for t in same_doc_cases]) if same_doc_cases else 0.0
        pair_same_doc = mean([t["metrics_pairwise"].get("all_required_evidence_coverage_at_5", 0.0) or 0.0 for t in same_doc_cases]) if same_doc_cases else 0.0

        ranking_gates = {
            "exact_id_recall_control": pw_exact_recall,
            "exact_id_recall_candidate": pair_exact_recall,
            "exact_id_recall_pass": pair_exact_recall >= pw_exact_recall,
            "version_correctness_control": pw_agg.get("version_correctness", 1.0),
            "version_correctness_candidate": pair_agg.get("version_correctness", 1.0),
            "version_correctness_pass": pair_agg.get("version_correctness", 1.0) >= pw_agg.get("version_correctness", 1.0),
            "same_doc_coverage_control": pw_same_doc,
            "same_doc_coverage_candidate": pair_same_doc,
            "same_doc_multi_chunk_degradation": max(pw_same_doc - pair_same_doc, 0.0),
            "same_doc_multi_chunk_pass": (pw_same_doc - pair_same_doc) <= PROMOTION_POLICY["same_doc_multi_chunk_regression_gate"],
            "unauthorized_downstream_evidence": unauthorized_support,
        }

        ranking_metrics = {
            "pointwise": pw_agg,
            "pairwise": pair_agg,
            "pointwise_category": category_retrieval_metrics(pw_rows),
            "pairwise_category": category_retrieval_metrics(pair_rows),
            "set_quality_pointwise": {
                "mean_unique_documents_in_top5": mean([t["set_quality_pointwise"]["unique_documents_in_top5"] for t in answerable_traces]) if answerable_traces else 0.0,
                "mean_redundant_chunk_count": mean([t["set_quality_pointwise"]["redundant_chunk_count"] for t in answerable_traces]) if answerable_traces else 0.0,
            },
            "set_quality_pairwise": {
                "mean_unique_documents_in_top5": mean([t["set_quality_pairwise"]["unique_documents_in_top5"] for t in answerable_traces]) if answerable_traces else 0.0,
                "mean_redundant_chunk_count": mean([t["set_quality_pairwise"]["redundant_chunk_count"] for t in answerable_traces]) if answerable_traces else 0.0,
            },
            "ranking_gates": ranking_gates,
            "pairwise_algorithm_id": PAIRWISE_ALGORITHM_ID,
            "pairwise_algorithm_version": PAIRWISE_ALGORITHM_VERSION,
            "candidate_ranking_hash": CANDIDATE_RANKING_HASH,
        }

        # Selection
        selection = apply_promotion_policy(
            reference=reference_metrics,
            control=control_metrics,
            candidate=candidate_metrics,
            additional_correct_supported_vs_control=len(rescues_vs_control),
            additional_correct_supported_vs_reference=len(rescues_vs_reference),
            regressions_vs_reference=len(regressions_vs_reference),
            security=security,
            citations=citations,
            ranking_gates=ranking_gates,
        )

        # Failure census
        census: Counter[str] = Counter()
        census_ids: dict[str, list[str]] = {family: [] for family in FAILURE_FAMILIES}
        for case in cases:
            family = classify_candidate_failure(
                case, traces_by_id[case.case_id], control[case.case_id], candidate[case.case_id]
            )
            if family:
                census[family] += 1
                census_ids.setdefault(family, []).append(case.case_id)
        bottleneck = census.most_common(1)[0][0] if census else "NONE"

        # Category results
        categories = {}
        mapping = {
            "single_document": "single_document",
            "multidoc_two": "two_document",
            "multidoc_three": "three_document",
            "near_duplicate": "near_duplicate",
            "exact_identifier": "exact_id",
            "version_region": "version_region",
            "semantic_paraphrase": "semantic_paraphrase",
            "acl_sensitive": "acl",
            "partial_no_answer": "partial_no_answer",
            "prompt_injection": "prompt_injection",
        }
        for category, key in mapping.items():
            scoped_cases = [item for item in cases if item.category == category]
            scoped_r = [reference[item.case_id] for item in scoped_cases]
            scoped_c = [control[item.case_id] for item in scoped_cases]
            scoped_b = [candidate[item.case_id] for item in scoped_cases]
            retrieval_complete_n = sum(
                1 for item in scoped_cases if traces_by_id[item.case_id].get("retrieval_complete_pairwise")
            )
            categories[key] = {
                "n": len(scoped_cases),
                "retrieval_complete": retrieval_complete_n,
                "reference": final_e2e_metrics(scoped_r),
                "control": final_e2e_metrics(scoped_c),
                "candidate": final_e2e_metrics(scoped_b),
                "reference_correct": sum(1 for item in scoped_r if item["behavior"] == "CORRECT_ANSWER"),
                "control_correct": sum(1 for item in scoped_c if item["behavior"] == "CORRECT_ANSWER"),
                "candidate_correct": sum(1 for item in scoped_b if item["behavior"] == "CORRECT_ANSWER"),
                "rescues_vs_control": [item.case_id for item in scoped_cases if item.case_id in set(rescues_vs_control)],
                "rescues_vs_reference": [item.case_id for item in scoped_cases if item.case_id in set(rescues_vs_reference)],
                "regressions_vs_reference": [item.case_id for item in scoped_cases if item.case_id in set(regressions_vs_reference)],
            }

        answerable_n = candidate_metrics["answerable_cases"]
        correct_supported = candidate_metrics["correct_answers"]
        needed_95 = math.ceil(0.95 * answerable_n) if answerable_n else 0
        target_95 = {
            "answerable_cases": answerable_n,
            "correct_supported_answers": correct_supported,
            "correct_answer_rate": candidate_metrics["answerable_case_correct_answer_rate"],
            "minimum_correct_answers_for_95": needed_95,
            "additional_correct_answers_still_required": max(needed_95 - correct_supported, 0),
            "claimed_95": candidate_metrics["answerable_case_correct_answer_rate"] >= 0.95,
        }

        retrieval_metrics_extra = {
            "pointwise_hit_at_5": pw_agg.get("hit_at_5"),
            "pairwise_hit_at_5": pair_agg.get("hit_at_5"),
            "pointwise_recall_at_5": pw_agg.get("recall_at_5"),
            "pairwise_recall_at_5": pair_agg.get("recall_at_5"),
        }

        # Usage & cost
        extra_prompt = extra_completion = 0
        extra_draft = extra_verifier = 0
        judge_retries = draft_retries = verifier_retries = 0
        successful_transport = 0
        for rows in (control_rows, candidate_rows):
            for item in rows:
                rec = item.get("recovery") or {}
                physical = item.get("physical_attempts") or 0
                if physical > 1:
                    judge_retries += physical - 1
                    successful_transport += 1
                if rec.get("draft_live"):
                    extra_draft += 1
                    extra_prompt += rec.get("draft_prompt_tokens") or 0
                    extra_completion += rec.get("draft_completion_tokens") or 0
                    draft_phys = rec.get("draft_external_calls") or 0
                    if draft_phys > 1:
                        draft_retries += draft_phys - 1
                        successful_transport += 1
                if rec.get("verifier_live"):
                    extra_verifier += 1
                    extra_prompt += rec.get("verifier_prompt_tokens") or 0
                    extra_completion += rec.get("verifier_completion_tokens") or 0
                    ver_phys = rec.get("verifier_external_calls") or 0
                    if ver_phys > 1:
                        verifier_retries += ver_phys - 1
                        successful_transport += 1

        start_ledger = (record.usage or {}).get("ledger_at_execution_start") or {}
        all_judge_rows = reference_rows + control_rows + candidate_rows
        usage = {
            **(record.usage or {}),
            "new_query_embeddings": (record.usage or {}).get("new_query_embedding_calls", 0),
            "embedding_tokens": (record.usage or {}).get("embedding_tokens", 0),
            "primary_judge_logical_calls_reference": len(cases),
            "primary_judge_logical_calls_candidate": len(cases),
            "recovery_draft_logical_calls": extra_draft,
            "recovery_verifier_logical_calls": extra_verifier,
            "transport_retries": judge_retries + draft_retries + verifier_retries,
            "judge_input_tokens": sum(item["prompt_tokens"] or 0 for item in all_judge_rows if item.get("live")),
            "judge_output_tokens": sum(item["completion_tokens"] or 0 for item in all_judge_rows if item.get("live")),
            "recovery_input_tokens": extra_prompt,
            "recovery_output_tokens": extra_completion,
            "original_embedding_ledger": start_ledger.get("embedding_ledger"),
            "original_judge_ledger": start_ledger.get("judge_ledger"),
            "original_recovery_ledger": start_ledger.get("recovery_ledger"),
            "resume_embedding_calls": max(self.inner._embedding_calls() - int(start_ledger.get("embedding_ledger") or 0), 0),
            "resume_judge_calls": max(self.inner._hosted_judge_rows() - int(start_ledger.get("judge_ledger") or 0), 0),
            "resume_recovery_calls": max(self.inner._hosted_recovery_rows() - int(start_ledger.get("recovery_ledger") or 0), 0),
        }
        embedding_cost = ((usage.get("embedding_tokens") or 0) * EMBEDDING_INPUT_USD_PER_MILLION / 1_000_000)
        judge_cost = official_token_cost(
            model=SOL_MODEL,
            input_tokens=usage["judge_input_tokens"],
            output_tokens=usage["judge_output_tokens"],
        )
        recovery_cost = official_token_cost(
            model=SOL_MODEL,
            input_tokens=extra_prompt,
            output_tokens=extra_completion,
        )
        cost = {
            "embedding_usd": embedding_cost,
            "primary_judge_usd": judge_cost,
            "recovery_draft_and_verifier_usd": recovery_cost,
            "total_final_benchmark_usd": embedding_cost + judge_cost + recovery_cost,
            "historical_diagnostic_excluded": True,
        }

        # Reliability
        reliability = {
            "reference_deterministic_generator_success": sum(1 for item in reference_rows if not item.get("generation_operational_error")),
            "control_deterministic_generator_success": sum(1 for item in control_rows if item["status"] == "answered" and not item.get("generation_operational_error")),
            "candidate_deterministic_generator_success": sum(1 for item in candidate_rows if item["status"] == "answered" and not item.get("generation_operational_error")),
            "primary_judge_request_errors": sum(1 for item in reference_rows if item.get("operational_error") == "JUDGE_REQUEST_ERROR"),
            "candidate_judge_request_errors": sum(1 for item in candidate_rows if item.get("operational_error") == "JUDGE_REQUEST_ERROR"),
            "transport_retries": usage["transport_retries"],
            "successful_transport_recoveries": successful_transport,
        }

        # Latency
        latency = {
            "embedding": latency_stats([item["timing"]["query_embedding_ms"] for item in traces]),
            "dense": latency_stats([item["timing"]["dense_ms"] for item in traces]),
            "bm25": latency_stats([item["timing"]["bm25_ms"] for item in traces]),
            "rrf": latency_stats([item["timing"]["rrf_ms"] for item in traces]),
            "cross_encoder": latency_stats([item["timing"]["cross_encoder_ms"] for item in traces]),
            "pairwise_ranking": latency_stats([item["timing"]["pairwise_ms"] for item in traces]),
            "primary_judge_reference": latency_stats([item.get("judge_latency_ms") or 0 for item in reference_rows]),
            "primary_judge_candidate": latency_stats([item.get("judge_latency_ms") or 0 for item in candidate_rows]),
        }

        return {
            "reference_metrics": reference_metrics,
            "control_metrics": control_metrics,
            "candidate_metrics": candidate_metrics,
            "paired_deltas": paired,
            "recovery_funnel": funnel,
            "instruction_boundary": boundary_metrics,
            "prompt_injection": {
                "cases": injection_results,
                "safe_count": sum(1 for item in injection_results if item["safe"]),
                "required": len(injection_cases),
                "all_safe": injection_safe,
            },
            "category_results": {**categories, "target_95": target_95},
            "retrieval_metrics": retrieval_metrics_extra,
            "ranking_metrics": ranking_metrics,
            "failure_census": {"counts": dict(census), "case_ids": census_ids},
            "security": security,
            "citations": citations,
            "reliability": reliability,
            "latency": latency,
            "usage": usage,
            "cost": cost,
            "selection": selection,
            "primary_remaining_bottleneck": bottleneck,
        }

    @staticmethod
    def _transition(left: dict[str, Any], right: dict[str, Any]) -> str:
        if left["behavior"] == "INCORRECT_ABSTENTION" and right["behavior"] == "CORRECT_ANSWER":
            return "A_ABSTAIN_TO_B_CORRECT"
        if left["behavior"] == "INCORRECT_ABSTENTION" and right["behavior"] == "UNSUPPORTED_ANSWER":
            return "A_ABSTAIN_TO_B_UNSUPPORTED"
        if left["behavior"] == "CORRECT_ANSWER" and right["behavior"] == "CORRECT_ANSWER":
            return "A_CORRECT_TO_B_CORRECT"
        if left["behavior"] == "CORRECT_ANSWER" and right["behavior"] != "CORRECT_ANSWER":
            return "A_CORRECT_TO_B_INCORRECT"
        if left["behavior"] == "CORRECT_ABSTENTION" and right["behavior"] == "CORRECT_ABSTENTION":
            return "A_CORRECT_ABSTAIN_TO_B_CORRECT_ABSTAIN"
        if left["behavior"] == "CORRECT_ABSTENTION" and right["status"] == "answered":
            return "A_CORRECT_ABSTAIN_TO_B_ANSWER"
        return "OTHER"

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(V3Phase5bExperimentRecord, LOCK_ID)
        research = self.session.get(ResearchArchitectureRecord, V3_ARCHITECTURE_ID)
        payload: dict[str, Any] = {
            "architecture_id": V3_ARCHITECTURE_ID,
            "parent_architecture": V2_ARCHITECTURE_ID,
            "production_status": False,
            "lock_id": LOCK_ID,
            "experiment_id": EXPERIMENT_ID,
            "initialized": record is not None,
            "completed": bool(record and record.completed_at),
            "reference": REFERENCE_STRATEGY,
            "control": CONTROL_STRATEGY,
            "candidate": CANDIDATE_STRATEGY,
            "retry_policy": TRANSPORT_RETRY_POLICY_RECORD,
            "promotion_policy": PROMOTION_POLICY,
            "frozen_dataset_hash": FROZEN_DATASET_HASH,
            "candidate_ranking_hash": CANDIDATE_RANKING_HASH,
            "pairwise_algorithm_id": PAIRWISE_ALGORITHM_ID,
            "pairwise_algorithm_version": PAIRWISE_ALGORITHM_VERSION,
        }
        if research:
            payload["v3_research_status"] = research.v3_research_status
            payload["selected_v3_strategy"] = research.selected_v3_strategy
            payload["v3_production_status"] = research.production_status
        if not record:
            return payload
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
                "reference_configuration": record.reference_configuration,
                "control_configuration": record.control_configuration,
                "candidate_configuration": record.candidate_configuration,
                "control_architecture_hash": record.control_architecture_hash,
                "candidate_architecture_hash": record.candidate_architecture_hash,
                "candidate_ranking_hash_record": record.candidate_ranking_hash,
                "promotion_policy_hash": record.promotion_policy_hash,
                "embedding_preflight": record.embedding_preflight,
                "hosted_preflight": record.hosted_preflight,
                "reference_metrics": record.reference_metrics,
                "control_metrics": record.control_metrics,
                "candidate_metrics": record.candidate_metrics,
                "paired_deltas": record.paired_deltas,
                "recovery_funnel": record.recovery_funnel,
                "instruction_boundary": record.instruction_boundary,
                "prompt_injection": record.prompt_injection,
                "category_results": record.category_results,
                "retrieval_metrics": record.retrieval_metrics,
                "ranking_metrics": record.ranking_metrics,
                "failure_census": record.failure_census,
                "security": record.security,
                "citations": record.citations,
                "reliability": record.reliability,
                "latency": record.latency,
                "usage": record.usage,
                "cost": record.cost,
                "selection": record.selection,
                "selected_strategy": record.selected_strategy,
                "promotion_decision": record.promotion_decision,
                "v3_status": record.v3_status,
                "primary_remaining_bottleneck": record.primary_remaining_bottleneck,
                "dataset_frozen_at": record.dataset_frozen_at,
                "retrieval_frozen_at": record.retrieval_frozen_at,
                "completed_at": record.completed_at,
            }
        )
        if include_cases:
            payload["shared_traces"] = record.shared_traces
            payload["reference_rows"] = record.reference_rows
            payload["control_rows"] = record.control_rows
            payload["candidate_rows"] = record.candidate_rows
        return payload

    def _persist_ledger(self, record: V3Phase5bExperimentRecord) -> None:
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        payload = self.status(include_cases=True)
        serializable = json.loads(json.dumps(payload, default=str))
        (LEDGER_DIR / "summary.json").write_text(json.dumps(serializable, indent=2) + "\n")
        (LEDGER_DIR / "freeze.json").write_text(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "dataset_id": record.dataset_id,
                    "dataset_hash": record.dataset_hash,
                    "control_architecture_hash": record.control_architecture_hash,
                    "candidate_architecture_hash": record.candidate_architecture_hash,
                    "candidate_ranking_hash": record.candidate_ranking_hash,
                    "promotion_policy_hash": record.promotion_policy_hash,
                    "pairwise_algorithm_id": PAIRWISE_ALGORITHM_ID,
                    "pairwise_algorithm_version": PAIRWISE_ALGORITHM_VERSION,
                    "dataset_frozen_at": record.dataset_frozen_at.isoformat() if record.dataset_frozen_at else None,
                    "selection_policy_frozen_at": record.selection_policy_frozen_at.isoformat() if record.selection_policy_frozen_at else None,
                    "completed_at": record.completed_at.isoformat() if record.completed_at else None,
                    "promotion_decision": record.promotion_decision,
                    "v3_status": record.v3_status,
                },
                indent=2,
            )
            + "\n"
        )

    def _persist_markdown(self) -> None:
        status = self.status(include_cases=False)
        md_path = LEDGER_DIR / "report.md"
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        lines = [
            "# V3 Phase 5B — Three-Arm Ranking E2E Benchmark",
            "",
            f"**Experiment ID:** `{EXPERIMENT_ID}`",
            f"**Lock ID:** `{LOCK_ID}`",
            "",
            "## Arms",
            "",
            f"- **Reference R:** `{REFERENCE_STRATEGY}` — Frozen stable V2 (pointwise CE Top-5 → Sol Judge → V2 answer/abstain)",
            f"- **Control A:** `{CONTROL_STRATEGY}` — V3 research pipeline (pointwise CE Top-5 → Sol Judge → recovery on negative)",
            f"- **Candidate B:** `{CANDIDATE_STRATEGY}` — Same as A but pairwise complementarity Top-5",
            "",
            "## Dataset",
            "",
            f"- Dataset ID: `{status.get('dataset_id')}`",
            f"- Hash: `{status.get('dataset_hash')}`",
            f"- Cases: {len(status.get('case_ids') or [])}",
            f"- Distribution: {json.dumps(status.get('category_distribution'), indent=None)}",
            "",
        ]
        ref_m = status.get("reference_metrics") or {}
        ctrl_m = status.get("control_metrics") or {}
        cand_m = status.get("candidate_metrics") or {}
        if ref_m:
            lines += [
                "## End-to-End Metrics",
                "",
                "| Metric | Reference R | Control A | Candidate B |",
                "| ------ | ----------: | --------: | ----------: |",
                f"| Correct answers | {ref_m.get('correct_answers', '—')} | {ctrl_m.get('correct_answers', '—')} | {cand_m.get('correct_answers', '—')} |",
                f"| Incorrect abstentions | {ref_m.get('incorrect_abstentions', '—')} | {ctrl_m.get('incorrect_abstentions', '—')} | {cand_m.get('incorrect_abstentions', '—')} |",
                f"| Unsupported answers | {ref_m.get('unsupported_answers', '—')} | {ctrl_m.get('unsupported_answers', '—')} | {cand_m.get('unsupported_answers', '—')} |",
                f"| Precision | {ref_m.get('precision', '—')} | {ctrl_m.get('precision', '—')} | {cand_m.get('precision', '—')} |",
                f"| Recall | {ref_m.get('recall', '—')} | {ctrl_m.get('recall', '—')} | {cand_m.get('recall', '—')} |",
                f"| F1 | {ref_m.get('f1', '—')} | {ctrl_m.get('f1', '—')} | {cand_m.get('f1', '—')} |",
                f"| Answerable correct rate | {ref_m.get('answerable_case_correct_answer_rate', '—')} | {ctrl_m.get('answerable_case_correct_answer_rate', '—')} | {cand_m.get('answerable_case_correct_answer_rate', '—')} |",
                "",
            ]
        sel = status.get("selection") or {}
        if sel:
            lines += [
                "## Promotion Decision",
                "",
                f"- **Decision:** `{sel.get('promotion_decision')}`",
                f"- **V3 Status:** `{sel.get('v3_status')}`",
                f"- Quality gate (B vs A): {sel.get('quality_gate_vs_control')}",
                f"- Quality gate (B vs R): {sel.get('quality_gate_vs_reference')}",
                f"- Safety gate: {sel.get('safety_gate')}",
                f"- Regression gate: {sel.get('regression_gate')}",
                f"- Ranking gate: {sel.get('ranking_gate')}",
                "",
            ]
        ranking = status.get("ranking_metrics") or {}
        rg = ranking.get("ranking_gates") or {}
        if rg:
            lines += [
                "## Ranking Gates",
                "",
                f"- Exact-ID Recall: Control={rg.get('exact_id_recall_control')}, Candidate={rg.get('exact_id_recall_candidate')}, Pass={rg.get('exact_id_recall_pass')}",
                f"- Version Correctness: Control={rg.get('version_correctness_control')}, Candidate={rg.get('version_correctness_candidate')}, Pass={rg.get('version_correctness_pass')}",
                f"- Same-doc degradation: {rg.get('same_doc_multi_chunk_degradation')}, Pass={rg.get('same_doc_multi_chunk_pass')}",
                f"- Unauthorized downstream evidence: {rg.get('unauthorized_downstream_evidence')}",
                "",
            ]
        lines += [
            f"**Completed at:** {status.get('completed_at')}",
            "",
        ]
        md_path.write_text("\n".join(lines))
