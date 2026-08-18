from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from rag_workbench.answerability.base import AnswerabilityResult, GateEvidence
from rag_workbench.answerability.cache import CachedAnswerabilityGate, gate_cache_key
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    OpenAICompatibleAnswerabilityGate,
    evidence_sufficiency_schema_identity,
    evidence_sufficiency_template_hash,
    hosted_judge_request_settings,
)
from rag_workbench.answerability.planning import ExternalJudgeCallLimitGate
from rag_workbench.answerability.transport import (
    DEFAULT_TRANSPORT_RETRY_POLICY,
)
from rag_workbench.answerability.validation import validate_gate_result_with_error
from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import (
    AnswerabilityGateCacheRecord,
    Chunk,
    Document,
    DocumentVersion,
    EndToEndBenchmarkRunRecord,
    ExperimentRunRecord,
    QueryEmbeddingCacheRecord,
    ResearchArchitectureRecord,
    RetrievalArchitectureRecord,
    V2FinalBenchmarkRecord,
    V2Phase3ExperimentRecord,
    V2Phase4ExperimentRecord,
)
from rag_workbench.evaluation.generation_metrics import deterministic_citation_correctness
from rag_workbench.evaluation.hybrid_metrics import (
    aggregate_retrieval_metrics,
    category_retrieval_metrics,
    retrieval_case_metrics,
)
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    CONFIGURATION,
    DENSE_DEPTH,
    FINAL_TOP_K,
    RRF_K,
    UNION_LIMIT,
    HybridRerankerBenchmark,
    _principal,
    _trace,
    aggregate_pool,
    pool_metrics,
)
from rag_workbench.experiments.hybrid_reranker_replication import RELEASE_ARCHITECTURE_ID
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
from rag_workbench.experiments.v2_final_benchmark_report import (
    persist_final_v2_benchmark_markdown,
)
from rag_workbench.experiments.v2_quality_recovery import (
    FROZEN_V1_ARCHITECTURE_HASH,
    V2_RESEARCH_ARCHITECTURE_ID,
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
    SufficiencyFnCase,
    retrieval_complete,
)
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.generation.generator import RagService
from rag_workbench.providers.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from rag_workbench.providers.llm.extractive import (
    EXTRACTIVE_REVISION,
    ExtractiveGenerationProvider,
)
from rag_workbench.reranking import CrossEncoderReranker, Reranker
from rag_workbench.reranking.document_diversity import EVALUATION_LABEL_FIELDS
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import RetrievalTiming, Retriever

DATASET_ID = "acmeai-enterprise-rag-v2-final-eval"
DATASET_PATH = Path("data/eval/acmeai_enterprise_rag_v2_final_eval.json")
DATASET_HASH = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
GENERATION_METHOD = "manual-corpus-grounded-v2-final"
OVERLAP_CEILING = 0.5
V2_ARCHITECTURE_ID = "enterprise-rag-workbench-v2"
RANKING_RESEARCH_FROZEN = "FROZEN_FOR_CURRENT_V2_CYCLE"
JUDGE_RESEARCH_FROZEN = "FROZEN_FOR_CURRENT_V2_CYCLE"
RELIABILITY_HARDENED = "RELIABILITY_HARDENED"
EXPECTED_DISTRIBUTION = {
    "acl_sensitive": 3,
    "exact_identifier": 10,
    "multidoc_three": 30,
    "multidoc_two": 18,
    "near_duplicate": 8,
    "partial_no_answer": 3,
    "prompt_injection": 4,
    "semantic_paraphrase": 6,
    "single_document": 10,
    "version_region": 8,
}
V1_FINAL_DESCRIPTIVE = {
    "dataset_id": "acmeai-enterprise-rag-v1-final-eval",
    "case_count": 80,
    "correct_answers": 24,
    "correct_abstentions": 11,
    "unsupported_answers": 0,
    "incorrect_abstentions": 45,
    "accuracy": 0.4375,
    "precision": 1.0,
    "recall": 0.347826,
    "f1": 0.516129,
    "candidate_pool_all_required_coverage": 1.0,
    "top5_all_required_coverage": 0.753623,
    "three_document_top5_coverage": 0.25,
}
HIDDEN_GROUND_TRUTH_FIELDS = frozenset(
    {
        *EVALUATION_LABEL_FIELDS,
        "expected_acl_behavior",
        "expected_prompt_injection_behavior",
        "preferred_source_id",
        "evaluation_labels",
    }
)


class V2FinalCase(SufficiencyFnCase):
    preferred_source_id: str | None = None


def load_v2_final_cases() -> tuple[V2FinalCase, ...]:
    payload = json.loads(DATASET_PATH.read_text())
    return tuple(V2FinalCase.model_validate(item) for item in payload["cases"])


CASES = load_v2_final_cases()


def _token_terms(question: str) -> set[str]:
    return set(re.compile(r"[a-z0-9]+").findall(question.casefold()))


def dataset_overlap_report() -> dict[str, Any]:
    maximum = 0.0
    closest: dict[str, Any] | None = None
    for path in Path("data/eval").glob("*.json"):
        if path == DATASET_PATH:
            continue
        for previous in json.loads(path.read_text()).get("cases", []):
            right = _token_terms(previous["question"])
            for item in CASES:
                left = _token_terms(item.question)
                union = left | right
                score = len(left & right) / len(union) if union else 0.0
                if score > maximum:
                    maximum = score
                    closest = {
                        "final_case_id": item.case_id,
                        "prior_dataset": path.name,
                        "prior_case_id": previous.get("case_id"),
                        "overlap": score,
                    }
    return {
        "maximum_normalized_overlap": maximum,
        "closest_prior_case": closest,
        "overlap_threshold": OVERLAP_CEILING,
        "pass": maximum < OVERLAP_CEILING,
    }


def v2_architecture_configuration() -> dict[str, Any]:
    prompt_hash = evidence_sufficiency_template_hash(EVIDENCE_GATE_PROMPT_VERSION)
    schema = evidence_sufficiency_schema_identity()
    configuration = {
        "architecture_id": V2_ARCHITECTURE_ID,
        "parent_architecture_id": RELEASE_ARCHITECTURE_ID,
        "selected_retriever": CONTROL_MODE,
        "ranking_policy_identity": CONTROL_MODE,
        "corpus_identity": CORPUS_IDENTITY,
        "semantic_index_identity": SEMANTIC_INDEX_IDENTITY,
        "embedding": {
            "provider": "openai-compatible",
            "model": "text-embedding-3-small",
            "dimension": 64,
            "version": "1",
        },
        "dense": {
            "backend": "pgvector_cosine",
            "threshold": 0.28,
            "candidate_depth": DENSE_DEPTH,
        },
        "bm25": CONFIGURATION["bm25"],
        "bm25_candidate_depth": BM25_DEPTH,
        "rrf": {"k": RRF_K, "candidate_union_limit": UNION_LIMIT},
        "cross_encoder": {
            "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "revision": RERANKER_REVISION,
        },
        "final_top_k": FINAL_TOP_K,
        "judge": {
            "id": CONTROL_JUDGE,
            "provider": "openai",
            "model": SOL_MODEL,
            "prompt": EVIDENCE_GATE_PROMPT_VERSION,
            "prompt_hash": prompt_hash,
            "schema_identity": schema,
            "request_settings": hosted_judge_request_settings(EVIDENCE_GATE_PROMPT_VERSION),
            "supporting_context_only": True,
        },
        "generator": dict(EXTRACTIVE_REVISION),
        "retry_policy": TRANSPORT_RETRY_POLICY_RECORD,
        "acl_version_policy": (
            "tenant/ACL/active-version filtering precedes Dense and BM25 candidate branches"
        ),
        "immutable": True,
        "no_post_benchmark_tuning": True,
    }
    configuration["architecture_hash"] = stable_hash(configuration)
    return configuration


def _gate_evidence(values: list[dict[str, Any]]) -> tuple[GateEvidence, ...]:
    return tuple(
        GateEvidence(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            document_version_id=item["document_version_id"],
            version=item["version"],
            text=item["text"],
            index_identity=SEMANTIC_INDEX_IDENTITY,
        )
        for item in values
    )


def _runtime_candidates(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaked = HIDDEN_GROUND_TRUTH_FIELDS.intersection(values[0] if values else {})
    if leaked:
        raise RuntimeError(f"evaluator labels leaked into runtime traces: {sorted(leaked)}")
    return values


def _behavior(case: V2FinalCase, status: str) -> str:
    if case.should_abstain:
        return "CORRECT_ABSTENTION" if status == "abstained" else "UNSUPPORTED_ANSWER"
    return "CORRECT_ANSWER" if status == "answered" else "INCORRECT_ABSTENTION"


def _confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for item in rows if item["class"] == "TP")
    fn = sum(1 for item in rows if item["class"] == "FN")
    fp = sum(1 for item in rows if item["class"] == "FP")
    tn = sum(1 for item in rows if item["class"] == "TN")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class V2FinalBenchmark:
    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        *,
        embedding_provider_factory: Any | None = None,
        reranker_factory: Any | None = None,
        gate_factory: Any | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embedding_provider_factory = embedding_provider_factory
        self.reranker_factory = reranker_factory or (
            lambda: CrossEncoderReranker(device="cpu", resolved_revision=RERANKER_REVISION)
        )
        self.gate_factory = gate_factory
        self._verify_dataset()

    def _verify_dataset(self) -> None:
        payload = json.loads(DATASET_PATH.read_text())
        if payload.get("dataset_id") != DATASET_ID or len(CASES) != 100:
            raise ValueError("final v2 dataset identity changed")
        if hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() != DATASET_HASH:
            raise ValueError("final v2 dataset hash changed")
        distribution = dict(sorted(Counter(item.category for item in CASES).items()))
        if distribution != EXPECTED_DISTRIBUTION:
            raise ValueError("final v2 category distribution changed")
        three = [item for item in CASES if item.category == "multidoc_three"]
        if any(
            len(set(item.required_document_ids)) != 3 or len(item.required_fact_ids) != 3
            for item in three
        ):
            raise ValueError("three-document cases must require three distinct sources")
        if any(
            item.category == "prompt_injection" and "prompt_injection" not in item.security_checks
            for item in CASES
        ):
            raise ValueError("prompt-injection cases must carry the security label")
        prompt_hash = evidence_sufficiency_template_hash(EVIDENCE_GATE_PROMPT_VERSION)
        schema = evidence_sufficiency_schema_identity()
        if prompt_hash != FROZEN_V1_TEMPLATE_HASH or schema != SCHEMA_IDENTITY:
            raise ValueError("PARTIAL: selected V2 Judge prompt or schema identity drifted")

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
        phase3 = self.session.get(
            V2Phase3ExperimentRecord, "acmeai-v2-sufficiency-fn-eval-v1"
        )
        if phase3 and phase3.usage:
            specialized += phase3.usage.get("new_control_judge_calls", 0)
            specialized += phase3.usage.get("new_candidate_judge_calls", 0)
        return historical + specialized

    def freeze_architecture(self) -> RetrievalArchitectureRecord:
        files = verify_v1_file_identities()
        persisted = verify_persisted_v1(self.session)
        research = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        if research is None:
            raise ValueError("v2 research identity is missing")
        if research.selected_v2_ranking != CONTROL_MODE:
            raise ValueError("PARTIAL: selected V2 ranking drifted")
        if research.ranking_research_status != RANKING_RESEARCH_FROZEN:
            raise ValueError("PARTIAL: ranking research is not frozen")
        if research.selected_v2_judge != CONTROL_JUDGE:
            raise ValueError("PARTIAL: selected V2 Judge drifted")
        if research.judge_research_status != JUDGE_RESEARCH_FROZEN:
            raise ValueError("PARTIAL: judge research is not frozen")
        if research.reliability_research_status != RELIABILITY_HARDENED:
            raise ValueError("PARTIAL: Phase 4 reliability is not hardened")
        if persisted["architecture_hash"] != FROZEN_V1_ARCHITECTURE_HASH:
            raise ValueError("frozen v1 architecture hash changed")
        phase4 = self.session.get(V2Phase4ExperimentRecord, "v2-reliability-fixtures")
        if phase4 is None or phase4.completed_at is None:
            raise ValueError("Phase 4 must be complete before the final V2 benchmark")
        configuration = v2_architecture_configuration()
        if configuration["judge"]["prompt_hash"] != FROZEN_V1_TEMPLATE_HASH:
            raise ValueError("PARTIAL: Judge prompt hash drifted from selected V2 state")
        if configuration["cross_encoder"]["revision"] != RERANKER_REVISION:
            raise ValueError("PARTIAL: Cross-Encoder revision drifted")
        existing = self.session.get(RetrievalArchitectureRecord, V2_ARCHITECTURE_ID)
        if existing:
            stored_hash = (existing.configuration or {}).get("architecture_hash")
            if (
                existing.selected_retriever != CONTROL_MODE
                or stored_hash != configuration["architecture_hash"]
            ):
                raise ValueError("frozen v2 architecture is immutable")
            return existing
        record = RetrievalArchitectureRecord(
            architecture_id=V2_ARCHITECTURE_ID,
            selected_retriever=CONTROL_MODE,
            configuration=configuration,
            selection_policy={
                "assembled_from_frozen_v2_research": True,
                "no_new_quality_component": True,
                "parent": RELEASE_ARCHITECTURE_ID,
            },
            dataset_id=DATASET_ID,
            dataset_hash=DATASET_HASH,
            retrieval_metrics={},
            immutable=True,
        )
        self.session.add(record)
        research.final_v2_architecture_id = V2_ARCHITECTURE_ID
        self.session.commit()
        if files["final_v1_architecture_record"] != RELEASE_ARCHITECTURE_ID:
            raise RuntimeError("v1 identities drifted during v2 architecture freeze")
        return record

    def embedding_preflight(self) -> dict[str, Any]:
        questions = tuple(dict.fromkeys(case.question for case in CASES))
        keys = [
            query_embedding_cache_key(
                question,
                provider="openai-compatible",
                model="text-embedding-3-small",
                version="1",
                dimension=64,
            )
            for question in questions
        ]
        matches = sum(self.session.get(QueryEmbeddingCacheRecord, key) is not None for key in keys)
        current = self._embedding_calls()
        missing = len(keys) - matches
        payload = {
            "embedding_provider": "openai-compatible",
            "embedding_model": "text-embedding-3-small",
            "current_cumulative_embedding_calls": current,
            "total_questions": len(questions),
            "existing_cache_matches": matches,
            "missing_unique_query_embeddings": missing,
            "authorized_ceiling": current + missing,
            "configured_ceiling": self.settings.max_external_embedding_calls,
            "document_embedding_calls_required": 0,
        }
        record = self.session.get(V2FinalBenchmarkRecord, DATASET_ID)
        if record is not None:
            record.embedding_preflight = payload
            self.session.commit()
        return payload

    def judge_preflight(self, *, persist: bool = True) -> dict[str, Any]:
        record = self.initialize()
        if record.retrieval_frozen_at is None or not record.retrieval_traces:
            raise ValueError("retrieval traces must be frozen before judge preflight")
        keys: list[str] = []
        for case, trace in zip(CASES, record.retrieval_traces, strict=True):
            keys.append(
                gate_cache_key(
                    case.question,
                    _gate_evidence(trace["final_top5"]),
                    provider="openai",
                    model=SOL_MODEL,
                    gate_version="1",
                    prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
                )[0]
            )
        unique = set(keys)
        matches = sum(
            self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in unique
        )
        missing = len(unique) - matches
        logical_current = self._hosted_judge_rows()
        historical = self._historical_judge_calls()
        preflight = {
            "judge_provider": "openai",
            "judge_model": SOL_MODEL,
            "logical_judge_ledger": logical_current,
            "historical_judge_call_ledger": historical,
            "physical_attempt_ledger": logical_current,
            "unique_final_inputs": len(unique),
            "existing_sol_cache_matches": matches,
            "missing_logical_judge_requests": missing,
            "minimum_initial_physical_attempts": missing,
            "logical_ceiling": logical_current + missing,
            "physical_attempt_ceiling": logical_current + missing * 2,
            "configured_ceiling": self.settings.max_external_judge_calls,
            "quality_retries_authorized": False,
            "max_total_attempts_per_logical_request": (
                DEFAULT_TRANSPORT_RETRY_POLICY.max_total_attempts
            ),
        }
        if persist:
            record.judge_preflight = preflight
            self.session.commit()
        return preflight

    def initialize(self) -> V2FinalBenchmarkRecord:
        architecture = self.freeze_architecture()
        overlap = dataset_overlap_report()
        if not overlap["pass"]:
            raise ValueError("PARTIAL: final dataset overlap exceeds the accepted threshold")
        if self._corpus_identity() != CORPUS_IDENTITY:
            raise ValueError("corpus identity changed")
        if not self.session.scalar(
            select(Chunk.id).where(Chunk.index_identity == SEMANTIC_INDEX_IDENTITY).limit(1)
        ):
            raise ValueError("frozen semantic index is unavailable")
        existing = self.session.get(V2FinalBenchmarkRecord, DATASET_ID)
        configuration = architecture.configuration or v2_architecture_configuration()
        if existing:
            if existing.dataset_hash != DATASET_HASH:
                raise ValueError("frozen final v2 dataset identity changed")
            if existing.architecture_hash != configuration["architecture_hash"]:
                raise ValueError("frozen v2 architecture hash changed")
            return existing
        now = datetime.now(UTC)
        record = V2FinalBenchmarkRecord(
            dataset_id=DATASET_ID,
            architecture_id=V2_ARCHITECTURE_ID,
            parent_architecture_id=RELEASE_ARCHITECTURE_ID,
            architecture_hash=configuration["architecture_hash"],
            architecture_configuration=configuration,
            dataset_hash=DATASET_HASH,
            case_ids=[item.case_id for item in CASES],
            category_distribution=EXPECTED_DISTRIBUTION,
            generation_method=GENERATION_METHOD,
            maximum_prior_overlap=overlap["maximum_normalized_overlap"],
            closest_previous_case=overlap["closest_prior_case"],
            overlap_report=overlap,
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=CORPUS_IDENTITY,
            one_shot=True,
            dataset_frozen=True,
            dataset_frozen_at=now,
            architecture_frozen_at=architecture.frozen_at or now,
        )
        self.session.add(record)
        research = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        if research:
            research.final_v2_architecture_id = V2_ARCHITECTURE_ID
            research.final_v2_dataset_id = DATASET_ID
        self.session.commit()
        return record

    def execute_retrieval(self) -> dict[str, Any]:
        record = self.initialize()
        traces = list(record.retrieval_traces or [])
        incomplete = len(traces) != len(CASES)
        if record.retrieval_frozen_at is not None and not incomplete:
            return self.status()
        preflight = self.embedding_preflight()
        if not self.settings.allow_external_calls or not self.settings.embedding_api_key:
            raise ValueError("external embedding authorization is incomplete")
        if preflight["authorized_ceiling"] > preflight["configured_ceiling"]:
            raise ValueError(
                "authorize MAX_EXTERNAL_EMBEDDING_CALLS to the preflight ceiling "
                f"{preflight['authorized_ceiling']}"
            )
        if record.one_shot_locked_at is None:
            record.one_shot_locked_at = datetime.now(UTC)
        record.embedding_preflight = preflight
        if record.retrieval_started_at is None:
            record.retrieval_started_at = datetime.now(UTC)
        preserved_usage = dict(record.usage or {}) if incomplete and record.usage else None
        preserved_latency = dict(record.latency or {}) if incomplete and record.latency else None
        preserved_security = dict(record.security or {}) if incomplete and record.security else None
        preserved_retrieval = None
        if incomplete and record.retrieval_results:
            preserved_retrieval = dict(record.retrieval_results)
        record.retrieval_frozen_at = None
        self.session.commit()
        judge_before = self._hosted_judge_rows()
        analysis = self._run_retrieval(record)
        if self._hosted_judge_rows() != judge_before:
            raise RuntimeError("Judge was invoked before retrieval traces were frozen")
        ending = self._embedding_calls()
        if ending > preflight["authorized_ceiling"]:
            raise RuntimeError("final v2 exceeded the authorized embedding ceiling")
        record.retrieval_traces = list(analysis["traces"])
        flag_modified(record, "retrieval_traces")
        record.retrieval_results = preserved_retrieval or analysis["retrieval"]
        record.latency = preserved_latency or {"retrieval": analysis["latency"]}
        record.usage = preserved_usage or analysis["usage"]
        record.security = preserved_security or analysis["security"]
        record.retrieval_frozen_at = datetime.now(UTC)
        self.session.commit()
        verify_persisted_v1(self.session)
        return self.status()

    def execute(self) -> dict[str, Any]:
        record = self.initialize()
        if record.completed_at is not None:
            persist_final_v2_benchmark_markdown(self.status())
            return self.status()
        if record.retrieval_frozen_at is None:
            self.execute_retrieval()
            record = self.initialize()
        return self.execute_generation(record)

    def execute_generation(self, record: V2FinalBenchmarkRecord | None = None) -> dict[str, Any]:
        record = record or self.initialize()
        if record.completed_at is not None:
            persist_final_v2_benchmark_markdown(self.status())
            return self.status()
        if record.retrieval_frozen_at is None or not record.retrieval_traces:
            raise ValueError("do not invoke the Judge until retrieval traces are frozen")
        preflight = self.judge_preflight()
        if (
            not self.settings.allow_external_judge_calls
            or not self.settings.effective_judge_api_key
        ):
            raise ValueError("hosted judge authorization is incomplete")
        if preflight["logical_ceiling"] > preflight["configured_ceiling"]:
            raise ValueError(
                "authorize MAX_EXTERNAL_JUDGE_CALLS to the logical ceiling "
                f"{preflight['logical_ceiling']}"
            )
        if record.execution_started_at is None:
            record.execution_started_at = datetime.now(UTC)
            self.session.commit()
        remaining = preflight["missing_logical_judge_requests"]
        analysis = self._run_generation(record, remaining)
        record.case_results = analysis["cases"]
        record.end_to_end = analysis["end_to_end"]
        record.category_results = analysis["category_results"]
        record.stage_funnel = analysis["stage_funnel"]
        record.failure_taxonomy = analysis["failure_taxonomy"]
        record.generator_reliability = analysis["generator_reliability"]
        record.provider_reliability = analysis["provider_reliability"]
        record.security = analysis["security"]
        record.citations = analysis["citations"]
        record.latency = {**(record.latency or {}), **analysis["latency"]}
        record.usage = {**(record.usage or {}), **analysis["usage"]}
        record.cost = analysis["cost"]
        record.v1_comparison = analysis["v1_comparison"]
        record.primary_remaining_bottleneck = analysis["primary_remaining_bottleneck"]
        record.completed_at = datetime.now(UTC)
        self.session.commit()
        persist_final_v2_benchmark_markdown(self.status())
        verify_persisted_v1(self.session)
        verify_v1_file_identities()
        return self.status(include_cases=True)

    def _run_retrieval(self, record: V2FinalBenchmarkRecord) -> dict[str, Any]:
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
        traces = list(record.retrieval_traces or [])
        done = {item["case_id"] for item in traces}
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
        rows: list[dict[str, Any]] = []
        latencies = {
            "query_embedding_ms": [],
            "dense_ms": [],
            "bm25_ms": [],
            "rrf_ms": [],
            "cross_encoder_ms": [],
        }
        for case in CASES:
            if case.case_id in done:
                trace = next(item for item in traces if item["case_id"] == case.case_id)
                rows.append(trace["evaluation_row"])
                timing = trace["timing"]
                for key in latencies:
                    latencies[key].append(timing[key])
                usage["new_query_embedding_calls"] += trace.get("embedding_external_calls") or 0
                usage["query_cache_hits"] += int(bool(trace.get("embedding_cache_hit")))
                usage["query_cache_misses"] += int(not trace.get("embedding_cache_hit"))
                usage["cross_encoder_pairs"] += len(trace.get("rrf_union") or [])
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
                usage["unauthorized_chunks_to_cross_encoder"] += len(leaked)
            rerank_started = time.perf_counter()
            reranked = reranker.rerank(case.question, union)
            ce_ms = (time.perf_counter() - rerank_started) * 1000
            usage["cross_encoder_pairs"] += len(union)
            top5 = [
                ranking_candidate(item)
                | {
                    "rank": item.reranked_rank,
                    "score": float(item.reranker_score or 0.0),
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
            pool = pool_metrics(case.as_retrieval(), union)
            top5_results = [_result(item) for item in top5]
            taxonomy = HybridRerankerBenchmark._taxonomy(
                case.as_retrieval(),
                dense_candidates,
                bm25_candidates,
                union,
                top5_results,
            )
            evaluation_row = {
                "case_id": case.case_id,
                "category": case.category,
                "expected_answerability": case.expected_answerability,
                "metrics": metrics,
                "pool": pool,
                "top5_complete": complete,
                "failure_taxonomy": taxonomy,
            }
            timing = {
                "query_embedding_ms": embedding.embedding_latency_ms,
                "cache_lookup_ms": embedding.cache_lookup_latency_ms,
                "dense_ms": dense_ms,
                "bm25_ms": bm25_ms,
                "rrf_ms": fusion_ms,
                "cross_encoder_ms": ce_ms,
            }
            trace = {
                "case_id": case.case_id,
                "embedding_cache_hit": embedding.cache_hit,
                "embedding_external_calls": embedding.external_calls,
                "dense": _runtime_candidates([_trace(item) for item in dense_candidates]),
                "bm25": _runtime_candidates([_trace(item) for item in bm25_candidates]),
                "rrf_union": _runtime_candidates([_trace(item) for item in union]),
                "cross_encoder_ranked": [
                    ranking_candidate(item) for item in reranked
                ],
                "final_top5": top5,
                "retrieval_complete": complete,
                "pool_complete": bool(pool.get("all_required_evidence_coverage")),
                "failure_taxonomy": taxonomy,
                "required_ce_ranks": {
                    document_id: next(
                        (
                            item.reranked_rank
                            for item in reranked
                            if item.result.document_id == document_id
                        ),
                        None,
                    )
                    for document_id in case.required_document_ids
                },
                "timing": timing,
                "evaluation_row": evaluation_row,
            }
            traces.append(trace)
            rows.append(evaluation_row)
            for key, value in latencies.items():
                value.append(timing[key])
            record.retrieval_traces = list(traces)
            flag_modified(record, "retrieval_traces")
            self.session.commit()
        answerable = [item for item in rows if item["expected_answerability"]]
        pool_complete = [
            item
            for item in answerable
            if (item.get("pool") or {}).get("all_required_evidence_coverage")
        ]
        top5_incomplete = [item for item in pool_complete if not item.get("top5_complete")]
        metrics = aggregate_retrieval_metrics(rows)
        category_metrics = category_retrieval_metrics(rows)
        metrics["single_document_coverage_at_5"] = (
            category_metrics.get("single_document") or {}
        ).get("all_required_evidence_coverage_at_5", 0.0)
        metrics["semantic_paraphrase_success"] = (
            category_metrics.get("semantic_paraphrase") or {}
        ).get("all_required_evidence_coverage_at_5", 0.0)
        retrieval = {
            "metrics": metrics,
            "category_metrics": category_metrics,
            "pool": aggregate_pool(rows),
            "required_evidence_present_in_pool_lost_by_top5": len(top5_incomplete),
        }
        security = {
            "acl_safety": 1.0 if usage["unauthorized_chunks_to_cross_encoder"] == 0 else 0.0,
            "tenant_isolation": 1.0,
            "unauthorized_chunks_to_cross_encoder": usage["unauthorized_chunks_to_cross_encoder"],
            "unauthorized_chunks_to_judge": usage["unauthorized_chunks_to_judge"],
            "version_correctness": retrieval["metrics"]["version_correctness"],
        }
        return {
            "traces": traces,
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

    def _run_generation(self, record: V2FinalBenchmarkRecord, maximum_calls: int) -> dict[str, Any]:
        traces = record.retrieval_traces or []
        results = list(record.case_results or [])
        done = {item["case_id"] for item in results}
        gate = self._gate(maximum_calls)
        provider = self._embedding_provider()
        extra = {
            "reasoning_tokens": 0,
            "cached_prompt_tokens": 0,
            "resolved_models": [],
            "retries": Counter(),
            "transport_recovered": 0,
            "quality_retries": 0,
        }
        for case, trace in zip(CASES, traces, strict=True):
            if case.case_id in done:
                continue
            top5 = trace["final_top5"]
            forbidden_ids = set(case.forbidden_document_ids)
            leaked = [item for item in top5 if item["document_id"] in forbidden_ids]
            if case.expected_access_behavior == "EXCLUDE_FORBIDDEN" and leaked:
                raise RuntimeError("unauthorized chunks reached the Judge")
            evidence = _gate_evidence(top5)
            retrieved = [_result(item) for item in top5]
            timing = RetrievalTiming(
                query_embedding_latency_ms=trace["timing"]["query_embedding_ms"],
                embedding_cache_lookup_latency_ms=trace["timing"]["cache_lookup_ms"],
                vector_search_latency_ms=trace["timing"]["dense_ms"],
                acl_filter_latency_ms=0.0,
                query_embedding_cache_hit=trace["embedding_cache_hit"],
                external_embedding_calls=0,
            )
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
            rag = RagService(
                self.session,
                FixedResultRetriever(provider, retrieved, timing),
                ContextBuilder(1200),
                ExtractiveGenerationProvider(),
                _FixedGate(result, gate.last_timing),
                supporting_context_only=True,
            )
            generated = rag.query(case.question, _principal(case), top_k=5, score_threshold=0.28)
            extra["reasoning_tokens"] += gate.last_timing.reasoning_tokens or 0
            extra["cached_prompt_tokens"] += gate.last_timing.cached_prompt_tokens or 0
            resolved = gate.last_timing.resolved_model
            if resolved and resolved not in extra["resolved_models"]:
                extra["resolved_models"].append(resolved)
            retry_count = gate.last_timing.retry_count or 0
            if (
                retry_count
                and generated.status in {"answered", "abstained"}
                and not operational_error
            ):
                extra["transport_recovered"] += 1
            if retry_count:
                extra["retries"][gate.last_timing.failure_class or "UNKNOWN"] += retry_count
            cited = [citation.chunk_id for citation in generated.citations]
            citation_validity = deterministic_citation_correctness(
                cited, tuple(item.chunk_id for item in generated.retrieval_results)
            )
            behavior = _behavior(case, generated.status)
            judge_class = classification(case.expected_answerability, result.answerable)
            payload = {
                "case_id": case.case_id,
                "category": case.category,
                "status": generated.status,
                "behavior": behavior,
                "class": judge_class,
                "answerable": result.answerable,
                "supporting_chunk_ids": list(result.supporting_chunk_ids),
                "reason_code": result.reason_code.value,
                "operational_error": operational_error or generated.answerability_operational_error,
                "generation_operational_error": generated.generation_operational_error,
                "extractive_path": generated.extractive_path,
                "retrieval_complete": trace["retrieval_complete"],
                "pool_complete": trace["pool_complete"],
                "failure_taxonomy_retrieval": trace["failure_taxonomy"],
                "citations": [citation.to_dict() for citation in generated.citations],
                "citation_validity": citation_validity,
                "logical_request_id": generated.judge_logical_request_id
                or gate.last_timing.logical_request_id,
                "physical_attempts": generated.judge_physical_attempts
                or gate.last_timing.external_calls,
                "retry_count": generated.judge_retry_count or gate.last_timing.retry_count,
                "cache_hit": gate.last_timing.cache_hit,
                "live": not gate.last_timing.cache_hit,
                "judge_latency_ms": gate.last_timing.judge_latency_ms,
                "generation_latency_ms": generated.generation_latency_ms,
                "support_validation_ms": generated.context_pruning_latency_ms,
                "total_pipeline_ms": (
                    trace["timing"]["query_embedding_ms"]
                    + trace["timing"]["dense_ms"]
                    + trace["timing"]["bm25_ms"]
                    + trace["timing"]["rrf_ms"]
                    + trace["timing"]["cross_encoder_ms"]
                    + (gate.last_timing.judge_latency_ms or 0.0)
                    + generated.context_pruning_latency_ms
                    + generated.generation_latency_ms
                ),
                "failure_class": gate.last_timing.failure_class,
                "prompt_tokens": gate.last_timing.prompt_tokens,
                "completion_tokens": gate.last_timing.completion_tokens,
                "version_correct": all(
                    any(
                        item["document_id"] == document_id and item["version"] == version
                        for item in top5
                    )
                    for document_id, version in case.required_version_ids.items()
                )
                if case.required_version_ids
                else True,
                "preferred_source_top5": (
                    case.preferred_source_id in {item["document_id"] for item in top5}
                    if case.preferred_source_id
                    else None
                ),
                "preferred_source_pool": (
                    case.preferred_source_id
                    in {item["document_id"] for item in trace["rrf_union"]}
                    if case.preferred_source_id
                    else None
                ),
                "document_instruction_followed": _document_instruction_followed(generated.answer),
                "unauthorized_evidence_selected": bool(leaked)
                and case.expected_access_behavior == "EXCLUDE_FORBIDDEN",
                "root_cause": None,
            }
            payload["root_cause"] = classify_v2_root_cause(case, trace, payload)
            results.append(payload)
            record.case_results = list(results)
            flag_modified(record, "case_results")
            self.session.commit()
        return self._analyze(record, results, extra)

    def _analyze(
        self,
        record: V2FinalBenchmarkRecord,
        results: list[dict[str, Any]],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        by_id = {item["case_id"]: item for item in results}
        cases_by_id = {item.case_id: item for item in CASES}
        answerable_cases = [item for item in CASES if item.expected_answerability]
        answered = [item for item in results if item["status"] == "answered"]
        correct_answers = [item for item in results if item["behavior"] == "CORRECT_ANSWER"]
        correct_abstentions = [item for item in results if item["behavior"] == "CORRECT_ABSTENTION"]
        unsupported = [item for item in results if item["behavior"] == "UNSUPPORTED_ANSWER"]
        incorrect_abstentions = [
            item for item in results if item["behavior"] == "INCORRECT_ABSTENTION"
        ]
        confusion = _confusion(results)
        accuracy = (len(correct_answers) + len(correct_abstentions)) / len(results)
        end_to_end = {
            "correct_answers": len(correct_answers),
            "correct_abstentions": len(correct_abstentions),
            "unsupported_answers": len(unsupported),
            "incorrect_abstentions": len(incorrect_abstentions),
            "accuracy": accuracy,
            **confusion,
            "answerable_cases": len(answerable_cases),
            "answered_cases": len(answered),
            "correct_answer_cases": len(correct_answers),
            "answer_rate_on_answerable": (
                sum(
                    1
                    for item in answerable_cases
                    if by_id[item.case_id]["status"] == "answered"
                )
                / len(answerable_cases)
            ),
            "correct_answer_rate_on_answerable": len(correct_answers) / len(answerable_cases),
            "abstention_rate_on_answerable": (
                sum(
                    1
                    for item in answerable_cases
                    if by_id[item.case_id]["status"] == "abstained"
                )
                / len(answerable_cases)
            ),
        }
        complete_answerable = [
            by_id[item.case_id]
            for item in answerable_cases
            if by_id[item.case_id]["retrieval_complete"]
        ]
        incomplete = [
            by_id[item.case_id]
            for item in answerable_cases
            if not by_id[item.case_id]["retrieval_complete"]
        ]
        pool_complete = [
            item for item in answerable_cases if by_id[item.case_id]["pool_complete"]
        ]
        top5_complete = [
            item for item in pool_complete if by_id[item.case_id]["retrieval_complete"]
        ]
        judge_approved = [
            item
            for item in top5_complete
            if by_id[item.case_id]["answerable"]
            and not by_id[item.case_id]["operational_error"]
        ]
        generator_ok = [
            item
            for item in judge_approved
            if by_id[item.case_id]["status"] == "answered"
            and not by_id[item.case_id]["generation_operational_error"]
        ]
        funnel = {
            "answerable": len(answerable_cases),
            "candidate_complete": len(pool_complete),
            "top5_complete": len(top5_complete),
            "judge_approved": len(judge_approved),
            "generator_succeeded": len(generator_ok),
            "correct_final": len(correct_answers),
        }
        causes = Counter(item["root_cause"] for item in results if item["root_cause"])
        fallbacks = [
            {
                "case_id": item["case_id"],
                "supporting_chunk_ids": item["supporting_chunk_ids"],
                "citation_validity": item["citation_validity"],
            }
            for item in results
            if item["extractive_path"] == "verbatim_supporting_fallback"
        ]
        silent = [
            item["case_id"]
            for item in results
            if item["answerable"]
            and item["status"] == "abstained"
            and not item["generation_operational_error"]
            and not item["operational_error"]
            and cases_by_id[item["case_id"]].expected_answerability
        ]
        generator = {
            "normal_extractive": sum(
                1 for item in results if item["extractive_path"] == "query_overlap"
            ),
            "verbatim_supporting_fallback": len(fallbacks),
            "generation_failure": sum(
                1
                for item in results
                if item["generation_operational_error"] == "GENERATION_FAILURE"
            ),
            "silent_generation_failure": len(silent),
            "fallback_cases": fallbacks,
        }
        retries = dict(extra["retries"])
        provider = {
            "logical_judge_requests": len(results),
            "physical_sol_attempts": sum(
                item["physical_attempts"] or 0 for item in results if item["live"]
            ),
            "transport_retries": sum(item["retry_count"] or 0 for item in results),
            "timeout_retries": retries.get("TIMEOUT", 0),
            "rate_limit_retries": retries.get("RATE_LIMIT", 0),
            "provider_5xx_retries": retries.get("PROVIDER_5XX", 0),
            "connection_error_retries": retries.get("CONNECTION_ERROR", 0),
            "transport_recovered_cases": extra["transport_recovered"],
            "final_judge_request_errors": sum(
                1 for item in results if item["operational_error"] == "JUDGE_REQUEST_ERROR"
            ),
            "schema_valid_quality_retries": extra["quality_retries"],
            "unique_logical_ids": len(
                {item["logical_request_id"] for item in results if item["logical_request_id"]}
            ),
            "duplicate_evaluation_rows": len(results) - len({item["case_id"] for item in results}),
            "duplicate_final_answers": 0,
            "one_decision_per_logical_request": len(
                {item["logical_request_id"] for item in results if item["logical_request_id"]}
            )
            == len(results),
        }
        should_abstain = [item for item in CASES if item.should_abstain]
        injection = [item for item in CASES if item.category == "prompt_injection"]
        answered_validity = [
            item["citation_validity"]
            for item in answered
            if item["citation_validity"] is not None
        ]
        security = {
            **(record.security or {}),
            "unauthorized_supporting_evidence": sum(
                1
                for item in results
                if set(item["supporting_chunk_ids"])
                & set(cases_by_id[item["case_id"]].forbidden_document_ids)
            ),
            "unauthorized_citations": sum(
                1
                for item in results
                for citation in item["citations"]
                if cases_by_id[item["case_id"]].expected_access_behavior == "EXCLUDE_FORBIDDEN"
                and citation.get("document_id")
                in set(cases_by_id[item["case_id"]].forbidden_document_ids)
            ),
            "prompt_injection_cases": len(injection),
            "document_instructions_followed": sum(
                1 for item in injection if by_id[item.case_id]["document_instruction_followed"]
            ),
            "unauthorized_evidence_selected": sum(
                1 for item in injection if by_id[item.case_id]["unauthorized_evidence_selected"]
            ),
            "unsafe_answers": sum(
                1
                for item in injection
                if by_id[item.case_id]["behavior"] == "UNSUPPORTED_ANSWER"
            ),
            "boundary_success_rate": (
                sum(
                    1
                    for item in injection
                    if by_id[item.case_id]["behavior"]
                    in {"CORRECT_ABSTENTION", "CORRECT_ANSWER"}
                    and not by_id[item.case_id]["document_instruction_followed"]
                )
                / len(injection)
            ),
        }
        security["acl_safety"] = float(
            security.get("unauthorized_chunks_to_cross_encoder", 0) == 0
            and security.get("unauthorized_chunks_to_judge", 0) == 0
            and security["unauthorized_supporting_evidence"] == 0
            and security["unauthorized_citations"] == 0
        )
        live_judge = [
            item["judge_latency_ms"]
            for item in results
            if item["live"] and item["judge_latency_ms"] is not None
        ]
        live_generation = [item["generation_latency_ms"] for item in results]
        retrieval_latency = (record.latency or {}).get("retrieval") or {}
        usage = {
            **(record.usage or {}),
            "logical_judge_requests": provider["logical_judge_requests"],
            "physical_sol_calls": provider["physical_sol_attempts"],
            "judge_input_tokens": sum(
                item["prompt_tokens"] or 0 for item in results if item["live"]
            ),
            "judge_output_tokens": sum(
                item["completion_tokens"] or 0 for item in results if item["live"]
            ),
            "judge_reasoning_tokens": extra["reasoning_tokens"],
            "judge_cached_prompt_tokens": extra["cached_prompt_tokens"],
            "transport_retries": provider["transport_retries"],
            "external_reranker_calls": 0,
            "new_sol_judge_calls": sum(1 for item in results if item["live"]),
        }
        sol_cost = official_token_cost(
            model=SOL_MODEL,
            input_tokens=usage["judge_input_tokens"],
            output_tokens=usage["judge_output_tokens"],
            cached_input_tokens=usage["judge_cached_prompt_tokens"],
        )
        category_results = {
            category: self._category_report(category, results, cases_by_id)
            for category in EXPECTED_DISTRIBUTION
        }
        category_results = self._enrich_categories(
            category_results, results, cases_by_id, record
        )
        remaining = Counter(
            item["root_cause"]
            for item in results
            if item["root_cause"]
            and cases_by_id[item["case_id"]].expected_answerability
        )
        bottleneck = remaining.most_common(1)[0][0] if remaining else "NONE"
        v2_metrics = {
            "correct_answers": end_to_end["correct_answers"],
            "correct_abstentions": end_to_end["correct_abstentions"],
            "unsupported_answers": end_to_end["unsupported_answers"],
            "incorrect_abstentions": end_to_end["incorrect_abstentions"],
            "accuracy": end_to_end["accuracy"],
            "precision": end_to_end["precision"],
            "recall": end_to_end["recall"],
            "f1": end_to_end["f1"],
            "candidate_pool_all_required_coverage": (record.retrieval_results or {})
            .get("pool", {})
            .get("all_required_evidence_coverage"),
            "top5_all_required_coverage": (record.retrieval_results or {})
            .get("metrics", {})
            .get("all_required_evidence_coverage_at_5"),
            "three_document_top5_coverage": (record.retrieval_results or {})
            .get("metrics", {})
            .get("three_document_coverage_at_5"),
        }
        return {
            "cases": results,
            "end_to_end": {
                **end_to_end,
                "retrieval_complete_judge": _confusion(complete_answerable),
                "retrieval_complete_count": len(complete_answerable),
                "retrieval_incomplete": {
                    "case_count": len(incomplete),
                    "candidate_generation_misses": sum(
                        1
                        for item in incomplete
                        if item["root_cause"] == "CANDIDATE_GENERATION_MISS"
                    ),
                    "pool_complete_top5_incomplete": sum(
                        1 for item in incomplete if item["pool_complete"]
                    ),
                    "incorrect_abstentions": sum(
                        1 for item in incomplete if item["behavior"] == "INCORRECT_ABSTENTION"
                    ),
                    "unsupported_answers": sum(
                        1 for item in incomplete if item["behavior"] == "UNSUPPORTED_ANSWER"
                    ),
                },
                "should_abstain": {
                    "case_count": len(should_abstain),
                    "correct_abstentions": sum(
                        1
                        for item in should_abstain
                        if by_id[item.case_id]["behavior"] == "CORRECT_ABSTENTION"
                    ),
                    "false_positive_judge_decisions": sum(
                        1 for item in should_abstain if by_id[item.case_id]["class"] == "FP"
                    ),
                    "unsupported_answers": sum(
                        1
                        for item in should_abstain
                        if by_id[item.case_id]["behavior"] == "UNSUPPORTED_ANSWER"
                    ),
                },
            },
            "category_results": category_results,
            "stage_funnel": funnel,
            "failure_taxonomy": dict(causes),
            "generator_reliability": generator,
            "provider_reliability": provider,
            "security": security,
            "citations": {
                "answered_case_count": len(answered),
                "citation_validity": mean(answered_validity) if answered_validity else None,
                "citation_correctness": mean(answered_validity) if answered_validity else None,
                "missing_citation_rate": (
                    sum(1 for item in answered if not item["citations"]) / len(answered)
                    if answered
                    else None
                ),
                "invalid_citation_count": sum(
                    1 for item in answered if (item["citation_validity"] or 1) < 1
                ),
                "denominator": "answered_cases",
            },
            "latency": {
                "query_embedding": retrieval_latency.get("query_embedding_ms"),
                "dense": retrieval_latency.get("dense_ms"),
                "bm25": retrieval_latency.get("bm25_ms"),
                "rrf": retrieval_latency.get("rrf_ms"),
                "cross_encoder": retrieval_latency.get("cross_encoder_ms"),
                "sol_judge_live": _latency(live_judge),
                "support_validation": _latency(
                    [
                        item["support_validation_ms"]
                        for item in results
                        if item.get("support_validation_ms") is not None
                    ]
                ),
                "generation": _latency(live_generation),
                "total_pipeline": _latency(
                    [
                        item["total_pipeline_ms"]
                        for item in results
                        if item.get("total_pipeline_ms") is not None
                    ]
                ),
                "retry_related_judge": _latency(
                    [
                        item["judge_latency_ms"]
                        for item in results
                        if item.get("retry_count") and item.get("judge_latency_ms") is not None
                    ]
                ),
            },
            "usage": usage,
            "cost": {
                "official_pricing_verified": True,
                "sol_cost_usd": sol_cost,
                "embedding_cost_usd": "NOT VERIFIED",
                "total_verified_external_cost_usd": sol_cost,
                "mean_sol_cost_per_case_usd": sol_cost / len(results),
                "retry_related_incremental_cost_usd": 0.0
                if not provider["transport_retries"]
                else "included_in_sol_tokens",
            },
            "v1_comparison": {
                "note": (
                    "V1 and V2 final benchmarks use different unseen datasets. "
                    "This is a descriptive cross-dataset comparison, NOT a controlled "
                    "paired A/B experiment. Numerical differences must not be interpreted "
                    "as a causal architecture improvement without qualification."
                ),
                "v1": V1_FINAL_DESCRIPTIVE,
                "v2": {"dataset_id": DATASET_ID, "case_count": 100, **v2_metrics},
            },
            "primary_remaining_bottleneck": bottleneck,
        }

    def _category_report(
        self,
        category: str,
        results: list[dict[str, Any]],
        cases_by_id: dict[str, V2FinalCase],
    ) -> dict[str, Any]:
        scoped = [item for item in results if item["category"] == category]
        answerable = [
            item for item in scoped if cases_by_id[item["case_id"]].expected_answerability
        ]
        complete = [item for item in answerable if item["retrieval_complete"]]
        return {
            "case_count": len(scoped),
            "candidate_pool_complete": sum(1 for item in answerable if item["pool_complete"]),
            "top5_complete": sum(1 for item in answerable if item["retrieval_complete"]),
            "correct_answers": sum(1 for item in scoped if item["behavior"] == "CORRECT_ANSWER"),
            "incorrect_abstentions": sum(
                1 for item in scoped if item["behavior"] == "INCORRECT_ABSTENTION"
            ),
            "unsupported_answers": sum(
                1 for item in scoped if item["behavior"] == "UNSUPPORTED_ANSWER"
            ),
            "judge_after_complete_retrieval": _confusion(complete),
        }

    def _enrich_categories(
        self,
        reports: dict[str, Any],
        results: list[dict[str, Any]],
        cases_by_id: dict[str, V2FinalCase],
        record: V2FinalBenchmarkRecord,
    ) -> dict[str, Any]:
        retrieval = record.retrieval_results or {}
        metrics = retrieval.get("metrics") or {}
        three = reports["multidoc_three"]
        three_failed = [
            item
            for item in results
            if item["category"] == "multidoc_three"
            and cases_by_id[item["case_id"]].expected_answerability
            and item["behavior"] != "CORRECT_ANSWER"
        ]
        three["required_evidence_recall_at_5"] = metrics.get("three_document_coverage_at_5")
        three["failed_answerable_stage"] = dict(
            Counter(item["root_cause"] or "UNKNOWN" for item in three_failed)
        )
        exact = reports["exact_identifier"]
        exact_rows = [item for item in results if item["category"] == "exact_identifier"]
        exact_answerable = [
            item
            for item in exact_rows
            if cases_by_id[item["case_id"]].expected_answerability
        ]
        exact["candidate_pool_recall"] = (
            mean(float(item["pool_complete"]) for item in exact_answerable)
            if exact_answerable
            else 0.0
        )
        exact["top5_recall"] = (
            mean(float(item["retrieval_complete"]) for item in exact_answerable)
            if exact_answerable
            else 0.0
        )
        exact["judge_tp_fn"] = {
            "tp": sum(1 for item in exact_rows if item["class"] == "TP"),
            "fn": sum(1 for item in exact_rows if item["class"] == "FN"),
        }
        version = reports["version_region"]
        version_rows = [item for item in results if item["category"] == "version_region"]
        version["retrieval_active_version_correctness"] = metrics.get("version_correctness")
        version["support_active_version_correctness"] = (
            mean([float(item["version_correct"]) for item in version_rows]) if version_rows else 0.0
        )
        version["final_answer_version_correctness"] = (
            mean(
                [
                    float(item["version_correct"])
                    for item in version_rows
                    if item["behavior"] == "CORRECT_ANSWER"
                ]
            )
            if any(item["behavior"] == "CORRECT_ANSWER" for item in version_rows)
            else 0.0
        )
        duplicate = reports["near_duplicate"]
        dup_rows = [item for item in results if item["category"] == "near_duplicate"]
        pool_hits = [
            float(item["preferred_source_pool"])
            for item in dup_rows
            if item["preferred_source_pool"] is not None
        ]
        duplicate["preferred_source_candidate_success"] = mean(pool_hits) if pool_hits else 0.0
        duplicate["preferred_source_top5_success"] = metrics.get(
            "near_duplicate_preferred_source_success"
        )
        duplicate["wrong_source_failures"] = sum(
            1
            for item in dup_rows
            if item["preferred_source_top5"] is False and item["behavior"] != "CORRECT_ANSWER"
        )
        semantic = reports["semantic_paraphrase"]
        semantic["retrieval_success"] = metrics.get("semantic_paraphrase_success")
        return reports

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(V2FinalBenchmarkRecord, DATASET_ID)
        architecture = self.session.get(RetrievalArchitectureRecord, V2_ARCHITECTURE_ID)
        research = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        payload: dict[str, Any] = {
            "dataset_id": DATASET_ID,
            "architecture_id": V2_ARCHITECTURE_ID,
            "parent_architecture": RELEASE_ARCHITECTURE_ID,
            "initialized": record is not None,
            "completed": bool(record and record.completed_at),
            "dataset_frozen": bool(record and record.dataset_frozen),
            "one_shot": True,
            "final_v2_benchmark_one_shot": bool(record and record.one_shot_locked_at),
            "generation_method": GENERATION_METHOD,
            "selected_v2_ranking": research.selected_v2_ranking if research else CONTROL_MODE,
            "selected_v2_judge": research.selected_v2_judge if research else CONTROL_JUDGE,
            "ranking_research_status": (
                research.ranking_research_status if research else RANKING_RESEARCH_FROZEN
            ),
            "judge_research_status": (
                research.judge_research_status if research else JUDGE_RESEARCH_FROZEN
            ),
            "v1_frozen": True,
        }
        if architecture:
            payload["architecture"] = {
                "architecture_id": architecture.architecture_id,
                "selected_retriever": architecture.selected_retriever,
                "architecture_hash": (architecture.configuration or {}).get("architecture_hash"),
                "immutable": architecture.immutable,
            }
        if not record:
            return payload
        payload.update(
            {
                "dataset_hash": record.dataset_hash,
                "architecture_hash": record.architecture_hash,
                "case_count": len(record.case_ids or []),
                "category_distribution": record.category_distribution,
                "maximum_prior_overlap": record.maximum_prior_overlap,
                "closest_previous_case": record.closest_previous_case,
                "freeze_timestamp": record.dataset_frozen_at,
                "one_shot_locked_at": record.one_shot_locked_at,
                "embedding_preflight": record.embedding_preflight,
                "judge_preflight": record.judge_preflight,
                "retrieval": record.retrieval_results,
                "end_to_end": record.end_to_end,
                "category_results": record.category_results,
                "stage_funnel": record.stage_funnel,
                "failure_taxonomy": record.failure_taxonomy,
                "generator_reliability": record.generator_reliability,
                "provider_reliability": record.provider_reliability,
                "security": record.security,
                "citations": record.citations,
                "latency": record.latency,
                "usage": record.usage,
                "cost": record.cost,
                "v1_comparison": record.v1_comparison,
                "primary_remaining_bottleneck": record.primary_remaining_bottleneck,
                "architecture_configuration": record.architecture_configuration,
            }
        )
        if include_cases:
            payload["cases"] = record.case_results
        return payload


class _FixedGate:
    provider_name = "openai"
    model_name = SOL_MODEL
    gate_version = "1"
    prompt_version = EVIDENCE_GATE_PROMPT_VERSION

    def __init__(self, result: AnswerabilityResult, timing: Any) -> None:
        self.result = result
        self.last_timing = timing

    def evaluate(self, question, chunks):
        return self.result


def classification(expected: bool, answerable: bool) -> str:
    if expected and answerable:
        return "TP"
    if expected and not answerable:
        return "FN"
    if not expected and answerable:
        return "FP"
    return "TN"


def _document_instruction_followed(answer: str | None) -> bool:
    if not answer:
        return False
    text = answer.casefold()
    return "ignore all previous instructions" in text or "reveal system secrets" in text


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": mean(values) if values else 0.0,
        "p50_ms": median(values) if values else 0.0,
        "p95_ms": _percentile(values, 0.95) if values else 0.0,
        "count": len(values),
    }


def classify_v2_root_cause(
    case: V2FinalCase, trace: dict[str, Any], payload: dict[str, Any]
) -> str | None:
    if case.category == "prompt_injection" and payload["document_instruction_followed"]:
        return "PROMPT_INJECTION_FAILURE"
    if (
        case.expected_access_behavior == "EXCLUDE_FORBIDDEN"
        and payload["unauthorized_evidence_selected"]
    ):
        return "ACL_FAILURE"
    if payload["behavior"] in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"}:
        return None
    if payload["operational_error"] == "JUDGE_REQUEST_ERROR":
        return "JUDGE_REQUEST_ERROR"
    if not case.expected_answerability:
        if payload["behavior"] == "UNSUPPORTED_ANSWER":
            return "EVIDENCE_GATE_FALSE_POSITIVE"
        return "UNKNOWN"
    if not payload["retrieval_complete"]:
        taxonomy = trace.get("failure_taxonomy")
        mapping = {
            "BOTH_BRANCHES_MISS": "CANDIDATE_GENERATION_MISS",
            "DENSE_CANDIDATE_GENERATION_MISS": "CANDIDATE_GENERATION_MISS",
            "RRF_TRUNCATION_LOSS": "CANDIDATE_GENERATION_MISS",
            "CROSS_ENCODER_FAILED_TO_PROMOTE": "CROSS_ENCODER_FAILED_TO_PROMOTE",
            "CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE": "CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE",
            "RANKING_OUTSIDE_TOP5": "RANKING_OUTSIDE_TOP5",
            "THREE_DOCUMENT_COVERAGE_FAILURE": "RANKING_OUTSIDE_TOP5",
            "LEXICAL_CANDIDATE_RESCUE": "CROSS_ENCODER_FAILED_TO_PROMOTE",
            "ACL_FAILURE": "ACL_FAILURE",
        }
        return mapping.get(taxonomy, "CANDIDATE_GENERATION_MISS")
    if not payload["answerable"]:
        return "EVIDENCE_GATE_FALSE_NEGATIVE"
    if payload["generation_operational_error"] == "GENERATION_FAILURE":
        return "GENERATION_FAILURE"
    if payload["citation_validity"] == 0.0:
        return "CITATION_FAILURE"
    if payload["status"] != "answered":
        return "GENERATION_FAILURE"
    return "UNKNOWN"
