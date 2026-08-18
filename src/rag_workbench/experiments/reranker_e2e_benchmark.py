from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.answerability.cache import CachedAnswerabilityGate, gate_cache_key
from rag_workbench.answerability.openai_compatible import OpenAICompatibleAnswerabilityGate
from rag_workbench.answerability.planning import ExternalJudgeCallLimitGate
from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import (
    AnswerabilityGateCacheRecord,
    Chunk,
    Document,
    DocumentVersion,
    EndToEndBenchmarkRecord,
    EndToEndBenchmarkRunRecord,
    ExperimentRunRecord,
    QueryEmbeddingCacheRecord,
    RerankingBenchmarkRecord,
)
from rag_workbench.evaluation.datasets import EvaluationCase, load_evaluation_dataset
from rag_workbench.evaluation.evaluator import CaseResult, EvaluationRunner
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.generation.generator import RagService
from rag_workbench.providers.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from rag_workbench.providers.llm import ExtractiveGenerationProvider
from rag_workbench.reranking import CrossEncoderReranker, Reranker
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import RetrievalTiming, Retriever
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal

DATASET_ID = "acmeai-reranker-e2e-eval-v1"
DATASET_PATH = Path("data/eval/acmeai_reranker_e2e_eval_v1.json")
DATASET_HASH = "288b26b0f1adc9c75b2b1d2c617c362f9aa0540ebb5591157bd728ff94fa2e16"
GENERATION_METHOD = "manual-corpus-grounded-v1"
MAXIMUM_PRIOR_OVERLAP = 0.391304347826087
SEMANTIC_INDEX_IDENTITY = "e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466"
CORPUS_IDENTITY = "f78baa9c2d891d2be09383eba5830c3aa3bc0b510b702e83a38f1a36d7b21ff5"
RERANKER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MODES = ("DENSE_C2", "DENSE_CROSS_ENCODER_RERANK_C2")
COMMON_CONFIGURATION = {
    "embedding_model": "text-embedding-3-small",
    "embedding_dimension": 64,
    "dense_distance": "pgvector_cosine",
    "dense_threshold": 0.28,
    "final_top_k": 5,
    "evidence_judge": "gpt-5.6-luna",
    "judge_version": "1",
    "judge_prompt": "evidence-sufficiency-v1",
    "supporting_context_only": True,
    "generator": "deterministic-extractive-v1",
    "context_budget": 1200,
}
PIPELINE_A = {**COMMON_CONFIGURATION, "retrieval_mode": "DENSE", "reranking": False}
PIPELINE_B = {
    **COMMON_CONFIGURATION,
    "retrieval_mode": "DENSE_CROSS_ENCODER_RERANK",
    "reranking": True,
    "dense_candidate_depth": 20,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "reranker_revision": RERANKER_REVISION,
    "reranker_device": "cpu",
}
SUCCESS_POLICY = {
    "minimum_correct_answer_gain": 3,
    "minimum_incorrect_abstention_reduction": 3,
    "maximum_unsupported_answer_increase": 0,
    "answerability_f1_must_not_regress": True,
    "hard_constraints": {
        "acl_safety": 1.0,
        "tenant_isolation": 1.0,
        "version_correctness": 1.0,
        "prompt_injection_boundary": 1.0,
    },
}

DATASET = load_evaluation_dataset(DATASET_PATH)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _principal(case: EvaluationCase) -> Principal:
    return Principal(
        case.principal.principal_id,
        case.principal.tenant_id,
        frozenset(case.principal.permission_groups),
    )


def _trace(result: RetrievalResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "document_version_id": result.document_version_id,
        "text": result.text,
        "rank": result.rank,
        "score": result.score,
        "source": result.source,
        "source_type": result.source_type,
        "title": result.title,
        "version": result.version,
        "page": result.page,
        "section": result.section,
        "metadata": result.metadata,
        "retrieval_source": result.retrieval_source,
        "dense_score": result.dense_score,
    }


def _result(value: dict[str, Any]) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=value["chunk_id"],
        document_id=value["document_id"],
        document_version_id=value["document_version_id"],
        text=value["text"],
        rank=value["rank"],
        score=value["score"],
        source=value["source"],
        source_type=value["source_type"],
        title=value["title"],
        version=value["version"],
        page=value["page"],
        section=value["section"],
        metadata=value["metadata"],
        retrieval_source=value["retrieval_source"],
        dense_score=value["dense_score"],
        found_by_dense=True,
    )


class FixedResultRetriever:
    """Production-compatible retriever facade over one frozen, authorized trace."""

    def __init__(
        self,
        embedding_provider: Any,
        results: list[RetrievalResult],
        timing: RetrievalTiming,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.results = results
        self.last_timing = timing
        self.index_identity = SEMANTIC_INDEX_IDENTITY

    def retrieve(self, *_: Any, **__: Any) -> list[RetrievalResult]:
        return list(self.results)


class RerankerEndToEndBenchmark:
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
        self._verify_dataset_file()

    def _verify_dataset_file(self) -> None:
        if DATASET.dataset_version != DATASET_ID or len(DATASET.cases) != 60:
            raise ValueError("end-to-end dataset identity changed")
        if hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() != DATASET_HASH:
            raise ValueError("end-to-end dataset hash changed")

    def _corpus_identity(self) -> str:
        rows = self.session.execute(
            select(Document.document_id, DocumentVersion.version, DocumentVersion.content_hash)
            .join(DocumentVersion, DocumentVersion.document_fk == Document.id)
            .where(DocumentVersion.is_active.is_(True))
            .order_by(Document.document_id, DocumentVersion.version)
        ).all()
        return hashlib.sha256("|".join(":".join(row) for row in rows).encode()).hexdigest()

    def initialize(self) -> EndToEndBenchmarkRecord:
        self._verify_frozen_dependencies()
        existing = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        if existing:
            self._verify_record(existing)
            return existing
        record = EndToEndBenchmarkRecord(
            dataset_id=DATASET_ID,
            dataset_hash=DATASET_HASH,
            case_ids=[case.case_id for case in DATASET.cases],
            category_distribution=dict(
                sorted(Counter(case.category for case in DATASET.cases).items())
            ),
            generation_method=GENERATION_METHOD,
            maximum_prior_overlap=MAXIMUM_PRIOR_OVERLAP,
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=CORPUS_IDENTITY,
            reranker_revision=RERANKER_REVISION,
            pipeline_a_configuration={**PIPELINE_A, "success_policy": SUCCESS_POLICY},
            pipeline_b_configuration={**PIPELINE_B, "success_policy": SUCCESS_POLICY},
        )
        self.session.add(record)
        self.session.commit()
        return record

    def _verify_frozen_dependencies(self) -> None:
        if self._corpus_identity() != CORPUS_IDENTITY:
            raise ValueError("frozen corpus identity changed")
        if not self.session.scalar(
            select(Chunk.id).where(Chunk.index_identity == SEMANTIC_INDEX_IDENTITY).limit(1)
        ):
            raise ValueError("frozen semantic index is unavailable")
        reranking = self.session.get(RerankingBenchmarkRecord, "acmeai-reranking-eval-v1")
        if (
            not reranking
            or reranking.selected_mode != "DENSE_CROSS_ENCODER_RERANK"
            or reranking.model_revision != RERANKER_REVISION
            or reranking.holdout_completed_at is None
        ):
            raise ValueError("frozen reranker selection changed")

    @staticmethod
    def _verify_record(record: EndToEndBenchmarkRecord) -> None:
        if not (
            record.dataset_hash == DATASET_HASH
            and record.case_ids == [case.case_id for case in DATASET.cases]
            and record.generation_method == GENERATION_METHOD
            and record.maximum_prior_overlap == MAXIMUM_PRIOR_OVERLAP
            and record.semantic_index_identity == SEMANTIC_INDEX_IDENTITY
            and record.corpus_identity == CORPUS_IDENTITY
            and record.reranker_revision == RERANKER_REVISION
            and record.pipeline_a_configuration == {**PIPELINE_A, "success_policy": SUCCESS_POLICY}
            and record.pipeline_b_configuration == {**PIPELINE_B, "success_policy": SUCCESS_POLICY}
        ):
            raise ValueError("sealed end-to-end benchmark identity changed")

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

    def embedding_preflight(self) -> dict[str, int]:
        queries = tuple(dict.fromkeys(case.question for case in DATASET.cases))
        keys = [
            query_embedding_cache_key(
                query,
                provider="openai-compatible",
                model="text-embedding-3-small",
                version="1",
                dimension=64,
            )
            for query in queries
        ]
        matches = sum(self.session.get(QueryEmbeddingCacheRecord, key) is not None for key in keys)
        current = self._embedding_calls()
        return {
            "current_cumulative_embedding_calls": current,
            "configured_ceiling": self.settings.max_external_embedding_calls,
            "dataset_unique_queries": len(queries),
            "existing_cache_matches": matches,
            "missing_embeddings": len(keys) - matches,
            "maximum_new_embedding_calls": len(keys) - matches,
            "expected_cumulative_ending_usage": current + len(keys) - matches,
        }

    def prepare(self) -> dict[str, Any]:
        record = self.initialize()
        if record.preparation_started_at is not None:
            raise ValueError("sealed end-to-end preparation is one-shot and already started")
        preflight = self.embedding_preflight()
        if not self.settings.allow_external_calls or not self.settings.embedding_api_key:
            raise ValueError("external embedding authorization is incomplete")
        if preflight["expected_cumulative_ending_usage"] > preflight["configured_ceiling"]:
            raise ValueError("end-to-end dataset exceeds the embedding-call ceiling")
        record.embedding_preflight = preflight
        record.preparation_started_at = datetime.now(UTC)
        self.session.commit()

        provider = self._embedding_provider()
        dense = Retriever(self.session, provider, SEMANTIC_INDEX_IDENTITY)
        reranker: Reranker = self.reranker_factory()
        if reranker.resolved_revision != RERANKER_REVISION:
            raise ValueError("local reranker revision differs from the frozen revision")
        prepared: list[dict[str, Any]] = []
        embedding_calls = embedding_tokens = cache_hits = cache_misses = pairs = 0
        unauthorized_to_reranker = 0
        for case in DATASET.cases:
            embedding = dense.query_embedding_cache.get_or_embed(case.question)
            self.session.commit()
            candidates = dense.retrieve_with_embedding(
                embedding,
                top_k=20,
                score_threshold=0.28,
                principal=_principal(case),
            )
            forbidden = set(case.forbidden_document_ids)
            if "acl" in case.security_checks:
                unauthorized_to_reranker += sum(
                    item.document_id in forbidden for item in candidates
                )
            reranked = reranker.rerank(case.question, candidates)
            reranked_results = [
                replace(
                    item.result,
                    rank=item.reranked_rank,
                    score=item.reranker_score,
                    retrieval_source="dense_cross_encoder_rerank",
                )
                for item in reranked
            ]
            prepared.append(
                {
                    "case_id": case.case_id,
                    "dense_candidates": [_trace(item) for item in candidates],
                    "pipeline_a_top5": [_trace(item) for item in candidates[:5]],
                    "pipeline_b_top5": [_trace(item) for item in reranked_results[:5]],
                    "reranking_trace": [
                        {
                            "chunk_id": item.result.chunk_id,
                            "document_id": item.result.document_id,
                            "original_dense_rank": item.original_dense_rank,
                            "dense_score": item.dense_score,
                            "reranker_score": item.reranker_score,
                            "reranked_rank": item.reranked_rank,
                        }
                        for item in reranked
                    ],
                    "timing": {
                        "query_embedding_ms": embedding.embedding_latency_ms,
                        "cache_lookup_ms": embedding.cache_lookup_latency_ms,
                        "dense_vector_search_ms": dense.last_timing.vector_search_latency_ms,
                        "dense_acl_filter_ms": dense.last_timing.acl_filter_latency_ms,
                        "reranker_inference_ms": getattr(
                            reranker, "last_inference_latency_ms", 0.0
                        ),
                        "reranker_sort_ms": getattr(reranker, "last_sort_latency_ms", 0.0),
                    },
                    "embedding_cache_hit": embedding.cache_hit,
                    "embedding_external_calls": embedding.external_calls,
                    "embedding_tokens": embedding.input_tokens,
                }
            )
            embedding_calls += embedding.external_calls
            embedding_tokens += embedding.input_tokens
            cache_hits += int(embedding.cache_hit)
            cache_misses += int(not embedding.cache_hit)
            pairs += len(candidates)
        if unauthorized_to_reranker:
            raise RuntimeError("unauthorized content reached the frozen Cross-Encoder")
        record = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        record.prepared_cases = prepared
        record.preparation_usage = {
            "new_query_embedding_calls": embedding_calls,
            "embedding_tokens": embedding_tokens,
            "query_cache_hits": cache_hits,
            "query_cache_misses": cache_misses,
            "document_embedding_calls": 0,
            "cross_encoder_pairs": pairs,
            "mean_pairs_per_query": pairs / len(DATASET.cases),
            "unauthorized_chunks_to_reranker": 0,
        }
        record.prepared_at = datetime.now(UTC)
        self.session.commit()
        return self.status(include_cases=False)

    @staticmethod
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

    def _historical_judge_calls(self) -> int:
        historical = sum(
            item.external_judge_calls or 0
            for item in self.session.scalars(select(ExperimentRunRecord)).all()
        )
        current = sum(
            (item.usage or {}).get("new_luna_judge_calls", 0)
            for item in self.session.scalars(select(EndToEndBenchmarkRunRecord)).all()
        )
        return historical + current

    def judge_preflight(self, *, persist: bool = True) -> dict[str, Any]:
        record = self.initialize()
        if not record.prepared_at or not record.prepared_cases:
            raise ValueError("frozen retrieval traces must be prepared before judge preflight")
        a_keys: list[str] = []
        b_keys: list[str] = []
        identical = 0
        for case, prepared in zip(DATASET.cases, record.prepared_cases, strict=True):
            a_key = gate_cache_key(
                case.question,
                self._gate_evidence(prepared["pipeline_a_top5"]),
                provider="openai",
                model="gpt-5.6-luna",
                gate_version="1",
                prompt_version="evidence-sufficiency-v1",
            )[0]
            b_key = gate_cache_key(
                case.question,
                self._gate_evidence(prepared["pipeline_b_top5"]),
                provider="openai",
                model="gpt-5.6-luna",
                gate_version="1",
                prompt_version="evidence-sufficiency-v1",
            )[0]
            a_keys.append(a_key)
            b_keys.append(b_key)
            identical += int(a_key == b_key)
        union = set(a_keys) | set(b_keys)
        existing_a = sum(
            self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in set(a_keys)
        )
        existing_b = sum(
            self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in set(b_keys)
        )
        existing_union = sum(
            self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in union
        )
        current = self._historical_judge_calls()
        preflight = {
            "current_cumulative_judge_calls": current,
            "configured_ceiling": self.settings.max_external_judge_calls,
            "unique_pipeline_a_gate_inputs": len(set(a_keys)),
            "unique_pipeline_b_gate_inputs": len(set(b_keys)),
            "identical_a_b_gate_inputs": identical,
            "unique_combined_gate_inputs": len(union),
            "existing_pipeline_a_cache_matches": existing_a,
            "existing_pipeline_b_cache_matches": existing_b,
            "existing_valid_gate_cache_matches": existing_union,
            "maximum_new_external_calls": len(union) - existing_union,
            "expected_worst_case_cumulative_calls": current + len(union) - existing_union,
        }
        if persist:
            record.judge_preflight = preflight
            self.session.commit()
        return preflight

    def _gate(self, maximum_calls: int) -> CachedAnswerabilityGate:
        if self.gate_factory:
            return self.gate_factory(maximum_calls)
        delegate = OpenAICompatibleAnswerabilityGate(
            api_key=self.settings.effective_judge_api_key or "",
            model="gpt-5.6-luna",
            base_url=self.settings.judge_base_url,
            gate_version="1",
            prompt_version="evidence-sufficiency-v1",
            provider_name="openai",
        )
        return CachedAnswerabilityGate(
            self.session, ExternalJudgeCallLimitGate(delegate, maximum_calls)
        )

    def execute(self) -> dict[str, Any]:
        record = self.initialize()
        if record.execution_started_at is not None:
            raise ValueError("sealed end-to-end evaluation is one-shot and already started")
        preflight = self.judge_preflight()
        if (
            not self.settings.allow_external_judge_calls
            or not self.settings.effective_judge_api_key
        ):
            raise ValueError("hosted judge authorization is incomplete")
        if preflight["expected_worst_case_cumulative_calls"] > preflight["configured_ceiling"]:
            raise ValueError("paired C2 inputs exceed the hosted judge-call ceiling")
        record.execution_started_at = datetime.now(UTC)
        self.session.commit()
        remaining = self.settings.max_external_judge_calls - self._historical_judge_calls()
        gate = self._gate(remaining)
        provider = self._embedding_provider()
        results: dict[str, list[CaseResult]] = {mode: [] for mode in MODES}
        unauthorized_to_judge = 0
        for case, prepared in zip(DATASET.cases, record.prepared_cases or [], strict=True):
            common_timing = RetrievalTiming(
                query_embedding_latency_ms=prepared["timing"]["query_embedding_ms"],
                embedding_cache_lookup_latency_ms=prepared["timing"]["cache_lookup_ms"],
                vector_search_latency_ms=prepared["timing"]["dense_vector_search_ms"],
                acl_filter_latency_ms=prepared["timing"]["dense_acl_filter_ms"],
                query_embedding_cache_hit=prepared["embedding_cache_hit"],
                external_embedding_calls=prepared["embedding_external_calls"],
            )
            for mode, trace in (
                (MODES[0], prepared["pipeline_a_top5"]),
                (MODES[1], prepared["pipeline_b_top5"]),
            ):
                retrieved = [_result(item) for item in trace]
                if "acl" in case.security_checks:
                    unauthorized_to_judge += sum(
                        item.document_id in set(case.forbidden_document_ids) for item in retrieved
                    )
                rag = RagService(
                    self.session,
                    FixedResultRetriever(provider, retrieved, common_timing),
                    ContextBuilder(1200),
                    ExtractiveGenerationProvider(),
                    gate,
                    supporting_context_only=True,
                )
                results[mode].append(EvaluationRunner(rag).run_case(case, 5, 0.28))
        if unauthorized_to_judge:
            raise RuntimeError("unauthorized content reached frozen C2")

        reporter = EvaluationRunner(rag)
        reports = {mode: reporter._report(tuple(values)) for mode, values in results.items()}
        transitions, conversion, regression = self._paired_analysis(
            results, record.prepared_cases or []
        )
        for mode in MODES:
            report = reports[mode]
            case_payload = [
                self._case_payload(item, mode, record.prepared_cases or []) for item in report.cases
            ]
            self.session.add(
                EndToEndBenchmarkRunRecord(
                    dataset_id=DATASET_ID,
                    mode=mode,
                    metrics=report.metrics,
                    retrieval_metrics=self._retrieval_metrics(report.cases),
                    category_metrics=report.category_metrics,
                    case_results=case_payload,
                    latency=self._latency(mode, report.cases, record.prepared_cases or []),
                    usage=self._usage(mode, report.cases, record),
                )
            )
        a, b = reports[MODES[0]].metrics, reports[MODES[1]].metrics
        material = (
            b["correct_answer_count"] - a["correct_answer_count"]
            >= SUCCESS_POLICY["minimum_correct_answer_gain"]
            or a["incorrect_abstention_count"] - b["incorrect_abstention_count"]
            >= SUCCESS_POLICY["minimum_incorrect_abstention_reduction"]
        )
        safe = (
            b["unsupported_answer_count"] - a["unsupported_answer_count"]
            <= SUCCESS_POLICY["maximum_unsupported_answer_increase"]
            and b["answerability_f1"] >= a["answerability_f1"]
            and b["acl_safety"] == 1.0
            and b["version_accuracy"] == 1.0
            and b["prompt_injection_boundary"] == 1.0
        )
        record = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        record.paired_transitions = transitions
        record.conversion_analysis = conversion
        record.regression_analysis = regression
        record.production_retriever_status = (
            "DENSE_CROSS_ENCODER_RERANK" if material and safe else "DENSE"
        )
        record.completed_at = datetime.now(UTC)
        self.session.commit()
        return self.status(include_cases=True)

    @staticmethod
    def _case_payload(
        case: CaseResult, mode: str, prepared_cases: list[dict[str, Any]]
    ) -> dict[str, Any]:
        payload = asdict(case)
        prepared = next(item for item in prepared_cases if item["case_id"] == case.case_id)
        payload["dense_candidate_trace"] = prepared["dense_candidates"]
        payload["reranking_trace"] = prepared["reranking_trace"]
        payload["root_causes"] = RerankerEndToEndBenchmark._root_causes(case, mode, prepared)
        payload["final_behavior"] = RerankerEndToEndBenchmark._behavior(case)
        return payload

    @staticmethod
    def _root_causes(case: CaseResult, mode: str, prepared: dict[str, Any]) -> list[str]:
        causes: list[str] = []
        required = set(case.expected_document_ids)
        candidate_docs = {item["document_id"] for item in prepared["dense_candidates"]}
        retrieved = set(case.retrieved_document_ids)
        if required and not required <= candidate_docs and not case.expected_abstain:
            causes.extend(["CANDIDATE_GENERATION_MISS", "RETRIEVAL_INCOMPLETE"])
        elif required and not required <= retrieved and not case.expected_abstain:
            causes.extend(["RANKING_OUTSIDE_TOP5", "RETRIEVAL_INCOMPLETE"])
            dense_docs = {item["document_id"] for item in prepared["pipeline_a_top5"]}
            if mode == MODES[1] and required <= dense_docs:
                causes.append("RERANKER_DEMOTED_REQUIRED_EVIDENCE")
        if (
            case.retrieval_coverage_complete
            and case.status == "abstained"
            and not case.expected_abstain
        ):
            causes.append("EVIDENCE_GATE_FALSE_NEGATIVE")
        if case.supporting_context_loss:
            causes.append("SUPPORTING_CONTEXT_LOSS")
        if case.expected_abstain and case.status == "answered":
            causes.extend(["EVIDENCE_GATE_FALSE_POSITIVE", "UNSUPPORTED_ANSWER"])
        if not case.expected_abstain and case.status == "abstained":
            causes.append("INCORRECT_ABSTENTION")
        if case.citation_correctness == 0:
            causes.append("CITATION_FAILURE")
        if case.version_correct == 0:
            causes.append("VERSION_FAILURE")
        if case.security_passed is False:
            causes.append("ACL_FAILURE")
        if case.error:
            causes.append("GENERATION_FAILURE")
        return list(dict.fromkeys(causes)) or ["NONE"]

    @staticmethod
    def _behavior(case: CaseResult) -> str:
        if case.expected_abstain:
            return "CORRECT_ABSTENTION" if case.status == "abstained" else "UNSUPPORTED_ANSWER"
        return "CORRECT_ANSWER" if case.status == "answered" else "INCORRECT_ABSTENTION"

    @staticmethod
    def _paired_analysis(
        results: dict[str, list[CaseResult]], prepared: list[dict[str, Any]]
    ) -> tuple[dict[str, int], dict[str, Any], dict[str, Any]]:
        a_by_id = {item.case_id: item for item in results[MODES[0]]}
        b_by_id = {item.case_id: item for item in results[MODES[1]]}
        transitions: Counter[str] = Counter()
        rescues: list[dict[str, Any]] = []
        regressions: list[dict[str, Any]] = []
        for case_id, a in a_by_id.items():
            b = b_by_id[case_id]
            a_behavior = RerankerEndToEndBenchmark._behavior(a)
            b_behavior = RerankerEndToEndBenchmark._behavior(b)
            transitions[f"{a_behavior}→{b_behavior}"] += 1
            if not a.retrieval_coverage_complete and b.retrieval_coverage_complete:
                outcome = (
                    "ANSWER_FIXED"
                    if b_behavior == "CORRECT_ANSWER"
                    else "JUDGE_FALSE_NEGATIVE"
                    if b.status == "abstained"
                    else "STILL_WRONG"
                )
                rescues.append({"case_id": case_id, "outcome": outcome})
            if a.retrieval_coverage_complete and not b.retrieval_coverage_complete:
                effect = (
                    "NEW_INCORRECT_ABSTENTION"
                    if a_behavior == "CORRECT_ANSWER" and b_behavior == "INCORRECT_ABSTENTION"
                    else "NEW_WRONG_ANSWER"
                    if b_behavior == "UNSUPPORTED_ANSWER"
                    else "NO_DOWNSTREAM_EFFECT"
                )
                regressions.append({"case_id": case_id, "effect": effect})
        fixed = sum(item["outcome"] == "ANSWER_FIXED" for item in rescues)
        conversion = {
            "retrieval_rescue_count": len(rescues),
            "retrieval_rescue_to_correct_answer_count": fixed,
            "retrieval_rescue_still_failed_count": len(rescues) - fixed,
            "retrieval_to_answer_conversion_rate": fixed / len(rescues) if rescues else 0.0,
            "cases": rescues,
        }
        regression = {
            "reranker_retrieval_regression_count": len(regressions),
            "new_incorrect_abstentions": sum(
                item["effect"] == "NEW_INCORRECT_ABSTENTION" for item in regressions
            ),
            "new_wrong_answers": sum(item["effect"] == "NEW_WRONG_ANSWER" for item in regressions),
            "no_downstream_effect": sum(
                item["effect"] == "NO_DOWNSTREAM_EFFECT" for item in regressions
            ),
            "cases": regressions,
        }
        return dict(sorted(transitions.items())), conversion, regression

    @staticmethod
    def _retrieval_metrics(cases: tuple[CaseResult, ...]) -> dict[str, Any]:
        answerable = [item for item in cases if not item.expected_abstain]

        def avg(attribute: str, rows: list[CaseResult] = answerable) -> float:
            values = [getattr(item, attribute) for item in rows]
            return mean(values) if values else 0.0

        def category(name: str) -> list[CaseResult]:
            return [item for item in answerable if item.category == name]

        exact = category("exact_identifier")
        version = category("version_region")
        near = category("near_duplicate")
        return {
            "hit_at_5": avg("hit_at_k"),
            "recall_at_5": avg("recall_at_k"),
            "mrr": avg("reciprocal_rank"),
            "ndcg_at_5": avg("ndcg_at_k"),
            "required_evidence_recall_at_5": avg("recall_at_k"),
            "all_required_evidence_coverage_at_5": avg("retrieval_coverage_complete"),
            "two_document_coverage_at_5": avg(
                "retrieval_coverage_complete", category("multidoc_two")
            ),
            "three_document_coverage_at_5": avg(
                "retrieval_coverage_complete", category("multidoc_three")
            ),
            "exact_identifier_recall_at_5": avg("recall_at_k", exact),
            "version_sensitive_recall_at_5": avg("recall_at_k", version),
            "near_duplicate_preferred_source_success": avg("retrieval_coverage_complete", near),
        }

    @staticmethod
    def _latency(
        mode: str, cases: tuple[CaseResult, ...], prepared: list[dict[str, Any]]
    ) -> dict[str, Any]:
        total: list[float] = []
        for case, prep in zip(cases, prepared, strict=True):
            timing = prep["timing"]
            retrieval = (
                timing["query_embedding_ms"]
                + timing["cache_lookup_ms"]
                + timing["dense_vector_search_ms"]
                + timing["dense_acl_filter_ms"]
            )
            reranker = (
                timing["reranker_inference_ms"] + timing["reranker_sort_ms"]
                if mode == MODES[1]
                else 0.0
            )
            total.append(case.total_latency_ms + retrieval + reranker)
        rerank = [item["timing"]["reranker_inference_ms"] for item in prepared]
        return {
            "query_embedding_ms": mean(item["timing"]["query_embedding_ms"] for item in prepared),
            "cache_lookup_ms": mean(item["timing"]["cache_lookup_ms"] for item in prepared),
            "dense_retrieval_ms": mean(
                item["timing"]["dense_vector_search_ms"] + item["timing"]["dense_acl_filter_ms"]
                for item in prepared
            ),
            "reranker_inference_mean_ms": mean(rerank) if mode == MODES[1] else 0.0,
            "reranker_inference_p50_ms": median(rerank) if mode == MODES[1] else 0.0,
            "reranker_inference_p95_ms": _percentile(rerank, 0.95) if mode == MODES[1] else 0.0,
            "judge_ms": mean(item.answerability_judge_latency_ms for item in cases),
            "context_construction_ms": mean(item.context_construction_latency_ms for item in cases),
            "generation_ms": mean(item.generation_latency_ms for item in cases),
            "total_mean_ms": mean(total),
            "total_p50_ms": median(total),
            "total_p95_ms": _percentile(total, 0.95),
        }

    @staticmethod
    def _usage(
        mode: str, cases: tuple[CaseResult, ...], record: EndToEndBenchmarkRecord
    ) -> dict[str, Any]:
        external = [item for item in cases if item.external_judge_calls]
        return {
            "new_query_embedding_calls": (
                record.preparation_usage["new_query_embedding_calls"] if mode == MODES[0] else 0
            ),
            "embedding_tokens": (
                record.preparation_usage["embedding_tokens"] if mode == MODES[0] else 0
            ),
            "query_cache_hits": record.preparation_usage["query_cache_hits"],
            "query_cache_misses": record.preparation_usage["query_cache_misses"],
            "new_luna_judge_calls": sum(item.external_judge_calls for item in cases),
            "luna_input_tokens": sum(item.judge_prompt_tokens or 0 for item in external),
            "luna_output_tokens": sum(item.judge_completion_tokens or 0 for item in external),
            "gate_cache_hits": sum(item.gate_cache_hit is True for item in cases),
            "gate_cache_misses": sum(item.gate_cache_hit is False for item in cases),
            "document_embedding_calls": 0,
            "cross_encoder_pairs": (
                record.preparation_usage["cross_encoder_pairs"] if mode == MODES[1] else 0
            ),
            "mean_cross_encoder_pairs": (
                record.preparation_usage["mean_pairs_per_query"] if mode == MODES[1] else 0
            ),
            "external_reranker_calls": 0,
            "unauthorized_chunks_to_reranker": 0,
            "unauthorized_chunks_to_judge": 0,
        }

    def _runs(self) -> dict[str, EndToEndBenchmarkRunRecord]:
        return {
            item.mode: item
            for item in self.session.scalars(
                select(EndToEndBenchmarkRunRecord).where(
                    EndToEndBenchmarkRunRecord.dataset_id == DATASET_ID
                )
            ).all()
        }

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        payload: dict[str, Any] = {
            "dataset_id": DATASET_ID,
            "dataset_hash": DATASET_HASH,
            "case_count": len(DATASET.cases),
            "category_distribution": dict(
                sorted(Counter(case.category for case in DATASET.cases).items())
            ),
            "generation_method": GENERATION_METHOD,
            "maximum_prior_overlap": MAXIMUM_PRIOR_OVERLAP,
            "pipeline_a": PIPELINE_A,
            "pipeline_b": PIPELINE_B,
            "success_policy": SUCCESS_POLICY,
            "embedding_preflight": self.embedding_preflight(),
            "prepared": False,
            "execution_started": False,
            "completed": False,
        }
        if not record:
            return payload
        self._verify_record(record)
        payload.update(
            {
                "frozen_at": record.frozen_at,
                "prepared": record.prepared_at is not None,
                "execution_started": record.execution_started_at is not None,
                "completed": record.completed_at is not None,
                "embedding_preflight_at_freeze": record.embedding_preflight,
                "judge_preflight": record.judge_preflight,
                "preparation_usage": record.preparation_usage,
                "paired_transitions": record.paired_transitions,
                "conversion_analysis": record.conversion_analysis,
                "regression_analysis": record.regression_analysis,
                "production_retriever_status": record.production_retriever_status,
            }
        )
        runs = self._runs()
        if runs:
            payload["results"] = {
                mode: {
                    "id": run.id,
                    "metrics": run.metrics,
                    "retrieval_metrics": run.retrieval_metrics,
                    "category_metrics": run.category_metrics,
                    "latency": run.latency,
                    "usage": run.usage,
                    **({"cases": run.case_results} if include_cases else {}),
                }
                for mode, run in runs.items()
            }
        return payload


def maximum_prior_dataset_overlap() -> float:
    token = re.compile(r"[a-z0-9]+")

    def terms(question: str) -> set[str]:
        return set(token.findall(question.casefold()))

    maximum = 0.0
    for path in Path("data/eval").glob("*.json"):
        if path == DATASET_PATH or path.name == "acmeai_evidence_judge_e2e_eval_v1.json":
            continue
        old_cases = json.loads(path.read_text()).get("cases", [])
        for case in DATASET.cases:
            for old in old_cases:
                left, right = terms(case.question), terms(old["question"])
                maximum = max(maximum, len(left & right) / len(left | right))
    return maximum
