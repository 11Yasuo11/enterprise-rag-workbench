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
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    OpenAICompatibleAnswerabilityGate,
    evidence_coverage_template_hash,
    evidence_gate_schema,
    hosted_judge_chat_payload,
    hosted_judge_request_settings,
)
from rag_workbench.answerability.planning import ExternalJudgeCallLimitGate
from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import (
    AnswerabilityGateCacheRecord,
    Chunk,
    EndToEndBenchmarkRecord,
    EndToEndBenchmarkRunRecord,
    ExperimentRunRecord,
    MultiDocumentBenchmarkRecord,
    QueryEmbeddingCacheRecord,
    RerankingBenchmarkRecord,
)
from rag_workbench.evaluation.datasets import load_evaluation_dataset
from rag_workbench.evaluation.evaluator import CaseResult, EvaluationRunner
from rag_workbench.experiments.judge_e2e_benchmark import (
    SHARED_RETRIEVAL,
    V2_TEMPLATE_HASH,
    FrozenJudgeEndToEndBenchmark,
    _percentile,
)
from rag_workbench.experiments.reranker_e2e_benchmark import (
    CORPUS_IDENTITY,
    RERANKER_REVISION,
    SEMANTIC_INDEX_IDENTITY,
    FixedResultRetriever,
    RerankerEndToEndBenchmark,
    _principal,
    _result,
    _trace,
)
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.generation.generator import RagService
from rag_workbench.providers.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from rag_workbench.providers.llm import ExtractiveGenerationProvider
from rag_workbench.reranking import CrossEncoderReranker, Reranker
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import RetrievalTiming, Retriever

DATASET_ID = "acmeai-sol-judge-e2e-eval-v1"
DATASET_PATH = Path("data/eval/acmeai_sol_judge_e2e_eval_v1.json")
DATASET_HASH = "10c947941644ab9e29ac6e4dc8d59ae87872859cee032eb5b8fb89aaadc2cf55"
MAXIMUM_PRIOR_OVERLAP = 0.4444444444444444
GENERATION_METHOD = "manual-corpus-grounded-v1"
SCHEMA_IDENTITY = hashlib.sha256(
    json.dumps(evidence_gate_schema(), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
HISTORICAL_SCHEMA_IDENTITY = "6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621"
LUNA_MODEL = "gpt-5.6-luna"
SOL_MODEL = "gpt-5.6-sol"
MODES = (
    "GPT_5_6_LUNA_EVIDENCE_SUFFICIENCY_V1",
    "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1",
)
REQUEST_SETTINGS = hosted_judge_request_settings(EVIDENCE_GATE_PROMPT_VERSION)
OFFICIAL_PRICING = {
    "source": {
        "luna": "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
        "sol": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        "pricing": "https://developers.openai.com/api/docs/pricing",
    },
    "context": "short-context <=272K list prices consulted from official OpenAI documentation",
    "luna": {
        "input_per_million": 0.20,
        "cached_input_per_million": 0.02,
        "output_per_million": 1.20,
    },
    "sol": {
        "input_per_million": 5.00,
        "cached_input_per_million": 0.50,
        "output_per_million": 30.00,
    },
}
COMMON_JUDGE = {
    "provider": "openai",
    "judge_version": "1",
    "prompt_version": EVIDENCE_GATE_PROMPT_VERSION,
    "schema_identity": SCHEMA_IDENTITY,
    "supporting_context_only": True,
    "generator": "deterministic-extractive-v1",
    "context_budget": 1200,
    "request_settings": REQUEST_SETTINGS,
}
JUDGE_A = {**COMMON_JUDGE, "model": LUNA_MODEL}
JUDGE_B = {**COMMON_JUDGE, "model": SOL_MODEL}
SUCCESS_POLICY = {
    "minimum_false_negative_fixes": 4,
    "minimum_retrieval_complete_recall_gain": 0.10,
    "maximum_unsupported_answer_increase": 0,
    "overall_f1_must_not_regress": True,
    "correct_answers_must_not_regress": True,
    "hard_constraints": {
        "acl_safety": 1.0,
        "tenant_isolation": 1.0,
        "version_correctness": 1.0,
        "prompt_injection_boundary": 1.0,
        "unauthorized_evidence_selection": 0,
    },
}
DATASET = load_evaluation_dataset(DATASET_PATH)


def request_parameter_parity() -> dict[str, Any]:
    dummy = (
        GateEvidence(
            "chunk",
            "document",
            "version-id",
            "1",
            "evidence",
            SEMANTIC_INDEX_IDENTITY,
        ),
    )
    luna = hosted_judge_chat_payload(
        model=LUNA_MODEL, question="parity", chunks=dummy, provider="openai"
    )
    sol = hosted_judge_chat_payload(
        model=SOL_MODEL, question="parity", chunks=dummy, provider="openai"
    )
    luna_model = luna.pop("model")
    sol_model = sol.pop("model")
    return {
        "matched": luna == sol,
        "luna_model": luna_model,
        "sol_model": sol_model,
        "shared_request_settings": REQUEST_SETTINGS,
        "schema_identity": SCHEMA_IDENTITY,
        "prompt_version": EVIDENCE_GATE_PROMPT_VERSION,
        "sol_pro_mode": False,
        "distinct_reasoning_effort": False,
    }


def official_token_cost(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    prices = OFFICIAL_PRICING["luna" if model == LUNA_MODEL else "sol"]
    uncached = max(input_tokens - cached_input_tokens, 0)
    return (
        uncached * prices["input_per_million"]
        + cached_input_tokens * prices["cached_input_per_million"]
        + output_tokens * prices["output_per_million"]
    ) / 1_000_000


class SolJudgeEndToEndBenchmark:
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
            raise ValueError("sol judge dataset identity changed")
        if hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() != DATASET_HASH:
            raise ValueError("sol judge dataset hash changed")
        if SCHEMA_IDENTITY != HISTORICAL_SCHEMA_IDENTITY:
            raise ValueError("evidence-sufficiency-v1 schema identity changed")
        if not request_parameter_parity()["matched"]:
            raise ValueError("Luna/Sol request-parameter parity cannot be maintained")

    def _verify_dependencies(self) -> None:
        reranker_e2e = self.session.get(EndToEndBenchmarkRecord, "acmeai-reranker-e2e-eval-v1")
        judge_e2e = self.session.get(EndToEndBenchmarkRecord, "acmeai-evidence-judge-e2e-eval-v1")
        reranker = self.session.get(RerankingBenchmarkRecord, "acmeai-reranking-eval-v1")
        v2 = self.session.get(MultiDocumentBenchmarkRecord, "acmeai-multidoc-eval-v2")
        if (
            not reranker_e2e
            or reranker_e2e.production_retriever_status != "DENSE_CROSS_ENCODER_RERANK"
        ):
            raise ValueError("selected production retrieval path changed")
        if not judge_e2e or judge_e2e.production_retriever_status != "EVIDENCE_SUFFICIENCY_V1":
            raise ValueError("historical evidence-sufficiency-v1 production judge changed")
        if (
            not reranker
            or reranker.selected_mode != "DENSE_CROSS_ENCODER_RERANK"
            or reranker.model_revision != RERANKER_REVISION
            or reranker.holdout_completed_at is None
        ):
            raise ValueError("frozen reranker identity changed")
        if (
            not v2
            or v2.candidate_prompt_hash != V2_TEMPLATE_HASH
            or evidence_coverage_template_hash() != V2_TEMPLATE_HASH
            or v2.holdout_completed_at is None
        ):
            raise ValueError("historical evidence-coverage-v2 identity cannot be reproduced")
        if not self.session.scalar(
            select(Chunk.id).where(Chunk.index_identity == SEMANTIC_INDEX_IDENTITY).limit(1)
        ):
            raise ValueError("frozen semantic index is unavailable")

    def initialize(self) -> EndToEndBenchmarkRecord:
        self._verify_dependencies()
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
            pipeline_a_configuration={
                **SHARED_RETRIEVAL,
                **JUDGE_A,
                "success_policy": SUCCESS_POLICY,
                "official_pricing": OFFICIAL_PRICING,
            },
            pipeline_b_configuration={
                **SHARED_RETRIEVAL,
                **JUDGE_B,
                "success_policy": SUCCESS_POLICY,
                "official_pricing": OFFICIAL_PRICING,
            },
        )
        self.session.add(record)
        self.session.commit()
        return record

    @staticmethod
    def _verify_record(record: EndToEndBenchmarkRecord) -> None:
        expected_a = {
            **SHARED_RETRIEVAL,
            **JUDGE_A,
            "success_policy": SUCCESS_POLICY,
            "official_pricing": OFFICIAL_PRICING,
        }
        expected_b = {
            **SHARED_RETRIEVAL,
            **JUDGE_B,
            "success_policy": SUCCESS_POLICY,
            "official_pricing": OFFICIAL_PRICING,
        }
        if not (
            record.dataset_hash == DATASET_HASH
            and record.case_ids == [case.case_id for case in DATASET.cases]
            and record.maximum_prior_overlap == MAXIMUM_PRIOR_OVERLAP
            and record.pipeline_a_configuration == expected_a
            and record.pipeline_b_configuration == expected_b
            and record.semantic_index_identity == SEMANTIC_INDEX_IDENTITY
            and record.corpus_identity == CORPUS_IDENTITY
            and record.reranker_revision == RERANKER_REVISION
        ):
            raise ValueError("sealed sol judge benchmark identity changed")

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
        questions = tuple(dict.fromkeys(case.question for case in DATASET.cases))
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
        return {
            "current_cumulative_embedding_calls": current,
            "configured_ceiling": self.settings.max_external_embedding_calls,
            "dataset_unique_queries": len(questions),
            "existing_cache_matches": matches,
            "missing_embeddings": len(keys) - matches,
            "maximum_new_embedding_calls": len(keys) - matches,
            "expected_cumulative_ending_usage": current + len(keys) - matches,
        }

    def prepare(self) -> dict[str, Any]:
        record = self.initialize()
        if record.preparation_started_at is not None:
            raise ValueError("sealed sol judge preparation is one-shot and already started")
        preflight = self.embedding_preflight()
        if not self.settings.allow_external_calls or not self.settings.embedding_api_key:
            raise ValueError("external embedding authorization is incomplete")
        if preflight["expected_cumulative_ending_usage"] > preflight["configured_ceiling"]:
            raise ValueError("sol judge dataset exceeds the embedding-call ceiling")
        record.embedding_preflight = preflight
        record.preparation_started_at = datetime.now(UTC)
        self.session.commit()
        provider = self._embedding_provider()
        dense = Retriever(self.session, provider, SEMANTIC_INDEX_IDENTITY)
        reranker: Reranker = self.reranker_factory()
        if reranker.resolved_revision != RERANKER_REVISION:
            raise ValueError("local reranker revision differs from frozen revision")
        prepared: list[dict[str, Any]] = []
        calls = tokens = hits = misses = pairs = unauthorized = 0
        for case in DATASET.cases:
            embedding = dense.query_embedding_cache.get_or_embed(case.question)
            self.session.commit()
            candidates = dense.retrieve_with_embedding(
                embedding, top_k=20, score_threshold=0.28, principal=_principal(case)
            )
            forbidden = set(case.forbidden_document_ids)
            unauthorized += sum(item.document_id in forbidden for item in candidates)
            reranked = reranker.rerank(case.question, candidates)
            top5 = [
                replace(
                    item.result,
                    rank=item.reranked_rank,
                    score=item.reranker_score,
                    retrieval_source="dense_cross_encoder_rerank",
                )
                for item in reranked[:5]
            ]
            prepared.append(
                {
                    "case_id": case.case_id,
                    "dense_candidates": [_trace(item) for item in candidates],
                    "shared_reranked_top5": [_trace(item) for item in top5],
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
                        "dense_ms": dense.last_timing.vector_search_latency_ms
                        + dense.last_timing.acl_filter_latency_ms,
                        "reranker_ms": getattr(reranker, "last_inference_latency_ms", 0.0)
                        + getattr(reranker, "last_sort_latency_ms", 0.0),
                    },
                    "embedding_cache_hit": embedding.cache_hit,
                    "embedding_external_calls": embedding.external_calls,
                    "embedding_tokens": embedding.input_tokens,
                }
            )
            calls += embedding.external_calls
            tokens += embedding.input_tokens
            hits += int(embedding.cache_hit)
            misses += int(not embedding.cache_hit)
            pairs += len(candidates)
        if unauthorized:
            raise RuntimeError("unauthorized content reached frozen Cross-Encoder")
        record = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        record.prepared_cases = prepared
        record.preparation_usage = {
            "new_query_embedding_calls": calls,
            "embedding_tokens": tokens,
            "query_cache_hits": hits,
            "query_cache_misses": misses,
            "document_embedding_calls": 0,
            "cross_encoder_pairs": pairs,
            "mean_pairs_per_query": pairs / len(DATASET.cases),
            "unauthorized_chunks_to_reranker": 0,
        }
        record.prepared_at = datetime.now(UTC)
        self.session.commit()
        return self.status()

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
        return historical + specialized

    def _mode_model(self, mode: str) -> str:
        return LUNA_MODEL if mode == MODES[0] else SOL_MODEL

    def judge_preflight(self, *, persist: bool = True) -> dict[str, Any]:
        record = self.initialize()
        if not record.prepared_cases or not record.prepared_at:
            raise ValueError("shared frozen retrieval must be prepared before judge preflight")
        by_mode: dict[str, list[str]] = {mode: [] for mode in MODES}
        for case, prepared in zip(DATASET.cases, record.prepared_cases, strict=True):
            evidence = tuple(
                RerankerEndToEndBenchmark._gate_evidence(prepared["shared_reranked_top5"])
            )
            for mode in MODES:
                by_mode[mode].append(
                    gate_cache_key(
                        case.question,
                        evidence,
                        provider="openai",
                        model=self._mode_model(mode),
                        gate_version="1",
                        prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
                    )[0]
                )
        sets = {mode: set(keys) for mode, keys in by_mode.items()}
        matches = {
            mode: sum(
                self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in keys
            )
            for mode, keys in sets.items()
        }
        current = self._historical_judge_calls()
        luna_missing = len(sets[MODES[0]]) - matches[MODES[0]]
        sol_missing = len(sets[MODES[1]]) - matches[MODES[1]]
        historical_tokens = 38265 / 60, 4625 / 60
        projected_luna = official_token_cost(
            model=LUNA_MODEL,
            input_tokens=int(historical_tokens[0] * luna_missing),
            output_tokens=int(historical_tokens[1] * luna_missing),
        )
        projected_sol = official_token_cost(
            model=SOL_MODEL,
            input_tokens=int(historical_tokens[0] * sol_missing),
            output_tokens=int(historical_tokens[1] * sol_missing),
        )
        preflight = {
            "current_cumulative_judge_calls": current,
            "configured_ceiling": self.settings.max_external_judge_calls,
            "unique_luna_gate_inputs": len(sets[MODES[0]]),
            "unique_sol_gate_inputs": len(sets[MODES[1]]),
            "existing_luna_cache_matches": matches[MODES[0]],
            "existing_sol_cache_matches": matches[MODES[1]],
            "cross_model_shared_cache_keys": len(sets[MODES[0]] & sets[MODES[1]]),
            "new_luna_calls_required": luna_missing,
            "new_sol_calls_required": sol_missing,
            "maximum_new_external_calls": luna_missing + sol_missing,
            "expected_worst_case_cumulative_calls": current + luna_missing + sol_missing,
            "request_parameter_parity": request_parameter_parity(),
            "official_cost_projection": {
                "basis": "historical evidence-sufficiency-v1 mean tokens from prior 60 Luna cases",
                "historical_mean_input_tokens": historical_tokens[0],
                "historical_mean_output_tokens": historical_tokens[1],
                "projected_luna_cost_usd": projected_luna,
                "projected_sol_cost_usd": projected_sol,
                "projected_maximum_usd": projected_luna + projected_sol,
                "projected_range_usd": [
                    round((projected_luna + projected_sol) * 0.8, 6),
                    round((projected_luna + projected_sol) * 2.5, 6),
                ],
            },
        }
        if persist:
            record.judge_preflight = preflight
            self.session.commit()
        return preflight

    def _gate(self, model: str, maximum_calls: int) -> CachedAnswerabilityGate:
        if self.gate_factory:
            return self.gate_factory(model, maximum_calls)
        delegate = OpenAICompatibleAnswerabilityGate(
            api_key=self.settings.effective_judge_api_key or "",
            model=model,
            base_url=self.settings.judge_base_url,
            gate_version="1",
            prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
            provider_name="openai",
        )
        return CachedAnswerabilityGate(
            self.session, ExternalJudgeCallLimitGate(delegate, maximum_calls)
        )

    def execute(self) -> dict[str, Any]:
        record = self.initialize()
        if record.execution_started_at is not None:
            raise ValueError("sealed sol judge evaluation is one-shot and already started")
        preflight = self.judge_preflight()
        if not preflight["request_parameter_parity"]["matched"]:
            raise ValueError("request-parameter parity cannot be maintained")
        if (
            not self.settings.allow_external_judge_calls
            or not self.settings.effective_judge_api_key
        ):
            raise ValueError("hosted judge authorization is incomplete")
        if preflight["expected_worst_case_cumulative_calls"] > preflight["configured_ceiling"]:
            raise ValueError("paired judge inputs exceed hosted judge-call ceiling")
        record.execution_started_at = datetime.now(UTC)
        self.session.commit()
        gates = {
            MODES[0]: self._gate(LUNA_MODEL, 60),
            MODES[1]: self._gate(SOL_MODEL, 60),
        }
        extra_usage = {
            mode: {
                "reasoning_tokens": 0,
                "cached_prompt_tokens": 0,
                "resolved_models": [],
            }
            for mode in MODES
        }
        provider = self._embedding_provider()
        results: dict[str, list[CaseResult]] = {mode: [] for mode in MODES}
        unauthorized = 0
        for case, prepared in zip(DATASET.cases, record.prepared_cases or [], strict=True):
            retrieved = [_result(item) for item in prepared["shared_reranked_top5"]]
            unauthorized += sum(
                item.document_id in set(case.forbidden_document_ids) for item in retrieved
            )
            timing = RetrievalTiming(
                query_embedding_latency_ms=prepared["timing"]["query_embedding_ms"],
                embedding_cache_lookup_latency_ms=prepared["timing"]["cache_lookup_ms"],
                vector_search_latency_ms=prepared["timing"]["dense_ms"],
                acl_filter_latency_ms=0.0,
                query_embedding_cache_hit=prepared["embedding_cache_hit"],
                external_embedding_calls=prepared["embedding_external_calls"],
            )
            for mode in MODES:
                rag = RagService(
                    self.session,
                    FixedResultRetriever(provider, retrieved, timing),
                    ContextBuilder(1200),
                    ExtractiveGenerationProvider(),
                    gates[mode],
                    supporting_context_only=True,
                )
                results[mode].append(EvaluationRunner(rag).run_case(case, 5, 0.28))
                extra = extra_usage[mode]
                extra["reasoning_tokens"] += gates[mode].last_timing.reasoning_tokens or 0
                extra["cached_prompt_tokens"] += (
                    gates[mode].last_timing.cached_prompt_tokens or 0
                )
                resolved = gates[mode].last_timing.resolved_model
                if resolved and resolved not in extra["resolved_models"]:
                    extra["resolved_models"].append(resolved)
        if unauthorized:
            raise RuntimeError("unauthorized content reached one or both judges")
        reporter = EvaluationRunner(rag)
        reports = {mode: reporter._report(tuple(rows)) for mode, rows in results.items()}
        analysis = self._paired_analysis(results)
        for mode in MODES:
            report = reports[mode]
            self.session.add(
                EndToEndBenchmarkRunRecord(
                    dataset_id=DATASET_ID,
                    mode=mode,
                    metrics=report.metrics,
                    retrieval_metrics=self._subset_metrics(report.cases),
                    category_metrics=report.category_metrics,
                    case_results=[
                        self._case_payload(case, mode, record.prepared_cases or [])
                        for case in report.cases
                    ],
                    latency=self._latency(mode, report.cases, record.prepared_cases or []),
                    usage=self._usage(mode, report.cases, record, extra_usage[mode]),
                )
            )
        a = reports[MODES[0]].metrics
        b = reports[MODES[1]].metrics
        a_complete = FrozenJudgeEndToEndBenchmark._aggregate_subset(reports[MODES[0]].cases, True)
        b_complete = FrozenJudgeEndToEndBenchmark._aggregate_subset(reports[MODES[1]].cases, True)
        material = (
            analysis["false_negative_rescues"] >= 4
            or b_complete["answerability_recall"] - a_complete["answerability_recall"] >= 0.10
        )
        safe = (
            b["unsupported_answer_count"] <= a["unsupported_answer_count"]
            and b["answerability_f1"] >= a["answerability_f1"]
            and b["correct_answer_count"] >= a["correct_answer_count"]
            and b["acl_safety"] == 1.0
            and b["version_accuracy"] == 1.0
            and b["prompt_injection_boundary"] == 1.0
            and b["unauthorized_evidence_selection"] == 0
        )
        record = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        record.paired_transitions = analysis["transitions"]
        record.conversion_analysis = {
            **analysis,
            "category_analysis": self._category_analysis(results),
            "supporting_evidence": {
                mode: self._supporting_evidence(report.cases)
                for mode, report in reports.items()
            },
        }
        record.regression_analysis = analysis["false_positive_regressions"]
        record.production_retriever_status = MODES[1] if material and safe else MODES[0]
        record.completed_at = datetime.now(UTC)
        self.session.commit()
        return self.status(include_cases=True)

    @classmethod
    def _paired_analysis(cls, results: dict[str, list[CaseResult]]) -> dict[str, Any]:
        a = {case.case_id: case for case in results[MODES[0]]}
        b = {case.case_id: case for case in results[MODES[1]]}
        transitions: Counter[str] = Counter()
        rescues: list[str] = []
        regressions: list[str] = []
        incomplete_answers: list[str] = []
        luna_correct_to_sol_fn: list[str] = []
        for case_id, left in a.items():
            right = b[case_id]
            left_behavior = FrozenJudgeEndToEndBenchmark._behavior(left)
            right_behavior = FrozenJudgeEndToEndBenchmark._behavior(right)
            transitions[f"{left_behavior}→{right_behavior}"] += 1
            if (
                left.retrieval_coverage_complete
                and FrozenJudgeEndToEndBenchmark._behavior(left) == "INCORRECT_ABSTENTION"
                and FrozenJudgeEndToEndBenchmark._behavior(right) == "CORRECT_ANSWER"
            ):
                rescues.append(case_id)
            if (
                left.retrieval_coverage_complete
                and FrozenJudgeEndToEndBenchmark._behavior(left) == "CORRECT_ANSWER"
                and FrozenJudgeEndToEndBenchmark._behavior(right) == "INCORRECT_ABSTENTION"
            ):
                luna_correct_to_sol_fn.append(case_id)
            if (
                FrozenJudgeEndToEndBenchmark._behavior(left) == "CORRECT_ANSWER"
                and FrozenJudgeEndToEndBenchmark._behavior(right) == "INCORRECT_ABSTENTION"
            ):
                regressions.append(case_id)
            if not right.retrieval_coverage_complete and right.status == "answered":
                incomplete_answers.append(case_id)
        false_positive = [
            case_id
            for case_id, left in a.items()
            if FrozenJudgeEndToEndBenchmark._behavior(left) == "CORRECT_ABSTENTION"
            and FrozenJudgeEndToEndBenchmark._behavior(b[case_id]) == "UNSUPPORTED_ANSWER"
        ]
        return {
            "transitions": dict(sorted(transitions.items())),
            "a_false_negatives": sum(
                FrozenJudgeEndToEndBenchmark._behavior(case) == "INCORRECT_ABSTENTION"
                for case in a.values()
            ),
            "b_false_negatives": sum(
                FrozenJudgeEndToEndBenchmark._behavior(case) == "INCORRECT_ABSTENTION"
                for case in b.values()
            ),
            "false_negative_rescues": len(rescues),
            "false_negative_rescue_case_ids": rescues,
            "a_correct_to_b_false_abstention": len(regressions),
            "judge_regression_case_ids": regressions,
            "luna_correct_to_sol_false_abstention": len(luna_correct_to_sol_fn),
            "luna_correct_to_sol_false_abstention_case_ids": luna_correct_to_sol_fn,
            "net_false_negative_improvement": len(rescues) - len(luna_correct_to_sol_fn),
            "false_positive_regressions": {
                "count": len(false_positive),
                "case_ids": false_positive,
                "answers_despite_incomplete_retrieval": incomplete_answers,
            },
        }

    @classmethod
    def _subset_metrics(cls, cases: tuple[CaseResult, ...]) -> dict[str, Any]:
        return FrozenJudgeEndToEndBenchmark._subset_metrics(cases)

    @classmethod
    def _case_payload(
        cls, case: CaseResult, mode: str, prepared_cases: list[dict[str, Any]]
    ) -> dict[str, Any]:
        payload = asdict(case)
        prepared = next(item for item in prepared_cases if item["case_id"] == case.case_id)
        payload["shared_retrieval_trace"] = prepared["shared_reranked_top5"]
        payload["dense_candidate_trace"] = prepared["dense_candidates"]
        payload["reranking_trace"] = prepared["reranking_trace"]
        payload["root_cause"] = cls._root_cause(case, mode)
        payload["final_behavior"] = FrozenJudgeEndToEndBenchmark._behavior(case)
        payload["judge_model"] = LUNA_MODEL if mode == MODES[0] else SOL_MODEL
        return payload

    @staticmethod
    def _root_cause(case: CaseResult, mode: str) -> str:
        judge_answerable = bool((case.answerability_result or {}).get("answerable"))
        error = case.answerability_operational_error
        if error == "INVALID_SUPPORTING_ID":
            return "INVALID_SUPPORTING_ID"
        if error == "UNAUTHORIZED_SUPPORTING_ID":
            return "ACL_FAILURE"
        if error in {"INACTIVE_VERSION", "VERSION_MISMATCH"}:
            return "VERSION_FAILURE"
        if not case.retrieval_coverage_complete and not case.expected_abstain:
            return "RETRIEVAL_INCOMPLETE"
        if case.status == "answered" and case.expected_abstain:
            return (
                "LUNA_EVIDENCE_GATE_FALSE_POSITIVE"
                if mode == MODES[0]
                else "SOL_EVIDENCE_GATE_FALSE_POSITIVE"
            )
        if case.status == "abstained" and not case.expected_abstain:
            if judge_answerable:
                return "GENERATION_FAILURE"
            return (
                "LUNA_EVIDENCE_GATE_FALSE_NEGATIVE"
                if mode == MODES[0]
                else "SOL_EVIDENCE_GATE_FALSE_NEGATIVE"
            )
        if case.supporting_context_loss:
            return "SUPPORTING_CONTEXT_LOSS"
        return "UNKNOWN" if case.failure_types else "NONE"

    @staticmethod
    def _latency(
        mode: str, cases: tuple[CaseResult, ...], prepared: list[dict[str, Any]]
    ) -> dict[str, Any]:
        shared = [
            item["timing"]["query_embedding_ms"]
            + item["timing"]["cache_lookup_ms"]
            + item["timing"]["dense_ms"]
            + item["timing"]["reranker_ms"]
            for item in prepared
        ]
        totals = [shared[index] + case.total_latency_ms for index, case in enumerate(cases)]
        judge = [case.answerability_judge_latency_ms for case in cases]
        validation = [
            case.context_pruning_latency_ms - case.answerability_judge_latency_ms for case in cases
        ]
        return {
            "shared_query_embedding_ms": mean(p["timing"]["query_embedding_ms"] for p in prepared),
            "shared_cache_lookup_ms": mean(p["timing"]["cache_lookup_ms"] for p in prepared),
            "shared_dense_ms": mean(p["timing"]["dense_ms"] for p in prepared),
            "shared_reranker_ms": mean(p["timing"]["reranker_ms"] for p in prepared),
            "judge_mean_ms": mean(judge),
            "judge_p50_ms": median(judge),
            "judge_p95_ms": _percentile(judge, 0.95),
            "validation_context_mean_ms": mean(validation),
            "generation_mean_ms": mean(case.generation_latency_ms for case in cases),
            "total_mean_ms": mean(totals),
            "total_p50_ms": median(totals),
            "total_p95_ms": _percentile(totals, 0.95),
            "model": LUNA_MODEL if mode == MODES[0] else SOL_MODEL,
            "prompt_version": EVIDENCE_GATE_PROMPT_VERSION,
        }

    @staticmethod
    def _usage(
        mode: str,
        cases: tuple[CaseResult, ...],
        record: EndToEndBenchmarkRecord,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        external = [case for case in cases if case.external_judge_calls]
        input_tokens = sum(case.judge_prompt_tokens or 0 for case in external)
        output_tokens = sum(case.judge_completion_tokens or 0 for case in external)
        model = LUNA_MODEL if mode == MODES[0] else SOL_MODEL
        prefix = "luna" if mode == MODES[0] else "sol"
        measured_cost = official_token_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=extra["cached_prompt_tokens"],
        )
        return {
            "new_query_embedding_calls": record.preparation_usage["new_query_embedding_calls"]
            if mode == MODES[0]
            else 0,
            "embedding_tokens": record.preparation_usage["embedding_tokens"]
            if mode == MODES[0]
            else 0,
            f"new_{prefix}_judge_calls": sum(case.external_judge_calls for case in cases),
            "new_luna_judge_calls": sum(case.external_judge_calls for case in cases)
            if mode == MODES[0]
            else 0,
            "new_sol_judge_calls": sum(case.external_judge_calls for case in cases)
            if mode == MODES[1]
            else 0,
            f"{prefix}_input_tokens": input_tokens,
            f"{prefix}_output_tokens": output_tokens,
            f"{prefix}_reasoning_tokens": extra["reasoning_tokens"],
            f"{prefix}_cached_input_tokens": extra["cached_prompt_tokens"],
            "resolved_models": extra["resolved_models"],
            "official_measured_cost_usd": measured_cost,
            "official_cost_per_case_usd": measured_cost / len(cases) if cases else 0.0,
            "gate_cache_hits": sum(case.gate_cache_hit is True for case in cases),
            "gate_cache_misses": sum(case.gate_cache_hit is False for case in cases),
            "cross_encoder_pairs": record.preparation_usage["cross_encoder_pairs"],
            "document_embedding_calls": 0,
            "external_reranker_calls": 0,
            "unauthorized_chunks_to_reranker": 0,
            "unauthorized_chunks_to_judge": 0,
        }

    @staticmethod
    def _supporting_evidence(cases: tuple[CaseResult, ...]) -> dict[str, Any]:
        answerable = [case for case in cases if not case.expected_abstain]
        complete = [case for case in answerable if case.retrieval_coverage_complete]
        empty = [
            case.case_id
            for case in complete
            if not case.supporting_chunk_ids
            and not bool((case.answerability_result or {}).get("answerable"))
        ]
        invented = [
            case.case_id
            for case in cases
            if case.answerability_operational_error == "INVALID_SUPPORTING_ID"
        ]
        return {
            "required_evidence_precision": EvaluationRunner._aggregate(cases).get(
                "required_evidence_precision"
            ),
            "required_evidence_recall": EvaluationRunner._aggregate(cases).get(
                "required_evidence_recall"
            ),
            "empty_support_despite_answerable_evidence": len(empty),
            "empty_support_case_ids": empty,
            "invented_or_invalid_supporting_ids": len(invented),
            "invalid_supporting_id_case_ids": invented,
        }

    @classmethod
    def _category_analysis(cls, results: dict[str, list[CaseResult]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for category in (
            "multidoc_two",
            "multidoc_three",
            "exact_identifier",
            "version_region",
            "near_duplicate",
            "acl_sensitive",
            "partial_no_answer",
        ):
            payload[category] = {
                mode: cls._category_metrics(
                    tuple(case for case in cases if case.category == category)
                )
                for mode, cases in results.items()
            }
        three = {
            mode: [case for case in cases if case.category == "multidoc_three"]
            for mode, cases in results.items()
        }
        payload["three_document_traces"] = [
            {
                "case_id": left.case_id,
                "retrieval_complete": left.retrieval_coverage_complete,
                "required_document_ids": list(left.expected_document_ids),
                "luna_judge_decision": (left.answerability_result or {}).get("answerable"),
                "sol_judge_decision": (right.answerability_result or {}).get("answerable"),
                "luna_supporting_ids": list(left.supporting_chunk_ids),
                "sol_supporting_ids": list(right.supporting_chunk_ids),
                "luna_final_behavior": FrozenJudgeEndToEndBenchmark._behavior(left),
                "sol_final_behavior": FrozenJudgeEndToEndBenchmark._behavior(right),
            }
            for left, right in zip(three[MODES[0]], three[MODES[1]], strict=True)
        ]
        return payload

    @staticmethod
    def _category_metrics(cases: tuple[CaseResult, ...]) -> dict[str, Any]:
        if not cases:
            return {}
        metrics = EvaluationRunner._aggregate(cases)
        complete = tuple(case for case in cases if case.retrieval_coverage_complete)
        return {
            "case_count": len(cases),
            "retrieval_complete_rate": mean(case.retrieval_coverage_complete for case in cases),
            "correct_answer_rate": metrics["correct_answer_count"] / len(cases),
            "incorrect_abstention_rate": metrics["incorrect_abstention_count"] / len(cases),
            "unsupported_answer_rate": metrics["unsupported_answer_count"] / len(cases),
            "answerability_f1": metrics["answerability_f1"],
            "judge_decision_retrieval_complete": (
                FrozenJudgeEndToEndBenchmark._judge_decision_metrics(complete)
            ),
        }

    def _runs(self) -> dict[str, EndToEndBenchmarkRunRecord]:
        return {
            run.mode: run
            for run in self.session.scalars(
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
                sorted(Counter(c.category for c in DATASET.cases).items())
            ),
            "maximum_prior_overlap": MAXIMUM_PRIOR_OVERLAP,
            "shared_retrieval": SHARED_RETRIEVAL,
            "judge_a": JUDGE_A,
            "judge_b": JUDGE_B,
            "success_policy": SUCCESS_POLICY,
            "request_parameter_parity": request_parameter_parity(),
            "official_pricing": OFFICIAL_PRICING,
            "production_retriever_status": "DENSE_CROSS_ENCODER_RERANK",
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
                "analysis": record.conversion_analysis,
                "false_positive_regressions": record.regression_analysis,
                "production_judge_status": record.production_retriever_status,
            }
        )
        runs = self._runs()
        if runs:
            payload["results"] = {
                mode: {
                    "id": run.id,
                    "metrics": run.metrics,
                    "subset_metrics": run.retrieval_metrics,
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
        if path == DATASET_PATH:
            continue
        for previous in json.loads(path.read_text()).get("cases", []):
            for case in DATASET.cases:
                left, right = terms(case.question), terms(previous["question"])
                maximum = max(maximum, len(left & right) / len(left | right))
    return maximum
