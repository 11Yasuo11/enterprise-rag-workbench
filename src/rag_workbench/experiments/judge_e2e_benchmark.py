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

from rag_workbench.answerability.cache import CachedAnswerabilityGate, gate_cache_key
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_COVERAGE_PROMPT_VERSION,
    EVIDENCE_GATE_PROMPT_VERSION,
    OpenAICompatibleAnswerabilityGate,
    evidence_coverage_template_hash,
    evidence_gate_schema,
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

DATASET_ID = "acmeai-evidence-judge-e2e-eval-v1"
DATASET_PATH = Path("data/eval/acmeai_evidence_judge_e2e_eval_v1.json")
DATASET_HASH = "7d9b0e6f45e7d385be92ce5c37d229a1602957316e85bfe0c5f828cf514f9651"
MAXIMUM_PRIOR_OVERLAP = 0.47058823529411764
GENERATION_METHOD = "manual-corpus-grounded-v1"
MODES = ("EVIDENCE_SUFFICIENCY_V1", "EVIDENCE_COVERAGE_V2")
V2_TEMPLATE_HASH = "dbcce103d21a48254c69819959297fbf981464fb99b1760e5f01d316a34a9940"
SHARED_RETRIEVAL = {
    "embedding_model": "text-embedding-3-small",
    "embedding_dimension": 64,
    "dense_backend": "pgvector_cosine",
    "dense_threshold": 0.28,
    "dense_candidate_depth": 20,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "reranker_revision": RERANKER_REVISION,
    "reranker_device": "cpu",
    "final_top_k": 5,
}
COMMON_JUDGE = {
    "provider": "openai",
    "model": "gpt-5.6-luna",
    "judge_version": "1",
    "supporting_context_only": True,
    "generator": "deterministic-extractive-v1",
    "context_budget": 1200,
}
JUDGE_A = {
    **COMMON_JUDGE,
    "prompt_version": EVIDENCE_GATE_PROMPT_VERSION,
    "schema_identity": hashlib.sha256(
        json.dumps(evidence_gate_schema(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest(),
}
JUDGE_B = {
    **COMMON_JUDGE,
    "prompt_version": EVIDENCE_COVERAGE_PROMPT_VERSION,
    "template_hash": V2_TEMPLATE_HASH,
    "statuses": ["SUPPORTED", "PARTIAL", "MISSING", "CONFLICTING"],
    "all_requirements_supported": True,
    "supporting_chunk_union_required": True,
}
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


def _percentile(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * value)]


class FrozenJudgeEndToEndBenchmark:
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
            raise ValueError("judge end-to-end dataset identity changed")
        if hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() != DATASET_HASH:
            raise ValueError("judge end-to-end dataset hash changed")

    def _verify_dependencies(self) -> None:
        previous = self.session.get(EndToEndBenchmarkRecord, "acmeai-reranker-e2e-eval-v1")
        reranker = self.session.get(RerankingBenchmarkRecord, "acmeai-reranking-eval-v1")
        v2 = self.session.get(MultiDocumentBenchmarkRecord, "acmeai-multidoc-eval-v2")
        if not previous or previous.production_retriever_status != "DENSE_CROSS_ENCODER_RERANK":
            raise ValueError("selected production retrieval path changed")
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
            },
            pipeline_b_configuration={
                **SHARED_RETRIEVAL,
                **JUDGE_B,
                "success_policy": SUCCESS_POLICY,
            },
        )
        self.session.add(record)
        self.session.commit()
        return record

    @staticmethod
    def _verify_record(record: EndToEndBenchmarkRecord) -> None:
        expected_a = {**SHARED_RETRIEVAL, **JUDGE_A, "success_policy": SUCCESS_POLICY}
        expected_b = {**SHARED_RETRIEVAL, **JUDGE_B, "success_policy": SUCCESS_POLICY}
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
            raise ValueError("sealed judge end-to-end benchmark identity changed")

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
            raise ValueError("sealed judge benchmark preparation is one-shot and already started")
        preflight = self.embedding_preflight()
        if not self.settings.allow_external_calls or not self.settings.embedding_api_key:
            raise ValueError("external embedding authorization is incomplete")
        if preflight["expected_cumulative_ending_usage"] > preflight["configured_ceiling"]:
            raise ValueError("judge dataset exceeds the embedding-call ceiling")
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
        specialized = sum(
            (item.usage or {}).get("new_luna_judge_calls", 0)
            for item in self.session.scalars(select(EndToEndBenchmarkRunRecord)).all()
        )
        return historical + specialized

    def judge_preflight(self, *, persist: bool = True) -> dict[str, Any]:
        record = self.initialize()
        if not record.prepared_cases or not record.prepared_at:
            raise ValueError("shared frozen retrieval must be prepared before judge preflight")
        by_mode: dict[str, list[str]] = {mode: [] for mode in MODES}
        for case, prepared in zip(DATASET.cases, record.prepared_cases, strict=True):
            evidence = tuple(
                RerankerEndToEndBenchmark._gate_evidence(prepared["shared_reranked_top5"])
            )
            for mode, prompt in zip(
                MODES,
                (EVIDENCE_GATE_PROMPT_VERSION, EVIDENCE_COVERAGE_PROMPT_VERSION),
                strict=True,
            ):
                by_mode[mode].append(
                    gate_cache_key(
                        case.question,
                        evidence,
                        provider="openai",
                        model="gpt-5.6-luna",
                        gate_version="1",
                        prompt_version=prompt,
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
        missing = sum(len(sets[mode]) - matches[mode] for mode in MODES)
        preflight = {
            "current_cumulative_judge_calls": current,
            "configured_ceiling": self.settings.max_external_judge_calls,
            "unique_a_gate_inputs": len(sets[MODES[0]]),
            "unique_b_gate_inputs": len(sets[MODES[1]]),
            "existing_a_cache_matches": matches[MODES[0]],
            "existing_b_cache_matches": matches[MODES[1]],
            "cross_prompt_shared_cache_keys": len(sets[MODES[0]] & sets[MODES[1]]),
            "maximum_new_external_calls": missing,
            "expected_worst_case_cumulative_calls": current + missing,
        }
        if persist:
            record.judge_preflight = preflight
            self.session.commit()
        return preflight

    def _gate(self, prompt_version: str, maximum_calls: int) -> CachedAnswerabilityGate:
        if self.gate_factory:
            return self.gate_factory(prompt_version, maximum_calls)
        delegate = OpenAICompatibleAnswerabilityGate(
            api_key=self.settings.effective_judge_api_key or "",
            model="gpt-5.6-luna",
            base_url=self.settings.judge_base_url,
            gate_version="1",
            prompt_version=prompt_version,
            provider_name="openai",
        )
        return CachedAnswerabilityGate(
            self.session, ExternalJudgeCallLimitGate(delegate, maximum_calls)
        )

    def execute(self) -> dict[str, Any]:
        record = self.initialize()
        if record.execution_started_at is not None:
            raise ValueError("sealed judge evaluation is one-shot and already started")
        preflight = self.judge_preflight()
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
            MODES[0]: self._gate(EVIDENCE_GATE_PROMPT_VERSION, 60),
            MODES[1]: self._gate(EVIDENCE_COVERAGE_PROMPT_VERSION, 60),
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
        if unauthorized:
            raise RuntimeError("unauthorized content reached one or both judges")
        reporter = EvaluationRunner(rag)
        reports = {mode: reporter._report(tuple(rows)) for mode, rows in results.items()}
        analysis = self._paired_analysis(results)
        diagnostics = self._requirement_diagnostics(reports[MODES[1]].cases)
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
                    usage=self._usage(mode, report.cases, record),
                )
            )
        a = reports[MODES[0]].metrics
        b = reports[MODES[1]].metrics
        a_complete = self._aggregate_subset(reports[MODES[0]].cases, True)
        b_complete = self._aggregate_subset(reports[MODES[1]].cases, True)
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
        record.conversion_analysis = {**analysis, "requirement_diagnostics": diagnostics}
        record.regression_analysis = analysis["false_positive_regressions"]
        record.production_retriever_status = (
            "EVIDENCE_COVERAGE_V2" if material and safe else "EVIDENCE_SUFFICIENCY_V1"
        )
        record.completed_at = datetime.now(UTC)
        self.session.commit()
        return self.status(include_cases=True)

    @staticmethod
    def _behavior(case: CaseResult) -> str:
        if case.expected_abstain:
            return "CORRECT_ABSTENTION" if case.status == "abstained" else "UNSUPPORTED_ANSWER"
        return "CORRECT_ANSWER" if case.status == "answered" else "INCORRECT_ABSTENTION"

    @classmethod
    def _paired_analysis(cls, results: dict[str, list[CaseResult]]) -> dict[str, Any]:
        a = {case.case_id: case for case in results[MODES[0]]}
        b = {case.case_id: case for case in results[MODES[1]]}
        transitions: Counter[str] = Counter()
        rescues: list[str] = []
        regressions: list[str] = []
        incomplete_answers: list[str] = []
        for case_id, left in a.items():
            right = b[case_id]
            transitions[f"{cls._behavior(left)}→{cls._behavior(right)}"] += 1
            if (
                left.retrieval_coverage_complete
                and cls._behavior(left) == "INCORRECT_ABSTENTION"
                and cls._behavior(right) == "CORRECT_ANSWER"
            ):
                rescues.append(case_id)
            if (
                cls._behavior(left) == "CORRECT_ANSWER"
                and cls._behavior(right) == "INCORRECT_ABSTENTION"
            ):
                regressions.append(case_id)
            if not right.retrieval_coverage_complete and right.status == "answered":
                incomplete_answers.append(case_id)
        false_positive = [
            case_id
            for case_id, left in a.items()
            if cls._behavior(left) == "CORRECT_ABSTENTION"
            and cls._behavior(b[case_id]) == "UNSUPPORTED_ANSWER"
        ]
        return {
            "transitions": dict(sorted(transitions.items())),
            "a_false_negatives": sum(
                cls._behavior(case) == "INCORRECT_ABSTENTION" for case in a.values()
            ),
            "b_false_negatives": sum(
                cls._behavior(case) == "INCORRECT_ABSTENTION" for case in b.values()
            ),
            "false_negative_rescues": len(rescues),
            "false_negative_rescue_case_ids": rescues,
            "a_correct_to_b_false_abstention": len(regressions),
            "judge_regression_case_ids": regressions,
            "net_false_negative_improvement": sum(
                cls._behavior(case) == "INCORRECT_ABSTENTION" for case in a.values()
            )
            - sum(cls._behavior(case) == "INCORRECT_ABSTENTION" for case in b.values()),
            "false_positive_regressions": {
                "count": len(false_positive),
                "case_ids": false_positive,
                "answers_despite_incomplete_retrieval": incomplete_answers,
            },
        }

    @staticmethod
    def _aggregate_subset(cases: tuple[CaseResult, ...], complete: bool) -> dict[str, Any]:
        rows = tuple(case for case in cases if case.retrieval_coverage_complete is complete)
        return EvaluationRunner._aggregate(rows)

    @classmethod
    def _subset_metrics(cls, cases: tuple[CaseResult, ...]) -> dict[str, Any]:
        return {
            "retrieval_complete": cls._aggregate_subset(cases, True),
            "retrieval_incomplete": cls._aggregate_subset(cases, False),
            "judge_decision_overall": cls._judge_decision_metrics(cases),
            "judge_decision_retrieval_complete": cls._judge_decision_metrics(
                tuple(case for case in cases if case.retrieval_coverage_complete)
            ),
            "judge_decision_retrieval_incomplete": cls._judge_decision_metrics(
                tuple(case for case in cases if not case.retrieval_coverage_complete)
            ),
            "retrieval_complete_case_count": sum(c.retrieval_coverage_complete for c in cases),
            "retrieval_incomplete_case_count": sum(
                not c.retrieval_coverage_complete for c in cases
            ),
        }

    @staticmethod
    def _judge_decision_metrics(cases: tuple[CaseResult, ...]) -> dict[str, Any]:
        true_positive = sum(
            not case.expected_abstain
            and bool((case.answerability_result or {}).get("answerable"))
            for case in cases
        )
        false_negative = sum(
            not case.expected_abstain
            and not bool((case.answerability_result or {}).get("answerable"))
            for case in cases
        )
        false_positive = sum(
            case.expected_abstain
            and bool((case.answerability_result or {}).get("answerable"))
            for case in cases
        )
        true_negative = sum(
            case.expected_abstain
            and not bool((case.answerability_result or {}).get("answerable"))
            for case in cases
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        return {
            "case_count": len(cases),
            "true_positive": true_positive,
            "false_negative": false_negative,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "accuracy": (true_positive + true_negative) / len(cases) if cases else None,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def _requirement_diagnostics(cases: tuple[CaseResult, ...]) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        total = 0
        false_negative_requirements: list[dict[str, Any]] = []
        for case in cases:
            requirements = (case.answerability_result or {}).get("requirements", [])
            total += len(requirements)
            for requirement in requirements:
                counts[requirement["status"]] += 1
            if (
                case.retrieval_coverage_complete
                and case.status == "abstained"
                and not case.expected_abstain
            ):
                false_negative_requirements.extend(
                    {
                        "case_id": case.case_id,
                        "requirement_id": requirement["requirement_id"],
                        "description": requirement["description"],
                        "status": requirement["status"],
                        "supporting_chunk_ids": requirement["supporting_chunk_ids"],
                        "support_was_present_in_top5": bool(
                            set(case.expected_document_ids) <= set(case.retrieved_document_ids)
                        ),
                    }
                    for requirement in requirements
                    if requirement["status"] != "SUPPORTED"
                )
        return {
            "mean_requirements_per_question": total / len(cases),
            "total_requirements": total,
            **{
                status: counts[status]
                for status in ("SUPPORTED", "PARTIAL", "MISSING", "CONFLICTING")
            },
            "false_negative_requirements": false_negative_requirements,
        }

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
        payload["final_behavior"] = cls._behavior(case)
        return payload

    @staticmethod
    def _root_cause(case: CaseResult, mode: str) -> str:
        judge_answerable = bool((case.answerability_result or {}).get("answerable"))
        if not case.retrieval_coverage_complete and not case.expected_abstain:
            root = "RETRIEVAL_INCOMPLETE"
        elif case.status == "abstained" and not case.expected_abstain:
            if judge_answerable:
                root = "GENERATION_FAILURE"
            elif mode == MODES[0]:
                root = "V1_EVIDENCE_GATE_FALSE_NEGATIVE"
            elif not (case.answerability_result or {}).get("requirements"):
                root = "V2_REQUIREMENT_DECOMPOSITION_FALSE_NEGATIVE"
            else:
                root = "V2_REQUIREMENT_COVERAGE_FALSE_NEGATIVE"
        elif case.supporting_context_loss:
            root = "SUPPORTING_CONTEXT_LOSS"
        else:
            root = "NONE"
        return root

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
            "prompt_version": JUDGE_A["prompt_version"]
            if mode == MODES[0]
            else JUDGE_B["prompt_version"],
        }

    @staticmethod
    def _usage(
        mode: str, cases: tuple[CaseResult, ...], record: EndToEndBenchmarkRecord
    ) -> dict[str, Any]:
        external = [case for case in cases if case.external_judge_calls]
        return {
            "new_query_embedding_calls": record.preparation_usage["new_query_embedding_calls"]
            if mode == MODES[0]
            else 0,
            "embedding_tokens": record.preparation_usage["embedding_tokens"]
            if mode == MODES[0]
            else 0,
            "new_luna_judge_calls": sum(case.external_judge_calls for case in cases),
            "luna_input_tokens": sum(case.judge_prompt_tokens or 0 for case in external),
            "luna_output_tokens": sum(case.judge_completion_tokens or 0 for case in external),
            "gate_cache_hits": sum(case.gate_cache_hit is True for case in cases),
            "gate_cache_misses": sum(case.gate_cache_hit is False for case in cases),
            "cross_encoder_pairs": record.preparation_usage["cross_encoder_pairs"],
            "document_embedding_calls": 0,
            "external_reranker_calls": 0,
            "unauthorized_chunks_to_reranker": 0,
            "unauthorized_chunks_to_judge": 0,
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
