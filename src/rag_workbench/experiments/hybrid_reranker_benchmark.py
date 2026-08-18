from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.cache import CachedAnswerabilityGate, gate_cache_key
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    OpenAICompatibleAnswerabilityGate,
    evidence_gate_schema,
    hosted_judge_request_settings,
)
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
    RetrievalBenchmarkRecord,
    RetrievalBenchmarkRunRecord,
)
from rag_workbench.evaluation.datasets import EvaluationCase, EvaluationPrincipal
from rag_workbench.evaluation.evaluator import CaseResult, EvaluationRunner
from rag_workbench.evaluation.hybrid_metrics import (
    aggregate_retrieval_metrics,
    category_retrieval_metrics,
    retrieval_case_metrics,
)
from rag_workbench.evaluation.retrieval_dataset import RetrievalGroundTruthCase
from rag_workbench.experiments.judge_e2e_benchmark import FrozenJudgeEndToEndBenchmark
from rag_workbench.experiments.reranker_e2e_benchmark import (
    CORPUS_IDENTITY,
    RERANKER_REVISION,
    SEMANTIC_INDEX_IDENTITY,
    FixedResultRetriever,
    RerankerEndToEndBenchmark,
    _percentile,
    _result,
)
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.generation.generator import RagService
from rag_workbench.providers.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from rag_workbench.providers.llm import ExtractiveGenerationProvider
from rag_workbench.reranking import CrossEncoderReranker, Reranker
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import RetrievalTiming, Retriever
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal

DATASET_ID = "acmeai-hybrid-reranker-eval-v1"
DATASET_PATH = Path("data/eval/acmeai_hybrid_reranker_eval_v1.json")
DATASET_HASH = "d5b69e67a4b322d03e8f9530e0526142c1176bf4112febc3cc5c61635ba9a451"
MAXIMUM_PRIOR_OVERLAP = 0.4
SPLIT_SEED = 2026081717
SOL_MODEL = "gpt-5.6-sol"
SCHEMA_IDENTITY = "6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621"
MODES = ("DENSE_CROSS_ENCODER_RERANK", "HYBRID_CROSS_ENCODER_RERANK")
DENSE_DEPTH = 20
BM25_DEPTH = 20
UNION_LIMIT = 30
RRF_K = 60
FINAL_TOP_K = 5
HOLDOUT_ALLOCATION = {
    "multidoc_two": 2,
    "multidoc_three": 9,
    "near_duplicate": 2,
    "exact_identifier": 2,
    "version_region": 2,
    "semantic_paraphrase": 1,
    "acl_sensitive": 1,
    "partial_no_answer": 1,
}
CONFIGURATION = {
    "embedding_model": "text-embedding-3-small",
    "embedding_dimension": 64,
    "dense_backend": "pgvector_cosine",
    "dense_threshold": 0.28,
    "dense_candidate_depth": DENSE_DEPTH,
    "bm25_candidate_depth": BM25_DEPTH,
    "bm25": {
        "implementation": "BM25 Okapi",
        "version": "bm25-okapi-v1",
        "k1": 1.2,
        "b": 0.75,
        "tokenization": "NFKC + casefold + words + preserved identifier and components",
    },
    "rrf_k": RRF_K,
    "candidate_union_limit": UNION_LIMIT,
    "final_top_k": FINAL_TOP_K,
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "reranker_revision": RERANKER_REVISION,
    "dense_threshold_policy": "0.28 applies to dense branch only; never to BM25 or RRF scores",
    "judge": {
        "model": SOL_MODEL,
        "prompt_version": EVIDENCE_GATE_PROMPT_VERSION,
        "schema_identity": SCHEMA_IDENTITY,
        "request_settings": hosted_judge_request_settings(EVIDENCE_GATE_PROMPT_VERSION),
        "used_for_retrieval_selection": False,
    },
}
SELECTION_POLICY = {
    "all_required_coverage_minimum_gain": 0.10,
    "three_document_coverage_minimum_gain": 0.20,
    "required_recall_maximum_regression": 0.02,
    "exact_identifier_maximum_regression": 0.05,
    "semantic_maximum_regression": 0.05,
    "version_correctness_required": 1.0,
    "acl_safety_required": 1.0,
}
PROMOTION_POLICY = {
    "coverage_must_improve_directionally": True,
    "three_document_must_improve_directionally": True,
    "correct_answers_must_not_regress": True,
    "maximum_incorrect_abstention_increase": 1,
    "maximum_unsupported_answer_increase": 0,
    "hard_constraints": {
        "acl_safety": 1.0,
        "tenant_isolation": 1.0,
        "version_correctness": 1.0,
        "prompt_injection_boundary": 1.0,
        "unauthorized_chunks_to_cross_encoder": 0,
        "unauthorized_chunks_to_sol": 0,
    },
}


class HybridRerankerCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: str
    question: str
    required_document_ids: tuple[str, ...] = ()
    required_chunk_ids: tuple[str, ...] = ()
    required_version_ids: dict[str, str] = Field(default_factory=dict)
    required_fact_ids: tuple[str, ...] = ()
    expected_facts: tuple[str, ...] = ()
    expected_answer: str | None = None
    forbidden_document_ids: tuple[str, ...] = ()
    expected_access_behavior: str
    expected_answerability: bool
    should_abstain: bool
    expected_document_ids: tuple[str, ...] = ()
    expected_versions: dict[str, str] = Field(default_factory=dict)
    security_checks: tuple[str, ...] = ()
    principal: EvaluationPrincipal = Field(default_factory=EvaluationPrincipal)
    expected_prompt_injection_behavior: str | None = None
    unavailable_required_fact_ids: tuple[str, ...] = ()

    def as_retrieval(self) -> RetrievalGroundTruthCase:
        return RetrievalGroundTruthCase(
            case_id=self.case_id,
            question=self.question,
            required_document_ids=self.required_document_ids,
            required_chunk_ids=self.required_chunk_ids,
            required_version_ids=self.required_version_ids,
            forbidden_document_ids=self.forbidden_document_ids,
            expected_access_behavior=self.expected_access_behavior,
            expected_answerability=self.expected_answerability,
            category=self.category,
            principal=self.principal,
        )

    def as_evaluation(self) -> EvaluationCase:
        return EvaluationCase(
            case_id=self.case_id,
            question=self.question,
            expected_answer=self.expected_answer,
            expected_document_ids=self.expected_document_ids,
            expected_chunk_ids=self.required_chunk_ids,
            forbidden_document_ids=self.forbidden_document_ids,
            expected_versions=self.expected_versions,
            required_fact_ids=self.required_fact_ids,
            unavailable_required_fact_ids=self.unavailable_required_fact_ids,
            expected_facts=self.expected_facts,
            security_checks=self.security_checks,
            should_abstain=self.should_abstain,
            category=self.category,  # type: ignore[arg-type]
            principal=self.principal,
        )


def load_hybrid_reranker_cases() -> tuple[HybridRerankerCase, ...]:
    payload = json.loads(DATASET_PATH.read_text())
    return tuple(HybridRerankerCase.model_validate(item) for item in payload["cases"])


def deterministic_hybrid_reranker_split(
    cases: tuple[HybridRerankerCase, ...], *, seed: int = SPLIT_SEED
) -> tuple[tuple[HybridRerankerCase, ...], tuple[HybridRerankerCase, ...], str]:
    grouped: dict[str, list[HybridRerankerCase]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)
    if set(grouped) != set(HOLDOUT_ALLOCATION):
        raise ValueError("hybrid reranker dataset category composition changed")
    holdout_ids: set[str] = set()
    for category, allocation in sorted(HOLDOUT_ALLOCATION.items()):
        values = sorted(grouped[category], key=lambda item: item.case_id)
        random.Random(f"{seed}:{category}").shuffle(values)
        holdout_ids.update(item.case_id for item in values[:allocation])
    calibration = tuple(item for item in cases if item.case_id not in holdout_ids)
    holdout = tuple(item for item in cases if item.case_id in holdout_ids)
    payload = "|".join(item.case_id for item in calibration)
    payload += "::" + "|".join(item.case_id for item in holdout)
    identity = hashlib.sha256(f"{seed}:{payload}".encode()).hexdigest()
    if len(calibration) != 40 or len(holdout) != 20:
        raise ValueError("hybrid reranker split must be exactly 40/20")
    return calibration, holdout, identity


CASES = load_hybrid_reranker_cases()
CALIBRATION_CASES, HOLDOUT_CASES, SPLIT_IDENTITY = deterministic_hybrid_reranker_split(CASES)


def _principal(case: HybridRerankerCase) -> Principal:
    return Principal(
        case.principal.principal_id,
        case.principal.tenant_id,
        frozenset(case.principal.permission_groups),
    )


def _trace(item: RetrievalResult) -> dict[str, Any]:
    return {
        "chunk_id": item.chunk_id,
        "document_id": item.document_id,
        "document_version_id": item.document_version_id,
        "text": item.text,
        "rank": item.rank,
        "score": item.score,
        "source": item.source,
        "source_type": item.source_type,
        "title": item.title,
        "version": item.version,
        "page": item.page,
        "section": item.section,
        "metadata": item.metadata,
        "retrieval_source": item.retrieval_source,
        "dense_score": item.dense_score,
        "lexical_score": item.lexical_score,
        "fusion_score": item.fusion_score,
        "found_by_dense": item.found_by_dense,
        "found_by_bm25": item.found_by_bm25,
        "found_by_both": bool(item.found_by_dense and item.found_by_bm25),
    }


def _union_trace(
    item: RetrievalResult, dense: list[RetrievalResult], bm25: list[RetrievalResult]
) -> dict[str, Any]:
    payload = _trace(item)
    payload["dense_rank"] = next(
        (entry.rank for entry in dense if entry.chunk_id == item.chunk_id), None
    )
    payload["bm25_rank"] = next(
        (entry.rank for entry in bm25 if entry.chunk_id == item.chunk_id), None
    )
    return payload


def _docs(results: list[RetrievalResult]) -> set[str]:
    return {item.document_id for item in results}


def pool_metrics(case: RetrievalGroundTruthCase, results: list[RetrievalResult]) -> dict[str, Any]:
    documents = _docs(results)
    required = set(case.required_document_ids)
    retrieved = required & documents
    unauthorized = (
        sum(item.document_id in set(case.forbidden_document_ids) for item in results)
        if case.expected_access_behavior == "EXCLUDE_FORBIDDEN"
        else 0
    )
    return {
        "required_evidence_recall": (len(retrieved) / len(required) if required else None),
        "all_required_evidence_coverage": (
            float(required <= documents) if case.expected_answerability else None
        ),
        "unauthorized_result_exposure": unauthorized,
        "candidate_count": len(results),
    }


def aggregate_pool(rows: list[dict[str, Any]], category: str | None = None) -> dict[str, Any]:
    answerable = [item for item in rows if item["expected_answerability"]]
    scoped = (
        [item for item in answerable if item["category"] == category] if category else answerable
    )
    values = [
        item["pool"]["all_required_evidence_coverage"]
        for item in scoped
        if item["pool"]["all_required_evidence_coverage"] is not None
    ]
    recall = [
        item["pool"]["required_evidence_recall"]
        for item in scoped
        if item["pool"]["required_evidence_recall"] is not None
    ]
    return {
        "required_evidence_recall": mean(recall) if recall else 0.0,
        "all_required_evidence_coverage": mean(values) if values else 0.0,
        "two_document_coverage": aggregate_pool(rows, "multidoc_two")[
            "all_required_evidence_coverage"
        ]
        if category is None
        else None,
        "three_document_coverage": aggregate_pool(rows, "multidoc_three")[
            "all_required_evidence_coverage"
        ]
        if category is None
        else None,
    }


def apply_selection_policy(dense: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
    coverage_gain = (
        hybrid["all_required_evidence_coverage_at_5"]
        - dense["all_required_evidence_coverage_at_5"]
    )
    three_gain = hybrid["three_document_coverage_at_5"] - dense["three_document_coverage_at_5"]
    recall_regression = (
        dense["required_evidence_recall_at_5"] - hybrid["required_evidence_recall_at_5"]
    )
    exact_regression = (
        dense["exact_identifier_recall_at_5"] - hybrid["exact_identifier_recall_at_5"]
    )
    semantic_regression = dense["semantic_success"] - hybrid["semantic_success"]
    material = coverage_gain >= 0.10 or three_gain >= 0.20
    safe = (
        recall_regression <= 0.02
        and exact_regression <= 0.05
        and semantic_regression <= 0.05
        and hybrid["version_correctness"] == 1.0
        and hybrid["acl_safety"] == 1.0
    )
    selected = MODES[1] if material and safe else MODES[0]
    return {
        "selected_mode": selected,
        "coverage_gain": coverage_gain,
        "three_document_gain": three_gain,
        "required_recall_regression": recall_regression,
        "exact_identifier_regression": exact_regression,
        "semantic_regression": semantic_regression,
        "material": material,
        "safe": safe,
        "reason": (
            f"coverage_gain={coverage_gain:.6f}; three_doc_gain={three_gain:.6f}; "
            f"recall_regression={recall_regression:.6f}; "
            f"exact_regression={exact_regression:.6f}; "
            f"semantic_regression={semantic_regression:.6f}; "
            f"version={hybrid['version_correctness']}; "
            f"acl={hybrid['acl_safety']}."
        ),
    }


def compare_three_document(
    dense_traces: list[dict[str, Any]], hybrid_traces: list[dict[str, Any]]
) -> dict[str, Any]:
    left = {item["case_id"]: item for item in dense_traces}
    right = {item["case_id"]: item for item in hybrid_traces}
    fixed: list[str] = []
    worsened: list[str] = []
    still_impossible: list[str] = []
    complete_in_both: list[str] = []
    for case_id, row in left.items():
        other = right[case_id]
        if row["top5_complete"] and other["top5_complete"]:
            complete_in_both.append(case_id)
        elif not row["top5_complete"] and other["top5_complete"]:
            fixed.append(case_id)
        elif row["top5_complete"] and not other["top5_complete"]:
            worsened.append(case_id)
        else:
            still_impossible.append(case_id)
    return {
        "fixed_by_hybrid_reranker": fixed,
        "worsened_by_hybrid_reranker": worsened,
        "still_impossible_after_candidate_expansion": still_impossible,
        "complete_in_both": complete_in_both,
    }


def maximum_prior_dataset_overlap() -> float:
    token = re.compile(r"[a-z0-9]+")

    def terms(question: str) -> set[str]:
        return set(token.findall(question.casefold()))

    maximum = 0.0
    for path in Path("data/eval").glob("*.json"):
        if path == DATASET_PATH:
            continue
        for previous in json.loads(path.read_text()).get("cases", []):
            for case in CASES:
                left, right = terms(case.question), terms(previous["question"])
                maximum = max(maximum, len(left & right) / len(left | right))
    return maximum


class HybridRerankerBenchmark:
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
        if hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() != DATASET_HASH:
            raise ValueError("hybrid reranker dataset identity changed")
        computed = hashlib.sha256(
            json.dumps(evidence_gate_schema(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if computed != SCHEMA_IDENTITY:
            raise ValueError("evidence-sufficiency-v1 schema identity changed")

    def _corpus_identity(self) -> str:
        rows = self.session.execute(
            select(Document.document_id, DocumentVersion.version, DocumentVersion.content_hash)
            .join(DocumentVersion, DocumentVersion.document_fk == Document.id)
            .where(DocumentVersion.is_active.is_(True))
            .order_by(Document.document_id, DocumentVersion.version)
        ).all()
        return hashlib.sha256("|".join(":".join(row) for row in rows).encode()).hexdigest()

    def initialize(self) -> RetrievalBenchmarkRecord:
        if not self.session.scalar(
            select(Chunk.id).where(Chunk.index_identity == SEMANTIC_INDEX_IDENTITY).limit(1)
        ):
            raise ValueError("frozen semantic index is unavailable")
        corpus = self._corpus_identity()
        if corpus != CORPUS_IDENTITY:
            raise ValueError("corpus identity changed")
        existing = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        if existing:
            self._verify(existing)
            return existing
        self._verify_production_identities()
        record = RetrievalBenchmarkRecord(
            dataset_id=DATASET_ID,
            dataset_hash=DATASET_HASH,
            split_identity=SPLIT_IDENTITY,
            split_seed=SPLIT_SEED,
            calibration_case_ids=[item.case_id for item in CALIBRATION_CASES],
            holdout_case_ids=[item.case_id for item in HOLDOUT_CASES],
            semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
            corpus_identity=CORPUS_IDENTITY,
            retrieval_configuration=CONFIGURATION,
            selection_policy={**SELECTION_POLICY, "promotion_policy": PROMOTION_POLICY},
        )
        self.session.add(record)
        self.session.commit()
        return record

    def _verify(self, record: RetrievalBenchmarkRecord) -> None:
        if not (
            record.dataset_hash == DATASET_HASH
            and record.split_identity == SPLIT_IDENTITY
            and record.split_seed == SPLIT_SEED
            and record.retrieval_configuration == CONFIGURATION
            and record.selection_policy
            == {**SELECTION_POLICY, "promotion_policy": PROMOTION_POLICY}
            and record.semantic_index_identity == SEMANTIC_INDEX_IDENTITY
            and record.corpus_identity == CORPUS_IDENTITY
        ):
            raise ValueError("sealed hybrid reranker identity changed")

    def _verify_production_identities(self) -> None:
        sol = self.session.get(EndToEndBenchmarkRecord, "acmeai-sol-judge-e2e-eval-v1")
        reranker = self.session.get(RerankingBenchmarkRecord, "acmeai-reranking-eval-v1")
        if not sol or sol.production_retriever_status != "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1":
            raise ValueError("production Sol judge identity changed")
        if (
            not reranker
            or reranker.selected_mode != "DENSE_CROSS_ENCODER_RERANK"
            or reranker.model_revision != RERANKER_REVISION
        ):
            raise ValueError("frozen Cross-Encoder identity changed")

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
        return {
            "current_cumulative_embedding_calls": current,
            "configured_ceiling": self.settings.max_external_embedding_calls,
            "new_unique_queries": len(questions),
            "existing_cache_matches": matches,
            "missing_embeddings": len(keys) - matches,
            "maximum_new_calls": len(keys) - matches,
            "expected_cumulative_ending_usage": current + len(keys) - matches,
        }

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

    def _partition_runs(self, partition: str) -> dict[str, RetrievalBenchmarkRunRecord]:
        return {
            run.retrieval_mode: run
            for run in self.session.scalars(
                select(RetrievalBenchmarkRunRecord).where(
                    RetrievalBenchmarkRunRecord.dataset_id == DATASET_ID,
                    RetrievalBenchmarkRunRecord.partition == partition,
                )
            ).all()
        }

    def calibrate(self) -> dict[str, Any]:
        record = self.initialize()
        if record.locked_at is not None:
            raise ValueError("hybrid reranker calibration is already locked")
        if self._partition_runs("calibration"):
            raise ValueError("hybrid reranker calibration was already executed")
        judge_before = self._historical_judge_calls()
        metrics = self._run_partition("calibration", CALIBRATION_CASES)
        if self._historical_judge_calls() != judge_before:
            raise RuntimeError("Sol was used during retrieval calibration")
        dense = metrics[MODES[0]]["metrics"]
        hybrid = metrics[MODES[1]]["metrics"]
        decision = apply_selection_policy(dense, hybrid)
        record = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        record.selected_retrieval_mode = decision["selected_mode"]
        record.calibration_metrics = {
            "modes": {mode: payload["metrics"] for mode, payload in metrics.items()},
            "candidate_pools": {
                mode: payload["candidate_pool"] for mode, payload in metrics.items()
            },
            "lexical_rescues": metrics[MODES[1]]["lexical_rescues"],
            "cross_encoder_conversion": metrics[MODES[1]]["cross_encoder_conversion"],
            "three_document": {
                mode: payload["three_document"] for mode, payload in metrics.items()
            },
            "three_document_effect": compare_three_document(
                metrics[MODES[0]]["three_document"]["traces"],
                metrics[MODES[1]]["three_document"]["traces"],
            ),
            "selection": decision,
            "sol_used_for_retrieval_selection": False,
        }
        record.selection_reason = decision["reason"]
        record.locked_at = datetime.now(UTC)
        self.session.commit()
        return self.status()

    def holdout_retrieve(self) -> dict[str, Any]:
        record = self.initialize()
        if record.locked_at is None or not record.selected_retrieval_mode:
            raise ValueError("calibration must be locked before holdout")
        if record.holdout_started_at is not None:
            raise ValueError("hybrid reranker holdout retrieval is one-shot")
        record.holdout_started_at = datetime.now(UTC)
        self.session.commit()
        judge_before = self._historical_judge_calls()
        metrics = self._run_partition("holdout", HOLDOUT_CASES)
        if self._historical_judge_calls() != judge_before:
            raise RuntimeError("Sol was used during holdout retrieval")
        prepared = [
            {
                "case_id": case.case_id,
                "pipeline_a_top5": next(
                    item["top5"]
                    for item in metrics[MODES[0]]["cases"]
                    if item["case_id"] == case.case_id
                ),
                "pipeline_b_top5": next(
                    item["top5"]
                    for item in metrics[MODES[1]]["cases"]
                    if item["case_id"] == case.case_id
                ),
                "timing": next(
                    item["timing"]
                    for item in metrics[MODES[1]]["cases"]
                    if item["case_id"] == case.case_id
                ),
                "shared_query_embedding": True,
            }
            for case in HOLDOUT_CASES
        ]
        e2e = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        if e2e is None:
            e2e = EndToEndBenchmarkRecord(
                dataset_id=DATASET_ID,
                dataset_hash=DATASET_HASH,
                case_ids=[item.case_id for item in HOLDOUT_CASES],
                category_distribution=dict(
                    sorted(Counter(item.category for item in HOLDOUT_CASES).items())
                ),
                generation_method="manual-corpus-grounded-v1",
                maximum_prior_overlap=MAXIMUM_PRIOR_OVERLAP,
                semantic_index_identity=SEMANTIC_INDEX_IDENTITY,
                corpus_identity=CORPUS_IDENTITY,
                reranker_revision=RERANKER_REVISION,
                pipeline_a_configuration={**CONFIGURATION, "mode": MODES[0]},
                pipeline_b_configuration={**CONFIGURATION, "mode": MODES[1]},
            )
            self.session.add(e2e)
            self.session.commit()
        e2e.prepared_cases = prepared
        e2e.prepared_at = datetime.now(UTC)
        e2e.preparation_usage = metrics[MODES[1]]["usage"]
        record.holdout_completed_at = datetime.now(UTC)
        self.session.commit()
        return self.status()

    def _run_partition(
        self, partition: str, cases: tuple[HybridRerankerCase, ...]
    ) -> dict[str, Any]:
        preflight = self.embedding_preflight()
        if not self.settings.allow_external_calls or not self.settings.embedding_api_key:
            raise ValueError("external embedding authorization is incomplete")
        if preflight["expected_cumulative_ending_usage"] > preflight["configured_ceiling"]:
            raise ValueError("hybrid reranker dataset exceeds the embedding-call ceiling")
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
        payloads: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
        usage = {
            "new_query_embedding_calls": 0,
            "embedding_tokens": 0,
            "query_cache_hits": 0,
            "query_cache_misses": 0,
            "document_embedding_calls": 0,
            "external_reranker_calls": 0,
            "cross_encoder_pairs_a": 0,
            "cross_encoder_pairs_b": 0,
            "bm25_index_size_bytes": 0,
            "bm25_build_ms": 0.0,
            "unauthorized_chunks_to_cross_encoder": 0,
        }
        for case in cases:
            retrieval = case.as_retrieval()
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
            usage["bm25_index_size_bytes"] = bm25.last_timing.index_size_bytes
            usage["bm25_build_ms"] = bm25.last_timing.index_build_latency_ms
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
            a_reranked = reranker.rerank(case.question, dense_candidates)
            a_pairs = len(dense_candidates)
            a_rerank_ms = reranker.last_inference_latency_ms + reranker.last_sort_latency_ms
            b_reranked = reranker.rerank(case.question, union)
            b_pairs = len(union)
            b_rerank_ms = reranker.last_inference_latency_ms + reranker.last_sort_latency_ms
            usage["cross_encoder_pairs_a"] += a_pairs
            usage["cross_encoder_pairs_b"] += b_pairs
            a_top5 = [
                replace(
                    item.result,
                    rank=item.reranked_rank,
                    score=item.reranker_score,
                    retrieval_source="dense_cross_encoder_rerank",
                )
                for item in a_reranked[:FINAL_TOP_K]
            ]
            b_top5 = [
                replace(
                    item.result,
                    rank=item.reranked_rank,
                    score=item.reranker_score,
                    retrieval_source="hybrid_cross_encoder_rerank",
                )
                for item in b_reranked[:FINAL_TOP_K]
            ]
            timing = {
                "query_embedding_ms": embedding.embedding_latency_ms,
                "cache_lookup_ms": embedding.cache_lookup_latency_ms,
                "dense_ms": dense_ms,
                "bm25_ms": bm25_ms,
                "rrf_ms": fusion_ms,
                "reranker_a_ms": a_rerank_ms,
                "reranker_b_ms": b_rerank_ms,
            }
            payloads[MODES[0]].append(
                self._case_payload(
                    retrieval,
                    dense_candidates,
                    [],
                    dense_candidates,
                    a_top5,
                    a_reranked,
                    timing,
                    a_pairs,
                    embedding.cache_hit,
                    union_all=dense_candidates,
                )
            )
            payloads[MODES[1]].append(
                self._case_payload(
                    retrieval,
                    dense_candidates,
                    bm25_candidates,
                    union,
                    b_top5,
                    b_reranked,
                    timing,
                    b_pairs,
                    embedding.cache_hit,
                    union_all=union_all,
                )
            )
        analysis = {mode: self._mode_analysis(payloads[mode], mode) for mode in MODES}
        analysis[MODES[1]]["lexical_rescues"] = self._lexical_rescues(payloads[MODES[1]])
        analysis[MODES[1]]["cross_encoder_conversion"] = self._conversion(payloads[MODES[1]])
        for mode in MODES:
            analysis[mode]["usage"] = usage
            contribution: dict[str, Any] = {
                "candidate_pool": analysis[mode]["candidate_pool"],
                "three_document": {
                    key: value
                    for key, value in analysis[mode]["three_document"].items()
                    if key != "traces"
                },
            }
            if mode == MODES[1]:
                contribution["lexical_rescues"] = analysis[mode]["lexical_rescues"]
                contribution["cross_encoder_conversion"] = analysis[mode][
                    "cross_encoder_conversion"
                ]
            self.session.add(
                RetrievalBenchmarkRunRecord(
                    dataset_id=DATASET_ID,
                    partition=partition,
                    retrieval_mode=mode,
                    metrics=analysis[mode]["metrics"],
                    category_metrics=analysis[mode]["category_metrics"],
                    case_results=payloads[mode],
                    branch_contribution=contribution,
                    latency=analysis[mode]["latency"],
                    usage=usage,
                )
            )
        self.session.commit()
        return analysis

    def _case_payload(
        self,
        case: RetrievalGroundTruthCase,
        dense: list[RetrievalResult],
        bm25: list[RetrievalResult],
        union: list[RetrievalResult],
        top5: list[RetrievalResult],
        reranked: list[Any],
        timing: dict[str, Any],
        pairs: int,
        cache_hit: bool,
        union_all: list[RetrievalResult] | None = None,
    ) -> dict[str, Any]:
        fused = union_all if union_all is not None else union
        ce_ranks = {
            item.result.document_id: item.reranked_rank
            for item in reranked
            if item.result.document_id in set(case.required_document_ids)
        }
        return {
            "case_id": case.case_id,
            "question": case.question,
            "category": case.category,
            "expected_answerability": case.expected_answerability,
            "required_document_ids": list(case.required_document_ids),
            "dense_candidates": [_trace(item) for item in dense],
            "bm25_candidates": [_trace(item) for item in bm25],
            "rrf_union": [_union_trace(item, dense, bm25) for item in union],
            "top5": [_trace(item) for item in top5],
            "candidate_count_before_dedup": len(dense) + len(bm25),
            "candidate_count_after_dedup": len({item.chunk_id for item in [*dense, *bm25]}),
            "candidate_count_sent_to_reranker": len(union) if union else len(dense),
            "cross_encoder_pairs": pairs,
            "metrics": retrieval_case_metrics(case, top5),
            "pool": pool_metrics(case, union if union else dense),
            "required_dense_ranks": {
                document_id: next(
                    (item.rank for item in dense if item.document_id == document_id), None
                )
                for document_id in case.required_document_ids
            },
            "required_bm25_ranks": {
                document_id: next(
                    (item.rank for item in bm25 if item.document_id == document_id), None
                )
                for document_id in case.required_document_ids
            },
            "required_rrf_ranks": {
                document_id: next(
                    (item.rank for item in fused if item.document_id == document_id), None
                )
                for document_id in case.required_document_ids
            },
            "required_ce_ranks": ce_ranks,
            "top5_complete": bool(set(case.required_document_ids) <= _docs(top5))
            if case.expected_answerability
            else None,
            "timing": timing,
            "embedding_cache_hit": cache_hit,
            "failure_taxonomy": self._taxonomy(case, dense, bm25, union, top5),
        }

    @staticmethod
    def _taxonomy(
        case: RetrievalGroundTruthCase,
        dense: list[RetrievalResult],
        bm25: list[RetrievalResult],
        union: list[RetrievalResult],
        top5: list[RetrievalResult],
    ) -> str:
        if case.expected_access_behavior == "EXCLUDE_FORBIDDEN" and any(
            item.document_id in set(case.forbidden_document_ids) for item in [*dense, *union, *top5]
        ):
            return "ACL_FAILURE"
        required = set(case.required_document_ids)
        if not case.expected_answerability or required <= _docs(top5):
            return "NONE"
        dense_docs, bm25_docs, union_docs = _docs(dense), _docs(bm25), _docs(union)
        if not (required & dense_docs) and not (required & bm25_docs):
            return "BOTH_BRANCHES_MISS"
        if not required <= dense_docs and required <= bm25_docs:
            if required <= union_docs and not required <= _docs(top5):
                return "CROSS_ENCODER_FAILED_TO_PROMOTE"
            if not required <= union_docs:
                return "RRF_TRUNCATION_LOSS"
            return "LEXICAL_CANDIDATE_RESCUE"
        if not required <= dense_docs:
            return "DENSE_CANDIDATE_GENERATION_MISS"
        if required <= dense_docs and not required <= _docs(top5):
            return "CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE"
        if case.category == "multidoc_three":
            return "THREE_DOCUMENT_COVERAGE_FAILURE"
        return "RANKING_OUTSIDE_TOP5"

    def _mode_analysis(self, rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
        metrics = aggregate_retrieval_metrics(rows)
        category = category_retrieval_metrics(rows)
        semantic = category.get("semantic_paraphrase", {}).get(
            "all_required_evidence_coverage_at_5", 0.0
        )
        metrics["semantic_success"] = semantic
        metrics["near_duplicate_success"] = metrics["near_duplicate_preferred_source_success"]
        pool_rows = [
            {
                "expected_answerability": item["expected_answerability"],
                "category": item["category"],
                "pool": item["pool"],
            }
            for item in rows
        ]
        three = [item for item in rows if item["category"] == "multidoc_three"]
        reranker_key = "reranker_b_ms" if mode == MODES[1] else "reranker_a_ms"
        rerank_ms = [item["timing"].get(reranker_key) or 0.0 for item in rows]
        return {
            "metrics": metrics,
            "category_metrics": category,
            "candidate_pool": {
                "required_evidence_recall": aggregate_pool(pool_rows)["required_evidence_recall"],
                "all_required_evidence_coverage": aggregate_pool(pool_rows)[
                    "all_required_evidence_coverage"
                ],
                "two_document_coverage": aggregate_pool(pool_rows, "multidoc_two")[
                    "all_required_evidence_coverage"
                ],
                "three_document_coverage": aggregate_pool(pool_rows, "multidoc_three")[
                    "all_required_evidence_coverage"
                ],
            },
            "three_document": {
                "case_count": len(three),
                "candidate_coverage": mean(
                    item["pool"]["all_required_evidence_coverage"] or 0.0 for item in three
                )
                if three
                else 0.0,
                "top5_coverage": mean(float(item["top5_complete"] or 0) for item in three)
                if three
                else 0.0,
                "traces": [
                    {
                        "case_id": item["case_id"],
                        "required_document_ids": item["required_document_ids"],
                        "dense_ranks": item["required_dense_ranks"],
                        "bm25_ranks": item["required_bm25_ranks"],
                        "rrf_ranks": item["required_rrf_ranks"],
                        "ce_ranks": item["required_ce_ranks"],
                        "top5_complete": item["top5_complete"],
                    }
                    for item in three
                ],
            },
            "latency": {
                "query_embedding_ms": mean(item["timing"]["query_embedding_ms"] for item in rows),
                "dense_ms": mean(item["timing"]["dense_ms"] for item in rows),
                "bm25_ms": mean(item["timing"]["bm25_ms"] for item in rows),
                "rrf_ms": mean(item["timing"]["rrf_ms"] for item in rows),
                "reranker_mean_ms": mean(rerank_ms),
                "reranker_p50_ms": median(rerank_ms),
                "reranker_p95_ms": _percentile(rerank_ms, 0.95),
                "pairs_per_query": mean(item["cross_encoder_pairs"] for item in rows),
            },
            "cases": rows,
        }

    @staticmethod
    def _lexical_rescues(rows: list[dict[str, Any]]) -> dict[str, Any]:
        rescued: list[dict[str, Any]] = []
        for item in rows:
            if not item["expected_answerability"]:
                continue
            for document_id in item["required_document_ids"]:
                dense_rank = item["required_dense_ranks"].get(document_id)
                bm25_rank = item["required_bm25_ranks"].get(document_id)
                if dense_rank is None and bm25_rank is not None:
                    rrf_rank = item["required_rrf_ranks"].get(document_id)
                    ce_rank = item["required_ce_ranks"].get(document_id)
                    rescued.append(
                        {
                            "case_id": item["case_id"],
                            "document_id": document_id,
                            "bm25_rank": bm25_rank,
                            "rrf_rank": rrf_rank,
                            "cross_encoder_rank": ce_rank,
                            "retained_by_rrf": rrf_rank is not None,
                            "sent_to_cross_encoder": (
                                rrf_rank is not None and rrf_rank <= UNION_LIMIT
                            ),
                            "promoted_to_top5": ce_rank is not None and ce_rank <= FINAL_TOP_K,
                        }
                    )
        sent = [item for item in rescued if item["sent_to_cross_encoder"]]
        promoted = [item for item in rescued if item["promoted_to_top5"]]
        return {
            "bm25_only_required_evidence": len(rescued),
            "retained_by_rrf": sum(item["retained_by_rrf"] for item in rescued),
            "sent_to_cross_encoder": len(sent),
            "promoted_to_top5": len(promoted),
            "lost": len(rescued) - len(promoted),
            "items": rescued,
        }

    @staticmethod
    def _conversion(rows: list[dict[str, Any]]) -> dict[str, Any]:
        lexical = HybridRerankerBenchmark._lexical_rescues(rows)
        sent = lexical["sent_to_cross_encoder"]
        promoted = lexical["promoted_to_top5"]
        dense_only_retained = dense_only_lost = 0
        for item in rows:
            if not item["expected_answerability"]:
                continue
            top5 = {entry["document_id"] for entry in item["top5"]}
            for document_id in item["required_document_ids"]:
                if (
                    item["required_dense_ranks"].get(document_id) is not None
                    and item["required_bm25_ranks"].get(document_id) is None
                ):
                    if document_id in top5:
                        dense_only_retained += 1
                    else:
                        dense_only_lost += 1
        return {
            "lexical_sent": sent,
            "lexical_promoted": promoted,
            "conversion_rate": (promoted / sent) if sent else None,
            "dense_only_required_retained": dense_only_retained,
            "dense_only_required_lost": dense_only_lost,
        }

    def judge_preflight(self) -> dict[str, Any]:
        e2e = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        if not e2e or not e2e.prepared_cases:
            raise ValueError("holdout retrieval traces must be frozen before Sol preflight")
        keys_a: list[str] = []
        keys_b: list[str] = []
        for case, prepared in zip(HOLDOUT_CASES, e2e.prepared_cases, strict=True):
            for collector, field in ((keys_a, "pipeline_a_top5"), (keys_b, "pipeline_b_top5")):
                collector.append(
                    gate_cache_key(
                        case.question,
                        RerankerEndToEndBenchmark._gate_evidence(prepared[field]),
                        provider="openai",
                        model=SOL_MODEL,
                        gate_version="1",
                        prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
                    )[0]
                )
        unique_a = set(keys_a)
        unique_b = set(keys_b)
        matches_a = sum(
            self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in unique_a
        )
        matches_b = sum(
            self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in unique_b
        )
        identical = sum(left == right for left, right in zip(keys_a, keys_b, strict=True))
        current = self._historical_judge_calls()
        new_a = max(len(unique_a) - matches_a, 0)
        new_b = max(len(unique_b) - matches_b, 0)
        preflight = {
            "current_cumulative_judge_calls": current,
            "configured_ceiling": self.settings.max_external_judge_calls,
            "existing_a_cache_matches": matches_a,
            "existing_b_cache_matches": matches_b,
            "identical_a_b_gate_inputs": identical,
            "new_a_calls_required": new_a,
            "new_b_calls_required": new_b,
            "maximum_ending_judge_usage": current + new_a + new_b,
            "calibration_sol_calls": 0,
        }
        if preflight["maximum_ending_judge_usage"] > current + 40:
            raise RuntimeError("holdout Sol call budget exceeded the authorized 40-call worst case")
        e2e.judge_preflight = preflight
        self.session.commit()
        return preflight

    def holdout_execute(self) -> dict[str, Any]:
        record = self.initialize()
        e2e = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        if record.locked_at is None or record.holdout_completed_at is None or not e2e:
            raise ValueError("retrieval lock and holdout traces are required before Sol")
        if e2e.execution_started_at is not None:
            raise ValueError("holdout Sol evaluation is one-shot")
        preflight = self.judge_preflight()
        if (
            not self.settings.allow_external_judge_calls
            or not self.settings.effective_judge_api_key
        ):
            raise ValueError("hosted judge authorization is incomplete")
        if preflight["maximum_ending_judge_usage"] > preflight["configured_ceiling"]:
            raise ValueError("holdout Sol calls exceed the judge-call ceiling")
        e2e.execution_started_at = datetime.now(UTC)
        self.session.commit()
        gate = (
            self.gate_factory(SOL_MODEL, 40)
            if self.gate_factory
            else CachedAnswerabilityGate(
                self.session,
                ExternalJudgeCallLimitGate(
                    OpenAICompatibleAnswerabilityGate(
                        api_key=self.settings.effective_judge_api_key or "",
                        model=SOL_MODEL,
                        base_url=self.settings.judge_base_url,
                        gate_version="1",
                        prompt_version=EVIDENCE_GATE_PROMPT_VERSION,
                        provider_name="openai",
                    ),
                    40,
                ),
            )
        )
        provider = self._embedding_provider()
        results: dict[str, list[CaseResult]] = {mode: [] for mode in MODES}
        unauthorized = 0
        for case, prepared in zip(HOLDOUT_CASES, e2e.prepared_cases or [], strict=True):
            evaluation = case.as_evaluation()
            for mode, field in zip(MODES, ("pipeline_a_top5", "pipeline_b_top5"), strict=True):
                retrieved = [_result(item) for item in prepared[field]]
                if case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                    unauthorized += sum(
                        item.document_id in set(case.forbidden_document_ids)
                        for item in retrieved
                    )
                timing = RetrievalTiming(
                    query_embedding_latency_ms=prepared["timing"]["query_embedding_ms"],
                    embedding_cache_lookup_latency_ms=prepared["timing"]["cache_lookup_ms"],
                    vector_search_latency_ms=prepared["timing"]["dense_ms"],
                    acl_filter_latency_ms=0.0,
                    query_embedding_cache_hit=True,
                    external_embedding_calls=0,
                )
                rag = RagService(
                    self.session,
                    FixedResultRetriever(provider, retrieved, timing),
                    ContextBuilder(1200),
                    ExtractiveGenerationProvider(),
                    gate,
                    supporting_context_only=True,
                )
                results[mode].append(EvaluationRunner(rag).run_case(evaluation, 5, 0.28))
        if unauthorized:
            raise RuntimeError("unauthorized content reached Sol")
        reporter = EvaluationRunner(rag)
        reports = {mode: reporter._report(tuple(rows)) for mode, rows in results.items()}
        analysis = self._holdout_analysis(results)
        for mode in MODES:
            report = reports[mode]
            self.session.add(
                EndToEndBenchmarkRunRecord(
                    dataset_id=DATASET_ID,
                    mode=mode,
                    metrics=report.metrics,
                    retrieval_metrics=analysis["retrieval"][mode],
                    category_metrics=report.category_metrics,
                    case_results=[asdict(item) for item in report.cases],
                    latency=self._e2e_latency(mode, report.cases, e2e.prepared_cases or []),
                    usage=self._e2e_usage(mode, report.cases),
                )
            )
        a, b = reports[MODES[0]].metrics, reports[MODES[1]].metrics
        holdout_runs = self._partition_runs("holdout")
        a_cov = holdout_runs[MODES[0]].metrics["all_required_evidence_coverage_at_5"]
        b_cov = holdout_runs[MODES[1]].metrics["all_required_evidence_coverage_at_5"]
        a_three = holdout_runs[MODES[0]].metrics["three_document_coverage_at_5"]
        b_three = holdout_runs[MODES[1]].metrics["three_document_coverage_at_5"]
        promote = (
            record.selected_retrieval_mode == MODES[1]
            and b_cov >= a_cov
            and b_three >= a_three
            and b["correct_answer_count"] >= a["correct_answer_count"]
            and b["incorrect_abstention_count"]
            <= a["incorrect_abstention_count"]
            + PROMOTION_POLICY["maximum_incorrect_abstention_increase"]
            and b["unsupported_answer_count"] <= a["unsupported_answer_count"]
            and b["acl_safety"] == 1.0
            and b["version_accuracy"] == 1.0
            and b.get("prompt_injection_boundary") in {1.0, None}
            and b["unauthorized_evidence_selection"] == 0
        )
        e2e = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        e2e.paired_transitions = analysis["transitions"]
        e2e.conversion_analysis = analysis
        e2e.regression_analysis = analysis["regressions"]
        e2e.production_retriever_status = MODES[1] if promote else MODES[0]
        e2e.completed_at = datetime.now(UTC)
        self.session.commit()
        return self.status()

    def _holdout_analysis(self, results: dict[str, list[CaseResult]]) -> dict[str, Any]:
        a = {case.case_id: case for case in results[MODES[0]]}
        b = {case.case_id: case for case in results[MODES[1]]}
        transitions: Counter[str] = Counter()
        rescues: list[dict[str, Any]] = []
        regressions: list[dict[str, Any]] = []
        for case_id, left in a.items():
            right = b[case_id]
            left_b = FrozenJudgeEndToEndBenchmark._behavior(left)
            right_b = FrozenJudgeEndToEndBenchmark._behavior(right)
            transitions[f"{left_b}→{right_b}"] += 1
            if (not left.retrieval_coverage_complete) and right.retrieval_coverage_complete:
                rescues.append({"case_id": case_id, "final_behavior": right_b})
            if left.retrieval_coverage_complete and not right.retrieval_coverage_complete:
                regressions.append({"case_id": case_id, "effect": f"{left_b}→{right_b}"})
        converted_answers = sum(item["final_behavior"] == "CORRECT_ANSWER" for item in rescues)
        remaining = sum(item["final_behavior"] == "INCORRECT_ABSTENTION" for item in rescues)
        generation = sum(
            item["final_behavior"] not in {"CORRECT_ANSWER", "INCORRECT_ABSTENTION"}
            for item in rescues
        )
        return {
            "transitions": dict(sorted(transitions.items())),
            "hybrid_retrieval_rescues": len(rescues),
            "rescues_to_correct_answer": converted_answers,
            "rescues_to_incorrect_abstention": remaining,
            "rescues_to_generation_or_other": generation,
            "conversion_rate": (converted_answers / len(rescues)) if rescues else None,
            "rescue_case_ids": [item["case_id"] for item in rescues],
            "regressions": {"count": len(regressions), "cases": regressions},
            "retrieval": {
                mode: self._e2e_slice(rows) for mode, rows in results.items()
            },
            "three_document": {
                mode: self._e2e_slice(
                    [case for case in rows if case.category == "multidoc_three"]
                )
                for mode, rows in results.items()
            },
            "abstention_safety": {
                mode: self._abstention_safety(rows) for mode, rows in results.items()
            },
        }

    @staticmethod
    def _e2e_slice(rows: list[CaseResult]) -> dict[str, Any]:
        if not rows:
            return {
                "retrieval_complete_rate": 0.0,
                "correct_answer_rate": 0.0,
                "incorrect_abstention_rate": 0.0,
                "unsupported_answer_rate": 0.0,
                "judge_false_negative_rate_after_complete_retrieval": None,
            }
        complete = [case for case in rows if case.retrieval_coverage_complete]
        complete_answerable = [case for case in complete if not case.expected_abstain]
        false_negatives = sum(case.status != "answered" for case in complete_answerable)
        return {
            "retrieval_complete_rate": mean(case.retrieval_coverage_complete for case in rows),
            "correct_answer_rate": mean(
                case.status == "answered" and not case.expected_abstain for case in rows
            ),
            "incorrect_abstention_rate": mean(
                case.status != "answered" and not case.expected_abstain for case in rows
            ),
            "unsupported_answer_rate": mean(
                case.status == "answered" and case.expected_abstain for case in rows
            ),
            "judge_false_negative_rate_after_complete_retrieval": (
                false_negatives / len(complete_answerable) if complete_answerable else None
            ),
        }

    @staticmethod
    def _abstention_safety(rows: list[CaseResult]) -> dict[str, Any]:
        should_abstain = [case for case in rows if case.expected_abstain]
        return {
            "should_abstain_count": len(should_abstain),
            "correct_abstentions": sum(case.status == "abstained" for case in should_abstain),
            "unsupported_answers": sum(case.status == "answered" for case in should_abstain),
            "judge_false_positives": sum(case.status == "answered" for case in should_abstain),
        }

    @staticmethod
    def _e2e_latency(
        mode: str, cases: tuple[CaseResult, ...], prepared: list[dict[str, Any]]
    ) -> dict[str, Any]:
        totals = []
        for case, item in zip(cases, prepared, strict=True):
            timing = item["timing"]
            retrieval = (
                timing["query_embedding_ms"]
                + timing["cache_lookup_ms"]
                + timing["dense_ms"]
                + (timing["bm25_ms"] + timing["rrf_ms"] if mode == MODES[1] else 0.0)
                + (timing["reranker_b_ms"] if mode == MODES[1] else timing["reranker_a_ms"])
            )
            totals.append(case.total_latency_ms + retrieval)
        return {
            "total_mean_ms": mean(totals),
            "total_p50_ms": median(totals),
            "total_p95_ms": _percentile(totals, 0.95),
            "judge_mean_ms": mean(case.answerability_judge_latency_ms for case in cases),
            "generation_mean_ms": mean(case.generation_latency_ms for case in cases),
        }

    def _e2e_usage(self, mode: str, cases: tuple[CaseResult, ...]) -> dict[str, Any]:
        return {
            "new_luna_judge_calls": 0,
            "new_sol_judge_calls": sum(case.external_judge_calls for case in cases),
            "sol_input_tokens": sum(case.judge_prompt_tokens or 0 for case in cases),
            "sol_output_tokens": sum(case.judge_completion_tokens or 0 for case in cases),
            "gate_cache_hits": sum(case.gate_cache_hit is True for case in cases),
            "gate_cache_misses": sum(case.gate_cache_hit is False for case in cases),
            "document_embedding_calls": 0,
            "external_reranker_calls": 0,
            "mode": mode,
        }

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(RetrievalBenchmarkRecord, DATASET_ID)
        payload: dict[str, Any] = {
            "dataset_id": DATASET_ID,
            "dataset_hash": DATASET_HASH,
            "case_count": len(CASES),
            "category_distribution": dict(sorted(Counter(item.category for item in CASES).items())),
            "maximum_prior_overlap": MAXIMUM_PRIOR_OVERLAP,
            "split_seed": SPLIT_SEED,
            "split_identity": SPLIT_IDENTITY,
            "calibration_count": len(CALIBRATION_CASES),
            "holdout_count": len(HOLDOUT_CASES),
            "configuration": CONFIGURATION,
            "selection_policy": SELECTION_POLICY,
            "production_judge_status": "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1",
            "embedding_preflight": self.embedding_preflight(),
            "selected_mode": None,
            "locked": False,
            "holdout_started": False,
            "holdout_completed": False,
            "e2e_completed": False,
            "production_retriever_status": "DENSE_CROSS_ENCODER_RERANK",
        }
        if not record:
            return payload
        self._verify(record)
        payload.update(
            {
                "selected_mode": record.selected_retrieval_mode,
                "selection_reason": record.selection_reason,
                "locked": record.locked_at is not None,
                "locked_at": record.locked_at,
                "holdout_started": record.holdout_started_at is not None,
                "holdout_completed": record.holdout_completed_at is not None,
                "calibration_metrics": record.calibration_metrics,
            }
        )
        runs: dict[str, dict[str, Any]] = {}
        for run in self.session.scalars(
            select(RetrievalBenchmarkRunRecord).where(
                RetrievalBenchmarkRunRecord.dataset_id == DATASET_ID
            )
        ).all():
            runs.setdefault(run.partition, {})[run.retrieval_mode] = {
                "metrics": run.metrics,
                "category_metrics": run.category_metrics,
                "latency": run.latency,
                "usage": run.usage,
                "branch_contribution": run.branch_contribution,
                **({"cases": run.case_results} if include_cases else {}),
            }
        payload["calibration"] = runs.get("calibration")
        payload["holdout"] = runs.get("holdout")
        e2e = self.session.get(EndToEndBenchmarkRecord, DATASET_ID)
        if e2e:
            payload["e2e_completed"] = e2e.completed_at is not None
            payload["judge_preflight"] = e2e.judge_preflight
            payload["production_retriever_status"] = (
                e2e.production_retriever_status or "DENSE_CROSS_ENCODER_RERANK"
            )
            payload["conversion_analysis"] = e2e.conversion_analysis
            payload["regressions"] = e2e.regression_analysis
            payload["paired_transitions"] = e2e.paired_transitions
            e2e_runs = {
                run.mode: {
                    "metrics": run.metrics,
                    "retrieval_metrics": run.retrieval_metrics,
                    "category_metrics": run.category_metrics,
                    "latency": run.latency,
                    "usage": run.usage,
                }
                for run in self.session.scalars(
                    select(EndToEndBenchmarkRunRecord).where(
                        EndToEndBenchmarkRunRecord.dataset_id == DATASET_ID
                    )
                ).all()
            }
            payload["end_to_end"] = e2e_runs or None
        return payload
