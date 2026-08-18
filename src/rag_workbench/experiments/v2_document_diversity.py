from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from pydantic import Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import (
    Chunk,
    Document,
    DocumentVersion,
    ExperimentRunRecord,
    QueryEmbeddingCacheRecord,
    ResearchArchitectureRecord,
    V2Phase1ExperimentRecord,
)
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
    HybridRerankerCase,
    _principal,
    _trace,
    aggregate_pool,
    pool_metrics,
)
from rag_workbench.experiments.hybrid_reranker_replication import (
    RELEASE_ARCHITECTURE_ID,
)
from rag_workbench.experiments.reranker_e2e_benchmark import (
    CORPUS_IDENTITY,
    RERANKER_REVISION,
    SEMANTIC_INDEX_IDENTITY,
    _percentile,
)
from rag_workbench.experiments.v2_quality_recovery import (
    FROZEN_V1_ARCHITECTURE_HASH,
    V2_RESEARCH_ARCHITECTURE_ID,
    V2QualityRecoveryBaseline,
    verify_persisted_v1,
    verify_v1_file_identities,
)
from rag_workbench.providers.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from rag_workbench.reranking import CrossEncoderReranker, Reranker
from rag_workbench.reranking.document_diversity import (
    as_retrieval_result,
    select_document_diversified_top5,
)
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.retrieval.vector_search import RetrievalResult

DATASET_ID = "acmeai-v2-document-diversity-eval-v1"
DATASET_PATH = Path("data/eval/acmeai_v2_document_diversity_eval_v1.json")
DATASET_HASH = "79229d49eac1091f648632531019064f73e0e877038c0d1d88618c783dd502c1"
GENERATION_METHOD = "manual-corpus-grounded-v1"
OVERLAP_CEILING = 0.5
CONTROL_MODE = "POINTWISE_CROSS_ENCODER_TOP5"
CANDIDATE_MODE = "DOCUMENT_DIVERSIFIED_TOP5"
PHASE1_BENCHMARK_HEADING = "## V2 Phase 1 — Document-Diversified Top-5 Ranking"
EXPECTED_DISTRIBUTION = {
    "acl_sensitive": 2,
    "exact_identifier": 4,
    "multidoc_three": 36,
    "multidoc_two": 14,
    "near_duplicate": 6,
    "partial_no_answer": 2,
    "semantic_paraphrase": 4,
    "single_document": 8,
    "version_region": 4,
}
SELECTION_POLICY = {
    "control": CONTROL_MODE,
    "candidate": CANDIDATE_MODE,
    "primary_condition_a_three_document_coverage_gain": 0.15,
    "primary_condition_b_all_required_coverage_gain": 0.08,
    "required_recall_maximum_regression": 0.02,
    "same_document_multichunk_coverage_maximum_regression": 0.10,
    "exact_identifier_maximum_regression": 0.05,
    "semantic_maximum_regression": 0.05,
    "version_correctness_required": 1.0,
    "acl_safety_required": 1.0,
    "regression_count_must_not_exceed_rescue_count": True,
    "frozen_before_first_result": True,
    "promotion_to_v1_forbidden": True,
    "reason": (
        "Candidate B wins only if Three-document Coverage@5 improves >= +0.15 or "
        "All Required Evidence Coverage@5 improves >= +0.08, and all guardrails hold. "
        "A-complete to B-incomplete regressions must not exceed crowding rescues."
    ),
}
CONTROL_CONFIGURATION = {
    **CONFIGURATION,
    "mode": CONTROL_MODE,
    "final_top5_construction": "highest Cross-Encoder score Top-5",
    "independent_variable": "final Top-5 set construction",
}
CANDIDATE_CONFIGURATION = {
    **CONFIGURATION,
    "mode": CANDIDATE_MODE,
    "final_top5_construction": (
        "canonical document_id diversification over frozen Cross-Encoder ranking; "
        "highest-scoring chunk retained per document_id; no rescoring, MMR, or LLM"
    ),
    "independent_variable": "final Top-5 set construction",
}


class DocumentDiversityCase(HybridRerankerCase):
    requires_multiple_chunks_same_document: bool = False
    required_chunk_markers: tuple[str, ...] = Field(default_factory=tuple)


def load_document_diversity_cases() -> tuple[DocumentDiversityCase, ...]:
    payload = json.loads(DATASET_PATH.read_text())
    return tuple(DocumentDiversityCase.model_validate(item) for item in payload["cases"])


CASES = load_document_diversity_cases()


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
            for case in CASES:
                left = _token_terms(case.question)
                score = len(left & right) / len(left | right)
                if score > maximum:
                    maximum = score
                    closest = {
                        "case_id": case.case_id,
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


def ranking_candidate(item: Any) -> dict[str, Any]:
    result = item.result if hasattr(item, "result") else item
    return {
        "chunk_id": result.chunk_id,
        "document_id": result.document_id,
        "document_version_id": result.document_version_id,
        "text": result.text,
        "rank": getattr(item, "reranked_rank", result.rank),
        "score": getattr(item, "reranker_score", result.score),
        "source": result.source,
        "source_type": result.source_type,
        "title": result.title,
        "version": result.version,
        "page": result.page,
        "section": result.section,
        "metadata": result.metadata,
        "retrieval_source": result.retrieval_source,
        "dense_score": result.dense_score,
        "lexical_score": result.lexical_score,
        "fusion_score": result.fusion_score,
        "found_by_dense": result.found_by_dense,
        "found_by_bm25": result.found_by_bm25,
    }


def marker_hits(case: DocumentDiversityCase, top5: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [str(item.get("text") or "") for item in top5]
    found = [
        marker
        for marker in case.required_chunk_markers
        if any(marker.casefold() in text.casefold() for text in texts)
    ]
    total = len(case.required_chunk_markers)
    return {
        "required_marker_count": total,
        "retrieved_marker_count": len(found),
        "required_evidence_recall_at_5": (len(found) / total) if total else None,
        "all_required_evidence_coverage_at_5": float(len(found) == total) if total else None,
        "found_markers": found,
    }


def document_complete(case: DocumentDiversityCase, top5: list[dict[str, Any]]) -> bool | None:
    if not case.expected_answerability:
        return None
    return set(case.required_document_ids) <= {item["document_id"] for item in top5}


def retrieval_complete(case: DocumentDiversityCase, top5: list[dict[str, Any]]) -> bool | None:
    docs = document_complete(case, top5)
    if docs is None:
        return None
    if not case.required_chunk_markers:
        return docs
    return docs and bool(marker_hits(case, top5)["all_required_evidence_coverage_at_5"])


def unique_document_stats(top5: list[dict[str, Any]]) -> dict[str, Any]:
    document_ids = [item["document_id"] for item in top5]
    unique = list(dict.fromkeys(document_ids))
    repeated = len(document_ids) - len(unique)
    occupied = 0
    seen: set[str] = set()
    for document_id in document_ids:
        if document_id in seen:
            occupied += 1
        seen.add(document_id)
    return {
        "unique_document_ids": len(unique),
        "repeated_document_chunks": repeated,
        "duplicate_slot_fraction": (occupied / len(document_ids)) if document_ids else 0.0,
        "document_ids": document_ids,
    }


def diversity_aggregate(rows: list[dict[str, Any]], *, key: str) -> dict[str, Any]:
    stats = [unique_document_stats(item[key]) for item in rows]
    unique_counts = [item["unique_document_ids"] for item in stats]
    repeated = [item["repeated_document_chunks"] for item in stats]
    duplicate_fraction = [item["duplicate_slot_fraction"] for item in stats]
    return {
        "mean_unique_document_ids": mean(unique_counts) if unique_counts else 0.0,
        "median_unique_document_ids": median(unique_counts) if unique_counts else 0.0,
        "mean_repeated_document_chunks": mean(repeated) if repeated else 0.0,
        "percentage_top5_slots_occupied_by_already_represented_documents": (
            mean(duplicate_fraction) if duplicate_fraction else 0.0
        ),
    }


def classify_regression(case: DocumentDiversityCase, row: dict[str, Any]) -> str:
    if case.requires_multiple_chunks_same_document:
        return "MULTIPLE_REQUIRED_CHUNKS_SAME_DOCUMENT"
    if case.category == "near_duplicate":
        return "AUTHORITATIVE_SOURCE_REPLACEMENT"
    if case.category == "exact_identifier":
        return "EXACT_ID_DISPLACEMENT"
    return "OTHER"


def apply_selection_policy(
    control: dict[str, Any],
    candidate: dict[str, Any],
    *,
    regressions: int,
    rescues: int,
    same_doc: dict[str, Any],
) -> dict[str, Any]:
    coverage_gain = (
        candidate["all_required_evidence_coverage_at_5"]
        - control["all_required_evidence_coverage_at_5"]
    )
    three_gain = (
        candidate["three_document_coverage_at_5"] - control["three_document_coverage_at_5"]
    )
    recall_regression = (
        control["required_evidence_recall_at_5"] - candidate["required_evidence_recall_at_5"]
    )
    exact_regression = (
        control["exact_identifier_recall_at_5"] - candidate["exact_identifier_recall_at_5"]
    )
    semantic_regression = control["semantic_success"] - candidate["semantic_success"]
    same_doc_regression = (
        (same_doc["control_all_required_coverage_at_5"] or 0.0)
        - (same_doc["candidate_all_required_coverage_at_5"] or 0.0)
    )
    primary = coverage_gain >= 0.08 or three_gain >= 0.15
    guardrails = (
        recall_regression <= 0.02
        and same_doc_regression <= 0.10
        and exact_regression <= 0.05
        and semantic_regression <= 0.05
        and candidate["version_correctness"] == 1.0
        and candidate["acl_safety"] == 1.0
        and regressions <= rescues
    )
    selected = CANDIDATE_MODE if primary and guardrails else CONTROL_MODE
    return {
        "selected_ranking": selected,
        "primary_condition_a": three_gain >= 0.15,
        "primary_condition_b": coverage_gain >= 0.08,
        "coverage_gain": coverage_gain,
        "three_document_gain": three_gain,
        "required_recall_regression": recall_regression,
        "same_document_multichunk_coverage_regression": same_doc_regression,
        "exact_identifier_regression": exact_regression,
        "semantic_regression": semantic_regression,
        "regressions": regressions,
        "rescues": rescues,
        "primary": primary,
        "guardrails": guardrails,
        "policy_modified_after_results": False,
        "reason": (
            f"three_doc_gain={three_gain:.6f}; coverage_gain={coverage_gain:.6f}; "
            f"recall_regression={recall_regression:.6f}; "
            f"same_doc_regression={same_doc_regression:.6f}; "
            f"exact_regression={exact_regression:.6f}; "
            f"semantic_regression={semantic_regression:.6f}; "
            f"version={candidate['version_correctness']}; "
            f"acl={candidate['acl_safety']}; "
            f"regressions={regressions}; rescues={rescues}."
        ),
    }


class V2DocumentDiversityBenchmark:
    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        *,
        embedding_provider_factory: Any | None = None,
        reranker_factory: Any | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embedding_provider_factory = embedding_provider_factory
        self.reranker_factory = reranker_factory or (
            lambda: CrossEncoderReranker(device="cpu", resolved_revision=RERANKER_REVISION)
        )
        if hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() != DATASET_HASH:
            raise ValueError("v2 document-diversity dataset identity changed")
        if len(CASES) == 80:
            distribution = dict(sorted(Counter(item.category for item in CASES).items()))
            if distribution != EXPECTED_DISTRIBUTION:
                raise ValueError("v2 document-diversity dataset composition changed")
            if sum(item.requires_multiple_chunks_same_document for item in CASES) < 8:
                raise ValueError("same-document multi-chunk guardrail cases are missing")

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

    def _historical_judge_calls(self) -> int:
        historical = sum(
            item.external_judge_calls or 0
            for item in self.session.scalars(select(ExperimentRunRecord)).all()
        )
        return historical

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
        missing = len(keys) - matches
        authorized = current + missing
        return {
            "current_cumulative_embedding_calls": current,
            "new_dataset_query_count": len(questions),
            "existing_cache_matches": matches,
            "missing_unique_query_embeddings": missing,
            "authorized_ceiling": authorized,
            "configured_ceiling": self.settings.max_external_embedding_calls,
            "maximum_new_calls": missing,
            "expected_cumulative_ending_usage": authorized,
        }

    def initialize(self) -> V2Phase1ExperimentRecord:
        files = verify_v1_file_identities()
        persisted = verify_persisted_v1(self.session)
        research = V2QualityRecoveryBaseline(self.session).initialize()
        if research.production_status is True:
            raise ValueError("v2 research identity must remain production=false")
        if persisted["architecture_hash"] != FROZEN_V1_ARCHITECTURE_HASH:
            raise ValueError("frozen v1 architecture hash changed")
        overlap = dataset_overlap_report()
        if not overlap["pass"]:
            raise ValueError("PARTIAL: dataset overlap exceeds the accepted threshold")
        existing = self.session.get(V2Phase1ExperimentRecord, DATASET_ID)
        payload = {
            "dataset_id": DATASET_ID,
            "architecture_id": V2_RESEARCH_ARCHITECTURE_ID,
            "dataset_hash": DATASET_HASH,
            "case_ids": [item.case_id for item in CASES],
            "category_distribution": dict(
                sorted(Counter(item.category for item in CASES).items())
            ),
            "generation_method": GENERATION_METHOD,
            "maximum_prior_overlap": overlap["maximum_normalized_overlap"],
            "closest_previous_case": overlap["closest_prior_case"],
            "overlap_report": overlap,
            "selection_policy": SELECTION_POLICY,
            "control_configuration": CONTROL_CONFIGURATION,
            "candidate_configuration": CANDIDATE_CONFIGURATION,
            "semantic_index_identity": SEMANTIC_INDEX_IDENTITY,
            "corpus_identity": CORPUS_IDENTITY,
        }
        now = datetime.now(UTC)
        if existing:
            if (
                existing.dataset_hash != DATASET_HASH
                or existing.selection_policy != SELECTION_POLICY
                or existing.case_ids != payload["case_ids"]
            ):
                raise ValueError("frozen Phase 1 dataset or selection policy changed")
            return existing
        if not self.session.scalar(
            select(Chunk.id).where(Chunk.index_identity == SEMANTIC_INDEX_IDENTITY).limit(1)
        ):
            raise ValueError("frozen semantic index is unavailable")
        if self._corpus_identity() != CORPUS_IDENTITY:
            raise ValueError("corpus identity changed")
        record = V2Phase1ExperimentRecord(
            **payload,
            dataset_frozen_at=now,
            selection_policy_frozen_at=now,
        )
        self.session.add(record)
        research.phase1_dataset_id = DATASET_ID
        self.session.commit()
        if files["final_v1_architecture_record"] != RELEASE_ARCHITECTURE_ID:
            raise RuntimeError("v1 identities drifted during Phase 1 initialize")
        return record

    def execute(self) -> dict[str, Any]:
        record = self.initialize()
        if record.selection_policy_frozen_at is None:
            raise ValueError("selection policy must be frozen before results")
        if record.execution_started_at is not None:
            raise ValueError("Phase 1 retrieval is one-shot")
        if record.selection_policy != SELECTION_POLICY:
            raise ValueError("frozen selection policy must not be modified")
        preflight = self.embedding_preflight()
        if not self.settings.allow_external_calls or not self.settings.embedding_api_key:
            raise ValueError("external embedding authorization is incomplete")
        if preflight["expected_cumulative_ending_usage"] > preflight["configured_ceiling"]:
            raise ValueError(
                "authorize MAX_EXTERNAL_EMBEDDING_CALLS to the preflight ceiling "
                f"{preflight['authorized_ceiling']}"
            )
        record.embedding_preflight = preflight
        record.execution_started_at = datetime.now(UTC)
        self.session.commit()
        judge_before = self._historical_judge_calls()
        analysis = self._run()
        if self._historical_judge_calls() != judge_before:
            raise RuntimeError("Sol was used during Phase 1 retrieval")
        if analysis["usage"]["new_sol_calls"] or analysis["usage"]["generator_calls"]:
            raise RuntimeError("Phase 1 must not invoke Sol or generation")
        ending = self._embedding_calls()
        if ending > preflight["authorized_ceiling"]:
            raise RuntimeError("Phase 1 exceeded the authorized embedding ceiling")
        record.shared_traces = analysis["shared_traces"]
        record.control_metrics = analysis["control"]
        record.candidate_metrics = analysis["candidate"]
        record.diversity_metrics = analysis["diversity"]
        record.crowding_rescues = analysis["crowding_rescues"]
        record.diversification_regressions = analysis["regressions"]
        record.same_document_multichunk = analysis["same_document_multichunk"]
        record.three_document_analysis = analysis["three_document"]
        record.exact_id_analysis = analysis["exact_id"]
        record.near_duplicate_analysis = analysis["near_duplicate"]
        record.semantic_analysis = analysis["semantic"]
        record.version_analysis = analysis["version"]
        record.security = analysis["security"]
        record.latency = analysis["latency"]
        record.local_compute = analysis["local_compute"]
        record.usage = analysis["usage"]
        selection = apply_selection_policy(
            analysis["control"]["metrics"],
            analysis["candidate"]["metrics"],
            regressions=analysis["regressions"]["count"],
            rescues=analysis["crowding_rescues"]["total"],
            same_doc=analysis["same_document_multichunk"],
        )
        record.selection = selection
        record.selected_ranking = selection["selected_ranking"]
        record.primary_remaining_bottleneck = analysis["primary_remaining_bottleneck"][
            selection["selected_ranking"]
        ]
        record.completed_at = datetime.now(UTC)
        research = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        if research is None:
            raise ValueError("v2 research identity is missing")
        research.selected_v2_ranking = selection["selected_ranking"]
        research.phase1_dataset_id = DATASET_ID
        research.production_status = False
        self.session.commit()
        verify_persisted_v1(self.session)
        return self.status()

    def _run(self) -> dict[str, Any]:
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
            "new_sol_calls": 0,
            "judge_gate_calls": 0,
            "generator_calls": 0,
            "cross_encoder_pairs": 0,
            "unauthorized_chunks_to_cross_encoder": 0,
            "unauthorized_chunks_entering_candidate_b": 0,
        }
        traces: list[dict[str, Any]] = []
        control_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        for case in CASES:
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
            leaked_pool = []
            if case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                leaked_pool = [
                    item
                    for item in [*dense_candidates, *bm25_candidates, *union]
                    if item.document_id in forbidden
                ]
                if leaked_pool:
                    raise RuntimeError("unauthorized content reached RRF or Cross-Encoder")
            rerank_started = time.perf_counter()
            reranked = reranker.rerank(case.question, union)
            ce_ms = (time.perf_counter() - rerank_started) * 1000
            usage["cross_encoder_pairs"] += len(union)
            ranked_view = [ranking_candidate(item) for item in reranked]
            a_top5 = [
                replace(
                    item.result,
                    rank=item.reranked_rank,
                    score=item.reranker_score,
                    retrieval_source="pointwise_cross_encoder_top5",
                )
                for item in reranked[:FINAL_TOP_K]
            ]
            diversify_started = time.perf_counter()
            b_selected = select_document_diversified_top5(ranked_view, top_k=FINAL_TOP_K)
            b_top5 = [
                as_retrieval_result(
                    replace(
                        reranked[item["rank"] - 1].result,
                        rank=index,
                        score=item["score"],
                    ),
                    rank=index,
                    retrieval_source="document_diversified_top5",
                )
                for index, item in enumerate(b_selected, start=1)
            ]
            diversify_ms = (time.perf_counter() - diversify_started) * 1000
            if case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
                leaked_b = [item for item in b_top5 if item.document_id in forbidden]
                usage["unauthorized_chunks_entering_candidate_b"] += len(leaked_b)
                if leaked_b:
                    raise RuntimeError("unauthorized content entered Candidate B")
            timing = {
                "query_embedding_ms": embedding.embedding_latency_ms,
                "cache_lookup_ms": embedding.cache_lookup_latency_ms,
                "dense_ms": dense_ms,
                "bm25_ms": bm25_ms,
                "rrf_ms": fusion_ms,
                "cross_encoder_ms": ce_ms,
                "document_id_dedup_ms": diversify_ms,
                "selection_ms": diversify_ms,
                "incremental_b_ms": diversify_ms,
            }
            a_payload = self._case_payload(case, dense_candidates, bm25_candidates, union, a_top5)
            b_payload = self._case_payload(case, dense_candidates, bm25_candidates, union, b_top5)
            a_payload["markers"] = marker_hits(case, a_payload["top5"])
            b_payload["markers"] = marker_hits(case, b_payload["top5"])
            a_payload["retrieval_complete"] = retrieval_complete(case, a_payload["top5"])
            b_payload["retrieval_complete"] = retrieval_complete(case, b_payload["top5"])
            control_rows.append(a_payload)
            candidate_rows.append(b_payload)
            traces.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "shared_query_embedding": True,
                    "shared_dense": True,
                    "shared_bm25": True,
                    "shared_rrf_union": True,
                    "shared_cross_encoder_scores": True,
                    "cross_encoder_ranked": ranked_view,
                    "control_top5": a_payload["top5"],
                    "candidate_top5": b_payload["top5"],
                    "timing": timing,
                    "embedding_cache_hit": embedding.cache_hit,
                    "cross_encoder_pairs": len(union),
                    "requires_multiple_chunks_same_document": (
                        case.requires_multiple_chunks_same_document
                    ),
                }
            )
        return self._analyze(
            control_rows,
            candidate_rows,
            traces,
            usage,
            reranker,
            bm25.last_timing.index_size_bytes,
            bm25.last_timing.index_build_latency_ms,
        )

    def _case_payload(
        self,
        case: DocumentDiversityCase,
        dense: list[RetrievalResult],
        bm25: list[RetrievalResult],
        union: list[RetrievalResult],
        top5: list[RetrievalResult],
    ) -> dict[str, Any]:
        retrieval = case.as_retrieval()
        return {
            "case_id": case.case_id,
            "category": case.category,
            "expected_answerability": case.expected_answerability,
            "required_document_ids": list(case.required_document_ids),
            "forbidden_document_ids": list(case.forbidden_document_ids),
            "requires_multiple_chunks_same_document": (
                case.requires_multiple_chunks_same_document
            ),
            "top5": [_trace(item) for item in top5],
            "metrics": retrieval_case_metrics(retrieval, top5),
            "pool": pool_metrics(retrieval, union),
            "top5_complete": document_complete(case, [_trace(item) for item in top5]),
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
                    (item.rank for item in union if item.document_id == document_id), None
                )
                for document_id in case.required_document_ids
            },
        }

    def _analyze(
        self,
        control_rows: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        traces: list[dict[str, Any]],
        usage: dict[str, Any],
        reranker: Reranker,
        bm25_index_size: int,
        bm25_build_ms: float,
    ) -> dict[str, Any]:
        control = self._mode_analysis(control_rows)
        candidate = self._mode_analysis(candidate_rows)
        same_doc = self._same_document_subset(control_rows, candidate_rows)
        rescues = self._crowding_rescues(control_rows, candidate_rows, traces)
        regressions = self._regressions(control_rows, candidate_rows, traces)
        three = self._three_document(control_rows, candidate_rows, traces)
        exact = self._category_pair(control_rows, candidate_rows, "exact_identifier")
        near = self._near_duplicate(control_rows, candidate_rows)
        semantic = self._category_pair(control_rows, candidate_rows, "semantic_paraphrase")
        version = {
            "control_version_correctness": control["metrics"]["version_correctness"],
            "candidate_version_correctness": candidate["metrics"]["version_correctness"],
            "active_version_correctness": candidate["metrics"]["version_correctness"],
        }
        security = {
            "acl_safety": candidate["metrics"]["acl_safety"],
            "tenant_isolation": 1.0,
            "version_correctness": candidate["metrics"]["version_correctness"],
            "unauthorized_chunks_to_cross_encoder": usage["unauthorized_chunks_to_cross_encoder"],
            "unauthorized_chunks_entering_candidate_b": usage[
                "unauthorized_chunks_entering_candidate_b"
            ],
        }
        diversity = {
            "control": {
                **diversity_aggregate(control_rows, key="top5"),
                "by_category": {
                    name: diversity_aggregate(
                        [item for item in control_rows if item["category"] == name],
                        key="top5",
                    )
                    for name in ("single_document", "multidoc_two", "multidoc_three")
                },
            },
            "candidate": {
                **diversity_aggregate(candidate_rows, key="top5"),
                "by_category": {
                    name: diversity_aggregate(
                        [item for item in candidate_rows if item["category"] == name],
                        key="top5",
                    )
                    for name in ("single_document", "multidoc_two", "multidoc_three")
                },
            },
        }
        latency = {
            "shared_cross_encoder_mean_ms": mean(
                item["timing"]["cross_encoder_ms"] for item in traces
            ),
            "shared_cross_encoder_p50_ms": median(
                item["timing"]["cross_encoder_ms"] for item in traces
            ),
            "shared_cross_encoder_p95_ms": _percentile(
                [item["timing"]["cross_encoder_ms"] for item in traces], 0.95
            ),
            "document_id_dedup_mean_ms": mean(
                item["timing"]["document_id_dedup_ms"] for item in traces
            ),
            "selection_mean_ms": mean(item["timing"]["selection_ms"] for item in traces),
            "incremental_b_mean_ms": mean(item["timing"]["incremental_b_ms"] for item in traces),
            "query_embedding_mean_ms": mean(
                item["timing"]["query_embedding_ms"] for item in traces
            ),
            "dense_mean_ms": mean(item["timing"]["dense_ms"] for item in traces),
            "bm25_mean_ms": mean(item["timing"]["bm25_ms"] for item in traces),
            "rrf_mean_ms": mean(item["timing"]["rrf_ms"] for item in traces),
        }
        local_compute = {
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "cross_encoder_revision": reranker.resolved_revision,
            "device": getattr(reranker, "device", "cpu"),
            "candidate_pairs_per_query": mean(item["cross_encoder_pairs"] for item in traces),
            "total_ce_pairs": usage["cross_encoder_pairs"],
            "document_diversification_cpu_overhead_mean_ms": latency["incremental_b_mean_ms"],
            "bm25_index_size_bytes": bm25_index_size,
            "bm25_build_ms": bm25_build_ms,
        }
        remaining = {
            CONTROL_MODE: self._remaining_bottleneck(control_rows),
            CANDIDATE_MODE: self._remaining_bottleneck(candidate_rows),
        }
        return {
            "control": control,
            "candidate": candidate,
            "diversity": diversity,
            "crowding_rescues": rescues,
            "regressions": regressions,
            "same_document_multichunk": same_doc,
            "three_document": three,
            "exact_id": exact,
            "near_duplicate": near,
            "semantic": semantic,
            "version": version,
            "security": security,
            "latency": latency,
            "local_compute": local_compute,
            "usage": usage,
            "shared_traces": traces,
            "primary_remaining_bottleneck": remaining,
        }

    def _mode_analysis(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        metrics = aggregate_retrieval_metrics(rows)
        category = category_retrieval_metrics(rows)
        single = [item for item in rows if item["category"] == "single_document"]
        metrics["single_document_coverage_at_5"] = (
            mean(
                item["metrics"]["all_required_evidence_coverage_at_5"]
                for item in single
                if item["metrics"]["all_required_evidence_coverage_at_5"] is not None
            )
            if single
            else 0.0
        )
        semantic = category.get("semantic_paraphrase", {}).get(
            "all_required_evidence_coverage_at_5", 0.0
        )
        metrics["semantic_success"] = semantic
        pool_rows = [
            {
                "expected_answerability": item["expected_answerability"],
                "category": item["category"],
                "pool": item["pool"],
            }
            for item in rows
        ]
        return {
            "metrics": metrics,
            "category_metrics": category,
            "candidate_pool": aggregate_pool(pool_rows),
        }

    def _same_document_subset(
        self, control_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        cases = {
            item.case_id: item for item in CASES if item.requires_multiple_chunks_same_document
        }
        left = [item for item in control_rows if item["case_id"] in cases]
        right = [item for item in candidate_rows if item["case_id"] in cases]
        complete_to_incomplete = [
            item["case_id"]
            for item, other in zip(left, right, strict=True)
            if item["retrieval_complete"] and not other["retrieval_complete"]
        ]
        def coverage(rows: list[dict[str, Any]], field: str) -> float:
            values = [row["markers"][field] for row in rows if row["markers"][field] is not None]
            return mean(values) if values else 0.0

        return {
            "case_count": len(left),
            "control_required_evidence_recall_at_5": coverage(
                left, "required_evidence_recall_at_5"
            ),
            "candidate_required_evidence_recall_at_5": coverage(
                right, "required_evidence_recall_at_5"
            ),
            "control_all_required_coverage_at_5": coverage(
                left, "all_required_evidence_coverage_at_5"
            ),
            "candidate_all_required_coverage_at_5": coverage(
                right, "all_required_evidence_coverage_at_5"
            ),
            "a_complete_to_b_incomplete_cases": complete_to_incomplete,
        }

    def _crowding_rescues(
        self,
        control_rows: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        traces: list[dict[str, Any]],
    ) -> dict[str, Any]:
        by_id = {item["case_id"]: item for item in candidate_rows}
        traces_by_id = {item["case_id"]: item for item in traces}
        items: list[dict[str, Any]] = []
        for row in control_rows:
            other = by_id[row["case_id"]]
            if row["top5_complete"] or not other["top5_complete"]:
                continue
            if not row["pool"]["all_required_evidence_coverage"]:
                continue
            control_docs = {item["document_id"] for item in row["top5"]}
            missing = [
                document_id
                for document_id in row["required_document_ids"]
                if document_id not in control_docs
            ]
            ranked = traces_by_id[row["case_id"]]["cross_encoder_ranked"]
            duplicate_slots = unique_document_stats(row["top5"])["repeated_document_chunks"]
            items.append(
                {
                    "case_id": row["case_id"],
                    "category": row["category"],
                    "missing_required_documents": missing,
                    "a_ce_ranks": {
                        document_id: next(
                            (
                                item["rank"]
                                for item in ranked
                                if item["document_id"] == document_id
                            ),
                            None,
                        )
                        for document_id in missing
                    },
                    "a_final_rank_result": {
                        "top5_complete": row["top5_complete"],
                        "document_ids": [item["document_id"] for item in row["top5"]],
                    },
                    "b_final_rank_result": {
                        "top5_complete": other["top5_complete"],
                        "document_ids": [item["document_id"] for item in other["top5"]],
                    },
                    "duplicate_document_slots_removed": duplicate_slots,
                }
            )
        three = [item for item in items if item["category"] == "multidoc_three"]
        two = [item for item in items if item["category"] == "multidoc_two"]
        return {
            "total": len(items),
            "three_document": len(three),
            "two_document": len(two),
            "other": len(items) - len(three) - len(two),
            "cases": items,
        }

    def _regressions(
        self,
        control_rows: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        traces: list[dict[str, Any]],
    ) -> dict[str, Any]:
        cases = {item.case_id: item for item in CASES}
        by_id = {item["case_id"]: item for item in candidate_rows}
        items: list[dict[str, Any]] = []
        for row in control_rows:
            other = by_id[row["case_id"]]
            if not row["retrieval_complete"] or other["retrieval_complete"]:
                continue
            case = cases[row["case_id"]]
            items.append(
                {
                    "case_id": row["case_id"],
                    "category": row["category"],
                    "cause": classify_regression(case, row),
                    "a_top5": [item["document_id"] for item in row["top5"]],
                    "b_top5": [item["document_id"] for item in other["top5"]],
                    "a_markers": row["markers"],
                    "b_markers": other["markers"],
                }
            )
        counts = Counter(item["cause"] for item in items)
        return {
            "count": len(items),
            "causes": {
                "MULTIPLE_REQUIRED_CHUNKS_SAME_DOCUMENT": counts.get(
                    "MULTIPLE_REQUIRED_CHUNKS_SAME_DOCUMENT", 0
                ),
                "AUTHORITATIVE_SOURCE_REPLACEMENT": counts.get(
                    "AUTHORITATIVE_SOURCE_REPLACEMENT", 0
                ),
                "EXACT_ID_DISPLACEMENT": counts.get("EXACT_ID_DISPLACEMENT", 0),
                "OTHER": counts.get("OTHER", 0),
            },
            "cases": items,
        }

    def _three_document(
        self,
        control_rows: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        traces: list[dict[str, Any]],
    ) -> dict[str, Any]:
        left = [item for item in control_rows if item["category"] == "multidoc_three"]
        right = [item for item in candidate_rows if item["category"] == "multidoc_three"]
        traces_by_id = {item["case_id"]: item for item in traces}
        fixed: list[dict[str, Any]] = []
        worsened: list[str] = []
        both_failed: list[str] = []
        for item, other in zip(left, right, strict=True):
            if not item["top5_complete"] and other["top5_complete"]:
                fixed.append(
                    {
                        "case_id": item["case_id"],
                        "duplicate_document_slots_removed": unique_document_stats(item["top5"])[
                            "repeated_document_chunks"
                        ],
                        "a_top5": [entry["document_id"] for entry in item["top5"]],
                        "b_top5": [entry["document_id"] for entry in other["top5"]],
                        "ce_ranks": {
                            document_id: next(
                                (
                                    ranked["rank"]
                                    for ranked in traces_by_id[item["case_id"]][
                                        "cross_encoder_ranked"
                                    ]
                                    if ranked["document_id"] == document_id
                                ),
                                None,
                            )
                            for document_id in item["required_document_ids"]
                        },
                    }
                )
            elif item["top5_complete"] and not other["top5_complete"]:
                worsened.append(item["case_id"])
            elif not item["top5_complete"] and not other["top5_complete"]:
                both_failed.append(item["case_id"])
        return {
            "case_count": len(left),
            "candidate_pool_complete_rate": mean(
                item["pool"]["all_required_evidence_coverage"] or 0.0 for item in left
            )
            if left
            else 0.0,
            "control_top5_complete_rate": mean(float(item["top5_complete"] or 0) for item in left)
            if left
            else 0.0,
            "candidate_top5_complete_rate": mean(
                float(item["top5_complete"] or 0) for item in right
            )
            if right
            else 0.0,
            "control_required_evidence_recall": mean(
                item["metrics"]["required_evidence_recall_at_5"] or 0.0 for item in left
            )
            if left
            else 0.0,
            "candidate_required_evidence_recall": mean(
                item["metrics"]["required_evidence_recall_at_5"] or 0.0 for item in right
            )
            if right
            else 0.0,
            "b_fixed_cases": fixed,
            "b_worsened_cases": worsened,
            "both_failed_cases": both_failed,
        }

    def _category_pair(
        self,
        control_rows: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        category: str,
    ) -> dict[str, Any]:
        left = [item for item in control_rows if item["category"] == category]
        right = [item for item in candidate_rows if item["category"] == category]
        a_success = [
            item["case_id"]
            for item in left
            if item["metrics"]["all_required_evidence_coverage_at_5"]
        ]
        b_success = {
            item["case_id"]
            for item in right
            if item["metrics"]["all_required_evidence_coverage_at_5"]
        }
        regressions = [case_id for case_id in a_success if case_id not in b_success]
        return {
            "control_recall_at_5": mean(
                item["metrics"]["required_evidence_recall_at_5"] or 0.0 for item in left
            )
            if left
            else 0.0,
            "candidate_recall_at_5": mean(
                item["metrics"]["required_evidence_recall_at_5"] or 0.0 for item in right
            )
            if right
            else 0.0,
            "control_success": mean(
                item["metrics"]["all_required_evidence_coverage_at_5"] or 0.0 for item in left
            )
            if left
            else 0.0,
            "candidate_success": mean(
                item["metrics"]["all_required_evidence_coverage_at_5"] or 0.0 for item in right
            )
            if right
            else 0.0,
            "a_success_to_b_failure_count": len(regressions),
            "a_success_to_b_failure_cases": regressions,
        }

    def _near_duplicate(
        self, control_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        left = [item for item in control_rows if item["category"] == "near_duplicate"]
        right = [item for item in candidate_rows if item["category"] == "near_duplicate"]
        a_success = {
            item["case_id"]
            for item in left
            if item["metrics"]["preferred_source_success"]
        }
        b_success = {
            item["case_id"]
            for item in right
            if item["metrics"]["preferred_source_success"]
        }
        return {
            "control_preferred_source_success": mean(
                item["metrics"]["preferred_source_success"] or 0.0 for item in left
            )
            if left
            else 0.0,
            "candidate_preferred_source_success": mean(
                item["metrics"]["preferred_source_success"] or 0.0 for item in right
            )
            if right
            else 0.0,
            "a_to_b_gains": sorted(b_success - a_success),
            "a_to_b_regressions": sorted(a_success - b_success),
        }

    def _remaining_bottleneck(self, rows: list[dict[str, Any]]) -> str:
        answerable = [item for item in rows if item["expected_answerability"]]
        incomplete = [item for item in answerable if not item["top5_complete"]]
        pool_complete = [
            item for item in incomplete if item["pool"]["all_required_evidence_coverage"]
        ]
        if pool_complete:
            appearances = Counter()
            for item in pool_complete:
                stats = unique_document_stats(item["top5"])
                missing = [
                    document_id
                    for document_id in item["required_document_ids"]
                    if document_id not in {entry["document_id"] for entry in item["top5"]}
                ]
                if stats["repeated_document_chunks"] >= len(missing) and missing:
                    appearances["CROSS_ENCODER_SAME_DOCUMENT_TOP5_CROWDING"] += 1
                else:
                    appearances["CROSS_ENCODER_POINTWISE_MISORDERING"] += 1
            return appearances.most_common(1)[0][0]
        incomplete_three = [
            item
            for item in rows
            if item["category"] == "multidoc_three" and not item["top5_complete"]
        ]
        if incomplete_three:
            return "THREE_DOCUMENT_TOP5_INCOMPLETE"
        same = [
            item
            for item in answerable
            if item["requires_multiple_chunks_same_document"]
            and item["markers"]["all_required_evidence_coverage_at_5"] == 0.0
        ]
        if same:
            return "SAME_DOCUMENT_MULTI_CHUNK_LOSS"
        return "NO_REMAINING_RETRIEVAL_BOTTLENECK"

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(V2Phase1ExperimentRecord, DATASET_ID)
        research = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        payload: dict[str, Any] = {
            "dataset_id": DATASET_ID,
            "dataset_hash": DATASET_HASH,
            "architecture_id": V2_RESEARCH_ARCHITECTURE_ID,
            "parent_architecture": RELEASE_ARCHITECTURE_ID,
            "research_status": "ACTIVE",
            "production_status": False,
            "control": CONTROL_MODE,
            "candidate": CANDIDATE_MODE,
            "selection_policy": SELECTION_POLICY,
            "initialized": record is not None,
            "completed": bool(record and record.completed_at),
            "selected_v2_ranking": research.selected_v2_ranking if research else None,
        }
        if not record:
            return payload
        traces = record.shared_traces if include_cases else None
        payload.update(
            {
                "case_ids": record.case_ids,
                "case_count": len(record.case_ids),
                "category_distribution": record.category_distribution,
                "generation_method": record.generation_method,
                "maximum_prior_overlap": record.maximum_prior_overlap,
                "closest_previous_case": record.closest_previous_case,
                "dataset_frozen_at": record.dataset_frozen_at,
                "selection_policy_frozen_at": record.selection_policy_frozen_at,
                "policy_frozen_before_results": (
                    record.execution_started_at is None
                    or record.selection_policy_frozen_at <= record.execution_started_at
                ),
                "embedding_preflight": record.embedding_preflight,
                "control_metrics": record.control_metrics,
                "candidate_metrics": record.candidate_metrics,
                "diversity_metrics": record.diversity_metrics,
                "crowding_rescues": _without_cases(record.crowding_rescues, include_cases),
                "diversification_regressions": _without_cases(
                    record.diversification_regressions, include_cases
                ),
                "same_document_multichunk": record.same_document_multichunk,
                "three_document_analysis": record.three_document_analysis,
                "exact_id_analysis": record.exact_id_analysis,
                "near_duplicate_analysis": record.near_duplicate_analysis,
                "semantic_analysis": record.semantic_analysis,
                "version_analysis": record.version_analysis,
                "security": record.security,
                "latency": record.latency,
                "local_compute": record.local_compute,
                "usage": record.usage,
                "selection": record.selection,
                "selected_ranking": record.selected_ranking,
                "primary_remaining_bottleneck": record.primary_remaining_bottleneck,
                "shared_traces": traces,
                "v1_frozen": True,
            }
        )
        return payload


def _without_cases(payload: dict[str, Any] | None, include_cases: bool) -> dict[str, Any] | None:
    if payload is None or include_cases:
        return payload
    return {key: value for key, value in payload.items() if key != "cases"}
