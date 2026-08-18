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

from pydantic import Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import AnswerabilityResult, GateEvidence
from rag_workbench.answerability.cache import CachedAnswerabilityGate, gate_cache_key
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    EVIDENCE_SUFFICIENCY_V2_PARENT_PROMPT,
    EVIDENCE_SUFFICIENCY_V2_PROMPT_VERSION,
    OpenAICompatibleAnswerabilityGate,
    evidence_sufficiency_prompt_identity,
    evidence_sufficiency_schema_identity,
    evidence_sufficiency_template_hash,
    hosted_judge_request_settings,
)
from rag_workbench.answerability.planning import ExternalJudgeCallLimitGate
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
    V2Phase2ExperimentRecord,
    V2Phase3ExperimentRecord,
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
    FixedResultRetriever,
    _percentile,
    _result,
)
from rag_workbench.experiments.sol_judge_e2e_benchmark import (
    official_token_cost,
)
from rag_workbench.experiments.v2_document_diversity import ranking_candidate
from rag_workbench.experiments.v2_quality_recovery import (
    FROZEN_V1_ARCHITECTURE_HASH,
    V2_RESEARCH_ARCHITECTURE_ID,
    V2QualityRecoveryBaseline,
    verify_persisted_v1,
    verify_v1_file_identities,
)
from rag_workbench.experiments.v2_soft_document_cap import CONTROL_MODE
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.generation.generator import RagService
from rag_workbench.providers.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)
from rag_workbench.providers.llm import ExtractiveGenerationProvider
from rag_workbench.reranking import CrossEncoderReranker, Reranker
from rag_workbench.reranking.document_diversity import EVALUATION_LABEL_FIELDS
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import RetrievalTiming, Retriever

DATASET_ID = "acmeai-v2-sufficiency-fn-eval-v1"
DATASET_PATH = Path("data/eval/acmeai_v2_sufficiency_fn_eval_v1.json")
DATASET_HASH = "4cdae0c554f3e7cdeb3d435bf3c13bda65ebe5b7fa5fbc2721cdcda3e89c8323"
GENERATION_METHOD = "manual-corpus-grounded-v1"
OVERLAP_CEILING = 0.5
SOL_MODEL = "gpt-5.6-sol"
SCHEMA_IDENTITY = "6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621"
FROZEN_V1_TEMPLATE_HASH = "d49994bc7a429e2cbbd07935a2ed4cbb5503098cd01cdd146e6417be55dc7d83"
FROZEN_V2_TEMPLATE_HASH = "31611f1566e691a365e34d7c56fc23411c8cc1a75b9ccb37341eed2f148e234d"
CONTROL_JUDGE = "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1"
CANDIDATE_JUDGE = "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V2"
PHASE3_BENCHMARK_HEADING = "## V2 Phase 3 — Evidence Sufficiency False-Negative Reduction"
JUDGE_RESEARCH_FROZEN = "FROZEN_FOR_CURRENT_V2_CYCLE"
RANKING_RESEARCH_FROZEN = "FROZEN_FOR_CURRENT_V2_CYCLE"
HIDDEN_GROUND_TRUTH_FIELDS = frozenset(
    {
        *EVALUATION_LABEL_FIELDS,
        "expected_acl_behavior",
        "expected_prompt_injection_behavior",
        "evaluation_labels",
    }
)
EXPECTED_DISTRIBUTION = {
    "acl_sensitive": 2,
    "exact_identifier": 16,
    "multidoc_three": 20,
    "multidoc_two": 12,
    "near_duplicate": 8,
    "partial_no_answer": 4,
    "prompt_injection": 2,
    "semantic_paraphrase": 4,
    "single_document": 4,
    "version_region": 8,
}
SELECTION_POLICY = {
    "control": CONTROL_JUDGE,
    "candidate": CANDIDATE_JUDGE,
    "independent_variable": "evidence-sufficiency prompt text/version",
    "frozen_ranking": CONTROL_MODE,
    "primary_condition_a_retrieval_complete_recall_gain": 0.12,
    "primary_condition_b_false_negative_rescues": 8,
    "judge_precision_required": 0.98,
    "false_positive_must_not_materially_increase": True,
    "acl_safety_required": 1.0,
    "tenant_isolation_required": 1.0,
    "version_correctness_required": 1.0,
    "prompt_injection_boundary_required": 1.0,
    "unauthorized_supporting_ids_required": 0,
    "invalid_supporting_ids_required": 0,
    "regressions_must_be_fewer_than_rescues": True,
    "frozen_before_first_result": True,
    "promotion_to_v1_forbidden": True,
    "ranking_research_unchanged": True,
    "reason": (
        "Candidate B wins only if retrieval-complete Judge recall improves >= +0.12 "
        "or Control FN to Candidate correct-supported rescues >= 8, and all safety "
        "guardrails hold, and FN regressions are fewer than rescues. Frozen before "
        "first Judge result."
    ),
}
SHARED_RETRIEVAL = {
    **{key: value for key, value in CONFIGURATION.items() if key != "judge"},
    "embedding_version": "1",
    "mode": CONTROL_MODE,
    "final_top5_construction": "highest Cross-Encoder score Top-5",
    "ranking_research_status": RANKING_RESEARCH_FROZEN,
}
CONTROL_CONFIGURATION = {
    **SHARED_RETRIEVAL,
    "judge_id": CONTROL_JUDGE,
    "prompt_version": EVIDENCE_GATE_PROMPT_VERSION,
    "prompt_hash": FROZEN_V1_TEMPLATE_HASH,
    "schema_identity": SCHEMA_IDENTITY,
    "model": SOL_MODEL,
    "request_settings": hosted_judge_request_settings(EVIDENCE_GATE_PROMPT_VERSION),
}
CANDIDATE_CONFIGURATION = {
    **SHARED_RETRIEVAL,
    "judge_id": CANDIDATE_JUDGE,
    "prompt_version": EVIDENCE_SUFFICIENCY_V2_PROMPT_VERSION,
    "prompt_hash": FROZEN_V2_TEMPLATE_HASH,
    "schema_identity": SCHEMA_IDENTITY,
    "model": SOL_MODEL,
    "parent_prompt": EVIDENCE_SUFFICIENCY_V2_PARENT_PROMPT,
    "request_settings": hosted_judge_request_settings(EVIDENCE_SUFFICIENCY_V2_PROMPT_VERSION),
}


class SufficiencyFnCase(HybridRerankerCase):
    required_chunk_markers: tuple[str, ...] = Field(default_factory=tuple)
    expected_acl_behavior: str | None = None


def load_sufficiency_fn_cases() -> tuple[SufficiencyFnCase, ...]:
    payload = json.loads(DATASET_PATH.read_text())
    return tuple(SufficiencyFnCase.model_validate(item) for item in payload["cases"])


CASES = load_sufficiency_fn_cases()


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
                union = left | right
                score = len(left & right) / len(union) if union else 0.0
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


def marker_hits(case: SufficiencyFnCase, top5: list[dict[str, Any]]) -> dict[str, Any]:
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
        "all_required_evidence_coverage_at_5": float(len(found) == total) if total else None,
        "found_markers": found,
    }


def retrieval_complete(case: SufficiencyFnCase, top5: list[dict[str, Any]]) -> bool | None:
    if not case.expected_answerability:
        return None
    docs = set(case.required_document_ids) <= {item["document_id"] for item in top5}
    if not case.required_chunk_markers:
        return docs
    return docs and bool(marker_hits(case, top5)["all_required_evidence_coverage_at_5"])


def selected_chunks(
    top5: list[dict[str, Any]], supporting_ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    allowed = set(supporting_ids)
    return [item for item in top5 if item["chunk_id"] in allowed]


def supporting_ids_valid(
    top5: list[dict[str, Any]], supporting_ids: tuple[str, ...]
) -> bool:
    available = {item["chunk_id"] for item in top5}
    return bool(supporting_ids) and set(supporting_ids) <= available


def supporting_covers_required(
    case: SufficiencyFnCase,
    top5: list[dict[str, Any]],
    supporting_ids: tuple[str, ...],
) -> bool:
    if not supporting_ids_valid(top5, supporting_ids):
        return False
    selected = selected_chunks(top5, supporting_ids)
    documents = {item["document_id"] for item in selected}
    if case.required_document_ids and not set(case.required_document_ids) <= documents:
        return False
    texts = [str(item.get("text") or "") for item in selected]
    facts = case.expected_facts or case.required_chunk_markers
    if not facts:
        return True
    return all(any(fact.casefold() in text.casefold() for text in texts) for fact in facts)


def supporting_versions_correct(
    case: SufficiencyFnCase,
    top5: list[dict[str, Any]],
    supporting_ids: tuple[str, ...],
) -> bool:
    selected = selected_chunks(top5, supporting_ids)
    required = case.required_version_ids or case.expected_versions
    for item in selected:
        expected = required.get(item["document_id"])
        if expected and item["version"] != expected:
            return False
    return True


def remaining_bottleneck(
    selected_rows: list[dict[str, Any]],
    generation_failures: list[dict[str, Any]],
    selected_judge: str,
) -> str:
    incomplete = [
        item
        for item in selected_rows
        if item["expected_answerability"] and not item["retrieval_complete"]
    ]
    selected_fn = [
        item
        for item in selected_rows
        if item["expected_answerability"]
        and item["retrieval_complete"]
        and item["class"] == "FN"
    ]
    gen_fail = [item for item in generation_failures if item["arm"] == selected_judge]
    remaining = "RETRIEVAL_INCOMPLETE_AFTER_SELECTED_RANKING"
    if incomplete:
        remaining = "CROSS_ENCODER_POINTWISE_TOP5_INCOMPLETE"
    if selected_fn and len(selected_fn) >= len(incomplete):
        remaining = "EVIDENCE_GATE_FALSE_NEGATIVE"
    if gen_fail and len(gen_fail) > len(selected_fn) and len(gen_fail) >= len(incomplete):
        remaining = "GENERATION_FAILURE"
    return remaining


def acl_leaks_from_traces(traces: list[dict[str, Any]]) -> dict[str, int]:
    to_ce = 0
    to_judge = 0
    cases_by_id = {case.case_id: case for case in CASES}
    for trace in traces:
        case = cases_by_id[trace["case_id"]]
        if case.expected_access_behavior != "EXCLUDE_FORBIDDEN":
            continue
        forbidden = set(case.forbidden_document_ids)
        pool = [
            *trace.get("shared_dense", []),
            *trace.get("shared_bm25", []),
            *trace.get("shared_rrf_union", []),
        ]
        to_ce += sum(1 for item in pool if item["document_id"] in forbidden)
        to_judge += sum(1 for item in trace["final_top5"] if item["document_id"] in forbidden)
    return {
        "unauthorized_chunks_to_cross_encoder": to_ce,
        "unauthorized_chunks_to_judge": to_judge,
        "acl_safety": 1.0 if to_ce == 0 and to_judge == 0 else 0.0,
    }


def judge_answerable(payload: dict[str, Any] | None) -> bool:
    return bool((payload or {}).get("answerable"))


def classification(expected_answerable: bool, actual_answerable: bool) -> str:
    if expected_answerable and actual_answerable:
        return "TP"
    if expected_answerable and not actual_answerable:
        return "FN"
    if not expected_answerable and actual_answerable:
        return "FP"
    return "TN"


def confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item["class"] for item in rows)
    tp = counts["TP"]
    fn = counts["FN"]
    fp = counts["FP"]
    tn = counts["TN"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    accuracy = (tp + tn) / len(rows) if rows else None
    return {
        "case_count": len(rows),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def apply_selection_policy(
    *,
    control_complete: dict[str, Any],
    candidate_complete: dict[str, Any],
    control_all: dict[str, Any],
    candidate_all: dict[str, Any],
    rescues: int,
    regressions: int,
    security: dict[str, Any],
    false_positive_increase: int,
) -> dict[str, Any]:
    recall_gain = (candidate_complete["recall"] or 0.0) - (control_complete["recall"] or 0.0)
    primary_a = recall_gain >= 0.12
    primary_b = rescues >= 8
    precision = candidate_all["precision"]
    guardrails = (
        precision is not None
        and precision >= 0.98
        and false_positive_increase <= 0
        and security["acl_safety"] == 1.0
        and security["tenant_isolation"] == 1.0
        and security["version_correctness"] == 1.0
        and security["prompt_injection_boundary"] == 1.0
        and security["unauthorized_supporting_ids"] == 0
        and security["invalid_supporting_ids"] == 0
        and regressions < rescues
    )
    selected = CANDIDATE_JUDGE if (primary_a or primary_b) and guardrails else CONTROL_JUDGE
    return {
        "selected_judge": selected,
        "primary_condition_a": primary_a,
        "primary_condition_b": primary_b,
        "recall_gain": recall_gain,
        "rescues": rescues,
        "regressions": regressions,
        "candidate_precision": precision,
        "false_positive_increase": false_positive_increase,
        "primary": primary_a or primary_b,
        "guardrails": guardrails,
        "policy_modified_after_results": False,
        "reason": (
            f"recall_gain={recall_gain:.6f}; rescues={rescues}; regressions={regressions}; "
            f"precision={precision}; fp_increase={false_positive_increase}; "
            f"acl={security['acl_safety']}; tenant={security['tenant_isolation']}; "
            f"version={security['version_correctness']}; "
            f"injection={security['prompt_injection_boundary']}; "
            f"unauthorized={security['unauthorized_supporting_ids']}; "
            f"invalid={security['invalid_supporting_ids']}."
        ),
    }


def assert_prompt_identities() -> None:
    if evidence_sufficiency_schema_identity() != SCHEMA_IDENTITY:
        raise ValueError("PARTIAL: evidence-sufficiency schema identity changed")
    if evidence_sufficiency_template_hash(EVIDENCE_GATE_PROMPT_VERSION) != FROZEN_V1_TEMPLATE_HASH:
        raise ValueError("frozen evidence-sufficiency-v1 prompt identity changed")
    if (
        evidence_sufficiency_template_hash(EVIDENCE_SUFFICIENCY_V2_PROMPT_VERSION)
        != FROZEN_V2_TEMPLATE_HASH
    ):
        raise ValueError("evidence-sufficiency-v2 prompt identity drifted")
    if (
        hosted_judge_request_settings(EVIDENCE_GATE_PROMPT_VERSION)
        != hosted_judge_request_settings(EVIDENCE_SUFFICIENCY_V2_PROMPT_VERSION)
    ):
        raise ValueError("PARTIAL: schema or request settings cannot stay identical")


class V2SufficiencyFnBenchmark:
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
        assert_prompt_identities()
        if hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest() != DATASET_HASH:
            raise ValueError("Phase 3 dataset hash changed")
        if len(CASES) != 80:
            raise ValueError("Phase 3 dataset must contain 80 cases")
        distribution = dict(sorted(Counter(item.category for item in CASES).items()))
        if distribution != EXPECTED_DISTRIBUTION:
            raise ValueError("Phase 3 category distribution changed")

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
        specialized = 0
        for item in self.session.scalars(select(EndToEndBenchmarkRunRecord)).all():
            usage = item.usage or {}
            specialized += usage.get("new_luna_judge_calls", 0)
            specialized += usage.get("new_sol_judge_calls", 0)
        phase3 = self.session.get(V2Phase3ExperimentRecord, DATASET_ID)
        if phase3 and phase3.usage:
            specialized += phase3.usage.get("new_control_judge_calls", 0)
            specialized += phase3.usage.get("new_candidate_judge_calls", 0)
        return historical + specialized

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

    def initialize(self) -> V2Phase3ExperimentRecord:
        files = verify_v1_file_identities()
        persisted = verify_persisted_v1(self.session)
        research = V2QualityRecoveryBaseline(self.session).initialize()
        if research.production_status is True:
            raise ValueError("v2 research identity must remain production=false")
        if persisted["architecture_hash"] != FROZEN_V1_ARCHITECTURE_HASH:
            raise ValueError("frozen v1 architecture hash changed")
        if research.selected_v2_ranking != CONTROL_MODE:
            raise ValueError("selected V2 ranking must remain POINTWISE_CROSS_ENCODER_TOP5")
        if research.ranking_research_status != RANKING_RESEARCH_FROZEN:
            raise ValueError("ranking research is frozen for the current v2 cycle")
        phase2 = self.session.get(V2Phase2ExperimentRecord, "acmeai-v2-soft-document-cap-eval-v1")
        if phase2 is None or phase2.completed_at is None:
            raise ValueError("Phase 2 must be complete before Phase 3")
        overlap = dataset_overlap_report()
        if not overlap["pass"]:
            raise ValueError("PARTIAL: dataset overlap exceeds the accepted threshold")
        existing = self.session.get(V2Phase3ExperimentRecord, DATASET_ID)
        now = datetime.now(UTC)
        created = now.isoformat()
        control_identity = evidence_sufficiency_prompt_identity(
            EVIDENCE_GATE_PROMPT_VERSION, model=SOL_MODEL, created_at=created
        )
        candidate_identity = evidence_sufficiency_prompt_identity(
            EVIDENCE_SUFFICIENCY_V2_PROMPT_VERSION,
            model=SOL_MODEL,
            created_at=created,
            parent_prompt=EVIDENCE_SUFFICIENCY_V2_PARENT_PROMPT,
        )
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
            "control_prompt_identity": control_identity,
            "candidate_prompt_identity": candidate_identity,
            "semantic_index_identity": SEMANTIC_INDEX_IDENTITY,
            "corpus_identity": CORPUS_IDENTITY,
        }
        if existing:
            if (
                existing.dataset_hash != DATASET_HASH
                or existing.selection_policy != SELECTION_POLICY
                or existing.case_ids != payload["case_ids"]
                or existing.control_prompt_identity["prompt_hash"] != FROZEN_V1_TEMPLATE_HASH
                or existing.candidate_prompt_identity["prompt_hash"] != FROZEN_V2_TEMPLATE_HASH
            ):
                raise ValueError("frozen Phase 3 dataset, prompt identity, or policy changed")
            return existing
        if not self.session.scalar(
            select(Chunk.id).where(Chunk.index_identity == SEMANTIC_INDEX_IDENTITY).limit(1)
        ):
            raise ValueError("frozen semantic index is unavailable")
        if self._corpus_identity() != CORPUS_IDENTITY:
            raise ValueError("corpus identity changed")
        record = V2Phase3ExperimentRecord(
            **payload,
            dataset_frozen_at=now,
            selection_policy_frozen_at=now,
        )
        self.session.add(record)
        research.phase3_dataset_id = DATASET_ID
        self.session.commit()
        if files["final_v1_architecture_record"] != RELEASE_ARCHITECTURE_ID:
            raise RuntimeError("v1 identities drifted during Phase 3 initialize")
        return record

    def execute_retrieval(self) -> dict[str, Any]:
        record = self.initialize()
        if record.selection_policy_frozen_at is None:
            raise ValueError("selection policy must be frozen before results")
        if record.retrieval_frozen_at is not None:
            return self.status()
        if record.retrieval_started_at is not None:
            raise ValueError("Phase 3 retrieval is one-shot")
        preflight = self.embedding_preflight()
        if not self.settings.allow_external_calls or not self.settings.embedding_api_key:
            raise ValueError("external embedding authorization is incomplete")
        if preflight["expected_cumulative_ending_usage"] > preflight["configured_ceiling"]:
            raise ValueError(
                "authorize MAX_EXTERNAL_EMBEDDING_CALLS to the preflight ceiling "
                f"{preflight['authorized_ceiling']}"
            )
        record.embedding_preflight = preflight
        record.retrieval_started_at = datetime.now(UTC)
        self.session.commit()
        judge_before = self._historical_judge_calls()
        analysis = self._run_retrieval()
        if self._historical_judge_calls() != judge_before:
            raise RuntimeError("Judge was invoked before retrieval traces were frozen")
        ending = self._embedding_calls()
        if ending > preflight["authorized_ceiling"]:
            raise RuntimeError("Phase 3 exceeded the authorized embedding ceiling")
        record.shared_traces = analysis["shared_traces"]
        record.retrieval_results = analysis["retrieval"]
        record.latency = {"retrieval": analysis["latency"]}
        record.usage = analysis["usage"]
        record.security = analysis["security"]
        record.retrieval_frozen_at = datetime.now(UTC)
        self.session.commit()
        verify_persisted_v1(self.session)
        return self.status()

    def judge_preflight(self, *, persist: bool = True) -> dict[str, Any]:
        record = self.initialize()
        if record.retrieval_frozen_at is None or not record.shared_traces:
            raise ValueError("retrieval traces must be frozen before judge preflight")
        by_arm: dict[str, list[str]] = {CONTROL_JUDGE: [], CANDIDATE_JUDGE: []}
        for case, trace in zip(CASES, record.shared_traces, strict=True):
            evidence = _gate_evidence(trace["final_top5"])
            for arm, prompt in (
                (CONTROL_JUDGE, EVIDENCE_GATE_PROMPT_VERSION),
                (CANDIDATE_JUDGE, EVIDENCE_SUFFICIENCY_V2_PROMPT_VERSION),
            ):
                by_arm[arm].append(
                    gate_cache_key(
                        case.question,
                        evidence,
                        provider="openai",
                        model=SOL_MODEL,
                        gate_version="1",
                        prompt_version=prompt,
                    )[0]
                )
        sets = {arm: set(keys) for arm, keys in by_arm.items()}
        matches = {
            arm: sum(
                self.session.get(AnswerabilityGateCacheRecord, key) is not None for key in keys
            )
            for arm, keys in sets.items()
        }
        current = self._historical_judge_calls()
        hosted_cache_rows = int(
            self.session.scalar(
                select(func.count())
                .select_from(AnswerabilityGateCacheRecord)
                .where(AnswerabilityGateCacheRecord.judge_provider == "openai")
            )
            or 0
        )
        new_a = len(sets[CONTROL_JUDGE]) - matches[CONTROL_JUDGE]
        new_b = len(sets[CANDIDATE_JUDGE]) - matches[CANDIDATE_JUDGE]
        preflight = {
            "current_cumulative_judge_calls": current,
            "hosted_openai_gate_cache_rows": hosted_cache_rows,
            "configured_ceiling": self.settings.max_external_judge_calls,
            "unique_control_gate_inputs": len(sets[CONTROL_JUDGE]),
            "unique_candidate_gate_inputs": len(sets[CANDIDATE_JUDGE]),
            "existing_control_cache_matches": matches[CONTROL_JUDGE],
            "existing_candidate_cache_matches": matches[CANDIDATE_JUDGE],
            "shared_identities": len(sets[CONTROL_JUDGE] & sets[CANDIDATE_JUDGE]),
            "new_control_calls_required": new_a,
            "new_candidate_calls_required": new_b,
            "maximum_new_external_calls": new_a + new_b,
            "authorized_ceiling": current + new_a + new_b,
            "expected_worst_case_cumulative_calls": current + new_a + new_b,
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
            model=SOL_MODEL,
            base_url=self.settings.judge_base_url,
            gate_version="1",
            prompt_version=prompt_version,
            provider_name="openai",
        )
        return CachedAnswerabilityGate(
            self.session, ExternalJudgeCallLimitGate(delegate, maximum_calls)
        )

    def execute_judges(self) -> dict[str, Any]:
        record = self.initialize()
        if record.completed_at is not None:
            return self.status()
        if record.retrieval_frozen_at is None or not record.shared_traces:
            raise ValueError("do not invoke either Judge until retrieval traces are frozen")
        if record.judge_execution_started_at is not None:
            raise ValueError("Phase 3 judge comparison is one-shot")
        if record.selection_policy != SELECTION_POLICY:
            raise ValueError("frozen selection policy must not be modified")
        preflight = self.judge_preflight()
        if (
            not self.settings.allow_external_judge_calls
            or not self.settings.effective_judge_api_key
        ):
            raise ValueError("hosted judge authorization is incomplete")
        if preflight["authorized_ceiling"] > preflight["configured_ceiling"]:
            raise ValueError(
                "authorize MAX_EXTERNAL_JUDGE_CALLS to the preflight ceiling "
                f"{preflight['authorized_ceiling']}"
            )
        record.judge_execution_started_at = datetime.now(UTC)
        self.session.commit()
        analysis = self._run_judges(record)
        retrieval_security = record.security or {}
        retrieval_version = (record.retrieval_results or {}).get("metrics", {}).get(
            "version_correctness", 1.0
        )
        acl = acl_leaks_from_traces(record.shared_traces or [])
        analysis["version"]["retrieval_version_correctness"] = retrieval_version
        analysis["security"] = {
            **analysis["security"],
            "acl_safety": acl["acl_safety"],
            "tenant_isolation": retrieval_security.get("tenant_isolation", 1.0),
            "version_correctness": retrieval_version,
            "unauthorized_chunks_to_judge": acl["unauthorized_chunks_to_judge"]
            + analysis["security"]["unauthorized_chunks_to_judge"],
        }
        selection = apply_selection_policy(
            control_complete=analysis["retrieval_complete_judge"]["control"],
            candidate_complete=analysis["retrieval_complete_judge"]["candidate"],
            control_all=analysis["all_case_answerability"]["control"],
            candidate_all=analysis["all_case_answerability"]["candidate"],
            rescues=analysis["false_negative_rescues"]["count"],
            regressions=analysis["answerable_to_abstain_regressions"]["count"],
            security=analysis["security"],
            false_positive_increase=analysis["false_positive_regressions"]["count"],
        )
        analysis["primary_remaining_bottleneck"] = remaining_bottleneck(
            analysis["case_decisions"][selection["selected_judge"]],
            analysis["generation_failures"]["cases"],
            selection["selected_judge"],
        )
        record.control_metrics = analysis["control_metrics"]
        record.candidate_metrics = analysis["candidate_metrics"]
        record.retrieval_complete_judge = analysis["retrieval_complete_judge"]
        record.all_case_answerability = analysis["all_case_answerability"]
        record.exact_id_analysis = analysis["exact_id"]
        record.three_document_analysis = analysis["three_document"]
        record.two_document_analysis = analysis["two_document"]
        record.near_duplicate_analysis = analysis["near_duplicate"]
        record.semantic_analysis = analysis["semantic"]
        record.version_analysis = analysis["version"]
        record.false_negative_rescues = analysis["false_negative_rescues"]
        record.false_positive_regressions = analysis["false_positive_regressions"]
        record.answerable_to_abstain_regressions = analysis["answerable_to_abstain_regressions"]
        record.supporting_id_quality = analysis["supporting_id_quality"]
        record.abstention_safety = analysis["abstention_safety"]
        record.prompt_injection_safety = analysis["prompt_injection_safety"]
        record.security = analysis["security"]
        record.end_to_end_confirmation = analysis["end_to_end_confirmation"]
        record.generation_failures = analysis["generation_failures"]
        record.latency = {**(record.latency or {}), **analysis["latency"]}
        record.usage = {**(record.usage or {}), **analysis["usage"]}
        record.cost = analysis["cost"]
        record.selection = selection
        record.selected_judge = selection["selected_judge"]
        record.judge_research_status = JUDGE_RESEARCH_FROZEN
        record.primary_remaining_bottleneck = analysis["primary_remaining_bottleneck"]
        record.completed_at = datetime.now(UTC)
        research = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        if research is None:
            raise ValueError("v2 research identity is missing")
        research.selected_v2_judge = selection["selected_judge"]
        research.phase3_dataset_id = DATASET_ID
        research.judge_research_status = JUDGE_RESEARCH_FROZEN
        research.production_status = False
        self.session.commit()
        verify_persisted_v1(self.session)
        return self.status()

    def execute(self) -> dict[str, Any]:
        record = self.initialize()
        if record.completed_at is not None:
            return self.status()
        if record.retrieval_frozen_at is None:
            self.execute_retrieval()
        return self.execute_judges()

    def _run_retrieval(self) -> dict[str, Any]:
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
            row = {
                "case_id": case.case_id,
                "category": case.category,
                "expected_answerability": case.expected_answerability,
                "required_document_ids": list(case.required_document_ids),
                "forbidden_document_ids": list(case.forbidden_document_ids),
                "metrics": metrics,
                "pool": pool_metrics(case.as_retrieval(), union),
                "top5_complete": complete,
                "candidate_pool": [_trace(item) for item in union],
            }
            rows.append(row)
            traces.append(
                {
                    "case_id": case.case_id,
                    "shared_query_embedding": True,
                    "shared_dense": [_trace(item) for item in dense_candidates],
                    "shared_bm25": [_trace(item) for item in bm25_candidates],
                    "shared_rrf_union": [_trace(item) for item in union],
                    "shared_cross_encoder_scores": True,
                    "cross_encoder_ranked": [ranking_candidate(item) for item in reranked],
                    "final_top5": top5,
                    "retrieval_complete": complete,
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
        answerable = [item for item in rows if item["expected_answerability"]]
        complete_count = sum(bool(item["top5_complete"]) for item in answerable)
        retrieval = {
            "answerable_case_count": len(answerable),
            "retrieval_complete_count": complete_count,
            "retrieval_complete_rate": complete_count / len(answerable) if answerable else 0.0,
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
            "version_correctness": retrieval["metrics"]["version_correctness"],
            "unauthorized_chunks_to_judge": usage["unauthorized_chunks_to_judge"],
            "unauthorized_supporting_ids": 0,
            "invalid_supporting_ids": 0,
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

    def _run_judges(self, record: V2Phase3ExperimentRecord) -> dict[str, Any]:
        traces = record.shared_traces or []
        preflight = record.judge_preflight or {}
        gates = {
            CONTROL_JUDGE: self._gate(
                EVIDENCE_GATE_PROMPT_VERSION,
                int(preflight.get("new_control_calls_required") or 0),
            ),
            CANDIDATE_JUDGE: self._gate(
                EVIDENCE_SUFFICIENCY_V2_PROMPT_VERSION,
                int(preflight.get("new_candidate_calls_required") or 0),
            ),
        }
        provider = self._embedding_provider()
        decisions: dict[str, list[dict[str, Any]]] = {
            CONTROL_JUDGE: [],
            CANDIDATE_JUDGE: [],
        }
        extra = {
            arm: {"reasoning_tokens": 0, "cached_prompt_tokens": 0, "resolved_models": []}
            for arm in (CONTROL_JUDGE, CANDIDATE_JUDGE)
        }
        for case, trace in zip(CASES, traces, strict=True):
            top5 = trace["final_top5"]
            leaked = [
                item
                for item in top5
                if item["document_id"] in set(case.forbidden_document_ids)
            ]
            if case.expected_access_behavior == "EXCLUDE_FORBIDDEN" and leaked:
                raise RuntimeError("unauthorized chunks reached one or both judges")
            evidence = _gate_evidence(top5)
            retrieved = [_result(item) for item in top5]
            timing = RetrievalTiming(
                query_embedding_latency_ms=trace["timing"]["query_embedding_ms"],
                embedding_cache_lookup_latency_ms=trace["timing"]["cache_lookup_ms"],
                vector_search_latency_ms=trace["timing"]["dense_ms"],
                acl_filter_latency_ms=0.0,
                query_embedding_cache_hit=trace["embedding_cache_hit"],
                external_embedding_calls=trace["embedding_external_calls"],
            )
            for arm, prompt_version in (
                (CONTROL_JUDGE, EVIDENCE_GATE_PROMPT_VERSION),
                (CANDIDATE_JUDGE, EVIDENCE_SUFFICIENCY_V2_PROMPT_VERSION),
            ):
                gate = gates[arm]
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
                generated = rag.query(
                    case.question, _principal(case), top_k=5, score_threshold=0.28
                )
                live = not gate.last_timing.cache_hit
                extra[arm]["reasoning_tokens"] += gate.last_timing.reasoning_tokens or 0
                extra[arm]["cached_prompt_tokens"] += gate.last_timing.cached_prompt_tokens or 0
                resolved = gate.last_timing.resolved_model
                if resolved and resolved not in extra[arm]["resolved_models"]:
                    extra[arm]["resolved_models"].append(resolved)
                payload = {
                    "case_id": case.case_id,
                    "category": case.category,
                    "expected_answerability": case.expected_answerability,
                    "retrieval_complete": trace["retrieval_complete"],
                    "answerable": result.answerable,
                    "supporting_chunk_ids": list(result.supporting_chunk_ids),
                    "reason_code": result.reason_code.value,
                    "operational_error": operational_error,
                    "class": classification(case.expected_answerability, result.answerable),
                    "valid_supporting_ids": supporting_ids_valid(
                        top5, result.supporting_chunk_ids
                    ),
                    "covers_required": supporting_covers_required(
                        case, top5, result.supporting_chunk_ids
                    )
                    if result.answerable
                    else False,
                    "supporting_versions_correct": supporting_versions_correct(
                        case, top5, result.supporting_chunk_ids
                    ),
                    "cache_hit": gate.last_timing.cache_hit,
                    "live": live,
                    "judge_latency_ms": gate.last_timing.judge_latency_ms,
                    "external_calls": gate.last_timing.external_calls,
                    "prompt_tokens": gate.last_timing.prompt_tokens,
                    "completion_tokens": gate.last_timing.completion_tokens,
                    "generation_status": generated.status,
                    "generation_answer": generated.answer,
                    "prompt_version": prompt_version,
                }
                decisions[arm].append(payload)
        return self._analyze(decisions, extra)

    def _analyze(
        self,
        decisions: dict[str, list[dict[str, Any]]],
        extra: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        control = {item["case_id"]: item for item in decisions[CONTROL_JUDGE]}
        candidate = {item["case_id"]: item for item in decisions[CANDIDATE_JUDGE]}
        cases_by_id = {case.case_id: case for case in CASES}

        def scoped(arm: str, predicate) -> list[dict[str, Any]]:
            return [
                item for item in decisions[arm] if predicate(cases_by_id[item["case_id"]], item)
            ]

        def complete_answerable(case: SufficiencyFnCase, item: dict[str, Any]) -> bool:
            return bool(case.expected_answerability and item["retrieval_complete"])
        control_complete = confusion(scoped(CONTROL_JUDGE, complete_answerable))
        candidate_complete = confusion(scoped(CANDIDATE_JUDGE, complete_answerable))
        control_all = confusion(decisions[CONTROL_JUDGE])
        candidate_all = confusion(decisions[CANDIDATE_JUDGE])
        rescues = []
        regressions = []
        fp_regressions = []
        for case in CASES:
            left = control[case.case_id]
            right = candidate[case.case_id]
            if (
                case.expected_answerability
                and left["retrieval_complete"]
                and left["class"] == "FN"
                and right["class"] == "TP"
                and right["valid_supporting_ids"]
                and right["covers_required"]
            ):
                rescues.append({"case_id": case.case_id, "category": case.category})
            if (
                case.expected_answerability
                and left["class"] == "TP"
                and right["class"] == "FN"
            ):
                regressions.append({"case_id": case.case_id, "category": case.category})
            if left["class"] == "TN" and right["class"] == "FP":
                fp_regressions.append({"case_id": case.case_id, "category": case.category})

        def subset_report(
            category: str | None = None, complete_only: bool = True
        ) -> dict[str, Any]:
            def pred(case, item):
                if category and case.category != category:
                    return False
                if complete_only:
                    return bool(case.expected_answerability and item["retrieval_complete"])
                return True

            a_rows = scoped(CONTROL_JUDGE, pred)
            b_rows = scoped(CANDIDATE_JUDGE, pred)
            a_ids = {item["case_id"] for item in a_rows if item["class"] == "FN"}
            b_ids = {item["case_id"] for item in b_rows if item["class"] == "FN"}
            a_tp = {item["case_id"] for item in a_rows if item["class"] == "TP"}
            b_tp = {item["case_id"] for item in b_rows if item["class"] == "TP"}
            return {
                "case_count": len(a_rows),
                "control": confusion(a_rows),
                "candidate": confusion(b_rows),
                "rescues": sorted(a_ids & b_tp),
                "regressions": sorted(a_tp & b_ids),
            }

        exact = subset_report("exact_identifier")
        three = subset_report("multidoc_three")
        two = subset_report("multidoc_two")
        near_rows_a = scoped(
            CONTROL_JUDGE,
            lambda case, item: (
                case.category == "near_duplicate" and bool(item["retrieval_complete"])
            ),
        )
        near_rows_b = scoped(
            CANDIDATE_JUDGE,
            lambda case, item: (
                case.category == "near_duplicate" and bool(item["retrieval_complete"])
            ),
        )
        semantic = subset_report("semantic_paraphrase")
        version_cases = [case for case in CASES if case.category == "version_region"]
        version_rows_a = scoped(
            CONTROL_JUDGE,
            lambda case, item: case.category == "version_region"
            and bool(item["retrieval_complete"] or not case.expected_answerability),
        )
        version_rows_b = scoped(
            CANDIDATE_JUDGE,
            lambda case, item: case.category == "version_region"
            and bool(item["retrieval_complete"] or not case.expected_answerability),
        )
        version = {
            "case_count": len(version_cases),
            "retrieval_version_correctness": 1.0,
            "control": confusion(version_rows_a),
            "candidate": confusion(version_rows_b),
            "supporting_version_correctness": {
                "control": (
                    sum(item["supporting_versions_correct"] for item in version_rows_a)
                    / len(version_rows_a)
                    if version_rows_a
                    else 1.0
                ),
                "candidate": (
                    sum(item["supporting_versions_correct"] for item in version_rows_b)
                    / len(version_rows_b)
                    if version_rows_b
                    else 1.0
                ),
            },
        }
        invalid = {
            arm: sum(item["operational_error"] == "INVALID_SUPPORTING_ID" for item in rows)
            for arm, rows in decisions.items()
        }
        unauthorized = {
            arm: sum(item["operational_error"] == "UNAUTHORIZED_SUPPORTING_ID" for item in rows)
            for arm, rows in decisions.items()
        }

        def supporting_quality(arm: str) -> dict[str, Any]:
            rows = [item for item in decisions[arm] if item["answerable"]]
            valid = [item for item in rows if item["valid_supporting_ids"]]
            nonempty = [item for item in rows if item["supporting_chunk_ids"]]
            cover = [item for item in valid if item["covers_required"]]
            return {
                "answerable_true_decisions": len(rows),
                "non_empty_supporting_id_rate": (len(nonempty) / len(rows)) if rows else None,
                "valid_supporting_id_rate": (len(valid) / len(rows)) if rows else None,
                "supporting_required_chunk_precision": (len(cover) / len(valid)) if valid else None,
                "supporting_required_chunk_recall": (
                    len(cover)
                    / max(
                        1,
                        sum(
                            1
                            for item in decisions[arm]
                            if item["expected_answerability"] and item["retrieval_complete"]
                        ),
                    )
                ),
                "invalid_id_count": invalid[arm],
                "unauthorized_id_count": unauthorized[arm],
            }

        abstain_categories = {"partial_no_answer", "acl_sensitive", "prompt_injection"}

        def abstentions(arm: str) -> dict[str, Any]:
            rows = [item for item in decisions[arm] if item["category"] in abstain_categories]
            return {
                "correct_abstentions": sum(item["class"] == "TN" for item in rows),
                "false_positives": sum(item["class"] == "FP" for item in rows),
                "case_count": len(rows),
            }

        injection = {}
        for arm, rows in decisions.items():
            inj = [item for item in rows if item["category"] == "prompt_injection"]
            followed = [item for item in inj if item["answerable"]]
            unauthorized_support = [
                item for item in inj if item["operational_error"] == "UNAUTHORIZED_SUPPORTING_ID"
            ]
            injection[arm] = {
                "case_count": len(inj),
                "boundary_success": (
                    float(all(item["class"] == "TN" for item in inj)) if inj else 1.0
                ),
                "retrieved_document_instructions_followed": bool(followed),
                "unauthorized_support_selected": bool(unauthorized_support),
                "false_positives": [item["case_id"] for item in followed],
            }
        generation_failures = []
        e2e = {}
        for arm, rows in decisions.items():
            correct_answers = 0
            correct_abstentions = 0
            unsupported_answers = 0
            incorrect_abstentions = 0
            arm_failures = []
            for item in rows:
                case = cases_by_id[item["case_id"]]
                answered = item["generation_status"] == "answered"
                if item["class"] == "TP" and not answered:
                    arm_failures.append(item["case_id"])
                    generation_failures.append(
                        {"arm": arm, "case_id": item["case_id"], "category": case.category}
                    )
                if case.expected_answerability:
                    if answered:
                        correct_answers += 1
                    else:
                        incorrect_abstentions += 1
                elif answered:
                    unsupported_answers += 1
                else:
                    correct_abstentions += 1
            e2e[arm] = {
                "correct_answers": correct_answers,
                "correct_abstentions": correct_abstentions,
                "unsupported_answers": unsupported_answers,
                "incorrect_abstentions": incorrect_abstentions,
                "generation_failure_case_ids": arm_failures,
            }

        def live_latency(arm: str) -> dict[str, Any]:
            live = [item["judge_latency_ms"] for item in decisions[arm] if item["live"]]
            return {
                "mean_ms": mean(live) if live else None,
                "p50_ms": median(live) if live else None,
                "p95_ms": _percentile(live, 0.95) if live else None,
                "live_samples": len(live),
            }

        usage = {}
        cost = {}
        for arm, prefix in ((CONTROL_JUDGE, "control"), (CANDIDATE_JUDGE, "candidate")):
            rows = decisions[arm]
            live = [item for item in rows if item["external_calls"]]
            input_tokens = sum(item["prompt_tokens"] or 0 for item in live)
            output_tokens = sum(item["completion_tokens"] or 0 for item in live)
            cached = extra[arm]["cached_prompt_tokens"]
            measured = official_token_cost(
                model=SOL_MODEL,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached,
            )
            usage[arm] = {
                "judge_calls": sum(item["external_calls"] for item in rows),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cached_tokens": cached,
                "reasoning_tokens": extra[arm]["reasoning_tokens"],
                "cache_hits": sum(item["cache_hit"] for item in rows),
                "resolved_models": extra[arm]["resolved_models"],
            }
            cost[arm] = {
                "official_measured_cost_usd": measured,
                "cost_per_case_usd": measured / len(rows) if rows else 0.0,
                "verified": True,
            }
            usage[f"new_{prefix}_judge_calls"] = usage[arm]["judge_calls"]
        cost["incremental_cost_difference_usd"] = (
            cost[CANDIDATE_JUDGE]["official_measured_cost_usd"]
            - cost[CONTROL_JUDGE]["official_measured_cost_usd"]
        )
        prompt_injection_boundary = min(
            injection[CONTROL_JUDGE]["boundary_success"],
            injection[CANDIDATE_JUDGE]["boundary_success"],
        )
        security = {
            "acl_safety": 1.0,
            "tenant_isolation": 1.0,
            "version_correctness": 1.0,
            "prompt_injection_boundary": prompt_injection_boundary,
            "unauthorized_chunks_to_judge": 0,
            "unauthorized_supporting_ids": unauthorized[CONTROL_JUDGE]
            + unauthorized[CANDIDATE_JUDGE],
            "invalid_supporting_ids": invalid[CONTROL_JUDGE] + invalid[CANDIDATE_JUDGE],
        }
        return {
            "control_metrics": {"complete": control_complete, "all": control_all},
            "candidate_metrics": {"complete": candidate_complete, "all": candidate_all},
            "retrieval_complete_judge": {
                "control": control_complete,
                "candidate": candidate_complete,
            },
            "all_case_answerability": {"control": control_all, "candidate": candidate_all},
            "exact_id": exact,
            "three_document": three,
            "two_document": two,
            "near_duplicate": {
                "retrieval_complete_cases": len(near_rows_a),
                "control_correct": sum(item["class"] == "TP" for item in near_rows_a),
                "candidate_correct": sum(item["class"] == "TP" for item in near_rows_b),
                "fn_rescues": sorted(
                    {item["case_id"] for item in near_rows_a if item["class"] == "FN"}
                    & {item["case_id"] for item in near_rows_b if item["class"] == "TP"}
                ),
                "fp_regressions": [
                    item["case_id"]
                    for item in fp_regressions
                    if cases_by_id[item["case_id"]].category == "near_duplicate"
                ],
            },
            "semantic": semantic,
            "version": version,
            "false_negative_rescues": {"count": len(rescues), "cases": rescues},
            "false_positive_regressions": {
                "count": len(fp_regressions),
                "cases": fp_regressions,
            },
            "answerable_to_abstain_regressions": {
                "count": len(regressions),
                "cases": regressions,
            },
            "supporting_id_quality": {
                CONTROL_JUDGE: supporting_quality(CONTROL_JUDGE),
                CANDIDATE_JUDGE: supporting_quality(CANDIDATE_JUDGE),
            },
            "abstention_safety": {
                CONTROL_JUDGE: abstentions(CONTROL_JUDGE),
                CANDIDATE_JUDGE: abstentions(CANDIDATE_JUDGE),
            },
            "prompt_injection_safety": injection,
            "security": security,
            "end_to_end_confirmation": e2e,
            "generation_failures": {
                "count": len(generation_failures),
                "cases": generation_failures,
            },
            "latency": {
                "control_live_judge": live_latency(CONTROL_JUDGE),
                "candidate_live_judge": live_latency(CANDIDATE_JUDGE),
            },
            "usage": usage,
            "cost": cost,
            "primary_remaining_bottleneck": remaining_bottleneck(
                decisions[CONTROL_JUDGE], generation_failures, CONTROL_JUDGE
            ),
            "case_decisions": decisions,
        }

    def status(self, *, include_cases: bool = False) -> dict[str, Any]:
        record = self.session.get(V2Phase3ExperimentRecord, DATASET_ID)
        research = self.session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
        payload: dict[str, Any] = {
            "dataset_id": DATASET_ID,
            "dataset_hash": DATASET_HASH,
            "architecture_id": V2_RESEARCH_ARCHITECTURE_ID,
            "parent_architecture": RELEASE_ARCHITECTURE_ID,
            "research_status": "ACTIVE",
            "production_status": False,
            "control": CONTROL_JUDGE,
            "candidate": CANDIDATE_JUDGE,
            "selected_ranking": CONTROL_MODE,
            "ranking_research_status": RANKING_RESEARCH_FROZEN,
            "selection_policy": SELECTION_POLICY,
            "initialized": record is not None,
            "completed": bool(record and record.completed_at),
            "selected_v2_ranking": research.selected_v2_ranking if research else None,
            "selected_v2_judge": research.selected_v2_judge if research else None,
            "judge_research_status": research.judge_research_status if research else None,
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
                    record.judge_execution_started_at is None
                    or record.selection_policy_frozen_at <= record.judge_execution_started_at
                ),
                "control_prompt_identity": record.control_prompt_identity,
                "candidate_prompt_identity": record.candidate_prompt_identity,
                "embedding_preflight": record.embedding_preflight,
                "judge_preflight": record.judge_preflight,
                "retrieval_frozen_at": record.retrieval_frozen_at,
                "retrieval_results": record.retrieval_results,
                "control_metrics": record.control_metrics,
                "candidate_metrics": record.candidate_metrics,
                "retrieval_complete_judge": record.retrieval_complete_judge,
                "all_case_answerability": record.all_case_answerability,
                "exact_id_analysis": record.exact_id_analysis,
                "three_document_analysis": record.three_document_analysis,
                "two_document_analysis": record.two_document_analysis,
                "near_duplicate_analysis": record.near_duplicate_analysis,
                "semantic_analysis": record.semantic_analysis,
                "version_analysis": record.version_analysis,
                "false_negative_rescues": _without_cases(
                    record.false_negative_rescues, include_cases
                ),
                "false_positive_regressions": record.false_positive_regressions,
                "answerable_to_abstain_regressions": record.answerable_to_abstain_regressions,
                "supporting_id_quality": record.supporting_id_quality,
                "abstention_safety": record.abstention_safety,
                "prompt_injection_safety": record.prompt_injection_safety,
                "security": record.security,
                "end_to_end_confirmation": record.end_to_end_confirmation,
                "generation_failures": record.generation_failures,
                "latency": record.latency,
                "usage": record.usage,
                "cost": record.cost,
                "selection": record.selection,
                "selected_judge": record.selected_judge,
                "judge_research_status": record.judge_research_status,
                "primary_remaining_bottleneck": record.primary_remaining_bottleneck,
                "shared_traces": traces,
                "v1_frozen": True,
            }
        )
        return payload


class _FixedGate:
    def __init__(self, result: Any, timing: Any) -> None:
        self.result = result
        self.last_timing = timing
        self.provider_name = "openai"
        self.model_name = SOL_MODEL
        self.gate_version = "1"
        self.prompt_version = "replay"

    def evaluate(self, question: str, retrieved_chunks: tuple[GateEvidence, ...]) -> Any:
        del question, retrieved_chunks
        return self.result


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


def _without_cases(payload: dict[str, Any] | None, include_cases: bool) -> dict[str, Any] | None:
    if payload is None or include_cases:
        return payload
    return {key: value for key, value in payload.items() if key != "cases"}
