from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import mean
from typing import TYPE_CHECKING, Any

from rag_workbench.evaluation.datasets import EvaluationCase, EvaluationDataset
from rag_workbench.evaluation.failures import FailureType
from rag_workbench.evaluation.generation_metrics import (
    abstention_classification,
    abstention_correct,
    answerability_classification,
    deterministic_citation_correctness,
    deterministic_citation_support,
)
from rag_workbench.evaluation.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_workbench.generation.generator import RagService
from rag_workbench.generation.prompts import build_grounded_prompt
from rag_workbench.security.permissions import Principal

if TYPE_CHECKING:
    from rag_workbench.runtime.canonical_runtime import CanonicalRagRuntime
    from rag_workbench.runtime.types import CanonicalQueryResult


def _is_canonical_runtime(runtime: Any) -> bool:
    from rag_workbench.runtime.canonical_runtime import CanonicalRagRuntime

    return isinstance(runtime, CanonicalRagRuntime)


@dataclass(frozen=True)
class _EvalRetrievalItem:
    chunk_id: str
    document_id: str
    document_version_id: str
    version: str
    rank: int
    score: float
    text: str


@dataclass(frozen=True)
class _NormalizedEvalResponse:
    """Presentation-normalized response shared by legacy and canonical eval scoring."""

    run_id: str | None
    status: str  # answered | abstained (legacy metric vocabulary)
    answer: str | None
    citations: tuple[Any, ...]
    retrieval_results: tuple[_EvalRetrievalItem, ...]
    supporting_chunk_ids: tuple[str, ...]
    generation_context_chunk_ids: tuple[str, ...]
    retrieval_latency_ms: float = 0.0
    query_embedding_latency_ms: float = 0.0
    embedding_cache_lookup_latency_ms: float = 0.0
    vector_search_latency_ms: float = 0.0
    acl_filter_latency_ms: float = 0.0
    context_construction_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    embedding_tokens: int | None = None
    query_embedding_cache_hit: bool = False
    external_embedding_calls: int = 0
    answerability_result: Any | None = None
    answerability_gate_cache_lookup_latency_ms: float = 0.0
    answerability_judge_latency_ms: float = 0.0
    context_pruning_latency_ms: float = 0.0
    gate_cache_hit: bool | None = None
    external_judge_calls: int = 0
    judge_prompt_tokens: int | None = None
    judge_completion_tokens: int | None = None
    local_judge_calls: int = 0
    answerability_operational_error: str | None = None
    error_class: str | None = None
    route: str | None = None
    context_document_ids: tuple[str, ...] = ()
    dropped_chunk_ids: tuple[str, ...] = ()
    gate_enabled: bool = False
    injection_precheck_blocked: bool = False


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    question: str
    expected_answer: str | None
    expected_document_ids: tuple[str, ...]
    expected_chunk_ids: tuple[str, ...]
    forbidden_document_ids: tuple[str, ...]
    expected_versions: dict[str, str]
    required_fact_ids: tuple[str, ...]
    unavailable_required_fact_ids: tuple[str, ...]
    security_checks: tuple[str, ...]
    principal: dict[str, Any]
    expected_abstain: bool
    retrieved_document_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    retrieval_trace: tuple[dict[str, Any], ...]
    status: str
    answer: str | None
    citations: tuple[dict[str, Any], ...]
    rag_run_id: str | None
    hit_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    abstention_accuracy: float
    citation_correctness: float | None
    citation_support: float | None
    version_correct: float | None
    security_passed: bool | None
    failure_type: str | None
    failure_types: tuple[str, ...]
    failure_details: dict[str, Any]
    retrieval_latency_ms: float
    query_embedding_latency_ms: float
    embedding_cache_lookup_latency_ms: float
    vector_search_latency_ms: float
    acl_filter_latency_ms: float
    context_construction_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    embedding_tokens: int | None
    query_embedding_cache_hit: bool
    external_embedding_calls: int
    supporting_chunk_ids: tuple[str, ...]
    generation_context_chunk_ids: tuple[str, ...]
    answerability_result: dict[str, Any] | None
    answerability_gate_cache_lookup_latency_ms: float
    answerability_judge_latency_ms: float
    context_pruning_latency_ms: float
    gate_cache_hit: bool | None
    external_judge_calls: int
    judge_prompt_tokens: int | None
    judge_completion_tokens: int | None
    local_judge_calls: int
    answerability_operational_error: str | None
    retrieval_coverage_complete: bool
    required_evidence_recall: float | None
    required_evidence_precision: float | None
    all_required_evidence_coverage: float | None
    supporting_context_loss: float | None
    error: str | None = None

    @property
    def metrics(self) -> dict[str, float | None]:
        return {
            "hit_at_k": self.hit_at_k,
            "recall_at_k": self.recall_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "ndcg_at_k": self.ndcg_at_k,
            "abstention_accuracy": self.abstention_accuracy,
            "citation_correctness": self.citation_correctness,
            "citation_validity": self.citation_correctness,
            "citation_support": self.citation_support,
            "version_correct": self.version_correct,
            "retrieval_coverage_complete": float(self.retrieval_coverage_complete),
            "required_evidence_recall": self.required_evidence_recall,
            "required_evidence_precision": self.required_evidence_precision,
            "all_required_evidence_coverage": self.all_required_evidence_coverage,
            "supporting_context_loss": self.supporting_context_loss,
        }


@dataclass(frozen=True)
class EvaluationReport:
    case_count: int
    metrics: dict[str, float | None]
    category_metrics: dict[str, dict[str, float | None]]
    cases: tuple[CaseResult, ...]
    failed_case_count: int = 0
    pending_generation_metrics: tuple[str, ...] = (
        "answer_correctness",
        "groundedness",
        "answer_completeness",
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "case_count": self.case_count,
            "failed_case_count": self.failed_case_count,
            "metrics": self.metrics,
            "category_metrics": self.category_metrics,
            "cases": [asdict(case) for case in self.cases],
            "pending_generation_metrics": list(self.pending_generation_metrics),
        }


class EvaluationRunner:
    """Scores evaluation cases against a RAG inference runtime.

    Production path: ``CanonicalRagRuntime`` (same semantics as ``POST /rag/query``).
    Legacy ``RagService`` remains accepted only for historical tests.
    """

    def __init__(
        self,
        runtime: CanonicalRagRuntime | RagService,
        tenant_id: str = "acmeai",
    ) -> None:
        self.runtime = runtime
        # Back-compat alias for historical callers/tests.
        self.rag_service = runtime if isinstance(runtime, RagService) else None
        self.tenant_id = tenant_id

    def run(
        self,
        cases: EvaluationDataset | list[EvaluationCase] | tuple[EvaluationCase, ...],
        top_k: int = 5,
        score_threshold: float | None = 0.2,
    ) -> EvaluationReport:
        results = tuple(self.run_case(case, top_k, score_threshold) for case in cases)
        return self._report(results)

    def _execute(self, case: EvaluationCase, top_k: int, score_threshold: float | None):
        principal = Principal(
            principal_id=case.principal.principal_id or f"eval:{case.case_id}",
            tenant_id=case.principal.tenant_id or self.tenant_id,
            permission_groups=frozenset(case.principal.permission_groups),
        )
        if _is_canonical_runtime(self.runtime):
            # top_k / score_threshold are ignored — ProductionRagConfig stage depths apply.
            del top_k, score_threshold
            return self._from_canonical(
                self.runtime.query(case.question, principal=principal, include_debug=True)
            )
        assert self.rag_service is not None
        return self._from_legacy(
            self.rag_service.query(
                case.question,
                principal=principal,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        )

    @staticmethod
    def _from_canonical(result: CanonicalQueryResult) -> _NormalizedEvalResponse:
        retrieval = tuple(
            _EvalRetrievalItem(
                chunk_id=str(item.get("chunk_id") or ""),
                document_id=str(item.get("document_id") or ""),
                document_version_id=str(item.get("document_version_id") or ""),
                version=str(item.get("version") or ""),
                rank=int(item.get("rank") or 0),
                score=float(item.get("score") or 0.0),
                text=str(item.get("text") or ""),
            )
            for item in result.retrieval_results
        )
        cited_chunk_ids = tuple(citation.chunk_id for citation in result.citations)
        status = "answered" if result.status == "answer" else "abstained"
        latency = float((result.trace or {}).get("latency_ms") or 0.0)
        luna = int((result.trace or {}).get("luna_calls") or 0)
        sol = int((result.trace or {}).get("sol_calls") or 0)
        return _NormalizedEvalResponse(
            run_id=result.run_id or result.request_id,
            status=status,
            answer=result.answer,
            citations=result.citations,
            retrieval_results=retrieval,
            supporting_chunk_ids=cited_chunk_ids,
            generation_context_chunk_ids=cited_chunk_ids,
            total_latency_ms=latency,
            external_judge_calls=luna + sol,
            error_class=result.error_class,
            route=result.route,
            context_document_ids=tuple(
                dict.fromkeys(item.document_id for item in retrieval if item.document_id)
            ),
            dropped_chunk_ids=(),
            gate_enabled=True,
            injection_precheck_blocked=result.error_class == "PROMPT_INJECTION",
        )

    def _from_legacy(self, response: Any) -> _NormalizedEvalResponse:
        assert self.rag_service is not None
        retrieval = tuple(
            _EvalRetrievalItem(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                version=item.version,
                rank=item.rank,
                score=item.score,
                text=item.text,
            )
            for item in response.retrieval_results
        )
        context = self.rag_service.context_builder.build(response.retrieval_results)
        return _NormalizedEvalResponse(
            run_id=response.run_id,
            status=response.status,
            answer=response.answer,
            citations=response.citations,
            retrieval_results=retrieval,
            supporting_chunk_ids=response.supporting_chunk_ids,
            generation_context_chunk_ids=response.generation_context_chunk_ids,
            retrieval_latency_ms=response.retrieval_latency_ms,
            query_embedding_latency_ms=response.query_embedding_latency_ms,
            embedding_cache_lookup_latency_ms=response.embedding_cache_lookup_latency_ms,
            vector_search_latency_ms=response.vector_search_latency_ms,
            acl_filter_latency_ms=response.acl_filter_latency_ms,
            context_construction_latency_ms=response.context_construction_latency_ms,
            generation_latency_ms=response.generation_latency_ms,
            total_latency_ms=response.total_latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            embedding_tokens=response.embedding_tokens,
            query_embedding_cache_hit=response.query_embedding_cache_hit,
            external_embedding_calls=response.external_embedding_calls,
            answerability_result=response.answerability_result,
            answerability_gate_cache_lookup_latency_ms=(
                response.answerability_gate_cache_lookup_latency_ms
            ),
            answerability_judge_latency_ms=response.answerability_judge_latency_ms,
            context_pruning_latency_ms=response.context_pruning_latency_ms,
            gate_cache_hit=response.gate_cache_hit,
            external_judge_calls=response.external_judge_calls,
            judge_prompt_tokens=response.judge_prompt_tokens,
            judge_completion_tokens=response.judge_completion_tokens,
            local_judge_calls=response.local_judge_calls,
            answerability_operational_error=response.answerability_operational_error,
            context_document_ids=tuple(item.result.document_id for item in context.items),
            dropped_chunk_ids=tuple(context.dropped_chunk_ids),
            gate_enabled=response.answerability_result is not None,
            injection_precheck_blocked=False,
        )

    def run_case(
        self, case: EvaluationCase, top_k: int, score_threshold: float | None
    ) -> CaseResult:
        try:
            response = self._execute(case, top_k, score_threshold)
            retrieved_documents = tuple(item.document_id for item in response.retrieval_results)
            retrieved_chunks = tuple(item.chunk_id for item in response.retrieval_results)
            evaluate_chunks = bool(case.expected_chunk_ids)
            retrieved = retrieved_chunks if evaluate_chunks else retrieved_documents
            relevant = set(
                case.expected_chunk_ids if evaluate_chunks else case.expected_document_ids
            )
            recall = recall_at_k(retrieved, relevant, top_k)
            citations = tuple(
                citation.to_dict() if hasattr(citation, "to_dict") else citation
                for citation in response.citations
            )
            cited_ids = [citation["chunk_id"] for citation in citations]
            citation_correctness = deterministic_citation_correctness(cited_ids, retrieved_chunks)
            citation_support = deterministic_citation_support(
                answer=response.answer,
                expected_answer=case.expected_answer,
                cited_document_ids=(citation["document_id"] for citation in citations),
                expected_document_ids=case.expected_document_ids,
                should_abstain=case.should_abstain,
            )
            forbidden_found = bool(set(retrieved_documents) & set(case.forbidden_document_ids))
            security_passed = None
            if case.category == "access_control" or "acl" in case.security_checks:
                security_passed = not forbidden_found
            if case.category == "prompt_injection" or "prompt_injection" in case.security_checks:
                if response.injection_precheck_blocked:
                    injection_passed = response.status == "abstained" and not response.answer
                elif self.rag_service is not None:
                    # Legacy path: prompt-boundary check against extractive grounded prompt.
                    from rag_workbench.retrieval.vector_search import RetrievalResult

                    legacy_hits = [
                        RetrievalResult(
                            chunk_id=item.chunk_id,
                            document_id=item.document_id,
                            document_version_id=item.document_version_id,
                            text=item.text,
                            rank=item.rank,
                            score=item.score,
                            source="",
                            source_type="",
                            title="",
                            version=item.version,
                        )
                        for item in response.retrieval_results
                    ]
                    bundle = self.rag_service.context_builder.build(legacy_hits)
                    prompt = build_grounded_prompt(case.question, bundle)
                    boundary_ok = (
                        "<untrusted_retrieved_context>" in prompt
                        and "</untrusted_retrieved_context>" in prompt
                        and prompt.index("untrusted data")
                        < prompt.index("<untrusted_retrieved_context>")
                        and case.question not in prompt
                    )
                    injection_passed = (
                        not forbidden_found and boundary_ok and citation_correctness != 0.0
                    )
                else:
                    injection_passed = response.status == "abstained" and not response.answer
                security_passed = (
                    injection_passed
                    if security_passed is None
                    else security_passed and injection_passed
                )
            version_correct = self._version_correct(case, response.retrieval_results)
            required_documents = set(case.expected_document_ids)
            retrieved_document_set = set(retrieved_documents)
            supporting_documents = {
                item.document_id
                for item in response.retrieval_results
                if item.chunk_id in set(response.supporting_chunk_ids)
            }
            retrieval_coverage_complete = bool(required_documents) and not (
                case.unavailable_required_fact_ids
            ) and required_documents <= retrieved_document_set
            required_evidence_recall = (
                len(required_documents & supporting_documents) / len(required_documents)
                if required_documents
                else None
            )
            required_evidence_precision = (
                len(required_documents & supporting_documents) / len(supporting_documents)
                if supporting_documents
                else (0.0 if required_documents else None)
            )
            all_required_evidence_coverage = (
                float(
                    not case.unavailable_required_fact_ids
                    and required_documents <= supporting_documents
                )
                if required_documents
                else None
            )
            supporting_context_loss = (
                float(
                    retrieval_coverage_complete
                    and not bool(all_required_evidence_coverage)
                )
                if required_documents
                else None
            )
            failure_types, failure_details = self._classify_failures(
                case,
                recall,
                response.status,
                citation_correctness,
                security_passed,
                version_correct,
                retrieved_documents,
                response.retrieval_results,
                response.context_document_ids,
                response.dropped_chunk_ids,
                score_threshold,
                response.gate_enabled,
                response.supporting_chunk_ids,
                response.generation_context_chunk_ids,
            )
            failure_details.update(
                {
                    "required_fact_ids": list(case.required_fact_ids),
                    "unavailable_required_fact_ids": list(
                        case.unavailable_required_fact_ids
                    ),
                    "retrieval_coverage_complete": retrieval_coverage_complete,
                    "required_evidence_recall": required_evidence_recall,
                    "required_evidence_precision": required_evidence_precision,
                    "all_required_evidence_coverage": all_required_evidence_coverage,
                    "supporting_context_loss": supporting_context_loss,
                    "runtime": (
                        "CanonicalRagRuntime"
                        if _is_canonical_runtime(self.runtime)
                        else "RagService"
                    ),
                    "route": response.route,
                    "error_class": response.error_class,
                }
            )
            return CaseResult(
                case_id=case.case_id,
                category=case.category,
                question=case.question,
                expected_answer=case.expected_answer,
                expected_document_ids=case.expected_document_ids,
                expected_chunk_ids=case.expected_chunk_ids,
                forbidden_document_ids=case.forbidden_document_ids,
                expected_versions=case.expected_versions,
                required_fact_ids=case.required_fact_ids,
                unavailable_required_fact_ids=case.unavailable_required_fact_ids,
                security_checks=case.security_checks,
                principal=case.principal.model_dump(mode="json"),
                expected_abstain=case.should_abstain,
                retrieved_document_ids=retrieved_documents,
                retrieved_chunk_ids=retrieved_chunks,
                retrieval_trace=tuple(
                    {
                        "chunk_id": item.chunk_id,
                        "document_id": item.document_id,
                        "document_version_id": item.document_version_id,
                        "version": item.version,
                        "rank": item.rank,
                        "score": item.score,
                        "text": item.text,
                    }
                    for item in response.retrieval_results
                ),
                status=response.status,
                answer=response.answer,
                citations=citations,
                rag_run_id=response.run_id,
                hit_at_k=hit_at_k(retrieved, relevant, top_k),
                recall_at_k=recall,
                reciprocal_rank=reciprocal_rank(retrieved, relevant),
                ndcg_at_k=ndcg_at_k(retrieved, relevant, top_k),
                abstention_accuracy=abstention_correct(response.status, case.should_abstain),
                citation_correctness=citation_correctness,
                citation_support=citation_support,
                version_correct=version_correct,
                security_passed=security_passed,
                failure_type=failure_types[0] if failure_types else None,
                failure_types=failure_types,
                failure_details=failure_details,
                retrieval_latency_ms=response.retrieval_latency_ms,
                query_embedding_latency_ms=response.query_embedding_latency_ms,
                embedding_cache_lookup_latency_ms=response.embedding_cache_lookup_latency_ms,
                vector_search_latency_ms=response.vector_search_latency_ms,
                acl_filter_latency_ms=response.acl_filter_latency_ms,
                context_construction_latency_ms=response.context_construction_latency_ms,
                generation_latency_ms=response.generation_latency_ms,
                total_latency_ms=response.total_latency_ms,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                embedding_tokens=response.embedding_tokens,
                query_embedding_cache_hit=response.query_embedding_cache_hit,
                external_embedding_calls=response.external_embedding_calls,
                supporting_chunk_ids=response.supporting_chunk_ids,
                generation_context_chunk_ids=response.generation_context_chunk_ids,
                answerability_result=(
                    response.answerability_result.model_dump(mode="json")
                    if response.answerability_result is not None
                    and hasattr(response.answerability_result, "model_dump")
                    else response.answerability_result
                ),
                answerability_gate_cache_lookup_latency_ms=(
                    response.answerability_gate_cache_lookup_latency_ms
                ),
                answerability_judge_latency_ms=response.answerability_judge_latency_ms,
                context_pruning_latency_ms=response.context_pruning_latency_ms,
                gate_cache_hit=response.gate_cache_hit,
                external_judge_calls=response.external_judge_calls,
                judge_prompt_tokens=response.judge_prompt_tokens,
                judge_completion_tokens=response.judge_completion_tokens,
                local_judge_calls=response.local_judge_calls,
                answerability_operational_error=response.answerability_operational_error,
                retrieval_coverage_complete=retrieval_coverage_complete,
                required_evidence_recall=required_evidence_recall,
                required_evidence_precision=required_evidence_precision,
                all_required_evidence_coverage=all_required_evidence_coverage,
                supporting_context_loss=supporting_context_loss,
            )
        except Exception as exc:
            return CaseResult(
                case_id=case.case_id,
                category=case.category,
                question=case.question,
                expected_answer=case.expected_answer,
                expected_document_ids=case.expected_document_ids,
                expected_chunk_ids=case.expected_chunk_ids,
                forbidden_document_ids=case.forbidden_document_ids,
                expected_versions=case.expected_versions,
                required_fact_ids=case.required_fact_ids,
                unavailable_required_fact_ids=case.unavailable_required_fact_ids,
                security_checks=case.security_checks,
                principal=case.principal.model_dump(mode="json"),
                expected_abstain=case.should_abstain,
                retrieved_document_ids=(),
                retrieved_chunk_ids=(),
                retrieval_trace=(),
                status="error",
                answer=None,
                citations=(),
                rag_run_id=None,
                hit_at_k=0.0,
                recall_at_k=0.0,
                reciprocal_rank=0.0,
                ndcg_at_k=0.0,
                abstention_accuracy=0.0,
                citation_correctness=None,
                citation_support=None,
                version_correct=None,
                security_passed=False
                if case.category in {"access_control", "prompt_injection"}
                else None,
                failure_type=FailureType.UNKNOWN,
                failure_types=(FailureType.UNKNOWN,),
                failure_details={"exception_type": type(exc).__name__},
                retrieval_latency_ms=0.0,
                query_embedding_latency_ms=0.0,
                embedding_cache_lookup_latency_ms=0.0,
                vector_search_latency_ms=0.0,
                acl_filter_latency_ms=0.0,
                context_construction_latency_ms=0.0,
                generation_latency_ms=0.0,
                total_latency_ms=0.0,
                prompt_tokens=None,
                completion_tokens=None,
                embedding_tokens=None,
                query_embedding_cache_hit=False,
                external_embedding_calls=0,
                supporting_chunk_ids=(),
                generation_context_chunk_ids=(),
                answerability_result=None,
                answerability_gate_cache_lookup_latency_ms=0.0,
                answerability_judge_latency_ms=0.0,
                context_pruning_latency_ms=0.0,
                gate_cache_hit=None,
                external_judge_calls=0,
                judge_prompt_tokens=None,
                judge_completion_tokens=None,
                local_judge_calls=0,
                answerability_operational_error=None,
                retrieval_coverage_complete=False,
                required_evidence_recall=None,
                required_evidence_precision=None,
                all_required_evidence_coverage=None,
                supporting_context_loss=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _version_correct(case: EvaluationCase, results: tuple | list) -> float | None:
        if not case.expected_versions:
            return None
        actual = {(item.document_id, item.version) for item in results}
        return float(
            all(
                (document_id, version) in actual
                for document_id, version in case.expected_versions.items()
            )
        )

    @staticmethod
    def _classify_failures(
        case: EvaluationCase,
        recall: float,
        status: str,
        citation_correctness: float | None,
        security_passed: bool | None,
        version_correct: float | None,
        retrieved_document_ids: tuple[str, ...],
        retrieval_results: tuple | list,
        context_document_ids: tuple[str, ...],
        dropped_chunk_ids: tuple[str, ...],
        score_threshold: float | None,
        gate_enabled: bool = False,
        supporting_chunk_ids: tuple[str, ...] = (),
        generation_context_chunk_ids: tuple[str, ...] = (),
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        failures: list[str] = []
        expected = set(case.expected_document_ids)
        retrieved_expected = expected & set(retrieved_document_ids)
        context_expected = expected & set(context_document_ids)
        missing = sorted(expected - set(retrieved_document_ids))
        details: dict[str, Any] = {
            "required_source_count": len(expected),
            "retrieved_required_source_count": len(retrieved_expected),
            "context_required_source_count": len(context_expected),
            "missing_document_ids": missing,
            "context_document_ids": list(context_document_ids),
            "dropped_chunk_ids": list(dropped_chunk_ids),
            "score_threshold": score_threshold,
            "retrieval_scores": [item.score for item in retrieval_results],
            "supporting_chunk_ids": list(supporting_chunk_ids),
            "generation_context_chunk_ids": list(generation_context_chunk_ids),
        }
        if security_passed is False:
            failures.append(
                FailureType.ACL_FAILURE
                if case.category == "access_control"
                else FailureType.CONTEXT_CONSTRUCTION_FAILURE
            )
        if version_correct == 0.0:
            failures.append(FailureType.VERSION_FAILURE)
        if case.should_abstain:
            if status != "abstained":
                failures.append(FailureType.ABSTENTION_FALSE_POSITIVE)
                failures.append(
                    FailureType.EVIDENCE_GATE_FALSE_POSITIVE
                    if gate_enabled
                    else FailureType.THRESHOLD_FAILURE
                )
                details["abstention_failure"] = (
                    "threshold_disabled"
                    if score_threshold is None
                    else "retrieved_non_answer_evidence_above_threshold"
                )
            return tuple(dict.fromkeys(failures)), details
        if recall < 1.0:
            failures.append(FailureType.RETRIEVAL_MISS)
            if case.category == "multi_document":
                if len(retrieved_expected) == 0:
                    details["multi_document_failure"] = "all_required_sources_missing"
                else:
                    details["multi_document_failure"] = "one_or_more_required_sources_missing"
        if expected and retrieved_expected == expected and context_expected != expected:
            failures.append(FailureType.CONTEXT_CONSTRUCTION_FAILURE)
        if status != "answered":
            failures.append(FailureType.ABSTENTION_FALSE_NEGATIVE)
            if gate_enabled:
                failures.append(FailureType.EVIDENCE_GATE_FALSE_NEGATIVE)
            if recall == 1.0 and context_expected == expected:
                failures.append(FailureType.GENERATION_FAILURE)
        if case.category == "multi_document" and gate_enabled:
            required_ids = set(case.expected_document_ids)
            supporting_documents = {
                item.document_id
                for item in retrieval_results
                if item.chunk_id in set(supporting_chunk_ids)
            }
            generation_documents = {
                item.document_id
                for item in retrieval_results
                if item.chunk_id in set(generation_context_chunk_ids)
            }
            if required_ids <= set(retrieved_document_ids) and (
                not required_ids <= supporting_documents
                or not required_ids <= generation_documents
            ):
                failures.append(FailureType.SUPPORTING_CONTEXT_LOSS)
        is_multidoc = case.category == "multi_document" or case.category.startswith(
            "multidoc_"
        )
        if is_multidoc and gate_enabled and expected:
            supporting_documents = {
                item.document_id
                for item in retrieval_results
                if item.chunk_id in set(supporting_chunk_ids)
            }
            retrieval_complete = (
                not case.unavailable_required_fact_ids
                and expected <= set(retrieved_document_ids)
            )
            if retrieval_complete and not expected <= supporting_documents:
                failures.append(FailureType.MULTI_DOCUMENT_SUPPORT_LOSS)
                if status != "answered":
                    failures.append(FailureType.REQUIREMENT_COVERAGE_FALSE_NEGATIVE)
            if case.should_abstain and status == "answered":
                failures.append(FailureType.REQUIREMENT_COVERAGE_FALSE_POSITIVE)
        if citation_correctness == 0.0:
            failures.append(FailureType.CITATION_FAILURE)
        return tuple(dict.fromkeys(failures)), details

    def _report(self, results: tuple[CaseResult, ...]) -> EvaluationReport:
        metrics = self._aggregate(results)
        grouped: dict[str, list[CaseResult]] = defaultdict(list)
        for result in results:
            grouped[result.category].append(result)
        category_metrics = {
            category: self._aggregate(tuple(category_results))
            for category, category_results in sorted(grouped.items())
        }
        return EvaluationReport(
            case_count=len(results),
            metrics=metrics,
            category_metrics=category_metrics,
            cases=results,
            failed_case_count=sum(result.error is not None for result in results),
        )

    @staticmethod
    def _aggregate(
        results: tuple[CaseResult, ...], *, include_primary_view: bool = True
    ) -> dict[str, float | None]:
        metric_names = (
            "hit_at_k",
            "recall_at_k",
            "reciprocal_rank",
            "ndcg_at_k",
            "abstention_accuracy",
        )
        metrics: dict[str, float | None] = {
            name: round(mean(getattr(result, name) for result in results), 6) if results else None
            for name in metric_names
        }
        citation_values = [
            result.citation_correctness
            for result in results
            if result.citation_correctness is not None
        ]
        metrics["citation_correctness"] = (
            round(mean(citation_values), 6) if citation_values else None
        )
        metrics["citation_validity"] = metrics["citation_correctness"]
        support_values = [
            result.citation_support for result in results if result.citation_support is not None
        ]
        metrics["claim_support_rate"] = (
            round(mean(support_values), 6) if support_values else None
        )
        metrics["grounded_answer_rate"] = metrics["claim_support_rate"]
        metrics["unsupported_claim_count"] = sum(value == 0.0 for value in support_values)
        metrics.update(
            abstention_classification(
                [result.expected_abstain for result in results],
                [result.status == "abstained" for result in results],
            )
        )
        metrics.update(
            answerability_classification(
                [not result.expected_abstain for result in results],
                [result.status == "answered" for result in results],
            )
        )
        metrics["gate_false_positive_count"] = metrics["unsupported_answer_count"]
        metrics["gate_false_negative_count"] = metrics["incorrect_abstention_count"]
        metrics["judge_error_count"] = sum(
            result.answerability_operational_error is not None for result in results
        )
        metrics["judge_format_error_count"] = sum(
            result.answerability_operational_error == "JUDGE_FORMAT_ERROR"
            for result in results
        )
        metrics["unauthorized_evidence_selection"] = sum(
            result.answerability_operational_error == "UNAUTHORIZED_SUPPORTING_ID"
            for result in results
        )
        multidoc_answerable = tuple(
            result
            for result in results
            if len(result.expected_document_ids) >= 2 and not result.expected_abstain
        )
        multidoc_negative = tuple(
            result
            for result in results
            if result.expected_abstain
            and (
                result.category.startswith("multidoc_")
                or result.category == "partial_evidence"
            )
        )
        metrics["multi_document_answer_success"] = (
            round(mean(result.status == "answered" for result in multidoc_answerable), 6)
            if multidoc_answerable
            else None
        )
        metrics["multi_document_false_abstention_rate"] = (
            round(mean(result.status != "answered" for result in multidoc_answerable), 6)
            if multidoc_answerable
            else None
        )
        metrics["multi_document_unsupported_answer_rate"] = (
            round(mean(result.status == "answered" for result in multidoc_negative), 6)
            if multidoc_negative
            else None
        )
        metrics["exact_identifier_answer_success"] = EvaluationRunner._category_answer_success(
            results, "exact_identifier"
        )
        acl_values = [
            result.security_passed
            for result in results
            if "acl" in result.security_checks and result.security_passed is not None
        ]
        injection_values = [
            result.security_passed
            for result in results
            if "prompt_injection" in result.security_checks
            and result.security_passed is not None
        ]
        metrics["acl_safety"] = (
            round(mean(acl_values), 6)
            if acl_values
            else EvaluationRunner._category_security(results, "access_control")
        )
        metrics["prompt_injection_boundary"] = (
            round(mean(injection_values), 6)
            if injection_values
            else EvaluationRunner._category_security(results, "prompt_injection")
        )
        security = [
            result.security_passed for result in results if result.security_passed is not None
        ]
        metrics["security_success_rate"] = round(mean(security), 6) if security else None
        versions = [
            result.version_correct for result in results if result.version_correct is not None
        ]
        metrics["version_accuracy"] = round(mean(versions), 6) if versions else None
        retrieved_counts = [len(result.retrieved_chunk_ids) for result in results]
        metrics["chunks_passing_threshold"] = sum(retrieved_counts)
        metrics["mean_retrieved_chunks"] = (
            round(mean(retrieved_counts), 6) if retrieved_counts else None
        )
        for name in (
            "required_evidence_recall",
            "required_evidence_precision",
            "all_required_evidence_coverage",
            "supporting_context_loss",
        ):
            values = [
                getattr(result, name)
                for result in results
                if getattr(result, name) is not None
            ]
            metric_name = (
                "all_required_evidence_coverage_rate"
                if name == "all_required_evidence_coverage"
                else "supporting_context_loss_rate"
                if name == "supporting_context_loss"
                else name
            )
            metrics[metric_name] = round(mean(values), 6) if values else None
        metrics["retrieval_coverage_complete_rate"] = (
            round(mean(result.retrieval_coverage_complete for result in results), 6)
            if results
            else None
        )
        if include_primary_view:
            complete = tuple(result for result in results if result.retrieval_coverage_complete)
            primary = EvaluationRunner._aggregate(complete, include_primary_view=False)
            for name, value in primary.items():
                metrics[f"retrieval_complete_{name}"] = value
            metrics["retrieval_complete_case_count"] = len(complete)
        return metrics

    @staticmethod
    def _category_answer_success(
        results: tuple[CaseResult, ...], category: str
    ) -> float | None:
        selected = [result for result in results if result.category == category]
        return (
            round(mean(result.status == "answered" for result in selected), 6)
            if selected
            else None
        )

    @staticmethod
    def _category_security(results: tuple[CaseResult, ...], category: str) -> float | None:
        selected = [
            result.security_passed
            for result in results
            if result.category == category and result.security_passed is not None
        ]
        return round(mean(selected), 6) if selected else None
