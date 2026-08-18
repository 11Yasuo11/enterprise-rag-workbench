import time
from dataclasses import dataclass

from sqlalchemy.orm import Session

from rag_workbench.answerability.base import (
    AnswerabilityGate,
    AnswerabilityGateError,
    AnswerabilityReason,
    AnswerabilityResult,
    GateEvidence,
    GateTiming,
)
from rag_workbench.answerability.validation import validate_gate_result_with_error
from rag_workbench.db.models import RagRun, RetrievalResultRecord
from rag_workbench.generation.citations import Citation, build_citations
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.generation.prompts import (
    GROUNDING_PROMPT_VERSION,
    PROMPT_VERSION,
    build_grounded_prompt,
)
from rag_workbench.providers.llm.base import GenerationContext, GenerationRequest, LLMProvider
from rag_workbench.providers.llm.extractive import EXTRACTIVE_REVISION, EXTRACTIVE_V1_1_MODEL
from rag_workbench.retrieval.filters import RetrievalFilters
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal


@dataclass(frozen=True)
class RagResponse:
    run_id: str
    status: str
    answer: str | None
    citations: tuple[Citation, ...]
    retrieval_results: tuple[RetrievalResult, ...]
    final_context: str | None = None
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
    supporting_chunk_ids: tuple[str, ...] = ()
    generation_context_chunk_ids: tuple[str, ...] = ()
    answerability_result: AnswerabilityResult | None = None
    answerability_gate_cache_lookup_latency_ms: float = 0.0
    answerability_judge_latency_ms: float = 0.0
    context_pruning_latency_ms: float = 0.0
    gate_cache_hit: bool | None = None
    external_judge_calls: int = 0
    judge_prompt_tokens: int | None = None
    judge_completion_tokens: int | None = None
    local_judge_calls: int = 0
    answerability_operational_error: str | None = None
    generation_operational_error: str | None = None
    judge_logical_request_id: str | None = None
    judge_attempt_count: int = 0
    judge_retry_count: int = 0
    judge_physical_attempts: int = 0
    extractive_path: str | None = None


class RagService:
    def __init__(
        self,
        session: Session,
        retriever: Retriever,
        context_builder: ContextBuilder,
        llm_provider: LLMProvider,
        answerability_gate: AnswerabilityGate | None = None,
        supporting_context_only: bool = False,
    ) -> None:
        self.session = session
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm_provider = llm_provider
        self.answerability_gate = answerability_gate
        self.supporting_context_only = supporting_context_only

    def query(
        self,
        question: str,
        principal: Principal,
        top_k: int = 5,
        score_threshold: float | None = 0.2,
        filters: RetrievalFilters | None = None,
        include_debug: bool = False,
    ) -> RagResponse:
        started = time.perf_counter()
        embedding_tokens_before = self.retriever.embedding_provider.usage.input_tokens
        retrieval_started = time.perf_counter()
        results = self.retriever.retrieve(
            question,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
            principal=principal,
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
        gate_result: AnswerabilityResult | None = None
        gate_timing = GateTiming()
        operational_error: str | None = None
        generation_results = results
        pruning_started = time.perf_counter()
        if results and self.answerability_gate is not None:
            evidence = tuple(
                GateEvidence(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    document_version_id=item.document_version_id,
                    version=item.version,
                    text=item.text,
                    index_identity=getattr(self.retriever, "index_identity", None),
                )
                for item in results
            )
            try:
                proposed = self.answerability_gate.evaluate(question, evidence)
                validated = validate_gate_result_with_error(
                    proposed, evidence, session=self.session, principal=principal
                )
                gate_result = validated.result
                operational_error = (
                    validated.operational_error.value if validated.operational_error else None
                )
            except AnswerabilityGateError as exc:
                gate_result = AnswerabilityResult.fail_closed(AnswerabilityReason.UNKNOWN)
                operational_error = exc.code.value
            except Exception:
                gate_result = AnswerabilityResult.fail_closed(AnswerabilityReason.UNKNOWN)
                operational_error = "JUDGE_REQUEST_ERROR"
            gate_timing = self.answerability_gate.last_timing
            if not gate_result.answerable:
                generation_results = []
            elif self.supporting_context_only:
                allowed = set(gate_result.supporting_chunk_ids)
                generation_results = [item for item in results if item.chunk_id in allowed]
        elif self.answerability_gate is not None:
            gate_result = AnswerabilityResult.fail_closed(AnswerabilityReason.MISSING_REQUIRED_FACT)
        context_pruning_latency_ms = (time.perf_counter() - pruning_started) * 1000
        context_started = time.perf_counter()
        context = self.context_builder.build(generation_results)
        context_construction_latency_ms = (time.perf_counter() - context_started) * 1000
        retrieval_timing = self.retriever.last_timing
        answer: str | None = None
        citations: list[Citation] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        status = "abstained"
        generation_latency_ms = 0.0
        generation_operational_error: str | None = None
        extractive_path: str | None = None
        if context.items:
            prompt = build_grounded_prompt(
                question, context, strict_evidence_only=self.answerability_gate is not None
            )
            generation_started = time.perf_counter()
            generated = self.llm_provider.generate(
                GenerationRequest(
                    question=question,
                    prompt=prompt,
                    contexts=tuple(
                        GenerationContext(
                            chunk_id=item.result.chunk_id,
                            citation_label=item.citation_label,
                            text=item.result.text,
                        )
                        for item in context.items
                    ),
                )
            )
            generation_latency_ms = (time.perf_counter() - generation_started) * 1000
            citations = build_citations(context, generated.used_chunk_ids)
            extractive_path = (generated.metadata or {}).get("extractive_path")
            if generated.answer and citations:
                answer = generated.answer
                status = "answered"
            elif gate_result is not None and gate_result.answerable:
                generation_operational_error = "GENERATION_FAILURE"
            prompt_tokens = generated.prompt_tokens
            completion_tokens = generated.completion_tokens
        elif gate_result is not None and gate_result.answerable and not operational_error:
            generation_operational_error = "GENERATION_FAILURE"
        elapsed_ms = (time.perf_counter() - started) * 1000
        embedding_tokens = (
            self.retriever.embedding_provider.usage.input_tokens - embedding_tokens_before
        )
        run = RagRun(
            tenant_id=principal.tenant_id,
            principal_id=principal.principal_id,
            query=question,
            status=status,
            retrieval_config={
                "top_k": top_k,
                "score_threshold": score_threshold,
                "filters": {
                    "document_ids": list(filters.document_ids) if filters else [],
                    "source_types": list(filters.source_types) if filters else [],
                },
                "prompt_version": (
                    GROUNDING_PROMPT_VERSION if self.answerability_gate else PROMPT_VERSION
                ),
                "generator": {
                    "identity": getattr(self.llm_provider, "model_name", None),
                    "parent": (
                        EXTRACTIVE_REVISION["parent"]
                        if getattr(self.llm_provider, "model_name", None)
                        == EXTRACTIVE_V1_1_MODEL
                        else None
                    ),
                    "reason": EXTRACTIVE_REVISION["reason"]
                    if getattr(self.llm_provider, "model_name", None) == EXTRACTIVE_V1_1_MODEL
                    else None,
                    "semantic_policy_changed": False,
                    "operational_error": generation_operational_error,
                    "extractive_path": extractive_path,
                },
                "answerability_gate": (
                    {
                        "provider": self.answerability_gate.provider_name,
                        "model": self.answerability_gate.model_name,
                        "gate_version": self.answerability_gate.gate_version,
                        "prompt_version": self.answerability_gate.prompt_version,
                        "supporting_context_only": self.supporting_context_only,
                    }
                    if self.answerability_gate
                    else None
                ),
            },
            context_references=[
                {"chunk_id": item.result.chunk_id, "citation_label": item.citation_label}
                for item in context.items
            ],
            retrieved_chunk_ids=[item.chunk_id for item in results],
            supporting_chunk_ids=(
                list(gate_result.supporting_chunk_ids) if gate_result is not None else []
            ),
            generation_context_chunk_ids=[item.result.chunk_id for item in context.items],
            answerability_result=(
                gate_result.model_dump(mode="json") if gate_result is not None else None
            ),
            answerability_operational_error=operational_error,
            provider=self.llm_provider.provider_name,
            model=self.llm_provider.model_name,
            answer=answer,
            citations=[citation.to_dict() for citation in citations],
            latency_ms=elapsed_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            query_embedding_latency_ms=retrieval_timing.query_embedding_latency_ms,
            embedding_cache_lookup_latency_ms=(
                retrieval_timing.embedding_cache_lookup_latency_ms
            ),
            vector_search_latency_ms=retrieval_timing.vector_search_latency_ms,
            acl_filter_latency_ms=retrieval_timing.acl_filter_latency_ms,
            context_construction_latency_ms=context_construction_latency_ms,
            answerability_gate_cache_lookup_latency_ms=gate_timing.cache_lookup_latency_ms,
            answerability_judge_latency_ms=gate_timing.judge_latency_ms,
            context_pruning_latency_ms=context_pruning_latency_ms,
            generation_latency_ms=generation_latency_ms,
            query_embedding_cache_hit=retrieval_timing.query_embedding_cache_hit,
            gate_cache_hit=gate_timing.cache_hit if self.answerability_gate else None,
            external_judge_calls=gate_timing.external_calls,
            local_judge_calls=gate_timing.local_calls,
            judge_prompt_tokens=gate_timing.prompt_tokens,
            judge_completion_tokens=gate_timing.completion_tokens,
            embedding_tokens=embedding_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self.session.add(run)
        self.session.flush()
        included = {item.result.chunk_id for item in context.items}
        for result in results:
            self.session.add(
                RetrievalResultRecord(
                    run_id=run.id,
                    chunk_id=result.chunk_id,
                    document_fk=self._document_fk(result.chunk_id),
                    rank=result.rank,
                    score=result.score,
                    included_in_context=result.chunk_id in included,
                )
            )
        self.session.commit()
        return RagResponse(
            run_id=run.id,
            status=status,
            answer=answer,
            citations=tuple(citations),
            retrieval_results=tuple(results),
            final_context=context.text if include_debug else None,
            retrieval_latency_ms=retrieval_latency_ms,
            query_embedding_latency_ms=retrieval_timing.query_embedding_latency_ms,
            embedding_cache_lookup_latency_ms=(
                retrieval_timing.embedding_cache_lookup_latency_ms
            ),
            vector_search_latency_ms=retrieval_timing.vector_search_latency_ms,
            acl_filter_latency_ms=retrieval_timing.acl_filter_latency_ms,
            context_construction_latency_ms=context_construction_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            embedding_tokens=embedding_tokens,
            query_embedding_cache_hit=retrieval_timing.query_embedding_cache_hit,
            external_embedding_calls=retrieval_timing.external_embedding_calls,
            supporting_chunk_ids=(
                gate_result.supporting_chunk_ids if gate_result is not None else ()
            ),
            generation_context_chunk_ids=tuple(item.result.chunk_id for item in context.items),
            answerability_result=gate_result,
            answerability_gate_cache_lookup_latency_ms=gate_timing.cache_lookup_latency_ms,
            answerability_judge_latency_ms=gate_timing.judge_latency_ms,
            context_pruning_latency_ms=context_pruning_latency_ms,
            gate_cache_hit=gate_timing.cache_hit if self.answerability_gate else None,
            external_judge_calls=gate_timing.external_calls,
            local_judge_calls=gate_timing.local_calls,
            answerability_operational_error=operational_error,
            generation_operational_error=generation_operational_error,
            judge_prompt_tokens=gate_timing.prompt_tokens,
            judge_completion_tokens=gate_timing.completion_tokens,
            judge_logical_request_id=gate_timing.logical_request_id,
            judge_attempt_count=gate_timing.attempt_count,
            judge_retry_count=gate_timing.retry_count,
            judge_physical_attempts=gate_timing.external_calls,
            extractive_path=extractive_path,
        )

    def _document_fk(self, chunk_id: str) -> str:
        from rag_workbench.db.models import Chunk

        chunk = self.session.get(Chunk, chunk_id)
        if chunk is None:
            raise RuntimeError(f"Retrieved chunk disappeared before run logging: {chunk_id}")
        return chunk.document_fk
