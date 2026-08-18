import json
from pathlib import Path

import httpx
import pytest

from rag_workbench.answerability.base import (
    AnswerabilityGateError,
    AnswerabilityReason,
    AnswerabilityResult,
    GateEvidence,
    GateTiming,
)
from rag_workbench.answerability.cache import (
    CachedAnswerabilityGate,
    judge_logical_request_identity,
)
from rag_workbench.answerability.openai_compatible import (
    OpenAICompatibleAnswerabilityGate,
    hosted_judge_chat_payload,
)
from rag_workbench.answerability.transport import (
    DEFAULT_TRANSPORT_RETRY_POLICY,
    ProviderFailureClass,
    no_sleep,
)
from rag_workbench.db.models import AnswerabilityGateCacheRecord, ResearchArchitectureRecord
from rag_workbench.evaluation.failures import FailureType
from rag_workbench.experiments.v2_document_diversity import PHASE1_BENCHMARK_HEADING
from rag_workbench.experiments.v2_quality_recovery import (
    V1_BENCHMARK_HEADING,
    V2_BENCHMARK_HEADING,
    V2_RESEARCH_ARCHITECTURE_ID,
)
from rag_workbench.experiments.v2_reliability import (
    OPERATIONAL_TAXONOMY,
    PHASE4_BENCHMARK_HEADING,
    classify_generator_root_cause,
    classify_historical_provider_failure,
    fixture_case,
    replay_historical_generator,
)
from rag_workbench.experiments.v2_soft_document_cap import PHASE2_BENCHMARK_HEADING
from rag_workbench.experiments.v2_sufficiency_fn import PHASE3_BENCHMARK_HEADING
from rag_workbench.generation.citations import build_citations
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.generation.generator import RagService
from rag_workbench.providers.llm.base import GenerationContext, GenerationRequest, GenerationResult
from rag_workbench.providers.llm.extractive import (
    EXTRACTIVE_REVISION,
    EXTRACTIVE_V1_1_MODEL,
    EXTRACTIVE_V1_MODEL,
    ExtractiveGenerationProvider,
    generate_extractive_v1_1,
)
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal


def evidence() -> tuple[GateEvidence, ...]:
    return (
        GateEvidence(
            "chunk-1", "policy", "version-1", "2026", "Managers review schedules every month."
        ),
        GateEvidence(
            "chunk-2", "ops", "version-2", "1.0", "The east drill runs on the first Wednesday."
        ),
        GateEvidence("chunk-3", "atlas", "version-3", "1.0", "Project Atlas uses API version v3."),
    )


def sufficient_json(ids: list[str] | None = None) -> str:
    return json.dumps(
        {
            "answerable": True,
            "supporting_chunk_ids": ids or ["chunk-1"],
            "confidence": 0.9,
            "reason_code": "SUFFICIENT_EVIDENCE",
        }
    )


def insufficient_json() -> str:
    return json.dumps(
        {
            "answerable": False,
            "supporting_chunk_ids": [],
            "confidence": 0.2,
            "reason_code": "MISSING_REQUIRED_FACT",
        }
    )


class StatusResponse:
    def __init__(
        self, status: int, payload: dict | None = None, headers: dict | None = None
    ) -> None:
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            response = httpx.Response(self.status_code, request=request, headers=self.headers)
            raise httpx.HTTPStatusError("provider error", request=request, response=response)

    def json(self) -> dict:
        return self._payload


def success_response(content: str) -> StatusResponse:
    return StatusResponse(
        200,
        {
            "model": "gpt-5.6-sol",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 18},
        },
    )


class SequencePoster:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.payloads: list[dict] = []

    def __call__(self, url, *args, **kwargs):
        self.calls += 1
        self.payloads.append(kwargs["json"])
        assert "Authorization" in kwargs["headers"]
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def hosted_gate() -> OpenAICompatibleAnswerabilityGate:
    return OpenAICompatibleAnswerabilityGate(
        api_key="not-a-real-key",
        model="gpt-5.6-sol",
        base_url="https://api.openai.com/v1",
        sleeper=no_sleep,
    )


def retrieval(chunk_id: str, document_id: str, text: str, rank: int = 1, version: str = "2026"):
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        document_version_id=f"{document_id}-{version}",
        text=text,
        rank=rank,
        score=0.9,
        source=f"{document_id}.md",
        source_type="markdown",
        title=document_id,
        version=version,
        section="Body",
    )


class FixedGate:
    provider_name = "openai"
    model_name = "gpt-5.6-sol"
    gate_version = "1"
    prompt_version = "evidence-sufficiency-v1"

    def __init__(self, result: AnswerabilityResult) -> None:
        self.result = result
        self.calls = 0
        self.last_timing = GateTiming(external_calls=1, attempt_count=1)

    def evaluate(self, question, chunks):
        self.calls += 1
        return self.result


class EmptyGenerator:
    provider_name = "local-extractive"
    model_name = EXTRACTIVE_V1_1_MODEL

    def generate(self, request):
        return GenerationResult(answer="", used_chunk_ids=())


def test_historical_generator_failure_is_extractive_span_match() -> None:
    case = fixture_case("fv1_ver_03")
    assert case["judge_decision"]["answerable"] is True
    assert case["supporting_chunk_ids"] == ["49679d97-020f-4418-9430-4c296bc824b1"]
    assert case["actual_generator_behavior"]["status"] == "abstained"
    assert case["actual_generator_behavior"]["answer"] is None
    assert classify_generator_root_cause(case) == "EXTRACTIVE_SPAN_MATCH_FAILURE"
    before = replay_historical_generator(case, revision=EXTRACTIVE_V1_MODEL)
    after = replay_historical_generator(case, revision=EXTRACTIVE_V1_1_MODEL)
    assert before["status"] == "abstained"
    assert before["answer"] is None
    assert after["status"] == "answered"
    assert "every month" in after["answer"].casefold()
    assert after["citation_validity"] == 1.0
    assert after["used_chunk_ids"] == case["supporting_chunk_ids"]


def test_historical_provider_failure_is_timeout() -> None:
    case = fixture_case("fv1_dup_05")
    assert case["answerability_operational_error"] == "JUDGE_REQUEST_ERROR"
    assert case["judge_decision"]["answerable"] is False
    assert classify_historical_provider_failure(case) == ProviderFailureClass.TIMEOUT.value


def test_extractive_v1_1_is_reliability_revision_only() -> None:
    assert EXTRACTIVE_REVISION["parent"] == EXTRACTIVE_V1_MODEL
    assert EXTRACTIVE_REVISION["id"] == EXTRACTIVE_V1_1_MODEL
    assert EXTRACTIVE_REVISION["semantic_policy_changed"] is False
    assert ExtractiveGenerationProvider().model_name == EXTRACTIVE_V1_1_MODEL


def test_one_two_three_chunk_and_ordering_generation() -> None:
    one = (GenerationContext("c1", "C1", "Managers review remote-work schedules every month."),)
    two = (
        *one,
        GenerationContext("c2", "C2", "Expense reports are due within ten business days."),
    )
    three = (*two, GenerationContext("c3", "C3", "Project Atlas uses API version v3."))
    question = "Yard docket asks unrelated calendar cadence without overlapping terms."
    for contexts in (one, two, three):
        result = generate_extractive_v1_1(GenerationRequest(question, "prompt", contexts))
        assert result.answer
        assert result.used_chunk_ids == tuple(item.chunk_id for item in contexts)
    ordered = generate_extractive_v1_1(GenerationRequest(question, "prompt", three))
    assert ordered.used_chunk_ids == ("c1", "c2", "c3")
    first = ordered.answer.index("[C1]")
    second = ordered.answer.index("[C2]")
    third = ordered.answer.index("[C3]")
    assert first < second < third


def test_same_and_different_document_and_near_duplicate_support() -> None:
    question = "Yard docket without lexical overlap."
    same = generate_extractive_v1_1(
        GenerationRequest(
            question,
            "p",
            (
                GenerationContext("a1", "C1", "Employees may work remotely two days per week."),
                GenerationContext("a2", "C2", "Managers review remote-work schedules every month."),
            ),
        )
    )
    different = generate_extractive_v1_1(
        GenerationRequest(
            question,
            "p",
            (
                GenerationContext(
                    "east",
                    "C1",
                    "The east recovery drill runs on the first Wednesday.",
                ),
                GenerationContext(
                    "policy",
                    "C2",
                    "Managers review remote-work schedules every month.",
                ),
            ),
        )
    )
    assert same.used_chunk_ids == ("a1", "a2")
    assert different.used_chunk_ids == ("east", "policy")
    bundle = ContextBuilder(1200).build(
        [
            retrieval(
                "dup-1",
                "recovery-runbook-east",
                "The east recovery drill runs on the first Wednesday.",
            ),
            retrieval(
                "dup-2",
                "recovery-runbook-west",
                "The east recovery drill runs on the first Wednesday.",
                2,
            ),
        ]
    )
    assert len(bundle.items) == 1
    citations = build_citations(bundle, ("dup-1",))
    assert citations[0].chunk_id == "dup-1"


def test_exact_id_and_version_region_extractive_support() -> None:
    exact = generate_extractive_v1_1(
        GenerationRequest(
            "What is identifier CS-1842?",
            "p",
            (GenerationContext("id-1", "C1", "The production identifier is CS-1842."),),
        )
    )
    versioned = generate_extractive_v1_1(
        GenerationRequest(
            "Yard docket asks 2026 manager cadence without overlapping nouns.",
            "p",
            (
                GenerationContext(
                    "ver-1",
                    "C1",
                    "Managers review remote-work schedules every month.",
                ),
            ),
        )
    )
    assert "CS-1842" in exact.answer
    assert versioned.used_chunk_ids == ("ver-1",)


def test_empty_support_is_normal_abstention_not_generation_failure(monkeypatch) -> None:
    service = _service(
        monkeypatch,
        FixedGate(
            AnswerabilityResult(
                answerable=False,
                supporting_chunk_ids=(),
                reason_code=AnswerabilityReason.MISSING_REQUIRED_FACT,
            )
        ),
        ExtractiveGenerationProvider(),
        [retrieval("chunk-1", "policy", "Managers review remote-work schedules every month.")],
    )
    response = service.query("Unknown future fact?", Principal("p", "acmeai"))
    assert response.status == "abstained"
    assert response.generation_operational_error is None
    assert response.generation_context_chunk_ids == ()


def test_valid_support_generates_and_empty_generator_is_typed_failure(monkeypatch) -> None:
    selected = retrieval("chunk-1", "policy", "Managers review remote-work schedules every month.")
    service = _service(
        monkeypatch,
        FixedGate(
            AnswerabilityResult(
                answerable=True,
                supporting_chunk_ids=("chunk-1",),
                reason_code=AnswerabilityReason.SUFFICIENT_EVIDENCE,
            )
        ),
        ExtractiveGenerationProvider(),
        [selected, retrieval("noise", "other", "The office is closed Sunday.", 2)],
    )
    response = service.query(
        "Yard docket asks manager cadence without overlapping nouns.",
        Principal("p", "acmeai"),
    )
    assert response.status == "answered"
    assert response.generation_operational_error is None
    assert response.citations[0].chunk_id == "chunk-1"
    failed = _service(
        monkeypatch,
        FixedGate(
            AnswerabilityResult(
                answerable=True,
                supporting_chunk_ids=("chunk-1",),
                reason_code=AnswerabilityReason.SUFFICIENT_EVIDENCE,
            )
        ),
        EmptyGenerator(),
        [selected],
    ).query("Yard docket asks manager cadence.", Principal("p", "acmeai"))
    assert failed.status == "abstained"
    assert failed.generation_operational_error == "GENERATION_FAILURE"


def test_citation_validity_only_validated_support(monkeypatch) -> None:
    selected = retrieval("chunk-1", "policy", "Managers review remote-work schedules every month.")
    response = _service(
        monkeypatch,
        FixedGate(
            AnswerabilityResult(
                answerable=True,
                supporting_chunk_ids=("chunk-1",),
                reason_code=AnswerabilityReason.SUFFICIENT_EVIDENCE,
            )
        ),
        ExtractiveGenerationProvider(),
        [selected, retrieval("unauthorized", "secret", "Do not cite this.", 2)],
    ).query("Yard docket asks cadence.", Principal("p", "acmeai"))
    cited = {item.chunk_id for item in response.citations}
    assert cited == {"chunk-1"}


def _service(monkeypatch, gate, generator, results):
    class Retriever:
        embedding_provider = type("E", (), {"usage": type("U", (), {"input_tokens": 0})()})()
        last_timing = type(
            "T",
            (),
            {
                "query_embedding_latency_ms": 0.0,
                "embedding_cache_lookup_latency_ms": 0.0,
                "vector_search_latency_ms": 0.0,
                "acl_filter_latency_ms": 0.0,
                "query_embedding_cache_hit": True,
                "external_embedding_calls": 0,
            },
        )()

        def retrieve(self, *args, **kwargs):
            return list(results)

    class Session:
        def add(self, value):
            if value.__class__.__name__ == "RagRun":
                value.id = "run-1"

        def flush(self):
            return None

        def commit(self):
            return None

    monkeypatch.setattr(RagService, "_document_fk", lambda self, chunk_id: "doc-fk")
    monkeypatch.setattr(
        "rag_workbench.generation.generator.validate_gate_result_with_error",
        lambda result, chunks, **kwargs: type(
            "V", (), {"result": result, "operational_error": None}
        )(),
    )
    return RagService(Session(), Retriever(), ContextBuilder(), generator, gate, True)


def test_timeout_then_success_is_one_logical_decision(monkeypatch) -> None:
    poster = SequencePoster(
        [httpx.TimeoutException("timed out"), success_response(sufficient_json())]
    )
    monkeypatch.setattr(httpx, "post", poster)
    gate = hosted_gate()
    result = gate.evaluate("How much leave?", evidence())
    assert result.answerable is True
    assert poster.calls == 2
    assert poster.payloads[0] == poster.payloads[1]
    assert gate.last_timing.attempt_count == 2
    assert gate.last_timing.retry_count == 1
    assert gate.last_timing.external_calls == 2
    assert [item["attempt_number"] for item in gate.last_timing.attempts] == [1, 2]
    assert gate.last_timing.attempts[0]["logical_request_id"] == gate.last_timing.attempts[1][
        "logical_request_id"
    ]
    assert {item["request_cache_identity"] for item in gate.last_timing.attempts} == {
        gate.last_timing.logical_request_id
    }
    assert gate.last_timing.final_outcome == "success"


@pytest.mark.parametrize(
    "first",
    [StatusResponse(429, headers={"Retry-After": "0"}), StatusResponse(503)],
)
def test_retryable_http_then_success(monkeypatch, first) -> None:
    poster = SequencePoster([first, success_response(sufficient_json())])
    monkeypatch.setattr(httpx, "post", poster)
    gate = hosted_gate()
    assert gate.evaluate("How much leave?", evidence()).answerable
    assert poster.calls == 2
    assert poster.payloads[0] == poster.payloads[1]


def test_two_timeouts_fail_closed(monkeypatch) -> None:
    poster = SequencePoster([httpx.TimeoutException("t1"), httpx.TimeoutException("t2")])
    monkeypatch.setattr(httpx, "post", poster)
    gate = hosted_gate()
    with pytest.raises(AnswerabilityGateError) as error:
        gate.evaluate("How much leave?", evidence())
    assert error.value.code == "JUDGE_REQUEST_ERROR"
    assert error.value.provider_failure_class == ProviderFailureClass.TIMEOUT.value
    assert poster.calls == 2
    assert gate.last_timing.final_outcome == "JUDGE_REQUEST_ERROR"


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_is_not_retried(monkeypatch, status) -> None:
    poster = SequencePoster([StatusResponse(status), success_response(sufficient_json())])
    monkeypatch.setattr(httpx, "post", poster)
    gate = hosted_gate()
    with pytest.raises(AnswerabilityGateError) as error:
        gate.evaluate("How much leave?", evidence())
    assert error.value.provider_failure_class == ProviderFailureClass.AUTHENTICATION_ERROR.value
    assert poster.calls == 1


def test_schema_valid_answer_and_abstention_are_not_retried(monkeypatch) -> None:
    for content in (sufficient_json(), insufficient_json()):
        poster = SequencePoster([success_response(content), success_response(sufficient_json())])
        monkeypatch.setattr(httpx, "post", poster)
        gate = hosted_gate()
        result = gate.evaluate("How much leave?", evidence())
        assert poster.calls == 1
        assert result.answerable is (content == sufficient_json())


def test_schema_valid_false_negative_is_not_retried(monkeypatch) -> None:
    poster = SequencePoster(
        [success_response(insufficient_json()), success_response(sufficient_json())]
    )
    monkeypatch.setattr(httpx, "post", poster)
    gate = hosted_gate()
    result = gate.evaluate("Evidence is actually sufficient.", evidence())
    assert result.answerable is False
    assert poster.calls == 1


def test_logical_request_identity_includes_quality_payload(monkeypatch) -> None:
    chunks = evidence()
    key_a, payload_a = judge_logical_request_identity(
        "q",
        chunks,
        provider="openai",
        model="gpt-5.6-sol",
        gate_version="1",
        prompt_version="evidence-sufficiency-v1",
    )
    key_b, _payload_b = judge_logical_request_identity(
        "q",
        chunks,
        provider="openai",
        model="gpt-5.6-sol",
        gate_version="1",
        prompt_version="evidence-sufficiency-v1",
    )
    assert key_a == key_b
    assert payload_a["ordered_top5"][0]["chunk_id"] == "chunk-1"
    poster = SequencePoster([httpx.TimeoutException("t"), success_response(sufficient_json())])
    monkeypatch.setattr(httpx, "post", poster)
    hosted_gate().evaluate("q", chunks)
    expected = hosted_judge_chat_payload(
        model="gpt-5.6-sol", question="q", chunks=chunks, provider="openai"
    )
    assert poster.payloads[0] == expected
    assert poster.payloads[1] == expected


def test_cache_does_not_duplicate_logical_rows(db_session, monkeypatch) -> None:
    from sqlalchemy import func, select

    poster = SequencePoster([httpx.TimeoutException("t"), success_response(sufficient_json())])
    monkeypatch.setattr(httpx, "post", poster)
    gate = CachedAnswerabilityGate(db_session, hosted_gate())
    first = gate.evaluate("How much leave?", evidence())
    second = gate.evaluate("How much leave?", evidence())
    assert first == second
    count = db_session.scalar(select(func.count()).select_from(AnswerabilityGateCacheRecord))
    assert count == 1
    assert poster.calls == 2
    assert gate.last_timing.cache_hit is True
    assert gate.delegate.last_timing.external_calls == 2
    assert gate.delegate.last_timing.attempt_count == 2


def test_retry_policy_constants() -> None:
    assert DEFAULT_TRANSPORT_RETRY_POLICY.max_total_attempts == 2
    assert ProviderFailureClass.TIMEOUT in DEFAULT_TRANSPORT_RETRY_POLICY.retryable
    assert ProviderFailureClass.AUTHENTICATION_ERROR not in DEFAULT_TRANSPORT_RETRY_POLICY.retryable
    assert set(OPERATIONAL_TAXONOMY) == {
        FailureType.EVIDENCE_GATE_FALSE_NEGATIVE.value,
        FailureType.EVIDENCE_GATE_FALSE_POSITIVE.value,
        FailureType.GENERATION_FAILURE.value,
        FailureType.JUDGE_REQUEST_ERROR.value,
        FailureType.INVALID_SUPPORTING_ID.value,
        FailureType.CITATION_FAILURE.value,
        FailureType.SECURITY_FAILURE.value,
    }


def test_v1_and_selected_v2_identities_unchanged() -> None:
    from rag_workbench.experiments.v2_quality_recovery import verify_v1_file_identities
    from rag_workbench.experiments.v2_sufficiency_fn import CONTROL_JUDGE, CONTROL_MODE

    files = verify_v1_file_identities()
    assert files["generator_identity"] == EXTRACTIVE_V1_MODEL
    assert CONTROL_MODE == "POINTWISE_CROSS_ENCODER_TOP5"
    assert CONTROL_JUDGE == "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1"
    text = Path("BENCHMARK.md").read_text()
    assert text.count(V1_BENCHMARK_HEADING) == 1
    assert text.count(V2_BENCHMARK_HEADING) == 1
    assert text.count(PHASE1_BENCHMARK_HEADING) == 1
    assert text.count(PHASE2_BENCHMARK_HEADING) == 1
    assert text.count(PHASE3_BENCHMARK_HEADING) == 1
    assert text.count(PHASE4_BENCHMARK_HEADING) == 1


def test_phase4_initialize_preserves_frozen_selection(db_session, monkeypatch) -> None:
    from rag_workbench.experiments.hybrid_reranker_replication import RELEASE_ARCHITECTURE_ID
    from rag_workbench.experiments.v2_reliability import V2ReliabilityHardening

    db_session.add(
        ResearchArchitectureRecord(
            architecture_id=V2_RESEARCH_ARCHITECTURE_ID,
            parent_architecture_id=RELEASE_ARCHITECTURE_ID,
            selected_retriever="HYBRID_CROSS_ENCODER_RERANK",
            research_status="ACTIVE",
            production_status=False,
            control_configuration={},
            control_equivalence_hash="abc",
            diagnosis_dataset_id="x",
            diagnosis_dataset_hash="y",
            security_guardrails={},
            v1_preservation={},
            failure_census={},
            ranking_diagnostic={},
            judge_false_negative_diagnostic={},
            operational_diagnostic={},
            primary_bottleneck="CROSS_ENCODER_SAME_DOCUMENT_TOP5_CROWDING",
            recommended_ranking_intervention="none",
            selected_v2_ranking="POINTWISE_CROSS_ENCODER_TOP5",
            ranking_research_status="FROZEN_FOR_CURRENT_V2_CYCLE",
            selected_v2_judge="GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1",
            judge_research_status="FROZEN_FOR_CURRENT_V2_CYCLE",
        )
    )
    db_session.flush()
    monkeypatch.setattr(
        "rag_workbench.experiments.v2_reliability.verify_v1_file_identities", lambda: {"ok": True}
    )
    monkeypatch.setattr(
        "rag_workbench.experiments.v2_reliability.verify_persisted_v1", lambda session: {"ok": True}
    )
    status = V2ReliabilityHardening(db_session).execute()
    assert status["completed"] is True
    assert status["selected_v2_ranking"] == "POINTWISE_CROSS_ENCODER_TOP5"
    assert status["selected_v2_judge"] == "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1"
    assert status["generator_root_cause"] == "EXTRACTIVE_SPAN_MATCH_FAILURE"
    assert status["provider_failure"]["classification"] == "TIMEOUT"
    assert status["usage"]["new_embedding_calls"] == 0
    assert status["usage"]["new_sol_calls"] == 0
    locked = db_session.get(ResearchArchitectureRecord, V2_RESEARCH_ARCHITECTURE_ID)
    assert locked.production_status is False


def test_content_mismatch_support_fails_closed(db_session) -> None:
    from sqlalchemy import select

    from rag_workbench.answerability.validation import validate_gate_result_with_error
    from rag_workbench.db.models import Chunk
    from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
    from rag_workbench.ingestion.pipeline import IngestionPipeline
    from rag_workbench.providers.embeddings import HashingEmbeddingProvider

    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        HashingEmbeddingProvider(64),
    )
    ingested = pipeline.ingest_path(Path("data/synthetic_company/remote-work-policy-2026.md"))
    stored = db_session.scalar(
        select(Chunk).where(Chunk.document_version_id == ingested.document_version_id)
    )
    proposed = AnswerabilityResult(
        answerable=True,
        supporting_chunk_ids=(stored.id,),
        reason_code=AnswerabilityReason.SUFFICIENT_EVIDENCE,
    )
    mismatched = GateEvidence(
        stored.id,
        "remote-work-policy",
        stored.document_version_id,
        ingested.version,
        "tampered content",
    )
    validated = validate_gate_result_with_error(
        proposed,
        (mismatched,),
        session=db_session,
        principal=Principal("employee", "acmeai", frozenset({"employees"})),
    )
    assert not validated.result.answerable
    assert validated.operational_error == "INVALID_SUPPORTING_ID"


def test_acl_and_inactive_support_fail_closed(db_session) -> None:
    from sqlalchemy import select

    from rag_workbench.answerability.validation import validate_gate_result_with_error
    from rag_workbench.db.models import Chunk, DocumentVersion
    from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig
    from rag_workbench.ingestion.pipeline import IngestionPipeline
    from rag_workbench.providers.embeddings import HashingEmbeddingProvider

    pipeline = IngestionPipeline(
        db_session,
        FixedTokenChunker(FixedTokenConfig(180, 30)),
        HashingEmbeddingProvider(64),
    )
    ingested = pipeline.ingest_path(Path("data/synthetic_company/hr-benefits-private.md"))
    stored = db_session.scalar(
        select(Chunk).where(Chunk.document_version_id == ingested.document_version_id)
    )
    evidence_row = GateEvidence(
        stored.id,
        "hr-benefits",
        stored.document_version_id,
        ingested.version,
        stored.text,
    )
    proposed = AnswerabilityResult(
        answerable=True,
        supporting_chunk_ids=(stored.id,),
        reason_code=AnswerabilityReason.SUFFICIENT_EVIDENCE,
    )
    unauthorized = validate_gate_result_with_error(
        proposed,
        (evidence_row,),
        session=db_session,
        principal=Principal("employee", "acmeai", frozenset({"employees"})),
    )
    assert unauthorized.operational_error == "UNAUTHORIZED_SUPPORTING_ID"
    version = db_session.get(DocumentVersion, stored.document_version_id)
    version.is_active = False
    db_session.flush()
    inactive = validate_gate_result_with_error(
        proposed,
        (evidence_row,),
        session=db_session,
        principal=Principal("hr", "acmeai", frozenset({"hr"})),
    )
    assert inactive.operational_error == "INACTIVE_VERSION"
