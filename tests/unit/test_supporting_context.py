from rag_workbench.answerability import (
    AnswerabilityReason,
    AnswerabilityResult,
    GateTiming,
    ValidatedAnswerabilityResult,
)
from rag_workbench.generation.context_builder import ContextBuilder
from rag_workbench.generation.generator import RagService
from rag_workbench.providers.llm import ExtractiveGenerationProvider
from rag_workbench.retrieval.vector_search import RetrievalResult


class FixedGate:
    provider_name = "test"
    model_name = "test"
    gate_version = "1"
    prompt_version = "v1"
    last_timing = GateTiming()

    def __init__(self, ids=(), answerable=True):
        self.result = AnswerabilityResult(
            answerable=answerable,
            supporting_chunk_ids=tuple(ids),
            reason_code=(
                AnswerabilityReason.SUFFICIENT_EVIDENCE
                if answerable
                else AnswerabilityReason.MISSING_REQUIRED_FACT
            ),
        )

    def evaluate(self, question, chunks):
        return self.result


def test_supporting_context_pruning_and_abstention(monkeypatch) -> None:
    retrieval_result = (
        RetrievalResult(
            chunk_id="chunk-1",
            document_id="policy",
            document_version_id="version-1",
            text="The recovery time is four hours.",
            rank=1,
            score=0.9,
            source="policy.md",
            source_type="markdown",
            title="Policy",
            version="1",
            section="Recovery",
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            document_id="noise",
            document_version_id="version-2",
            text="The office is closed Sunday.",
            rank=2,
            score=0.8,
            source="noise.md",
            source_type="markdown",
            title="Noise",
            version="1",
            section="Hours",
        ),
    )
    class Retriever:
        embedding_provider = type(
            "Embedding", (), {"usage": type("U", (), {"input_tokens": 0})()}
        )()
        last_timing = type(
            "Timing",
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
            return list(retrieval_result)

    class Session:
        def add(self, value):
            if value.__class__.__name__ == "RagRun":
                value.id = "run-1"

        def flush(self):
            pass

        def commit(self):
            pass

    monkeypatch.setattr(RagService, "_document_fk", lambda self, chunk_id: "doc-fk")
    selected = retrieval_result[0].chunk_id
    service = RagService(
        Session(),
        Retriever(),
        ContextBuilder(),
        ExtractiveGenerationProvider(),
        FixedGate((selected,)),
        supporting_context_only=True,
    )
    from rag_workbench.security.permissions import Principal

    monkeypatch.setattr(
        "rag_workbench.generation.generator.validate_gate_result_with_error",
        lambda result, chunks, **kwargs: ValidatedAnswerabilityResult(result),
    )
    response = service.query("What is the recovery time?", Principal("p", "acmeai"))
    assert response.generation_context_chunk_ids == (selected,)

    service.answerability_gate = FixedGate(answerable=False)
    response = service.query("Unknown future fact?", Principal("p", "acmeai"))
    assert response.status == "abstained"
    assert response.generation_context_chunk_ids == ()
