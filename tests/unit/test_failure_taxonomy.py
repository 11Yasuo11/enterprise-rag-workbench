from rag_workbench.evaluation.datasets import EvaluationCase
from rag_workbench.evaluation.evaluator import EvaluationRunner
from rag_workbench.evaluation.failures import FailureType
from rag_workbench.retrieval.retriever import meets_score_threshold
from rag_workbench.retrieval.vector_search import RetrievalResult


def _case(**overrides) -> EvaluationCase:
    payload = {
        "case_id": "case-1",
        "question": "What is the answer?",
        "expected_answer": "answer",
        "expected_document_ids": ["doc-a", "doc-b"],
        "category": "multi_document",
    }
    payload.update(overrides)
    return EvaluationCase.model_validate(payload)


def _result(document_id: str, score: float = 0.4) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"chunk-{document_id}",
        document_id=document_id,
        document_version_id=f"version-{document_id}",
        text="answer evidence",
        rank=1,
        score=score,
        source=f"{document_id}.md",
        source_type="markdown",
        title=document_id,
        version="1",
    )


def test_multi_document_failure_records_missing_source_evidence() -> None:
    failures, details = EvaluationRunner._classify_failures(
        _case(),
        0.5,
        "answered",
        1.0,
        None,
        None,
        ("doc-a",),
        (_result("doc-a"),),
        ("doc-a",),
        (),
        0.2,
    )
    assert FailureType.RETRIEVAL_MISS in failures
    assert details["missing_document_ids"] == ["doc-b"]
    assert details["multi_document_failure"] == "one_or_more_required_sources_missing"


def test_false_positive_abstention_has_distinct_taxonomy() -> None:
    failures, details = EvaluationRunner._classify_failures(
        _case(
            expected_answer=None,
            expected_document_ids=[],
            should_abstain=True,
            category="abstention",
        ),
        0.0,
        "answered",
        1.0,
        None,
        None,
        ("public-noise",),
        (_result("public-noise", 0.25),),
        ("public-noise",),
        (),
        0.2,
    )
    assert failures == (
        FailureType.ABSTENTION_FALSE_POSITIVE,
        FailureType.THRESHOLD_FAILURE,
    )
    assert details["retrieval_scores"] == [0.25]


def test_similarity_threshold_semantics_are_higher_is_better() -> None:
    assert meets_score_threshold(0.4, 0.3)
    assert not meets_score_threshold(0.2, 0.3)
    assert meets_score_threshold(-1.0, None)


def test_multi_document_gate_context_loss_has_distinct_taxonomy() -> None:
    results = (_result("doc-a"), _result("doc-b"))
    failures, _ = EvaluationRunner._classify_failures(
        _case(),
        1.0,
        "answered",
        1.0,
        None,
        None,
        ("doc-a", "doc-b"),
        results,
        ("doc-a",),
        (),
        0.28,
        True,
        (results[0].chunk_id,),
        (results[0].chunk_id,),
    )
    assert FailureType.SUPPORTING_CONTEXT_LOSS in failures
