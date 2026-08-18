from rag_workbench.evaluation.generation_metrics import (
    deterministic_citation_correctness,
    deterministic_citation_support,
)


def test_citation_validity_and_support_are_distinct() -> None:
    assert deterministic_citation_correctness(["noise"], ["noise"]) == 1.0
    assert (
        deterministic_citation_support(
            answer="The launch was in 2026.",
            expected_answer=None,
            cited_document_ids=["noise-doc"],
            expected_document_ids=[],
            should_abstain=True,
        )
        == 0.0
    )
