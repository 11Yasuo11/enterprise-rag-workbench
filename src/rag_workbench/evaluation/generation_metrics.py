import re
from collections.abc import Iterable


def abstention_correct(actual_status: str, should_abstain: bool) -> float:
    return float((actual_status == "abstained") == should_abstain)


def abstention_classification(
    expected: Iterable[bool], actual: Iterable[bool]
) -> dict[str, float | None]:
    pairs = list(zip(expected, actual, strict=True))
    true_positive = sum(expected_value and actual_value for expected_value, actual_value in pairs)
    false_positive = sum(
        not expected_value and actual_value for expected_value, actual_value in pairs
    )
    false_negative = sum(
        expected_value and not actual_value for expected_value, actual_value in pairs
    )
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else None
    recall = true_positive / recall_denominator if recall_denominator else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": f1,
    }


def deterministic_citation_correctness(
    cited_chunk_ids: Iterable[str], retrieved_chunk_ids: Iterable[str]
) -> float | None:
    citations = set(cited_chunk_ids)
    if not citations:
        return None
    return float(citations <= set(retrieved_chunk_ids))


def deterministic_citation_support(
    *,
    answer: str | None,
    expected_answer: str | None,
    cited_document_ids: Iterable[str],
    expected_document_ids: Iterable[str],
    should_abstain: bool,
) -> float | None:
    """Document-level support using only explicit dataset answer/source labels.

    The dataset has no claim spans or evidence quotes, so this deliberately does not claim
    sentence-level entailment. It requires expected-answer text and every required source.
    """
    if not answer:
        return None
    if should_abstain or not expected_answer:
        return 0.0

    stop_words = {"a", "an", "and", "are", "is", "of", "the", "to"}

    def terms(value: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", value.lower())) - stop_words

    expected_sources = set(expected_document_ids)
    cited_sources = set(cited_document_ids)
    expected_terms = terms(expected_answer)
    answer_supported = bool(expected_terms) and expected_terms <= terms(answer)
    sources_supported = bool(expected_sources) and expected_sources <= cited_sources
    return float(answer_supported and sources_supported)


def answerability_classification(
    expected_answerable: Iterable[bool], actual_answerable: Iterable[bool]
) -> dict[str, float | int | None]:
    pairs = list(zip(expected_answerable, actual_answerable, strict=True))
    tp = sum(expected and actual for expected, actual in pairs)
    tn = sum(not expected and not actual for expected, actual in pairs)
    fp = sum(not expected and actual for expected, actual in pairs)
    fn = sum(expected and not actual for expected, actual in pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "answerability_accuracy": (tp + tn) / len(pairs) if pairs else None,
        "answerability_precision": precision,
        "answerability_recall": recall,
        "answerability_f1": f1,
        "correct_answer_count": tp,
        "correct_abstention_count": tn,
        "unsupported_answer_count": fp,
        "incorrect_abstention_count": fn,
    }


# Claim-level entailment and completeness still require claim annotations or a separately
# validated judge. The deterministic support metric above is intentionally document-level.
