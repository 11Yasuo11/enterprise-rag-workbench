"""FINAL_E2E_SCORER_V2 — schema-native deterministic E2E scoring for frozen Phase-5K GT."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCORER_ID = "FINAL_E2E_SCORER_V2"
SCORER_VERSION = "v2.0.0"


def normalize_fact(text: str) -> str:
    return text.casefold().strip()


def fact_in_text(fact: str, text: str) -> bool:
    return normalize_fact(fact) in text.casefold()


@dataclass(frozen=True)
class FactSupportRecord:
    required_fact: str
    supporting_citation_ids: tuple[str, ...]
    support_status: str  # SUPPORTED | NOT_SUPPORTED | CITATION_SUPPORT_INDETERMINATE


@dataclass
class ScorerInput:
    arm: str
    query_id: str
    question: str
    category: str
    should_abstain: bool
    expected_answerable: bool
    required_facts: tuple[str, ...]
    required_document_ids: tuple[str, ...]
    final_answer: str | None
    final_answer_present: bool
    citation_ids: tuple[str, ...]
    cited_document_ids: tuple[str, ...]
    cited_chunk_texts: dict[str, str]
    retrieved_top_k_ids: tuple[str, ...]
    authorized_citation_ids: tuple[str, ...]
    phase5kc_behavior_label: str | None = None


@dataclass
class ScorerResult:
    facts_satisfied: list[str] = field(default_factory=list)
    facts_missing: list[str] = field(default_factory=list)
    fact_completeness_pass: bool = False
    citation_validity_pass: bool | None = None
    citation_correctness_pass: bool | None = None
    citation_completeness_pass: bool | None = None
    required_documents_satisfied: bool = False
    fact_support_records: list[FactSupportRecord] = field(default_factory=list)
    behavior: str = "SCORING_INDETERMINATE"
    citation_validity_rate: float | None = None
    citation_correctness_rate: float | None = None
    citation_completeness_rate: float | None = None
    indeterminate_reason: str | None = None


def evaluate_fact_completeness(required_facts: tuple[str, ...], answer: str | None) -> tuple[list[str], list[str], bool]:
    if not required_facts:
        return [], [], True
    if not answer:
        return [], list(required_facts), False
    satisfied = [f for f in required_facts if fact_in_text(f, answer)]
    missing = [f for f in required_facts if f not in satisfied]
    return satisfied, missing, len(missing) == 0


def evaluate_citation_validity(
    citation_ids: tuple[str, ...],
    retrieved_top_k_ids: tuple[str, ...],
    authorized_citation_ids: tuple[str, ...],
) -> tuple[bool | None, float | None]:
    if not citation_ids:
        return None, None
    allowed = set(retrieved_top_k_ids)
    authorized = set(authorized_citation_ids)
    per_cite = [
        cid in allowed and cid in authorized for cid in citation_ids
    ]
    if not per_cite:
        return None, None
    rate = sum(per_cite) / len(per_cite)
    return rate == 1.0, rate


def evaluate_fact_citation_support(
    required_facts: tuple[str, ...],
    citation_ids: tuple[str, ...],
    cited_chunk_texts: dict[str, str],
) -> list[FactSupportRecord]:
    records: list[FactSupportRecord] = []
    if not required_facts:
        return records
    if not citation_ids:
        for fact in required_facts:
            records.append(FactSupportRecord(fact, (), "NOT_SUPPORTED"))
        return records
    cited_text = " ".join(cited_chunk_texts.get(cid, "") for cid in citation_ids)
    for fact in required_facts:
        supporting = tuple(cid for cid in citation_ids if fact_in_text(fact, cited_chunk_texts.get(cid, "")))
        if supporting:
            status = "SUPPORTED"
        elif fact_in_text(fact, cited_text):
            supporting = tuple(citation_ids)
            status = "SUPPORTED"
        elif not any(cited_chunk_texts.get(cid) for cid in citation_ids):
            status = "CITATION_SUPPORT_INDETERMINATE"
        else:
            status = "NOT_SUPPORTED"
        records.append(FactSupportRecord(fact, supporting, status))
    return records


def evaluate_required_documents(
    required_document_ids: tuple[str, ...],
    cited_document_ids: tuple[str, ...],
) -> bool:
    if not required_document_ids:
        return True
    return set(required_document_ids) <= set(cited_document_ids)


def score_case(row: ScorerInput) -> ScorerResult:
    result = ScorerResult()
    answer = row.final_answer
    has_answer = row.final_answer_present and bool(answer and answer.strip())

    if row.should_abstain:
        result.behavior = "CORRECT_ABSTENTION" if not has_answer else "UNSUPPORTED_ANSWER"
        return result

    if not has_answer:
        result.behavior = "INCORRECT_ABSTENTION"
        return result

    satisfied, missing, fact_pass = evaluate_fact_completeness(row.required_facts, answer)
    result.facts_satisfied = satisfied
    result.facts_missing = missing
    result.fact_completeness_pass = fact_pass

    cvalid_pass, cvalid_rate = evaluate_citation_validity(
        row.citation_ids, row.retrieved_top_k_ids, row.authorized_citation_ids
    )
    result.citation_validity_pass = cvalid_pass
    result.citation_validity_rate = cvalid_rate

    result.fact_support_records = evaluate_fact_citation_support(
        row.required_facts, row.citation_ids, row.cited_chunk_texts
    )
    if any(r.support_status == "CITATION_SUPPORT_INDETERMINATE" for r in result.fact_support_records):
        result.behavior = "SCORING_INDETERMINATE"
        result.indeterminate_reason = "CITATION_SUPPORT_INDETERMINATE"
        return result

    supported_facts = [r for r in result.fact_support_records if r.support_status == "SUPPORTED"]
    correctness_rate = (
        len(supported_facts) / len(result.fact_support_records) if result.fact_support_records else 1.0
    )
    result.citation_correctness_rate = correctness_rate
    result.citation_correctness_pass = correctness_rate == 1.0 if result.fact_support_records else True
    result.citation_completeness_rate = correctness_rate
    result.citation_completeness_pass = result.citation_correctness_pass

    result.required_documents_satisfied = evaluate_required_documents(
        row.required_document_ids, row.cited_document_ids
    )

    all_pass = (
        result.fact_completeness_pass
        and result.citation_validity_pass is not False
        and result.citation_correctness_pass is not False
        and result.required_documents_satisfied
        and (cvalid_pass is not False)
    )
    if cvalid_pass is None and row.citation_ids:
        result.behavior = "SCORING_INDETERMINATE"
        result.indeterminate_reason = "CITATION_VALIDITY_INDETERMINATE"
        return result

    result.behavior = "CORRECT_COMPLETE_ANSWER" if all_pass else "UNSUPPORTED_ANSWER"
    return result


def scorer_definition_payload() -> dict[str, Any]:
    impl_path = Path(__file__)
    impl_hash = hashlib.sha256(impl_path.read_bytes()).hexdigest()
    full_module_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "scorer_id": SCORER_ID,
        "scorer_version": SCORER_VERSION,
        "implementation_path": str(impl_path),
        "implementation_hash": full_module_hash,
        "configuration": {
            "fact_matching_rule": "Unicode casefold substring: normalize_fact(f) in answer.casefold()",
            "citation_validity_rule": "Each citation chunk_id exists in retrieved top-k AND is authorized for principal",
            "citation_correctness_rule": "Each required_fact has >=1 cited chunk whose text casefold-contains the fact",
            "citation_completeness_rule": "All required_facts have supporting citations (same as correctness aggregate)",
            "required_document_rule": "set(required_document_ids) <= set(cited_document_ids)",
            "expected_answer_field_used": False,
            "required_chunk_ids_field_used": False,
            "llm_evaluator_used": False,
        },
        "ground_truth_fields_consumed": [
            "required_facts",
            "required_document_ids",
            "expected_answerable",
            "should_abstain",
            "category",
            "principal",
        ],
        "metric_definitions": {
            "CORRECT_COMPLETE_ANSWER": "Answerable case: answer present, all facts satisfied, citations valid/correct/complete, required docs cited, evidence authorized",
            "CORRECT_ABSTENTION": "should_abstain=true and no substantive answer",
            "INCORRECT_ABSTENTION": "Answerable case abstained",
            "INCORRECT_ANSWER": "Not primary label; folded into UNSUPPORTED_ANSWER when facts/citations fail",
            "UNSUPPORTED_ANSWER": "Answer returned but fails completeness/citation/document/authorization checks, or should_abstain with answer",
            "SCORING_INDETERMINATE": "Deterministic support cannot be resolved (missing cited text or ambiguous validity inputs)",
        },
        "behavior_priority": [
            "should_abstain guard",
            "answer presence",
            "fact completeness",
            "citation validity",
            "citation correctness/completeness",
            "required documents",
        ],
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    behaviors = [r["behavior"] for r in rows]
    cca = sum(1 for b in behaviors if b == "CORRECT_COMPLETE_ANSWER")
    cab = sum(1 for b in behaviors if b == "CORRECT_ABSTENTION")
    ia = sum(1 for b in behaviors if b == "INCORRECT_ANSWER")
    iab = sum(1 for b in behaviors if b == "INCORRECT_ABSTENTION")
    ua = sum(1 for b in behaviors if b == "UNSUPPORTED_ANSWER")
    ind = sum(1 for b in behaviors if b == "SCORING_INDETERMINATE")
    tp = cca
    fp = ua + ia
    fn = iab
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    answered = [r for r in rows if r.get("final_answer_present")]
    cvalid_vals = [r["citation_validity_rate"] for r in answered if r.get("citation_validity_rate") is not None]
    ccorr_vals = [r["citation_correctness_rate"] for r in answered if r.get("citation_correctness_rate") is not None]
    ccomp_vals = [r["citation_completeness_rate"] for r in answered if r.get("citation_completeness_rate") is not None]

    ans_rows = [r for r in rows if r.get("expected_answerable")]
    gen_comp = (
        sum(
            1
            for r in ans_rows
            if r.get("top5_complete_evidence") == 1.0 and r["behavior"] == "CORRECT_COMPLETE_ANSWER"
        )
        / sum(1 for r in ans_rows if r.get("top5_complete_evidence") == 1.0)
        if any(r.get("top5_complete_evidence") == 1.0 for r in ans_rows)
        else 0.0
    )

    from collections import Counter

    cat_total = Counter(r["category"] for r in rows)
    cat_correct = Counter(
        r["category"] for r in rows if r["behavior"] in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"}
    )

    return {
        "total": total,
        "correct_complete_answers": cca,
        "correct_abstentions": cab,
        "incorrect_answers": ia,
        "incorrect_abstentions": iab,
        "unsupported_answers": ua,
        "scoring_indeterminate": ind,
        "strict_e2e_accuracy": (cca + cab) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "citation_validity": sum(cvalid_vals) / len(cvalid_vals) if cvalid_vals else 1.0,
        "citation_correctness": sum(ccorr_vals) / len(ccorr_vals) if ccorr_vals else 1.0,
        "citation_completeness": sum(ccomp_vals) / len(ccomp_vals) if ccomp_vals else 1.0,
        "generator_completeness_given_complete_evidence": gen_comp,
        "category_correctness": {k: cat_correct[k] / v if v else 0.0 for k, v in cat_total.items()},
    }
