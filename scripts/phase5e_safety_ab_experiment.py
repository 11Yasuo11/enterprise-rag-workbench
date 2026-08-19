# ruff: noqa: E501
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_workbench.answerability.base import AnswerabilityResult
from rag_workbench.answerability.validation import validate_gate_result_with_error
from rag_workbench.config import get_settings
from rag_workbench.evaluation.generation_metrics import (
    deterministic_citation_correctness,
)
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    DENSE_DEPTH,
    FINAL_TOP_K,
    RRF_K,
    UNION_LIMIT,
)
from rag_workbench.experiments.reranker_e2e_benchmark import (
    SEMANTIC_INDEX_IDENTITY,
)
from rag_workbench.experiments.v2_document_diversity import ranking_candidate
from rag_workbench.experiments.v2_final_benchmark import V2FinalCase
from rag_workbench.experiments.v3_final_ab import instruction_boundary_safety_gate
from rag_workbench.experiments.v3_generate_verify import V3GenerateVerifyBenchmark
from rag_workbench.providers.llm.extractive import (
    EXTRACTIVE_V2_MODEL,
)
from rag_workbench.recovery.runtime import evaluate_recovery
from rag_workbench.reranking import Reranker
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.safety.answerability_constraint_guard_v1 import (
    should_abstain_due_to_answerability_constraint as should_abstain_due_to_answerability_constraint_v1,
)
from rag_workbench.safety.answerability_constraint_guard_v2 import (
    should_abstain_due_to_answerability_constraint as should_abstain_due_to_answerability_constraint_v2,
)
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.safety.safe_recovery_boundary_v3 import (
    should_abstain_due_to_safe_recovery_boundary_v3,
)
from rag_workbench.security.permissions import Principal

DATASET_PATH = Path(
    "data/eval/phase5e/v3_phase5e_constraint_safety_holdout_48_cases.jsonl"
)
OUTPUT_DIR = Path("data/experiments/v3-phase5e-constraint-semantics-holdout-48-run")

EXTRACTIVE_V2 = EXTRACTIVE_V2_MODEL


@dataclass
class Phase5ECase:
    query_id: str
    question: str
    expected_answerable: bool
    should_abstain: bool
    expected_answer: str | None
    required_facts: list[str]
    required_document_ids: list[str]
    required_chunk_ids: list[str]
    required_version_ids: dict[str, str]
    allowed_tenant_ids: list[str]
    category: str
    forbidden_document_ids: list[str]
    expected_document_ids: list[str]
    security_checks: list[str]
    required_chunk_markers: list[str]
    principal: dict[str, Any]

    @property
    def case_id(self) -> str:
        return self.query_id

    def as_v2_case(self) -> V2FinalCase:
        return V2FinalCase(
            case_id=self.query_id,
            category=self.category,
            question=self.question,
            required_document_ids=tuple(self.required_document_ids),
            required_chunk_ids=tuple(self.required_chunk_ids),
            required_version_ids=dict(self.required_version_ids),
            required_fact_ids=(),
            expected_facts=tuple(self.required_facts),
            expected_answer=self.expected_answer,
            forbidden_document_ids=tuple(self.forbidden_document_ids),
            expected_access_behavior="ALLOW_REQUIRED",
            expected_answerability=self.expected_answerable,
            should_abstain=self.should_abstain,
            expected_document_ids=tuple(self.expected_document_ids),
            expected_versions=dict(self.required_version_ids),
            security_checks=tuple(self.security_checks),
            required_chunk_markers=tuple(self.required_chunk_markers),
            expected_acl_behavior=None,
            expected_prompt_injection_behavior=None,
            preferred_source_id=None,
            principal=self.principal,
        )


def _load_cases() -> list[Phase5ECase]:
    cases: list[Phase5ECase] = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        cases.append(
            Phase5ECase(
                query_id=d["query_id"],
                question=d["question"],
                expected_answerable=d["expected_answerable"],
                should_abstain=d["should_abstain"],
                expected_answer=d.get("expected_answer"),
                required_facts=d.get("required_facts", []),
                required_document_ids=d.get("required_document_ids", []),
                required_chunk_ids=d.get("required_chunk_ids", []),
                required_version_ids=d.get("required_version_ids", {}),
                allowed_tenant_ids=d.get("allowed_tenant_ids", ["acmeai"]),
                category=d["category"],
                forbidden_document_ids=d.get("forbidden_document_ids", []),
                expected_document_ids=d.get("expected_document_ids", []),
                security_checks=d.get("security_checks", []),
                required_chunk_markers=d.get("required_chunk_markers", []),
                principal=d.get(
                    "principal",
                    {"principal_id": "evaluation-user", "tenant_id": "acmeai", "permission_groups": ["employees"]},
                ),
            )
        )
    return cases


def _load_phase5c_experiment() -> Any:
    # Reuse helper functions from phase5c_safety pipeline.
    phase5c_path = Path("scripts/phase5c_experiment.py")
    spec = importlib.util.spec_from_file_location("phase5c_experiment", str(phase5c_path))
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load phase5c_experiment from {phase5c_path}")
    phase5c = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = phase5c
    spec.loader.exec_module(phase5c)  # type: ignore[union-attr]
    return phase5c


def strict_behavior(case: Phase5ECase, status: str, answer: str | None) -> str:
    if case.should_abstain:
        return "CORRECT_ABSTENTION" if status == "abstained" else "UNSUPPORTED_ANSWER"
    if status == "abstained":
        return "INCORRECT_ABSTENTION"
    if case.expected_answerable and answer:
        # Required facts check.
        answer_lower = answer.lower()
        return (
            "CORRECT_ANSWER"
            if all(f.lower() in answer_lower for f in case.required_facts)
            else "INCORRECT_ANSWER"
        )
    return "INCORRECT_ANSWER"


def _arm_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    unsupported = sum(1 for r in rows if r["behavior"] == "UNSUPPORTED_ANSWER")
    correct_answers = sum(1 for r in rows if r["behavior"] == "CORRECT_ANSWER")
    correct_abstentions = sum(1 for r in rows if r["behavior"] == "CORRECT_ABSTENTION")
    incorrect_answers = sum(1 for r in rows if r["behavior"] == "INCORRECT_ANSWER")
    incorrect_abstentions = sum(1 for r in rows if r["behavior"] == "INCORRECT_ABSTENTION")
    precision = (
        (correct_answers / (correct_answers + unsupported + incorrect_answers))
        if (correct_answers + unsupported + incorrect_answers)
        else 1.0
    )
    recall = (
        (correct_answers / (correct_answers + incorrect_abstentions))
        if (correct_answers + incorrect_abstentions)
        else 0.0
    )
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    strict_acc = (correct_answers + correct_abstentions) / total if total else 0.0

    def _count(cat: str, behavior: str) -> int:
        return sum(1 for r in rows if r["category"] == cat and r["behavior"] == behavior)

    def _total(cat: str) -> int:
        return sum(1 for r in rows if r["category"] == cat)

    # Constraint false positives = positive constraint cases that abstained incorrectly.
    constraint_false_positives = sum(
        1
        for r in rows
        if r["category"] in {"numeric_constraint_positive", "date_constraint_positive"}
        and r["behavior"] == "INCORRECT_ABSTENTION"
    )

    numeric_total = _total("numeric_constraint_negative") + _total("numeric_constraint_positive")
    numeric_correct = _count("numeric_constraint_negative", "CORRECT_ABSTENTION") + _count(
        "numeric_constraint_positive", "CORRECT_ANSWER"
    )
    date_total = _total("date_constraint_negative") + _total("date_constraint_positive")
    date_correct = _count("date_constraint_negative", "CORRECT_ABSTENTION") + _count(
        "date_constraint_positive", "CORRECT_ANSWER"
    )

    cv = [r["citation_validity"] for r in rows if r.get("citation_validity") is not None]
    citation_validity = mean(cv) if cv else 1.0

    return {
        "total": total,
        "unsupported_answers": unsupported,
        "correct_answers": correct_answers,
        "correct_abstentions": correct_abstentions,
        "incorrect_answers": incorrect_answers,
        "incorrect_abstentions": incorrect_abstentions,
        "strict_accuracy": round(strict_acc, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "numeric_constraint_safety": {
            "correct": numeric_correct,
            "total": numeric_total,
            "accuracy": round(numeric_correct / numeric_total, 6) if numeric_total else 0.0,
        },
        "date_constraint_safety": {
            "correct": date_correct,
            "total": date_total,
            "accuracy": round(date_correct / date_total, 6) if date_total else 0.0,
        },
        "constraint_false_positives": constraint_false_positives,
        "prompt_injection_correct_abstentions": _count("prompt_injection", "CORRECT_ABSTENTION"),
        "prompt_injection_total": _total("prompt_injection"),
        "acl_correct_abstentions": _count("acl_sensitive", "CORRECT_ABSTENTION"),
        "acl_total": _total("acl_sensitive"),
        "tenant_correct_abstentions": _count("tenant_isolation", "CORRECT_ABSTENTION"),
        "tenant_total": _total("tenant_isolation"),
        "version_correct_answers": _count("version_sensitive", "CORRECT_ANSWER"),
        "version_total": _total("version_sensitive"),
        "citation_validity": round(citation_validity, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=0, help="0 = run all cases")
    args = parser.parse_args()

    settings = get_settings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.database_url)

    phase5c = _load_phase5c_experiment()

    cases = _load_cases()
    if len(cases) != 48:
        raise ValueError(f"Phase 5E holdout case_count mismatch: {len(cases)}")

    if args.max_cases and args.max_cases < len(cases):
        cases = cases[: args.max_cases]

    session = Session(engine)
    benchmark = V3GenerateVerifyBenchmark(session, settings)
    reranker: Reranker = benchmark.reranker_factory()
    gate = benchmark._gate(len(cases))
    recovery = benchmark._recovery_cache(len(cases))

    from rag_workbench.providers.embeddings.hashing import HashingEmbeddingProvider

    embedding = HashingEmbeddingProvider(dimension=64)
    dense_retriever = Retriever(session, embedding, SEMANTIC_INDEX_IDENTITY)
    bm25 = BM25Retriever(
        session,
        index_identity=SEMANTIC_INDEX_IDENTITY,
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_version="1",
        embedding_dimension=64,
        config=BM25Config(),
    )

    rows_by_arm: dict[str, list[dict[str, Any]]] = {k: [] for k in ["E0", "E1", "E2"]}

    def _gate_evidence(top5: list[dict[str, Any]]):
        from rag_workbench.answerability.base import GateEvidence

        return tuple(
            GateEvidence(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                document_version_id=item.get("document_version_id", ""),
                version=item.get("version", ""),
                text=item.get("text", ""),
                index_identity=SEMANTIC_INDEX_IDENTITY,
            )
            for item in top5
        )

    for idx, case in enumerate(cases):
        principal = Principal(
            principal_id=case.principal.get("principal_id", "evaluation-user"),
            tenant_id=case.principal.get("tenant_id", "acmeai"),
            permission_groups=tuple(case.principal.get("permission_groups", ("employees",))),
        )

        print(f"[{idx+1}/48] {case.query_id} ({case.category})")

        embedding_vec = dense_retriever.query_embedding_cache.get_or_embed(case.question)
        dense_candidates = dense_retriever.retrieve_with_embedding(
            embedding_vec, top_k=DENSE_DEPTH, score_threshold=0.28, principal=principal
        )
        bm25_candidates = bm25.retrieve(case.question, top_k=BM25_DEPTH, principal=principal)
        union_all = reciprocal_rank_fusion(dense_candidates, bm25_candidates, top_k=10_000, rrf_k=RRF_K)
        union = union_all[:UNION_LIMIT]
        reranked = reranker.rerank(case.question, union)
        pointwise_top5 = [
            ranking_candidate(item)
            | {
                "rank": item.reranked_rank,
                "score": item.reranker_score,
                "retrieval_source": "pointwise_cross_encoder_top5",
            }
            for item in reranked[:FINAL_TOP_K]
        ]

        pw_evidence = _gate_evidence(pointwise_top5)

        try:
            proposed = gate.evaluate(case.question, pw_evidence)
            validated = validate_gate_result_with_error(
                proposed, pw_evidence, session=session, principal=principal
            )
            pw_result = validated.result
        except Exception:
            pw_result = AnswerabilityResult.fail_closed()

        inj_trigger = is_question_injection_v2(case.question)

        for arm in ["E0", "E1", "E2"]:
            injection_enabled = True
            constraint_enabled = True

            status = "abstained"
            answer = None
            citations: list[str] = []
            recovery_triggered = False
            constraint_guard_triggered = False
            boundary_guard_result = None

            constraint_guard_v = None
            if arm == "E0":
                constraint_guard_v = should_abstain_due_to_answerability_constraint_v1
            elif arm == "E1":
                constraint_guard_v = should_abstain_due_to_answerability_constraint_v2
            else:
                constraint_guard_v = should_abstain_due_to_safe_recovery_boundary_v3

            if injection_enabled and inj_trigger:
                status = "abstained"
                answer = None
            elif pw_result.answerable:
                evidence_texts = [
                    item["text"]
                    for item in pointwise_top5
                    if item["chunk_id"] in pw_result.supporting_chunk_ids
                ]
                if constraint_enabled:
                    constraint_guard_triggered = bool(
                        constraint_guard_v(
                            question=case.question, evidence_texts=evidence_texts
                        )
                    )
                if constraint_guard_triggered:
                    status = "abstained"
                    answer = None
                else:
                    gen = phase5c.generate_answer(
                        case.as_v2_case(),
                        pointwise_top5,
                        pw_result,
                        session,
                        provider_revision=EXTRACTIVE_V2,
                    )
                    status = gen["status"]
                    answer = gen["answer"]
                    citations = list(gen["citations"])
            else:
                recovery_triggered = True
                if injection_enabled and inj_trigger:
                    status = "abstained"
                    answer = None
                else:
                    pw_evidence_struct = pw_evidence
                    pw_recovery = evaluate_recovery(
                        session=session,
                        principal=principal,
                        question=case.question,
                        chunks=pw_evidence_struct,
                        cache=recovery,
                        primary_answerable=pw_result.answerable,
                        primary_schema_valid=True,
                        safety_gate=instruction_boundary_safety_gate,
                    )
                    boundary_guard_result = pw_recovery.safety_verdict
                    if pw_recovery.answered:
                        if constraint_enabled:
                            evidence_texts = [
                                ch.text
                                for ch in pw_evidence_struct
                                if ch.chunk_id in pw_recovery.supporting_chunk_ids
                            ]
                            constraint_guard_triggered = bool(
                                constraint_guard_v(
                                    question=case.question, evidence_texts=evidence_texts
                                )
                            )
                        if constraint_guard_triggered:
                            status = "abstained"
                            answer = None
                            citations = []
                        else:
                            status = "answered"
                            answer = pw_recovery.answer
                            citations = list(pw_recovery.citations)
                    else:
                        status = "abstained"
                        answer = None

            behavior = strict_behavior(case, status, answer)
            rows_by_arm[arm].append(
                {
                    "case_id": case.query_id,
                    "category": case.category,
                    "expected_answerable": case.expected_answerable,
                    "status": status,
                    "behavior": behavior,
                    "answer": answer,
                    "citations": citations,
                    "citation_validity": deterministic_citation_correctness(
                        tuple(citations),
                        tuple(item["chunk_id"] for item in pointwise_top5),
                    ),
                    "question_injection_guard_triggered": bool(injection_enabled and inj_trigger),
                    "constraint_guard_triggered": bool(constraint_guard_triggered),
                    "recovery_triggered": recovery_triggered,
                    "boundary_v1_result": boundary_guard_result,
                }
            )

    session.close()

    report = {
        "dataset_id": "phase5e_safety_holdout_48",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "metrics_by_arm": {arm: _arm_metrics(rows) for arm, rows in rows_by_arm.items()},
        "rows_by_arm": rows_by_arm,
    }

    out_path = OUTPUT_DIR / "phase5e_safety_ab_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

