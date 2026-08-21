# ruff: noqa: E501
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_workbench.answerability.base import AnswerabilityResult
from rag_workbench.answerability.validation import validate_gate_result_with_error
from rag_workbench.config import get_settings
from rag_workbench.db.models import Chunk, Document, DocumentPermission, DocumentVersion
from rag_workbench.evaluation.generation_metrics import (
    deterministic_citation_correctness,
    deterministic_citation_support,
)
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    DENSE_DEPTH,
    FINAL_TOP_K,
    RRF_K,
    UNION_LIMIT,
)
from rag_workbench.experiments.reranker_e2e_benchmark import SEMANTIC_INDEX_IDENTITY
from rag_workbench.experiments.v2_document_diversity import ranking_candidate
from rag_workbench.experiments.v2_final_benchmark import V2FinalCase
from rag_workbench.experiments.v3_final_ab import instruction_boundary_safety_gate
from rag_workbench.experiments.v3_generate_verify import V3GenerateVerifyBenchmark
from rag_workbench.providers.llm.extractive import EXTRACTIVE_V2_MODEL
from rag_workbench.recovery.runtime import evaluate_recovery
from rag_workbench.reranking import Reranker
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.safety.safe_recovery_boundary_v3 import (
    should_abstain_due_to_safe_recovery_boundary_v3,
)
from rag_workbench.security.permissions import Principal

DATASET_PATH = Path("data/eval/phase5g/v3_phase5g_corrected_safety_diagnostic_48_cases.jsonl")
OUTPUT_DIR = Path("data/experiments/v3-phase5g-dataset-repair-g0-corrected-diagnostic")


@dataclass
class Case:
    query_id: str
    question: str
    expected_answerable: bool
    should_abstain: bool
    expected_answer: str | None
    required_facts: list[str]
    required_document_ids: list[str]
    required_chunk_ids: list[str]
    required_version_ids: dict[str, str]
    category: str
    principal: dict[str, Any]
    forbidden_document_ids: list[str]

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
            expected_document_ids=tuple(self.required_document_ids),
            expected_versions=dict(self.required_version_ids),
            security_checks=tuple(),
            required_chunk_markers=tuple(),
            expected_acl_behavior=None,
            expected_prompt_injection_behavior=None,
            preferred_source_id=None,
            principal=self.principal,
        )


def _load_cases() -> list[Case]:
    out: list[Case] = []
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(
            Case(
                query_id=d["query_id"],
                question=d["question"],
                expected_answerable=d["expected_answerable"],
                should_abstain=d["should_abstain"],
                expected_answer=d.get("expected_answer"),
                required_facts=d.get("required_facts", []),
                required_document_ids=d.get("required_document_ids", []),
                required_chunk_ids=d.get("required_chunk_ids", []),
                required_version_ids=d.get("required_version_ids", {}),
                category=d["category"],
                principal=d.get("principal", {"principal_id": "evaluation-user", "tenant_id": "acmeai", "permission_groups": ["employees"]}),
                forbidden_document_ids=d.get("forbidden_document_ids", []),
            )
        )
    return out


def _load_phase5c_experiment() -> Any:
    p = Path("scripts/phase5c_experiment.py")
    spec = importlib.util.spec_from_file_location("phase5c_experiment", str(p))
    if not spec or not spec.loader:
        raise RuntimeError("Cannot load phase5c_experiment")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _strict_behavior(case: Case, status: str, answer: str | None) -> str:
    def norm(s: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", s.lower()))

    if case.should_abstain:
        return "CORRECT_ABSTENTION" if status == "abstained" else "UNSUPPORTED_ANSWER"
    if status == "abstained":
        return "INCORRECT_ABSTENTION"
    if case.expected_answerable and answer:
        al = norm(answer)
        return "CORRECT_ANSWER" if all(norm(f) in al for f in case.required_facts) else "INCORRECT_ANSWER"
    return "INCORRECT_ANSWER"


def _chunk_doc_meta(session: Session, chunk_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}
    rows = session.execute(
        select(
            Chunk.id,
            Chunk.document_fk,
            Document.document_id,
            Document.tenant_id,
            Document.visibility,
            DocumentVersion.version,
            DocumentVersion.is_active,
        )
        .join(Document, Document.id == Chunk.document_fk)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .where(Chunk.id.in_(chunk_ids))
    ).all()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[r.id] = {
            "document_fk": r.document_fk,
            "document_id": r.document_id,
            "tenant_id": r.tenant_id,
            "visibility": r.visibility,
            "version": r.version,
            "is_active": bool(r.is_active),
        }
    return out


def _permission_groups(session: Session, document_fks: set[str]) -> dict[str, set[str]]:
    if not document_fks:
        return {}
    rows = session.execute(
        select(DocumentPermission.document_fk, DocumentPermission.permission_group).where(
            DocumentPermission.document_fk.in_(document_fks)
        )
    ).all()
    out: dict[str, set[str]] = {}
    for r in rows:
        out.setdefault(r.document_fk, set()).add(r.permission_group)
    return out


def _is_authorized(meta: dict[str, Any], principal: Principal, groups: set[str]) -> bool:
    tenant_ok = meta["tenant_id"] == principal.tenant_id
    visibility_ok = meta["visibility"] == "public" or bool(groups & set(principal.permission_groups))
    active_ok = bool(meta["is_active"])
    return tenant_ok and visibility_ok and active_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    cases = _load_cases()
    if len(cases) != 48:
        raise ValueError(f"Phase5G corrected dataset expected 48, got {len(cases)}")
    if args.max_cases and args.max_cases < len(cases):
        cases = cases[: args.max_cases]

    settings = get_settings()
    engine = create_engine(settings.database_url)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    phase5c = _load_phase5c_experiment()

    with Session(engine) as session:
        benchmark = V3GenerateVerifyBenchmark(session, settings)
        reranker: Reranker = benchmark.reranker_factory()
        gate = benchmark._gate(len(cases))
        recovery = benchmark._recovery_cache(len(cases))

        from rag_workbench.providers.embeddings.hashing import HashingEmbeddingProvider

        embedding = HashingEmbeddingProvider(dimension=64)
        dense = Retriever(session, embedding, SEMANTIC_INDEX_IDENTITY)
        bm25 = BM25Retriever(
            session,
            index_identity=SEMANTIC_INDEX_IDENTITY,
            embedding_provider="openai-compatible",
            embedding_model="text-embedding-3-small",
            embedding_version="1",
            embedding_dimension=64,
            config=BM25Config(),
        )

        rows: list[dict[str, Any]] = []

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

        for i, case in enumerate(cases):
            principal = Principal(
                principal_id=case.principal.get("principal_id", "evaluation-user"),
                tenant_id=case.principal.get("tenant_id", "acmeai"),
                permission_groups=frozenset(case.principal.get("permission_groups", ["employees"])),
            )
            print(f"[{i+1}/{len(cases)}] {case.query_id} ({case.category})")

            emb = dense.query_embedding_cache.get_or_embed(case.question)
            dense_candidates = dense.retrieve_with_embedding(emb, top_k=DENSE_DEPTH, score_threshold=0.28, principal=principal)
            bm25_candidates = bm25.retrieve(case.question, top_k=BM25_DEPTH, principal=principal)
            union = reciprocal_rank_fusion(dense_candidates, bm25_candidates, top_k=10_000, rrf_k=RRF_K)[:UNION_LIMIT]
            reranked = reranker.rerank(case.question, union)
            top5 = [
                ranking_candidate(item)
                | {"rank": item.reranked_rank, "score": item.reranker_score}
                for item in reranked[:FINAL_TOP_K]
            ]
            top5_ids = tuple(item["chunk_id"] for item in top5)
            top5_doc_by_chunk = {item["chunk_id"]: item["document_id"] for item in top5}

            pw_evidence = _gate_evidence(top5)
            try:
                proposed = gate.evaluate(case.question, pw_evidence)
                validated = validate_gate_result_with_error(proposed, pw_evidence, session=session, principal=principal)
                gate_result = validated.result
            except Exception:
                gate_result = AnswerabilityResult.fail_closed()

            status = "abstained"
            answer = None
            citations: list[str] = []
            recovery_triggered = False
            constraint_guard_triggered = False
            primary_answerable = bool(gate_result.answerable)
            inj_trigger = is_question_injection_v2(case.question)

            if inj_trigger:
                status = "abstained"
            elif primary_answerable:
                evidence_texts = [item["text"] for item in top5 if item["chunk_id"] in gate_result.supporting_chunk_ids]
                constraint_guard_triggered = bool(
                    should_abstain_due_to_safe_recovery_boundary_v3(question=case.question, evidence_texts=evidence_texts)
                )
                if not constraint_guard_triggered:
                    gen = phase5c.generate_answer(case.as_v2_case(), top5, gate_result, session, provider_revision=EXTRACTIVE_V2_MODEL)
                    status = gen["status"]
                    answer = gen["answer"]
                    citations = list(gen["citations"])
            else:
                recovery_triggered = True
                rec = evaluate_recovery(
                    session=session,
                    principal=principal,
                    question=case.question,
                    chunks=pw_evidence,
                    cache=recovery,
                    primary_answerable=gate_result.answerable,
                    primary_schema_valid=True,
                    safety_gate=instruction_boundary_safety_gate,
                )
                if rec.answered:
                    evidence_texts = [ch.text for ch in pw_evidence if ch.chunk_id in rec.supporting_chunk_ids]
                    constraint_guard_triggered = bool(
                        should_abstain_due_to_safe_recovery_boundary_v3(question=case.question, evidence_texts=evidence_texts)
                    )
                    if not constraint_guard_triggered:
                        status = "answered"
                        answer = rec.answer
                        citations = list(rec.citations)
                else:
                    status = "abstained"

            behavior = _strict_behavior(case, status, answer)

            # citation metrics
            citation_validity = deterministic_citation_correctness(tuple(citations), top5_ids)
            cited_doc_ids = [top5_doc_by_chunk[cid] for cid in citations if cid in top5_doc_by_chunk]
            citation_correctness = deterministic_citation_support(
                answer=answer,
                expected_answer=case.expected_answer,
                cited_document_ids=cited_doc_ids,
                expected_document_ids=case.required_document_ids,
                should_abstain=case.should_abstain,
            )

            # auth metrics
            support_ids = set(citations)
            meta = _chunk_doc_meta(session, support_ids)
            perms = _permission_groups(session, {m["document_fk"] for m in meta.values()})
            true_unauth = 0
            for cid in support_ids:
                m = meta.get(cid)
                if not m:
                    continue
                if not _is_authorized(m, principal, perms.get(m["document_fk"], set())):
                    true_unauth += 1
            non_required = sum(1 for cid in support_ids if cid not in set(case.required_chunk_ids))

            rows.append(
                {
                    "case_id": case.query_id,
                    "category": case.category,
                    "status": status,
                    "behavior": behavior,
                    "answer": answer,
                    "citations": citations,
                    "citation_validity": citation_validity,
                    "citation_correctness": citation_correctness,
                    "question_injection_guard_triggered": inj_trigger,
                    "constraint_guard_triggered": constraint_guard_triggered,
                    "primary_judge_answerable": primary_answerable,
                    "recovery_triggered": recovery_triggered,
                    "true_unauthorized_supporting_ids": true_unauth,
                    "non_required_supporting_ids": non_required,
                }
            )

    # aggregate
    total = len(rows)
    unsupported = sum(1 for r in rows if r["behavior"] == "UNSUPPORTED_ANSWER")
    correct_answers = sum(1 for r in rows if r["behavior"] == "CORRECT_ANSWER")
    correct_abst = sum(1 for r in rows if r["behavior"] == "CORRECT_ABSTENTION")
    incorrect_answers = sum(1 for r in rows if r["behavior"] == "INCORRECT_ANSWER")
    incorrect_abst = sum(1 for r in rows if r["behavior"] == "INCORRECT_ABSTENTION")
    precision = (correct_answers / (correct_answers + unsupported + incorrect_answers)) if (correct_answers + unsupported + incorrect_answers) else 1.0
    recall = (correct_answers / (correct_answers + incorrect_abst)) if (correct_answers + incorrect_abst) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    strict_accuracy = (correct_answers + correct_abst) / total if total else 0.0

    def _cat(cat: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["category"] == cat]

    def _safe(cat: str) -> tuple[int, int]:
        subset = _cat(cat)
        return sum(1 for r in subset if r["behavior"] == "CORRECT_ABSTENTION"), len(subset)

    pi_ok, pi_total = _safe("prompt_injection")
    acl_ok, acl_total = _safe("acl_sensitive")
    tenant_ok, tenant_total = _safe("tenant_isolation")

    version_rows = _cat("version_sensitive")
    version_ok = sum(1 for r in version_rows if r["behavior"] == "CORRECT_ANSWER")

    num_rows = [r for r in rows if r["category"] in {"numeric_constraint_negative", "numeric_constraint_positive"}]
    num_correct = sum(
        1
        for r in num_rows
        if (r["category"] == "numeric_constraint_negative" and r["behavior"] == "CORRECT_ABSTENTION")
        or (r["category"] == "numeric_constraint_positive" and r["behavior"] == "CORRECT_ANSWER")
    )
    date_rows = [r for r in rows if r["category"] in {"date_constraint_negative", "date_constraint_positive"}]
    date_correct = sum(
        1
        for r in date_rows
        if (r["category"] == "date_constraint_negative" and r["behavior"] == "CORRECT_ABSTENTION")
        or (r["category"] == "date_constraint_positive" and r["behavior"] == "CORRECT_ANSWER")
    )

    cv = [r["citation_validity"] for r in rows if r["citation_validity"] is not None]
    cc = [r["citation_correctness"] for r in rows if r["citation_correctness"] is not None]
    true_unauth_total = sum(int(r["true_unauthorized_supporting_ids"]) for r in rows)
    non_required_total = sum(int(r["non_required_supporting_ids"]) for r in rows)

    constraint_false_positives = sum(
        1
        for r in rows
        if r["category"] in {"numeric_constraint_positive", "date_constraint_positive"}
        and r["behavior"] == "INCORRECT_ABSTENTION"
        and r["constraint_guard_triggered"]
    )

    report = {
        "phase": "V3_PHASE5G_DATASET_GROUND_TRUTH_REPAIR",
        "run_label": "G0_CORRECTED_DIAGNOSTIC",
        "dataset_path": str(DATASET_PATH),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "metrics": {
            "total": total,
            "correct_answers": correct_answers,
            "correct_abstentions": correct_abst,
            "incorrect_answers": incorrect_answers,
            "incorrect_abstentions": incorrect_abst,
            "unsupported_answers": unsupported,
            "strict_accuracy": round(strict_accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "prompt_injection_safety": {"correct": pi_ok, "total": pi_total},
            "acl_safety": {"correct": acl_ok, "total": acl_total},
            "tenant_isolation": {"correct": tenant_ok, "total": tenant_total},
            "version_correctness": {"correct": version_ok, "total": len(version_rows)},
            "numeric_constraint_accuracy": {"correct": num_correct, "total": len(num_rows)},
            "date_constraint_accuracy": {"correct": date_correct, "total": len(date_rows)},
            "true_unauthorized_supporting_ids": true_unauth_total,
            "non_required_supporting_ids": non_required_total,
            "citation_validity": round(mean(cv), 6) if cv else None,
            "citation_correctness": round(mean(cc), 6) if cc else None,
            "constraint_false_positives": constraint_false_positives,
        },
        "rows": rows,
    }

    out = OUTPUT_DIR / "phase5g_g0_corrected_diagnostic_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

