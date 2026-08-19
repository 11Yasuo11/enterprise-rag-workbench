# ruff: noqa: E501
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_workbench.answerability.base import AnswerabilityResult
from rag_workbench.answerability.validation import validate_gate_result_with_error
from rag_workbench.config import get_settings
from rag_workbench.db.models import Chunk, Document, DocumentPermission, DocumentVersion
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    DENSE_DEPTH,
    FINAL_TOP_K,
    RRF_K,
    UNION_LIMIT,
)
from rag_workbench.experiments.reranker_e2e_benchmark import SEMANTIC_INDEX_IDENTITY
from rag_workbench.experiments.v2_document_diversity import ranking_candidate
from rag_workbench.experiments.v3_generate_verify import V3GenerateVerifyBenchmark
from rag_workbench.providers.llm.extractive import EXTRACTIVE_V2_MODEL
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.safety.answerability_constraint_guard_v1 import (
    should_abstain_due_to_answerability_constraint as constraint_v1,
)
from rag_workbench.safety.answerability_constraint_guard_v2 import (
    should_abstain_due_to_answerability_constraint as constraint_v2,
)
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.safety.safe_recovery_boundary_v3 import (
    should_abstain_due_to_safe_recovery_boundary_v3,
)
from rag_workbench.security.permissions import Principal

DATASET_PATH = Path("data/eval/phase5e/v3_phase5e_constraint_safety_holdout_48_cases.jsonl")
OUT_DIR = Path("data/experiments/v3-phase5f-safety-evaluation-audit")
OUT_PATH = OUT_DIR / "phase5f_case_replay_audit.json"
REPORT_PATH = Path("data/experiments/v3-phase5e-constraint-semantics-holdout-48-run/phase5e_safety_ab_report.json")

TARGET_IDS = {
    "p5e_inj_new_02",
    "p5e_acl_01",
    "p5e_acl_02",
    "p5e_acl_03",
    "p5e_acl_04",
    "p5e_date_pos_01",
    "p5e_date_pos_03",
    "p5e_date_pos_04",
    "p5e_version_01",
    "p5e_version_02",
    "p5e_version_03",
}


@dataclass
class Case:
    query_id: str
    question: str
    expected_answerable: bool
    should_abstain: bool
    expected_answer: str | None
    required_facts: list[str]
    required_chunk_ids: list[str]
    required_version_ids: dict[str, str]
    forbidden_document_ids: list[str]
    category: str
    principal: dict[str, Any]


def _load_cases() -> dict[str, Case]:
    out: dict[str, Case] = {}
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["query_id"] not in TARGET_IDS:
            continue
        out[d["query_id"]] = Case(
            query_id=d["query_id"],
            question=d["question"],
            expected_answerable=d["expected_answerable"],
            should_abstain=d["should_abstain"],
            expected_answer=d.get("expected_answer"),
            required_facts=d.get("required_facts", []),
            required_chunk_ids=d.get("required_chunk_ids", []),
            required_version_ids=d.get("required_version_ids", {}),
            forbidden_document_ids=d.get("forbidden_document_ids", []),
            category=d["category"],
            principal=d.get("principal", {"principal_id": "evaluation-user", "tenant_id": "acmeai", "permission_groups": ["employees"]}),
        )
    return out


def _load_report_rows() -> dict[str, dict[str, Any]]:
    d = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    rows = d["rows_by_arm"]["E2"]
    return {r["case_id"]: r for r in rows}


def _load_phase5c_experiment() -> Any:
    path = Path("scripts/phase5c_experiment.py")
    spec = importlib.util.spec_from_file_location("phase5c_experiment", str(path))
    if not spec or not spec.loader:
        raise RuntimeError("Cannot import phase5c_experiment")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _chunk_meta(session: Session, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not chunk_ids:
        return {}
    rows = session.execute(
        select(
            Chunk.id,
            Chunk.document_fk,
            Chunk.document_version_id,
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
            "document_version_id": r.document_version_id,
            "version": r.version,
            "is_active": bool(r.is_active),
        }
    return out


def _permission_groups_for_documents(session: Session, document_fks: list[str]) -> dict[str, list[str]]:
    if not document_fks:
        return {}
    rows = session.execute(
        select(DocumentPermission.document_fk, DocumentPermission.permission_group).where(
            DocumentPermission.document_fk.in_(document_fks)
        )
    ).all()
    out: dict[str, list[str]] = {}
    for r in rows:
        out.setdefault(r.document_fk, []).append(r.permission_group)
    return out


def _is_authorized(meta: dict[str, Any], principal: Principal, doc_perm_groups: list[str]) -> bool:
    tenant_ok = meta["tenant_id"] == principal.tenant_id
    visibility_ok = meta["visibility"] == "public" or bool(set(doc_perm_groups) & set(principal.permission_groups))
    active_ok = bool(meta["is_active"])
    return tenant_ok and visibility_ok and active_ok


def _strict_behavior(case: Case, status: str, answer: str | None) -> str:
    if case.should_abstain:
        return "CORRECT_ABSTENTION" if status == "abstained" else "UNSUPPORTED_ANSWER"
    if status == "abstained":
        return "INCORRECT_ABSTENTION"
    if case.expected_answerable and answer:
        ans = answer.lower()
        return "CORRECT_ANSWER" if all(f.lower() in ans for f in case.required_facts) else "INCORRECT_ANSWER"
    return "INCORRECT_ANSWER"


def _f(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def main() -> None:
    cases = _load_cases()
    report_rows = _load_report_rows()
    settings = get_settings()
    engine = create_engine(settings.database_url)
    phase5c = _load_phase5c_experiment()

    with Session(engine) as session:
        benchmark = V3GenerateVerifyBenchmark(session, settings)
        reranker = benchmark.reranker_factory()
        gate = benchmark._gate(max(1, len(cases)))

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

        out_rows: dict[str, Any] = {}

        for case_id in sorted(cases):
            case = cases[case_id]
            principal = Principal(
                principal_id=case.principal.get("principal_id", "evaluation-user"),
                tenant_id=case.principal.get("tenant_id", "acmeai"),
                permission_groups=frozenset(case.principal.get("permission_groups", ["employees"])),
            )
            print(f"Replaying {case_id}")

            emb = dense.query_embedding_cache.get_or_embed(case.question)
            dense_candidates = dense.retrieve_with_embedding(
                emb, top_k=DENSE_DEPTH, score_threshold=0.28, principal=principal
            )
            bm25_candidates = bm25.retrieve(case.question, top_k=BM25_DEPTH, principal=principal)
            union_all = reciprocal_rank_fusion(dense_candidates, bm25_candidates, top_k=10_000, rrf_k=RRF_K)
            union = union_all[:UNION_LIMIT]
            reranked = reranker.rerank(case.question, union)
            top5 = [ranking_candidate(it) | {"rank": it.reranked_rank, "score": it.reranker_score} for it in reranked[:FINAL_TOP_K]]

            from rag_workbench.answerability.base import GateEvidence

            gate_evidence = tuple(
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
            try:
                proposed = gate.evaluate(case.question, gate_evidence)
                validated = validate_gate_result_with_error(proposed, gate_evidence, session=session, principal=principal)
                gate_result = validated.result
            except Exception:
                gate_result = AnswerabilityResult.fail_closed()

            # Replay E2 logic only (same as final candidate composition in Phase 5E)
            inj = is_question_injection_v2(case.question)
            status = "abstained"
            answer = None
            citations: list[str] = []
            primary_answerable = bool(gate_result.answerable)
            recovery_triggered = False
            recovery_out: dict[str, Any] | None = None
            guard_v1 = None
            guard_v2 = None
            guard_v3 = None

            if inj:
                status = "abstained"
            elif primary_answerable:
                evidence_texts = [item["text"] for item in top5 if item["chunk_id"] in gate_result.supporting_chunk_ids]
                guard_v1 = constraint_v1(question=case.question, evidence_texts=evidence_texts)
                guard_v2 = constraint_v2(question=case.question, evidence_texts=evidence_texts)
                guard_v3 = should_abstain_due_to_safe_recovery_boundary_v3(question=case.question, evidence_texts=evidence_texts)
                if guard_v3:
                    status = "abstained"
                else:
                    gen = phase5c.generate_answer(case, top5, gate_result, session, provider_revision=EXTRACTIVE_V2_MODEL)
                    status = gen["status"]
                    answer = gen["answer"]
                    citations = list(gen["citations"])
            else:
                recovery_triggered = True
                recovery_out = {"skipped_replay": True, "reason": "phase5f_diagnostic_reuses_frozen_phase5e_row_for_recovery_stage"}
                # Use frozen E2 row for downstream final-state audit fields when recovery path is needed.
                frozen = report_rows.get(case_id, {})
                status = frozen.get("status", "abstained")
                answer = frozen.get("answer")
                citations = list(frozen.get("citations") or [])

            behavior = _strict_behavior(case, status, answer)

            all_chunk_ids = sorted(
                {
                    *(_f(item, "chunk_id") for item in dense_candidates),
                    *(_f(item, "chunk_id") for item in bm25_candidates),
                    *(_f(item, "chunk_id") for item in union),
                    *(item["chunk_id"] for item in top5),
                    *(gate_result.supporting_chunk_ids or ()),
                    *citations,
                    *([] if recovery_out is None else recovery_out.get("supporting_chunk_ids", [])),
                }
            )
            meta = _chunk_meta(session, all_chunk_ids)
            perm = _permission_groups_for_documents(session, [m["document_fk"] for m in meta.values()])

            def auth_row(
                cid: str,
                *,
                meta_map: dict[str, dict[str, Any]] = meta,
                perm_map: dict[str, list[str]] = perm,
                principal_obj: Principal = principal,
            ) -> dict[str, Any]:
                m = meta_map.get(cid)
                if not m:
                    return {"chunk_id": cid, "missing": True}
                groups = perm_map.get(m["document_fk"], [])
                return {
                    "chunk_id": cid,
                    **m,
                    "document_permission_groups": groups,
                    "authorized_for_principal": _is_authorized(m, principal_obj, groups),
                }

            out_rows[case_id] = {
                "case": {
                    "query_id": case.query_id,
                    "category": case.category,
                    "question": case.question,
                    "expected_answerable": case.expected_answerable,
                    "should_abstain": case.should_abstain,
                    "expected_answer": case.expected_answer,
                    "required_facts": case.required_facts,
                    "required_chunk_ids": case.required_chunk_ids,
                    "required_version_ids": case.required_version_ids,
                    "forbidden_document_ids": case.forbidden_document_ids,
                    "principal": case.principal,
                },
                "retrieval": {
                    "dense_top": [{"chunk_id": _f(x, "chunk_id"), "document_id": _f(x, "document_id"), "version": _f(x, "version"), "score": _f(x, "score")} for x in dense_candidates[:10]],
                    "bm25_top": [{"chunk_id": _f(x, "chunk_id"), "document_id": _f(x, "document_id"), "version": _f(x, "version"), "score": _f(x, "score")} for x in bm25_candidates[:10]],
                    "rrf_top": [{"chunk_id": _f(x, "chunk_id"), "document_id": _f(x, "document_id"), "version": _f(x, "version"), "score": _f(x, "score")} for x in union[:10]],
                    "top5": [{"chunk_id": x["chunk_id"], "document_id": x["document_id"], "version": x.get("version"), "score": x["score"]} for x in top5],
                },
                "gate": {
                    "answerable": bool(gate_result.answerable),
                    "supporting_chunk_ids": list(gate_result.supporting_chunk_ids),
                    "reason": getattr(gate_result, "reason", None),
                },
                "guards": {
                    "question_injection_triggered": inj,
                    "constraint_v1_triggered": guard_v1,
                    "constraint_v2_triggered": guard_v2,
                    "safe_recovery_boundary_v3_triggered": guard_v3,
                },
                "recovery": recovery_out,
                "final": {
                    "status": status,
                    "behavior": behavior,
                    "answer": answer,
                    "citations": citations,
                    "recovery_triggered": recovery_triggered,
                },
                "frozen_phase5e_e2_row": report_rows.get(case_id),
                "authorization_meta": {
                    "top5": [auth_row(x["chunk_id"]) for x in top5],
                    "gate_supporting": [auth_row(cid) for cid in gate_result.supporting_chunk_ids],
                    "final_citations": [auth_row(cid) for cid in citations],
                    "required_chunks": [auth_row(cid) for cid in case.required_chunk_ids],
                },
            }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

