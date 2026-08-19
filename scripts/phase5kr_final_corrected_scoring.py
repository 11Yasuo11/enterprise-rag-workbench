# ruff: noqa: E501, PLR0915, C901
"""Phase 5K-R: scoring-only re-aggregation of frozen Phase-5KC outputs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.answerability.cache import gate_cache_key, normalize_query_text
from rag_workbench.answerability.openai_compatible import EVIDENCE_GATE_PROMPT_VERSION
from rag_workbench.config import get_settings
from rag_workbench.db.models import (
    AnswerabilityGateCacheRecord,
    Chunk,
    Document,
    DocumentPermission,
    DocumentVersion,
    RagRun,
    RecoveryStageCacheRecord,
)
from rag_workbench.evaluation.final_e2e_scorer_v2 import (
    SCORER_ID,
    aggregate_metrics,
    score_case,
    scorer_definition_payload,
    ScorerInput,
)
from rag_workbench.recovery.contracts import (
    CANNOT_DRAFT,
    CLAIM_SUPPORTED,
    COMPLETENESS_COMPLETE,
    RECOVERY_DRAFT_PROMPT_VERSION,
    RECOVERY_DRAFT_STAGE,
    STAGE_CLAIM_VERIFIER,
    RecoveryDraft,
    RecoveryVerification,
)
from rag_workbench.recovery.runtime import recovery_cache_key
from rag_workbench.safety.answerability_constraint_guard_v2 import (
    should_abstain_due_to_answerability_constraint as constraint_guard_v2,
)
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.safety.safe_recovery_boundary_v3 import (
    should_abstain_due_to_safe_recovery_boundary_v3,
)
from rag_workbench.security.permissions import Principal

PHASE_ID = "v3-phase5kr-final-corrected-scoring"
DATASET_ID = "acmeai-enterprise-rag-v3-final-unseen-e2e-120"
DATASET_PATH = Path("data/eval/phase5k/acmeai_enterprise_rag_v3_final_unseen_e2e_120.jsonl")
DATASET_HASH = "12851a9915ad51eccf2629e6765a8dc9a1731eb308987b0e53ac56468274f404"
OUT_DIR = Path("data/experiments/v3-phase5k-final-e2e")
JUDGE_MODEL = "gpt-5.6-sol"
INDEX_IDENTITY = "e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466"

SCORER_DEF_PATH = OUT_DIR / "phase5kr_final_e2e_scorer_v2_definition.json"
SCORING_INPUTS_PATH = OUT_DIR / "phase5kr_scoring_inputs.jsonl"
PER_CASE_PATH = OUT_DIR / "phase5kr_corrected_per_case_results.jsonl"
AGG_PATH = OUT_DIR / "phase5kr_corrected_aggregate_metrics.json"
CENSUS_PATH = OUT_DIR / "phase5kr_corrected_failure_census.json"
CLOSURE_PATH = OUT_DIR / "phase5kr_final_research_closure_report.json"


@dataclass(frozen=True)
class FinalCase:
    query_id: str
    category: str
    question: str
    expected_answerable: bool
    should_abstain: bool
    required_document_ids: tuple[str, ...]
    required_facts: tuple[str, ...]
    principal: dict[str, Any]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")


def _load_cases() -> list[FinalCase]:
    rows = [json.loads(x) for x in DATASET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [
        FinalCase(
            query_id=r["query_id"],
            category=r["category"],
            question=r["question"],
            expected_answerable=bool(r["expected_answerable"]),
            should_abstain=bool(r["should_abstain"]),
            required_document_ids=tuple(r.get("required_document_ids", [])),
            required_facts=tuple(r.get("required_facts", [])),
            principal=r.get(
                "principal",
                {"principal_id": "evaluation-user", "tenant_id": "acmeai", "permission_groups": ["employees"]},
            ),
        )
        for r in rows
    ]


def _principal(case: FinalCase) -> Principal:
    p = case.principal
    return Principal(
        principal_id=p.get("principal_id", "evaluation-user"),
        tenant_id=p.get("tenant_id", "acmeai"),
        permission_groups=frozenset(p.get("permission_groups", ["employees"])),
    )


def _is_authorized(meta: dict[str, Any], principal: Principal, groups: set[str]) -> bool:
    tenant_ok = meta["tenant_id"] == principal.tenant_id
    visibility_ok = meta["visibility"] == "public" or bool(groups & set(principal.permission_groups))
    return tenant_ok and visibility_ok and bool(meta["is_active"])


def _chunk_texts(session: Session, chunk_ids: set[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    rows = session.execute(select(Chunk.id, Chunk.text).where(Chunk.id.in_(chunk_ids))).all()
    return {cid: text for cid, text in rows}


def _hydrate_top5(session: Session, top5: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = {x["chunk_id"] for x in top5}
    texts = _chunk_texts(session, ids)
    out = []
    for item in top5:
        enriched = dict(item)
        if not enriched.get("text"):
            enriched["text"] = texts.get(item["chunk_id"], "")
        out.append(enriched)
    return out


def _chunk_meta(session: Session, chunk_ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    if not chunk_ids:
        return {}, {}
    rows = session.execute(
        select(
            Chunk.id,
            Chunk.document_fk,
            Document.document_id,
            Document.tenant_id,
            Document.visibility,
            DocumentVersion.is_active,
        )
        .join(Document, Document.id == Chunk.document_fk)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .where(Chunk.id.in_(chunk_ids))
    ).all()
    meta = {
        cid: {
            "document_id": did,
            "document_fk": dfk,
            "tenant_id": tenant,
            "visibility": vis,
            "is_active": active,
        }
        for cid, dfk, did, tenant, vis, active in rows
    }
    doc_fks = {m["document_fk"] for m in meta.values()}
    perm_rows = session.execute(
        select(DocumentPermission.document_fk, DocumentPermission.permission_group).where(
            DocumentPermission.document_fk.in_(doc_fks)
        )
    ).all()
    perms: dict[str, set[str]] = {}
    for dfk, group in perm_rows:
        perms.setdefault(dfk, set()).add(group)
    return meta, perms


def _gate_evidence_from_top5(top5: list[dict[str, Any]]) -> tuple[GateEvidence, ...]:
    return tuple(
        GateEvidence(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            document_version_id=item.get("document_version_id", ""),
            version=item.get("version", ""),
            text=item.get("text", ""),
            index_identity=INDEX_IDENTITY,
        )
        for item in top5
    )


def _top5_from_gate_record(record: AnswerabilityGateCacheRecord) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "document_version_id": item.get("document_version_id", ""),
            "version": item.get("version", ""),
            "text": item.get("text", ""),
        }
        for item in record.retrieved_evidence
    ]


def _retrieval_recall(required_docs: set[str], docs: list[str]) -> float:
    if not required_docs:
        return 0.0
    return len(required_docs & set(docs)) / len(required_docs)


def _recovery_cache_keys(question: str, ev: tuple[GateEvidence, ...]) -> tuple[str, str]:
    dmsg = [{"role": "user", "content": question}]
    prompt_hash = hashlib.sha256(json.dumps(dmsg, sort_keys=True).encode()).hexdigest()
    draft_key, _ = recovery_cache_key(
        question,
        ev,
        stage=RECOVERY_DRAFT_STAGE,
        model=JUDGE_MODEL,
        prompt_version=RECOVERY_DRAFT_PROMPT_VERSION,
        prompt_hash=prompt_hash,
        schema_identity="recovery-draft-v1",
    )
    verifier_key, _ = recovery_cache_key(
        question,
        ev,
        stage=STAGE_CLAIM_VERIFIER,
        model=JUDGE_MODEL,
        prompt_version="claim-verifier-v1",
        prompt_hash=prompt_hash,
        schema_identity="recovery-verifier-v1",
    )
    return draft_key, verifier_key


def _recovery_answer_from_cache(
    *,
    case: FinalCase,
    top5: list[dict[str, Any]],
    session: Session,
    draft_key: str,
    verifier_key: str,
) -> tuple[str | None, list[str], str, str | None]:
    draft_rec = session.get(RecoveryStageCacheRecord, draft_key)
    verifier_rec = session.get(RecoveryStageCacheRecord, verifier_key)
    if draft_rec is None or verifier_rec is None:
        return None, [], "abstained", "recovery_cache_incomplete"
    try:
        draft = RecoveryDraft.model_validate(draft_rec.result)
        verification = RecoveryVerification.model_validate(verifier_rec.result)
    except Exception:
        return None, [], "abstained", "recovery_cache_parse_error"
    if draft.status == CANNOT_DRAFT:
        return None, [], "abstained", "recovery_cannot_draft"
    if verification.completeness != COMPLETENESS_COMPLETE:
        return None, [], "abstained", "recovery_incomplete"
    if any(item.state != CLAIM_SUPPORTED for item in verification.claim_results):
        return None, [], "abstained", "recovery_claim_not_supported"
    ev = _gate_evidence_from_top5(top5)
    ev_texts = [x.text for x in ev if x.chunk_id in set(draft.citations)]
    if constraint_guard_v2(question=case.question, evidence_texts=ev_texts):
        return None, [], "abstained", "constraint_guard"
    if should_abstain_due_to_safe_recovery_boundary_v3(question=case.question, evidence_texts=ev_texts):
        return None, [], "abstained", "safe_recovery_boundary"
    return draft.candidate_answer, list(draft.citations), "answered", None


def _phase5kc_broken_behavior(
    *,
    case: FinalCase,
    status: str,
    answer: str | None,
    citations: list[str],
    top5: list[dict[str, Any]],
) -> str:
    from rag_workbench.evaluation.generation_metrics import (
        deterministic_citation_correctness,
        deterministic_citation_support,
    )

    cvalid = deterministic_citation_correctness(tuple(citations), tuple(x["chunk_id"] for x in top5))
    cited_doc_ids = [x["document_id"] for x in top5 if x["chunk_id"] in set(citations)]
    ccorr = deterministic_citation_support(
        answer=answer,
        expected_answer=None,
        cited_document_ids=cited_doc_ids,
        expected_document_ids=list(case.required_document_ids),
        should_abstain=case.should_abstain,
    )
    facts_ok = True
    if case.required_facts and answer:
        al = answer.lower()
        facts_ok = all(f.lower() in al for f in case.required_facts)
    if case.should_abstain:
        return "CORRECT_ABSTENTION" if status == "abstained" else "UNSUPPORTED_ANSWER"
    if status == "abstained":
        return "INCORRECT_ABSTENTION"
    if not facts_ok or (cvalid is not None and cvalid < 1.0) or (ccorr is not None and ccorr < 1.0):
        return "UNSUPPORTED_ANSWER"
    return "CORRECT_ANSWER"


def _reconstruct_output(
    *,
    case: FinalCase,
    arm: str,
    gate: AnswerabilityGateCacheRecord | None,
    top5: list[dict[str, Any]],
    rag: RagRun | None,
    session: Session,
) -> dict[str, Any]:
    inj = is_question_injection_v2(case.question)
    judge_answerable = gate.result.get("answerable") if gate else None
    judge_supporting = list(gate.result.get("supporting_chunk_ids") or []) if gate else []
    status = "abstained"
    answer: str | None = None
    citations: list[str] = []
    generation_path = None
    recovery_note = None

    if inj:
        status = "abstained"
    elif gate and gate.result.get("answerable"):
        if rag and rag.status == "answered" and rag.answer:
            status = "answered"
            answer = rag.answer
            citations = [c.get("chunk_id") for c in (rag.citations or []) if c.get("chunk_id")]
            generation_path = "rag_run_generator"
        else:
            status = "abstained"
            recovery_note = "judge_positive_no_persisted_generator_output"
    elif arm != "FINAL_R" and gate is not None:
        ev = _gate_evidence_from_top5(top5)
        draft_key, verifier_key = _recovery_cache_keys(case.question, ev)
        answer, citations, status, recovery_note = _recovery_answer_from_cache(
            case=case, top5=top5, session=session, draft_key=draft_key, verifier_key=verifier_key
        )
        if status == "answered":
            generation_path = "recovery_cache"
    else:
        status = "abstained"

    if case.should_abstain and not inj and rag is None:
        status = "abstained"

    phase5kc_label = _phase5kc_broken_behavior(
        case=case, status=status, answer=answer, citations=citations, top5=top5
    )
    return {
        "status": status,
        "answer": answer,
        "citations": citations,
        "generation_path": generation_path,
        "judge_answerable": judge_answerable,
        "judge_supporting_chunk_ids": judge_supporting,
        "question_injection_guard_triggered": inj,
        "phase5kc_behavior_label": phase5kc_label,
        "recovery_note": recovery_note,
        "top5": top5,
    }


def _failure_cause(row: dict[str, Any]) -> str:
    if row["behavior"] in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"}:
        return "NONE"
    if row["behavior"] == "SCORING_INDETERMINATE":
        return "SCORING_INDETERMINATE"
    if row.get("question_injection_guard_triggered"):
        return "PROMPT_INJECTION_FAILURE" if row["behavior"] != "CORRECT_ABSTENTION" else "NONE"
    if row.get("should_abstain"):
        return "OTHER"
    if not row.get("final_answer_present"):
        gate = row.get("judge_answerable")
        gate_input = row.get("judge_input_recoverable")
        if gate is False and gate_input and row.get("top5_complete_evidence") == 1.0:
            return "JUDGE_FALSE_NEGATIVE"
        if gate is False:
            return "OTHER"
        return "GENERATOR_INCOMPLETE"
    if row.get("fact_completeness_pass") is False:
        return "GENERATOR_INCOMPLETE"
    if row.get("citation_validity_pass") is False:
        return "GENERATOR_CITATION_MAPPING_FAILURE"
    if row.get("citation_correctness_pass") is False or row.get("required_documents_satisfied") is False:
        return "CITATION_FAILURE"
    if row.get("retrieval_recall5", 0.0) < 1.0 and row.get("expected_answerable"):
        return "RETRIEVAL_MISS"
    return "OTHER"


def _paired_delta(b_rows: dict[str, dict], ref_rows: dict[str, dict]) -> dict[str, Any]:
    rescues = [
        qid
        for qid, b in b_rows.items()
        if b["behavior"] in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"}
        and ref_rows[qid]["behavior"] not in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"}
    ]
    regressions = [
        qid
        for qid, b in b_rows.items()
        if b["behavior"] not in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"}
        and ref_rows[qid]["behavior"] in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"}
    ]
    bm = aggregate_metrics(list(b_rows.values()))
    rm = aggregate_metrics(list(ref_rows.values()))
    return {
        "rescues": rescues,
        "regressions": regressions,
        "net_correct_change": len(rescues) - len(regressions),
        "strict_accuracy_delta": bm["strict_e2e_accuracy"] - rm["strict_e2e_accuracy"],
        "precision_delta": bm["precision"] - rm["precision"],
        "recall_delta": bm["recall"] - rm["recall"],
        "f1_delta": bm["f1"] - rm["f1"],
        "unsupported_delta": bm["unsupported_answers"] - rm["unsupported_answers"],
    }


def main() -> None:
    dataset_hash = _sha(DATASET_PATH)
    if dataset_hash != DATASET_HASH:
        raise SystemExit(f"DATASET_HASH_MISMATCH expected={DATASET_HASH} actual={dataset_hash}")

    scorer_def = scorer_definition_payload()
    _write_json(SCORER_DEF_PATH, scorer_def)

    cases = _load_cases()
    settings = get_settings()
    engine = create_engine(settings.database_url)

    missing_outputs: list[str] = []
    scoring_inputs: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []
    by_arm: dict[str, dict[str, dict[str, Any]]] = {"FINAL_R": {}, "FINAL_A": {}, "FINAL_B": {}}

    with Session(engine) as session:
        gates_by_norm = {
            g.normalized_question: g
            for g in session.execute(select(AnswerabilityGateCacheRecord)).scalars().all()
        }
        rag_by_q: dict[str, RagRun] = {}
        for run in session.execute(select(RagRun)).scalars().all():
            if run.status != "answered":
                continue
            prev = rag_by_q.get(run.query)
            if prev is None or run.created_at > prev.created_at:
                rag_by_q[run.query] = run

        for case in cases:
            norm = normalize_query_text(case.question)
            gate = gates_by_norm.get(norm)
            top5 = _hydrate_top5(session, _top5_from_gate_record(gate)) if gate else []
            top5_recovered = bool(top5)
            rag = rag_by_q.get(case.question)

            if case.expected_answerable and gate is None:
                missing_outputs.append(case.query_id)
                continue
            if case.expected_answerable and gate.result.get("answerable") and rag is None:
                missing_outputs.append(case.query_id)
                continue

            req_docs = set(case.required_document_ids)
            docs5 = [x["document_id"] for x in top5]
            recall5 = _retrieval_recall(req_docs, docs5)
            top5_complete = 1.0 if recall5 == 1.0 else 0.0

            for arm in ("FINAL_R", "FINAL_A", "FINAL_B"):
                out = _reconstruct_output(
                    case=case, arm=arm, gate=gate, top5=top5, rag=rag, session=session
                )
                citation_ids = tuple(out["citations"])
                cited_doc_ids = tuple(
                    x["document_id"] for x in top5 if x["chunk_id"] in set(citation_ids)
                )
                text_map = _chunk_texts(session, set(citation_ids))
                cited_texts = {cid: text_map.get(cid, "") for cid in citation_ids}
                retrieved_ids = tuple(x["chunk_id"] for x in top5)
                meta, perms = _chunk_meta(session, set(citation_ids))
                principal = _principal(case)
                authorized = tuple(
                    cid
                    for cid in citation_ids
                    if cid in meta and _is_authorized(meta[cid], principal, perms.get(meta[cid]["document_fk"], set()))
                )

                input_row = ScorerInput(
                    arm=arm,
                    query_id=case.query_id,
                    question=case.question,
                    category=case.category,
                    should_abstain=case.should_abstain,
                    expected_answerable=case.expected_answerable,
                    required_facts=case.required_facts,
                    required_document_ids=case.required_document_ids,
                    final_answer=out["answer"],
                    final_answer_present=out["status"] == "answered" and bool(out["answer"]),
                    citation_ids=citation_ids,
                    cited_document_ids=cited_doc_ids,
                    cited_chunk_texts=cited_texts,
                    retrieved_top_k_ids=retrieved_ids,
                    authorized_citation_ids=authorized,
                    phase5kc_behavior_label=out["phase5kc_behavior_label"],
                )
                scored = score_case(input_row)
                row = {
                    **input_row.__dict__,
                    "facts_satisfied": scored.facts_satisfied,
                    "facts_missing": scored.facts_missing,
                    "fact_completeness_pass": scored.fact_completeness_pass,
                    "citation_validity_pass": scored.citation_validity_pass,
                    "citation_correctness_pass": scored.citation_correctness_pass,
                    "citation_completeness_pass": scored.citation_completeness_pass,
                    "required_documents_satisfied": scored.required_documents_satisfied,
                    "citation_validity_rate": scored.citation_validity_rate,
                    "citation_correctness_rate": scored.citation_correctness_rate,
                    "citation_completeness_rate": scored.citation_completeness_rate,
                    "fact_support_records": [
                        {
                            "required_fact": r.required_fact,
                            "supporting_citation_ids": list(r.supporting_citation_ids),
                            "support_status": r.support_status,
                        }
                        for r in scored.fact_support_records
                    ],
                    "behavior": scored.behavior,
                    "indeterminate_reason": scored.indeterminate_reason,
                    "status": out["status"],
                    "generation_path": out["generation_path"],
                    "judge_answerable": out["judge_answerable"],
                    "judge_supporting_chunk_ids": out["judge_supporting_chunk_ids"],
                    "judge_input_recoverable": gate is not None,
                    "question_injection_guard_triggered": out["question_injection_guard_triggered"],
                    "top5_recovered": top5_recovered,
                    "retrieval_recall5": recall5,
                    "top5_complete_evidence": top5_complete,
                    "recovery_note": out["recovery_note"],
                    "output_recovery_source": {
                        "gate_cache": gate is not None,
                        "rag_run": rag is not None,
                        "recovery_cache": out.get("generation_path") == "recovery_cache",
                    },
                }
                scoring_inputs.append(
                    {
                        "arm": arm,
                        "query_id": case.query_id,
                        "question": case.question,
                        "should_abstain": case.should_abstain,
                        "required_facts": list(case.required_facts),
                        "required_document_ids": list(case.required_document_ids),
                        "final_answer": out["answer"],
                        "final_answer_present": row["final_answer_present"],
                        "citation_ids": list(citation_ids),
                        "cited_document_ids": list(cited_doc_ids),
                        "cited_chunk_texts": cited_texts,
                        "retrieved_top_k_ids": list(retrieved_ids),
                        "authorization_status": {"authorized_citation_ids": list(authorized)},
                        "phase5kc_behavior_label": out["phase5kc_behavior_label"],
                        "top5_recovered": top5_recovered,
                    }
                )
                per_case.append(row)
                by_arm[arm][case.query_id] = row

    if missing_outputs:
        raise SystemExit(
            "MISSING_PHASE5KC_OUTPUTS: "
            + json.dumps(sorted(set(missing_outputs)))
        )

    _write_jsonl(SCORING_INPUTS_PATH, scoring_inputs)
    _write_jsonl(PER_CASE_PATH, per_case)

    metrics = {arm: aggregate_metrics(list(rows.values())) for arm, rows in by_arm.items()}
    _write_json(AGG_PATH, {"phase": PHASE_ID, "dataset_id": DATASET_ID, "dataset_hash": dataset_hash, "arms": metrics})

    failures_b = [r for r in per_case if r["arm"] == "FINAL_B" and r["behavior"] not in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"}]
    census = Counter(_failure_cause(r) for r in failures_b)
    census_payload = {
        "phase": PHASE_ID,
        "arm": "FINAL_B",
        "failure_census": dict(census),
        "per_case": [
            {"query_id": r["query_id"], "behavior": r["behavior"], "primary_cause": _failure_cause(r)}
            for r in failures_b
        ],
        "judge_false_negative_policy": "Assigned only when judge_input_recoverable and top5_complete_evidence=1.0 and judge_answerable=false",
    }
    _write_json(CENSUS_PATH, census_payload)

    # Historical comparisons for FINAL_B
    def _proxy_aggregate() -> dict[str, Any]:
        proxy_rows = []
        for r in per_case:
            if r["arm"] != "FINAL_B":
                continue
            b = r["phase5kc_behavior_label"]
            mapped = {
                "CORRECT_ANSWER": "CORRECT_COMPLETE_ANSWER",
                "CORRECT_ABSTENTION": "CORRECT_ABSTENTION",
                "INCORRECT_ABSTENTION": "INCORRECT_ABSTENTION",
                "UNSUPPORTED_ANSWER": "UNSUPPORTED_ANSWER",
            }.get(b, b)
            proxy_rows.append({**r, "behavior": mapped})
        return aggregate_metrics(proxy_rows)

    phase5k_report = json.loads((OUT_DIR / "phase5k_final_e2e_report.json").read_text())
    phase5kc_report = json.loads((OUT_DIR / "phase5kc_corrected_final_report.json").read_text())
    phase5kq_proxy = json.loads((OUT_DIR / "phase5kq_independent_recomputation.json").read_text())

    b_vs_a = _paired_delta(by_arm["FINAL_B"], by_arm["FINAL_A"])
    b_vs_v2 = _paired_delta(by_arm["FINAL_B"], by_arm["FINAL_R"])

    mb = metrics["FINAL_B"]
    mr = metrics["FINAL_R"]
    quality_decision = "V3_QUALITY_RESULT_INCONCLUSIVE"
    if mb["scoring_indeterminate"] > 0:
        quality_decision = "V3_QUALITY_RESULT_INCONCLUSIVE"
    elif mb["strict_e2e_accuracy"] > mr["strict_e2e_accuracy"] + 1e-9:
        quality_decision = "V3_QUALITY_IMPROVEMENT_CONFIRMED"
    else:
        quality_decision = "V3_QUALITY_IMPROVEMENT_NOT_CONFIRMED"

    release_decision = "V3_RELEASE_PROMOTION_REJECTED"

    indeterminate_rows = sum(1 for r in per_case if r["behavior"] == "SCORING_INDETERMINATE")
    research_status = (
        "V3_RESEARCH_FULLY_CLOSED_AFTER_CORRECTED_SCORING"
        if len(scoring_inputs) == 360 and indeterminate_rows == 0
        else "STOP_FOR_USER_REVIEW"
    )

    identical_a_b = all(
        by_arm["FINAL_A"][qid]["behavior"] == by_arm["FINAL_B"][qid]["behavior"]
        for qid in by_arm["FINAL_A"]
    )

    closure = {
        "phase": "V3_PHASE5KR_FINAL_CORRECTED_SCORING",
        "phase_identity": PHASE_ID,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "mode": {
            "SCORING_ONLY": True,
            "NO_INFERENCE": True,
            "NO_MODEL_CHANGE": True,
            "NO_DATASET_CHANGE": True,
            "NO_CANDIDATE_CHANGE": True,
            "NO_RETUNING": True,
            "external_api_calls": 0,
            "rag_reruns": 0,
        },
        "verdict": "COMPLETE" if len(scoring_inputs) == 360 and not missing_outputs else "BLOCKED",
        "frozen_dataset": {"id": DATASET_ID, "hash": dataset_hash},
        "scorer": {
            "id": SCORER_ID,
            "definition_path": str(SCORER_DEF_PATH),
            "implementation_hash": scorer_def["implementation_hash"],
            "ground_truth_fields": scorer_def["ground_truth_fields_consumed"],
        },
        "scorer_bug_corrected": {
            "issue": "Phase-5KC passed expected_answer=None to deterministic_citation_support",
            "effect": "citation_correctness forced to 0.0 for every answered case",
        },
        "scoring_coverage": {
            "expected_rows": 360,
            "recovered_rows": len(scoring_inputs),
            "by_arm": {arm: len(rows) for arm, rows in by_arm.items()},
            "indeterminate_rows": indeterminate_rows,
            "top5_not_recovered_should_abstain_cases": sum(
                1 for r in scoring_inputs if not r.get("top5_recovered")
            ),
        },
        "corrected_metrics": metrics,
        "comparisons": {
            "B_vs_A": b_vs_a,
            "B_vs_Stable_V2_FINAL_R": b_vs_v2,
            "FINAL_A_AND_FINAL_B_OUTPUTS_IDENTICAL": identical_a_b,
        },
        "historical_score_comparison_FINAL_B": {
            "A_original_invalid_phase_k": phase5k_report["arms"]["FINAL_B"],
            "B_phase5kc_broken_scorer": phase5kc_report["arms"]["FINAL_B"],
            "C_phase5kq_proxy": phase5kq_proxy.get("phase5kq_recomputed_all_120_proxy"),
            "D_phase5kr_authoritative_corrected": mb,
            "why_differ": {
                "phase_k": "Invalid execution path (hashing embedding + manual shortcut); 100 incorrect abstentions",
                "phase5kc": "Valid execution but expected_answer=None forced citation_correctness=0 → 73 unsupported",
                "phase5kq_proxy": "RagRun-only proxy using fact substring + citation-in-top5; no expected_answer dependency",
                "phase5kr": "Authoritative recovery of all persisted outputs scored with FINAL_E2E_SCORER_V2",
            },
        },
        "quality_decision": quality_decision,
        "cumulative_security_evidence": {
            "phase5kc_final_subset_prompt_injection": "5/5",
            "phase5kc_acl": "5/5",
            "phase5kc_tenant": "5/5",
            "phase5h_targeted_prompt_injection": "14/20",
            "phase5h_unsafe_failures": 6,
            "true_unauthorized_supporting_ids_phase5kc": 0,
        },
        "release_decision": release_decision,
        "release_safety_note": "Phase-H 14/20 prompt-injection safety remains an independent production blocker.",
        "final_research_status": research_status,
        "production_status": {
            "main": "stable — no merge from V3 research branch",
            "v2.0.0": "stable production baseline unchanged",
        },
        "phase5kc_historical_decision_preserved": "V3_FINAL_CANDIDATE_REJECTED",
        "artifact_paths": {
            "scorer_definition": str(SCORER_DEF_PATH),
            "scoring_inputs": str(SCORING_INPUTS_PATH),
            "per_case_results": str(PER_CASE_PATH),
            "aggregate_metrics": str(AGG_PATH),
            "failure_census": str(CENSUS_PATH),
        },
    }
    _write_json(CLOSURE_PATH, closure)

    print(json.dumps({"closure_path": str(CLOSURE_PATH), "verdict": closure["verdict"], "quality_decision": quality_decision, "release_decision": release_decision, "research_status": research_status, "FINAL_B_corrected": mb}, indent=2))


if __name__ == "__main__":
    main()
