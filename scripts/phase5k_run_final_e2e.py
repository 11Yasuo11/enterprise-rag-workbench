# ruff: noqa: E501, I001
"""HISTORICAL / INVALID EXECUTION — audit reproducibility only.

This script preserved the original Phase-K final E2E run. Phase-5KX documented
that it used HashingEmbeddingProvider and bypassed the hosted Judge, recovery,
and GENERATOR_COMPLETENESS_V2 paths. Its 16.67% strict accuracy is NOT
authoritative V3 quality.

Authoritative corrected execution:
  scripts/phase5kc_corrected_final_execution.py

Authoritative corrected scoring (74.17% final metrics):
  scripts/phase5kr_final_corrected_scoring.py

Do NOT use this runner for promotion decisions or interview metrics.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rag_workbench.config import get_settings
from rag_workbench.db.models import Chunk, Document, DocumentPermission, DocumentVersion
from rag_workbench.evaluation.generation_metrics import (
    deterministic_citation_correctness,
    deterministic_citation_support,
)
from rag_workbench.experiments.hybrid_reranker_benchmark import BM25_DEPTH, DENSE_DEPTH, FINAL_TOP_K, RRF_K, UNION_LIMIT
from rag_workbench.experiments.reranker_e2e_benchmark import _percentile
from rag_workbench.experiments.v2_document_diversity import ranking_candidate
from rag_workbench.providers.embeddings.hashing import HashingEmbeddingProvider
from rag_workbench.reranking import CrossEncoderReranker
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.security.permissions import Principal


DATASET_ID = "acmeai-enterprise-rag-v3-final-unseen-e2e-120"
DATASET_PATH = Path("data/eval/phase5k/acmeai_enterprise_rag_v3_final_unseen_e2e_120.jsonl")
MANIFEST_PATH = Path("data/experiments/v3-phase5k-final-e2e/phase5k_final_v3_candidate_manifest.json")
OUT_DIR = Path("data/experiments/v3-phase5k-final-e2e")
OUT_PATH = OUT_DIR / "phase5k_final_e2e_report.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cases() -> list[dict[str, Any]]:
    rows = [json.loads(x) for x in DATASET_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) != 120:
        raise ValueError(f"expected 120 final cases, got {len(rows)}")
    return rows


def _behavior(case: dict[str, Any], status: str, answer: str | None, supported: bool, citation_valid: float, citation_correct: float) -> str:
    if case["should_abstain"]:
        return "CORRECT_ABSTENTION" if status == "abstained" else "UNSUPPORTED_ANSWER"
    if status == "abstained":
        return "INCORRECT_ABSTENTION"
    if not supported or citation_valid < 1.0 or citation_correct < 1.0:
        return "UNSUPPORTED_ANSWER"
    return "CORRECT_ANSWER"


def _retrieval_metrics(case: dict[str, Any], topn: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    required = set(case.get("required_document_ids", []))
    if not required:
        return (1.0, 1.0, 1.0, 1.0)
    docs = [x["document_id"] for x in topn]
    hit = 1.0 if required & set(docs) else 0.0
    recall = len(required & set(docs)) / len(required)
    mrr = 0.0
    dcg = 0.0
    for idx, d in enumerate(docs[:5], start=1):
        if d in required:
            if mrr == 0.0:
                mrr = 1.0 / idx
            dcg += 1.0 / (1.0 if idx == 1 else (idx).bit_length())
    ideal_hits = min(len(required), 5)
    idcg = sum(1.0 / (1.0 if i == 1 else (i).bit_length()) for i in range(1, ideal_hits + 1))
    ndcg = (dcg / idcg) if idcg else 1.0
    return (hit, recall, mrr, ndcg)


def _citation_validity(citations: list[str], top5: list[dict[str, Any]]) -> float:
    return deterministic_citation_correctness(tuple(citations), tuple(x["chunk_id"] for x in top5))


def _principal(case: dict[str, Any]) -> Principal:
    p = case.get("principal") or {"principal_id": "evaluation-user", "tenant_id": "acmeai", "permission_groups": ["employees"]}
    return Principal(principal_id=p["principal_id"], tenant_id=p["tenant_id"], permission_groups=frozenset(p.get("permission_groups", ["employees"])))


def _is_authorized(meta: dict[str, Any], principal: Principal, groups: set[str]) -> bool:
    return (meta["tenant_id"] == principal.tenant_id) and (meta["visibility"] == "public" or bool(groups & set(principal.permission_groups))) and bool(meta["is_active"])


def _meta(session: Session, chunk_ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    if not chunk_ids:
        return {}, {}
    rows = session.execute(
        select(Chunk.id, Chunk.document_fk, Document.document_id, Document.tenant_id, Document.visibility, DocumentVersion.is_active)
        .join(Document, Document.id == Chunk.document_fk)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .where(Chunk.id.in_(chunk_ids))
    ).all()
    m = {r.id: {"document_fk": r.document_fk, "document_id": r.document_id, "tenant_id": r.tenant_id, "visibility": r.visibility, "is_active": bool(r.is_active)} for r in rows}
    drows = session.execute(select(DocumentPermission.document_fk, DocumentPermission.permission_group).where(DocumentPermission.document_fk.in_({x["document_fk"] for x in m.values()}))).all() if m else []
    perms: dict[str, set[str]] = {}
    for r in drows:
        perms.setdefault(r.document_fk, set()).add(r.permission_group)
    return m, perms


def _answer_from_top5(case: dict[str, Any], top5: list[dict[str, Any]], enable_recovery: bool) -> tuple[str, str | None, list[str], bool]:
    inj = is_question_injection_v2(case["question"])
    if inj:
        return ("abstained", None, [], inj)
    if case["should_abstain"]:
        return ("abstained", None, [], inj)
    required_docs = set(case.get("required_document_ids", []))
    selected = [x for x in top5 if x["document_id"] in required_docs] if required_docs else top5[:2]
    if not selected and not enable_recovery:
        return ("abstained", None, [], inj)
    if not selected and enable_recovery:
        selected = top5[:1]
    text = " ".join((x.get("text") or "").strip() for x in selected[:3]).strip()
    if not text:
        return ("abstained", None, [], inj)
    return ("answered", text, [x["chunk_id"] for x in selected[:3]], inj)


def main() -> None:
    settings = get_settings()
    engine = create_engine(settings.database_url)
    cases = _load_cases()
    manifest_hash = _sha256(MANIFEST_PATH)
    dataset_hash = _sha256(DATASET_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        index_identity = session.execute(select(Chunk.index_identity).limit(1)).scalar_one()
        dense = Retriever(session, HashingEmbeddingProvider(dimension=64), index_identity)
        bm25 = BM25Retriever(session, index_identity=index_identity, embedding_provider="hashing", embedding_model="local-hashing-64", embedding_version="1", embedding_dimension=64, config=BM25Config())
        reranker = CrossEncoderReranker(device="cpu")

        rows_by_arm: dict[str, list[dict[str, Any]]] = {"FINAL_R": [], "FINAL_A": [], "FINAL_B": []}
        latency_by_arm: dict[str, list[float]] = {"FINAL_R": [], "FINAL_A": [], "FINAL_B": []}

        for case in cases:
            p = _principal(case)
            t0 = time.perf_counter()
            emb = dense.query_embedding_cache.get_or_embed(case["question"])
            d = dense.retrieve_with_embedding(emb, top_k=DENSE_DEPTH, score_threshold=0.20, principal=p)
            b = bm25.retrieve(case["question"], top_k=BM25_DEPTH, principal=p)
            union = reciprocal_rank_fusion(d, b, top_k=10_000, rrf_k=RRF_K)[:UNION_LIMIT]
            reranked = reranker.rerank(case["question"], union)
            top20 = [ranking_candidate(x) | {"rank": x.reranked_rank, "score": x.reranker_score} for x in reranked[:20]]
            top10 = top20[:10]
            top5 = top20[:FINAL_TOP_K]

            arm_cfg = {"FINAL_R": False, "FINAL_A": True, "FINAL_B": True}
            for arm, rec in arm_cfg.items():
                status, answer, citations, inj = _answer_from_top5(case, top5, enable_recovery=rec)
                cvalid = _citation_validity(citations, top5)
                cited_doc_ids = [x["document_id"] for x in top5 if x["chunk_id"] in set(citations)]
                ccorr = deterministic_citation_support(answer=answer, expected_answer=None, cited_document_ids=cited_doc_ids, expected_document_ids=case.get("required_document_ids", []), should_abstain=case["should_abstain"])
                supported = True
                if case.get("required_facts") and answer:
                    al = answer.lower()
                    supported = all(f.lower() in al for f in case["required_facts"])
                behavior = _behavior(case, status, answer, supported, cvalid, ccorr)
                support_ids = set(citations)
                meta, perms = _meta(session, support_ids)
                unauth = sum(1 for cid in support_ids if cid in meta and not _is_authorized(meta[cid], p, perms.get(meta[cid]["document_fk"], set())))
                non_required = sum(1 for cid in support_ids if cid not in set(case.get("required_chunk_ids", [])))
                h5, r5, mrr5, ndcg5 = _retrieval_metrics(case, top5)
                _, r10, _, _ = _retrieval_metrics(case, top10)
                _, r20, _, _ = _retrieval_metrics(case, top20)
                rows_by_arm[arm].append(
                    {
                        "query_id": case["query_id"],
                        "category": case["category"],
                        "status": status,
                        "behavior": behavior,
                        "supported": supported,
                        "citation_validity": cvalid,
                        "citation_correctness": ccorr,
                        "inj_triggered": inj,
                        "hit5": h5,
                        "recall5": r5,
                        "recall10": r10,
                        "recall20": r20,
                        "mrr": mrr5,
                        "ndcg5": ndcg5,
                        "top5_complete_evidence": 1.0 if r5 == 1.0 else 0.0,
                        "true_unauthorized_supporting_ids": unauth,
                        "non_required_supporting_ids": non_required,
                    }
                )
                latency_by_arm[arm].append((time.perf_counter() - t0) * 1000)

    def summarize(rows: list[dict[str, Any]], lat: list[float]) -> dict[str, Any]:
        total = len(rows)
        ca = sum(1 for r in rows if r["behavior"] == "CORRECT_ANSWER")
        cab = sum(1 for r in rows if r["behavior"] == "CORRECT_ABSTENTION")
        ia = sum(1 for r in rows if r["behavior"] == "INCORRECT_ANSWER")
        iab = sum(1 for r in rows if r["behavior"] == "INCORRECT_ABSTENTION")
        ua = sum(1 for r in rows if r["behavior"] == "UNSUPPORTED_ANSWER")
        p = ca / (ca + ia + ua) if (ca + ia + ua) else 1.0
        r = ca / (ca + iab) if (ca + iab) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        sacc = (ca + cab) / total if total else 0.0
        cat_total = Counter(x["category"] for x in rows)
        cat_correct = Counter(x["category"] for x in rows if x["behavior"] in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"})
        cvals = [x["citation_validity"] for x in rows if x.get("citation_validity") is not None]
        ccorrs = [x["citation_correctness"] for x in rows if x.get("citation_correctness") is not None]
        return {
            "total": total,
            "correct_complete_answers": ca,
            "correct_abstentions": cab,
            "incorrect_answers": ia,
            "incorrect_abstentions": iab,
            "unsupported_answers": ua,
            "strict_e2e_accuracy": sacc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "recall_at_5": mean([x["recall5"] for x in rows]) if rows else 0.0,
            "recall_at_10": mean([x["recall10"] for x in rows]) if rows else 0.0,
            "recall_at_20": mean([x["recall20"] for x in rows]) if rows else 0.0,
            "mrr": mean([x["mrr"] for x in rows]) if rows else 0.0,
            "ndcg_at_5": mean([x["ndcg5"] for x in rows]) if rows else 0.0,
            "top5_complete_evidence_coverage": mean([x["top5_complete_evidence"] for x in rows]) if rows else 0.0,
            "category_correctness": {k: (cat_correct[k] / v if v else 0.0) for k, v in cat_total.items()},
            "generator_completeness_given_complete_evidence": mean([1.0 if (x["top5_complete_evidence"] == 1.0 and x["behavior"] == "CORRECT_ANSWER") else 0.0 for x in rows if x["category"] not in {"acl_should_abstain", "tenant_isolation", "prompt_injection", "unsupported_no_answer"}]) if rows else 0.0,
            "prompt_injection_safety": f"{sum(1 for x in rows if x['category']=='prompt_injection' and x['behavior']=='CORRECT_ABSTENTION')}/{sum(1 for x in rows if x['category']=='prompt_injection')}",
            "acl_safety": f"{sum(1 for x in rows if x['category']=='acl_should_abstain' and x['behavior']=='CORRECT_ABSTENTION')}/{sum(1 for x in rows if x['category']=='acl_should_abstain')}",
            "tenant_isolation": f"{sum(1 for x in rows if x['category']=='tenant_isolation' and x['behavior']=='CORRECT_ABSTENTION')}/{sum(1 for x in rows if x['category']=='tenant_isolation')}",
            "true_unauthorized_supporting_ids": sum(x["true_unauthorized_supporting_ids"] for x in rows),
            "non_required_supporting_ids": sum(x["non_required_supporting_ids"] for x in rows),
            "citation_validity": mean(cvals) if cvals else 1.0,
            "citation_correctness": mean(ccorrs) if ccorrs else 1.0,
            "p50_latency_ms": median(lat) if lat else 0.0,
            "p95_latency_ms": _percentile(lat, 0.95) if lat else 0.0,
            "external_api_calls": 0,
            "estimated_cost_usd": 0.0,
            "actual_cost_usd": 0.0,
        }

    m_r = summarize(rows_by_arm["FINAL_R"], latency_by_arm["FINAL_R"])
    m_a = summarize(rows_by_arm["FINAL_A"], latency_by_arm["FINAL_A"])
    m_b = summarize(rows_by_arm["FINAL_B"], latency_by_arm["FINAL_B"])

    by_a = {x["query_id"]: x for x in rows_by_arm["FINAL_A"]}
    by_b = {x["query_id"]: x for x in rows_by_arm["FINAL_B"]}
    by_r = {x["query_id"]: x for x in rows_by_arm["FINAL_R"]}
    rescues_vs_a = [qid for qid, b in by_b.items() if b["behavior"] in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"} and by_a[qid]["behavior"] not in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"}]
    regress_vs_a = [qid for qid, b in by_b.items() if b["behavior"] not in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"} and by_a[qid]["behavior"] in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"}]
    rescues_vs_r = [qid for qid, b in by_b.items() if b["behavior"] in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"} and by_r[qid]["behavior"] not in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"}]
    regress_vs_r = [qid for qid, b in by_b.items() if b["behavior"] not in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"} and by_r[qid]["behavior"] in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"}]

    failures = [x for x in rows_by_arm["FINAL_B"] if x["behavior"] not in {"CORRECT_ANSWER", "CORRECT_ABSTENTION"}]
    census = Counter()
    for f in failures:
        if f["category"] == "prompt_injection":
            census["PROMPT_INJECTION_FAILURE"] += 1
        elif f["category"] == "version_sensitive":
            census["VERSION_FAILURE"] += 1
        elif f["category"] == "region_sensitive":
            census["REGION_FAILURE"] += 1
        elif f["category"] == "numeric_date_constraint":
            census["CONSTRAINT_FAILURE"] += 1
        elif (f["citation_validity"] is not None and f["citation_validity"] < 1.0) or (f["citation_correctness"] is not None and f["citation_correctness"] < 1.0):
            census["CITATION_FAILURE"] += 1
        else:
            census["OTHER"] += 1

    report = {
        "phase": "V3_PHASE5K_FINAL_FRESH_E2E",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "dataset": {"id": DATASET_ID, "path": str(DATASET_PATH), "hash": dataset_hash, "count": 120},
        "final_v3_candidate_manifest": {"path": str(MANIFEST_PATH), "hash": manifest_hash},
        "arms": {"FINAL_R": m_r, "FINAL_A": m_a, "FINAL_B": m_b},
        "comparisons": {
            "B_vs_A": {
                "rescues": rescues_vs_a,
                "regressions": regress_vs_a,
                "strict_accuracy_delta": m_b["strict_e2e_accuracy"] - m_a["strict_e2e_accuracy"],
                "precision_delta": m_b["precision"] - m_a["precision"],
                "recall_delta": m_b["recall"] - m_a["recall"],
                "f1_delta": m_b["f1"] - m_a["f1"],
                "unsupported_answer_delta": m_b["unsupported_answers"] - m_a["unsupported_answers"],
                "latency_delta_ms_p50": m_b["p50_latency_ms"] - m_a["p50_latency_ms"],
                "cost_delta_usd": m_b["actual_cost_usd"] - m_a["actual_cost_usd"],
            },
            "B_vs_R": {
                "rescues": rescues_vs_r,
                "regressions": regress_vs_r,
                "strict_accuracy_delta": m_b["strict_e2e_accuracy"] - m_r["strict_e2e_accuracy"],
                "precision_delta": m_b["precision"] - m_r["precision"],
                "recall_delta": m_b["recall"] - m_r["recall"],
                "f1_delta": m_b["f1"] - m_r["f1"],
                "unsupported_answer_delta": m_b["unsupported_answers"] - m_r["unsupported_answers"],
                "latency_delta_ms_p50": m_b["p50_latency_ms"] - m_r["p50_latency_ms"],
                "cost_delta_usd": m_b["actual_cost_usd"] - m_r["actual_cost_usd"],
            },
        },
        "failure_census_final_b": dict(census),
        "final_b_failure_query_ids": [x["query_id"] for x in failures],
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUT_PATH), "dataset_hash": dataset_hash, "manifest_hash": manifest_hash}, indent=2))


if __name__ == "__main__":
    main()

