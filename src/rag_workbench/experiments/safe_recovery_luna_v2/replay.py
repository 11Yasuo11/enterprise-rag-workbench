# ruff: noqa: E501
"""Diagnostic replay of historical OTHER=23 through the frozen V2 retrieval+judge path."""

from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.db.models import Chunk, Document, DocumentVersion
from rag_workbench.security.permissions import Principal, apply_document_acl

ALLOWED_CAUSES = (
    "INGESTION_MISSING_EVIDENCE",
    "ACL_FILTER_ERROR",
    "TENANT_FILTER_ERROR",
    "VERSION_FILTER_FAILURE",
    "REGION_FILTER_FAILURE",
    "RETRIEVAL_MISSING_REQUIRED_EVIDENCE",
    "RANKING_TOPK_INCOMPLETE",
    "JUDGE_FALSE_NEGATIVE",
    "LUNA_VERIFIER_FALSE_NEGATIVE",
    "SOL_JUDGE_FALSE_NEGATIVE",
    "GENERATOR_INCOMPLETE",
    "CITATION_FAILURE",
    "CONTRADICTORY_EVIDENCE",
    "DATASET_OR_SCORER_ISSUE",
    "MULTI_HOP_EVIDENCE_FAILURE",
    "UNKNOWN",
)


def _fact_chunks(
    session: Session, document_ids: tuple[str, ...], facts: tuple[str, ...]
) -> list[dict[str, Any]]:
    if not document_ids:
        return []
    rows = session.execute(
        select(Chunk, Document, DocumentVersion)
        .join(Document, Chunk.document_fk == Document.id)
        .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
        .where(Document.document_id.in_(document_ids))
    ).all()
    hits: list[dict[str, Any]] = []
    for chunk, document, version in rows:
        covered = [f for f in facts if f.casefold() in chunk.text.casefold()]
        if facts and not covered:
            continue
        hits.append(
            {
                "chunk_id": chunk.id,
                "document_id": document.document_id,
                "tenant_id": document.tenant_id,
                "visibility": document.visibility,
                "is_active": bool(version.is_active),
                "version": version.version,
                "covered_facts": covered,
                "region": (chunk.metadata_ or {}).get("region")
                or (document.metadata_ or {}).get("region"),
            }
        )
    return hits


def classify_other_case(
    *,
    session: Session,
    case: Any,
    principal: Principal,
    trace: dict[str, Any],
    frozen_row: dict[str, Any] | None,
) -> dict[str, Any]:
    facts = tuple(case.required_facts)
    docs = tuple(case.required_document_ids)
    corpus_hits = _fact_chunks(session, docs, facts)
    evidence_exists = bool(corpus_hits) if facts or docs else True
    active_hits = [h for h in corpus_hits if h["is_active"]]
    tenant_hits = [h for h in corpus_hits if h["tenant_id"] == principal.tenant_id]
    ids = [h["chunk_id"] for h in corpus_hits]
    authorized: set[str] = set()
    if ids:
        stmt = (
            select(Chunk.id)
            .join(Document, Chunk.document_fk == Document.id)
            .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
            .where(Chunk.id.in_(ids), DocumentVersion.is_active.is_(True))
        )
        authorized = set(session.scalars(apply_document_acl(stmt, principal)).all())
    acl_hits = [h for h in active_hits if h["chunk_id"] in authorized]
    union_docs = {item["document_id"] for item in trace.get("union", [])}
    union_ids = {item["chunk_id"] for item in trace.get("union", [])}
    top5 = trace.get("top5") or []
    top5_docs = {x["document_id"] for x in top5}
    top5_ids = {x["chunk_id"] for x in top5}
    required_doc_set = set(docs)
    pool_complete = required_doc_set <= union_docs if required_doc_set else True
    topk_complete = required_doc_set <= top5_docs if required_doc_set else True
    fact_in_top5 = (
        all(any(f.casefold() in (x.get("text") or "").casefold() for x in top5) for f in facts)
        if facts
        else True
    )
    judge_answerable = None if frozen_row is None else frozen_row.get("judge_answerable")
    ranks: list[dict[str, Any]] = []
    for hit in acl_hits:
        union_item = next((u for u in trace.get("union", []) if u["chunk_id"] == hit["chunk_id"]), None)
        ce_rank = next(
            (i + 1 for i, x in enumerate(trace.get("top20") or []) if x["chunk_id"] == hit["chunk_id"]),
            None,
        )
        ranks.append(
            {
                "chunk_id": hit["chunk_id"],
                "document_id": hit["document_id"],
                "dense_rank": None if union_item is None else union_item.get("dense_rank"),
                "bm25_rank": None if union_item is None else union_item.get("bm25_rank"),
                "rrf_rank": None if union_item is None else union_item.get("rrf_rank"),
                "ce_rank": ce_rank,
                "in_pool": hit["chunk_id"] in union_ids,
                "in_top5": hit["chunk_id"] in top5_ids,
            }
        )

    passed_acl = bool(acl_hits) if evidence_exists else False
    passed_tenant = bool(tenant_hits) if evidence_exists else False
    passed_version = bool(active_hits) if evidence_exists else False
    passed_region = True
    if case.category == "region_sensitive" and any(h.get("region") for h in corpus_hits):
        passed_region = any(h.get("region") for h in acl_hits)

    cause = "UNKNOWN"
    evidence = ""
    if not evidence_exists:
        cause = "INGESTION_MISSING_EVIDENCE"
        evidence = "required document/fact not found in corpus"
    elif evidence_exists and not passed_tenant:
        cause = "TENANT_FILTER_ERROR"
        evidence = "required evidence exists but tenant filter excluded it"
    elif evidence_exists and passed_tenant and not passed_acl:
        cause = "ACL_FILTER_ERROR"
        evidence = "required evidence exists for tenant but ACL excluded it"
    elif evidence_exists and not passed_version:
        cause = "VERSION_FILTER_FAILURE"
        evidence = "required fact exists only on inactive versions"
    elif not passed_region:
        cause = "REGION_FILTER_FAILURE"
        evidence = "region metadata excluded required evidence"
    elif not pool_complete:
        cause = "RETRIEVAL_MISSING_REQUIRED_EVIDENCE"
        evidence = f"required docs {sorted(required_doc_set - union_docs)} missing from RRF union"
    elif pool_complete and not topk_complete:
        cause = "RANKING_TOPK_INCOMPLETE"
        evidence = f"required docs {sorted(required_doc_set - top5_docs)} missing from Top-5 after CE"
    elif topk_complete and not fact_in_top5:
        cause = "RANKING_TOPK_INCOMPLETE"
        evidence = "required documents in Top-5 but required fact spans missing"
    elif topk_complete and fact_in_top5 and judge_answerable is False:
        cause = "SOL_JUDGE_FALSE_NEGATIVE"
        evidence = "Top-5 contains required docs and facts; frozen Sol judge returned answerable=false"
    elif judge_answerable is True and frozen_row and frozen_row.get("behavior") == "INCORRECT_ABSTENTION":
        cause = "GENERATOR_INCOMPLETE"
        evidence = "judge GO but no complete extractive answer"
    elif len(docs) >= 3 and not topk_complete:
        cause = "MULTI_HOP_EVIDENCE_FAILURE"
        evidence = "three-document requirement not fully present in Top-5"
    else:
        cause = "UNKNOWN"
        evidence = (
            f"pool_complete={pool_complete} topk_complete={topk_complete} "
            f"fact_in_top5={fact_in_top5} judge_answerable={judge_answerable}"
        )

    return {
        "query_id": case.query_id,
        "category": case.category,
        "required_fact_ids": list(facts),
        "required_chunk_ids": [h["chunk_id"] for h in acl_hits],
        "evidence_exists_in_corpus": evidence_exists,
        "passed_acl": passed_acl,
        "passed_tenant": passed_tenant,
        "passed_version": passed_version,
        "passed_region": passed_region,
        "candidate_pool_complete": pool_complete,
        "top_k_complete": topk_complete,
        "judge_input_complete": bool(topk_complete and fact_in_top5),
        "judge_decision": (
            "NO" if judge_answerable is False else ("YES" if judge_answerable is True else "UNKNOWN")
        ),
        "generator_status": (frozen_row or {}).get("behavior"),
        "final_root_cause": cause if cause in ALLOWED_CAUSES else "UNKNOWN",
        "evidence": evidence,
        "ranks": ranks,
        "top5_document_ids": [x["document_id"] for x in top5],
        "diagnostic_only": True,
    }


def summarize_causes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(r["final_root_cause"] for r in rows)
    total = max(len(rows), 1)
    return [
        {"root_cause": cause, "count": counts.get(cause, 0), "pct_of_other": round(100 * counts.get(cause, 0) / total, 2)}
        for cause in (*ALLOWED_CAUSES,)
        if counts.get(cause)
    ]
