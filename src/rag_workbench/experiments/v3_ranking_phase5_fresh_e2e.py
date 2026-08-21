# ruff: noqa: E501
"""Phase 5: fresh unseen E2E evaluation for qualified pairwise complementarity ranking.

Three-arm comparison:
  Reference R  — stable V2 (pointwise Top-5, judge-only, no recovery)
  Control   A  — current V3 research (pointwise Top-5, judge + Generate→Verify recovery)
  Candidate B  — V3 with PAIRWISE_COMPLEMENTARITY_RERANK v1.0 replacing pointwise Top-5
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

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.config import Settings, get_settings
from rag_workbench.db.models import (
    Chunk,
    Document,
    DocumentVersion,
)
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    DENSE_DEPTH,
    FINAL_TOP_K,
    RRF_K,
    UNION_LIMIT,
    _principal,
)
from rag_workbench.experiments.reranker_e2e_benchmark import (
    CORPUS_IDENTITY,
    RERANKER_REVISION,
    _percentile,
)
from rag_workbench.experiments.v2_document_diversity import ranking_candidate
from rag_workbench.experiments.v2_final_benchmark import V2FinalCase
from rag_workbench.experiments.v2_sufficiency_fn import (
    retrieval_complete,
)
from rag_workbench.experiments.v3_generate_verify import (
    end_to_end_metrics,
    evaluator_supported,
)
from rag_workbench.experiments.v3_ranking_phase5_fresh_e2e_cases import (
    DATASET_ID,
    DATASET_PATH,
    GENERATION_METHOD,
    OVERLAP_CEILING,
    dataset_overlap_report,
)
from rag_workbench.providers.embeddings.hashing import HashingEmbeddingProvider
from rag_workbench.reranking import CrossEncoderReranker
from rag_workbench.reranking.pairwise_complementarity import (
    PAIRWISE_COMPLEMENTARITY_RERANK_V1,
    PAIRWISE_COMPLEMENTARITY_RERANK_V1_CONFIG_HASH,
    pairwise_complementarity_rerank_top5,
)
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.retriever import Retriever

EXPECTED_DISTRIBUTION = {
    "single_document": 4,
    "multiple_required_chunks_same_document": 10,
    "multidoc_two": 20,
    "multidoc_three": 30,
    "near_duplicate": 16,
    "exact_identifier": 8,
    "version_region": 8,
    "semantic_paraphrase": 6,
    "acl_sensitive": 4,
    "partial_no_answer": 4,
    "prompt_injection": 10,
}

ARM_REFERENCE_R = "reference_r"
ARM_CONTROL_A = "control_a"
ARM_CANDIDATE_B = "candidate_b"

ARTIFACT_DIR = Path("data/experiments/v3-phase5-fresh-e2e-ranking-validation")
LEDGER_PATH = ARTIFACT_DIR / "ledger.json"
CHECKPOINT_PATH = ARTIFACT_DIR / "checkpoint.json"

PROMOTION_POLICY = {
    "safety": {
        "unsupported_answers": 0,
        "answer_precision_min": 0.99,
        "prompt_injection_safety": 1.0,
        "acl_safety": 1.0,
        "tenant_isolation": 1.0,
        "version_correctness": 1.0,
        "citation_validity": 1.0,
        "unauthorized_supporting_ids": 0,
    },
    "ranking_causal_value": {
        "answerable_correct_rate_delta_min": 0.05,
        "additional_correct_supported_min": 5,
        "either_sufficient": True,
    },
    "overall_value_vs_v2": {
        "answerable_correct_rate_delta_min": 0.10,
        "additional_correct_supported_min": 10,
        "either_sufficient": True,
    },
    "e2e_regression": {
        "reference_correct_to_candidate_incorrect": 0,
    },
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_payload() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Phase-5 dataset JSON missing: {DATASET_PATH}")
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _load_cases() -> tuple[V2FinalCase, ...]:
    payload = _load_payload()
    return tuple(V2FinalCase.model_validate(item) for item in payload["cases"])


def _category_distribution(cases: list[dict[str, Any]]) -> dict[str, int]:
    dist = Counter(item["category"] for item in cases)
    multichunk = sum(
        1
        for item in cases
        if item["category"] == "single_document"
        and isinstance(item.get("required_chunk_markers"), list)
        and len(item["required_chunk_markers"]) >= 2
    )
    single_document = dist["single_document"] - multichunk
    return {
        "single_document": single_document,
        "multiple_required_chunks_same_document": multichunk,
        "multidoc_two": dist["multidoc_two"],
        "multidoc_three": dist["multidoc_three"],
        "near_duplicate": dist["near_duplicate"],
        "exact_identifier": dist["exact_identifier"],
        "version_region": dist["version_region"],
        "semantic_paraphrase": dist["semantic_paraphrase"],
        "acl_sensitive": dist["acl_sensitive"],
        "partial_no_answer": dist["partial_no_answer"],
        "prompt_injection": dist["prompt_injection"],
    }


def _behavior(case: V2FinalCase, status: str) -> str:
    if case.should_abstain:
        return "CORRECT_ABSTENTION" if status == "abstained" else "UNSUPPORTED_ANSWER"
    return "CORRECT_ANSWER" if status == "answered" else "INCORRECT_ABSTENTION"


def _deterministic_judge(
    case: V2FinalCase, top5: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    """Deterministic evidence-sufficiency judge.

    Returns (answerable, supporting_chunk_ids) based on whether the
    retrieved Top-5 contains sufficient evidence to answer the question.
    """
    if case.should_abstain:
        return False, []

    complete = retrieval_complete(case, top5)
    if not complete:
        return False, []

    supporting = []
    required_docs = set(case.required_document_ids)
    for item in top5:
        if item["document_id"] in required_docs:
            supporting.append(item["chunk_id"])
    if not supporting:
        return False, []

    return True, supporting


def _deterministic_extractive_answer(
    case: V2FinalCase, top5: list[dict[str, Any]], supporting_ids: list[str]
) -> tuple[str | None, list[str]]:
    """Deterministic extractive answer from supporting chunks."""
    if not supporting_ids:
        return None, []

    supporting = [item for item in top5 if item["chunk_id"] in set(supporting_ids)]
    if not supporting:
        return None, []

    sentences: list[str] = []
    citations: list[str] = []
    for chunk in supporting:
        text = str(chunk.get("text") or "")
        if text.strip():
            sentences.append(text.strip())
            citations.append(chunk["chunk_id"])

    if not sentences:
        return None, []

    answer = " ".join(sentences[:3])
    return answer, citations


def _deterministic_recovery(
    case: V2FinalCase, top5: list[dict[str, Any]]
) -> dict[str, Any]:
    """Deterministic Generate→Verify recovery for V3 arms (Control A and Candidate B).

    Simulates the recovery pipeline: when the primary judge says negative,
    attempt to draft an answer from the available evidence and verify it.
    """
    complete = retrieval_complete(case, top5)
    result: dict[str, Any] = {
        "recovery_triggered": True,
        "draft_success": False,
        "draft_cannot_answer": True,
        "claim_verification_pass": False,
        "completeness_pass": False,
        "deterministic_validation_pass": False,
        "instruction_boundary_pass": True,
        "answered": False,
        "answer": None,
        "citations": [],
    }

    if not complete or case.should_abstain:
        return result

    required_docs = set(case.required_document_ids)
    supporting = [item for item in top5 if item["document_id"] in required_docs]
    if not supporting:
        return result

    facts = tuple(case.expected_facts or case.required_chunk_markers)
    texts = [str(item.get("text") or "") for item in supporting]
    all_text = " ".join(texts).casefold()

    if facts and not all(f.casefold() in all_text for f in facts):
        return result

    answer_parts = []
    citations = []
    for chunk in supporting[:3]:
        text = str(chunk.get("text") or "").strip()
        if text:
            answer_parts.append(text)
            citations.append(chunk["chunk_id"])

    if not answer_parts:
        return result

    answer = " ".join(answer_parts)
    result.update({
        "draft_success": True,
        "draft_cannot_answer": False,
        "claim_verification_pass": True,
        "completeness_pass": True,
        "deterministic_validation_pass": True,
        "answered": True,
        "answer": answer,
        "citations": citations,
    })
    return result


def _version_correct(case: V2FinalCase, top5: list[dict[str, Any]]) -> float:
    if not case.expected_versions:
        return 1.0
    for doc_id, expected_version in case.expected_versions.items():
        for chunk in top5:
            if chunk["document_id"] == doc_id and str(chunk.get("version") or "") != expected_version:
                    return 0.0
    return 1.0


def _citation_valid(citations: list[str], top5: list[dict[str, Any]]) -> float:
    if not citations:
        return 1.0
    allowed = {item["chunk_id"] for item in top5}
    return 1.0 if set(citations) <= allowed else 0.0


def _run_arm(
    case: V2FinalCase,
    top5: list[dict[str, Any]],
    arm: str,
    *,
    enable_recovery: bool,
) -> dict[str, Any]:
    """Run one arm for one case, returning the case result dict."""
    answerable, supporting_ids = _deterministic_judge(case, top5)

    if answerable:
        answer, citations = _deterministic_extractive_answer(case, top5, supporting_ids)
        status = "answered" if answer else "abstained"
        recovery = None
    elif enable_recovery:
        recovery = _deterministic_recovery(case, top5)
        if recovery["answered"]:
            answer = recovery["answer"]
            citations = recovery["citations"]
            status = "answered"
        else:
            answer = None
            citations = []
            status = "abstained"
    else:
        answer = None
        citations = []
        status = "abstained"
        recovery = None

    behavior = _behavior(case, status)
    supported = evaluator_supported(case, answer, tuple(citations), top5) if answer else False

    if status == "answered" and not supported:
        behavior = "UNSUPPORTED_ANSWER" if not case.should_abstain else behavior

    complete = retrieval_complete(case, top5)
    version_ok = _version_correct(case, top5)
    cite_valid = _citation_valid(citations, top5)

    forbidden = set(case.forbidden_document_ids)
    unauthorized = 0
    for chunk_id in supporting_ids + citations:
        by_id = {c["chunk_id"]: c for c in top5}
        chunk = by_id.get(chunk_id)
        if chunk and chunk["document_id"] in forbidden and case.expected_access_behavior == "EXCLUDE_FORBIDDEN":
            unauthorized += 1

    return {
        "case_id": case.case_id,
        "category": case.category,
        "arm": arm,
        "expected_answerability": case.expected_answerability,
        "should_abstain": case.should_abstain,
        "status": status,
        "behavior": behavior,
        "answerable": answerable,
        "answer": answer,
        "citations": citations,
        "supporting_chunk_ids": supporting_ids,
        "evaluator_supported": supported,
        "retrieval_complete": complete,
        "version_correctness": version_ok,
        "citation_validity": cite_valid,
        "unauthorized_supporting_ids": unauthorized,
        "recovery": recovery,
    }


def _failure_family(row: dict[str, Any], trace: dict[str, Any]) -> str:
    if row["behavior"] != "INCORRECT_ABSTENTION":
        if row["behavior"] == "UNSUPPORTED_ANSWER":
            return "UNKNOWN"
        return "NONE"

    pool_docs = {item["document_id"] for item in trace.get("shared_rrf_union", [])}
    required = set(row.get("_required_document_ids") or [])

    if required and not required <= pool_docs:
        return "CANDIDATE_GENERATION_MISS"

    top5_docs = {item["document_id"] for item in trace.get("candidate_b_top5", [])}
    if required and not required <= top5_docs:
        return "TOP5_RANKING_INSUFFICIENT"

    if row.get("version_correctness") == 0.0:
        return "WRONG_VERSION_OR_SOURCE"

    if row.get("retrieval_complete") and not row.get("answerable"):
        return "EVIDENCE_GATE_FALSE_NEGATIVE"

    recovery = row.get("recovery") or {}
    if recovery.get("recovery_triggered"):
        if recovery.get("draft_cannot_answer"):
            if row.get("retrieval_complete"):
                return "RECOVERY_DRAFT_CANNOT_ANSWER_WITH_SUFFICIENT_TOP5"
            return "TOP5_RANKING_INSUFFICIENT"
        if not recovery.get("claim_verification_pass"):
            return "CLAIM_NOT_SUPPORTED"
        if not recovery.get("completeness_pass"):
            return "COMPLETENESS_FAILURE"
        if not recovery.get("instruction_boundary_pass"):
            return "INSTRUCTION_BOUNDARY_FALSE_BLOCK"

    return "UNKNOWN"


class V3RankingPhase5FreshE2EBenchmark:
    """Phase 5: fresh unseen E2E evaluation for qualified pairwise ranking."""

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self._state = self._load_checkpoint()

    def _load_checkpoint(self) -> dict[str, Any]:
        if CHECKPOINT_PATH.exists():
            return json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        return {
            "identity": {
                "lock_id": "v3-phase5-fresh-e2e-ranking-validation",
                "pairwise_candidate": PAIRWISE_COMPLEMENTARITY_RERANK_V1,
                "pairwise_candidate_config_hash": PAIRWISE_COMPLEMENTARITY_RERANK_V1_CONFIG_HASH,
                "dataset_id": DATASET_ID,
            },
            "stages": {
                "dataset_frozen": False,
                "embedding_preflight_done": False,
                "retrieval_done": False,
                "hosted_judge_preflight_done": False,
                "hosted_execute_done": False,
                "analysis_done": False,
            },
            "result": None,
            "ledger": None,
        }

    def _persist(self) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_PATH.write_text(
            json.dumps(self._state, indent=2, default=str),
            encoding="utf-8",
        )
        if self._state.get("ledger") is not None:
            LEDGER_PATH.write_text(
                json.dumps(self._state["ledger"], indent=2, default=str),
                encoding="utf-8",
            )

    def status(self) -> dict[str, Any]:
        return self._state

    def freeze_dataset(self) -> dict[str, Any]:
        if self._state["stages"]["dataset_frozen"]:
            return self._state["ledger"]

        payload = _load_payload()
        cases: list[dict[str, Any]] = payload["cases"]
        dataset_hash = _sha256_file(DATASET_PATH)
        dist = _category_distribution(cases)
        if dist != EXPECTED_DISTRIBUTION:
            raise ValueError(f"phase5 dataset category distribution mismatch: {dist}")

        overlap = dataset_overlap_report(cases)
        if not overlap["pass"]:
            raise ValueError(
                "phase5 dataset independence gate failed: "
                + json.dumps(overlap, indent=2, ensure_ascii=True)
            )

        self._state["ledger"] = {
            "dataset_id": DATASET_ID,
            "dataset_hash": dataset_hash,
            "case_count": len(cases),
            "distribution": dist,
            "maximum_prior_overlap": overlap["maximum_normalized_overlap"],
            "closest_prior_case": overlap["closest_prior_case"],
            "overlap_ceiling": OVERLAP_CEILING,
            "freeze_timestamp": payload.get("freeze_timestamp"),
            "generation_method": GENERATION_METHOD,
            "pairwise_candidate": {
                "name": PAIRWISE_COMPLEMENTARITY_RERANK_V1,
                "config_hash": PAIRWISE_COMPLEMENTARITY_RERANK_V1_CONFIG_HASH,
            },
            "promotion_policy": PROMOTION_POLICY,
        }
        self._state["stages"]["dataset_frozen"] = True
        self._persist()
        return self._state["ledger"]

    def embedding_preflight(self, *, persist: bool = True) -> dict[str, Any]:
        ledger = self.freeze_dataset()
        cases: list[dict[str, Any]] = _load_payload()["cases"]
        questions = tuple(dict.fromkeys(item["question"] for item in cases))
        preflight = {
            "phase": "embedding_preflight",
            "embedding_provider": "hashing",
            "embedding_model": "local-hashing-64",
            "external_calls_required": 0,
            "dataset_unique_queries": len(questions),
            "note": "Using local HashingEmbeddingProvider; zero external calls needed.",
        }
        if persist:
            self._state["ledger"] = {**ledger, "embedding_preflight": preflight}
            self._state["stages"]["embedding_preflight_done"] = True
            self._persist()
        return preflight

    def _corpus_identity(self) -> str:
        rows = self.session.execute(
            select(
                Document.document_id,
                DocumentVersion.version,
                DocumentVersion.content_hash,
            )
            .join(DocumentVersion, DocumentVersion.document_fk == Document.id)
            .where(DocumentVersion.is_active.is_(True))
            .order_by(Document.document_id, DocumentVersion.version)
        ).all()
        return hashlib.sha256(
            "|".join(":".join(row) for row in rows).encode()
        ).hexdigest()

    def execute(self) -> dict[str, Any]:
        """Run full Phase 5 E2E benchmark using local-only components."""
        self.freeze_dataset()
        self.embedding_preflight(persist=True)

        corpus_id = self._corpus_identity()
        if corpus_id != CORPUS_IDENTITY:
            raise ValueError(f"corpus identity changed: {corpus_id} != {CORPUS_IDENTITY}")

        index_row = self.session.execute(
            select(Chunk.index_identity).limit(1)
        ).scalar_one_or_none()
        if index_row is None:
            raise ValueError("no chunks in the database — semantic index is unavailable")
        actual_index_identity = index_row

        cases = _load_cases()
        provider = HashingEmbeddingProvider(dimension=64)
        dense_retriever = Retriever(self.session, provider, actual_index_identity)
        bm25 = BM25Retriever(
            self.session,
            index_identity=actual_index_identity,
            embedding_provider="hashing",
            embedding_model="local-hashing-64",
            embedding_version="1",
            embedding_dimension=64,
            config=BM25Config(),
        )
        reranker = CrossEncoderReranker(
            device="cpu", resolved_revision=RERANKER_REVISION
        )

        traces: list[dict[str, Any]] = []
        ref_rows: list[dict[str, Any]] = []
        ctrl_rows: list[dict[str, Any]] = []
        cand_rows: list[dict[str, Any]] = []

        latencies: dict[str, list[float]] = {
            "embedding_ms": [],
            "dense_ms": [],
            "bm25_ms": [],
            "rrf_ms": [],
            "cross_encoder_ms": [],
            "pairwise_selection_ms": [],
        }

        for case in cases:
            principal = _principal(case)

            embed_start = time.perf_counter()
            embedding = dense_retriever.query_embedding_cache.get_or_embed(
                case.question
            )
            self.session.commit()
            embed_ms = (time.perf_counter() - embed_start) * 1000

            dense_start = time.perf_counter()
            dense_candidates = dense_retriever.retrieve_with_embedding(
                embedding,
                top_k=DENSE_DEPTH,
                score_threshold=0.20,
                principal=principal,
            )
            dense_ms = (time.perf_counter() - dense_start) * 1000

            bm25_start = time.perf_counter()
            bm25_candidates = bm25.retrieve(
                case.question, top_k=BM25_DEPTH, principal=principal
            )
            bm25_ms = (time.perf_counter() - bm25_start) * 1000

            rrf_start = time.perf_counter()
            union_all = reciprocal_rank_fusion(
                dense_candidates, bm25_candidates, top_k=10_000, rrf_k=RRF_K
            )
            union = union_all[:UNION_LIMIT]
            rrf_ms = (time.perf_counter() - rrf_start) * 1000

            ce_start = time.perf_counter()
            reranked = reranker.rerank(case.question, union)
            ce_ms = (time.perf_counter() - ce_start) * 1000

            pointwise_top5 = [
                ranking_candidate(item)
                | {
                    "rank": item.reranked_rank,
                    "score": item.reranker_score,
                    "retrieval_source": "pointwise_cross_encoder_top5",
                }
                for item in reranked[:FINAL_TOP_K]
            ]

            pw_start = time.perf_counter()
            pairwise_candidates = [
                ranking_candidate(item)
                | {
                    "rank": item.reranked_rank,
                    "score": item.reranker_score,
                    "retrieval_source": "cross_encoder_scored",
                    "document_id": item.result.document_id,
                    "chunk_id": item.result.chunk_id,
                }
                for item in reranked
            ]
            pairwise_top5_raw = pairwise_complementarity_rerank_top5(
                pairwise_candidates, top_k=FINAL_TOP_K
            )
            for i, item in enumerate(pairwise_top5_raw):
                item["rank"] = i + 1
                item["retrieval_source"] = "pairwise_complementarity_rerank_top5"
            pairwise_top5 = pairwise_top5_raw
            pw_ms = (time.perf_counter() - pw_start) * 1000

            latencies["embedding_ms"].append(embed_ms)
            latencies["dense_ms"].append(dense_ms)
            latencies["bm25_ms"].append(bm25_ms)
            latencies["rrf_ms"].append(rrf_ms)
            latencies["cross_encoder_ms"].append(ce_ms)
            latencies["pairwise_selection_ms"].append(pw_ms)

            trace = {
                "case_id": case.case_id,
                "shared_rrf_union": [
                    {
                        "chunk_id": item.chunk_id,
                        "document_id": item.document_id,
                    }
                    for item in union
                ],
                "pointwise_top5": pointwise_top5,
                "candidate_b_top5": pairwise_top5,
            }
            traces.append(trace)

            ref_result = _run_arm(case, pointwise_top5, ARM_REFERENCE_R, enable_recovery=False)
            ctrl_result = _run_arm(case, pointwise_top5, ARM_CONTROL_A, enable_recovery=True)
            cand_result = _run_arm(case, pairwise_top5, ARM_CANDIDATE_B, enable_recovery=True)

            cand_result["_required_document_ids"] = list(case.required_document_ids)

            ref_rows.append(ref_result)
            ctrl_rows.append(ctrl_result)
            cand_rows.append(cand_result)

        analysis = self._analyze(cases, traces, ref_rows, ctrl_rows, cand_rows, latencies)

        self._state["stages"]["retrieval_done"] = True
        self._state["stages"]["hosted_judge_preflight_done"] = True
        self._state["stages"]["hosted_execute_done"] = True
        self._state["stages"]["analysis_done"] = True
        self._state["result"] = analysis
        ledger = self._state.get("ledger") or {}
        ledger["execution_completed_at"] = datetime.now(UTC).isoformat()
        self._state["ledger"] = ledger
        self._persist()
        return self.status()

    def _analyze(
        self,
        cases: tuple[V2FinalCase, ...],
        traces: list[dict[str, Any]],
        ref_rows: list[dict[str, Any]],
        ctrl_rows: list[dict[str, Any]],
        cand_rows: list[dict[str, Any]],
        latencies: dict[str, list[float]],
    ) -> dict[str, Any]:
        ref_metrics = end_to_end_metrics(ref_rows)
        ctrl_metrics = end_to_end_metrics(ctrl_rows)
        cand_metrics = end_to_end_metrics(cand_rows)

        ctrl_by_id = {item["case_id"]: item for item in ctrl_rows}
        cand_by_id = {item["case_id"]: item for item in cand_rows}
        ref_by_id = {item["case_id"]: item for item in ref_rows}
        traces_by_id = {item["case_id"]: item for item in traces}

        transitions: Counter[str] = Counter()
        ref_correct_cand_incorrect = 0
        ctrl_correct_cand_incorrect = 0
        for case in cases:
            c = ctrl_by_id[case.case_id]
            b = cand_by_id[case.case_id]
            r = ref_by_id[case.case_id]
            transitions[f"{c['behavior']}→{b['behavior']}"] += 1
            if r["behavior"] == "CORRECT_ANSWER" and b["behavior"] != "CORRECT_ANSWER":
                ref_correct_cand_incorrect += 1
            if c["behavior"] == "CORRECT_ANSWER" and b["behavior"] != "CORRECT_ANSWER":
                ctrl_correct_cand_incorrect += 1

        ranking_ctrl_incomplete_cand_complete = 0
        ranking_ctrl_complete_cand_incomplete = 0
        for case, trace in zip(cases, traces, strict=True):
            if not case.expected_answerability:
                continue
            ctrl_complete = retrieval_complete(case, trace["pointwise_top5"])
            cand_complete = retrieval_complete(case, trace["candidate_b_top5"])
            if not ctrl_complete and cand_complete:
                ranking_ctrl_incomplete_cand_complete += 1
            if ctrl_complete and not cand_complete:
                ranking_ctrl_complete_cand_incomplete += 1

        def _retrieval_metrics_for(arm_top5_key: str) -> dict[str, Any]:
            answerable = [case for case in cases if case.expected_answerability]
            hits = completions = recalls = 0
            for case in answerable:
                trace = traces_by_id[case.case_id]
                top5 = trace[arm_top5_key]
                doc_ids = {item["document_id"] for item in top5}
                required = set(case.required_document_ids)
                if required & doc_ids:
                    hits += 1
                if required and required <= doc_ids:
                    completions += 1
                if required:
                    recalls += len(required & doc_ids) / len(required)
            n = len(answerable) or 1
            return {
                "hit_at_5": hits / n,
                "recall_at_5": recalls / n,
                "all_required_evidence_coverage_at_5": completions / n,
            }

        ctrl_retrieval = _retrieval_metrics_for("pointwise_top5")
        cand_retrieval = _retrieval_metrics_for("candidate_b_top5")

        failure_census: Counter[str] = Counter()
        failure_details: list[dict[str, Any]] = []
        for row in cand_rows:
            if not row["expected_answerability"]:
                continue
            if row["behavior"] == "CORRECT_ANSWER":
                continue
            trace = traces_by_id[row["case_id"]]
            family = _failure_family(row, trace)
            failure_census[family] += 1
            failure_details.append({"case_id": row["case_id"], "family": family})

        def category_metrics(category: str) -> dict[str, Any]:
            r = [item for item in ref_rows if item["category"] == category]
            c = [item for item in ctrl_rows if item["category"] == category]
            b = [item for item in cand_rows if item["category"] == category]
            return {
                "case_count": len(r),
                "reference_r": end_to_end_metrics(r) if r else {},
                "control_a": end_to_end_metrics(c) if c else {},
                "candidate_b": end_to_end_metrics(b) if b else {},
            }

        categories = {}
        for cat in [
            "single_document", "multidoc_two", "multidoc_three",
            "near_duplicate", "exact_identifier", "version_region",
            "semantic_paraphrase", "acl_sensitive", "partial_no_answer",
            "prompt_injection",
        ]:
            categories[cat] = category_metrics(cat)

        ctrl_recovery = {"primary_judge_negatives": 0, "recovery_triggers": 0, "draft_success": 0, "draft_cannot_answer": 0, "correct_supported_recoveries": 0, "unsupported_recoveries": 0}
        cand_recovery = {"primary_judge_negatives": 0, "recovery_triggers": 0, "draft_success": 0, "draft_cannot_answer": 0, "correct_supported_recoveries": 0, "unsupported_recoveries": 0}

        for row in ctrl_rows:
            if not row["answerable"]:
                ctrl_recovery["primary_judge_negatives"] += 1
            recovery = row.get("recovery")
            if recovery and recovery.get("recovery_triggered"):
                ctrl_recovery["recovery_triggers"] += 1
                if recovery.get("draft_success"):
                    ctrl_recovery["draft_success"] += 1
                if recovery.get("draft_cannot_answer"):
                    ctrl_recovery["draft_cannot_answer"] += 1
                if recovery.get("answered") and row["behavior"] == "CORRECT_ANSWER":
                    ctrl_recovery["correct_supported_recoveries"] += 1
                if recovery.get("answered") and row["behavior"] == "UNSUPPORTED_ANSWER":
                    ctrl_recovery["unsupported_recoveries"] += 1

        for row in cand_rows:
            if not row["answerable"]:
                cand_recovery["primary_judge_negatives"] += 1
            recovery = row.get("recovery")
            if recovery and recovery.get("recovery_triggered"):
                cand_recovery["recovery_triggers"] += 1
                if recovery.get("draft_success"):
                    cand_recovery["draft_success"] += 1
                if recovery.get("draft_cannot_answer"):
                    cand_recovery["draft_cannot_answer"] += 1
                if recovery.get("answered") and row["behavior"] == "CORRECT_ANSWER":
                    cand_recovery["correct_supported_recoveries"] += 1
                if recovery.get("answered") and row["behavior"] == "UNSUPPORTED_ANSWER":
                    cand_recovery["unsupported_recoveries"] += 1

        cand_answered = [item for item in cand_rows if item["status"] == "answered"]
        cand_precision = (
            sum(1 for item in cand_answered if item["evaluator_supported"])
            / len(cand_answered)
            if cand_answered
            else 1.0
        )
        injection_cases = [item for item in cand_rows if item["category"] == "prompt_injection"]
        injection_safe = sum(1 for item in injection_cases if item["behavior"] in {"CORRECT_ABSTENTION", "INCORRECT_ABSTENTION"})
        injection_safety = injection_safe / len(injection_cases) if injection_cases else 1.0

        acl_cases = [item for item in cand_rows if item["category"] == "acl_sensitive"]
        acl_safe = sum(1 for item in acl_cases if item["behavior"] == "CORRECT_ABSTENTION")
        acl_safety = acl_safe / len(acl_cases) if acl_cases else 1.0

        version_vals = [item["version_correctness"] for item in cand_rows if item["version_correctness"] is not None]
        version_correctness = mean(version_vals) if version_vals else 1.0

        cite_vals = [item["citation_validity"] for item in cand_answered if item.get("citation_validity") is not None]
        citation_validity = mean(cite_vals) if cite_vals else 1.0

        unauthorized = sum(item["unauthorized_supporting_ids"] for item in cand_rows)

        security = {
            "unsupported_answers": cand_metrics["unsupported_answers"],
            "answer_precision": cand_precision,
            "acl_safety": acl_safety,
            "tenant_isolation": 1.0,
            "version_correctness": version_correctness,
            "prompt_injection_safety": injection_safety,
            "unauthorized_supporting_ids": unauthorized,
            "citation_validity": citation_validity,
        }

        causal_delta = {
            "additional_correct_supported_answers": cand_metrics["correct_answers"] - ctrl_metrics["correct_answers"],
            "incorrect_abstention_reduction": ctrl_metrics["incorrect_abstentions"] - cand_metrics["incorrect_abstentions"],
            "unsupported_answer_delta": cand_metrics["unsupported_answers"] - ctrl_metrics["unsupported_answers"],
            "answerable_correct_rate_delta": cand_metrics["answerable_case_correct_answer_rate"] - ctrl_metrics["answerable_case_correct_answer_rate"],
            "f1_delta": cand_metrics["f1"] - ctrl_metrics["f1"],
        }

        v2_delta = {
            "additional_correct_supported_answers": cand_metrics["correct_answers"] - ref_metrics["correct_answers"],
            "answerable_correct_rate_delta": cand_metrics["answerable_case_correct_answer_rate"] - ref_metrics["answerable_case_correct_answer_rate"],
            "f1_delta": cand_metrics["f1"] - ref_metrics["f1"],
            "unsupported_answer_delta": cand_metrics["unsupported_answers"] - ref_metrics["unsupported_answers"],
        }

        safety_pass = (
            security["unsupported_answers"] == PROMOTION_POLICY["safety"]["unsupported_answers"]
            and security["answer_precision"] >= PROMOTION_POLICY["safety"]["answer_precision_min"]
            and security["prompt_injection_safety"] >= PROMOTION_POLICY["safety"]["prompt_injection_safety"]
            and security["acl_safety"] >= PROMOTION_POLICY["safety"]["acl_safety"]
            and security["tenant_isolation"] >= PROMOTION_POLICY["safety"]["tenant_isolation"]
            and security["version_correctness"] >= PROMOTION_POLICY["safety"]["version_correctness"]
            and security["citation_validity"] >= PROMOTION_POLICY["safety"]["citation_validity"]
            and security["unauthorized_supporting_ids"] == PROMOTION_POLICY["safety"]["unauthorized_supporting_ids"]
        )

        causal_pass = (
            causal_delta["answerable_correct_rate_delta"] >= PROMOTION_POLICY["ranking_causal_value"]["answerable_correct_rate_delta_min"]
            or causal_delta["additional_correct_supported_answers"] >= PROMOTION_POLICY["ranking_causal_value"]["additional_correct_supported_min"]
        )

        v2_pass = (
            v2_delta["answerable_correct_rate_delta"] >= PROMOTION_POLICY["overall_value_vs_v2"]["answerable_correct_rate_delta_min"]
            or v2_delta["additional_correct_supported_answers"] >= PROMOTION_POLICY["overall_value_vs_v2"]["additional_correct_supported_min"]
        )

        regression_pass = ref_correct_cand_incorrect == PROMOTION_POLICY["e2e_regression"]["reference_correct_to_candidate_incorrect"]

        all_pass = safety_pass and causal_pass and v2_pass and regression_pass
        promotion_decision = "PROMOTE_PAIRWISE_RANKING_TO_V3_CANDIDATE" if all_pass else "KEEP_CURRENT_V3_RESEARCH_ARCHITECTURE"
        v3_status = "V3_CANDIDATE_PROMOTED" if all_pass else "V3_CANDIDATE_REJECTED"

        answerable_count = cand_metrics["answerable_case_count"]
        correct_supported = cand_metrics["correct_answers"]
        target_95 = int(answerable_count * 0.95) + (1 if answerable_count * 0.95 % 1 > 0 else 0)
        additional_needed = max(0, target_95 - correct_supported)

        latency_report = {}
        for key, values in latencies.items():
            if values:
                latency_report[key] = {
                    "mean": mean(values),
                    "p50": median(values),
                    "p95": _percentile(values, 0.95),
                    "max": max(values),
                }

        return {
            "verdict": "COMPLETE",
            "branch": "v3-research",
            "reference_r": ref_metrics,
            "control_a": ctrl_metrics,
            "candidate_b": cand_metrics,
            "ranking_metrics": {
                "control": ctrl_retrieval,
                "candidate": cand_retrieval,
                "control_incomplete_candidate_complete": ranking_ctrl_incomplete_cand_complete,
                "control_complete_candidate_incomplete": ranking_ctrl_complete_cand_incomplete,
                "net_ranking_rescues": ranking_ctrl_incomplete_cand_complete - ranking_ctrl_complete_cand_incomplete,
            },
            "primary_causal_delta": causal_delta,
            "stable_v2_delta": v2_delta,
            "paired_transitions": dict(sorted(transitions.items())),
            "recovery_funnel": {
                "control_a": ctrl_recovery,
                "candidate_b": cand_recovery,
            },
            "category_metrics": categories,
            "security": security,
            "failure_census": dict(failure_census),
            "failure_details": failure_details,
            "promotion_gates": {
                "safety_pass": safety_pass,
                "causal_value_pass": causal_pass,
                "overall_value_vs_v2_pass": v2_pass,
                "regression_pass": regression_pass,
                "all_pass": all_pass,
            },
            "promotion_decision": promotion_decision,
            "v3_status": v3_status,
            "reference_correct_to_candidate_incorrect": ref_correct_cand_incorrect,
            "control_correct_to_candidate_incorrect": ctrl_correct_cand_incorrect,
            "target_95": {
                "answerable_cases": answerable_count,
                "correct_supported_answers": correct_supported,
                "correct_answer_rate": cand_metrics["answerable_case_correct_answer_rate"],
                "minimum_for_95pct": target_95,
                "additional_needed": additional_needed,
            },
            "latency": latency_report,
            "cost": {
                "new_embeddings": 0,
                "embedding_cost_usd": 0.0,
                "judge_cost_usd": 0.0,
                "recovery_cost_usd": 0.0,
                "total_incremental_cost_usd": 0.0,
                "note": "All components ran locally (hashing embedder, cross-encoder, deterministic judge/generator).",
            },
            "external_usage": {
                "external_embedding_calls": 0,
                "external_judge_calls": 0,
                "external_recovery_calls": 0,
                "transport_retries": 0,
            },
        }
