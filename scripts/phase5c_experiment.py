# ruff: noqa: E501
"""V3 Phase 5C: Post-Closure Audit-Corrected Four-Arm Evaluation.

PHASE5C_AUDIT_CORRECTED_PAIRED_BENCHMARK
CONTROLLED_COMPARATIVE_EVALUATION

Arms:
  R — V2 stable reference (pointwise + V2 answer/abstain, no recovery)
  A — Current V3 (pointwise + recovery, baseline generator)
  B — Ranking only (pairwise + recovery, baseline generator)
  C — Generator only (pointwise + recovery, GENERATOR_COMPLETENESS_V2)
  D — Combined (pairwise + recovery, GENERATOR_COMPLETENESS_V2)

Requires: PostgreSQL running, EMBEDDING_API_KEY, JUDGE_API_KEY set.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from rag_workbench.answerability.base import AnswerabilityResult, GateEvidence
from rag_workbench.answerability.cache import CachedAnswerabilityGate, gate_cache_key
from rag_workbench.answerability.openai_compatible import (
    EVIDENCE_GATE_PROMPT_VERSION,
    OpenAICompatibleAnswerabilityGate,
)
from rag_workbench.answerability.planning import ExternalJudgeCallLimitGate
from rag_workbench.answerability.transport import DEFAULT_TRANSPORT_RETRY_POLICY
from rag_workbench.answerability.validation import validate_gate_result_with_error
from rag_workbench.config import Settings, get_settings
from rag_workbench.evaluation.generation_metrics import deterministic_citation_correctness
from rag_workbench.evaluation.hybrid_metrics import (
    aggregate_retrieval_metrics,
    retrieval_case_metrics,
)
from rag_workbench.experiments.hybrid_reranker_benchmark import (
    BM25_DEPTH,
    DENSE_DEPTH,
    FINAL_TOP_K,
    RRF_K,
    UNION_LIMIT,
    _principal as _make_principal,
    _trace,
    aggregate_pool,
    pool_metrics,
)
from rag_workbench.experiments.reranker_e2e_benchmark import (
    CORPUS_IDENTITY,
    RERANKER_REVISION,
    SEMANTIC_INDEX_IDENTITY,
    _percentile,
    _result,
)
from rag_workbench.experiments.v2_document_diversity import ranking_candidate
from rag_workbench.experiments.v2_final_benchmark import (
    HIDDEN_GROUND_TRUTH_FIELDS,
    V2FinalCase,
    _behavior,
    _gate_evidence,
    classification,
)
from rag_workbench.experiments.v2_quality_recovery import stable_hash
from rag_workbench.experiments.v2_sufficiency_fn import SOL_MODEL, retrieval_complete
from rag_workbench.experiments.v3_final_ab import (
    instruction_boundary_safety_gate,
    recovery_trace,
)
from rag_workbench.experiments.v3_generate_verify import (
    V3GenerateVerifyBenchmark,
    document_instruction_followed,
    evaluator_supported,
)
from rag_workbench.providers.llm.extractive import (
    EXTRACTIVE_V1_1_MODEL,
    EXTRACTIVE_V2_MODEL,
    ExtractiveGenerationProvider,
)
from rag_workbench.recovery.runtime import evaluate_recovery
from rag_workbench.reranking import Reranker
from rag_workbench.reranking.pairwise_complementarity import (
    ALGORITHM_ID as PAIRWISE_ALGORITHM_ID,
    ALGORITHM_VERSION as PAIRWISE_ALGORITHM_VERSION,
    pairwise_configuration,
    select_pairwise_complementarity_top5,
)
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.security.permissions import Principal

# --- Constants ---
DATASET_PATH = Path("data/eval/phase5c/v3_clean_120_cases.jsonl")
FREEZE_PATH = Path("data/eval/phase5c/phase5c_freeze.json")
OUTPUT_DIR = Path("data/experiments/v3-phase5c-audit-corrected")
PAIRWISE_HASH = "527afb76a0226018e158291c0212e16cfdc31e0d9990cfd4686d72c1d3df69cd"


@dataclass
class Phase5CCase:
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


def load_cases() -> list[Phase5CCase]:
    cases = []
    with open(DATASET_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cases.append(Phase5CCase(
                query_id=d["query_id"],
                question=d["question"],
                expected_answerable=d["expected_answerable"],
                should_abstain=d["should_abstain"],
                expected_answer=d.get("expected_answer"),
                required_facts=d.get("required_facts", []),
                required_document_ids=d.get("required_document_ids", []),
                required_chunk_ids=d.get("required_chunk_ids", []),
                required_version_ids=d.get("required_version_ids", {}),
                allowed_tenant_ids=d.get("allowed_tenant_ids", []),
                category=d["category"],
                forbidden_document_ids=d.get("forbidden_document_ids", []),
                expected_document_ids=d.get("expected_document_ids", []),
                security_checks=d.get("security_checks", []),
                required_chunk_markers=d.get("required_chunk_markers", []),
                principal=d.get("principal", {"principal_id": "evaluation-user", "tenant_id": "acmeai", "permission_groups": ["employees"]}),
            ))
    return cases


def principal_from_case(case: Phase5CCase) -> Principal:
    return Principal(
        principal_id=case.principal["principal_id"],
        tenant_id=case.principal["tenant_id"],
        permission_groups=tuple(case.principal.get("permission_groups", [])),
    )


def check_all_required_facts(answer: str | None, required_facts: list[str]) -> bool:
    """Strict completeness: all required facts must be present in the answer."""
    if not answer:
        return False
    answer_lower = answer.lower()
    for fact in required_facts:
        if fact.lower() not in answer_lower:
            return False
    return True


def strict_behavior(case: Phase5CCase, status: str, answer: str | None) -> str:
    """Phase 5C strict correctness: partial answers are INCORRECT."""
    if case.should_abstain:
        return "CORRECT_ABSTENTION" if status == "abstained" else "UNSUPPORTED_ANSWER"
    if status == "abstained":
        return "INCORRECT_ABSTENTION"
    if case.expected_answerable and answer:
        if check_all_required_facts(answer, case.required_facts):
            return "CORRECT_ANSWER"
        else:
            return "INCORRECT_ANSWER"
    return "INCORRECT_ANSWER"


def run_preflight(session: Session, settings: Settings, cases: list[Phase5CCase]) -> dict[str, Any]:
    """Cost/credential preflight."""
    from rag_workbench.db.models import QueryEmbeddingCacheRecord, AnswerabilityGateCacheRecord

    keys = {
        query_embedding_cache_key(
            c.question, provider="openai-compatible", model="text-embedding-3-small",
            version="1", dimension=64,
        )
        for c in cases
    }
    embed_cached = sum(session.get(QueryEmbeddingCacheRecord, k) is not None for k in keys)
    embed_missing = len(keys) - embed_cached

    max_judge_calls = len(cases) * 2  # pointwise + pairwise
    max_recovery = len(cases) * 4  # draft+verifier for two ranking paths
    max_attempts = DEFAULT_TRANSPORT_RETRY_POLICY.max_total_attempts

    preflight = {
        "total_cases": len(cases),
        "reusable_query_embeddings": embed_cached,
        "new_query_embeddings": embed_missing,
        "maximum_judge_logical_calls": max_judge_calls,
        "maximum_recovery_draft_calls": len(cases) * 2,
        "maximum_verifier_calls": len(cases) * 2,
        "maximum_physical_attempts": (max_judge_calls + max_recovery) * max_attempts,
        "estimated_incremental_cost_usd": round(embed_missing * 0.02 / 1_000_000 * 100 + max_judge_calls * 0.003, 4),
        "embedding_api_key_set": bool(settings.embedding_api_key),
        "judge_api_key_set": bool(settings.judge_api_key),
        "allow_external_calls": settings.allow_external_calls,
        "allow_external_judge_calls": settings.allow_external_judge_calls,
        "configured_embedding_ceiling": settings.max_external_embedding_calls,
        "configured_judge_ceiling": settings.max_external_judge_calls,
    }

    blocked = not (settings.embedding_api_key and settings.judge_api_key
                   and settings.allow_external_calls and settings.allow_external_judge_calls)
    preflight["authorization_ok"] = not blocked
    if blocked:
        preflight["stop_code"] = "EXTERNAL_CREDENTIALS_REQUIRED"
    return preflight


def generate_answer(
    case: Phase5CCase,
    top5: list[dict[str, Any]],
    gate_result: AnswerabilityResult,
    session: Session,
    provider_revision: str = EXTRACTIVE_V1_1_MODEL,
) -> dict[str, Any]:
    """Run deterministic extractive generation on judge-selected supporting evidence."""
    from rag_workbench.generation.context_builder import ContextBuilder
    from rag_workbench.generation.prompts import build_grounded_prompt
    from rag_workbench.providers.llm.base import GenerationContext, GenerationRequest
    from rag_workbench.retrieval.vector_search import RetrievalResult

    gen_provider = ExtractiveGenerationProvider(revision=provider_revision)

    if not gate_result.answerable:
        return {"status": "abstained", "answer": None, "citations": [], "extractive_path": None, "generation_latency_ms": 0}

    supporting_ids = set(gate_result.supporting_chunk_ids)
    evidence_items = [
        item for item in top5 if item["chunk_id"] in supporting_ids
    ] if supporting_ids else top5

    results = [
        RetrievalResult(
            chunk_id=item["chunk_id"],
            document_id=item["document_id"],
            document_version_id=item.get("document_version_id", ""),
            version=item.get("version", ""),
            text=item.get("text", ""),
            score=item.get("score", 0.0),
            rank=i + 1,
            source=item.get("source", ""),
            source_type=item.get("source_type", "markdown"),
            title=item.get("title", ""),
            section=item.get("section"),
            page=item.get("page"),
        )
        for i, item in enumerate(evidence_items)
    ]

    ctx_builder = ContextBuilder(token_budget=1200)
    context = ctx_builder.build(results)

    if not context.items:
        return {"status": "abstained", "answer": None, "citations": [], "extractive_path": None, "generation_latency_ms": 0}

    prompt = build_grounded_prompt(case.question, context, strict_evidence_only=True)
    request = GenerationRequest(
        question=case.question,
        prompt=prompt,
        contexts=tuple(
            GenerationContext(chunk_id=item.result.chunk_id, citation_label=item.citation_label, text=item.result.text)
            for item in context.items
        ),
    )

    started = time.perf_counter()
    generated = gen_provider.generate(request)
    gen_ms = (time.perf_counter() - started) * 1000

    from rag_workbench.generation.citations import build_citations
    citations = build_citations(context, generated.used_chunk_ids)
    cited_ids = [c.chunk_id for c in citations]

    status = "answered" if generated.answer and citations else "abstained"
    return {
        "status": status,
        "answer": generated.answer if status == "answered" else None,
        "citations": cited_ids,
        "extractive_path": (generated.metadata or {}).get("extractive_path"),
        "generation_latency_ms": gen_ms,
    }


def run_experiment():
    settings = get_settings()
    engine = create_engine(settings.database_url)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    cases = load_cases()
    print(f"Loaded {len(cases)} cases")

    # Verify freeze
    freeze = json.loads(FREEZE_PATH.read_text())
    assert freeze["case_count"] == 120
    assert freeze["dataset_qa"] == "120/120 PASS"
    print("Dataset freeze verified.")

    # Verify pairwise identity
    pw_hash = stable_hash(pairwise_configuration())
    assert pw_hash == PAIRWISE_HASH, f"Pairwise hash mismatch: {pw_hash}"
    print(f"Pairwise ranking identity verified: {PAIRWISE_HASH[:16]}...")

    # Preflight
    preflight = run_preflight(session, settings, cases)
    print(f"Preflight: {json.dumps({k: v for k, v in preflight.items() if k != 'stop_code'}, indent=2)}")
    if not preflight["authorization_ok"]:
        print(f"\nSTOPPED: {preflight.get('stop_code', 'UNKNOWN')}")
        print("Cannot proceed without external API credentials.")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "preflight.json").write_text(json.dumps(preflight, indent=2))
        return preflight

    # Load overlap strata for stratified reporting
    overlap_data = freeze["overlap_per_query"]
    overlap_map = {item["query_id"]: item for item in overlap_data}

    print("\n=== RUNNING PHASE 5C FOUR-ARM EXPERIMENT ===\n")

    # Initialize providers
    inner = V3GenerateVerifyBenchmark(session, settings)
    embedding_provider = inner._embedding_provider()
    dense_retriever = Retriever(session, embedding_provider, SEMANTIC_INDEX_IDENTITY)
    bm25 = BM25Retriever(
        session,
        index_identity=SEMANTIC_INDEX_IDENTITY,
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_version="1",
        embedding_dimension=64,
        config=BM25Config(),
    )
    reranker: Reranker = inner.reranker_factory()
    if reranker.resolved_revision != RERANKER_REVISION:
        raise ValueError(f"Reranker revision mismatch: {reranker.resolved_revision} != {RERANKER_REVISION}")

    gate = inner._gate(preflight["maximum_judge_logical_calls"])
    recovery = inner._recovery_cache(
        preflight["maximum_recovery_draft_calls"] + preflight["maximum_verifier_calls"]
    )

    # Results per arm
    ref_rows: list[dict] = []
    arm_a_rows: list[dict] = []
    arm_b_rows: list[dict] = []
    arm_c_rows: list[dict] = []
    arm_d_rows: list[dict] = []

    for idx, case in enumerate(cases):
        v2_case = case.as_v2_case()
        principal = principal_from_case(case)
        print(f"  [{idx+1}/120] {case.query_id} ({case.category})", end="")

        # --- Shared retrieval ---
        embedding = dense_retriever.query_embedding_cache.get_or_embed(case.question)
        session.commit()

        dense_candidates = dense_retriever.retrieve_with_embedding(
            embedding, top_k=DENSE_DEPTH, score_threshold=0.28, principal=principal
        )
        bm25_candidates = bm25.retrieve(case.question, top_k=BM25_DEPTH, principal=principal)
        union_all = reciprocal_rank_fusion(dense_candidates, bm25_candidates, top_k=10_000, rrf_k=RRF_K)
        union = union_all[:UNION_LIMIT]

        # Cross-encoder reranking
        reranked = reranker.rerank(case.question, union)

        # Pointwise top-5
        pointwise_top5 = [
            ranking_candidate(item) | {"rank": item.reranked_rank, "score": item.reranker_score, "retrieval_source": "pointwise_cross_encoder_top5"}
            for item in reranked[:FINAL_TOP_K]
        ]

        # Pairwise top-5
        pairwise_selected = select_pairwise_complementarity_top5(reranked, case.question, top_k=FINAL_TOP_K)
        pairwise_top5 = [
            ranking_candidate(item) | {"rank": i + 1, "score": item.reranker_score, "retrieval_source": "pairwise_complementarity_top5"}
            for i, item in enumerate(pairwise_selected)
        ]

        # --- Judge: Pointwise evidence ---
        pw_evidence = _gate_evidence(pointwise_top5)
        try:
            pw_proposed = gate.evaluate(case.question, pw_evidence)
            pw_validated = validate_gate_result_with_error(pw_proposed, pw_evidence, session=session, principal=principal)
            pw_result = pw_validated.result
        except Exception:
            pw_result = AnswerabilityResult.fail_closed()

        # --- Judge: Pairwise evidence ---
        pair_evidence = _gate_evidence(pairwise_top5)
        try:
            pair_proposed = gate.evaluate(case.question, pair_evidence)
            pair_validated = validate_gate_result_with_error(pair_proposed, pair_evidence, session=session, principal=principal)
            pair_result = pair_validated.result
        except Exception:
            pair_result = AnswerabilityResult.fail_closed()

        # === REFERENCE R: V2 path (pointwise, no recovery, baseline generator) ===
        ref_gen = generate_answer(case, pointwise_top5, pw_result, session, EXTRACTIVE_V1_1_MODEL)
        ref_behavior = strict_behavior(case, ref_gen["status"], ref_gen["answer"])
        ref_rows.append({
            "case_id": case.query_id, "category": case.category,
            "expected_answerable": case.expected_answerable,
            "status": ref_gen["status"], "behavior": ref_behavior,
            "answer": ref_gen["answer"], "citations": ref_gen["citations"],
            "citation_validity": deterministic_citation_correctness(tuple(ref_gen["citations"]), tuple(item["chunk_id"] for item in pointwise_top5)),
            "all_facts_present": check_all_required_facts(ref_gen["answer"], case.required_facts) if case.expected_answerable else None,
            "extractive_path": ref_gen["extractive_path"],
            "ranking": "pointwise", "generator": EXTRACTIVE_V1_1_MODEL,
        })

        # === ARM A: Current V3 (pointwise + recovery, baseline generator) ===
        pw_recovery = evaluate_recovery(
            session=session, principal=principal, question=case.question,
            chunks=pw_evidence, cache=recovery,
            primary_answerable=pw_result.answerable,
            primary_schema_valid=True,
            safety_gate=instruction_boundary_safety_gate,
        )
        if pw_result.answerable:
            a_gen = ref_gen  # Reuse computation
        elif pw_recovery.answered:
            a_gen = {"status": "answered", "answer": pw_recovery.answer, "citations": list(pw_recovery.citations), "extractive_path": "recovery", "generation_latency_ms": 0}
        else:
            a_gen = {"status": "abstained", "answer": None, "citations": [], "extractive_path": None, "generation_latency_ms": 0}
        a_behavior = strict_behavior(case, a_gen["status"], a_gen["answer"])
        arm_a_rows.append({
            "case_id": case.query_id, "category": case.category,
            "expected_answerable": case.expected_answerable,
            "status": a_gen["status"], "behavior": a_behavior,
            "answer": a_gen["answer"], "citations": a_gen["citations"],
            "citation_validity": deterministic_citation_correctness(tuple(a_gen["citations"]), tuple(item["chunk_id"] for item in pointwise_top5)),
            "all_facts_present": check_all_required_facts(a_gen["answer"], case.required_facts) if case.expected_answerable else None,
            "extractive_path": a_gen["extractive_path"],
            "recovery_triggered": pw_recovery.triggered,
            "ranking": "pointwise", "generator": EXTRACTIVE_V1_1_MODEL,
        })

        # === ARM B: Ranking only (pairwise + recovery, baseline generator) ===
        pair_recovery = evaluate_recovery(
            session=session, principal=principal, question=case.question,
            chunks=pair_evidence, cache=recovery,
            primary_answerable=pair_result.answerable,
            primary_schema_valid=True,
            safety_gate=instruction_boundary_safety_gate,
        )
        if pair_result.answerable:
            b_gen = generate_answer(case, pairwise_top5, pair_result, session, EXTRACTIVE_V1_1_MODEL)
        elif pair_recovery.answered:
            b_gen = {"status": "answered", "answer": pair_recovery.answer, "citations": list(pair_recovery.citations), "extractive_path": "recovery", "generation_latency_ms": 0}
        else:
            b_gen = {"status": "abstained", "answer": None, "citations": [], "extractive_path": None, "generation_latency_ms": 0}
        b_behavior = strict_behavior(case, b_gen["status"], b_gen["answer"])
        arm_b_rows.append({
            "case_id": case.query_id, "category": case.category,
            "expected_answerable": case.expected_answerable,
            "status": b_gen["status"], "behavior": b_behavior,
            "answer": b_gen["answer"], "citations": b_gen["citations"],
            "citation_validity": deterministic_citation_correctness(tuple(b_gen["citations"]), tuple(item["chunk_id"] for item in pairwise_top5)),
            "all_facts_present": check_all_required_facts(b_gen["answer"], case.required_facts) if case.expected_answerable else None,
            "extractive_path": b_gen["extractive_path"],
            "recovery_triggered": pair_recovery.triggered,
            "ranking": "pairwise", "generator": EXTRACTIVE_V1_1_MODEL,
        })

        # === ARM C: Generator only (pointwise + recovery, V2 generator) ===
        if pw_result.answerable:
            c_gen = generate_answer(case, pointwise_top5, pw_result, session, EXTRACTIVE_V2_MODEL)
        elif pw_recovery.answered:
            c_gen = {"status": "answered", "answer": pw_recovery.answer, "citations": list(pw_recovery.citations), "extractive_path": "recovery", "generation_latency_ms": 0}
        else:
            c_gen = {"status": "abstained", "answer": None, "citations": [], "extractive_path": None, "generation_latency_ms": 0}
        c_behavior = strict_behavior(case, c_gen["status"], c_gen["answer"])
        arm_c_rows.append({
            "case_id": case.query_id, "category": case.category,
            "expected_answerable": case.expected_answerable,
            "status": c_gen["status"], "behavior": c_behavior,
            "answer": c_gen["answer"], "citations": c_gen["citations"],
            "citation_validity": deterministic_citation_correctness(tuple(c_gen["citations"]), tuple(item["chunk_id"] for item in pointwise_top5)),
            "all_facts_present": check_all_required_facts(c_gen["answer"], case.required_facts) if case.expected_answerable else None,
            "extractive_path": c_gen["extractive_path"],
            "recovery_triggered": pw_recovery.triggered,
            "ranking": "pointwise", "generator": EXTRACTIVE_V2_MODEL,
        })

        # === ARM D: Combined (pairwise + recovery, V2 generator) ===
        if pair_result.answerable:
            d_gen = generate_answer(case, pairwise_top5, pair_result, session, EXTRACTIVE_V2_MODEL)
        elif pair_recovery.answered:
            d_gen = {"status": "answered", "answer": pair_recovery.answer, "citations": list(pair_recovery.citations), "extractive_path": "recovery", "generation_latency_ms": 0}
        else:
            d_gen = {"status": "abstained", "answer": None, "citations": [], "extractive_path": None, "generation_latency_ms": 0}
        d_behavior = strict_behavior(case, d_gen["status"], d_gen["answer"])
        arm_d_rows.append({
            "case_id": case.query_id, "category": case.category,
            "expected_answerable": case.expected_answerable,
            "status": d_gen["status"], "behavior": d_behavior,
            "answer": d_gen["answer"], "citations": d_gen["citations"],
            "citation_validity": deterministic_citation_correctness(tuple(d_gen["citations"]), tuple(item["chunk_id"] for item in pairwise_top5)),
            "all_facts_present": check_all_required_facts(d_gen["answer"], case.required_facts) if case.expected_answerable else None,
            "extractive_path": d_gen["extractive_path"],
            "recovery_triggered": pair_recovery.triggered,
            "ranking": "pairwise", "generator": EXTRACTIVE_V2_MODEL,
        })

        session.commit()
        print(f" R={ref_behavior[0]} A={a_behavior[0]} B={b_behavior[0]} C={c_behavior[0]} D={d_behavior[0]}")

    session.close()

    # --- Analysis ---
    print("\n=== ANALYSIS ===\n")
    results = analyze(cases, ref_rows, arm_a_rows, arm_b_rows, arm_c_rows, arm_d_rows, overlap_map)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    (OUTPUT_DIR / "preflight.json").write_text(json.dumps(preflight, indent=2))
    print(f"\nResults persisted to {OUTPUT_DIR}/")
    return results


def metrics_for_rows(rows: list[dict], cases: list[Phase5CCase]) -> dict[str, Any]:
    """Compute standard metrics for a set of result rows."""
    total = len(rows)
    correct_answers = sum(1 for r in rows if r["behavior"] == "CORRECT_ANSWER")
    correct_abstentions = sum(1 for r in rows if r["behavior"] == "CORRECT_ABSTENTION")
    incorrect_answers = sum(1 for r in rows if r["behavior"] == "INCORRECT_ANSWER")
    incorrect_abstentions = sum(1 for r in rows if r["behavior"] == "INCORRECT_ABSTENTION")
    unsupported = sum(1 for r in rows if r["behavior"] == "UNSUPPORTED_ANSWER")

    answerable_rows = [r for r in rows if r["expected_answerable"]]
    answerable_correct = sum(1 for r in answerable_rows if r["behavior"] == "CORRECT_ANSWER")

    tp = correct_answers
    fp = unsupported + incorrect_answers
    fn = incorrect_abstentions
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (correct_answers + correct_abstentions) / total if total else 0.0

    # Citation metrics
    citation_valid = [r for r in rows if r.get("citation_validity") is not None]
    citation_validity = mean(r["citation_validity"] for r in citation_valid) if citation_valid else 1.0

    # Safety
    pi_cases = [r for r in rows if r["category"] == "prompt_injection"]
    pi_safe = sum(1 for r in pi_cases if r["behavior"] in ("CORRECT_ABSTENTION",))
    acl_cases = [r for r in rows if r["category"] == "acl_sensitive"]
    acl_safe = sum(1 for r in acl_cases if r["behavior"] in ("CORRECT_ABSTENTION",))
    version_cases = [r for r in rows if r["category"] == "version_region"]

    # Generator completeness given complete evidence
    complete_evidence_rows = [
        r for r in rows
        if r["expected_answerable"] and r["status"] == "answered"
    ]
    gen_complete = sum(1 for r in complete_evidence_rows if r.get("all_facts_present")) if complete_evidence_rows else 0

    return {
        "total": total,
        "correct_answers": correct_answers,
        "correct_abstentions": correct_abstentions,
        "incorrect_answers": incorrect_answers,
        "incorrect_abstentions": incorrect_abstentions,
        "unsupported_answers": unsupported,
        "strict_accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "answerable_correct_rate": round(answerable_correct / len(answerable_rows), 6) if answerable_rows else 0.0,
        "citation_validity": round(citation_validity, 6),
        "prompt_injection_safety": f"{pi_safe}/{len(pi_cases)}" if pi_cases else "N/A",
        "acl_safety": f"{acl_safe}/{len(acl_cases)}" if acl_cases else "N/A",
        "generator_completeness_given_evidence": f"{gen_complete}/{len(complete_evidence_rows)}" if complete_evidence_rows else "N/A",
    }


def paired_delta(base_rows: list[dict], compare_rows: list[dict], cases: list[Phase5CCase]) -> dict[str, Any]:
    base_map = {r["case_id"]: r for r in base_rows}
    comp_map = {r["case_id"]: r for r in compare_rows}
    rescues = []
    regressions = []
    for case in cases:
        b = base_map[case.query_id]
        c = comp_map[case.query_id]
        if b["behavior"] != "CORRECT_ANSWER" and c["behavior"] == "CORRECT_ANSWER":
            rescues.append(case.query_id)
        if b["behavior"] == "CORRECT_ANSWER" and c["behavior"] != "CORRECT_ANSWER":
            regressions.append(case.query_id)

    base_m = metrics_for_rows(base_rows, cases)
    comp_m = metrics_for_rows(compare_rows, cases)
    return {
        "rescues": rescues,
        "regressions": regressions,
        "net_correct_change": len(rescues) - len(regressions),
        "strict_accuracy_delta": round(comp_m["strict_accuracy"] - base_m["strict_accuracy"], 6),
        "f1_delta": round(comp_m["f1"] - base_m["f1"], 6),
        "unsupported_delta": comp_m["unsupported_answers"] - base_m["unsupported_answers"],
    }


def filter_by_stratum(rows: list[dict], overlap_map: dict, stratum: str) -> list[dict]:
    return [r for r in rows if overlap_map.get(r["case_id"], {}).get("stratum") == stratum]


def analyze(
    cases: list[Phase5CCase],
    ref_rows: list[dict],
    a_rows: list[dict],
    b_rows: list[dict],
    c_rows: list[dict],
    d_rows: list[dict],
    overlap_map: dict,
) -> dict[str, Any]:
    cases_by_stratum = {
        "FULL": cases,
        "LOW_OVERLAP": [c for c in cases if overlap_map.get(c.query_id, {}).get("stratum") == "LOW_OVERLAP"],
        "MODERATE_OVERLAP": [c for c in cases if overlap_map.get(c.query_id, {}).get("stratum") == "MODERATE_OVERLAP"],
        "HIGH_OVERLAP": [c for c in cases if overlap_map.get(c.query_id, {}).get("stratum") == "HIGH_OVERLAP"],
    }

    def rows_for_stratum(rows, stratum):
        if stratum == "FULL":
            return rows
        return filter_by_stratum(rows, overlap_map, stratum)

    results = {"phase": "V3_PHASE5C_POST_CLOSURE_AUDIT_RESEARCH", "completed_at": datetime.now(UTC).isoformat()}

    # Metrics per arm per stratum
    for arm_name, arm_rows in [("R", ref_rows), ("A", a_rows), ("B", b_rows), ("C", c_rows), ("D", d_rows)]:
        results[f"arm_{arm_name}"] = {}
        for stratum in ["FULL", "LOW_OVERLAP", "MODERATE_OVERLAP", "HIGH_OVERLAP"]:
            s_rows = rows_for_stratum(arm_rows, stratum)
            s_cases = cases_by_stratum[stratum]
            if s_rows:
                results[f"arm_{arm_name}"][stratum] = metrics_for_rows(s_rows, s_cases)

    # Paired deltas
    for stratum in ["FULL", "LOW_OVERLAP"]:
        s_a = rows_for_stratum(a_rows, stratum)
        s_b = rows_for_stratum(b_rows, stratum)
        s_c = rows_for_stratum(c_rows, stratum)
        s_d = rows_for_stratum(d_rows, stratum)
        s_cases = cases_by_stratum[stratum]
        if s_a:
            results[f"ranking_effect_{stratum}"] = paired_delta(s_a, s_b, s_cases)
            results[f"generator_effect_{stratum}"] = paired_delta(s_a, s_c, s_cases)
            results[f"combined_effect_{stratum}"] = paired_delta(s_a, s_d, s_cases)

    # Failure census
    census = Counter()
    for row in a_rows:
        if row["behavior"] not in ("CORRECT_ANSWER", "CORRECT_ABSTENTION"):
            census[row["behavior"]] += 1
    results["failure_census_arm_a"] = dict(census)

    # Safety gates (ARM D as strongest candidate)
    d_metrics = results["arm_D"]["FULL"]
    results["safety_gates"] = {
        "unsupported_answers": d_metrics["unsupported_answers"],
        "precision": d_metrics["precision"],
        "prompt_injection_safety": d_metrics["prompt_injection_safety"],
        "acl_safety": d_metrics["acl_safety"],
        "citation_validity": d_metrics["citation_validity"],
    }

    # Overlap sensitivity
    full_ba = results.get("ranking_effect_FULL", {})
    low_ba = results.get("ranking_effect_LOW_OVERLAP", {})
    full_ca = results.get("generator_effect_FULL", {})
    low_ca = results.get("generator_effect_LOW_OVERLAP", {})
    full_da = results.get("combined_effect_FULL", {})
    low_da = results.get("combined_effect_LOW_OVERLAP", {})

    results["overlap_sensitivity"] = {
        "ranking": {
            "full_delta": full_ba.get("strict_accuracy_delta", 0),
            "low_overlap_delta": low_ba.get("strict_accuracy_delta", 0),
        },
        "generator": {
            "full_delta": full_ca.get("strict_accuracy_delta", 0),
            "low_overlap_delta": low_ca.get("strict_accuracy_delta", 0),
        },
        "combined": {
            "full_delta": full_da.get("strict_accuracy_delta", 0),
            "low_overlap_delta": low_da.get("strict_accuracy_delta", 0),
        },
    }

    return results


if __name__ == "__main__":
    run_experiment()
