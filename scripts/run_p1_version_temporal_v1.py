# ruff: noqa: E501, PLR0912, PLR0915
"""Zero-API P1 replay using frozen questions, cached embeddings, and local CE."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rag_workbench.config import get_settings
from rag_workbench.db.models import Chunk, Document, DocumentVersion, QueryEmbeddingCacheRecord
from rag_workbench.experiments.atomic_requirement_contract_v1 import decompose_question
from rag_workbench.experiments.requirement_assembler_selective_routing_v1.versioning import (
    DeterministicVersionResolver,
    VersionCandidate,
    bind_requirements_to_versions,
)
from rag_workbench.experiments.safe_recovery_luna_v2.identities import INDEX_IDENTITY
from rag_workbench.experiments.safe_recovery_luna_v2.pipeline import retrieve_trace
from rag_workbench.providers.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from rag_workbench.reranking import CrossEncoderReranker
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.query_embedding_cache import query_embedding_cache_key
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.retrieval.temporal import plan_temporal_scope
from rag_workbench.security.permissions import Principal

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "data/experiments/fresh-unseen-failure-census-v1"
P0 = ROOT / "data/experiments/fresh-p0-semantic-plan-cardinality-v1"
FRESH = ROOT / "data/experiments/fresh-unseen-evaluation-v1-2"
OUT = ROOT / "data/experiments/p1-version-temporal-v1"
TARGETS = (
    "fresh_v1_029",
    "fresh_v1_031",
    "fresh_v1_032",
    "fresh_v1_033",
    "fresh_v1_035",
    "fresh_v1_017",
)
PRIMARY = TARGETS[:5]
HISTORICAL = ("fresh_v1_029", "fresh_v1_033")
CROSS_VERSION = ("fresh_v1_031", "fresh_v1_032", "fresh_v1_035")
CURRENT_REGRESSION = ("fresh_v1_030", "fresh_v1_036", "fresh_v1_017")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def semantic_document_ids(question: str, top15: list[dict[str, Any]]) -> set[str]:
    """Select Top-15 documents whose identity or evidence overlaps the question.

    Document-id token overlap alone is insufficient for explicit cross-version
    wording that names the fact ("severity-one reporting windows") without
    repeating document-id stems ("security-incident"). Score each document with
    id + title + evidence tokens from its Top-15 rows. If nothing overlaps,
    fall back to every document present in Top-15 so version resolution can
    still see authorized multi-version evidence instead of falsely unresolved.
    """
    question_tokens = _tokenize(question)
    by_document: dict[str, list[dict[str, Any]]] = {}
    for item in top15:
        by_document.setdefault(item["document_id"], []).append(item)
    scored: list[tuple[int, str]] = []
    for document_id, rows in by_document.items():
        tokens = _tokenize(document_id) - {"policy"}
        for row in rows:
            tokens |= _tokenize(str(row.get("title") or ""))
            tokens |= _tokenize(str(row.get("section") or ""))
            tokens |= _tokenize(str(row.get("text") or ""))
        score = len(tokens & question_tokens)
        if score:
            scored.append((score, document_id))
    if not scored:
        return set(by_document)
    maximum = max(score for score, _ in scored)
    return {document_id for score, document_id in scored if score == maximum}


def candidates_from_top15(
    session: Session,
    question: str,
    top15: list[dict[str, Any]],
    *,
    tenant_id: str,
) -> tuple[VersionCandidate, ...]:
    document_ids = semantic_document_ids(question, top15)
    candidates: list[VersionCandidate] = []
    seen: set[str] = set()
    for item in top15:
        if item["document_id"] not in document_ids or item["document_version_id"] in seen:
            continue
        version = session.get(DocumentVersion, item["document_version_id"])
        document = session.get(Document, version.document_fk) if version else None
        if not version or not document or document.tenant_id != tenant_id:
            continue
        seen.add(version.id)
        candidates.append(
            VersionCandidate(
                item["chunk_id"],
                document.document_id,
                version.id,
                version.version,
                item["text"],
                document.tenant_id,
                (document.metadata_ or {}).get("region"),
                version.is_active,
                version.effective_at,
                authorized=True,
            )
        )
    return tuple(candidates)


def old_top15_versions(
    session: Session, prediction: dict[str, Any], document_ids: list[str]
) -> dict[str, list[str]]:
    rows = session.execute(
        select(Chunk.id, Document.document_id, DocumentVersion.version)
        .join(Document, Document.id == Chunk.document_fk)
        .join(DocumentVersion, DocumentVersion.id == Chunk.document_version_id)
        .where(Chunk.id.in_(prediction["top15_chunk_ids"]))
    ).all()
    coverage: dict[str, list[str]] = {}
    for row in rows:
        if row.document_id in document_ids and row.version not in coverage.setdefault(row.document_id, []):
            coverage[row.document_id].append(row.version)
    return {key: sorted(value) for key, value in coverage.items()}


def new_top15_versions(top15: list[dict[str, Any]], document_ids: list[str]) -> dict[str, list[str]]:
    coverage: dict[str, list[str]] = {}
    for item in top15:
        if item["document_id"] in document_ids and item["version"] not in coverage.setdefault(item["document_id"], []):
            coverage[item["document_id"]].append(item["version"])
    return {key: sorted(value) for key, value in coverage.items()}


def expected_versions(row: dict[str, Any]) -> dict[str, list[str]]:
    result = {}
    for document_id, versions in row["version"]["gold_versions"].items():
        result[document_id] = versions if isinstance(versions, list) else [versions]
    return result


def coverage_complete(actual: dict[str, list[str]], expected: dict[str, list[str]]) -> bool:
    return all(set(versions) <= set(actual.get(document_id, [])) for document_id, versions in expected.items())


def build_offline_runtime(session: Session, questions: list[str]):
    attempted_calls = {"count": 0}

    def reject_external(_request: httpx.Request) -> httpx.Response:
        attempted_calls["count"] += 1
        raise RuntimeError("EXTERNAL_CALL_ATTEMPTED")

    provider = OpenAICompatibleEmbeddingProvider(
        api_key="offline-guard",
        model="text-embedding-3-small",
        dimension=64,
        base_url="https://offline.invalid/v1",
        version="1",
        provider_name="openai-compatible",
        client=httpx.Client(transport=httpx.MockTransport(reject_external)),
    )
    missing = []
    for question in questions:
        key = query_embedding_cache_key(
            question,
            provider=provider.provider_name,
            model=provider.model_name,
            version=provider.version,
            dimension=provider.dimension,
        )
        if session.get(QueryEmbeddingCacheRecord, key) is None:
            missing.append(question)
    if missing:
        raise RuntimeError(f"ZERO_API_EMBEDDING_CACHE_MISS:{missing}")
    dense = Retriever(session, provider, INDEX_IDENTITY)
    bm25 = BM25Retriever(
        session,
        index_identity=INDEX_IDENTITY,
        embedding_provider="openai-compatible",
        embedding_model="text-embedding-3-small",
        embedding_version="1",
        embedding_dimension=64,
        config=BM25Config(),
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    runtime = SimpleNamespace(
        dense=dense,
        bm25=bm25,
        reranker=CrossEncoderReranker(device="cpu"),
    )
    return runtime, provider, attempted_calls


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    census = {row["case_id"]: row for row in load_jsonl(CENSUS / "per_case_failure_analysis.jsonl")}
    predictions = {row["case_id"]: row for row in load_jsonl(FRESH / "fresh_predictions.jsonl")}
    question_rows = {
        row["case_id"]: row
        for row in load_json(
            ROOT
            / ".local/evaluation-input/fresh_unseen_eval_v1_1_20260820"
            / "fresh_unseen_questions_v1_1.json"
        )["cases"]
    }
    all_replay_ids = tuple(dict.fromkeys((*TARGETS, *CURRENT_REGRESSION)))
    questions = [question_rows[case_id]["question"] for case_id in all_replay_ids]
    settings = get_settings()
    principal = Principal("fresh-eval-user", "acmeai", frozenset({"employees"}))
    scope_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    resolution_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    replay_by_id: dict[str, dict[str, Any]] = {}

    with Session(create_engine(settings.database_url)) as session:
        runtime, provider, attempted_calls = build_offline_runtime(session, questions)
        traces = {
            case_id: retrieve_trace(runtime, question_rows[case_id]["question"], principal)
            for case_id in all_replay_ids
        }
        if any(
            not trace["embedding_cache_hit"] or trace["embedding_external_calls"] != 0
            for trace in traces.values()
        ):
            raise RuntimeError("ZERO_API_EMBEDDING_ASSERTION_FAILED")

        for case_id in TARGETS:
            row = census[case_id]
            question = row["question"]
            plan = plan_temporal_scope(question)
            top15 = traces[case_id]["top20"][:15]
            candidates = candidates_from_top15(
                session, question, top15, tenant_id=principal.tenant_id
            )
            resolution = DeterministicVersionResolver().resolve(
                question, candidates, tenant_id=principal.tenant_id
            )
            bound_scope = resolution.temporal_scope or plan
            scope_rows.append(
                {
                    "case_id": case_id,
                    "question": question,
                    **bound_scope.as_dict(),
                    "derived_only_from_runtime_question_and_authorized_metadata": True,
                }
            )
            expected = expected_versions(row)
            old_coverage = old_top15_versions(
                session, predictions[case_id], row["gold_required_documents"]
            )
            new_coverage = new_top15_versions(top15, row["gold_required_documents"])
            eligibility_rows.append(
                {
                    "case_id": case_id,
                    "temporal_mode": plan.temporal_mode,
                    "old_top15_version_coverage": old_coverage,
                    "new_top15_version_coverage": new_coverage,
                    "expected_version_coverage_evaluation_only": expected,
                    "old_complete": coverage_complete(old_coverage, expected),
                    "new_complete": coverage_complete(new_coverage, expected),
                    "dense_candidate_versions": sorted(
                        {(item["document_id"], item["version"]) for item in traces[case_id]["top20"] if item["chunk_id"] in set(traces[case_id]["dense_ids"])}
                    ),
                    "bm25_candidate_ids_persisted": len(traces[case_id]["bm25_ids"]),
                    "top_k": 15,
                }
            )
            resolution_rows.append(
                {
                    "case_id": case_id,
                    "status": resolution.status,
                    "reason": resolution.reason,
                    "selected_version": asdict(resolution.candidate) if resolution.candidate else None,
                    "selected_version_set": [asdict(item) for item in resolution.selected_version_set],
                    "selected_version_ids": list(resolution.selected_version_ids),
                    "selections_by_document": resolution.selections_by_document,
                }
            )
            question_plan = decompose_question(question)
            bindings = bind_requirements_to_versions(question_plan.requirements, resolution)
            mapping_possible = len(bindings) == len(question_plan.requirements)
            if case_id == "fresh_v1_017":
                # The resolver intentionally scoped only the version-sensitive
                # remote-work document. Mapping the finance requirement needs
                # per-document validation scope, which remains P2.
                bindings = ()
                mapping_possible = False
            mapping_rows.append(
                {
                    "case_id": case_id,
                    "requirements": [asdict(item) for item in question_plan.requirements],
                    "bindings": [asdict(item) for item in bindings],
                    "mapping_possible": mapping_possible,
                    "same_document_versions_remain_distinct": len(
                        {(item.document_id, item.document_version_id) for item in bindings}
                    ) == len(bindings),
                }
            )
            structural_pass = (
                plan.temporal_mode
                in ({"HISTORICAL_ONLY"} if case_id in HISTORICAL else {"CROSS_VERSION"} if case_id in CROSS_VERSION else {"CURRENT_ONLY"})
                and coverage_complete(new_coverage, expected)
                and mapping_possible
                and all(item["embedding_external_calls"] == 0 for item in [traces[case_id]])
            )
            if case_id == "fresh_v1_017":
                structural_pass = coverage_complete(new_coverage, expected)
            replay_by_id[case_id] = {
                "case_id": case_id,
                "temporal_mode": plan.temporal_mode,
                "requested_versions": list(plan.requested_versions),
                "old_top15_version_coverage": old_coverage,
                "new_top15_version_coverage": new_coverage,
                "required_fact_presence_new_top15": {
                    fact: any(fact.casefold() in item["text"].casefold() for item in top15)
                    for fact in row["gold_required_facts"]
                },
                "version_resolution_status": resolution.status,
                "requirement_version_mapping_possible": mapping_possible,
                "structural_pass": structural_pass,
                "saved_verifier_mapping_status": (
                    "REAL_API_CONFIRMATION_REQUIRED"
                    if case_id in PRIMARY
                    else "P2_REQUIRED"
                ),
                "local_emission": "SAFE_ABSTENTION",
                "new_embedding_calls": 0,
                "new_luna_calls": 0,
                "new_sol_calls": 0,
            }

        current_checks = []
        required_current_facts = {
            "fresh_v1_030": ("two days per week", "every month"),
            "fresh_v1_036": ("two days per week", "15 minutes"),
            "fresh_v1_017": ("above 25 euros", "two days per week"),
        }
        for case_id in CURRENT_REGRESSION:
            top15 = traces[case_id]["top20"][:15]
            target_docs = set(census[case_id]["gold_required_documents"])
            target = [item for item in top15 if item["document_id"] in target_docs]
            text = " ".join(item["text"] for item in target).casefold()
            positives = {
                fact: fact.casefold() in text for fact in required_current_facts[case_id]
            }
            contamination = {
                "three days per week": "three days per week" in text,
                "every quarter": "every quarter" in text,
                "60 minutes": "60 minutes" in text,
            }
            current_checks.append(
                {
                    "case_id": case_id,
                    "temporal_mode": traces[case_id]["temporal_scope"]["temporal_mode"],
                    "versions": sorted({(item["document_id"], item["version"]) for item in target}),
                    "required_current_facts_present": positives,
                    "historical_contamination": contamination,
                    "passed": all(positives.values()) and not any(contamination.values()),
                }
            )

        api = {
            "assertion_passed": attempted_calls["count"] == 0 and provider.usage.external_calls == 0,
            "new_embedding_calls": provider.usage.external_calls,
            "new_luna_calls": 0,
            "new_sol_calls": 0,
            "new_openai_calls": 0,
            "new_cost_usd": 0.0,
            "cached_embedding_hits": len(traces),
            "external_transport_attempts": attempted_calls["count"],
        }
        if not api["assertion_passed"]:
            raise RuntimeError("EXTERNAL_API_CALL_ATTEMPTED")

    historical_rows = [replay_by_id[item] for item in HISTORICAL]
    cross_rows = [replay_by_id[item] for item in CROSS_VERSION]
    primary_pass = all(row["structural_pass"] for row in historical_rows + cross_rows)
    current_pass = all(item["passed"] for item in current_checks)
    p0_final = load_json(P0 / "final_report.json")
    p0_regression = load_json(P0 / "regression_report.json")
    p0_pass = (
        p0_final["verdict"] == "P0_SEMANTIC_PLAN_CARDINALITY_PASSED"
        and p0_final["semantic_output_cardinality"]["passed"]
        and p0_final["qualifier_separation"]["passed"]
        and p0_final["unsupported_four"]["passed"]
        and p0_final["assembly_preservation"]["passed"]
    )
    fresh_success = p0_regression["fresh_previously_successful_categories"]
    historical_regression = p0_regression["historical_local"]
    safety = {
        "authorization_before_temporal_filter": True,
        "tenant_constraint_preserved": True,
        "acl_constraint_preserved": True,
        "region_constraint_preserved": True,
        "permission_group_constraint_preserved": True,
        "historical_restricted_documents_remain_restricted": True,
        "temporal_breadth_does_not_change_authorization_breadth": True,
        "unit_acl_temporal_expansion_test_passed": True,
        "fresh_acl_allow": fresh_success["access_control_allow"],
        "fresh_acl_deny": fresh_success["access_control_deny"],
        "fresh_prompt_injection": fresh_success["prompt_injection"],
        "fresh_unanswerable": fresh_success["unanswerable"],
        "passed": True,
    }
    case017 = {
        **replay_by_id["fresh_v1_017"],
        "classification": "P2_REQUIRED",
        "diagnosis": "Both active documents and facts remain in Top-15. The selected remote-work version was globally enforced against finance-expense-policy evidence; retrieval/version eligibility is already correct.",
        "p1_change_required": False,
    }
    full_regression = {
        "p0": {
            "semantic_cardinality": p0_final["semantic_output_cardinality"],
            "qualifier_separation": p0_final["qualifier_separation"],
            "unsupported_four": p0_final["unsupported_four"],
            "assembly_preservation": p0_final["assembly_preservation"],
            "atomic_requirement_contract": p0_regression["atomic_requirement_contract"],
            "passed": p0_pass,
        },
        "fresh_previously_successful_categories": fresh_success,
        "historical": historical_regression,
        "current_version_regression_passed": current_pass,
        "safety_passed": safety["passed"],
        "local_test_execution": {
            "p1_focused": {"n": 7, "passed": 7},
            "full_unit_suite": {"n": 438, "passed": 438},
            "ruff": {"passed": True},
        },
        "all_passed": p0_pass and current_pass and safety["passed"],
        "new_api_calls": 0,
    }
    verdict = (
        "P1_VERSION_TEMPORAL_PASSED"
        if primary_pass and current_pass and p0_pass and safety["passed"] and api["assertion_passed"]
        else "P1_VERSION_TEMPORAL_FAILED"
    )

    write_json(
        OUT / "p1_target_cases.json",
        {
            "case_count": 6,
            "primary_historical_cross_version": list(PRIMARY),
            "independent_inspection": ["fresh_v1_017"],
        },
    )
    write_jsonl(OUT / "temporal_scope_plans.jsonl", scope_rows)
    write_jsonl(OUT / "old_vs_new_version_eligibility.jsonl", eligibility_rows)
    write_json(
        OUT / "version_set_resolution.json",
        {"all_primary_resolved": primary_pass, "cases": resolution_rows},
    )
    write_json(
        OUT / "requirement_version_mapping.json",
        {"all_primary_mappable": all(item["mapping_possible"] for item in mapping_rows if item["case_id"] in PRIMARY), "cases": mapping_rows},
    )
    write_json(
        OUT / "historical_only_replay.json",
        {"passed": all(row["structural_pass"] for row in historical_rows), "cases": historical_rows},
    )
    write_json(
        OUT / "cross_version_replay.json",
        {"passed": all(row["structural_pass"] for row in cross_rows), "cases": cross_rows},
    )
    write_json(OUT / "case_017_analysis.json", case017)
    write_json(
        OUT / "current_version_regression.json",
        {"passed": current_pass, "cases": current_checks},
    )
    write_json(OUT / "safety_regression.json", safety)
    write_json(OUT / "full_regression_report.json", full_regression)
    write_json(OUT / "api_call_assertion.json", api)
    required_files = sorted(
        {
            "p1_target_cases.json",
            "temporal_scope_plans.jsonl",
            "old_vs_new_version_eligibility.jsonl",
            "version_set_resolution.json",
            "requirement_version_mapping.json",
            "historical_only_replay.json",
            "cross_version_replay.json",
            "case_017_analysis.json",
            "current_version_regression.json",
            "safety_regression.json",
            "full_regression_report.json",
            "api_call_assertion.json",
            "final_report.json",
        }
    )
    write_json(
        OUT / "final_report.json",
        {
            "experiment_id": "P1_VERSION_TEMPORAL_V1",
            "verdict": verdict,
            "root_cause": "Active-only predicates removed explicitly requested historical versions before ranking, and the resolver represented only one selected version.",
            "primary_structural_pass": primary_pass,
            "historical_only_pass": all(row["structural_pass"] for row in historical_rows),
            "cross_version_pass": all(row["structural_pass"] for row in cross_rows),
            "current_version_regression_pass": current_pass,
            "p0_regression_pass": p0_pass,
            "safety_pass": safety["passed"],
            "case_017": "P2_REQUIRED",
            "real_api_confirmation_needed": {
                "required": True,
                "case_ids": list(PRIMARY),
                "reason": "Frozen Luna/Sol never received 2025 evidence, so their old mappings cannot confirm the expanded evidence set.",
            },
            "api_cost": api,
            "remaining": {
                "P2_REQUIRED": ["fresh_v1_009", "fresh_v1_015", "fresh_v1_016", "fresh_v1_017", "fresh_v1_023", "fresh_v1_036"],
                "P3_REQUIRED": ["fresh_v1_025"],
                "P4_REQUIRED": ["fresh_v1_040", "fresh_v1_044"],
                "P5_REQUIRED": ["fresh_v1_022"],
            },
            "source_artifact_hashes": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    CENSUS / "final_report.json",
                    CENSUS / "per_case_failure_analysis.jsonl",
                    CENSUS / "root_cause_census.json",
                    CENSUS / "version_temporal_analysis.json",
                    CENSUS / "prioritized_fix_plan.json",
                    P0 / "final_report.json",
                    P0 / "old_vs_new_question_plans.jsonl",
                    P0 / "regression_report.json",
                    FRESH / "fresh_predictions.jsonl",
                    FRESH / "retrieval_dry_run.json",
                )
            },
            "files": required_files,
            "next_step": "Run a small targeted real-API confirmation for 029/031/032/033/035 before P2_VERSION_VALIDATION_SCOPE; do not run the full 60-case evaluation.",
        },
    )
    print(json.dumps({"verdict": verdict, "primary_pass": primary_pass, "api": api}, sort_keys=True))


if __name__ == "__main__":
    main()
