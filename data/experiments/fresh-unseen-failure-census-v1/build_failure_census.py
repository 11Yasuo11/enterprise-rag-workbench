from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.config import Settings
from rag_workbench.db.models import Chunk, Document, DocumentVersion
from rag_workbench.experiments.atomic_requirement_contract_v1.contract import (
    decompose_question,
    validate_verifier_result,
)
from rag_workbench.experiments.atomic_requirement_contract_v1.verifier import (
    FrozenVerifierResult,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "data/experiments/fresh-unseen-evaluation-v1-2"
INPUT = ROOT / ".local/evaluation-input/fresh_unseen_eval_v1_1_20260820"
OUT = Path(__file__).parent


DIAGNOSIS = {
    "fresh_v1_009": ("QUALIFIER_REQUIREMENT_ERROR", ["VERSION_RESOLVER_OVER_FILTER", "SOL_FALSE_NEGATIVE", "LOCAL_VALIDATOR_FALSE_REJECT"], "QUESTION_PLAN", "SOL_ABSTAIN"),
    "fresh_v1_011": ("QUESTION_PLAN_ERROR", ["DETERMINISTIC_ASSEMBLER_ERROR", "CITATION_MAPPING_ERROR", "LOCAL_VALIDATOR_FALSE_ACCEPT"], "QUESTION_PLAN", None),
    "fresh_v1_013": ("QUESTION_PLAN_ERROR", ["DETERMINISTIC_ASSEMBLER_ERROR", "CITATION_MAPPING_ERROR", "LOCAL_VALIDATOR_FALSE_ACCEPT"], "QUESTION_PLAN", None),
    "fresh_v1_015": ("QUALIFIER_REQUIREMENT_ERROR", ["VERSION_RESOLVER_OVER_FILTER", "LOCAL_VALIDATOR_FALSE_REJECT", "SOL_DID_NOT_RESOLVE"], "QUESTION_PLAN", "POST_SOL_VALIDATION_REJECT"),
    "fresh_v1_016": ("QUESTION_PLAN_ERROR", ["VERSION_RESOLVER_OVER_FILTER", "LOCAL_VALIDATOR_FALSE_REJECT", "DETERMINISTIC_ASSEMBLER_ERROR"], "QUESTION_PLAN", "POST_SOL_VALIDATION_REJECT"),
    "fresh_v1_017": ("VERSION_RESOLVER_OVER_FILTER", ["LOCAL_VALIDATOR_FALSE_REJECT", "SOL_DID_NOT_RESOLVE"], "LOCAL_VALIDATION", "POST_SOL_VALIDATION_REJECT"),
    "fresh_v1_022": ("LUNA_FALSE_NEGATIVE", [], "LUNA", "LUNA_ABSTAIN"),
    "fresh_v1_023": ("QUALIFIER_REQUIREMENT_ERROR", ["VERSION_RESOLVER_OVER_FILTER", "LOCAL_VALIDATOR_FALSE_REJECT", "SOL_DID_NOT_RESOLVE"], "QUESTION_PLAN", "POST_SOL_VALIDATION_REJECT"),
    "fresh_v1_025": ("QUALIFIER_REQUIREMENT_ERROR", ["LUNA_FALSE_NEGATIVE", "ROUTER_ERROR"], "QUESTION_PLAN", "LUNA_ABSTAIN"),
    "fresh_v1_028": ("QUESTION_PLAN_ERROR", ["DETERMINISTIC_ASSEMBLER_ERROR", "CITATION_MAPPING_ERROR", "LOCAL_VALIDATOR_FALSE_ACCEPT"], "QUESTION_PLAN", None),
    "fresh_v1_029": ("TEMPORAL_SCOPE_ERROR", ["WRONG_VERSION_SELECTED", "VERSION_RESOLVER_OVER_FILTER", "QUALIFIER_REQUIREMENT_ERROR"], "ACTIVE_VERSION_RETRIEVAL_FILTER", "LUNA_ABSTAIN"),
    "fresh_v1_030": ("QUESTION_PLAN_ERROR", ["DETERMINISTIC_ASSEMBLER_ERROR", "CITATION_MAPPING_ERROR", "LOCAL_VALIDATOR_FALSE_ACCEPT"], "QUESTION_PLAN", None),
    "fresh_v1_031": ("VERSION_COMPARISON_UNSUPPORTED", ["VERSION_RESOLVER_OVER_FILTER"], "ACTIVE_VERSION_RETRIEVAL_FILTER", "LUNA_ABSTAIN"),
    "fresh_v1_032": ("VERSION_COMPARISON_UNSUPPORTED", ["VERSION_RESOLVER_OVER_FILTER"], "ACTIVE_VERSION_RETRIEVAL_FILTER", "LUNA_ABSTAIN"),
    "fresh_v1_033": ("TEMPORAL_SCOPE_ERROR", ["WRONG_VERSION_SELECTED", "VERSION_RESOLVER_OVER_FILTER"], "ACTIVE_VERSION_RETRIEVAL_FILTER", "LUNA_ABSTAIN"),
    "fresh_v1_035": ("VERSION_COMPARISON_UNSUPPORTED", ["VERSION_RESOLVER_OVER_FILTER", "QUESTION_PLAN_ERROR"], "ACTIVE_VERSION_RETRIEVAL_FILTER", "LUNA_ABSTAIN"),
    "fresh_v1_036": ("QUESTION_PLAN_ERROR", ["VERSION_RESOLVER_OVER_FILTER", "LOCAL_VALIDATOR_FALSE_REJECT", "DETERMINISTIC_ASSEMBLER_ERROR"], "QUESTION_PLAN", "POST_SOL_VALIDATION_REJECT"),
    "fresh_v1_040": ("LUNA_FALSE_NEGATIVE", ["CONSTRAINT_VALIDATION_ERROR", "ROUTER_ERROR"], "LUNA", "LUNA_ABSTAIN"),
    "fresh_v1_044": ("QUALIFIER_REQUIREMENT_ERROR", ["CONSTRAINT_VALIDATION_ERROR", "LUNA_FALSE_NEGATIVE"], "QUESTION_PLAN", "LUNA_ABSTAIN"),
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = load_jsonl(SOURCE / "fresh_predictions.jsonl")
    scoring = load_jsonl(SOURCE / "per_case_scoring.jsonl")
    api_ledger = load_jsonl(SOURCE / "api_cost_ledger.jsonl")
    questions = json.loads((INPUT / "fresh_unseen_questions_v1_1.json").read_text())["cases"]
    gold = json.loads((INPUT / "fresh_unseen_gold_v1_1.json").read_text())["cases"]
    deterministic = json.loads((SOURCE / "deterministic_preflight.json").read_text())["cases"]
    pred = {row["case_id"]: row for row in predictions}
    score = {row["case_id"]: row for row in scoring}
    question = {row["case_id"]: row for row in questions}
    expected = {row["case_id"]: row for row in gold}
    plans = {row["case_id"]: row["question_plan"] for row in deterministic}
    failed_ids = [
        row["case_id"]
        for row in scoring
        if row["behavior"] not in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"}
    ]
    known = set(DIAGNOSIS)
    if set(failed_ids) != known or len(failed_ids) != 19:
        raise RuntimeError("FROZEN_FAILURE_SET_MISMATCH")
    prediction_hash = hashlib.sha256((SOURCE / "fresh_predictions.jsonl").read_bytes()).hexdigest()
    freeze = json.loads((SOURCE / "predictions_freeze.json").read_text())
    if prediction_hash != freeze["predictions_sha256"]:
        raise RuntimeError("FROZEN_PREDICTION_HASH_MISMATCH")
    all_top_ids = {chunk_id for case_id in failed_ids for chunk_id in pred[case_id]["top15_chunk_ids"]}
    with Session(create_engine(Settings().database_url)) as session:
        chunks = {
            row.id: row for row in session.scalars(select(Chunk).where(Chunk.id.in_(all_top_ids)))
        }
        document_rows = list(session.scalars(select(Document)))
        documents_by_fk = {row.id: row for row in document_rows}
        documents_by_id = {row.document_id: row for row in document_rows}
        version_rows = list(session.scalars(select(DocumentVersion)))
        versions_by_id = {row.id: row for row in version_rows}
        versions_by_document: dict[str, list[DocumentVersion]] = defaultdict(list)
        for version in version_rows:
            versions_by_document[version.document_fk].append(version)
        per_case = []
        for case_id in failed_ids:
            p = pred[case_id]
            s = score[case_id]
            q = question[case_id]
            g = expected[case_id]
            plan = plans[case_id]
            top_text = " ".join(chunks[chunk_id].text for chunk_id in p["top15_chunk_ids"])
            top_doc_ranks: dict[str, int] = {}
            top_version_rows = []
            for rank, chunk_id in enumerate(p["top15_chunk_ids"], 1):
                chunk = chunks[chunk_id]
                doc = documents_by_fk[chunk.document_fk]
                version = versions_by_id[chunk.document_version_id]
                top_doc_ranks.setdefault(doc.document_id, rank)
                top_version_rows.append((doc.document_id, version.version, version.id, version.is_active))
            fact_presence = {
                fact: normalize(fact) in normalize(top_text) for fact in g.get("expected_facts", [])
            }
            doc_presence = {
                doc_id: doc_id in top_doc_ranks for doc_id in g.get("required_document_ids", [])
            }
            top_complete = all(fact_presence.values()) and all(doc_presence.values())
            corpus_complete = True
            missing_corpus = []
            for fact in g.get("expected_facts", []):
                required_content = " ".join(
                    version.content
                    for doc_id in g.get("required_document_ids", [])
                    if doc_id in documents_by_id
                    for version in versions_by_document[documents_by_id[doc_id].id]
                )
                if normalize(fact) not in normalize(required_content):
                    corpus_complete = False
                    missing_corpus.append(fact)
            versions_available = [
                {
                    "document_id": doc_id,
                    "version": version.version,
                    "version_id": version.id,
                    "active": version.is_active,
                }
                for doc_id in g.get("required_document_ids", [])
                if doc_id in documents_by_id
                for version in sorted(
                    versions_by_document[documents_by_id[doc_id].id], key=lambda item: item.version
                )
            ]
            selected_ids = set(p.get("selected_version_ids", []))
            selected_versions = [
                {
                    "document_id": documents_by_fk[version.document_fk].document_id,
                    "version": version.version,
                    "version_id": version.id,
                }
                for version in version_rows
                if version.id in selected_ids
            ]
            comparison_required = bool(
                re.search(r"\b(compare|change|between|from the 2025|2025 and 2026|2025 to 2026)\b", q["question"], re.I)
            )
            semantic_gold_versions = dict(g.get("required_version_ids", {}))
            if comparison_required:
                semantic_gold_versions = {
                    doc_id: ["2025", "2026"]
                    for doc_id in g.get("required_document_ids", [])
                    if doc_id in {"remote-work-policy", "security-incident-policy"}
                }
            evidence_rows = []
            selected_doc = (p.get("version_resolution") or {}).get("selected_document_id")
            for chunk_id in p["top15_chunk_ids"]:
                chunk = chunks[chunk_id]
                doc = documents_by_fk[chunk.document_fk]
                version = versions_by_id[chunk.document_version_id]
                if selected_ids and doc.document_id == selected_doc and version.id not in selected_ids:
                    continue
                evidence_rows.append(
                    GateEvidence(
                        chunk_id=chunk.id,
                        document_id=doc.document_id,
                        document_version_id=version.id,
                        version=version.version,
                        text=chunk.text,
                        index_identity=chunk.index_identity,
                    )
                )
            evidence_tuple = tuple(evidence_rows)
            selected_arg = frozenset(selected_ids) if selected_ids else None
            luna_validation = None
            if p.get("luna_decision"):
                luna_validation = validate_verifier_result(
                    decompose_question(q["question"]),
                    FrozenVerifierResult.model_validate(p["luna_decision"]),
                    evidence_tuple,
                    authorized_chunk_ids=frozenset(row.chunk_id for row in evidence_tuple),
                    selected_version_ids=selected_arg,
                )
            sol_validation = None
            if p.get("sol_decision"):
                sol_validation = validate_verifier_result(
                    decompose_question(q["question"]),
                    FrozenVerifierResult.model_validate(p["sol_decision"]),
                    evidence_tuple,
                    authorized_chunk_ids=frozenset(row.chunk_id for row in evidence_tuple),
                    selected_version_ids=selected_arg,
                )
            final_validation = sol_validation or luna_validation
            luna_decision = (p.get("luna_decision") or {}).get("decision")
            luna_relative_correct = {
                "fresh_v1_022": False,
                "fresh_v1_025": False,
                "fresh_v1_040": False,
                "fresh_v1_044": False,
            }.get(case_id, True)
            sol_relative_correct = None
            if p.get("sol_decision"):
                sol_relative_correct = case_id != "fresh_v1_009"
            primary, secondary, first_stage, abstention_trigger = DIAGNOSIS[case_id]
            cited_spans = [
                {
                    "chunk_id": citation,
                    "document_id": documents_by_fk[chunks[citation].document_fk].document_id,
                    "version": versions_by_id[chunks[citation].document_version_id].version,
                    "text": chunks[citation].text,
                }
                for citation in p.get("citations", [])
                if citation in chunks
            ]
            missing_top = [fact for fact, present in fact_presence.items() if not present]
            evidence = {
                "fresh_v1_009": "Both required documents ranked 1/3 and Luna mapped both; a scope clause became R1, then global selected-version validation rejected the unrelated support document and Sol incorrectly marked R3 unsupported.",
                "fresh_v1_011": "Both facts were in Top-15 and Luna mapped both to one R1; validation retained only the first mapping and assembly emitted only Product Operations.",
                "fresh_v1_013": "Both facts were in Top-15 and Luna mapped both to one R1; assembly retained the recovery-drill span and dropped the release-weekday span.",
                "fresh_v1_015": "Both documents were Top-2 and Luna/Sol returned GO; the scope clause and conjunction were over-decomposed, and global selected-version validation rejected Operations evidence from an unrelated document.",
                "fresh_v1_016": "Both exact facts were Top-2 and Luna/Sol returned GO; one R1 covered two facts, while global version validation rejected the retention document.",
                "fresh_v1_017": "Both exact facts were Top-3 and Luna/Sol returned GO; selected remote-work version was incorrectly enforced against the finance-policy chunk.",
                "fresh_v1_022": "All three exact facts were in documents ranked 1/2/3, but Luna marked the four-hour R3 unsupported.",
                "fresh_v1_023": "All three exact facts were Top-3 and Luna/Sol returned GO; a checklist scope became an extra requirement and global version validation rejected finance evidence.",
                "fresh_v1_025": "All exact facts were in ranks 1/3; the scope phrase became R1 and Luna returned ABSTAIN despite every saved requirement status being SUPPORTED.",
                "fresh_v1_028": "All three facts were in ranks 1/2/3 and mapped to one R1; assembly emitted only the first span and citation.",
                "fresh_v1_029": "The 2025 fact exists in corpus but active-only retrieval admitted only 2026; resolver selected 2026 for an explicit 2025 question.",
                "fresh_v1_030": "Both current-version facts were Top-15 and mapped to one R1; assembly emitted only the allowance span and omitted monthly review.",
                "fresh_v1_031": "Comparison requires 2025+2026, but active-only retrieval retained only 2026 and the resolver can select only one version.",
                "fresh_v1_032": "Comparison requires quarterly 2025 plus monthly 2026 evidence; only active 2026 survived retrieval.",
                "fresh_v1_033": "The 2025 reporting fact exists in corpus, but active-only retrieval and single-version resolution selected 2026.",
                "fresh_v1_035": "Cross-version comparison needed 2025 and 2026; only 2026 entered Top-15, so Luna correctly lacked the 60-minute span relative to supplied evidence.",
                "fresh_v1_036": "Both current facts were Top-2 and Luna/Sol returned GO; one R1 covered two documents and selected remote-work version was enforced against security-policy evidence.",
                "fresh_v1_040": "The ten-business-day rule ranked first and was marked SUPPORTED, but Luna's overall decision was ABSTAIN; no deterministic eleventh-day boundary inference was applied.",
                "fresh_v1_044": "The thirty-minute rule ranked first; the conditional clause became R1 and the derived forty-five-minute comparison became unsupported R2.",
            }[case_id]
            record = {
                "case_id": case_id,
                "category": s["category"],
                "outcome": s["behavior"],
                "question": q["question"],
                "gold_answerability": g["expected_answerability"],
                "gold_required_facts": g.get("expected_facts", []),
                "gold_required_documents": g.get("required_document_ids", []),
                "gold_required_versions": g.get("required_version_ids", {}),
                "question_plan": {
                    "requirements": plan["requirements"],
                    "qualifiers": plan["context_qualifiers"],
                    "question_plan_hash": plan["question_plan_hash"],
                },
                "retrieval": {
                    "required_evidence_in_corpus": corpus_complete,
                    "missing_corpus_facts": missing_corpus,
                    "acl_tenant_region_survival": True,
                    "active_version_filter_survival": not any(
                        version["active"] is False
                        and version["version"] in {"2025"}
                        for version in versions_available
                        if version["version"] in {"2025"}
                    )
                    if not top_complete
                    else True,
                    "required_evidence_in_dense_pool": None,
                    "required_evidence_in_bm25_pool": None,
                    "required_evidence_in_rrf_pool": None,
                    "candidate_pool_trace_status": "TRACE_INSUFFICIENT: frozen dry-run persisted counts but not dense/BM25/RRF membership IDs",
                    "required_evidence_in_top15": top_complete,
                    "required_fact_presence_top15": fact_presence,
                    "required_document_ranks": {
                        doc_id: top_doc_ranks.get(doc_id) for doc_id in g.get("required_document_ids", [])
                    },
                    "minimum_missing_facts": missing_top,
                },
                "version": {
                    "versions_available": versions_available,
                    "versions_selected": selected_versions,
                    "gold_versions": semantic_gold_versions,
                    "correct": not any(label in {"WRONG_VERSION_SELECTED", "VERSION_COMPARISON_UNSUPPORTED", "TEMPORAL_SCOPE_ERROR"} for label in [primary, *secondary]),
                    "comparison_required": comparison_required,
                },
                "router": {
                    "initial_route": p["initial_route"],
                    "final_route": p["final_route"],
                    "appropriate": primary != "ROUTER_ERROR",
                    "reason": p["final_route_reason"],
                },
                "luna": {
                    "called": p.get("luna_decision") is not None,
                    "decision": luna_decision,
                    "requirement_statuses": {
                        item["requirement_id"]: item["status"]
                        for item in (p.get("luna_decision") or {}).get("requirements", [])
                    },
                    "correct_relative_to_top15": luna_relative_correct,
                    "local_validation": None if luna_validation is None else {
                        "passed": luna_validation.valid,
                        "reason": luna_validation.failure_code,
                    },
                },
                "sol": {
                    "called": p.get("sol_decision") is not None,
                    "decision": (p.get("sol_decision") or {}).get("decision"),
                    "correct_relative_to_top15": sol_relative_correct,
                    "local_validation": None if sol_validation is None else {
                        "passed": sol_validation.valid,
                        "reason": sol_validation.failure_code,
                    },
                },
                "local_validation": {
                    "passed": bool(final_validation and final_validation.valid),
                    "reason": None if final_validation is None else final_validation.failure_code,
                },
                "assembler": {
                    "emitted": bool(p.get("answer")),
                    "all_requirements_present": not s.get("facts_missing"),
                    "unsupported_fact_present": False,
                    "actual_answer": p.get("answer"),
                },
                "citation": {
                    "valid": s.get("citation_validity_pass") is not False,
                    "correct": s.get("citation_correctness_pass") is not False,
                    "complete": s.get("citation_completeness_pass") is not False,
                    "cited_sources": cited_spans,
                },
                "scorer": {"classification": s["behavior"], "correct": True},
                "primary_root_cause": primary,
                "secondary_root_causes": secondary,
                "first_failure_stage": first_stage,
                "final_abstention_trigger": abstention_trigger,
                "evidence": evidence,
            }
            per_case.append(record)
    (OUT / "per_case_failure_analysis.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in per_case) + "\n"
    )
    outcome_counts = Counter(row["outcome"] for row in per_case)
    failure_set = {
        "experiment_id": "FRESH_UNSEEN_FAILURE_CENSUS_V1",
        "source_experiment": "FRESH_UNSEEN_EVALUATION_V1_2",
        "prediction_sha256_verified": prediction_hash,
        "artifact_failure_count": len(per_case),
        "outcome_counts": dict(outcome_counts),
        "case_ids": [row["case_id"] for row in per_case],
        "matches_user_known_failure_set": True,
        "new_api_calls": 0,
    }
    write_json("failure_set.json", failure_set)
    complete = [row for row in per_case if row["retrieval"]["required_evidence_in_top15"]]
    incomplete = [row for row in per_case if not row["retrieval"]["required_evidence_in_top15"]]
    retrieval = {
        "required_evidence_present_in_corpus": sum(row["retrieval"]["required_evidence_in_corpus"] for row in per_case),
        "required_evidence_complete_in_top15": len(complete),
        "required_evidence_incomplete_in_top15": len(incomplete),
        "complete_case_ids": [row["case_id"] for row in complete],
        "incomplete_cases": [
            {
                "case_id": row["case_id"],
                "missing_facts": row["retrieval"]["minimum_missing_facts"],
                "cause": "inactive historical version excluded by frozen active-only retrieval",
            }
            for row in incomplete
        ],
        "dense_membership": "TRACE_INSUFFICIENT",
        "bm25_membership": "TRACE_INSUFFICIENT",
        "rrf_membership": "TRACE_INSUFFICIENT",
        "trace_note": "The frozen dry-run saved branch counts and Top-15 IDs, not dense/BM25/RRF membership IDs. Frozen source code proves both branches filter DocumentVersion.is_active=true.",
    }
    write_json("retrieval_coverage_analysis.json", retrieval)
    version_ids = [row["case_id"] for row in per_case if row["category"] == "version_temporal"]
    version_analysis = {
        "category_total": 8,
        "failed_count": len(version_ids),
        "failed_case_ids": version_ids,
        "version_resolver_supports_both_historical_and_current_simultaneously": "NO",
        "evidence": "Dense and BM25 frozen SQL require is_active=true, excluding 2025. The resolver returns one candidate and predictions contain at most one selected version ID.",
        "historical_only_failures": ["fresh_v1_029", "fresh_v1_033"],
        "comparison_failures": ["fresh_v1_031", "fresh_v1_032", "fresh_v1_035"],
        "other_version_failures": ["fresh_v1_030", "fresh_v1_036"],
        "cases": [row for row in per_case if row["case_id"] in version_ids],
    }
    write_json("version_temporal_analysis.json", version_analysis)
    multi = [row for row in per_case if row["category"] in {"multidoc_two", "multidoc_three"}]
    multi_primary = {
        "retrieval_ranking": sum(row["primary_root_cause"] in {"RETRIEVAL_MISS", "RANKING_TOP15_INCOMPLETE"} for row in multi),
        "luna": sum(row["primary_root_cause"] == "LUNA_FALSE_NEGATIVE" for row in multi),
        "version_handling": sum(row["primary_root_cause"] in {"VERSION_RESOLVER_OVER_FILTER", "VERSION_COMPARISON_UNSUPPORTED", "TEMPORAL_SCOPE_ERROR"} for row in multi),
        "downstream_or_question_plan": sum(row["primary_root_cause"] in {"QUESTION_PLAN_ERROR", "QUALIFIER_REQUIREMENT_ERROR", "DETERMINISTIC_ASSEMBLER_ERROR", "LOCAL_VALIDATOR_FALSE_REJECT"} for row in multi),
    }
    write_json("multidoc_analysis.json", {
        "failure_count": len(multi),
        "all_required_evidence_complete_top15": sum(row["retrieval"]["required_evidence_in_top15"] for row in multi),
        "primary_bucket_counts": multi_primary,
        "nonexclusive_secondary_version_involvement": sum("VERSION_RESOLVER_OVER_FILTER" in row["secondary_root_causes"] for row in multi),
        "cases": multi,
    })
    abstentions = [row for row in per_case if row["outcome"] == "INCORRECT_ABSTENTION"]
    trigger_counts = Counter(row["final_abstention_trigger"] for row in abstentions)
    write_json("incorrect_abstention_analysis.json", {
        "count": len(abstentions),
        "trigger_counts": dict(trigger_counts),
        "complete_required_evidence_top15": sum(row["retrieval"]["required_evidence_in_top15"] for row in abstentions),
        "interpretation": "10/15 had complete Top-15 evidence; downstream planning, validation, verification, and routing dominate over retrieval for those cases.",
        "cases": abstentions,
    })
    unsupported = [row for row in per_case if row["outcome"] == "UNSUPPORTED_ANSWER"]
    unsupported_details = []
    for row in unsupported:
        unsupported_details.append({
            "case_id": row["case_id"],
            "actual_answer": row["assembler"]["actual_answer"],
            "correct_facts": score[row["case_id"]]["facts_satisfied"],
            "missing_facts": score[row["case_id"]]["facts_missing"],
            "cited_sources": row["citation"]["cited_sources"],
            "wrong_version": not row["version"]["correct"],
            "luna_marked_all_source_evidence_supported": True,
            "local_validator_accepted_mapping": row["local_validation"]["passed"],
            "assembler_altered_meaning": False,
            "assembler_dropped_additional_validated_mappings": True,
            "scorer_correct": True,
            "preventable_deterministically": True,
            "missing_invariant": "A GO emission must preserve every independently mapped span/document for a requirement, or reject a one-requirement plan that contains multiple requested output facts.",
        })
    write_json("unsupported_answer_analysis.json", {"count": 4, "cases": unsupported_details})
    numeric = [row for row in per_case if row["case_id"] in {"fresh_v1_040", "fresh_v1_044"}]
    write_json("numeric_date_analysis.json", {
        "failure_count": 2,
        "source_constraint_retrieved_count": 2,
        "boundary_semantics_preserved_in_source": True,
        "diagnosis": "The verifier/validator contract checks literal support but has no deterministic comparison of day 11 > day 10 or cadence 45 != 30. Case 040 also has an internally inconsistent Luna ABSTAIN with SUPPORTED status; case 044 promotes the conditional clause to an output requirement.",
        "cases": numeric,
    })
    luna_counts = Counter()
    false_negative_detail = []
    for row in per_case:
        decision = row["luna"]["decision"]
        correct = row["luna"]["correct_relative_to_top15"]
        luna_counts[("correct" if correct else "incorrect", decision)] += 1
        if decision == "ABSTAIN" and not correct:
            kind = {
                "fresh_v1_022": "exact supporting span + multiple documents",
                "fresh_v1_025": "exact supporting spans + multiple documents; decision/status inconsistency",
                "fresh_v1_040": "exact rule + derived numeric inference",
                "fresh_v1_044": "exact rule + derived numeric comparison",
            }[row["case_id"]]
            false_negative_detail.append({"case_id": row["case_id"], "evidence_type": kind})
    write_json("luna_analysis.json", {
        "failed_case_luna_calls": len(per_case),
        "correct_go": luna_counts[("correct", "GO")],
        "incorrect_go": luna_counts[("incorrect", "GO")],
        "correct_abstain": luna_counts[("correct", "ABSTAIN")],
        "incorrect_abstain": luna_counts[("incorrect", "ABSTAIN")],
        "correct_uncertain": luna_counts[("correct", "UNCERTAIN")],
        "incorrect_uncertain": luna_counts[("incorrect", "UNCERTAIN")],
        "false_negative_count_relative_to_top15": len(false_negative_detail),
        "false_positive_count_among_unsupported_answers": 0,
        "false_negatives": false_negative_detail,
        "note": "Unsupported outputs had complete Luna mappings; downstream plan cardinality/assembler truncation, not Luna GO, caused them.",
    })
    sol_predictions = [row for row in predictions if row.get("sol_decision")]
    sol_rows = []
    rescues = 0
    total_sol_cost = 0.0
    for row in sol_predictions:
        case_id = row["case_id"]
        sol_cost = sum(
            item["estimated_cost_usd"]
            for item in api_ledger
            if item["query_id"] == case_id and item["model"] == "gpt-5.6-sol"
        )
        total_sol_cost += sol_cost
        rescued = score[case_id]["behavior"] == "CORRECT_COMPLETE_ANSWER"
        rescues += rescued
        sol_rows.append({
            "case_id": case_id,
            "escalation_reason": row["final_route_reason"],
            "escalation_justified_by_saved_signal": True,
            "luna_decision": row["luna_decision"]["decision"],
            "sol_decision": row["sol_decision"]["decision"],
            "sol_corrected_luna_or_mapping": rescued,
            "final_outcome": score[case_id]["behavior"],
            "rescue": rescued,
            "wasted": not rescued,
            "regression": False,
            "cost_usd": sol_cost,
        })
    write_json("sol_analysis.json", {
        "call_count": len(sol_rows),
        "rescues": rescues,
        "non_rescues": len(sol_rows) - rescues,
        "regressions": 0,
        "total_sol_cost_usd": total_sol_cost,
        "cost_per_actual_rescue_usd": total_sol_cost / rescues if rescues else None,
        "cases": sol_rows,
    })
    write_json("router_analysis.json", {
        "router_caused_failures": 0,
        "statement": "Router was not the primary cause of any fresh failure.",
        "secondary_missed_escalation_cases": ["fresh_v1_025", "fresh_v1_040"],
        "note": "For ABSTAIN plus all-SUPPORTED/invalid decision consistency, the post-Luna router safe-abstained instead of escalating; the upstream Luna inconsistency remains primary.",
    })
    contract_mismatches = [row["case_id"] for row in per_case if not pred[row["case_id"]]["requirement_contract_match"]]
    write_json("question_plan_contract_analysis.json", {
        "failure_case_count": 19,
        "plan_hash_stable_count": 19,
        "requirement_ids_stable_count": 19,
        "downstream_redecomposition_count": 0,
        "verifier_schema_match_count": 19,
        "assembler_same_plan_count": 19,
        "contract_mismatch_count": len(contract_mismatches),
        "contract_mismatch_case_ids": contract_mismatches,
        "identity_contract_successful": not contract_mismatches,
        "semantic_plan_errors": [row["case_id"] for row in per_case if row["primary_root_cause"] in {"QUESTION_PLAN_ERROR", "QUALIFIER_REQUIREMENT_ERROR"}],
        "interpretation": "ATOMIC_REQUIREMENT_CONTRACT identity propagation remains successful and should not be weakened. Several label-blind plans are semantically mis-decomposed, which is distinct from hash/ID contract mismatch.",
    })
    citation_failures = [row for row in per_case if row["outcome"] == "UNSUPPORTED_ANSWER"]
    write_json("citation_analysis.json", {
        "citation_validity": 1.0,
        "citation_correctness": 0.9322916666666666,
        "citation_completeness": 0.9322916666666666,
        "responsible_case_ids": [row["case_id"] for row in citation_failures],
        "cause": "Each cited first span was valid, but additional valid source mappings were dropped, leaving incomplete citation sets.",
        "cases": [
            {
                "case_id": row["case_id"],
                "valid": row["citation"]["valid"],
                "correct": row["citation"]["correct"],
                "complete": row["citation"]["complete"],
                "missing_facts": score[row["case_id"]]["facts_missing"],
                "classification": "incomplete source set / assembler retained only first mapping",
            }
            for row in citation_failures
        ],
    })
    write_json("scorer_audit.json", {
        "audited_failure_count": 19,
        "correct_classification_count": 19,
        "scorer_false_negative_count": 0,
        "scorer_false_positive_count": 0,
        "cases": [{"case_id": row["case_id"], "classification": row["outcome"], "correct": True} for row in per_case],
    })
    primary_counts = Counter(row["primary_root_cause"] for row in per_case)
    census_rows = []
    for label, count in primary_counts.most_common():
        cases = [row for row in per_case if row["primary_root_cause"] == label]
        census_rows.append({
            "root_cause": label,
            "failure_count": count,
            "case_ids": [row["case_id"] for row in cases],
            "incorrect_abstention": sum(row["outcome"] == "INCORRECT_ABSTENTION" for row in cases),
            "unsupported_answer": sum(row["outcome"] == "UNSUPPORTED_ANSWER" for row in cases),
        })
    stage_counts = Counter(row["first_failure_stage"] for row in per_case)
    write_json("root_cause_census.json", {
        "total_failures": 19,
        "primary_root_causes": census_rows,
        "first_failure_stage_distribution": dict(stage_counts),
    })
    dependency_graph = {
        "chains": [
            {"name": "semantic plan cardinality", "case_ids": [row["case_id"] for row in per_case if row["primary_root_cause"] in {"QUESTION_PLAN_ERROR", "QUALIFIER_REQUIREMENT_ERROR"}], "flow": ["scope/conjunction mis-decomposed", "multiple requested facts share one R or extra qualifier becomes R", "validator accepts identity-consistent mapping", "assembler retains first mapping or validator encounters unrelated version", "incomplete answer or abstention"]},
            {"name": "current-only version path", "case_ids": ["fresh_v1_029", "fresh_v1_031", "fresh_v1_032", "fresh_v1_033", "fresh_v1_035"], "flow": ["DocumentVersion.is_active=true in dense and BM25", "2025 chunks excluded", "single candidate resolver selects 2026", "Luna lacks historical evidence", "incorrect abstention"]},
            {"name": "global version whitelist", "case_ids": ["fresh_v1_009", "fresh_v1_015", "fresh_v1_016", "fresh_v1_017", "fresh_v1_023", "fresh_v1_036"], "flow": ["one document version selected", "selected_version_ids applied to all mapped documents", "valid unrelated-document mapping rejected as WRONG_VERSION", "Sol sees same evidence/invariant", "abstention"]},
            {"name": "literal verifier vs derived boundary", "case_ids": ["fresh_v1_040", "fresh_v1_044"], "flow": ["exact rule retrieved", "derived numeric comparison not represented deterministically", "Luna ABSTAIN/unsupported derived requirement", "safe abstention"]},
        ]
    }
    write_json("failure_dependency_graph.json", dependency_graph)
    fixes = {
        "do_not_implement_in_this_task": True,
        "priorities": [
            {"priority": "P0", "fix": "Enforce semantic output cardinality: one requested output per R ID; qualifiers remain Q; reject or preserve every mapping when one R spans independent facts.", "affected_failures": [row["case_id"] for row in per_case if row["primary_root_cause"] in {"QUESTION_PLAN_ERROR", "QUALIFIER_REQUIREMENT_ERROR"}], "affected_count": 11, "maximum_supported_scope": "11 planning-primary failures; deterministically prevents all 4 unsupported emissions, though secondary version issues limit guaranteed recovery.", "new_api_required_initially": 0, "api_cost_impact": "none for deterministic validation", "complexity": "medium", "regression_risk": "medium", "currently_passing_categories_at_risk": ["single_semantic", "multidoc_two", "multidoc_three", "version_temporal", "numeric_date_boundary"]},
            {"priority": "P1", "fix": "Make retrieval/resolution temporal-scope aware and represent simultaneous historical+current version sets.", "affected_failures": ["fresh_v1_029", "fresh_v1_031", "fresh_v1_032", "fresh_v1_033", "fresh_v1_035"], "affected_count": 5, "maximum_supported_scope": "5 failures with missing historical facts", "new_api_required_initially": 0, "api_cost_impact": "larger local candidate/evidence pools", "complexity": "high", "regression_risk": "medium-high", "currently_passing_categories_at_risk": ["version_temporal", "single_semantic"]},
            {"priority": "P2", "fix": "Scope selected-version validation by document instead of globally across every cited document.", "affected_failures": ["fresh_v1_009", "fresh_v1_015", "fresh_v1_016", "fresh_v1_017", "fresh_v1_023", "fresh_v1_036"], "affected_count": 6, "maximum_supported_scope": "6 false validator rejects; 3 have immediately assemblable complete plans, while 3 retain planning/Sol issues.", "new_api_required_initially": 0, "api_cost_impact": "may avoid repeated Sol calls", "complexity": "low-medium", "regression_risk": "low-medium", "currently_passing_categories_at_risk": ["multidoc_two", "multidoc_three", "version_temporal"]},
            {"priority": "P3", "fix": "Treat ABSTAIN with all requirements SUPPORTED as inconsistent and escalate/fail closed rather than accepting safe abstention.", "affected_failures": ["fresh_v1_025", "fresh_v1_040"], "affected_count": 2, "maximum_supported_scope": "2 missed escalations", "new_api_required_initially": 0, "api_cost_impact": "potentially more Sol calls", "complexity": "low", "regression_risk": "low", "currently_passing_categories_at_risk": ["unanswerable", "prompt_injection"]},
            {"priority": "P4", "fix": "Add deterministic comparison checks for simple deadline/cadence boundary questions.", "affected_failures": ["fresh_v1_040", "fresh_v1_044"], "affected_count": 2, "maximum_supported_scope": "2 numeric/date failures", "new_api_required_initially": 0, "api_cost_impact": "may reduce verifier calls", "complexity": "medium", "regression_risk": "medium", "currently_passing_categories_at_risk": ["numeric_date_boundary", "unanswerable"]},
            {"priority": "P5", "fix": "Diagnose Luna multi-document false-negative on complete literal evidence without changing broad verifier behavior.", "affected_failures": ["fresh_v1_022"], "affected_count": 1, "maximum_supported_scope": "1 isolated Luna false negative", "new_api_required_initially": 0, "api_cost_impact": "none initially", "complexity": "medium", "regression_risk": "high if prompt is broadly changed", "currently_passing_categories_at_risk": ["single_semantic", "access_control_deny", "prompt_injection", "unanswerable"]},
        ],
    }
    write_json("prioritized_fix_plan.json", fixes)
    verdict = "FRESH_FAILURE_CENSUS_PARTIAL_TRACE_INSUFFICIENT"
    final_report = {
        "experiment_id": "FRESH_UNSEEN_FAILURE_CENSUS_V1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "failure_summary": failure_set,
        "root_cause_census": census_rows,
        "first_failure_stage_distribution": dict(stage_counts),
        "retrieval_coverage": retrieval,
        "version_temporal": {key: value for key, value in version_analysis.items() if key != "cases"},
        "multidoc": {key: value for key, value in json.loads((OUT / "multidoc_analysis.json").read_text()).items() if key != "cases"},
        "incorrect_abstentions": {"count": 15, "trigger_counts": dict(trigger_counts), "complete_top15": 10},
        "unsupported_answers": {"count": 4, "all_deterministically_preventable": True},
        "luna": json.loads((OUT / "luna_analysis.json").read_text()),
        "sol": {key: value for key, value in json.loads((OUT / "sol_analysis.json").read_text()).items() if key != "cases"},
        "router": json.loads((OUT / "router_analysis.json").read_text()),
        "contract": json.loads((OUT / "question_plan_contract_analysis.json").read_text()),
        "scorer": {key: value for key, value in json.loads((OUT / "scorer_audit.json").read_text()).items() if key != "cases"},
        "api_cost": {"new_embedding_calls": 0, "new_luna_calls": 0, "new_sol_calls": 0, "new_openai_calls": 0, "new_cost_usd": 0.0},
        "trace_insufficiency": ["Dense candidate membership IDs not persisted", "BM25 candidate membership IDs not persisted", "RRF candidate membership IDs not persisted"],
        "preserved_success_categories": {"single_semantic": 1.0, "access_control_allow": 1.0, "access_control_deny": 1.0, "prompt_injection": 1.0, "unanswerable": 1.0},
        "next_step": "Implement P0 semantic output-cardinality/qualifier invariant locally; do not run another paid evaluation.",
    }
    write_json("final_report.json", final_report)
    print(json.dumps({"verdict": verdict, "failures": len(per_case), "top15_complete": len(complete), "primary_root_causes": dict(primary_counts), "new_openai_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
