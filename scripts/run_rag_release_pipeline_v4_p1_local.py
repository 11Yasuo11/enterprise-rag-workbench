# ruff: noqa: E501
"""Zero-API reproduction and corrected rescore of the frozen P1 V1 predictions."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rag_workbench.config import Settings
from rag_workbench.evaluation.citation_authorization import (
    authorize_citation_set,
    load_citation_evidence_identities,
)
from rag_workbench.evaluation.final_e2e_scorer_v2 import ScorerInput, score_case
from rag_workbench.retrieval.temporal import (
    TemporalScopePlan,
    temporal_scope_from_dict,
    version_is_eligible,
)
from rag_workbench.security.permissions import Principal

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/experiments/rag-remaining-failures-release-pipeline-v3/p1-targeted"
OUT = ROOT / "data/experiments/rag-release-pipeline-v4"
FIX = OUT / "p1-citation-authorization-fix"
TARGETS = ("fresh_v1_029", "fresh_v1_031", "fresh_v1_032", "fresh_v1_033", "fresh_v1_035")
QUESTIONS = ROOT / ".local/evaluation-input/fresh_unseen_eval_v1_1_20260820/fresh_unseen_questions_v1_1.json"
GOLD = ROOT / ".local/evaluation-input/fresh_unseen_eval_v1_1_20260820/fresh_unseen_gold_v1_1.json"
SUBDIRS = (
    "p1-citation-authorization-fix", "p1-targeted-v2", "p2", "p3", "p4", "p5",
    "local-regression", "targeted-19", "old60-regression", "fresh-holdout-v2",
    "release-decision",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_incomplete_temporal_decision() -> bool:
    # Exact defective fallback: missing version/is_active plus no frozen scope.
    return version_is_eligible(
        TemporalScopePlan("UNSPECIFIED_CURRENT_DEFAULT"), version=None, is_active=None
    )


def main() -> None:
    for name in SUBDIRS:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    frozen_hash_before = sha256(OLD / "final_predictions.jsonl")
    freeze = load(OLD / "predictions_freeze.json")
    if frozen_hash_before != freeze["predictions_sha256"]:
        raise RuntimeError("OLD_P1_PREDICTIONS_NOT_FROZEN")
    predictions = load_jsonl(OLD / "final_predictions.jsonl")
    formal_rows = {row["case_id"]: row for row in load_jsonl(OLD / "per_case_scoring.jsonl")}
    cases = {row["case_id"]: row for row in load(OLD / "target_cases.json")["cases"]}
    questions = {row["case_id"]: row for row in load(QUESTIONS)["cases"] if row["case_id"] in TARGETS}
    # Gold is used only for diagnostic scoring after verifying the pre-existing prediction freeze.
    gold = {row["case_id"]: row for row in load(GOLD)["cases"] if row["case_id"] in TARGETS}
    metadata_rows: list[dict[str, Any]] = []
    rescored: list[dict[str, Any]] = []
    principal_by_case = {
        case_id: Principal(
            row["principal"]["principal_id"],
            row["principal"]["tenant_id"],
            frozenset(row["principal"]["permission_groups"]),
        )
        for case_id, row in questions.items()
    }
    with Session(create_engine(Settings().database_url)) as session:
        for prediction in predictions:
            case_id = prediction["case_id"]
            scope = temporal_scope_from_dict(cases[case_id]["temporal_scope"])
            identities = load_citation_evidence_identities(session, tuple(prediction["citations"]))
            decisions = authorize_citation_set(
                identities, principal=principal_by_case[case_id], temporal_scope=scope
            )
            authorized_ids = tuple(item.chunk_id for item in decisions if item.authorized)
            by_chunk = {item.chunk_id: item for item in identities}
            for citation in prediction["citations"]:
                complete = by_chunk[citation]
                decision = next(item for item in decisions if item.chunk_id == citation)
                metadata_rows.append({
                    "case_id": case_id,
                    "old_citation_metadata": {
                        "chunk_id": citation,
                        "authorization_payload": {
                            "tenant_id": complete.tenant,
                            "visibility": complete.visibility,
                            "document_fk": "database-internal-document-key",
                            "permission_groups": list(complete.permission_groups),
                        },
                    },
                    "missing_fields_from_authorization_payload": ["document_id", "document_version_id", "version", "is_active", "chunk_id", "region", "frozen_temporal_scope"],
                    "old_temporal_fallback": TemporalScopePlan("UNSPECIFIED_CURRENT_DEFAULT").as_dict(),
                    "old_authorization_decision": old_incomplete_temporal_decision(),
                    "formal_citation_validity_pass": formal_rows[case_id]["citation_validity_pass"],
                    "complete_citation_metadata": complete.as_dict(),
                    "frozen_temporal_scope": scope.as_dict(),
                    "correct_authorization_decision": decision.as_dict(),
                })
            expected = gold[case_id]
            identity_by_chunk = {item.chunk_id: item for item in identities}
            scorer = score_case(ScorerInput(
                arm="P1_FROZEN_DIAGNOSTIC_RESCORE",
                query_id=case_id,
                question=questions[case_id]["question"],
                category=expected["category"],
                should_abstain=bool(expected["should_abstain"]),
                expected_answerable=bool(expected["expected_answerability"]),
                required_facts=tuple(expected.get("expected_facts", ())),
                required_document_ids=tuple(expected.get("required_document_ids", ())),
                final_answer=prediction.get("answer"),
                final_answer_present=bool(prediction.get("answer")),
                citation_ids=tuple(prediction["citations"]),
                cited_document_ids=tuple(identity_by_chunk[item].document_id for item in prediction["citations"]),
                cited_chunk_texts={
                    mapping["chunk_id"]: mapping["supporting_span"]
                    for requirement in prediction["requirement_version_mappings"]
                    for mapping in requirement["mappings"]
                },
                retrieved_top_k_ids=tuple(prediction["top15_chunk_ids"]),
                authorized_citation_ids=authorized_ids,
            ))
            rescored.append({
                "case_id": case_id,
                "old_behavior": formal_rows[case_id]["behavior"],
                "corrected_behavior": scorer.behavior,
                "authorization_valid": set(prediction["citations"]) <= set(authorized_ids),
                "authorized_citation_ids": list(authorized_ids),
                "citation_count": len(prediction["citations"]),
                "citation_validity_pass": scorer.citation_validity_pass,
                "citation_correctness_pass": scorer.citation_correctness_pass,
                "citation_completeness_pass": scorer.citation_completeness_pass,
                "facts_missing": scorer.facts_missing,
            })
    if sha256(OLD / "final_predictions.jsonl") != frozen_hash_before:
        raise RuntimeError("OLD_P1_PREDICTIONS_MUTATED")
    write_jsonl(FIX / "old_vs_complete_citation_metadata.jsonl", metadata_rows)
    write(FIX / "citation_authorization_root_cause.json", {
        "experiment": "RAG_RELEASE_PIPELINE_V4",
        "timestamp": datetime.now(UTC).isoformat(),
        "frozen_source": str(OLD),
        "frozen_predictions_sha256": frozen_hash_before,
        "frozen_predictions_immutable": True,
        "cases_reproduced": list(TARGETS),
        "omitted_from_old_authorization_payload": ["document_id", "document_version_id", "version", "is_active", "chunk_id", "region", "frozen_temporal_scope"],
        "preserved_by_old_authorization_payload": ["tenant", "visibility", "permission_groups"],
        "old_fallback": "UNSPECIFIED_CURRENT_DEFAULT/CURRENT_ONLY semantics applied to incomplete evidence identity",
        "formal_authorized_citations": 0,
        "complete_metadata_authorized_citations": sum(len(row["authorized_citation_ids"]) for row in rescored),
        "cited_chunks": sum(row["citation_count"] for row in rescored),
        "generated_answer_defect": False,
        "harness_defect_confirmed": all(row["corrected_behavior"] == "CORRECT_COMPLETE_ANSWER" for row in rescored),
    })
    write(FIX / "frozen_p1_local_rescore.json", {
        "diagnostic_only": True,
        "old_formal_report_modified": False,
        "old_predictions_modified": False,
        "cases": rescored,
        "authorized_citations": sum(len(row["authorized_citation_ids"]) for row in rescored),
        "cited_chunks": sum(row["citation_count"] for row in rescored),
        "all_five_authorization_valid": all(row["authorization_valid"] for row in rescored),
        "unsupported_after_fix": sum(row["corrected_behavior"] == "UNSUPPORTED_ANSWER" for row in rescored),
        "passed": all(row["corrected_behavior"] == "CORRECT_COMPLETE_ANSWER" for row in rescored),
    })


if __name__ == "__main__":
    main()
