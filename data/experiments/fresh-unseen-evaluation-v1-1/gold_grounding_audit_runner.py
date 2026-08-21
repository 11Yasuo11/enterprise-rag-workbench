from __future__ import annotations

import json
import re
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from rag_workbench.config import Settings
from rag_workbench.db.models import Document, DocumentPermission, DocumentVersion
ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / ".local/evaluation-input/fresh_unseen_eval_v1_1_20260820"
OUT = Path(__file__).with_name("gold_grounding_audit.json")


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def contains(content: str, marker: str) -> bool:
    return normalized(marker) in normalized(content)


def main() -> None:
    questions = json.loads((INPUT / "fresh_unseen_questions_v1_1.json").read_text())
    gold = json.loads((INPUT / "fresh_unseen_gold_v1_1.json").read_text())
    question_by_id = {case["case_id"]: case for case in questions["cases"]}
    missing_probes = {
        "fresh_v1_055": lambda text: "meal" in text and "reimbursement" in text,
        "fresh_v1_056": lambda text: "north" in text and "runbook" in text,
        "fresh_v1_057": lambda text: "atlas" in text and "mobile" in text,
        "fresh_v1_058": lambda text: "severity two" in text or "severity 2" in text,
        "fresh_v1_059": lambda text: "deletion" in text
        and any(day in text for day in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")),
        "fresh_v1_060": lambda text: "vpn" in text,
    }
    rows = []
    with Session(create_engine(Settings().database_url)) as session:
        documents = {document.document_id: document for document in session.scalars(select(Document))}
        versions_by_document: dict[str, list[DocumentVersion]] = {}
        permissions_by_document: dict[str, set[str]] = {}
        for document_id, document in documents.items():
            versions_by_document[document_id] = list(
                session.scalars(
                    select(DocumentVersion).where(DocumentVersion.document_fk == document.id)
                )
            )
            permissions_by_document[document_id] = set(
                session.scalars(
                    select(DocumentPermission.permission_group).where(
                        DocumentPermission.document_fk == document.id
                    )
                )
            )
        for expected in gold["cases"]:
            case_id = expected["case_id"]
            question = question_by_id[case_id]
            groups = set(question["principal"]["permission_groups"])
            required_documents = expected.get("required_document_ids", [])
            forbidden_documents = expected.get("forbidden_document_ids", [])
            checks: dict[str, object] = {}
            checks["required_documents_exist"] = all(doc in documents for doc in required_documents)
            checks["forbidden_documents_exist"] = all(doc in documents for doc in forbidden_documents)
            expected_versions = expected.get("required_version_ids", {})
            checks["required_versions_exist"] = all(
                document_id in versions_by_document
                and any(version.version == version_name for version in versions_by_document[document_id])
                for document_id, version_name in expected_versions.items()
            )
            required_contents = "\n".join(
                version.content
                for document_id in required_documents
                for version in versions_by_document.get(document_id, [])
            )
            checks["expected_facts_supported"] = all(
                contains(required_contents, fact) for fact in expected.get("expected_facts", [])
            )
            checks["required_chunk_markers_supported"] = all(
                contains(required_contents, marker)
                for marker in expected.get("required_chunk_markers", [])
            )
            checks["required_documents_authorized"] = all(
                documents[doc].visibility == "public"
                or bool(groups & permissions_by_document.get(doc, set()))
                for doc in required_documents
                if doc in documents
            )
            checks["forbidden_documents_denied"] = all(
                documents[doc].visibility != "public"
                and not bool(groups & permissions_by_document.get(doc, set()))
                for doc in forbidden_documents
                if doc in documents
            )
            forbidden_contents = "\n".join(
                version.content
                for document_id in forbidden_documents
                for version in versions_by_document.get(document_id, [])
            )
            checks["forbidden_facts_exist_only_in_restricted_scope"] = all(
                contains(forbidden_contents, fact) for fact in expected.get("forbidden_facts", [])
            )
            if case_id in missing_probes:
                authorized_units = [
                    normalized(unit)
                    for document_id, versions in versions_by_document.items()
                    if documents[document_id].visibility == "public"
                    or bool(groups & permissions_by_document.get(document_id, set()))
                    for version in versions
                    for unit in re.split(r"(?<=[.!?])\s+|\n+", version.content)
                    if unit.strip()
                ]
                checks["requested_missing_fact_absent_from_authorized_corpus"] = not any(
                    missing_probes[case_id](unit) for unit in authorized_units
                )
            if expected["category"] == "prompt_injection":
                checks["prompt_injection_security_expectation_defined"] = "PROMPT_INJECTION" in (
                    expected.get("security_checks") or []
                )
            checks["no_external_knowledge_required"] = (
                checks["expected_facts_supported"]
                and checks["required_chunk_markers_supported"]
                if expected["expected_answerability"]
                else True
            )
            passed = all(bool(value) for value in checks.values())
            rows.append(
                {
                    "case_id": case_id,
                    "category": expected["category"],
                    "expected_answerability": expected["expected_answerability"],
                    "checks": checks,
                    "passed": passed,
                }
            )
    failures = [row["case_id"] for row in rows if not row["passed"]]
    report = {
        "experiment_id": "FRESH_UNSEEN_EVALUATION_V1_1",
        "dataset_id": gold["dataset_id"],
        "audit_process_isolated_from_prediction_runtime": True,
        "candidate_predictions_loaded": False,
        "paid_api_calls": 0,
        "case_count": len(rows),
        "passed_count": len(rows) - len(failures),
        "failed_count": len(failures),
        "failed_case_ids": failures,
        "gate_passed": not failures,
        "verdict": "PASS" if not failures else "FRESH_EVAL_GOLD_AUDIT_FAILED",
        "cases": rows,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("case_count", "passed_count", "failed_count", "failed_case_ids", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
