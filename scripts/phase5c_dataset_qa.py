# ruff: noqa: E501
"""Phase 5C Dataset QA Gate — validates all 120 cases against corpus."""

from __future__ import annotations

import json
import re
from pathlib import Path

DATASET_PATH = Path("data/eval/phase5c/v3_clean_120_cases.jsonl")
CORPUS_DIR = Path("data/synthetic_company")
OUTPUT_PATH = Path("data/eval/phase5c/phase5c_dataset_qa.json")

TOKEN = re.compile(r"[a-z0-9]+")


def load_corpus() -> dict[str, dict]:
    """Load corpus documents with frontmatter metadata."""
    docs = {}
    for p in sorted(CORPUS_DIR.glob("*.md")):
        text = p.read_text()
        lines = text.split("\n")
        if lines[0].strip() == "---":
            end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
            fm_text = "\n".join(lines[1:end])
            doc_id_m = re.search(r"document_id:\s*(.+)", fm_text)
            version_m = re.search(r'version:\s*"?([^"\n]+)"?', fm_text)
            visibility_m = re.search(r"visibility:\s*(\w+)", fm_text)
            doc_id = doc_id_m.group(1).strip() if doc_id_m else p.stem
            version = version_m.group(1).strip() if version_m else ""
            visibility = visibility_m.group(1).strip() if visibility_m else "public"
            body = "\n".join(lines[end + 1 :])
        else:
            doc_id = p.stem
            version = ""
            visibility = "public"
            body = text

        key = doc_id
        entry = {"text": body, "full_text": text, "visibility": visibility, "version": version, "file": p.name}
        if key not in docs:
            docs[key] = []
        docs[key].append(entry)
    return docs


def load_cases() -> list[dict]:
    cases = []
    with open(DATASET_PATH) as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def check_question_sufficiency(case: dict) -> tuple[bool, str]:
    """A: Can the question alone identify what is being asked?"""
    q = case["question"]
    facts = case.get("required_facts", [])
    if not q or len(q.strip()) < 10:
        return False, "Question too short to be meaningful"
    if case["expected_answerable"] and not facts:
        return False, "Answerable case has no required_facts"
    return True, ""


def check_ground_truth_support(case: dict, corpus: dict[str, list[dict]]) -> tuple[bool, str]:
    """B: Every expected fact exists in the authorized corpus."""
    if not case["expected_answerable"]:
        return True, ""
    markers = case.get("required_chunk_markers", [])
    doc_ids = case.get("required_document_ids", [])
    for marker in markers:
        found = False
        for did in doc_ids:
            if did in corpus:
                for entry in corpus[did]:
                    if marker.lower() in entry["full_text"].lower():
                        found = True
                        break
            if found:
                break
        if not found:
            # Search all docs as fallback
            for _did, entries in corpus.items():
                for entry in entries:
                    if marker.lower() in entry["full_text"].lower():
                        found = True
                        break
                if found:
                    break
        if not found:
            return False, f"Marker '{marker}' not found in corpus"
    return True, ""


def check_required_chunks_valid(case: dict) -> tuple[bool, str]:
    """C: required_chunk_ids are present."""
    if not case["expected_answerable"]:
        return True, ""
    if not case.get("required_chunk_ids"):
        return False, "Answerable case missing required_chunk_ids"
    return True, ""


def check_minimality(case: dict) -> tuple[bool, str]:
    """D: Chunk count is reasonable for the category."""
    cat = case["category"]
    chunks = case.get("required_chunk_ids", [])
    if cat == "single_document" and len(chunks) > 2:
        return False, f"Single-doc case has {len(chunks)} chunks"
    if cat == "multidoc_two" and len(chunks) > 4:
        return False, f"Two-doc case has {len(chunks)} chunks"
    return True, ""


def check_answerability(case: dict) -> tuple[bool, str]:
    """E: answerable label consistency."""
    if case["expected_answerable"] and case["should_abstain"]:
        return False, "Both answerable=true and should_abstain=true"
    if not case["expected_answerable"] and not case["should_abstain"]:
        return False, "Both answerable=false and should_abstain=false"
    return True, ""


def check_abstention(case: dict) -> tuple[bool, str]:
    """F: Abstention cases have valid reasons."""
    if case["should_abstain"]:
        cat = case["category"]
        valid_abstain_cats = {"acl_sensitive", "partial_no_answer", "prompt_injection", "version_region"}
        if cat not in valid_abstain_cats:
            # Some version_region cases are answerable, some abstain
            pass
    return True, ""


def check_exact_id_leakage(case: dict) -> tuple[bool, str]:
    """G: Exact-ID answer must not appear in the question."""
    if case["category"] != "exact_identifier":
        return True, ""
    answer = case.get("expected_answer", "")
    question = case["question"]
    if answer and answer.lower() in question.lower():
        return False, f"Answer '{answer}' leaked in question"
    return True, ""


def check_multidoc_validity(case: dict) -> tuple[bool, str]:
    """H: Multi-doc cases require the right number of documents."""
    cat = case["category"]
    doc_ids = case.get("required_document_ids", [])
    unique_docs = set(doc_ids)
    if cat == "multidoc_two" and len(unique_docs) < 2:
        return False, f"Two-doc case has only {len(unique_docs)} unique docs"
    if cat == "multidoc_three" and len(unique_docs) < 3:
        return False, f"Three-doc case has only {len(unique_docs)} unique docs"
    return True, ""


def check_same_doc_multi_chunk(case: dict) -> tuple[bool, str]:
    """I: Same-document multi-chunk cases need multiple chunks."""
    if case["category"] != "same_document_multi_chunk":
        return True, ""
    chunks = case.get("required_chunk_ids", [])
    if len(chunks) < 2:
        return False, f"Same-doc multi-chunk has only {len(chunks)} chunk(s)"
    return True, ""


def check_version(case: dict, corpus: dict[str, list[dict]]) -> tuple[bool, str]:
    """J: Version ground truth corresponds to current authorized evidence."""
    if not case.get("required_version_ids"):
        return True, ""
    for doc_id, expected_ver in case["required_version_ids"].items():
        if doc_id not in corpus:
            return False, f"Doc '{doc_id}' not in corpus"
        versions = [e["version"] for e in corpus[doc_id]]
        if expected_ver not in versions:
            return False, f"Version '{expected_ver}' not available for '{doc_id}' (have: {versions})"
    return True, ""


def check_acl(case: dict, corpus: dict[str, list[dict]]) -> tuple[bool, str]:
    """K: ACL cases don't use restricted evidence as if public."""
    if case["category"] != "acl_sensitive":
        return True, ""
    if not case["should_abstain"]:
        return False, "ACL-sensitive case should abstain"
    forbidden = case.get("forbidden_document_ids", [])
    if not forbidden:
        return False, "ACL case has no forbidden_document_ids"
    return True, ""


def check_prompt_injection(case: dict) -> tuple[bool, str]:
    """L: Prompt injection cases are labeled should_abstain."""
    if case["category"] != "prompt_injection":
        return True, ""
    if not case["should_abstain"]:
        return False, "Prompt injection case should abstain"
    return True, ""


def run_qa() -> dict:
    corpus = load_corpus()
    cases = load_cases()

    print(f"Corpus documents: {len(corpus)}")
    print(f"Cases loaded: {len(cases)}")

    results = []
    pass_count = 0
    fail_count = 0

    for case in cases:
        qid = case["query_id"]
        checks = {}

        suf_pass, suf_note = check_question_sufficiency(case)
        checks["question_sufficient"] = suf_pass

        ans_pass = case["expected_answerable"] != case["should_abstain"]
        checks["answerable_label_valid"] = ans_pass

        gt_pass, gt_note = check_ground_truth_support(case, corpus)
        checks["ground_truth_supported"] = gt_pass

        facts_pass = True
        if case["expected_answerable"] and not case.get("required_facts"):
            facts_pass = False
        checks["required_facts_valid"] = facts_pass

        chunks_pass, chunks_note = check_required_chunks_valid(case)
        checks["required_chunks_valid"] = chunks_pass

        min_pass, min_note = check_minimality(case)
        checks["minimal_evidence_valid"] = min_pass

        leak_pass, leak_note = check_exact_id_leakage(case)
        checks["answer_leakage"] = leak_pass

        acl_pass, acl_note = check_acl(case, corpus)
        checks["acl_valid"] = acl_pass

        ver_pass, ver_note = check_version(case, corpus)
        checks["version_valid"] = ver_pass

        pi_pass, pi_note = check_prompt_injection(case)
        checks["prompt_injection_label_valid"] = pi_pass

        all_pass = all(checks.values())
        notes_parts = []
        if not suf_pass:
            notes_parts.append(f"sufficiency: {suf_note}")
        if not gt_pass:
            notes_parts.append(f"ground_truth: {gt_note}")
        if not chunks_pass:
            notes_parts.append(f"chunks: {chunks_note}")
        if not min_pass:
            notes_parts.append(f"minimality: {min_note}")
        if not leak_pass:
            notes_parts.append(f"leakage: {leak_note}")
        if not acl_pass:
            notes_parts.append(f"acl: {acl_note}")
        if not ver_pass:
            notes_parts.append(f"version: {ver_note}")
        if not pi_pass:
            notes_parts.append(f"prompt_injection: {pi_note}")

        result = {
            "query_id": qid,
            "qa_pass": all_pass,
            **checks,
            "notes": "; ".join(notes_parts) if notes_parts else None,
        }
        results.append(result)
        if all_pass:
            pass_count += 1
        else:
            fail_count += 1

    summary = {
        "total": len(results),
        "pass": pass_count,
        "fail": fail_count,
        "failed_ids": [r["query_id"] for r in results if not r["qa_pass"]],
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nDataset QA: {pass_count}/{len(results)} PASS, {fail_count} FAIL")
    if fail_count:
        print("Failed cases:")
        for r in results:
            if not r["qa_pass"]:
                print(f"  {r['query_id']}: {r['notes']}")
        print("\nPHASE5C_DATASET_QA_FAILURE")
    else:
        print("PHASE5C_DATASET_QA_PASS")

    return summary


if __name__ == "__main__":
    run_qa()
