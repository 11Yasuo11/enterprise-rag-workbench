from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("historical_overlap_audit.json")
QUESTIONS = (
    ROOT
    / ".local/evaluation-input/fresh_unseen_eval_v1_1_20260820"
    / "fresh_unseen_questions_v1_1.json"
)


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def normalize_ids(text: str) -> str:
    text = normalize(text)
    text = re.sub(
        r"\b(?:p5\w*|phase\w*|v\d+|r\d+|q\d+|\d+(?:st|nd|rd|th)?)\b",
        "<id>",
        text,
    )
    return re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)


def template(text: str) -> str:
    stop = {
        "what", "which", "who", "when", "where", "why", "how", "is", "are",
        "was", "were", "the", "a", "an", "for", "from", "under", "both", "and",
        "or", "of", "to", "in", "on", "with", "without", "asks", "ask", "provide",
        "give", "state", "employee", "employees", "current", "policy", "document",
        "according",
    }
    return " ".join(
        token if token in stop or token in {"<id>", "<num>"} else "<slot>"
        for token in normalize_ids(text).split()
    )


def grams(text: str, size: int = 4) -> set[str]:
    value = " " + normalize_ids(text) + " "
    return {value[index : index + size] for index in range(max(0, len(value) - size + 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left | right else 1.0


def visit(value: object, source: str, rows: list[tuple[str, str | None, str]]) -> None:
    if isinstance(value, dict):
        question = value.get("question") or value.get("query")
        if isinstance(question, str) and len(question.strip()) >= 8:
            case_id = value.get("query_id") or value.get("case_id") or value.get("id")
            rows.append((question, str(case_id) if case_id is not None else None, source))
        for child in value.values():
            if isinstance(child, (dict, list)):
                visit(child, source, rows)
    elif isinstance(value, list):
        for child in value:
            visit(child, source, rows)


def historical_paths() -> list[Path]:
    paths = list((ROOT / "data/eval").rglob("*.json"))
    paths += list((ROOT / "data/eval").rglob("*.jsonl"))
    for path in (ROOT / "data/experiments").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        value = str(path)
        if "/result_cache/" in value or "fresh-unseen-evaluation-v1" in value:
            continue
        if any(
            marker in path.name.casefold()
            for marker in ("case", "result", "trace", "replay", "dataset")
        ):
            paths.append(path)
    return sorted(set(paths))


def load_historical(paths: list[Path]) -> list[dict[str, object]]:
    raw: list[tuple[str, str | None, str]] = []
    for path in paths:
        try:
            if path.suffix == ".jsonl":
                for line in path.read_text(errors="ignore").splitlines():
                    if line.strip():
                        visit(json.loads(line), str(path.relative_to(ROOT)), raw)
            else:
                visit(json.loads(path.read_text(errors="ignore")), str(path.relative_to(ROOT)), raw)
        except (OSError, json.JSONDecodeError):
            continue
    unique: dict[str, tuple[str, str | None, str]] = {}
    for question, case_id, source in raw:
        unique.setdefault(normalize(question), (question, case_id, source))
    rows = []
    for question, case_id, source in unique.values():
        normalized_ids = normalize_ids(question)
        rows.append(
            {
                "question": question,
                "case_id": case_id,
                "source": source,
                "normalized": normalize(question),
                "normalized_ids": normalized_ids,
                "tokens": set(normalized_ids.split()),
                "grams": grams(question),
                "template": template(question),
            }
        )
    return rows


def main() -> None:
    fresh = json.loads(QUESTIONS.read_text())["cases"]
    paths = historical_paths()
    historical = load_historical(paths)
    cases = []
    for case in fresh:
        question = case["question"]
        normalized = normalize(question)
        normalized_ids = normalize_ids(question)
        tokens = set(normalized_ids.split())
        char_grams = grams(question)
        question_template = template(question)
        cheap = []
        exact_matches = []
        for neighbor in historical:
            exact = normalized == neighbor["normalized"]
            id_exact = normalized_ids == neighbor["normalized_ids"]
            token_score = jaccard(tokens, neighbor["tokens"])
            char_score = jaccard(char_grams, neighbor["grams"])
            cheap_score = 0.45 * token_score + 0.55 * char_score
            item = (cheap_score, exact, id_exact, token_score, char_score, neighbor)
            cheap.append(item)
            if exact or id_exact:
                exact_matches.append(item)
        candidates = exact_matches + sorted(cheap, key=lambda item: item[0], reverse=True)[:40]
        detailed = []
        seen = set()
        for _, exact, id_exact, token_score, char_score, neighbor in candidates:
            key = (neighbor["source"], neighbor["case_id"], neighbor["question"])
            if key in seen:
                continue
            seen.add(key)
            sequence = SequenceMatcher(None, normalized_ids, neighbor["normalized_ids"]).ratio()
            template_score = SequenceMatcher(None, question_template, neighbor["template"]).ratio()
            composite = (
                0.40 * sequence
                + 0.25 * token_score
                + 0.25 * char_score
                + 0.10 * template_score
            )
            detailed.append(
                {
                    "source": neighbor["source"],
                    "case_id": neighbor["case_id"],
                    "question": neighbor["question"],
                    "exact": exact,
                    "id_exact": id_exact,
                    "scores": {
                        "composite": round(composite, 6),
                        "sequence": round(sequence, 6),
                        "token_jaccard": round(token_score, 6),
                        "character_4gram_jaccard": round(char_score, 6),
                        "template_sequence": round(template_score, 6),
                    },
                }
            )
        detailed.sort(key=lambda item: item["scores"]["composite"], reverse=True)
        best = detailed[0]
        suspicious = bool(
            best["exact"]
            or best["id_exact"]
            or best["scores"]["composite"] >= 0.86
            or (
                best["scores"]["sequence"] >= 0.94
                and best["scores"]["character_4gram_jaccard"] >= 0.82
            )
            or (
                best["scores"]["composite"] >= 0.80
                and best["scores"]["template_sequence"] >= 0.92
            )
        )
        cases.append(
            {
                "case_id": case["case_id"],
                "normalized_exact_duplicate": any(item["exact"] for item in detailed),
                "ids_numbers_normalized_exact_duplicate": any(
                    item["id_exact"] for item in detailed
                ),
                "suspicious_near_duplicate": suspicious,
                "nearest_historical_neighbors": detailed[:5],
            }
        )
    exact_ids = [
        case["case_id"]
        for case in cases
        if case["normalized_exact_duplicate"]
        or case["ids_numbers_normalized_exact_duplicate"]
    ]
    suspicious_ids = [case["case_id"] for case in cases if case["suspicious_near_duplicate"]]
    verdict = (
        "FRESH_EVAL_DATASET_OVERLAP_REQUIRES_REPLACEMENT"
        if exact_ids or len(suspicious_ids) > 5
        else "PASS"
    )
    replacement = next(case for case in cases if case["case_id"] == "fresh_v1_048")
    report = {
        "experiment_id": "FRESH_UNSEEN_EVALUATION_V1_1",
        "dataset_id": "acmeai-fresh-unseen-eval-v1.1-20260820",
        "method": {
            "historical_file_count": len(paths),
            "historical_unique_question_count": len(historical),
            "sources": "All data/eval JSON/JSONL and historical case/result/trace/replay/dataset artifacts; provider caches and fresh-evaluation artifacts excluded.",
            "candidate_selection": "Exact matches plus top 40 by token/character similarity, then full sequence/template scoring.",
            "checks": [
                "normalized exact duplicates",
                "ID/number-normalized duplicates",
                "high lexical similarity",
                "character 4-gram similarity",
                "suspicious template reuse",
            ],
            "similarity": "0.40 sequence + 0.25 token Jaccard + 0.25 character 4-gram Jaccard + 0.10 template sequence",
            "suspicious_threshold": "Composite >= 0.86; sequence >= 0.94 with char >= 0.82; or composite >= 0.80 with template >= 0.92.",
            "paid_api_calls": 0,
        },
        "case_count": 60,
        "generalization_scope": "Fresh question-level generalization over the existing corpus.",
        "underlying_corpus_fact_reuse_allowed": True,
        "exact_or_id_swapped_duplicate_count": len(exact_ids),
        "exact_or_id_swapped_duplicate_case_ids": exact_ids,
        "suspicious_near_duplicate_count": len(suspicious_ids),
        "suspicious_near_duplicate_case_ids": suspicious_ids,
        "replacement_case_audit": replacement,
        "gate_passed": verdict == "PASS",
        "verdict": verdict,
        "cases": cases,
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "historical_unique_question_count": len(historical),
                "exact": exact_ids,
                "suspicious": suspicious_ids,
                "replacement_case_audit": replacement,
                "verdict": verdict,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
