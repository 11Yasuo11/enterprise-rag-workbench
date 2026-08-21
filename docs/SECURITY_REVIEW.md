# Security Review

**Audit date:** 2026-08-21
**Branch:** `chore/repository-hygiene-cleanup`
**Scope:** Secrets, tracking, history, frontend exposure, evaluation leakage, auth/injection, logging, high-risk files.
**Policy:** No secret values are reproduced in this document.

---

## Executive Result

| Gate | Result |
|---|---|
| TRACKED_REAL_SECRETS | **0** |
| HISTORY_UNROTATED_REAL_SECRETS | **0** (gitleaks hits classified as false positives) |
| CLIENT_EXPOSED_SERVER_SECRETS | **0** |
| PRODUCTION_GOLD_LEAKAGE | **0** |
| CASE_ID_RUNTIME_HARDCODING | **0** |
| UNAUTHORIZED_EVIDENCE_PATH | **0** (guards present; no proven bug) |
| DEBUG_SECRET_LEAK | **0** |
| Overall | **SECURITY_REVIEW_PASS** (local untracked keys noted below) |

---

## Working Tree Secret Scan

- Tooling: pattern scan + classification of key-shaped values **without printing values**.
- Tracked sensitive filenames: only `.env.example` (placeholders).
- Local `.env` is **gitignored** and present on the workstation.
- Local `.env` contains **API-key-shaped values** for `EMBEDDING_API_KEY` and `JUDGE_API_KEY` (type: OpenAI-style project key). `OPENAI_API_KEY` is empty. `ALLOW_EXTERNAL_CALLS=false` / `ALLOW_EXTERNAL_JUDGE_CALLS=false`.
- Classification: **LOCAL_UNTRACKED** — not a tracked leak. Do not commit `.env`. Optional hygiene: clear unused local keys before sharing the machine.

Docker Compose uses local placeholder DB credentials (`rag` / `rag`) documented for local-only use.

---

## Git History Secret Scan

- Tool: **gitleaks** (`gitleaks detect`), 20 commits scanned.
- Raw findings: **28**
- Classification: **false positives**
  - SHA-256 integrity hashes in freeze manifests (`*_hash`, script file digests)
  - Freeze IDs such as `RC_FINAL_PRE_API_V*`
  - Synthetic corpus/eval identifiers (e.g., `ATLAS-API-301`-style tokens in builders/datasets)
- No `sk-` / OpenAI-style credential rule hits in history.
- `history_secret_findings_count` (real credentials): **0**

---

## Environment Files

| File | Tracked? | Notes |
|---|---|---|
| `.env` | No (gitignore) | Local keys may exist; keep untracked |
| `.env.example` | Yes | Placeholders only; safe |
| `.env.local` / `.env.production` | Not present | — |

`.gitignore` covers `.env`, `.env.*` with `!.env.example`.

---

## Frontend Exposure

- Only browser-exposed env usage found: `NEXT_PUBLIC_API_URL` → API base URL (`apps/web/lib/api.ts`).
- No `NEXT_PUBLIC_*` provider secrets.
- **CLIENT_EXPOSED_SERVER_SECRETS = 0**

---

## Evaluation Leakage

- `src/rag_workbench/runtime/` has **no** matches for `gold`, `expected_answer`, `expected_route`, `should_abstain`, `fresh_v`, or case-id branching.
- Gold / expected_* appear in **evaluation / experiment / scorer** code paths (legitimate).
- Atomic requirement contract explicitly documents that gold/case IDs are not inputs to serving logic.
- **PRODUCTION_GOLD_LEAKAGE = 0**, **CASE_ID_RUNTIME_HARDCODING = 0**

Artifacts: `data/experiments/interview-final-audit/evaluation_leakage_audit.json`

---

## Authorization

- Canonical runtime applies principal-scoped retrieval and gates evidence before packets / Luna / Sol / assembler / citations (`canonical_runtime.py`, retrieval filters).
- Citation authorization tests exist under unit tests.
- No unauthorized-evidence path bug proven in this audit.

---

## Prompt Injection

- `is_question_injection_v2` runs **before** retrieval/answer routing in `CanonicalRagRuntime.query`.
- Injection cases are hard-abstained with `error_class=PROMPT_INJECTION`.
- Unit coverage: `test_prompt_injection_safety_v13.py`, `test_question_injection_guard_v2.py`.
- Status: **PASS** (do not claim universal novel-attack immunity).

---

## Logging

- `/rag/query` logs request_id, route, status, error_class, latency — not API keys or full prompts/secrets.
- Health failures log exception **type**, not stack traces to clients.
- Debug traces require `include_debug` + `RAG_ADMIN_DEBUG`.
- Request IDs are UUIDs (safe correlation tokens).

---

## High-Risk Files

| Class | Result |
|---|---|
| Tracked `.env` / PEM / SSH keys / DB dumps | **SAFE** (none found tracked) |
| Customer production data | **SAFE** (synthetic AcmeAI corpus) |
| Local browser/session dumps | **SAFE** |
| Root one-off `export_audit.py` | **REVIEW** (large export utility; excluded from Ruff product gate; no secrets found in scan classification) |

---

## Findings

1. **F1** — Gitleaks noise on SHA256 / synthetic IDs → false positive; no rotation.
2. **F2** — Local untracked API-key-shaped values in `.env` → keep ignored; optional clear/rotate if the workstation is shared.
3. **F3** — Historical docs may still mention Top-5 research windows → documentation drift risk, not a secret issue (addressed in interview docs).

---

## Required Actions

| Action | Owner | Blocking? |
|---|---|---|
| Keep `.env` untracked | Maintainer | Yes (ongoing) |
| Optional: clear local unused provider keys before machine sharing | Maintainer | No |
| Do not rewrite git history for false-positive hashes | Maintainer | N/A |
| Prefer README / interview cheat sheet over V3 Top-5 cheat sheet | Interviewer prep | No |

**No ROTATION_REQUIRED for tracked or historical real credentials.**
