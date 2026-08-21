# Interview Repository Audit

First-look audit as if an interviewer opened this repo cold.

**Date:** 2026-08-21
**Branch:** `chore/repository-hygiene-cleanup`
**Scope:** Readability / navigation only (no RAG quality tuning).

---

## Verdicts (30s / 2m questions)

| Question | Verdict | Notes |
|---|---|---|
| 1. Understand what it does in 30s? | **PASS** | README “What It Is” + grounded/auth/version/abstention focus |
| 2. Understand architecture in 2 minutes? | **PASS** | README Mermaid + `docs/PRODUCTION_RAG_ARCHITECTURE.md` |
| 3. Identify interesting decisions? | **PASS** | Hybrid/RRF/CE Top-15, requirements, deterministic support, Luna/Sol hierarchy |
| 4. Understand evaluation? | **PASS** | README Results separates regression vs historical holdouts |
| 5. Historical vs current? | **PASS*** | README + production docs clear; older V3/cheat sheets still say Top-5 — treat as historical |
| 6. Is it deployed? | **PASS** | Explicit: integrated/validated; **not** externally deployed |
| 7. Run locally without reverse engineering? | **PASS** | `docker compose up`, `.env.example`, demo runbook + ingest |

\*Minor residual risk: `docs/V3_RAG_INTERVIEW_CHEAT_SHEET.md` still describes historical Top-5 path — prefer README / new interview cheat sheet.

---

## What an interviewer should open first

1. `README.md`
2. `docs/INTERVIEW_CHEAT_SHEET.md`
3. `docs/INTERVIEW_DEMO_RUNBOOK.md`
4. `docs/PRODUCTION_RAG_ARCHITECTURE.md`
5. `src/rag_workbench/runtime/canonical_runtime.py`

Avoid leading with `data/experiments/` dumps unless asked for evidence.

---

## Navigation map

| Need | Location |
|---|---|
| Current architecture | README, `PRODUCTION_RAG_ARCHITECTURE.md`, `CURRENT_SERVING_RAG_ARCHITECTURE.md` |
| Why decisions exist | `ARCHITECTURE_DECISIONS.md` (CURRENT banner), failure analysis |
| Latest regression evidence | `data/experiments/rag-release-pipeline-v14/` |
| Closed V3 research | `V3_RAG_RESEARCH_POSTMORTEM_AND_INTERVIEW_GUIDE.md` |
| Local demo | `INTERVIEW_DEMO_RUNBOOK.md` |
| Security | `SECURITY_REVIEW.md` |

---

## Remaining readability risks (non-blocking)

1. Large frozen experiment trees under `data/experiments/` — evidence, not the narrative.
2. Historical docs (V3 cheat sheet, older failure analyses) still mention Top-5 as the research window.
3. Local hashing demo ≠ production embedding quality path — must be stated verbally.

## Overall

**Repository understandable: PASS**
