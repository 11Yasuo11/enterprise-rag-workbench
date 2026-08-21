# Repository Quality Status

Snapshot after repository-hygiene cleanup on branch `chore/repository-hygiene-cleanup`.

## Tests

- **pytest:** 338 passed
- Warning (non-blocking): Starlette/`httpx` TestClient deprecation (see below)

## Python Lint

- **Ruff (`ruff check .`):** PASS — 0 findings
- Baseline before this hygiene pass: **155** findings (F401, E501, I001, F841, SIM*, B007, E741, UP015)
- Lint configuration in `pyproject.toml` was **not** weakened

## Database

- **`alembic upgrade head`:** PASS
- **`alembic check`:** PASS (no new upgrade operations detected)

## Web

- **`npm --prefix apps/web run typecheck`:** PASS
- **`npm --prefix apps/web run build`:** PASS

## Known Non-Blocking Warnings

### Starlette / httpx TestClient deprecation

- **Status:** `NON_BLOCKING_DEPENDENCY_DEPRECATION_WARNING`
- **Source:** FastAPI re-exports Starlette `TestClient`; Starlette 1.6 warns that `httpx` is deprecated for the test client and recommends `httpx2`
- **Why not fixed here:** Removing the warning requires adding/upgrading `httpx2` (dependency / lockfile change). Out of scope for hygiene-only cleanup.

## Research Status

- **Current executing quality+serving:** V14 packet/year-token logic inside `CanonicalRagRuntime` (Web + Eval)
- **Latest V2 regression:** 60/60 after V14 (`data/experiments/rag-release-pipeline-v14/`) — **regression set, not unseen**
- **Production finalization:** `READY_FOR_PRODUCTION_DEPLOYMENT_REVIEW` (not deployed)
- **V3 research:** closed — not promoted (`V3_QUALITY_IMPROVEMENT_NOT_CONFIRMED`)
- **Historical V2 release baseline:** `main` / tag `v2.0.0`

## Remaining Product Limitations

These are **RAG RESEARCH LIMITATIONS**, not repository code-quality failures:

| Limitation | Evidence |
|---|---|
| Prompt Injection targeted holdout | **14/20** safe (6 unsafe) |
| Three-document Final category | **0%** |
| Version-sensitive Final category | **0%** |
| Numeric/date Final category | **50%** |
| Incorrect abstentions | **27** |

## Hygiene Scope Note

This cleanup changed formatting, imports, unused locals, and equivalent lint-safe syntax only. Retrieval, ranking, Judge, generator, safety, datasets, and recorded experiment metrics were not modified.

## Interview polish docs

- [`ARCHITECTURE_DECISIONS.md`](ARCHITECTURE_DECISIONS.md)
- [`RAG_FAILURE_ANALYSIS.md`](RAG_FAILURE_ANALYSIS.md)
- [`INTERVIEW_DEMO_RUNBOOK.md`](INTERVIEW_DEMO_RUNBOOK.md)
- Readiness script: `./scripts/interview_readiness_check.sh`
