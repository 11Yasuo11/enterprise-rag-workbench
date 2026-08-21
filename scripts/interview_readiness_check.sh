#!/usr/bin/env bash
# Interview readiness check — deterministic gates only.
# Does NOT run paid LLM inference, research benchmarks, or data mutations
# beyond alembic check (schema drift detection).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PASS=0
FAIL=0

run_step() {
  local name="$1"
  shift
  echo ""
  echo "=== ${name} ==="
  if "$@"; then
    echo "PASS: ${name}"
    PASS=$((PASS + 1))
  else
    echo "FAIL: ${name}"
    FAIL=$((FAIL + 1))
  fi
}

echo "Enterprise RAG Workbench — interview readiness check"
echo "Root: ${ROOT}"

run_step "ruff" uv run ruff check .
run_step "pytest" uv run pytest -q
run_step "alembic check" uv run alembic check
run_step "web typecheck" npm --prefix apps/web run typecheck
run_step "web build" npm --prefix apps/web run build

echo ""
echo "========================================"
echo "Summary: ${PASS} passed, ${FAIL} failed"
if [[ "${FAIL}" -eq 0 ]]; then
  echo "OVERALL: PASS"
  exit 0
fi
echo "OVERALL: FAIL"
exit 1
