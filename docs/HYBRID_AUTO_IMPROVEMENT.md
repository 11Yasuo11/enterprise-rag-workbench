# Hybrid Auto-Improvement Loop (Zero API Mode)

## Repository audit

### ALREADY_IMPLEMENTED

- Frozen Phase-5KR dataset and immutable dataset hashing
- Phase5 retrieval pipeline: Dense, BM25, RRF, local CrossEncoder
- Common stage trace and document-level failure evidence
- Deterministic metrics, failure census, slices, comparison, and promotion gate
- Cache-only Experiment Runner with per-case checkpoint/resume

### REUSABLE

- `Phase5CachedRunner` and `execute_experiment`
- Existing embedding and Judge caches, read-only on cache hit
- Pointwise and one-chunk-per-document selectors
- Existing frozen max-two-chunks-per-document selector
- Current trace-enabled baseline in `artifacts/eval/current-baseline`

### NEEDS_EXTENSION

- Connect the existing experiment lifecycle to diagnosis and bounded candidate selection
- Expose the existing max-two selector through the common runner
- Rank candidates by target-stage structural objectives when quality metrics are unavailable

### MISSING BEFORE IMPLEMENTATION

- Failure-to-strategy mapping and deterministic known-problem guard
- Algorithmic search abstraction and candidate config generation
- Disabled LLM planner extension point
- Zero-API controller, CLI, run-level artifacts, and best-structural-candidate report

## Active architecture

```text
Failure census + traces
          ↓
RuleBasedPlanner / Diagnosis Engine
          ↓
Known deterministic stage problem?
     ├─ no  → NEEDS_HUMAN_OR_LLM_HYPOTHESIS
     └─ yes → GridSearchPlanner (maximum 3 candidates)
                         ↓
              Existing Experiment Runner
                         ↓
              Deterministic evaluation
                         ↓
                 Regression gate
                         ↓
             BEST_STRUCTURAL_CANDIDATE
                         ↓
                  MANUAL_REVIEW
```

`HypothesisPlanner` and `CandidateSearch` are extension protocols. `LLMPlanner` exists only as a
disabled fail-closed stub and raises `LLM_PLANNER_DISABLED`. Random, Bayesian, or TPE search can be
added through `CandidateSearch`; the only enabled search implementation is bounded grid search.

## Safety invariants

- `llm_planner.enabled: false`
- `semantic_judge.enabled: false`
- `paid_api_fallback.enabled: false`
- `auto_promote.enabled: false`
- `max_iterations: 1`
- No cache write or external fallback on an embedding/Judge cache miss
- Unknown or semantic failures stop as `NEEDS_HUMAN_OR_LLM_HYPOTHESIS`
- Changed Judge inputs produce null quality metrics and `MANUAL_REVIEW`
- Generated candidates differ from the baseline in exactly one RAG behavior field

## Current supported ranking search

Supported:

- `reranking.top_k`: 5 → 7
- `reranking.top_k`: 5 → 10
- `reranking.selection_strategy`: pointwise → max-two-chunks-per-document

The already-tested one-chunk-per-document strategy is supported by the runner but excluded from this
search because it did not reduce the target three-document failure bucket. Arbitrary per-document
caps and `diversity_weight` are unsupported and are not fabricated.

## Run

```bash
uv run python scripts/run_auto_improvement.py \
  --baseline artifacts/eval/current-baseline \
  --planner rule_based \
  --llm-planner off \
  --semantic-judge off \
  --max-candidates 3 \
  --max-iterations 1
```

The controller never changes `current-baseline` or production configuration. Each candidate retains
the normal experiment artifacts under `artifacts/eval/`; the controller summary is written under
`artifacts/auto_improvement/`.
