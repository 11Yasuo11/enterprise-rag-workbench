# Deterministic Eval and Error Analysis Loop

## Frozen Eval Dataset

A Frozen Eval Dataset is an immutable, hash-identified set of questions and explicit ground
truth. Control and Candidate must use the same file hash. The adapter accepts the repository's
legacy JSON schema and Phase-5 JSONL schema without editing either source. Missing labels stay
`null`, `unknown`, or an empty list; the evaluator never synthesizes gold labels.

## Eval-driven development

```text
Run
↓
Evaluate
↓
Classify Failure
↓
Aggregate
↓
Analyze Root Cause
↓
Change ONE Thing
↓
Re-run
↓
Regression Check
↓
Promote / Reject
↓
Repeat
```

The evaluation layer is artifact-first and independent of production. Capture one trace per
case from the same frozen dataset, change one experimental variable, evaluate both arms, then
compare them. Existing V2/V3 and Phase-5 runners, judges, datasets, and artifacts are unchanged.

## Run an eval

The connected cache-only end-to-end loop is:

```bash
uv run python scripts/run_experiment.py \
  --candidate configs/phase5kr-control-cache-only.yaml \
  --baseline artifacts/eval/current-baseline \
  --semantic-judge off
```

It validates the Frozen Dataset hash, executes cached Dense plus local BM25/RRF/Cross-Encoder,
captures filter and ranking stages, reuses exact Judge/artifact cache outcomes, evaluates,
compares, gates, and reports. `--resume` continues after the last flushed case. Cache misses never
call external providers and are reported as `PAID_SEMANTIC_EVAL_REQUIRED`.

The zero-cost default replays the authoritative Phase-5KR `FINAL_R` captured output:

```bash
uv run python scripts/run_eval.py --semantic-judge off
```

For a fresh experiment, export one JSON/JSONL trace per frozen case and pass it explicitly:

```bash
uv run python scripts/run_eval.py \
  --dataset data/eval/phase5k/acmeai_enterprise_rag_v3_final_unseen_e2e_120.jsonl \
  --traces path/to/captured-traces.jsonl \
  --arm CANDIDATE \
  --experiment-id one-change-only-v1
```

The command does not invoke the RAG pipeline or an LLM; this prevents accidental paid calls and
keeps evaluation deterministic. Pipeline execution remains owned by the existing experiment
runners. A trace may include ACL, tenant, version, Dense, BM25, RRF, Cross-Encoder, Judge,
Generator, and Citation state under `stage_trace`.

Outputs are written under `artifacts/eval/<experiment_id>/`: `results.json`, `traces.jsonl`,
`metrics.json`, `failure_census.json`, `slice_analysis.json`, `regression.json`, and `report.md`.
An existing experiment directory is never overwritten.

## Failure classification

Rules are applied only when their required evidence is present: retrieval miss, ranking loss,
filter failure, insufficient evidence, Judge false negative/positive, generator incomplete or
incorrect, and citation failure. Passing cases are `PASS`. Insufficient evidence becomes
`OTHER` or `UNRESOLVED`; it is not guessed into a root cause.

Every `OTHER`/`UNRESOLVED` case retains question type, retrieval and ranking status, Judge
decision, generator output, citation state, version sensitivity, document count, and secondary
tags. This makes the bucket actionable without silently changing historical semantics.

## Slice analysis

Only available metadata is used. Results are grouped by category, difficulty, document count,
version, ACL, tenant, and answerability. Each slice reports sample size, precision, recall, F1,
accuracy, and failure counts. Slices below five cases are marked `small_sample`.

## Regression gate

```bash
uv run python scripts/compare_eval.py \
  --baseline artifacts/eval/control \
  --candidate artifacts/eval/candidate
```

Dataset hash mismatch produces `MANUAL_REVIEW`. Otherwise the command reports metric deltas,
`fixed_cases`, `regressed_cases`, `unchanged_failures`, and `new_failures`, then applies
`configs/deterministic-eval-gate.yaml`. Existing experiment-specific frozen promotion policies
remain authoritative; this generic gate does not replace them.

## Deterministic versus semantic metrics

Free/deterministic metrics include Hit/Recall/nDCG at configurable K, MRR, candidate/top-K
recall, answer decision metrics, citation presence/existence/gold-ID matches, failure census,
slices, and regression gates. Deterministic evaluation always records zero API calls.

Evidence sufficiency, semantic answer correctness, answer completeness, and claim-level citation
support are isolated in `semantic_eval.py`. They require a separately reviewed local/provider
adapter and are off by default. `--semantic-judge sol` intentionally fails closed until such an
adapter is explicitly wired; it never silently calls the existing paid Sol Judge. Existing
GPT-5.6 Sol answerability artifacts can be replayed as traces without generating new calls.
