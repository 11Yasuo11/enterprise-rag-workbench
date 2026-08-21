# Repository Audit: Deterministic Eval Loop

Audit date: 2026-08-20. This audit preceded implementation. No frozen dataset, production
pipeline, existing Judge, or Phase-5 artifact was modified.

## ALREADY_IMPLEMENTED

- Retrieval primitives: `hit_at_k`, `recall_at_k`, reciprocal rank/MRR input, and `ndcg_at_k`
  in `evaluation/retrieval_metrics.py`; these are reused.
- Deterministic answerability, abstention, and document-level citation helpers in
  `evaluation/generation_metrics.py`.
- Frozen/validated legacy evaluation models in `evaluation/datasets.py`, plus immutable Phase-5
  freeze manifests and JSONL datasets under `data/eval/phase5*`.
- Full production `EvaluationRunner` metrics/tracing and persisted run cost fields.
- Detailed existing `FailureType` taxonomy and experiment-specific failure analyses.
- `FINAL_E2E_SCORER_V2`, the authoritative deterministic Phase-5KR behavior and aggregate metric
  semantics. The new layer preserves its precision/recall/F1 definitions.
- Existing experiment-specific regression thresholds and frozen promotion policies.
- Hosted GPT-5.6 Sol answerability Judge, caching, hard call limits, and default-off external calls.

## PARTIALLY_IMPLEMENTED

- Trace data existed across database records and experiment artifacts, but there was no common
  portable JSONL contract covering candidates, top-K, answer, citation, stages, and error.
- Failure census existed per experiment, including Phase-5KR, but `OTHER` lacked a generic
  structured drill-down contract.
- Category metrics existed, but there was no reusable metadata-driven slice engine with sample
  size warnings.
- Control/Candidate comparisons and gates existed inside experiments, but there was no generic
  artifact CLI with dataset identity enforcement and case-level fixed/regressed lists.
- Cost accounting existed in RAG/Judge runs, but deterministic evaluation cost was not explicitly
  separated in a portable report.

## MISSING BEFORE IMPLEMENTATION

- One additive, production-independent deterministic loop producing the requested seven files.
- Backward-compatible adapters for legacy JSON, Phase-5 JSONL, and captured multi-arm outputs.
- Generic rule-based taxonomy with `OTHER`/`UNRESOLVED` non-guessing behavior.
- Automatic failure percentages and detailed unresolved secondary tags.
- Configurable K values `[1, 3, 5, 10]` from a CLI.
- Generic slices for category, difficulty, document count, version, ACL, tenant, answerability.
- Dataset-hash-locked comparison and a generic configurable promotion gate.
- A hard module boundary separating semantic evaluation from deterministic evaluation.

## Compatibility decision

The implementation is additive under `rag_workbench/evaluation`, `scripts`, `configs`, `docs`,
`tests`, and a new `artifacts/eval` result. It does not widen existing Pydantic schemas, rewrite
ground truth, change an existing scorer, or import the new runner into production. Existing
experiment-specific promotion policy remains authoritative over the generic gate.

