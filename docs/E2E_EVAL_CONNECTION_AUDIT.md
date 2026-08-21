# End-to-End Eval Connection Audit

Audit date: 2026-08-20. This work is additive and does not modify Frozen Dataset, Phase-5
artifacts, production dependency wiring, or Judge behavior.

## EXISTING_RUNNERS

- `EvaluationRunner` calls the production `RagService`, but its portable trace ends at the final
  retrieval list.
- Phase-5B/C/E/I runners explicitly execute Dense → BM25 → RRF → Cross-Encoder → Top-5 → Judge →
  Generator. Their stage logic is reusable but their orchestration and artifact schemas are
  experiment-specific.
- Phase-5KC executes the final unseen 120 cases. Phase-5KR is the authoritative scoring-only
  reconstruction from persisted Top-5, Judge cache, generator runs, and recovery cache.
- V2/V3 runners contain frozen, experiment-specific promotion policies that remain authoritative.

## REUSABLE_COMPONENTS

- `Retriever` and `QueryEmbeddingCache` for cached Dense retrieval.
- `BM25Retriever` for local lexical retrieval.
- `reciprocal_rank_fusion` for candidate union provenance.
- `CrossEncoderReranker` revision `233902...` from the local Hugging Face snapshot.
- Existing `select_document_diversified_top5`; it reads no evaluation labels.
- `AnswerabilityGateCacheRecord` and exact `gate_cache_key` for zero-call Sol replay.
- Deterministic extractive generator and `FINAL_E2E_SCORER_V2`.
- ACL/tenant SQL semantics from `apply_document_acl`.

## TRACE_ALREADY_AVAILABLE

- Dense/BM25 result objects expose chunk/document/version, score, rank, and branch provenance.
- RRF output exposes dense/lexical/fusion scores.
- Cross-Encoder output exposes input rank, score, and final rank.
- Judge cache stores ordered evidence identity, decision, supporting IDs, tokens, and model.
- Phase-5KR stores answer, abstention, citations, facts missing, and final behavior.

## TRACE_MISSING BEFORE THIS CHANGE

- One common candidate/reranker/Top-K document-level contract.
- Pre/post ACL, tenant, and active-version inventory with removal reasons.
- Evidence-backed stage attribution attached to every failure.
- Checkpoint/resume orchestration from Candidate config through comparison and gate.
- Dataset/config/code identities in one experiment manifest.

## Root-cause result

Fresh, cache-only retrieval reproduced the Frozen Control Top-5 for all answerable cases and
preserved the official aggregate metrics. The previous `OTHER=23` decomposes to
`RANKING_FAILURE=23`: required documents survived filters and RRF candidates but were incomplete
in Top-5. All 16 three-document failures are ranking losses. The 8 version-sensitive cases split
into 4 ranking losses and 4 cached Sol false negatives.

The numeric-date failures are more precisely `EVIDENCE_INSUFFICIENT=4`: the required document is
present, but the structured required date is absent from the captured Top-5 text. This supersedes
the earlier coarse generator attribution without changing the historical artifact.

## API safety

The connected Runner uses a cache-identity embedding provider that raises on cache miss, local
BM25/RRF/Cross-Encoder, direct read-only Judge-cache lookup, and deterministic generation.
Changed Judge inputs with no exact cache record become `PAID_SEMANTIC_EVAL_REQUIRED`; no fallback
API call or fabricated decision occurs.

