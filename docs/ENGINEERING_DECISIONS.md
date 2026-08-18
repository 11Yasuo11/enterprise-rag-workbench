# Engineering Decisions

This document explains the frozen v2 design. Official v2 remains
`enterprise-rag-workbench-v2` at architecture hash
`964a639905502159e9e66d1b3b1bbfe7013a7f6e8cbc73a33d8412a91419a5da`.
Parent `enterprise-rag-workbench-v1` is immutable at
`e76aa8834d995c03d47d6d80c9917cc9e16e4f31b24dcd742250329239a23c85`.

The workbench is an evaluation system. Decisions were accepted or rejected from
measured, one-change experiments, not from technique popularity.

## Why a deterministic baseline existed

`HashingEmbeddingProvider` and `deterministic-extractive-v1` make the full
PostgreSQL / pgvector / retrieval / citation path runnable without paid APIs.
That baseline exists so ingestion, ACL filtering, experiment hashes, and CI can
be exercised independently of semantic quality. It is not a claim of production
retrieval quality.

## Why semantic embeddings

On `acmeai-eval-v1`, replacing hashing with `text-embedding-3-small` (64
dimensions) was a single independent variable. Recall@5 moved 0.740000 →
0.800000, MRR 0.688667 → 0.749167, nDCG 0.683385 → 0.757290. Multi-document
Recall@5 moved 0.750000 → 1.000000. ACL safety and citation correctness stayed
1.000000. Semantic embeddings therefore replaced hashing for measured quality
work. Hashing remains the offline/CI default.

## Why BM25

Dense retrieval can miss identifier-heavy or lexically exact evidence. BM25
Okapi v1 (`k1=1.2`, `b=0.75`, NFKC / casefold / identifier tokenization) is a
deterministic lexical branch. Tenant, ACL, and active-version filters construct
its candidate set before scoring. BM25 is not a learned sparse model and is not
SPLADE.

## Why RRF

Dense and BM25 scores are incomparable. Reciprocal Rank Fusion at fixed `k=60`
with `chunk_id` union ≤ 30 combines ranks without treating BM25 scores as cosine
similarities. The 0.28 dense threshold is never applied to BM25 or RRF scores.

The first Dense-vs-Hybrid calibration missed a +0.10 coverage materiality bar.
A later sealed 80-case replication met a pre-registered rule (coverage@5
+0.135135, three-document coverage@5 +0.225000) and selected
`HYBRID_CROSS_ENCODER_RERANK` as the v1 retriever. V2 kept that hybrid candidate
generation path.

## Why Cross-Encoder

Candidate generation can retrieve required evidence that still sits below Top-5.
A local `cross-encoder/ms-marco-MiniLM-L6-v2` reranker, pinned to revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, reorders the authorized union into
final Top-5. On the sealed reranking dataset, All Required Evidence Coverage@5
moved 0.777778 → 0.972222 under a frozen policy requiring ≥ +0.10. External
reranker APIs were not used.

## Why deterministic extractive generation

Frozen V2 answers with `deterministic-extractive-v1.1`, not a hosted chat model.
The generator may only emit text from validated supporting chunks, with a
verbatim supporting-text fallback when extractive synthesis cannot form a
citation-safe answer. That isolates retrieval and Judge quality from
hallucinated generation. Semantic answer-correctness scoring remains
unimplemented and is not faked.

## Why an Evidence Sufficiency Gate

Extractive generation will answer from any ACL-accessible chunk that clears a
score threshold. Early semantic runs had perfect ACL retrieval safety and still
produced unsupported answers from unrelated public context. The gate classifies
whether Top-5 evidence can answer the question. Fail-closed supporting-ID checks
prevent the generator from seeing unauthorized or uncited chunks. The gate does
not grade semantic answer correctness.

## Why Evidence Coverage v2 was rejected

A requirement-level `evidence-coverage-v2` judge was compared with frozen
`evidence-sufficiency-v1` on a sealed 60-case set, sharing one retrieval trace.
Coverage v2 reduced correct answers 37 → 33, increased incorrect abstentions
15 → 19, and added five regressions for one rescue. The frozen policy was not
met. Production Judge stayed `EVIDENCE_SUFFICIENCY_V1`.

## Why Sol replaced Luna

The independent variable was judge model identity only: `gpt-5.6-luna` versus
`gpt-5.6-sol`, same `evidence-sufficiency-v1` prompt, schema, temperature 0,
and request settings. On a sealed 60-case set, Sol moved correct answers
28 → 36 and F1 0.700000 → 0.818182, with 10 retrieval-complete false-negative
rescues and 0 unsupported-answer increase. Sol became
`GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`.

## Why fail-closed

Invalid supporting IDs, content-identity mismatches, schema failures, transport
exhaustion, and generation exceptions become abstention or typed
`GENERATION_FAILURE`. The workbench prefers an incorrect abstention over an
unsupported answer. Official v2 reports precision 1.000000 and unsupported
answers 0, with 31 incorrect abstentions left visible rather than patched after
the fact.

## Why quality retries are forbidden

Retrying a completed quality decision (Judge insufficient, abstention, or
extractive miss) would silently change one-shot benchmarks. Transport retries
are allowed only for `TIMEOUT`, `CONNECTION_ERROR`, `RATE_LIMIT`, and
`PROVIDER_5XX`, with at most two physical attempts. Schema-valid quality
outcomes are never retried.

## Why datasets are frozen before evaluation

Questions, category mix, and hidden labels are sealed before the first inference
of a benchmark. Hidden ground truth never reaches Dense, BM25, RRF,
Cross-Encoder, Sol, or the generator. Token-set overlap with prior datasets is
checked against a 0.5 ceiling so later suites are not paraphrases of earlier
failures.

## Why final benchmarks are one-shot

A final architecture is executed once on one unseen dataset. Interrupted runs
resume persisted checkpoints and do not replay successful paid calls. After
results exist, retrieval depth, thresholds, Judge prompts, and generator
semantics are not retuned against that dataset.

## Why official metrics are versioned

V2 numbers are a historical frozen record. Later V3 research must add a new
section rather than overwrite V2. Parent V1 remains immutable at
`e76aa8834d995c03d47d6d80c9917cc9e16e4f31b24dcd742250329239a23c85`.
