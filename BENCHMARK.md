# Benchmark

> Measured synthetic benchmark results are engineering evidence, not production performance
> claims. Pending work is explicitly separated from completed runs.

## Dataset

The current benchmark is `acmeai-eval-v1`: 100 cases against corpus `acmeai-v1`.

| Category | Cases |
|---|---:|
| single-document factual | 20 |
| multi-document | 20 |
| exact identifier | 10 |
| versioning | 10 |
| ACL | 10 |
| abstention | 15 |
| prompt injection | 10 |
| duplicate/similar | 5 |

## Baseline Hashing Retrieval

All three historical benchmark names remain persisted. The 100-case runs below were measured
again after migration `0004`; their quality metrics match the preserved historical values.

| Experiment | Cases | top_k | Threshold | Recall@K | MRR | nDCG | Abstention F1 | ACL safety | Version accuracy | Mean retrieval ms | Mean total ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline-hashing-v1` | 6 | 5 | 0.2 | 0.666667 | 0.666667 | 0.666667 | 1.000000 | 1.000000 | pending | 1.747917 | 1.825320 |
| `baseline-hashing-eval-v1` | 100 | 5 | 0.2 | 0.740000 | 0.688667 | 0.683385 | 0.666667 | 1.000000 | 1.000000 | 1.172498 | 1.237244 |
| `hashing-top10-eval-v1` | 100 | 10 | 0.2 | 0.750000 | 0.688667 | 0.687613 | 0.580645 | 1.000000 | 1.000000 | 1.501244 | 1.587495 |

The score is cosine similarity (`1 - cosine_distance`), so higher is better. Threshold filtering
keeps candidates with `similarity >= score_threshold` (implemented in SQL as
`cosine_distance <= 1 - score_threshold`).

## Hashing vs Semantic Embeddings

`semantic-baseline-v1` completed all 100 cases on 2026-08-13. The only configuration
differences from `baseline-hashing-eval-v1` are the embedding provider and model.

### Configuration

| Setting | Hashing | Semantic |
|---|---|---|
| Provider | `hashing` | `openai-compatible` |
| Model | `local-hashing-64` | `text-embedding-3-small` |
| Embedding version | `1` | `1` |
| Dimension | 64 | 64 |
| Chunk size / overlap | 180 / 30 | 180 / 30 |
| Top K / score threshold | 5 / 0.2 | 5 / 0.2 |
| Generator | deterministic extractive | deterministic extractive |
| Corpus / dataset | `acmeai-v1` / `acmeai-eval-v1` | `acmeai-v1` / `acmeai-eval-v1` |
| Index identity | `41cfe34cc2691d18b2d900fb9088860f6ea763f732dbccced090f4d5e01d87f4` | `e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466` |

### Overall metrics

| Metric | Hashing | Semantic | Semantic - Hashing |
|---|---:|---:|---:|
| Hit@5 | 0.780000 | 0.800000 | +0.020000 |
| Recall@5 | 0.740000 | 0.800000 | +0.060000 |
| MRR | 0.688667 | 0.749167 | +0.060500 |
| nDCG | 0.683385 | 0.757290 | +0.073905 |
| Abstention accuracy | 0.890000 | 0.900000 | +0.010000 |
| Abstention precision | 0.846154 | 1.000000 | +0.153846 |
| Abstention recall | 0.550000 | 0.500000 | -0.050000 |
| Abstention F1 | 0.666667 | 0.666667 | 0.000000 |
| Citation correctness | 1.000000 | 1.000000 | 0.000000 |
| ACL safety | 1.000000 | 1.000000 | 0.000000 |
| Version accuracy | 1.000000 | 1.000000 | 0.000000 |

### Category metrics

| Category | Hash Recall | Semantic Recall | Delta | Hash MRR | Semantic MRR | Delta | Hash nDCG | Semantic nDCG | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single-document factual | 0.950000 | 1.000000 | +0.050000 | 0.850000 | 0.950000 | +0.100000 | 0.876186 | 0.963093 | +0.086907 |
| multi-document | 0.750000 | 1.000000 | +0.250000 | 0.816667 | 0.941667 | +0.125000 | 0.707119 | 0.932182 | +0.225063 |
| exact identifier | 1.000000 | 1.000000 | 0.000000 | 0.883333 | 0.933333 | +0.050000 | 0.913093 | 0.950000 | +0.036907 |
| versioning | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 | 0.000000 |
| ACL | 0.500000 | 0.500000 | 0.000000 | 0.500000 | 0.500000 | 0.000000 | 0.500000 | 0.500000 | 0.000000 |
| abstention | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| prompt injection | 1.000000 | 1.000000 | 0.000000 | 0.770000 | 0.925000 | +0.155000 | 0.827964 | 0.943068 | +0.115104 |
| duplicate/similar | 1.000000 | 1.000000 | 0.000000 | 0.800000 | 0.700000 | -0.100000 | 0.852372 | 0.778558 | -0.073814 |

The semantic abstention-only category has accuracy 0.666667 and F1 0.800000, compared with
0.733333 and 0.846154 for hashing. ACL safety, prompt-injection boundary success, and version
accuracy are each 1.000000.

### Latency and usage

| Measurement | Hashing | Semantic | Delta |
|---|---:|---:|---:|
| Mean retrieval latency (ms) | 1.172498 | 404.004512 | +402.832014 |
| Mean generation latency (ms) | 0.044867 | 0.154268 | +0.109401 |
| Mean total latency (ms) | 1.237244 | 404.261618 | +403.024374 |

The semantic run made 116 application-level embedding requests: 16 document requests and 100
query requests. The provider returned 1,505 embedding input tokens: 540 for documents and 965
for queries. It created 24 vectors from 16 documents. This first index build had 0 document
cache hits and 16 misses; a post-run plan requires 0 document embedding calls, confirming reuse
of the completed index. The semantic index contains exactly 24 unique chunk keys, all 64
dimensional, with no metadata mismatch and no hashing vectors. No monetary cost is reported.

### Multi-document analysis

Semantic Recall@5 is 1.000000, up 0.250000 from hashing. All 20 cases retrieved all required
documents and were answered; none had a missing source, required evidence below Top K,
context-construction loss, or generation failure. The latest required source occurred at rank 5
in the observed traces, so the clean baseline does not support reducing K without calibration.

### Exact identifier analysis

Exact-identifier Recall@5 remains 1.000000. MRR improves by 0.050000 and nDCG by 0.036907.

| Case | Required-source first rank | Required-source best score |
|---|---:|---:|
| `identifier_01` (`CS-1842`) | 1 | 0.712844 |
| `identifier_02` (`ENG-DEP-17`) | 1 | 0.649509 |
| `identifier_03` (`SEC-TRAIN-44`) | 1 | 0.591306 |
| `identifier_04` (`ATLAS-API-301`) | 1 | 0.723880 |
| `identifier_05` (`FIN-TRAVEL-52`) | 1 | 0.500903 |
| `identifier_06` (`LEGAL-DEL-08`) | 1 | 0.839850 |
| `identifier_07` (`OPS-REC-E17`) | 3 | 0.603029 |
| `identifier_08` (`OPS-REC-W29`) | 1 | 0.673559 |
| `identifier_09` (`HR-COMP-900`) | 1 | 0.646440 |
| `identifier_10` (`HR-BEN-771`) | 1 | 0.797401 |

The required document is rank 1 in nine cases and rank 3 in one. Best required-document scores
range from 0.500903 to 0.839850 (mean 0.673872). There is no measured identifier failure to
justify hybrid search in this baseline.

### Abstention analysis

Across all categories, 20 cases should abstain. Semantic correctly abstains on 10 and answers
10; all 80 answerable cases are answered, yielding precision 1.000000, recall 0.500000, and F1
0.666667. Hashing correctly abstained on 11, answered 9 unsupported cases, and incorrectly
abstained on 2 answerable cases. Semantic therefore removes two incorrect abstentions but
increases unsupported answers by one; it does not improve F1 or solve the
accessible-but-unsupported-context problem.

Every unsupported answer below had one or more unrelated, ACL-accessible chunks above the
unchanged 0.2 threshold. `doc@score` entries are the complete observed Top-5 retrieval trace.

| Case | Top retrieved chunks (`document@score`) | Generated answer | Citations |
|---|---|---|---|
| `abstention_03` | `project-atlas-launch@0.480269`; `security-training-example@0.426740`; `operations-continuity-plan@0.259756`; `project-atlas-api@0.258211`; `engineering-deployment-handbook@0.233464` | Operations owns the business continuity plan. | `operations-continuity.md` |
| `abstention_04` | `finance-expense-policy@0.354109`; `remote-work-policy@0.332936`; `security-training-example@0.269339`; `finance-expense-policy@0.266714`; `project-atlas-api@0.237727` | The international travel approval code is FIN-TRAVEL-52. | `finance-expenses.md` |
| `abstention_11` | `remote-work-policy@0.350268`; `operations-continuity-plan@0.298995`; `project-atlas-launch@0.279259`; `security-incident-policy@0.276416`; `finance-expense-policy@0.258726` | Employees may work remotely two days per week; severity-one incidents must be reported within 15 minutes. | `remote-work-policy-2026.md`; `security-incident-policy-2026.md` |
| `abstention_13` | `project-atlas-launch@0.351180`; `project-atlas-api@0.279570`; `engineering-deployment-handbook@0.242892`; `data-retention-standard@0.234810`; `remote-work-policy@0.233974` | Project Atlas launched on April 12, 2026 and uses API version v3. | `project-atlas-launch.md`; `project-atlas-api.md` |
| `abstention_14` | `finance-expense-policy@0.501982`; `remote-work-policy@0.385682`; `operations-continuity-plan@0.340535`; `security-incident-policy@0.328424`; `data-retention-standard@0.317936` | The customer API recovery time objective is four hours. | `operations-continuity.md` |
| `acl_benefit_denied_01` | `security-training-example@0.435240`; `finance-expense-policy@0.408559`; `data-retention-standard@0.393515`; `engineering-deployment-handbook@0.370871`; `data-retention-standard@0.346178` | The training reference is SEC-TRAIN-44; Finance Operations owns exception review. | `security-training-untrusted.md`; `finance-expenses.md` |
| `acl_benefit_denied_02` | same trace as `acl_benefit_denied_01` | same answer as `acl_benefit_denied_01` | same citations as `acl_benefit_denied_01` |
| `acl_comp_denied_01` | `finance-expense-policy@0.488042`; `remote-work-policy@0.341442`; `security-training-example@0.334657`; `finance-expense-policy@0.314372`; `recovery-runbook-east@0.276496` | The international travel approval code is FIN-TRAVEL-52; Finance Operations owns exception review. | `finance-expenses.md` |
| `acl_comp_denied_02` | same trace as `acl_comp_denied_01` | same answer as `acl_comp_denied_01` | same citation as `acl_comp_denied_01` |
| `acl_comp_denied_03` | same trace as `acl_comp_denied_01` | same answer as `acl_comp_denied_01` | same citation as `acl_comp_denied_01` |

The restricted benefits and compensation documents never appear in these traces. These are
threshold/abstention failures after correct ACL filtering, not ACL leaks. There are no semantic
false-positive abstentions on answerable questions.

### Failure taxonomy

| Taxonomy | Hashing | Semantic |
|---|---:|---:|
| `PARSING_FAILURE` | 0 | 0 |
| `CHUNKING_FAILURE` | 0 | 0 |
| `RETRIEVAL_MISS` | 10 | 0 |
| `RANKING_FAILURE` | 0 | 0 |
| `ACL_FAILURE` | 0 | 0 |
| `VERSION_FAILURE` | 0 | 0 |
| `THRESHOLD_FAILURE` | 9 | 10 |
| `CONTEXT_CONSTRUCTION_FAILURE` | 0 | 0 |
| `GENERATION_FAILURE` | 0 | 0 |
| `CITATION_FAILURE` | 0 | 0 |
| `ABSTENTION_FALSE_POSITIVE` | 9 | 10 |
| `ABSTENTION_FALSE_NEGATIVE` | 2 | 0 |
| `UNKNOWN` | 0 | 0 |

### Regressions and limitations

- Mean total latency increases by 403.024374 ms because every semantic query uses the remote
  embedding endpoint; hashing is local.
- Abstention-category F1 decreases by 0.046154 and accuracy by 0.066666. Overall abstention F1
  is unchanged when rounded to six decimals.
- Duplicate/similar MRR decreases by 0.100000 and nDCG by 0.073814, although Recall remains
  1.000000 and all five cases are answered.
- The configured regression detector reports no threshold breach, and both hard constraints
  remain perfect.
- This is a synthetic corpus with deterministic extractive generation. Answer correctness,
  groundedness, and completeness judging remain pending.

## Semantic Threshold Calibration

This phase preserved `semantic-baseline-v1`, its 100 case results, configuration, index identity,
and historical latency values. Calibration used the same 24 semantic document vectors, fixed
`top_k=5`, unchanged corpus/dataset/chunking/model/dimension, and a new persisted query cache.
No document was re-embedded.

### Score distributions

The persisted `semantic-baseline-v1` traces show substantial overlap between answerable and
abstention cases. For answerable cases, “lowest required” is the lowest best score among all
required documents in a case. Rank statistics for abstention cases are the score at that rank.

| Population / score | N | Min | P10 | P25 | Median | Mean | P75 | P90 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Answerable top relevant | 80 | 0.287598 | 0.530281 | 0.594565 | 0.688694 | 0.670485 | 0.734418 | 0.797853 | 0.839850 |
| Answerable lowest required | 80 | 0.287598 | 0.476775 | 0.558485 | 0.647975 | 0.646756 | 0.730334 | 0.797853 | 0.839850 |
| Answerable highest irrelevant | 80 | 0.257415 | 0.360010 | 0.435997 | 0.523801 | 0.503351 | 0.575372 | 0.640755 | 0.734601 |
| Multi-document lowest required | 20 | 0.336398 | 0.417175 | 0.441769 | 0.539217 | 0.532025 | 0.610153 | 0.636302 | 0.710697 |
| Abstention rank 1 | 20 | 0.345772 | 0.350260 | 0.353377 | 0.430952 | 0.423637 | 0.482212 | 0.489436 | 0.546604 |
| Abstention rank 2 | 20 | 0.279570 | 0.297398 | 0.304772 | 0.341442 | 0.352639 | 0.404934 | 0.409634 | 0.426740 |
| Abstention rank 3 | 20 | 0.242892 | 0.256495 | 0.273456 | 0.334657 | 0.323580 | 0.364456 | 0.393515 | 0.419178 |
| Abstention rank 4 | 20 | 0.234810 | 0.243929 | 0.264588 | 0.314372 | 0.307295 | 0.335358 | 0.370871 | 0.406334 |
| Abstention rank 5 | 20 | 0.217717 | 0.232750 | 0.238323 | 0.276496 | 0.285496 | 0.319211 | 0.347986 | 0.402503 |

The answerable minimum (0.287598) is below every abstention top-1 score, and the lowest required
multi-document score (0.336398) is below the top score of every remaining unsupported-answer
case. A single score boundary therefore cannot cleanly separate the two populations.

### Threshold experiments

All experiments are persisted. “Chunks” is the total passing threshold across 100 cases; mean
chunks is per query. The security constraints (ACL safety, version correctness, and prompt-
injection boundary success) and citation correctness were 1.000000 for every row.

| Experiment | Threshold | Hit@5 | Recall@5 | MRR | nDCG | Abstention F1 | Unsupported | Incorrect abstentions | Chunks | Mean chunks | Multi Recall | Identifier Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `semantic-threshold-020` | 0.200000 | 0.800000 | 0.800000 | 0.749167 | 0.757290 | 0.666667 | 10 | 0 | 497 | 4.970000 | 1.000000 | 1.000000 |
| `semantic-threshold-025` | 0.250000 | 0.800000 | 0.800000 | 0.749167 | 0.757290 | 0.666667 | 10 | 0 | 482 | 4.820000 | 1.000000 | 1.000000 |
| `semantic-threshold-026` | 0.260000 | 0.800000 | 0.800000 | 0.749167 | 0.757290 | 0.709677 | 9 | 0 | 469 | 4.690000 | 1.000000 | 1.000000 |
| `semantic-threshold-028` | 0.280000 | 0.800000 | 0.800000 | 0.749167 | 0.757290 | 0.709677 | 9 | 0 | 456 | 4.560000 | 1.000000 | 1.000000 |
| `semantic-threshold-0285` | 0.285000 | 0.800000 | 0.800000 | 0.749167 | 0.757290 | 0.709677 | 9 | 0 | 451 | 4.510000 | 1.000000 | 1.000000 |
| `semantic-threshold-0287598` | 0.287598 | 0.790000 | 0.790000 | 0.746667 | 0.752983 | 0.687500 | 9 | 1 | 446 | 4.460000 | 1.000000 | 1.000000 |
| `semantic-threshold-030` | 0.300000 | 0.790000 | 0.790000 | 0.746667 | 0.752983 | 0.687500 | 9 | 1 | 436 | 4.360000 | 1.000000 | 1.000000 |
| `semantic-threshold-0336398` | 0.336398 | 0.790000 | 0.790000 | 0.746667 | 0.752983 | 0.687500 | 9 | 1 | 393 | 3.930000 | 1.000000 | 1.000000 |
| `semantic-threshold-035` | 0.350000 | 0.790000 | 0.795000 | 0.746667 | 0.759114 | 0.727273 | 8 | 1 | 367 | 3.670000 | 0.975000 | 1.000000 |
| `semantic-threshold-0350269` | 0.350269 | 0.790000 | 0.815000 | 0.746667 | 0.779114 | 0.764706 | 7 | 1 | 365 | 3.650000 | 0.975000 | 1.000000 |
| `semantic-threshold-040` | 0.400000 | 0.790000 | 0.855000 | 0.746667 | 0.819114 | 0.833333 | 5 | 1 | 291 | 2.910000 | 0.975000 | 1.000000 |
| `semantic-threshold-045` | 0.450000 | 0.790000 | 0.900000 | 0.746667 | 0.873993 | 0.894737 | 3 | 1 | 208 | 2.080000 | 0.850000 | 1.000000 |
| `semantic-threshold-050` | 0.500000 | 0.780000 | 0.925000 | 0.736667 | 0.900124 | 0.952381 | 0 | 2 | 168 | 1.680000 | 0.825000 | 1.000000 |
| `semantic-threshold-0501983` | 0.501983 | 0.770000 | 0.925000 | 0.726667 | 0.900124 | 0.930233 | 0 | 3 | 165 | 1.650000 | 0.825000 | 0.900000 |
| `semantic-threshold-055` | 0.550000 | 0.690000 | 0.855000 | 0.653333 | 0.833993 | 0.800000 | 0 | 10 | 121 | 1.210000 | 0.675000 | 0.900000 |
| `semantic-threshold-060` | 0.600000 | 0.580000 | 0.745000 | 0.548333 | 0.729289 | 0.655738 | 0 | 21 | 86 | 0.860000 | 0.475000 | 0.800000 |

### Selected threshold and remaining failures

`semantic-threshold-028` (0.28) is the selected calibrated configuration. It is Pareto-safe
against the historical semantic baseline: Hit@5, Recall@5, MRR, nDCG, multi-document Recall,
exact-identifier Recall, ACL safety, version correctness, prompt-injection boundary success,
and citation correctness are unchanged. Unsupported answers fall from 10 to 9, abstention
recall rises from 0.500000 to 0.550000, abstention F1 rises from 0.666667 to 0.709677, and no
incorrect abstention is introduced. The 0.285 candidate has the same measured quality but less
margin below the first observed answerable evidence boundary; 0.287598 introduces an incorrect
abstention and reduces Recall/Hit.

Of the original 10 unsupported-answer cases, `abstention_03` is fixed because the misleading
`operations-continuity-plan` chunk at 0.259756 no longer enters context. The other nine remain:
`abstention_04`, `abstention_11`, `abstention_13`, `abstention_14`, both denied benefit cases,
and all three denied compensation cases. Their only classifications are nine
`ABSTENTION_FALSE_POSITIVE` and nine paired `THRESHOLD_FAILURE` entries. There are zero retrieval,
ranking, context-construction, generation, citation, ACL, version, or unknown failures.

Static-threshold sufficiency: **NO**. Removing all unsupported answers requires a boundary above
the observed unsupported scores (through 0.501982), but threshold 0.50 reduces multi-document
Recall to 0.825000 and threshold 0.501983 also reduces exact-identifier Recall to 0.900000.
Future work should measure an evidence-sufficiency/answerability gate; it is not implemented here.

### Query embedding cache and latency decomposition

The deterministic SHA-256 cache identity includes normalized query text, embedding provider,
model, version, and dimension. The evaluation has 100 cases but 84 unique normalized queries.
The first calibration run made 84 external query calls and had 16 within-run cache hits; the
remaining 15 threshold runs were fully warm. Across this phase: 1,516 hits, 84 misses, 84
external calls, 777 embedding input tokens, 84 persisted cache rows, and zero document calls.

| Mean latency component (ms) | Cold/new query (84) | Warm selected run (100) |
|---|---:|---:|
| Query embedding provider | 573.481641 | 0.000000 |
| Embedding cache lookup | 2.006637 | 0.417620 |
| Vector search SQL execution | 4.063204 | 0.489491 |
| ACL predicate construction | 0.289615 | 0.047882 |
| Context construction | 0.078613 | 0.014120 |
| Generation | 0.167980 | 0.042722 |
| Total | 584.166554 | 1.177643 |

ACL remains part of the pre-retrieval SQL statement. The ACL timing above measures predicate
construction; database execution of ACL plus cosine search is included in vector-search SQL
execution. The decomposition shows that the historical approximately 404 ms semantic retrieval
latency was provider/network dominated, not pgvector dominated. Historical latency rows were
not redefined or backfilled.

### Duplicate / similar analysis

The semantic regression is a near-tie ordering change, not a recall failure. Expected preferred
source ranks are 2, 1, 2, 1, 2 for `duplicate_01` through `duplicate_05`, versus 2, 1, 1, 2, 1
under hashing. Semantic chooses West before East in cases 01 and 03 and East before West in case
05; score gaps are only 0.015901, 0.038801, and 0.024821. This produces MRR 0.700000 and nDCG
0.778558 while Recall remains 1.000000. Threshold 0.28 changes none of the five rankings or
scores. No reranking or deduplication change was implemented.

## Deferred Experiments

No chunking grid, retrieval-parameter grid, hybrid search, reranking, or query rewrite was run
after observing the semantic baseline. The measured traces support exactly one next experiment:
**Top-K / threshold calibration**, focused on the semantic score threshold. All answerable and
multi-document cases already have complete Top-5 source coverage, while all 10 remaining
failures are unsupported answers caused by unrelated accessible chunks clearing 0.2.

Hybrid search, reranking, and query rewrite are not justified by this baseline: exact identifiers
have perfect Recall, no relevant source is known to be below Top K, and there are no query-wording
retrieval misses.

## Evidence Sufficiency / Answerability Gate

Status: **PARTIAL — implementation complete; external judge calibration blocked by missing
explicit provider/model authorization.** No judge outputs were invented, no holdout outcomes
were inspected, and no model was silently selected.

### Baseline preservation

The historical `semantic-baseline-v1` configuration hash remains
`5dd80a49706e66e240c0bdf1494bff5f3c5fe9858213b00c1f70626309f13e23` and the selected
`semantic-threshold-028` configuration hash remains
`39ff41408fd8a3798cd653e1756ab5278246f5a63c3a1c22fd05b57b5ea2c7ee`. Their completed run
and case rows were not rewritten. The semantic index remains 24 unique 64-dimensional
`text-embedding-3-small` version-1 vectors at index identity
`e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466`.

Preflight content identities:

| Artifact | SHA-256 |
|---|---|
| `data/eval/eval_v1.json` | `afa08683b1f816fe18df4549bdc6d09267550a77fb668c045723b39664b4d537` |
| `data/corpus_manifests/acmeai-v0.1.txt` | `b42c2765a76da3370fe26c59ecdb4f7a8faeef8f2c4d164ec7d712c664be3b89` |
| `configs/semantic-threshold-calibration-v1.yaml` | `735dcb00f3ac4283649976960ae43a9890d50ac67a9c998842f396f87dad8a19` |

The persisted query-embedding cache contained 84 entries at preflight, and its hit, miss, and
provider/model/version/dimension isolation tests pass.

### Evaluation split methodology

The fixed seed is `20260813`. The split is deterministic and stratified on
`(category, should_abstain)`, with proportional largest-remainder allocation. The split identity
is `208f41caafcf5d05e8aa7c32f912a56324ee6a52d57fa073d49b6ddc5c35adf1`.

| Category | Calibration | Holdout |
|---|---:|---:|
| single-document factual | 14 | 6 |
| multi-document | 14 | 6 |
| exact identifier | 7 | 3 |
| versioning | 7 | 3 |
| ACL | 7 | 3 |
| abstention | 10 | 5 |
| prompt injection | 7 | 3 |
| duplicate/similar | 4 | 1 |
| **Total** | **70** | **30** |

Calibration contains 56 answerable and 14 abstention-behavior cases. Holdout contains 24
answerable and 6 abstention-behavior cases. Code prevents configuration selection from a
holdout partition and makes a locked holdout authorization one-shot. Runtime retrieval,
generation, and gating receive only the question, principal, and authorized chunks; evaluation
labels are never passed to them.

### Candidate configurations

| Candidate | Retrieval | Gate | Generation context | Calibration status |
|---|---|---|---|---|
| A | top-5, threshold 0.28 | none | all context-budgeted chunks | measured from preserved trace |
| B1 | unchanged | semantic evidence judge | all context-budgeted chunks | pending external configuration |
| B2 | unchanged | same judge | validated supporting chunks only | pending external configuration |

The gate contract is provider independent and returns strict structured fields:
`answerable`, `supporting_chunk_ids`, optional `confidence`, and enum `reason_code`. The prompt
version is `evidence-sufficiency-v1`; retrieved chunks are delimited as untrusted data. An
answerable decision fails closed unless every selected ID is non-empty, in the ordered retrieved
set, active, and authorized for the current principal.

The gate cache identity includes normalized question, ordered chunk IDs, document-version IDs,
version labels, content SHA-256 values, judge provider, judge model, judge version, and prompt
version. Model, prompt, and retrieval-context isolation are tested.

### Calibration results

Candidate A on the 70 calibration cases:

| Metric | A | B1 | B2 |
|---|---:|---:|---:|
| Answerability accuracy | 0.914286 | pending | pending |
| Answerability precision | 0.903226 | pending | pending |
| Answerability recall | 1.000000 | pending | pending |
| Answerability F1 | 0.949153 | pending | pending |
| Unsupported answers | 6 | pending | pending |
| Incorrect abstentions | 0 | pending | pending |
| Recall@5 | 0.800000 | pending | pending |
| MRR | 0.736905 | pending | pending |
| nDCG | 0.747641 | pending | pending |
| Multi-document retrieval Recall@5 | 1.000000 | pending | pending |
| Exact-identifier retrieval Recall@5 | 1.000000 | pending | pending |
| Citation validity | 1.000000 | pending | pending |
| Conservative expected-answer/source support | 0.564516 | pending | pending |
| ACL safety | 1.000000 | pending | pending |
| Version correctness | 1.000000 | pending | pending |
| Prompt-injection boundary | 1.000000 | pending | pending |

The support metric is deterministic but conservative: it requires the expected-answer content
terms and all expected source documents in the citations. The dataset contains expected answers
and expected documents, but no claim spans or evidence quotes. Therefore this is not presented
as a general semantic entailment score, and answer completeness remains unmeasured.

### External-call safety and blocked selection

The repository has no configured answerability judge provider/model, and defaults remain
`ALLOW_EXTERNAL_JUDGE_CALLS=false` and `MAX_EXTERNAL_JUDGE_CALLS=0`. On calibration there are
62 unique gate inputs after identical question/evidence inputs are deduplicated, zero cached
gate results, and 62 missing results. The configured ceiling is zero, so B1/B2 were stopped
before an external call. B2 would reuse B1 cache entries for the same inputs.

Because B1 and B2 have no real outputs, no final gate configuration was selected or locked.
Consequently the 30-case holdout was not evaluated and remains unconsumed. This is intentional:
selecting A merely to produce a holdout number would not answer whether an evidence gate helps,
and fabricating judge results would invalidate the experiment.

### Original-nine failure analysis

All nine preserved failures remain available with questions, top-five source/score traces, and
generated behavior in `semantic-threshold-028`. They comprise `abstention_04`,
`abstention_11`, `abstention_13`, `abstention_14`, `acl_benefit_denied_01`,
`acl_benefit_denied_02`, `acl_comp_denied_01`, `acl_comp_denied_02`, and
`acl_comp_denied_03`. A real gate decision and supporting IDs do not exist yet, so none is
classified as fixed or still unsupported by a gate. Their baseline final behavior remains an
unsupported answer.

### Latency and usage

For Candidate A calibration, warm-cache means were: query-cache lookup 0.417794 ms, query
embedding API 0 ms, vector search 0.490890 ms, context construction 0.014195 ms, generation
0.043060 ms, and total 1.179189 ms. All 70 calibration queries were query-embedding cache hits.
Gate cache lookup, judge API, and context pruning are not applicable to A. B1/B2 cold and warm
latency cannot be measured without selecting and authorizing a real judge. No judge calls or
judge tokens were used; no document or query embedding calls were required for this analysis.

### Limitations and next decision

- Holdout was not used for tuning and has not been evaluated.
- B1/B2 calibration, configuration lock, primary holdout result, original-nine gate outcomes,
  multi-document support selection, and duplicate-source selection remain pending one explicit
  judge provider/model plus a reviewed call ceiling.
- Retrieval architecture remains unchanged. Current evidence does not justify hybrid search,
  reranking, or query rewrite.

## Evidence Judge Benchmark — Qwen3-8B vs GPT-5.6 Luna

### Methodology and integrity

Five candidates were evaluated on the deterministic 70-case calibration split: A (threshold-only), B1/B2 (one cached local Qwen decision with all/supporting-only generation context), and C1/C2 (one cached Luna decision with the same two context policies). Retrieval remained Top-5 dense semantic search at threshold 0.28 using the existing 64-dimensional `text-embedding-3-small` index. No document was re-embedded.

Both providers used prompt version `evidence-sufficiency-v1`, the same task, evidence delimiters, reason codes, supporting-ID rule, and fail-closed semantics. Judges received only the question and authorized retrieved evidence; evaluation labels and expected sources/answers were not provided.

Qwen v4 used Ollama's native `/api/chat` endpoint with `think=false`, deterministic bounded inference, and native JSON Schema output. Versions 1 and 2 used the unsuitable OpenAI-compatible Ollama transport, where reasoning consumed the bounded output and structured content became unreliable; version 3 has no persisted cache result. All older rows remain for auditability and were excluded from B1/B2 quality metrics. Cache identity includes provider, model, judge and prompt versions, normalized question, ordered evidence identity/content hashes, index identity, and prompt-render hash.

**The holdout set was not used to select the judge model or context-pruning policy.**

### Calibration results

| Metric | A | B1 | B2 | C1 | C2 |
|---|---:|---:|---:|---:|---:|
| answerability_accuracy | 0.914286 | 0.900000 | 0.900000 | 0.985714 | 0.985714 |
| answerability_precision | 0.903226 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| answerability_recall | 1.000000 | 0.875000 | 0.875000 | 0.982143 | 0.982143 |
| answerability_f1 | 0.949153 | 0.933333 | 0.933333 | 0.990991 | 0.990991 |
| correct_answer_count | 56 | 49 | 49 | 55 | 55 |
| correct_abstention_count | 8 | 14 | 14 | 14 | 14 |
| unsupported_answer_count | 6 | 0 | 0 | 0 | 0 |
| incorrect_abstention_count | 0 | 7 | 7 | 1 | 1 |
| judge_error_count | 0 | 5 | 5 | 0 | 0 |
| judge_format_error_count | 0 | 5 | 5 | 0 | 0 |
| gate_false_positive_count | 6 | 0 | 0 | 0 | 0 |
| gate_false_negative_count | 0 | 7 | 7 | 1 | 1 |
| multi_document_answer_success | 1 | 0.857143 | 0.857143 | 0.928571 | 0.928571 |
| exact_identifier_answer_success | 1 | 0.857143 | 0.857143 | 1 | 1 |
| citation_validity | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| claim_support_rate | 0.564516 | 0.673469 | 0.714286 | 0.636364 | 0.745455 |
| grounded_answer_rate | 0.564516 | 0.673469 | 0.714286 | 0.636364 | 0.745455 |
| recall_at_k | 0.800000 | 0.800000 | 0.800000 | 0.800000 | 0.800000 |
| reciprocal_rank | 0.736905 | 0.736905 | 0.736905 | 0.736905 | 0.736905 |
| ndcg_at_k | 0.747641 | 0.747641 | 0.747641 | 0.747641 | 0.747641 |
| acl_safety | 1 | 1 | 1 | 1 | 1 |
| version_accuracy | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| prompt_injection_boundary | 1 | 1 | 1 | 1 | 1 |
| unauthorized_evidence_selection | 0 | 0 | 0 | 0 | 0 |

Retrieval metrics are preserved for comparison; the evidence judges did not change retrieval ranking or index quality.

### Calibration confusion matrices

Order: expected-answerable/predicted-answerable, expected-answerable/predicted-abstain, expected-abstain/predicted-answerable, expected-abstain/predicted-abstain.

| Candidate | TP / FN / FP / TN |
|---|---:|
| A | 56 / 0 / 6 / 8 |
| B1 | 49 / 7 / 0 / 14 |
| B2 | 49 / 7 / 0 / 14 |
| C1 | 55 / 1 / 0 / 14 |
| C2 | 55 / 1 / 0 / 14 |

### Qwen vs Luna trade-offs

| Dimension | Qwen (B1/B2) | Luna (C1/C2) |
|---|---|---|
| Quality | B1 F1 0.933333; B2 F1 0.933333 | C1 F1 0.990991; C2 F1 0.990991 |
| Unsupported / incorrect abstentions | 0 / 7 | 0 / 1 |
| Format / request / timeout errors | 5 / 0 / 0 | 0 / 0 / 0 |
| Supporting evidence selection | Qwen had four supporting-context-loss cases; B2 conservative support was 0.714286. | Luna had one supporting-context-loss case; C2 conservative support was 0.745455. |
| Mean cold judge latency | 6382.996094 ms | 2167.811773 ms |
| Cache behavior B1/B2 or C1/C2 | 8/62 then 70/0 | 8/62 then 70/0 |
| Privacy / operations | Local evidence stays on the developer machine; no external judge API usage, but local hardware, memory, energy, model storage, and runtime management remain real costs. | Authorized evidence is sent to OpenAI; network availability, latency, token billing, and external-service governance apply. |

### Selected configuration

**C2** was locked at `2026-08-13 19:37:18.887585+00:00`. Calibration-only selection minimized unsupported answers to 0 with 1 incorrect abstentions and answerability F1 0.9909909909909909; all hard constraints passed.

### Original-nine calibration analysis

Only six of the original nine unsupported-answer cases were in calibration; the remaining three stayed sealed as part of holdout and were not used for selection.

| Case | Retrieved chunks (score) | Qwen decision / reason / support / status | Luna decision / reason / support / status | Qwen / Luna outcome |
|---|---|---|---|---|
| `abstention_04` | 2798f984-b356-4e40-9ed2-574ffbe26e86 (0.354109), 803b1d29-ee55-4c31-abc8-13e9e5be6485 (0.332936) | False / ACCESS_RESTRICTED_EVIDENCE / none / OK | False / IRRELEVANT_EVIDENCE / none / OK | FIXED_BY_GATE / FIXED_BY_GATE |
| `abstention_11` | 803b1d29-ee55-4c31-abc8-13e9e5be6485 (0.350268), 2aafc123-32cd-4c8b-ad3e-845392de29e6 (0.298995) | False / MISSING_REQUIRED_FACT / none / OK | False / MISSING_REQUIRED_FACT / none / OK | FIXED_BY_GATE / FIXED_BY_GATE |
| `acl_benefit_denied_02` | c19243d4-0a12-4d2b-b727-de35e742516f (0.435240), 2798f984-b356-4e40-9ed2-574ffbe26e86 (0.408559), c91ee2ff-af75-461a-9372-0c562e46051d (0.393515), 01423f11-8d28-418b-bd8f-b548fa0491ef (0.370871), 054089a4-d1c8-4628-aed2-03dbd8b4e8e9 (0.346178) | False / MISSING_REQUIRED_FACT / none / OK | False / MISSING_REQUIRED_FACT / none / OK | FIXED_BY_GATE / FIXED_BY_GATE |
| `acl_comp_denied_01` | 2798f984-b356-4e40-9ed2-574ffbe26e86 (0.488042), 49679d97-020f-4418-9430-4c296bc824b1 (0.341442), c19243d4-0a12-4d2b-b727-de35e742516f (0.334657), 7bdffe91-3861-44db-8be2-d6c7328997e8 (0.314372) | False / ACCESS_RESTRICTED_EVIDENCE / none / OK | False / MISSING_REQUIRED_FACT / none / OK | FIXED_BY_GATE / FIXED_BY_GATE |
| `acl_comp_denied_02` | 2798f984-b356-4e40-9ed2-574ffbe26e86 (0.488042), 49679d97-020f-4418-9430-4c296bc824b1 (0.341442), c19243d4-0a12-4d2b-b727-de35e742516f (0.334657), 7bdffe91-3861-44db-8be2-d6c7328997e8 (0.314372) | False / ACCESS_RESTRICTED_EVIDENCE / none / OK | False / MISSING_REQUIRED_FACT / none / OK | FIXED_BY_GATE / FIXED_BY_GATE |
| `acl_comp_denied_03` | 2798f984-b356-4e40-9ed2-574ffbe26e86 (0.488042), 49679d97-020f-4418-9430-4c296bc824b1 (0.341442), c19243d4-0a12-4d2b-b727-de35e742516f (0.334657), 7bdffe91-3861-44db-8be2-d6c7328997e8 (0.314372) | False / ACCESS_RESTRICTED_EVIDENCE / none / OK | False / MISSING_REQUIRED_FACT / none / OK | FIXED_BY_GATE / FIXED_BY_GATE |

### Latency and usage

| Candidate | Judge ms | Gate-cache ms | Pruning ms | Generation ms | Total ms | Gate H/M | External calls | Local calls | Judge input/output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.000000 | 0.000000 | 0.000224 | 0.055630 | 2.599924 | 0/0 | 0 | 0 | 0/0 |
| B1 | 6382.996094 | 1.338139 | 6430.633282 | 0.284279 | 6443.847660 | 8/62 | 0 | 62 | 39295/5646 |
| B2 | 0.000000 | 0.381718 | 1.328117 | 0.024105 | 3.208451 | 70/0 | 0 | 0 | 39295/5646 |
| C1 | 2167.811773 | 1.144651 | 2184.155567 | 0.157336 | 2190.597092 | 8/62 | 62 | 0 | 43421/4156 |
| C2 | 0.000000 | 0.264665 | 0.879166 | 0.019059 | 2.187723 | 70/0 | 0 | 0 | 43421/4156 |

Actual model requests/cache records: Qwen 62/62 with 34750/5067 input/output tokens; Luna 86/86 with 52879/5160 input/output tokens.

Qwen v4 outcomes were 57 successful decisions and 5 fail-closed format errors (zero request errors and zero timeouts). Older persisted Qwen diagnostics made 37 local calls: 29 v1 cache rows and 6 v2 cache rows; v3 had 0. Cumulative persisted local calls are 99 against the configured ceiling 92. This exceeds the requested cumulative interpretation of the ceiling by 7 because the persisted runner enforced it per judge version. No further Qwen calls were issued after this was discovered.

Calibration-only Luna usage was 62 external calls with 38,436 input and 3,719 output tokens. Holdout reused five cached inputs and made 24 calls, producing cumulative Luna usage of 86 calls, 52,879 input tokens, and 5,160 output tokens. This phase made zero embedding calls; all query embeddings were cache hits.

No project-local verified pricing configuration exists, so monetary cost is not calculated. Token usage is reported instead.

### Sealed holdout results

The holdout was executed exactly once for A and the single locked winner C2. No other judge candidate was run on holdout, and no configuration was changed afterward. The configuration lock was persisted before the first holdout call.

| Metric | A | C2 |
|---|---:|---:|
| correct_answer_count | 24 | 24 |
| correct_abstention_count | 3 | 6 |
| unsupported_answer_count | 3 | 0 |
| incorrect_abstention_count | 0 | 0 |
| answerability_accuracy | 0.900000 | 1.000000 |
| answerability_precision | 0.888889 | 1.000000 |
| answerability_recall | 1.000000 | 1.000000 |
| answerability_f1 | 0.941176 | 1.000000 |
| multi_document_answer_success | 1 | 1 |
| exact_identifier_answer_success | 1 | 1 |
| citation_validity | 1.000000 | 1.000000 |
| claim_support_rate | 0.629630 | 0.750000 |
| acl_safety | 1 | 1 |
| version_accuracy | 1.000000 | 1.000000 |
| prompt_injection_boundary | 1 | 1 |
| unauthorized_evidence_selection | 0 | 0 |

### Generalization: GOOD

F1 change +0.009009; unsupported-answer rate change +0.000000; security constraints held. This is descriptive evidence, not a statistical-significance claim.

### Security and limitations

Supporting IDs were checked server-side against retrieved IDs, ACLs, active and expected versions, and index identity. Invalid selections fail closed and the operational error is persisted separately. Retrieved content is explicitly delimited as untrusted evidence.

The benchmark uses a synthetic 100-case dataset, deterministic extractive final generation, and one local Qwen quantization/runtime. Claim-level entailment remains unmeasured because the dataset has no claim-span annotations; `claim_support_rate` is the existing conservative expected-answer/source metric.

Largest remaining measured problem: supporting-evidence selection for multi-document questions (C2 calibration multi-document success was 0.928571, driven by one false abstention/context-loss case). A future experiment should evaluate a new, frozen multi-document-focused judge prompt or model on newly created unseen evaluation data; it must not reuse this consumed holdout.

## Multi-Document Evidence Coverage Judge v2

This experiment tested one independent variable: frozen `evidence-sufficiency-v1` (C2) versus requirement-level `evidence-coverage-v2` (D), both using GPT-5.6 Luna, `text-embedding-3-small` at 64 dimensions, dense pgvector retrieval, Top-5, threshold 0.28, the same corpus/security model/generator, and supporting-context-only generation. The previously consumed holdout was not reused. Historical experiments, Qwen diagnostics, query/gate caches, the corpus, and semantic index were not modified.

### New dataset and sealed split

`acmeai-multidoc-eval-v2` contains 60 new questions: 20 two-document answerable, 10 three-document answerable, 10 deliberately partial-evidence, 6 version-sensitive, 6 ACL-sensitive, 4 near-duplicate, and 4 multi-document no-answer cases. Answerable cases require at least two distinct sources. Hidden ground truth includes required documents/facts, expected answerability, principal, and version/security behavior; none is rendered into runtime judge prompts.

- Dataset SHA-256: `a5d652ab76ee68f5adce4dd420a65bd7cc285f2b9fa4097aa070435ddb58e6b9`
- Deterministic stratified seed: `20260817`
- Calibration / holdout: 40 / 20
- Split identity: `32b1bf1a29dc1fd612931b372810f8a186398c529c76054c18d58d4549e0bd7c`

The new holdout stayed sealed until calibration-only selection was persisted. Only category counts were used to validate stratification before lock; holdout case outcomes were not inspected. The old `multi_05` case served only as historical motivation and is not part of this dataset.

### Frozen configurations

| Candidate | Judge prompt | Context policy |
|---|---|---|
| C2 control | `evidence-sufficiency-v1` | validated supporting chunks only |
| D | `evidence-coverage-v2` | validated union of requirement-level supporting chunks only |

The D prompt and strict JSON Schema were frozen before scoring with template hash `dbcce103d21a48254c69819959297fbf981464fb99b1760e5f01d316a34a9940`. D decomposes each question into concise atomic requirements with `SUPPORTED`, `PARTIAL`, `MISSING`, or `CONFLICTING` states. The server permits `answerable=true` only when every requirement is supported, every requirement has evidence, and the top-level supporting IDs equal the validated union. IDs must occur in the current authorized retrieval trace, active version, tenant, and ACL boundary; invalid output fails closed.

### Calibration: end-to-end view

| Metric | C2 | D |
|---|---:|---:|
| Accuracy | 0.650000 | 0.675000 |
| Precision | 1.000000 | 1.000000 |
| Recall | 0.500000 | 0.535714 |
| F1 | 0.666667 | 0.697674 |
| Correct answers | 14 | 15 |
| Correct abstentions | 12 | 12 |
| Unsupported answers | 0 | 0 |
| Incorrect abstentions | 14 | 13 |
| Multi-document answer success | 0.500000 | 0.535714 |
| Multi-document false-abstention rate | 0.500000 | 0.464286 |
| Required Evidence Recall | 0.378378 | 0.405405 |
| Required Evidence Precision | 0.378378 | 0.405405 |
| All-Required-Evidence Coverage Rate | 0.378378 | 0.405405 |
| Supporting Context Loss Rate | 0.189189 | 0.162162 |
| Citation validity | 1.000000 | 1.000000 |
| Conservative answer/source support | 0.428571 | 0.400000 |
| Judge request/format errors | 2 / 0 | 2 / 0 |

Frozen retrieval supplied all required documents in 21/40 calibration cases (0.525000). Retrieval misses remain end-to-end failures but are not attributed to the judge.

### Calibration: retrieval-complete primary gate view

| Metric | C2 | D |
|---|---:|---:|
| Cases | 21 | 21 |
| Accuracy | 0.666667 | 0.714286 |
| Precision | 1.000000 | 1.000000 |
| Recall | 0.666667 | 0.714286 |
| F1 | 0.800000 | 0.833333 |
| Required Evidence Recall | 0.666667 | 0.714286 |
| Required Evidence Precision | 0.666667 | 0.714286 |
| All-Required-Evidence Coverage Rate | 0.666667 | 0.714286 |
| Supporting Context Loss Rate | 0.333333 | 0.285714 |
| Version correctness | 1.000000 | 1.000000 |

D improved one answer and all-required coverage by 0.027027 end-to-end (0.047619 on the retrieval-complete subset), with no unsupported-answer regression. This did not cross the pre-frozen material-improvement rule, and D did not improve operational failures or conservative support. Therefore frozen C2 was retained and locked at `2026-08-16 23:14:30.766827+00:00` before holdout. No prompt, threshold, retrieval, or selection change was made after holdout became visible.

### New one-shot holdout

Because C2 remained selected, only one C2 judge execution was needed; redundant selected-candidate calls were avoided.

| Metric | C2 |
|---|---:|
| Cases | 20 |
| Accuracy | 0.650000 |
| Precision | 1.000000 |
| Recall | 0.533333 |
| F1 | 0.695652 |
| Correct answers / abstentions | 8 / 5 |
| Unsupported answers / incorrect abstentions | 0 / 7 |
| Multi-document success / false-abstention rate | 0.533333 / 0.466667 |
| Required Evidence Recall / Precision | 0.403509 / 0.403509 |
| All-Required-Evidence Coverage Rate | 0.368421 |
| Supporting Context Loss Rate | 0.105263 |
| Retrieval-complete cases | 9 (0.450000) |
| Retrieval-complete accuracy / F1 | 0.888889 / 0.941176 |
| Retrieval-complete Required Evidence Recall / Precision | 0.851852 / 0.851852 |
| Retrieval-complete All-Required-Evidence Coverage | 0.777778 |
| Retrieval-complete Supporting Context Loss | 0.222222 |

Generalization is **GOOD** descriptively: unsupported answers remained zero, multi-document success rose from 0.500000 to 0.533333, false abstention fell from 0.500000 to 0.466667, security held, and operational errors fell from two to zero. All-required coverage changed only slightly end-to-end; no statistical-significance claim is made.

### Latency, cache, and usage

| Split/candidate | Judge ms | Gate-cache ms | Context pruning ms | Generation ms | Total ms | Gate H/M | Judge calls | Judge input/output tokens | Embedding calls/tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Calibration C2 | 1425.175 | 1.835 | 2655.759 | 0.049 | 2664.479 | 0/40 | 40 | 25802/2843 | 40/674 |
| Calibration D | 1692.190 | 0.775 | 3867.164 | 0.040 | 3872.701 | 0/40 | 40 | 30135/6683 | 0/0 |
| Holdout C2 | 1291.806 | 1.371 | 1302.249 | 0.044 | 1309.851 | 0/20 | 20 | 12892/1556 | 20/364 |

Each of the 60 new questions was embedded once and cached; D reused all 40 calibration query embeddings from C2. This phase used 100 hosted judge calls (C2 60, D 40) and 60 embedding calls. Cumulative persisted usage is 186/206 hosted judge calls and 144/144 external embedding calls. Cumulative OpenAI gate caches contain 146 v1 decisions and 40 v2 decisions; prompt-version identity prevents cross-use. No monetary cost is invented because no verified local pricing configuration exists.

### Security, failures, and limitations

ACL safety, version correctness on the retrieval-complete subset, and prompt-injection boundary were 1.000000 for both candidates; unauthorized evidence selections were zero. Overall calibration version behavior was 0.600000 because Top-5 missed required versioned evidence, while the judge-only retrieval-complete view was 1.000000. Operational failures were cached, not retried, and failed closed.

The largest measured quality problem is incomplete Top-5 retrieval coverage (0.525000 calibration and 0.450000 holdout), which constrains both judges before evidence selection. Requirement-level coverage showed a small directional improvement but did not materially outperform frozen C2 under the precommitted policy. The next experiment should test one frozen retrieval-coverage intervention on a new sealed dataset; this report does not execute it.

## Dense vs BM25 vs Hybrid Retrieval Benchmark

This retrieval-only experiment compares the frozen dense control with a deterministic BM25 diagnostic and a fixed Reciprocal Rank Fusion candidate. It does not change the corpus, chunks, embeddings, threshold, C2 judge, generation, ACL model, versions, query rewrite, reranking, or Top-5 output size. **No LLM evidence-judge calls were used for retriever selection.**

### Dataset methodology and sealed split

`acmeai-hybrid-retrieval-eval-v1` contains 60 new questions grounded in the active corpus: 12 two-document, 8 three-document, 10 exact-identifier, 8 version/region-sensitive, 8 near-duplicate discrimination, 6 semantic paraphrase, 4 ACL-sensitive, and 4 partial/no-answer cases. The questions are not exact copies of earlier evaluation questions; normalized token-set comparison against both old datasets found a maximum Jaccard overlap of 0.360. Multi-document cases use new fact combinations and require distinct sources. Ground truth is benchmark-only and includes stable document identities, active version labels, access behavior, answerability, category, and stable chunk IDs only where available; database-generated chunk UUIDs were not treated as portable ground truth.

- Dataset SHA-256: `b935a5bcad77ea03128f280cf6dacb06cacad2c62b4c4ec0da477f37c7b43f3b`
- Explicit seed: `2026081703`
- Calibration / holdout: 40 / 20
- Split identity: `087dda3545a8027ddc1f4886ad9e123b61a852da2fe7052b77d34a3fd58c00e3`
- Corpus identity: `f78baa9c2d891d2be09383eba5830c3aa3bc0b510b702e83a38f1a36d7b21ff5`
- Semantic index identity: `e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466`

The 20-case holdout remained sealed until the calibration result, selection decision, configuration, and timestamp were persisted. All three frozen retrieval views then ran on holdout exactly once. No post-holdout tuning was permitted.

### Retrieval configurations

Dense uses `text-embedding-3-small`, 64 dimensions, pgvector cosine, threshold 0.28, and final Top-5. The control ranking behavior is unchanged. Hybrid uses dense depth 20 and BM25 depth 20, applies 0.28 only at the dense-branch boundary, deduplicates by canonical chunk ID, and fuses with unweighted RRF at fixed `k=60` before final Top-5. RRF values are stored separately and are never interpreted as cosine scores.

BM25 is an isolated BM25 Okapi implementation (`k1=1.2`, `b=0.75`). Tokenization uses Unicode NFKC, case folding, ordinary word tokens, and exact whole identifier tokens plus components (for example `POL-2026-004`, `Atlas-X17`, `API-V3`, and `West-Runbook`). There is no stemming, synonym expansion, external lexical service, or learned weighting. Tenant, ACL, and active-version predicates construct the candidate set before document frequencies and ranking are computed. Deterministic ordering uses score descending and chunk ID ascending.

### Calibration results and selection

| Metric | Dense | BM25 | Hybrid RRF |
|---|---:|---:|---:|
| Hit@5 | 1.000000 | 0.944444 | 1.000000 |
| Recall@5 | 0.912037 | 0.893519 | 0.953704 |
| MRR | 0.845833 | 0.812500 | 0.862037 |
| nDCG@5 | 0.807157 | 0.803197 | 0.856516 |
| Required Evidence Recall@5 | 0.912037 | 0.893519 | 0.953704 |
| All Required Evidence Coverage@5 | 0.805556 | 0.805556 | 0.888889 |
| Two-document Coverage@5 | 0.625000 | 0.875000 | 0.750000 |
| Three-document Coverage@5 | 0.200000 | 0.200000 | 0.600000 |
| Exact Identifier Recall@5 | 1.000000 | 1.000000 | 1.000000 |
| Version-sensitive Recall@5 | 1.000000 | 1.000000 | 1.000000 |
| Near-duplicate preferred-source success | 0.800000 | 0.600000 | 0.600000 |
| ACL safety | 1.000000 | 1.000000 | 1.000000 |
| Unauthorized results | 0 | 0 | 0 |

The precommitted rule required Hybrid to improve All Required Evidence Coverage@5 by at least 0.10 absolute, without required-recall or semantic regression and with ACL safety 1.000000. Hybrid improved coverage by 0.083333, improved required recall, produced no semantic regression, and preserved ACL safety, but missed the materiality threshold. `DENSE` was therefore locked at `2026-08-17 08:49:48.912629+00:00` before holdout.

### Sealed holdout results

| Metric | Dense | BM25 | Hybrid RRF |
|---|---:|---:|---:|
| Hit@5 | 1.000000 | 0.888889 | 1.000000 |
| Recall@5 | 0.916667 | 0.888889 | 0.944444 |
| MRR | 0.842593 | 0.861111 | 0.916667 |
| nDCG@5 | 0.824607 | 0.840030 | 0.890812 |
| Required Evidence Recall@5 | 0.916667 | 0.888889 | 0.944444 |
| All Required Evidence Coverage@5 | 0.833333 | 0.888889 | 0.888889 |
| Two-document Coverage@5 | 0.750000 | 1.000000 | 1.000000 |
| Three-document Coverage@5 | 0.333333 | 1.000000 | 0.333333 |
| Exact Identifier Recall@5 | 1.000000 | 1.000000 | 1.000000 |
| Version-sensitive Recall@5 | 1.000000 | 1.000000 | 1.000000 |
| Near-duplicate preferred-source success | 0.666667 | 0.666667 | 1.000000 |
| ACL safety | 1.000000 | 1.000000 | 1.000000 |
| Unauthorized results | 0 | 0 | 0 |

On sealed holdout, Hybrid's coverage gain over Dense was 0.055556, again below the frozen +0.10 rule. The selected Dense retriever generalized **GOOD** descriptively: calibration-to-holdout All Required Coverage changed from 0.805556 to 0.833333, required recall from 0.912037 to 0.916667, exact-ID recall remained 1.000000, and security remained perfect. No statistical-significance claim is made.

### Exact-identifier ranks

| Split | Case | Required source | Dense | BM25 | Hybrid |
|---|---|---|---:|---:|---:|
| Calibration | `hyb_id_02` | engineering deployment | 3 | 1 | 1 |
| Calibration | `hyb_id_03` | security training | 1 | 1 | 1 |
| Calibration | `hyb_id_05` | finance expense | 1 | 1 | 1 |
| Calibration | `hyb_id_06` | data retention | 1 | 1 | 1 |
| Calibration | `hyb_id_08` | west recovery | 1 | 2 | 2 |
| Calibration | `hyb_id_09` | HR compensation | 1 | 1 | 1 |
| Calibration | `hyb_id_10` | HR benefits | 1 | 1 | 1 |
| Holdout | `hyb_id_01` | support escalation | 1 | 1 | 1 |
| Holdout | `hyb_id_04` | Atlas API | 1 | 1 | 1 |
| Holdout | `hyb_id_07` | east recovery | 2 | 1 | 1 |

BM25 improved the dense rank in two identifier cases, tied in seven, and ranked one west-recovery case one position lower. All modes achieved 1.000000 Exact Identifier Recall@5, so lexical retrieval did not improve identifier recall at Top-5 on this dataset.

### Multi-document evidence accounting

Each cell is `required evidence retrieved / all required`. This is document-level required evidence; repeated chunks from one document do not satisfy a missing source.

| Split | Case | Required | Dense | BM25 | Hybrid |
|---|---|---:|---|---|---|
| Calibration | `hyb_two_02` | 2 | 2 / yes | 2 / yes | 2 / yes |
| Calibration | `hyb_two_04` | 2 | 2 / yes | 2 / yes | 2 / yes |
| Calibration | `hyb_two_05` | 2 | 2 / yes | 2 / yes | 2 / yes |
| Calibration | `hyb_two_06` | 2 | 1 / no | 2 / yes | 2 / yes |
| Calibration | `hyb_two_07` | 2 | 2 / yes | 2 / yes | 2 / yes |
| Calibration | `hyb_two_08` | 2 | 2 / yes | 2 / yes | 2 / yes |
| Calibration | `hyb_two_10` | 2 | 1 / no | 2 / yes | 1 / no |
| Calibration | `hyb_two_11` | 2 | 1 / no | 1 / no | 1 / no |
| Calibration | `hyb_three_01` | 3 | 3 / yes | 3 / yes | 3 / yes |
| Calibration | `hyb_three_03` | 3 | 1 / no | 2 / no | 2 / no |
| Calibration | `hyb_three_04` | 3 | 2 / no | 2 / no | 3 / yes |
| Calibration | `hyb_three_07` | 3 | 2 / no | 2 / no | 2 / no |
| Calibration | `hyb_three_08` | 3 | 2 / no | 2 / no | 3 / yes |
| Holdout | `hyb_two_01` | 2 | 1 / no | 2 / yes | 2 / yes |
| Holdout | `hyb_two_03` | 2 | 2 / yes | 2 / yes | 2 / yes |
| Holdout | `hyb_two_09` | 2 | 2 / yes | 2 / yes | 2 / yes |
| Holdout | `hyb_two_12` | 2 | 2 / yes | 2 / yes | 2 / yes |
| Holdout | `hyb_three_02` | 3 | 1 / no | 3 / yes | 1 / no |
| Holdout | `hyb_three_05` | 3 | 3 / yes | 3 / yes | 3 / yes |
| Holdout | `hyb_three_06` | 3 | 2 / no | 3 / yes | 2 / no |

### Branch contribution and semantic regression

On calibration, BM25 rescued 4 required evidence items absent from Dense Top-5; Dense rescued 5 absent from BM25 Top-5; 66 final required-evidence chunk appearances were present in both depth-20 branches. On holdout, these counts were 1, 2, and 31. Candidate-depth provenance is distinct from final Top-5 rescue accounting. Hybrid introduced zero Dense-success/Hybrid-failure cases in the semantic-paraphrase category. Dense and Hybrid covered every semantic case on both splits; standalone BM25 missed two of four calibration semantic cases and both holdout semantic cases.

### Failure analysis

Actual stored traces show that remaining Hybrid failures were predominantly fusion ranking failures, not security or version failures. On calibration, Hybrid missed four answerable cases: `hyb_two_10` (engineering deployment displaced), `hyb_two_11` (finance expense displaced), `hyb_three_03` (data retention displaced), and `hyb_three_07` (remote-work evidence displaced). All four required sources existed inside the union of depth-20 branch candidates but landed outside final Top-5 (`RANKING_OUTSIDE_TOP5` + `HYBRID_MISS`).

On holdout, Hybrid missed two three-document cases. For `hyb_three_02`, finance and security evidence were outside Top-5 behind Atlas launch, continuity, Atlas API, and deployment chunks. For `hyb_three_06`, data-retention evidence was outside Top-5 behind deployment, Atlas launch, two Atlas API chunks, and west recovery. The selected Dense view missed three holdout cases: `hyb_two_01` (deployment), `hyb_three_02` (finance and security), and `hyb_three_06` (data retention). Standalone BM25's only holdout failures were synonym-heavy `hyb_sem_04` and `hyb_sem_05`, both `RANKING_OUTSIDE_TOP5` + `LEXICAL_MISS`. There were no exact-identifier misses, version-evidence misses, ACL exposures, or inactive-version results.

### Latency, index overhead, cache, and external usage

| Split | Query embedding | Cache lookup | Dense search | BM25 total | BM25 build | BM25 scoring | RRF fusion | Dense total | Hybrid total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Calibration | 338.844 ms | 1.315 ms | 5.565 ms | 6.324 ms | 1.161 ms | 0.130 ms | 0.106 ms | 345.724 ms | 352.154 ms |
| Holdout | 284.019 ms | 1.252 ms | 5.837 ms | 5.737 ms | 1.101 ms | 0.141 ms | 0.103 ms | 291.108 ms | 296.948 ms |

Totals reflect cold query-cache misses and therefore include the one external query embedding for Dense/Hybrid. BM25 requires no embedding. The in-process, ACL-specific BM25 representation used at most 5,832 serialized bytes on calibration and 5,328 bytes on holdout; it is rebuilt over the prefiltered candidate set and adds no persistent database storage or external infrastructure. The dense HNSW/pgvector index is unchanged.

All 60 questions were cache misses on their first and only paid embedding, then persisted for reuse. Calibration used 40 calls / 770 embedding tokens and holdout used 20 calls / 379 tokens: 60 calls / 1,149 tokens total. Query cache grew from 144 to 204 matching entries. Document embedding calls were zero. Hosted judge calls were zero; the historical hosted-judge ledger remained 186/206. The cumulative embedding ledger ended at the authorized 204/204 ceiling.

### Security and limitations

ACL safety, tenant filtering, and active-version correctness were 1.000000 for every mode on both splits, with zero unauthorized result exposures. Lexical statistics were computed only after tenant/permission/version filtering. Tests cover ACL and tenant isolation with adversarial candidates, inactive-version exclusion, deterministic fusion and tokenization, score-type separation, branch provenance, locks, and the no-judge invariant.

This is a small synthetic corpus and descriptive one-shot benchmark. BM25 is intentionally minimal and in-memory; no infrastructure cost is inferred. Hybrid improved several metrics directionally but did not meet the frozen materiality policy on calibration or sealed holdout. The largest remaining measured retrieval problem is final Top-5 coverage for three-document questions: selected Dense reached 0.333333 on holdout, and Hybrid also reached 0.333333 because duplicated/highly similar chunks displaced distinct required sources.

## Dense Cross-Encoder Reranking Benchmark

This retrieval-only experiment changes one quality variable: frozen Dense ordering versus local Cross-Encoder ordering of the same authorized Dense Top-20 trace. It does not change the corpus, chunks, active versions, ACL/tenant boundary, `text-embedding-3-small` 64-dimensional embeddings, pgvector index, cosine threshold 0.28, final Top-5, C2 judge, or generation. Hybrid/BM25, query rewrite, expansion, larger context, and alternate rerankers were not evaluated. **No LLM evidence-judge calls were used for reranker selection.**

### Dataset methodology and sealed split

`acmeai-reranking-eval-v1` contains 60 new corpus-grounded questions: 10 two-document, 16 three-document, 8 near-duplicate discrimination, 6 exact-identifier, 6 version/region-sensitive, 6 semantic paraphrase, 4 ACL-sensitive, and 4 partial/no-answer cases. Benchmark-only ground truth contains required documents, stable chunks where available, versions, access behavior, answerability, and category; neither Dense retrieval nor the Cross-Encoder receives these labels. Maximum normalized token-set Jaccard overlap with the three earlier datasets is 0.380952, below the repository guard of 0.5.

- Dataset SHA-256: `36ad031f08f4dbbc0a1c67ab292eda549cf7291e91807e2cd0ba4f89b051b12f`
- Explicit deterministic seed: `2026081711`
- Calibration / holdout: 40 / 20
- Split identity: `ea2110dbced9a208c0a40f7feca6d4c5aa36253e83432c7650cc7ba4b8de88f0`
- Corpus identity: `f78baa9c2d891d2be09383eba5830c3aa3bc0b510b702e83a38f1a36d7b21ff5`
- Semantic index identity: `e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466`

Selection was persisted at `2026-08-17 09:41:57.414456+00:00`. Holdout started later at `09:42:06.667844+00:00` and completed once at `09:42:16.512052+00:00`; no configuration or selection tuning followed.

### Frozen configurations

| Candidate | Candidate generation | Final ordering |
|---|---|---|
| `DENSE` | authorized/current-version pgvector cosine, threshold 0.28, depth 20 | original Dense first 5 |
| `DENSE_CROSS_ENCODER_RERANK` | identical persisted Dense trace | local Cross-Encoder descending logit, first 5 |

The model is exactly `cross-encoder/ms-marco-MiniLM-L6-v2`, resolved revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, through Sentence Transformers `CrossEncoder` / PyTorch on CPU (`aarch64`). Reranker scores remain a separate type: they are not embeddings, are not compared with cosine similarity, and never receive the 0.28 threshold. Ties resolve deterministically by original Dense rank and canonical chunk ID. ACL, tenant, and active-version filtering occurs inside Dense retrieval before chunk text reaches the local model.

### Candidate-pool coverage

| Metric | Calibration Top-20 | Holdout Top-20 |
|---|---:|---:|
| Required Evidence Recall@20 | 0.990741 | 0.972222 |
| All Required Evidence Coverage@20 | 0.972222 | 0.944444 |
| Two-document Coverage@20 | 1.000000 | 0.666667 |
| Three-document Coverage@20 | 0.909091 | 1.000000 |
| Mean surviving candidates | 9.125 | 8.600 |
| Version correctness@20 | 1.000000 | 1.000000 |
| ACL safety@20 | 1.000000 | 1.000000 |

The frozen threshold left fewer than 20 candidates on average. One calibration and one holdout case lacked all required evidence in the surviving pool and were impossible for the reranker to repair.

### Calibration results and selection

| Metric | Dense | Dense + Cross-Encoder |
|---|---:|---:|
| Hit@5 | 1.000000 | 1.000000 |
| Recall@5 | 0.916667 | 0.990741 |
| MRR | 0.935185 | 0.967593 |
| nDCG@5 | 0.877745 | 0.957576 |
| Required Evidence Recall@5 | 0.916667 | 0.990741 |
| All Required Evidence Coverage@5 | 0.777778 | 0.972222 |
| Two-document Coverage@5 | 0.714286 | 1.000000 |
| Three-document Coverage@5 | 0.454545 | 0.909091 |
| Exact Identifier Recall@5 | 1.000000 | 1.000000 |
| Version-sensitive Recall@5 | 1.000000 | 1.000000 |
| Near-duplicate preferred-source success | 0.600000 | 0.800000 |
| Semantic/paraphrase all-required success | 1.000000 | 1.000000 |
| ACL safety | 1.000000 | 1.000000 |

The precommitted policy selected the reranker: All Required Evidence Coverage@5 gained 0.194444 (at least +0.10), and Three-document Coverage@5 gained 0.454545 (at least +0.20). Required recall improved; exact-ID and semantic performance did not regress; version correctness and ACL safety remained 1.000000. `DENSE_CROSS_ENCODER_RERANK` was locked before holdout.

### Rerankable subset

This view includes only answerable cases where every required source exists in Dense Top-20: 35 calibration and 17 holdout cases.

| Split / metric | Dense | Dense + Cross-Encoder |
|---|---:|---:|
| Calibration All Required Coverage@5 | 0.800000 | 1.000000 |
| Calibration Required Evidence Recall@5 | 0.923810 | 1.000000 |
| Calibration Three-document Coverage@5 | 0.500000 | 1.000000 |
| Calibration MRR / nDCG@5 | 0.933333 / 0.880956 | 0.966667 / 0.963068 |
| Holdout All Required Coverage@5 | 0.823529 | 0.941176 |
| Holdout Required Evidence Recall@5 | 0.911765 | 0.980392 |
| Holdout Three-document Coverage@5 | 0.600000 | 0.800000 |
| Holdout MRR / nDCG@5 | 0.960784 / 0.894261 | 1.000000 / 0.969620 |

This isolates ordering quality rather than attributing Top-20 candidate misses to the Cross-Encoder.

### Sealed holdout results

| Metric | Dense | Dense + Cross-Encoder |
|---|---:|---:|
| Hit@5 | 1.000000 | 1.000000 |
| Recall@5 | 0.888889 | 0.953704 |
| MRR | 0.962963 | 1.000000 |
| nDCG@5 | 0.878644 | 0.949816 |
| Required Evidence Recall@5 | 0.888889 | 0.953704 |
| All Required Evidence Coverage@5 | 0.777778 | 0.888889 |
| Two-document Coverage@5 | 0.333333 | 0.666667 |
| Three-document Coverage@5 | 0.600000 | 0.800000 |
| Exact Identifier Recall@5 | 1.000000 | 1.000000 |
| Version-sensitive Recall@5 | 1.000000 | 1.000000 |
| Near-duplicate preferred-source success | 0.666667 | 1.000000 |
| Semantic/paraphrase all-required success | 1.000000 | 1.000000 |
| ACL safety | 1.000000 | 1.000000 |

The selected candidate generalized **GOOD** descriptively. Calibration-to-holdout all-required coverage changed from 0.972222 to 0.888889, three-document coverage from 0.909091 to 0.800000, and required recall from 0.990741 to 0.953704 while remaining above sealed Dense. Exact-ID, semantic, version, and security constraints held. This is not a statistical-significance claim.

### Rank movement and three-document analysis

Movement is aggregated by required canonical evidence source while every underlying chunk-level Dense/reranker score and rank is persisted. Calibration promoted 7 required sources into Top-5, demoted 0, removed 42 irrelevant Dense Top-5 chunk appearances, and produced a mean required-evidence rank delta of +0.703125. Holdout promoted 4, demoted 1, removed 20 irrelevant appearances, and produced a mean delta of +0.468750. Required-source ranks below are `Dense→reranked`; `none` means absent from thresholded Top-20.

| Split | Case | Required-source ranks | Dense all | Reranked all |
|---|---|---|---:|---:|
| Calibration | `rer_three_01` | continuity 2→3; east recovery 1→1; support 6→4 | 0 | 1 |
| Calibration | `rer_three_04` | incident 2→3; continuity 1→2; support 3→1 | 1 | 1 |
| Calibration | `rer_three_05` | east recovery 2→3; west recovery 3→2; Atlas launch 1→1 | 1 | 1 |
| Calibration | `rer_three_06` | retention 1→1; deployment none; expense 2→2 | 0 | 0 |
| Calibration | `rer_three_08` | remote policy 1→3; support 2→1; retention 3→2 | 1 | 1 |
| Calibration | `rer_three_09` | Atlas launch 1→1; Atlas API 3→3; expense 2→2 | 1 | 1 |
| Calibration | `rer_three_10` | continuity 3→4; retention 4→3; west recovery 1→1 | 1 | 1 |
| Calibration | `rer_three_11` | deployment 7→1; incident 3→2; support 1→4 | 0 | 1 |
| Calibration | `rer_three_12` | Atlas launch 1→1; continuity 5→2; east recovery 6→3 | 0 | 1 |
| Calibration | `rer_three_13` | remote policy 9→3; deployment 1→2; Atlas API 3→1 | 0 | 1 |
| Calibration | `rer_three_14` | expense 3→3; retention 1→1; support 8→2 | 0 | 1 |
| Holdout | `rer_three_02` | Atlas API 1→2; Atlas launch 2→4; deployment 4→1 | 1 | 1 |
| Holdout | `rer_three_03` | remote policy 3→2; expense 2→4; retention 1→1 | 1 | 1 |
| Holdout | `rer_three_07` | Atlas API 1→2; incident 9→1; west recovery 6→3 | 0 | 1 |
| Holdout | `rer_three_15` | east recovery 2→2; west recovery 1→1; incident 7→4 | 0 | 1 |
| Holdout | `rer_three_16` | Atlas API 1→1; retention 2→3; expense 4→7 | 1 | 0 |

Calibration fixed five three-document cases, worsened zero, left one failed, and had one impossible because required evidence was outside Top-20. Holdout fixed two, worsened one, left one failed, and had zero three-document cases outside Top-20. The holdout regression `rer_three_16` demoted expense evidence from rank 4 to 7.

### Exact identifiers, versions, semantic cases, and near-duplicates

All six exact-identifier cases retained their required source in Top-5: every required source was Dense rank 1 and reranked rank 1. Exact Identifier Recall@5 remained 1.000000 on both splits. Version-sensitive Recall@5 and version correctness remained 1.000000 for both candidates on both splits. All semantic/paraphrase cases remained covered; there were zero Dense-success/reranker-failure semantic cases.

| Split | Near-duplicate case | Preferred and competing ranks (`Dense→reranked`) |
|---|---|---|
| Calibration | `rer_near_02` | west recovery 2→1; east recovery 1→2 |
| Calibration | `rer_near_03` | current remote policy 1→1 |
| Calibration | `rer_near_06` | Atlas API 1→1; Atlas launch 3→3 |
| Calibration | `rer_near_07` | retention 1→1; expense 2→2 |
| Calibration | `rer_near_08` | expense 2→2; retention 1→1 |
| Holdout | `rer_near_01` | east recovery 1→1; west recovery 2→2 |
| Holdout | `rer_near_04` | incident policy 1→1 |
| Holdout | `rer_near_05` | Atlas launch 3→1; Atlas API 1→2 |

Preferred-source success improved from 0.600000 to 0.800000 on calibration and from 0.666667 to 1.000000 on holdout; no near-duplicate reranker failure remained.

### Failure analysis

The selected calibration candidate had one real miss: `rer_three_06` was `CANDIDATE_GENERATION_MISS`, `REQUIRED_EVIDENCE_OUTSIDE_TOP20`, and `THREE_DOCUMENT_COVERAGE_FAILURE` because deployment evidence did not survive threshold 0.28. The selected holdout candidate had two misses. `rer_two_09` lacked finance-expense evidence in Top-20 (`CANDIDATE_GENERATION_MISS`, `REQUIRED_EVIDENCE_OUTSIDE_TOP20`). `rer_three_16` contained all three sources but demoted expense from 4 to 7 (`RANKING_OUTSIDE_TOP5`, `RERANKER_FAILED_TO_PROMOTE`, `RERANKER_DEMOTED_REQUIRED_EVIDENCE`, `THREE_DOCUMENT_COVERAGE_FAILURE`). There were no ACL, version, exact-identifier, semantic, or near-duplicate failures.

### Latency and local resource usage

| Split | Query embedding | Cache lookup | Dense Top-20 | Cross-Encoder mean | Reranker p50 / p95 | Sort | Dense total | Reranked total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Calibration | 326.556 ms | 0.578 ms | 4.921 ms | 76.299 ms | 67.613 / 125.728 ms | 0.004 ms | 332.055 ms | 408.358 ms |
| Holdout | 291.574 ms | 0.597 ms | 5.577 ms | 67.913 ms | 67.062 / 92.877 ms | 0.004 ms | 297.749 ms | 365.665 ms |

The local snapshot contains 273,581,448 measurable bytes and was downloaded once into the persistent container cache; holdout reused it. Process maximum RSS increased from approximately 88.2 MiB before model load to 492.1 MiB after load (about 403.8 MiB); this process-scoped measurement is not an isolated model-memory benchmark. Calibration scored 365 pairs (9.125/query mean), and holdout scored 172 (8.600/query mean), on CPU. External reranking inference calls and cost were zero; total operational cost is not claimed to be zero.

### Cache, external usage, security, and limitations

Preflight observed 204 cumulative matching query embeddings, a ceiling of 204, 60 unique new queries, 0 cache matches, and 60 missing embeddings. The ignored local ceiling was then changed to the explicitly authorized 264. Calibration used 40 new calls / 706 tokens and holdout 20 / 347: 60 calls / 1,053 tokens total. Each query was embedded once and the identical result was reused for Dense and reranked views. The cache ended at 264. Document embedding calls were zero. New hosted judge calls were zero, the historical judge ledger remained 186/206, and external reranker calls were zero.

ACL safety, tenant isolation, active-version correctness, and Top-20 ACL safety were 1.000000. Unauthorized results and unauthorized chunks sent to the Cross-Encoder were both zero. The model saw only question/chunk-text pairs from the already filtered candidate trace.

This remains a small synthetic-corpus, one-model, one-shot descriptive benchmark. The Cross-Encoder was trained for passage ranking and may demote an otherwise necessary source in composite questions, as `rer_three_16` demonstrates. The largest remaining measured retrieval problem is candidate generation for composite evidence: two selected-candidate cases across calibration and holdout had a required source absent from thresholded Dense Top-20. The next experiment should use a new sealed end-to-end dataset to compare frozen Dense+C2 against frozen Dense+Cross-Encoder+C2; it is not executed here.

## Dense + Cross-Encoder + C2 End-to-End Benchmark

The reranker and C2 judge were frozen before this end-to-end evaluation. This was a single sealed paired evaluation, not a calibration or model-selection split. Pipeline A was frozen Dense cosine retrieval at threshold 0.28 and Top-5. Pipeline B used the same cached query embedding and authorized thresholded Dense Top-20 candidates, local `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a` on CPU, and reranked Top-5. Both then used OpenAI `gpt-5.6-luna`, judge version 1, prompt `evidence-sufficiency-v1`, validated supporting chunks only, and the deterministic extractive generator. No component was tuned after execution began.

### Dataset isolation and freeze

`acmeai-reranker-e2e-eval-v1` contains 60 new manually corpus-grounded cases: 12 two-document, 16 three-document, 8 near-duplicate, 6 exact-identifier, 6 version/region, 4 semantic/paraphrase, 4 ACL-sensitive, and 4 partial-evidence/no-answer cases. Its SHA-256 is `288b26b0f1adc9c75b2b1d2c617c362f9aa0540ebb5591157bd728ff94fa2e16`; maximum token-Jaccard overlap against every prior evaluation question is 0.391304. The dataset identity, ordered case IDs, generation method `manual-corpus-grounded-v1`, frozen component configurations, and precommitted success policy were persisted before preparation. The one-shot preparation and execution timestamps prevent reuse or post-result tuning.

The semantic index identity remained `e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466`; the active corpus identity remained `f78baa9c2d891d2be09383eba5830c3aa3bc0b510b702e83a38f1a36d7b21ff5`. Historical benchmark rows and consumed holdout locks were unchanged.

### Retrieval and end-to-end results

| Metric | Dense + C2 | Dense + Cross-Encoder + C2 | Delta |
|---|---:|---:|---:|
| Hit@5 | 1.000000 | 1.000000 | 0.000000 |
| Recall@5 / Required Evidence Recall@5 | 0.903846 | 0.961538 | +0.057692 |
| MRR | 0.866026 | 0.990385 | +0.124359 |
| nDCG@5 | 0.822522 | 0.941815 | +0.119293 |
| All Required Evidence Coverage@5 | 0.769231 | 0.884615 | +0.115384 |
| Two-document Coverage@5 | 0.833333 | 1.000000 | +0.166667 |
| Three-document Coverage@5 | 0.375000 | 0.625000 | +0.250000 |
| Exact-ID Recall@5 | 1.000000 | 1.000000 | 0.000000 |
| Version-sensitive Recall@5 | 1.000000 | 1.000000 | 0.000000 |
| Near-duplicate preferred-source success | 1.000000 | 1.000000 | 0.000000 |

| Final behavior | Dense + C2 | Dense + Cross-Encoder + C2 | Delta |
|---|---:|---:|---:|
| Correct answers | 30 | 38 | +8 |
| Correct abstentions | 8 | 8 | 0 |
| Unsupported answers | 0 | 0 | 0 |
| Incorrect abstentions | 22 | 14 | -8 |
| Answerability accuracy | 0.633333 | 0.766667 | +0.133334 |
| Answerability precision | 1.000000 | 1.000000 | 0.000000 |
| Answerability recall | 0.576923 | 0.730769 | +0.153846 |
| Answerability F1 | 0.731707 | 0.844444 | +0.112737 |

The precommitted production policy passed: correct answers increased by eight, incorrect abstentions decreased by eight, unsupported answers did not increase, Answerability F1 improved, and all hard security/version constraints remained 1.000000. The observed direction is consistent with the persisted retrieval traces rather than architecture preference alone.

### Retrieval-to-answer conversion and paired transitions

There were seven `RERANKER_RETRIEVAL_RESCUE` cases. Five became correct answers (`e2e_two_01`, `e2e_two_04`, `e2e_three_04`, `e2e_three_08`, `e2e_three_16`); two remained incorrect abstentions because C2 returned not-answerable despite complete B retrieval (`e2e_three_07`, `e2e_three_11`). Conversion was therefore 5/7, or 0.714286. One coverage regression, `e2e_three_03`, demoted required evidence from a complete A Top-5 to an incomplete B Top-5; both pipelines already abstained, so it caused no additional final-behavior loss.

| Paired A → B transition | Count |
|---|---:|
| Correct answer → correct answer | 28 |
| Incorrect abstention → correct answer | 10 |
| Incorrect abstention → incorrect abstention | 12 |
| Correct answer → incorrect abstention | 2 |
| Correct abstention → correct abstention | 8 |

The two answer-to-abstention transitions were `e2e_three_13` and `e2e_near_06`. Both retained required-source coverage, but their changed ordered evidence caused isolated C2 decisions and a false negative; they are downstream evidence-selection/judge regressions, not unauthorized cache sharing.

### Three-document paired analysis

Each cell is `retrieval complete; C2 decision; supporting chunk-ID prefixes; final behavior`. Full UUIDs, ordered retrieval/reranking traces, judge traces, and generation-context IDs are persisted in the comparison API.

| Case | Dense + C2 | Dense + Cross-Encoder + C2 |
|---|---|---|
| `e2e_three_01` | no; not answerable; none; incorrect abstention | no; not answerable; none; incorrect abstention |
| `e2e_three_02` | yes; not answerable; none; incorrect abstention | yes; answerable; `803b1d29`, `01423f11`, `7bdffe91`; correct answer |
| `e2e_three_03` | yes; not answerable; none; incorrect abstention | no; not answerable; none; incorrect abstention |
| `e2e_three_04` | no; not answerable; none; incorrect abstention | yes; answerable; `054089a4`, `7e6c9c91`, `39622c27`; correct answer |
| `e2e_three_05` | yes; answerable; `054089a4`, `ab9cf7c9`, `bd832158`; correct answer | yes; answerable; same three; correct answer |
| `e2e_three_06` | yes; answerable; `01423f11`, `7e6c9c91`, `2798f984`; correct answer | yes; answerable; same three; correct answer |
| `e2e_three_07` | no; not answerable; none; incorrect abstention | yes; not answerable; none; incorrect abstention |
| `e2e_three_08` | no; not answerable; none; incorrect abstention | yes; answerable; `7bdffe91`, `2aafc123`, `3f2b1506`; correct answer |
| `e2e_three_09` | no; not answerable; none; incorrect abstention | no; not answerable; none; incorrect abstention |
| `e2e_three_10` | no; not answerable; none; incorrect abstention | no; not answerable; none; incorrect abstention |
| `e2e_three_11` | no; not answerable; none; incorrect abstention | yes; not answerable; none; incorrect abstention |
| `e2e_three_12` | yes; not answerable; none; incorrect abstention | yes; not answerable; none; incorrect abstention |
| `e2e_three_13` | yes; answerable; `ab9cf7c9`, `01423f11`, `9e94a1f1`; correct answer | yes; not answerable; none; incorrect abstention |
| `e2e_three_14` | no; not answerable; none; incorrect abstention | no; not answerable; none; incorrect abstention |
| `e2e_three_15` | no; not answerable; none; incorrect abstention | no; not answerable; none; incorrect abstention |
| `e2e_three_16` | no; not answerable; none; incorrect abstention | yes; answerable; `2798f984`, `ab9cf7c9`, `3f2b1506`; correct answer |

Three-document retrieval-complete rate improved 0.375000→0.625000, correct-answer rate improved 0.187500→0.375000, and incorrect-abstention rate fell 0.812500→0.625000; unsupported-answer rate remained 0.000000. Two-document retrieval-complete rate improved 0.833333→1.000000, correct-answer rate improved 0.583333→1.000000, incorrect-abstention rate fell 0.416667→0.000000, and unsupported-answer rate remained 0.000000.

### Exact identifiers, versions, near-duplicates, and abstention safety

Exact-ID retrieval stayed 1.000000 for A and B; each produced five correct answers and one C2 incorrect abstention, for final success 0.833333. Version-sensitive retrieval and active-version correctness stayed 1.000000; each produced five correct answers and one C2 incorrect abstention, so final answer/version success was 0.833333 while retrieval version correctness was 1.000000. Near-duplicate preferred-source retrieval stayed 1.000000. Each pipeline produced seven correct answers and one incorrect abstention, but the failed case changed from `e2e_near_08` in A to `e2e_near_06` in B, showing that perfect preferred-source retrieval does not guarantee a stable C2 decision.

All eight should-abstain cases (four ACL-sensitive and four partial/no-answer) correctly abstained in both pipelines. Unsupported answers, C2 false positives, and unauthorized evidence selections were zero. B did promote irrelevant evidence in some no-answer traces, but frozen C2 rejected it; no unsupported final answer resulted.

### Grounding, citations, and failures

Citation validity was 1.000000 for both pipelines. Conservative answer/source support was 0.233333 for A and 0.210526 for B; the associated zero-support counts were 23 and 30. This exact-fact/source heuristic is intentionally conservative and does not measure semantic entailment. The increase in zero-support count partly follows B answering eight more cases; it must not be read as 30 proven hallucinations. Unsupported final answers remained zero, and no semantic-entailment claim is made.

B's remaining 14 incorrect abstentions break down by trace root causes: one candidate-generation miss (`e2e_three_15`), five ranking-outside-Top-5 cases (`e2e_three_01`, `09`, `10`, `14`, plus the reranker demotion `03`), and eight C2 false negatives after required retrieval remained complete or otherwise had sufficient evidence. Supporting-context loss accompanied six of those C2 false negatives. There were no generation, citation, ACL, version, judge-format, or judge-operational failures. Root cause and final behavior are persisted separately for every case.

### Latency, cache, and resource usage

| Mean unless noted | Dense + C2 | Dense + Cross-Encoder + C2 | Delta |
|---|---:|---:|---:|
| Query embedding | 424.128 ms | shared 424.128 ms | 0 |
| Query-cache lookup | 0.714 ms | shared 0.714 ms | 0 |
| Dense retrieval | 4.283 ms | shared 4.283 ms | 0 |
| Cross-Encoder inference | 0 | 105.995 ms | +105.995 ms |
| C2 gate | 1546.142 ms | 1535.142 ms | -10.999 ms |
| Context construction | 0.050 ms | 0.066 ms | +0.017 ms |
| Generation | 0.050 ms | 0.073 ms | +0.024 ms |
| Total mean | 1985.107 ms | 2082.940 ms | +97.833 ms |
| Total p50 | 1825.573 ms | 1937.146 ms | +111.573 ms |
| Total p95 | 2282.694 ms | 2624.875 ms | +342.181 ms |

Reranker inference p50/p95 was 91.637/225.040 ms. It scored 597 authorized local pairs, 9.95/query, on CPU at the frozen revision; external reranker calls were zero. All 60 query embeddings were cold during preparation and then shared by A and B. All 60 A gate inputs and 58 distinct B inputs were cold; two B inputs reused the just-created valid A cache entry because question, ordered evidence identity, and C2 configuration were identical. Different evidence inputs produced different keys. Thus the observed paths were cold embedding+cold judge for A, shared prepared embedding+cold judge for 58 B cases, and shared prepared embedding+cached judge for two B cases.

Embedding preflight began at 264 calls with 60 unique queries, zero matches, and a ceiling raised only in ignored local `.env` to 324. Actual new query embeddings were 60 / 1,078 tokens, ending at 324; document embeddings were zero. Judge preflight began at 186 calls, found 118 unique combined uncached inputs, and projected 304 against the authorized ceiling 306. Actual new Luna usage was 118 calls, 76,480 input tokens, and 8,375 output tokens, ending cumulative usage at 304. Gate-cache hits/misses were 2/118 across the paired run. The query cache ended at 324 records and the gate cache at 401.

### Security and limitations

For both pipelines ACL safety, tenant isolation, active-version correctness, and prompt-injection boundary were 1.000000. Unauthorized chunks sent to the Cross-Encoder, unauthorized chunks sent to Luna, and unauthorized evidence selections were all zero. Tenant, ACL, and active-version constraints were applied before Dense ranking, so B's local model saw only authorized candidates.

This is a 60-case synthetic-corpus, one-shot descriptive evaluation and is not a statistical-significance claim. The frozen generator is extractive and the conservative support heuristic does not score semantic answer correctness. Nevertheless, B met the precommitted end-to-end policy with a trace-consistent eight-answer gain and no unsupported-answer or security regression. The measured production retrieval path is therefore `DENSE_CROSS_ENCODER_RERANK`. The largest remaining end-to-end problem is C2 false-negative abstention after sufficient evidence retrieval: eight of B's 14 incorrect abstentions had this root cause. A single next experiment should compare the frozen C2 prompt with an Evidence Coverage v2 requirement-level judge on a new sealed end-to-end dataset, keeping the selected reranker and all other components frozen; it is not executed here.

## Frozen Evidence Sufficiency v1 vs Evidence Coverage v2

Both judge configurations were frozen before evaluation. This was a direct, one-shot comparison with no calibration, prompt tuning, or post-result configuration changes. The new manually authored, corpus-grounded sealed dataset is `acmeai-evidence-judge-e2e-eval-v1`: 60 cases comprising 12 two-document, 18 three-document, 6 near-duplicate, 6 exact-identifier, 6 version/region, 4 semantic/paraphrase, 4 ACL-sensitive, and 4 partial/no-answer cases. Its SHA-256 is `7d9b0e6f45e7d385be92ce5c37d229a1602957316e85bfe0c5f828cf514f9651`; maximum token-set overlap with any prior evaluation question is 0.470588, below the existing 0.5 guard.

The shared path remained `text-embedding-3-small` at 64 dimensions, pgvector cosine Dense Top-20 with the frozen 0.28 threshold, local `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, and reranked Top-5. One persisted ordered retrieval trace and one query embedding per case were shared by both arms. Tenant, ACL, and active-version filtering preceded both reranking and judging.

Judge A remained OpenAI `gpt-5.6-luna`, judge version 1, prompt `evidence-sufficiency-v1`, schema identity `6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621`. Judge B remained the historical frozen `evidence-coverage-v2` contract with prompt/schema hash `dbcce103d21a48254c69819959297fbf981464fb99b1760e5f01d316a34a9940`, statuses `SUPPORTED`, `PARTIAL`, `MISSING`, and `CONFLICTING`, all-requirements-supported invariant, validated requirement support, and validated top-level supporting-chunk union. Both fed only validated supporting chunks to the same deterministic extractive generator.

The precommitted policy required v2 to fix at least four A false abstentions or improve retrieval-complete recall by at least 0.10, without an unsupported-answer increase, overall F1 or correct-answer regression, or any security/version regression. V2 did not meet it, so the selected production judge remains `EVIDENCE_SUFFICIENCY_V1`.

### Overall and retrieval-complete results

| Final end-to-end metric | Sufficiency v1 | Coverage v2 | Delta v2−v1 |
|---|---:|---:|---:|
| Correct answers | 37 | 33 | -4 |
| Correct abstentions | 8 | 8 | 0 |
| Unsupported answers | 0 | 0 | 0 |
| Incorrect abstentions | 15 | 19 | +4 |
| Accuracy | 0.750000 | 0.683333 | -0.066667 |
| Precision | 1.000000 | 1.000000 | 0 |
| Recall | 0.711538 | 0.634615 | -0.076923 |
| F1 | 0.831461 | 0.776471 | -0.054990 |

All required evidence reached Top-5 for the same 48 cases; 12 were retrieval-incomplete. On the retrieval-complete subset, final v1/v2 accuracy and recall were 0.770833/0.687500, precision was 1.000000/1.000000, and F1 was 0.870588/0.814815. Correct answers were 37/33, incorrect abstentions 11/15, and unsupported answers 0/0. Measuring the judge decision itself before the common generator, v1/v2 had 39/35 true-positive answerable decisions and 9/13 false negatives; retrieval-complete accuracy and recall were 0.812500/0.729167, precision 1.000000/1.000000, and judge-decision F1 0.896552/0.843373. The common generator subsequently failed closed on the same two answerable decisions in both arms (`judge_three_01`, `judge_sem_01`).

On the 12 retrieval-incomplete cases, both arms produced eight correct abstentions, four answerable-case abstentions attributable to missing retrieval evidence, and zero unsupported answers. Neither judge answered despite incomplete evidence.

### Paired transitions and false-negative protection

| Final behavior transition, v1 → v2 | Count |
|---|---:|
| Correct answer → correct answer | 32 |
| Incorrect abstention → correct answer | 1 |
| Incorrect abstention → incorrect abstention | 14 |
| Correct answer → incorrect abstention | 5 |
| Correct abstention → correct abstention | 8 |
| Correct abstention → unsupported answer | 0 |
| Unsupported answer → correct abstention | 0 |
| Unsupported answer → unsupported answer | 0 |

The sole `V2_FALSE_NEGATIVE_RESCUE` was `judge_two_11`. Five v1 correct answers became v2 false abstentions: `judge_two_04`, `judge_three_16`, `judge_near_03`, `judge_near_04`, and `judge_id_05`. Final false negatives therefore moved 15→19, a net change of -4 rescues. There were no should-abstain false-positive regressions and no answers despite incomplete retrieval.

### Multi-document and requirement-level analysis

Two-document retrieval completeness was 0.916667. Both judges produced 9/12 correct answers, 3/12 incorrect abstentions, 0 unsupported answers, and final F1 0.857143; v2 did not systematically improve this category. Three-document retrieval completeness was 0.833333. V1 versus v2 correct-answer rate was 0.555556 versus 0.500000, incorrect-abstention rate 0.444444 versus 0.500000, unsupported-answer rate 0 versus 0, and overall category F1 0.714286 versus 0.666667. On the 15 retrieval-complete three-document cases, final F1 was 0.800000 for v1 and 0.750000 for v2; judge-decision F1 was 0.846154 and 0.800000, respectively.

For v2, Luna identified 134 requirements, mean 2.233333 per question: 105 `SUPPORTED`, 7 `PARTIAL`, 22 `MISSING`, and 0 `CONFLICTING`. Ten non-supported requirement judgments on retrieval-complete false-abstention cases were directly traceable to evidence that was present in Top-5: `reimbursable_cost_deadline`, `combined_schedule`, `incident_notification_limit`, `governing_role`, `severity_one_reporting_interval`, `overseas_travel_authorization_token`, `ENG-DEP-17` approval, `CS-1842` handling, `SEC-TRAIN-44` classification, and the current receipt cutoff. The evidence shows recognition/coverage failures rather than retrieval misses. One additional v2 false negative (`judge_three_12`) failed closed after an invalid supporting ID and had no valid requirement payload.

All 18 three-document cases retain their complete shared retrieval trace, A support IDs, B requirement-to-support mappings, decisions, and final behavior in `/experiments/judge-e2e-comparison`. Summary: retrieval was incomplete for cases 02, 05, and 14; both arms safely abstained. Cases 01 and 12 failed closed identically after an answerable decision/common generation failure and invalid support respectively. Both answered cases 04, 06–11, 13, and 17. Both judge paths abstained on complete evidence for 03, 15, and 18. Case 16 was the three-document regression: v1 answered correctly while v2 marked `severity_one_reporting_interval` partial and abstained.

### Exact identifiers, versions, near-duplicates, and abstention safety

Exact-identifier final success regressed from 4/6 (0.666667) to 3/6 (0.500000); `judge_id_05` changed from a v1 correct answer to a v2 false abstention. Version/region behavior was unchanged at 5/6 correct answers and 1/6 incorrect abstentions, with retrieval and final active-version correctness 1.000000 for both.

The preferred near-duplicate source was present in all six shared traces. V1 answered 6/6; v2 answered 4/6. In cases 03 and 04, v1 selected authoritative chunks `826df8a1` and `3f2b1506`/`39622c27`, respectively, while v2 selected none and abstained. The other four cases retained valid authoritative support and correct answers. Thus requirement-level reasoning introduced two downstream near-duplicate regressions even though retrieval was identical.

All eight should-abstain cases—four ACL-sensitive and four partial/no-answer—correctly abstained under both judges. Unsupported answers, false positives, and selected unauthorized evidence were zero for each arm.

### Citation quality, grounding, and failures

Citation validity and citation correctness were 1.000000 for both arms. Conservative exact answer/source support was 0.243243 over 37 v1 answered cases and 0.272727 over 33 v2 answered cases. The denominators differ, so the raw rates are not evidence that v2 improved semantic grounding. The heuristic recorded 28 and 24 unsupported exact-claim matches, respectively, but unsupported final answers under the evaluation ground truth were zero; no semantic-entailment claim is made.

Among v1's 15 final incorrect abstentions, root causes were 4 `RETRIEVAL_INCOMPLETE`, 9 `V1_EVIDENCE_GATE_FALSE_NEGATIVE`, and 2 `GENERATION_FAILURE`. Among v2's 19, causes were 4 `RETRIEVAL_INCOMPLETE`, 12 `V2_REQUIREMENT_COVERAGE_FALSE_NEGATIVE`, 1 `V2_REQUIREMENT_DECOMPOSITION_FALSE_NEGATIVE`, and the same 2 `GENERATION_FAILURE` cases. These classifications are derived from immutable run traces, and retrieval misses are not counted as judge failures.

### Latency, tokens, cache, and security

| Mean unless noted | Sufficiency v1 | Coverage v2 | Incremental v2 |
|---|---:|---:|---:|
| Shared query embedding | 313.203 ms | 313.203 ms | 0 |
| Shared cache lookup | 1.202 ms | 1.202 ms | 0 |
| Shared Dense retrieval | 5.113 ms | 5.113 ms | 0 |
| Shared Cross-Encoder | 122.020 ms | 122.020 ms | 0 |
| Judge | 1644.496 ms | 2159.805 ms | +515.309 ms |
| Validation/context | 12.123 ms | 11.592 ms | -0.531 ms |
| Generation | 0.095 ms | 0.061 ms | -0.034 ms |
| Total mean | 2098.373 ms | 2613.120 ms | +514.747 ms |
| Total p50 | 1976.847 ms | 2508.129 ms | +531.282 ms |
| Total p95 | 2777.791 ms | 3127.109 ms | +349.318 ms |

Embedding preflight began at 324 cumulative calls with 60 unique queries, 0 valid cache matches, 60 misses, and a locally configured ceiling of 384. Preparation made exactly 60 query calls using 1,115 tokens, zero document-embedding calls, and ended at 384. The prepared embeddings and 537 authorized local Cross-Encoder pairs (8.95/query) were reused by both arms; external reranker calls were zero.

Judge preflight began at 304 cumulative calls with a ceiling of 424. Each prompt namespace had 60 unique inputs, zero valid matches, and zero cross-prompt shared keys. Each arm made 60 Luna calls with 0 cache hits/60 misses. V1 used 38,265 input and 4,625 output tokens; v2 used 44,745 input and 10,813 output tokens. Cumulative judge use ended at 424.

ACL safety, tenant isolation, active-version correctness, and prompt-injection boundary were 1.000000 for both. Unauthorized chunks reaching the reranker, v1 judge, or v2 judge were zero; unauthorized supporting evidence selections were zero. One invalid support-ID output occurred in each arm on the same case and failed closed as required.

### Limitations and production decision

This is a 60-case synthetic-corpus, one-shot descriptive benchmark, not a statistical-significance claim. Retrieval completeness uses hidden document/chunk ground truth; judges never received it. The extractive generator and conservative lexical support heuristic limit claims about semantic answer quality. Even so, the paired trace evidence is unambiguous under the frozen policy: v2 reduced neither judge-decision nor final false negatives, added five regressions for one rescue, increased mean total latency by 514.747 ms, and did not improve safety beyond v1's already perfect result. Production retrieval remains `DENSE_CROSS_ENCODER_RERANK`; production judge remains `EVIDENCE_SUFFICIENCY_V1`.

## GPT-5.6 Luna vs GPT-5.6 Sol Evidence Judge Benchmark

The evidence-sufficiency prompt and retrieval pipeline were frozen before model comparison. This phase did not retune prompts, change retrieval, add a third model, or create a Sol-specific schema. The only intended quality variable was judge model identity: OpenAI `gpt-5.6-luna` versus OpenAI `gpt-5.6-sol`, both on historically frozen `evidence-sufficiency-v1`.

### Dataset methodology

The new sealed dataset is `acmeai-sol-judge-e2e-eval-v1`: 60 genuinely new corpus-grounded cases, generated by `manual-corpus-grounded-v1`, with 10 two-document, 20 three-document, 6 near-duplicate, 6 exact-identifier, 6 version/region, 4 semantic/paraphrase, 4 ACL-sensitive, and 4 partial/no-answer cases. Three-document questions were deliberately emphasized because composite evidence remained the difficult judge category. Cases are not reused from prior calibration, holdout, reranker, or v1-v2 judge datasets and were not templated from the previous nine Luna false negatives.

- Dataset SHA-256: `10c947941644ab9e29ac6e4dc8d59ae87872859cee032eb5b8fb89aaadc2cf55`
- Maximum prior-dataset token-set overlap: 0.4444444444444444
- Frozen at: `2026-08-17 13:13:02.603177+00:00`
- Corpus identity: `f78baa9c2d891d2be09383eba5830c3aa3bc0b510b702e83a38f1a36d7b21ff5`
- Semantic index identity: `e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466`

Hidden evaluation-only ground truth includes expected answerability, required document/chunk/version IDs where deterministic, expected ACL behavior, and category. Neither model received expected answers, ground-truth chunks, answerability labels, or failure categories. After the first judge inference the dataset was immutable.

### Shared retrieval identity

Both models consumed one persisted retrieval trace per question: one `text-embedding-3-small` 64-dimensional query embedding, one pgvector cosine Dense Top-20 at threshold 0.28, one local `cross-encoder/ms-marco-MiniLM-L6-v2` execution at revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, and one frozen reranked Top-5. Tenant, ACL, and active-version filtering preceded reranking and judging. No BM25, hybrid, query rewrite, HyDE, expansion, depth change, or threshold change was introduced.

Preparation used 60 new query embeddings (1,210 tokens), 0 cache hits / 60 misses, 0 document embeddings, and 582 authorized local Cross-Encoder pairs (9.7/query). Unauthorized chunks reaching the reranker were 0. Cumulative embedding usage moved 384 → 444 against a local ceiling of 444.

### Frozen prompt, schema, and request-parameter parity

Both models used prompt `evidence-sufficiency-v1` and schema identity `6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621`. Official OpenAI documentation lists the stable model ID `gpt-5.6-sol` with Chat Completions structured-output support; no dated snapshot is published, and the repository does not pin judge snapshots. Returned/resolved model metadata was `gpt-5.6-luna` and `gpt-5.6-sol`.

Historical Luna request settings were copied exactly for Sol:

| Setting | Luna | Sol |
|---|---|---|
| Provider | OpenAI | OpenAI |
| Model | `gpt-5.6-luna` | `gpt-5.6-sol` |
| Prompt | `evidence-sufficiency-v1` | `evidence-sufficiency-v1` |
| Schema identity | `6ce9db94…de5f2621` | `6ce9db94…de5f2621` |
| Temperature | 0 | 0 |
| Reasoning effort | `none` | `none` |
| Max completion tokens | 160 | 160 |
| Timeout | 45 s | 45 s |
| Retry policy | single request, no extra retries | single request, no extra retries |
| Sol pro mode | n/a | false |
| Response format | `json_schema` `evidence_sufficiency` strict | identical |

Request-parameter parity matched before inference. Sol did not receive a different reasoning effort, output budget, retry policy, or Sol-only instructions.

### Precommitted selection policy

Frozen before inference: recommend Sol only if, on the shared `RETRIEVAL_COMPLETE` subset, Sol fixes at least 4 Luna false negatives **or** retrieval-complete answerability recall improves by at least +0.10 absolute, **and** unsupported answers do not increase, overall answerability F1 does not regress, correct-answer count does not regress, and ACL safety, tenant isolation, version correctness, and prompt-injection boundary remain 1.000000 with unauthorized evidence selection 0.

### Overall results

| Final end-to-end metric | Luna | Sol | Delta Sol−Luna |
|---|---:|---:|---:|
| Correct answers | 28 | 36 | +8 |
| Correct abstentions | 8 | 8 | 0 |
| Unsupported answers | 0 | 0 | 0 |
| Incorrect abstentions | 24 | 16 | -8 |
| Accuracy | 0.600000 | 0.733333 | +0.133333 |
| Precision | 1.000000 | 1.000000 | 0 |
| Recall | 0.538462 | 0.692308 | +0.153846 |
| F1 | 0.700000 | 0.818182 | +0.118182 |

Sol reduced incorrect abstentions without introducing unsupported answers.

### Retrieval-complete subset

All required ground-truth evidence existed in the shared reranked Top-5 for the same 43 cases. This subset is identical for both models and is the primary judge-quality comparison.

Judge decision before the common generator:

| Judge-only metric | Luna | Sol | Delta |
|---|---:|---:|---:|
| True-positive decisions | 30 | 38 | +8 |
| False-negative decisions | 13 | 5 | -8 |
| False-positive decisions | 0 | 0 | 0 |
| Accuracy | 0.697674 | 0.883721 | +0.186047 |
| Precision | 1.000000 | 1.000000 | 0 |
| Recall | 0.697674 | 0.883721 | +0.186047 |
| F1 | 0.821918 | 0.938272 | +0.116354 |

Final behavior on the same 43 cases: Luna 28 correct answers / 15 incorrect abstentions; Sol 36 / 7. Final retrieval-complete recall moved 0.651163 → 0.837209 (+0.186047), and F1 moved 0.788732 → 0.911392. The common generator subsequently failed closed on the same two answerable decisions in both arms (`sol_two_01`, `sol_sem_02`); those are `GENERATION_FAILURE`, not judge false negatives.

### False-negative rescues and false-positive regressions

`SOL_FALSE_NEGATIVE_RESCUE` required retrieval-complete evidence, a Luna incorrect abstention, and a Sol correct answer.

| Metric | Count | Case IDs |
|---|---:|---|
| Luna retrieval-complete false negatives (final) | 15 | see failure analysis |
| Sol retrieval-complete false negatives (final) | 7 | `sol_two_01`, `sol_three_13`, `sol_three_18`, `sol_ver_05`, `sol_sem_02`, `sol_sem_03`, `sol_sem_04` |
| Luna FN → Sol correct answer | 10 | `sol_two_02`, `sol_two_04`, `sol_two_06`, `sol_three_08`, `sol_three_11`, `sol_three_14`, `sol_three_17`, `sol_near_01`, `sol_id_01`, `sol_id_04` |
| Luna correct answer → Sol false abstention | 2 | `sol_sem_03`, `sol_sem_04` |
| Net false-negative improvement | 8 | 10 − 2 |

There were no Sol false-positive regressions: 0 Luna correct abstention → Sol unsupported answer, and 0 answers despite incomplete retrieval.

The two Luna-correct → Sol-FN cases failed closed with `JUDGE_REQUEST_ERROR` rather than a completed Sol sufficiency decision. A third Sol request error (`sol_acl_01`) remained a correct abstention because the case should abstain.

### Paired transitions

| Final behavior transition, Luna → Sol | Count |
|---|---:|
| Correct answer → correct answer | 26 |
| Incorrect abstention → correct answer | 10 |
| Incorrect abstention → incorrect abstention | 14 |
| Correct answer → incorrect abstention | 2 |
| Correct abstention → correct abstention | 8 |
| Correct abstention → unsupported answer | 0 |
| Unsupported answer → correct abstention | 0 |
| Unsupported answer → unsupported answer | 0 |

### Three-document results

Three-document retrieval completeness was 0.600000 (12/20). Incomplete traces: `sol_three_01`, `03`, `04`, `05`, `09`, `12`, `15`, `20`. Both models safely abstained on those eight retrieval misses.

| Three-document metric | Luna | Sol |
|---|---:|---:|
| Correct-answer rate | 0.300000 | 0.500000 |
| Incorrect-abstention rate | 0.700000 | 0.500000 |
| Unsupported-answer rate | 0.000000 | 0.000000 |
| Retrieval-complete judge recall | 0.500000 | 0.833333 |
| Retrieval-complete judge F1 | 0.666667 | 0.909091 |

On retrieval-complete three-document cases, Luna true-positives/false-negatives were 6/6; Sol 10/2. Sol rescued `sol_three_08`, `11`, `14`, and `17`. Remaining Sol three-document capability false negatives after complete retrieval are `sol_three_13` and `sol_three_18`. All 20 paired traces persist in `/experiments/sol-judge-e2e-comparison`.

### Two-document results

Two-document retrieval completeness was 0.900000 (9/10); `sol_two_08` missed required evidence. Luna versus Sol correct-answer rate was 0.500000 versus 0.800000, incorrect-abstention rate 0.500000 versus 0.200000, unsupported-answer rate 0 versus 0, and category F1 0.666667 versus 0.888889. On the 9 retrieval-complete two-document cases, judge-decision F1 moved 0.800000 → 1.000000 (Luna FN 3, Sol FN 0). Sol improvement appeared in both two- and three-document questions, not only the most complex composites.

### Exact identifiers, versions, near-duplicates, and abstention safety

Exact-identifier retrieval was complete for 6/6. Luna answered 4/6 and Sol answered 6/6; Luna false abstentions `sol_id_01` and `sol_id_04` were Sol rescues. Because retrieval was identical, those differences are judge-only.

Version/region retrieval and final active-version correctness were 1.000000 for both. No inactive version reached the reranker or either judge. Both models answered 5/6 and incorrectly abstained on the same retrieval-complete case `sol_ver_05`.

The preferred near-duplicate source was present in all six shared traces. Luna answered 5/6; Sol answered 6/6. In `sol_near_01` Luna selected no supporting IDs and abstained, while Sol selected authoritative chunk `d60d3666-be70-4f64-b83e-7fe72c86f426` and answered. The other five cases retained valid authoritative support under both models.

All eight should-abstain cases—four ACL-sensitive and four partial/no-answer—correctly abstained under both judges. Unsupported answers, judge false positives, and selected unauthorized evidence were zero for each arm.

### Supporting evidence and citation quality

Where required-chunk ground truth is defensible, required-evidence precision/recall were 0.536364/0.545455 for Luna and 0.681818/0.690909 for Sol. Empty support despite retrieval-complete answerable evidence: Luna 13, Sol 5. Invented or invalid supporting IDs: Luna 1 (`sol_three_14`, fail-closed), Sol 0. No provider-specific ID repair was applied.

Citation validity and citation correctness were 1.000000 for both arms. Conservative exact answer/source support was 0.642857 over 28 Luna answered cases and 0.555556 over 36 Sol answered cases. The denominators differ, so the lower Sol rate is not evidence of worse semantic grounding. The heuristic recorded 10 and 16 unsupported exact-claim matches, respectively, but unsupported final answers under evaluation ground truth were zero; no semantic-entailment claim is made.

### Failure analysis

Root cause and final behavior remain separate. Luna: 9 `RETRIEVAL_INCOMPLETE`, 12 `LUNA_EVIDENCE_GATE_FALSE_NEGATIVE`, 1 `INVALID_SUPPORTING_ID`, 2 `GENERATION_FAILURE`. Sol: 9 `RETRIEVAL_INCOMPLETE`, 5 `SOL_EVIDENCE_GATE_FALSE_NEGATIVE` (3 completed-capability misses: `sol_three_13`, `sol_three_18`, `sol_ver_05`; plus 2 request-error fail-closed answerable cases), and the same 2 `GENERATION_FAILURE` cases. Unauthorized, ACL, and version failures were 0.

### Latency, tokens, cache, and verified cost

Official short-context list prices consulted from OpenAI documentation were Luna $0.20 / $0.02 cached / $1.20 output per 1M tokens and Sol $5.00 / $0.50 cached / $30.00 output per 1M tokens.

| Mean unless noted | Luna | Sol | Incremental Sol |
|---|---:|---:|---:|
| Shared query embedding | 332.376 ms | 332.376 ms | 0 |
| Shared cache lookup | 0.715 ms | 0.715 ms | 0 |
| Shared Dense retrieval | 6.612 ms | 6.612 ms | 0 |
| Shared Cross-Encoder | 104.858 ms | 104.858 ms | 0 |
| Judge mean | 1817.188 ms | 3485.189 ms | +1668.001 ms |
| Judge p50 | 1683.146 ms | 2288.399 ms | +605.253 ms |
| Judge p95 | 2568.587 ms | 17041.454 ms | +14472.867 ms |
| Total mean | 2281.441 ms | 3309.928 ms | +1028.487 ms |
| Total p50 | 2084.704 ms | 2723.976 ms | +639.272 ms |
| Total p95 | 3049.603 ms | 4644.308 ms | +1594.705 ms |
| Input tokens | 38,582 | 37,919 | −663 |
| Output tokens | 4,232 | 4,595 | +363 |
| Official reasoning tokens | 0 | 0 | 0 |
| Official cached input tokens | 0 | 0 | 0 |
| Official measured cost | $0.012795 | $0.327445 | +$0.314650 |
| Official cost per case | $0.000213 | $0.005457 | +$0.005244 |

Cost per additional correct answer, using official list prices on measured tokens, is $0.039331. Judge preflight projected $0.013203 Luna + $0.330075 Sol from historical sufficiency-v1 mean sizes; measured totals were close. Sol p95 judge latency is dominated by three `JUDGE_REQUEST_ERROR` fail-closed events.

Embedding preflight began at 384 cumulative calls with 60 unique queries, 0 cache matches, 60 misses, and a locally configured ceiling of 444. Judge preflight began at 424 cumulative calls with a ceiling of 544, 60 unique Luna inputs, 60 unique Sol inputs, 0 same-model cache matches, and 0 cross-model shared cache keys. Each arm made 60 hosted calls (0 hits / 60 misses). Cumulative hosted judge use ended at 544. Document embedding calls and external reranker calls were 0.

### Security

ACL safety, tenant isolation, active-version correctness, and prompt-injection boundary were 1.000000 for both. Unauthorized chunks reaching the reranker, Luna, or Sol were zero; unauthorized supporting evidence selections were zero.

### Limitations and production decision

This is a 60-case synthetic-corpus, one-shot descriptive benchmark, not a statistical-significance claim. Retrieval completeness uses hidden document/chunk ground truth; judges never received it. The extractive generator and conservative lexical support heuristic limit claims about semantic answer quality. Sol request errors fail closed and therefore can look like false abstentions; they are not additional reasoning-capability measurements.

Under the frozen policy Sol is the selected production judge: 10 retrieval-complete false-negative rescues (≥ 4), retrieval-complete recall +0.186047 (≥ 0.10), unsupported answers unchanged at 0, overall F1 0.700000 → 0.818182, correct answers 28 → 36, and all hard security/version constraints held. Production retrieval remains `DENSE_CROSS_ENCODER_RERANK`. Production judge is `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`. The quality/cost/latency trade-off is real: Sol is the higher-capability frozen sufficiency judge, but official list cost is about 25× Luna per case and mean/p95 judge latency are substantially higher.

## Dense+Cross-Encoder vs Hybrid+Cross-Encoder Benchmark

This phase compared frozen production retrieval `DENSE_CROSS_ENCODER_RERANK` against one fixed Hybrid candidate that reuses the same Cross-Encoder. Sol was not used for retrieval configuration selection. Query rewrite, HyDE, decomposition, reranker tuning, and judge changes were not implemented.

### Dataset methodology

New sealed dataset `acmeai-hybrid-reranker-eval-v1`: 60 new corpus-grounded cases, generation method `manual-corpus-grounded-v1`, emphasizing three-document coverage. Hidden ground truth never reached Dense, BM25, RRF, Cross-Encoder, Sol, or the generator.

| Category | Count |
|---|---:|
| Two-document answerable | 8 |
| Three-document answerable | 26 |
| Near-duplicate answerable | 6 |
| Exact-identifier answerable | 6 |
| Version/region-sensitive answerable | 6 |
| Semantic/paraphrase answerable | 4 |
| ACL-sensitive should-abstain | 2 |
| Partial/no-answer should-abstain | 2 |

- Dataset SHA-256: `d5b69e67a4b322d03e8f9530e0526142c1176bf4112febc3cc5c61635ba9a451`
- Maximum prior-dataset token-set overlap: 0.400000
- Split seed: `2026081717`
- Split identity: `1d5be5d43588856d6d930798992f497554b9480675df6a9d7914b384be786737`
- Calibration / holdout: 40 / 20
- Calibration IDs: `hrr_two_01,02,04,05,06,08`; `hrr_three_01,04,05,08,09,10,11,13,16,17,19,21,22,23,24,25,26`; `hrr_near_01,02,04,05`; `hrr_id_01,02,05,06`; `hrr_ver_01,02,04,06`; `hrr_sem_01,02,04`; `hrr_acl_01`; `hrr_no_02`
- Holdout IDs: `hrr_two_03,07`; `hrr_three_02,03,06,07,12,14,15,18,20`; `hrr_near_03,06`; `hrr_id_03,04`; `hrr_ver_03,05`; `hrr_sem_03`; `hrr_acl_02`; `hrr_no_01`
- Corpus identity: `f78baa9c2d891d2be09383eba5830c3aa3bc0b510b702e83a38f1a36d7b21ff5`
- Semantic index identity: `e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466`

Historical holdouts remained one-shot locked. Production identities going in were `DENSE_CROSS_ENCODER_RERANK` and `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`. Cross-Encoder revision remained `233902d25c440f23af6f7d6e94d2946bac0bee0a`. Historical BM25 Okapi (`k1=1.2`, `b=0.75`, NFKC/casefold/identifier tokenization, ACL before scoring) and RRF `k=60` were reused, not rewritten.

### Frozen control A and fixed candidate B

Control A `DENSE_CROSS_ENCODER_RERANK`: Dense authorized/current-version search, cosine ≥ 0.28, Top-20, same local Cross-Encoder, final Top-5.

Candidate B `HYBRID_CROSS_ENCODER_RERANK`: Dense Top-20 @ 0.28 plus BM25 Top-20, `chunk_id` union, RRF `k=60` pre-order, Cross-Encoder input bounded at 30, same Cross-Encoder, final Top-5. The 0.28 threshold is never applied to BM25 or RRF scores. Tenant/ACL/active-version filtering precedes both branches.

No third retrieval candidate. No grid search of `k1`, `b`, RRF k, depths, union depth, or reranker identity.

### Precommitted calibration selection policy

Frozen before calibration: B wins retrieval selection only if All Required Evidence Coverage@5 improves by at least +0.10 **or** Three-document Coverage@5 improves by at least +0.20, **and** Required Evidence Recall@5 does not regress by more than 0.02, exact-ID and semantic/paraphrase success do not materially regress, version correctness = 1.000000, and ACL safety = 1.000000.

### External-call preflight

Embedding preflight before any new query embeddings: cumulative 444, ceiling 504, 60 unique new questions, 0 cache matches, 60 missing, expected ending 504. A and B shared one query embedding per question.

Judge preflight after retrieval lock and holdout traces: cumulative 544, ceiling 584, calibration Sol calls 0, 20 unique A inputs, 20 unique B inputs, 5 identical A/B ordered Top-5 keys, 0 prior cache matches, worst-case ending 584.

### Calibration candidate-pool coverage

Sol was not called. Generation was not run.

| Before reranking | A Dense@20 | B union≤30 |
|---|---:|---:|
| Required evidence recall | 0.973684 | 1.000000 |
| All required evidence coverage | 0.921053 | 1.000000 |
| Two-document coverage | 1.000000 | 1.000000 |
| Three-document coverage | 0.823529 | 1.000000 |

Hybrid candidate generation recovered every required document that Dense missed in the 40 calibration cases.

### Calibration Top-5 results

| Metric | A Dense+CE | B Hybrid+CE | B−A |
|---|---:|---:|---:|
| Hit@5 | 1.000000 | 1.000000 | 0 |
| Recall@5 | 0.956140 | 0.982456 | +0.026316 |
| MRR | 0.953947 | 0.953947 | 0 |
| nDCG@5 | 0.926655 | 0.949172 | +0.022516 |
| Required evidence recall@5 | 0.956140 | 0.982456 | +0.026316 |
| All required evidence coverage@5 | 0.868421 | 0.947368 | +0.078947 |
| Two-document coverage@5 | 1.000000 | 1.000000 | 0 |
| Three-document coverage@5 | 0.705882 | 0.882353 | +0.176471 |
| Exact-ID recall@5 | 1.000000 | 1.000000 | 0 |
| Version-sensitive recall@5 | 1.000000 | 1.000000 | 0 |
| Version correctness | 1.000000 | 1.000000 | 0 |
| Semantic/paraphrase success | 1.000000 | 1.000000 | 0 |
| Near-duplicate success | 0.750000 | 0.750000 | 0 |
| ACL safety | 1.000000 | 1.000000 | 0 |

### Lexical rescues and Cross-Encoder conversion

Calibration BM25-only required evidence: 3. Retained by RRF: 3. Sent to Cross-Encoder: 3. Promoted into final Top-5: 2. Lost: 1 (`hrr_three_22` / `finance-expense-policy`, BM25 rank 5, RRF 10, CE 10). Conversion rate 0.666667. Dense-only required evidence retained 0 / lost 0; Hybrid did not discard Dense-only semantic evidence.

### Three-document calibration

17 calibration three-document cases. A candidate coverage 0.823529 vs B 1.000000. A Top-5 coverage 0.705882 vs B 0.882353. Fixed by Hybrid+CE: `hrr_three_11`, `hrr_three_16`, `hrr_three_26`. Worsened: none. Still incomplete after candidate expansion: `hrr_three_04` (required evidence already in Dense Top-20; Cross-Encoder left `engineering-deployment-handbook` at rank 6) and `hrr_three_22` (BM25 recovered `finance-expense-policy`; Cross-Encoder did not promote it into Top-5).

### Calibration selection

Selected `DENSE_CROSS_ENCODER_RERANK`. Coverage gain +0.078947 was below +0.10, and three-document gain +0.176471 was below +0.20. Safety constraints held (recall improved; exact-ID/semantic/version/ACL did not regress). The lock timestamp is `2026-08-17 15:03:01.430877+00:00`, before holdout retrieval started at `2026-08-17 15:06:22.676645+00:00`.

### Holdout retrieval

Configuration was not changed after lock.

| Metric | A | B |
|---|---:|---:|
| All required evidence coverage@5 | 0.722222 | 1.000000 |
| Required evidence recall@5 | 0.907407 | 1.000000 |
| Three-document coverage@5 | 0.444444 | 1.000000 |
| Two-document coverage@5 | 1.000000 | 1.000000 |
| Hit@5 / MRR | 1.000000 / 1.000000 | 1.000000 / 1.000000 |
| nDCG@5 | 0.918017 | 0.984801 |
| Exact-ID recall@5 | 1.000000 | 1.000000 |
| Version correctness | 1.000000 | 1.000000 |
| Semantic/paraphrase success | 1.000000 | 1.000000 |
| Near-duplicate success | 1.000000 | 1.000000 |
| ACL safety | 1.000000 | 1.000000 |
| Candidate-pool coverage | 0.722222 | 1.000000 |
| Three-document candidate coverage | 0.444444 | 1.000000 |

Holdout BM25-only required evidence: 5, all retained by RRF and sent to the Cross-Encoder; 4 promoted into Top-5. Dense-only required lost: 0. Unauthorized chunks to Cross-Encoder: 0.

### Holdout Sol end-to-end

Frozen judge `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1` on A Top-5 and B Top-5. Prompt, schema, request settings, supporting-context builder, and extractive generator were unchanged. No judge tuning.

A first Sol pass populated 35 unique cache entries (20 A + 15 B after 5 identical ordered Top-5 keys). An overly broad near-duplicate `forbidden_document_ids` guard then aborted before persisting runs; ACL traces had already shown zero unauthorized documents. The guard was narrowed to `EXCLUDE_FORBIDDEN` only, matching retrieval security, and the one-shot execute was completed from cache without additional unique Sol calls.

| Final end-to-end metric | A Dense+CE+Sol | B Hybrid+CE+Sol |
|---|---:|---:|
| Correct answers | 9 | 14 |
| Correct abstentions | 2 | 2 |
| Unsupported answers | 0 | 0 |
| Incorrect abstentions | 9 | 4 |
| Accuracy | 0.550000 | 0.800000 |
| Precision | 1.000000 | 1.000000 |
| Recall | 0.500000 | 0.777778 |
| F1 | 0.666667 | 0.875000 |

### Retrieval-to-answer conversion

Hybrid retrieval rescues (A Top-5 incomplete and B Top-5 complete): 5 (`hrr_three_02,03,12,14,18`). Rescues → Sol correct answer: 4. Rescues → remaining incorrect abstention: 1 (`hrr_three_02`). Conversion rate 0.800000. A-complete → B-incomplete regressions: 0.

Paired transitions: correct answer → correct answer 9; incorrect abstention → correct answer 5; incorrect abstention → incorrect abstention 4; correct abstention → correct abstention 2.

### Three-document holdout

Nine holdout three-document cases.

| Three-document metric | A | B |
|---|---:|---:|
| Retrieval-complete rate | 0.444444 | 1.000000 |
| Correct-answer rate | 0.222222 | 0.777778 |
| Incorrect-abstention rate | 0.777778 | 0.222222 |
| Unsupported-answer rate | 0.000000 | 0.000000 |
| Judge FN rate after complete retrieval | 0.500000 | 0.222222 |

Correct answers 2 vs 7. Remaining B three-document incorrect abstentions after complete retrieval: `hrr_three_02` and `hrr_three_06`.

### Exact identifiers, version/region, semantic/paraphrase, abstention safety

Exact-ID holdout retrieval and answers were 2/2 for both. Version/region retrieval correctness was 1.000000 for both; each arm answered `hrr_ver_05` and incorrectly abstained on retrieval-complete `hrr_ver_03`. Semantic/paraphrase: 1/1 both, no regression. Near-duplicate: 2/2 both. Should-abstain cases (1 ACL, 1 partial/no-answer): both pipelines correctly abstained; unsupported answers 0; judge false positives 0. Hybrid candidate expansion did not create unsafe unsupported answers.

### Security

ACL safety 1.000000. Tenant isolation 1.000000. Version correctness 1.000000 on retrieval traces. Prompt-injection boundary not applicable on this dataset (no prompt-injection cases; evaluator value is null, treated as vacuously passing). Unauthorized chunks to Cross-Encoder 0. Unauthorized chunks to Sol 0 on `EXCLUDE_FORBIDDEN` ACL traces. Unauthorized evidence selections 0.

### Latency and local compute

Calibration query embedding mean 895.515 ms (includes 40 cache misses). Dense 11.958 ms. Additional BM25 6.436 ms. RRF 0.230 ms. Cross-Encoder pairs/query 9.35 A vs 17.00 B. Reranker mean/p50/p95: A 238.402 / 109.885 / 439.419 ms; B 364.140 / 147.584 / 1297.915 ms.

Holdout retrieval: embedding mean 469.435 ms; Dense 7.414 ms; BM25 5.760 ms; RRF 0.142 ms. Pairs/query 7.35 A vs 16.90 B. Reranker mean/p50/p95: A 163.225 / 110.112 / 252.128 ms; B 200.728 / 150.853 / 414.128 ms.

BM25 index size 5,328 bytes; build about 1.03 ms; no Elasticsearch; no persistent BM25 store beyond the in-process authorized candidate set.

Recorded holdout end-to-end totals used Sol cache hits on replay (judge mean 0 ms) and therefore understate live Sol latency. Unique Sol work is the 35 first-pass cache writes, not the replay.

### Cache and external usage

Query embeddings: 60 new calls (40 calibration + 20 holdout), 0 hits during those partitions, 1,278 tokens (856 + 422), cumulative 444 → 504. Document embedding calls 0. External reranker calls 0. Calibration Sol calls 0.

Sol holdout: 35 unique hosted calls, 21,928 input tokens, 2,702 output tokens, cumulative hosted judge 544 → 579 against ceiling 584. Replay recorded 20/20 A and 20/20 B gate-cache hits. Identical ordered Top-5 evidence reused the same Sol gate key; different ordered Top-5 evidence did not.

### Production decision

Holdout retrieval and Sol answers moved in Hybrid's favor (coverage 0.722222 → 1.000000, three-document 0.444444 → 1.000000, correct answers 9 → 14, unsupported 0 → 0). Promotion still requires the frozen calibration selection of B, which did not occur. Production retriever remains `DENSE_CROSS_ENCODER_RERANK`. Production judge remains `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`.

This is a 60-case synthetic-corpus descriptive benchmark, not a statistical-significance claim. The calibration materiality bar was missed by 0.021053 coverage and 0.023529 three-document coverage. The holdout magnitude was larger, so Hybrid+the same Cross-Encoder is a confirmed candidate-generation effect that this locked policy did not promote.

## Final Hybrid+Cross-Encoder Replication

This was the final Dense-vs-Hybrid replication for Enterprise RAG Workbench v1.

### Motivation

The previous calibration selected `DENSE_CROSS_ENCODER_RERANK` because All Required Evidence Coverage@5 gained +0.078947 (below +0.10) and Three-document Coverage@5 gained +0.176471 (below +0.20). The sealed holdout of that same experiment then showed a much larger Hybrid effect (coverage 0.722222 → 1.000000; three-document 0.444444 → 1.000000; correct answers 9 → 14; unsupported 0 → 0). That holdout was not allowed to retroactively change the original selection rule. This phase therefore replicated the frozen A/B pipelines on one new unseen 80-case dataset, with a new pre-registered replication rule frozen before retrieval.

Query rewrite, HyDE, decomposition, BM25/RRF/depth/Top-K/embedding/reranker tuning, Evidence Coverage v2, and judge changes were not implemented.

### Previous contradictory calibration/holdout evidence

The previous experiment `acmeai-hybrid-reranker-eval-v1` remains unchanged: calibration selected Dense; holdout was one-shot consumed; production identities going in were `DENSE_CROSS_ENCODER_RERANK` and `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`.

### New frozen replication policy

Frozen at initialize `2026-08-17 15:59:40.640279+00:00`, before retrieval lock `2026-08-17 16:21:58.989764+00:00`. This is not a retroactive change to the previous experiment.

Promote B only if the full 80-case dataset shows All Required Evidence Coverage@5 B−A ≥ +0.08 **or** Three-document Coverage@5 B−A ≥ +0.15, **and** Required Evidence Recall@5 does not regress by more than 0.02, exact-ID and semantic/paraphrase success do not materially regress (≤ 0.05), version correctness = 1.000000, ACL safety = 1.000000, and unauthorized results = 0.

No calibration/holdout split. No parameter or prompt tuning.

### Dataset

New sealed dataset `acmeai-hybrid-reranker-replication-v1`: 80 new corpus-grounded cases, generation method `manual-corpus-grounded-v1`, three-document questions dominant. Hidden ground truth never reached Dense, BM25, RRF, Cross-Encoder, Sol, or the generator.

| Category | Count |
|---|---:|
| Two-document answerable | 10 |
| Three-document answerable | 40 |
| Near-duplicate answerable | 8 |
| Exact-identifier answerable | 6 |
| Version/region-sensitive answerable | 6 |
| Semantic/paraphrase answerable | 4 |
| ACL-sensitive should-abstain | 3 |
| Partial/no-answer should-abstain | 3 |

- Dataset SHA-256: `f9144a0bbc6f15de030963199b8a784d9e23031a017258a8b6636b131cb04114`
- Maximum prior-dataset token-set overlap: 0.416667 (accepted ceiling 0.5; did not raise frozen prior maxima)
- No-split identity: `76812ca5976347687382bdf9979c4b2e1ddff7fe928791cc6e2321c4ab6cc35b`
- Freeze timestamp: `2026-08-17 15:59:40.640279+00:00`
- Previous holdout case IDs were not reused
- Corpus identity: `f78baa9c2d891d2be09383eba5830c3aa3bc0b510b702e83a38f1a36d7b21ff5`
- Semantic index identity: `e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466`
- Cross-Encoder revision: `233902d25c440f23af6f7d6e94d2946bac0bee0a`

### Frozen control A and candidate B

Control A `DENSE_CROSS_ENCODER_RERANK`: text-embedding-3-small, Dense authorized/current-version search, cosine ≥ 0.28, Top-20, `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, final Top-5.

Candidate B `HYBRID_CROSS_ENCODER_RERANK`: Dense Top-20 @ 0.28 plus BM25 Okapi v1 Top-20 (`k1=1.2`, `b=0.75`), canonical `chunk_id` union, RRF `k=60`, union limit 30, same Cross-Encoder, final Top-5. The 0.28 threshold is never applied to BM25 or RRF scores. Tenant/ACL/active-version filtering precedes both branches.

### External-call preflight

Embedding preflight before retrieval: cumulative 504, ceiling set to 584, 80 unique new questions, 0 cache matches, 80 missing, expected ending 584.

Judge preflight after retrieval lock: cumulative hosted-judge ledger 544, 80 unique A inputs, 80 unique B inputs, 34 identical ordered Top-5 keys, 0 prior cache matches, 126 new unique calls, ceiling set to 670.

### Candidate-pool coverage

Sol was not called during retrieval. Generation was not run for selection.

| Before reranking | A Dense@20 | B union≤30 |
|---|---:|---:|
| Required evidence recall | 0.930180 | 1.000000 |
| All required evidence coverage | 0.824324 | 1.000000 |
| Two-document coverage | 0.900000 | 1.000000 |
| Three-document coverage | 0.700000 | 1.000000 |

Hybrid candidate generation recovered every required document that Dense missed on this dataset.

### Top-5 retrieval results

| Metric | A Dense+CE | B Hybrid+CE | B−A |
|---|---:|---:|---:|
| Hit@5 | 1.000000 | 1.000000 | 0 |
| Recall@5 | 0.921171 | 0.977477 | +0.056306 |
| MRR | 0.993243 | 0.993243 | 0 |
| nDCG@5 | 0.920441 | 0.961910 | +0.041469 |
| Required evidence recall@5 | 0.921171 | 0.977477 | +0.056306 |
| All required evidence coverage@5 | 0.797297 | 0.932432 | +0.135135 |
| Two-document coverage@5 | 0.900000 | 1.000000 | +0.100000 |
| Three-document coverage@5 | 0.650000 | 0.875000 | +0.225000 |
| Exact-ID recall@5 | 1.000000 | 1.000000 | 0 |
| Version-sensitive recall@5 | 1.000000 | 1.000000 | 0 |
| Version correctness | 1.000000 | 1.000000 | 0 |
| Semantic/paraphrase success | 1.000000 | 1.000000 | 0 |
| Near-duplicate success | 0.875000 | 0.875000 | 0 |
| ACL safety | 1.000000 | 1.000000 | 0 |

### Lexical rescues and Cross-Encoder conversion

BM25-only required evidence: 15. Retained by RRF: 15. Sent to Cross-Encoder: 15. Promoted into final Top-5: 12. Lost: 3 (`hrp_three_02` / `engineering-deployment-handbook` CE rank 6; `hrp_three_15` / `finance-expense-policy` CE rank 6; `hrp_three_28` / `remote-work-policy` CE rank 8). Conversion rate 0.800000. Dense-only required evidence retained 0 / lost 0.

### Three-document performance

40 three-document cases. A candidate coverage 0.700000 vs B 1.000000. A Top-5 coverage 0.650000 vs B 0.875000. Fixed by Hybrid+CE: `hrp_three_11,12,15,17,22,28,30,33,37,40`. Worsened: `hrp_three_04`. Still incomplete after candidate expansion: `hrp_three_02,07,27,35`. Remaining B failures: `hrp_three_02` and `hrp_three_27` `CROSS_ENCODER_FAILED_TO_PROMOTE`; `hrp_three_04`, `hrp_three_07`, `hrp_three_35` `CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE`.

### Regressions

One retrieval regression: `hrp_three_04` A-complete → B-incomplete (`CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE`). No exact-ID, semantic, version, or near-duplicate retrieval regressions.

### Final architecture decision

Selected `HYBRID_CROSS_ENCODER_RERANK`. Coverage gain +0.135135 met +0.08, and three-document gain +0.225000 met +0.15. Required recall improved; exact-ID/semantic/version/ACL/unauthorized guardrails held. Architecture record `enterprise-rag-v1-retriever` frozen at `2026-08-17 16:21:59.015033+00:00`. This is the final Enterprise RAG Workbench v1 retriever. No further v1 Dense-vs-Hybrid replication, BM25/RRF/depth/Top-K/embedding/reranker tuning, Query Rewrite, HyDE, or multi-query work.

### End-to-end confirmation

Frozen judge `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1` on A Top-5 and B Top-5. Prompt, schema, request settings, supporting-context builder, and extractive generator were unchanged. Sol was not used to modify retrieval selection.

| Final end-to-end metric | A Dense+CE+Sol | B Hybrid+CE+Sol |
|---|---:|---:|
| Correct answers | 34 | 41 |
| Correct abstentions | 6 | 6 |
| Unsupported answers | 0 | 0 |
| Incorrect abstentions | 40 | 33 |
| Accuracy | 0.500000 | 0.587500 |
| Precision | 1.000000 | 1.000000 |
| Recall | 0.459459 | 0.554054 |
| F1 | 0.629630 | 0.713043 |

Hybrid retrieval rescues (A Top-5 incomplete and B Top-5 complete): 11. Rescues → Sol correct answer: 5 (`hrp_two_04`, `hrp_three_11,12,17,37`). Rescues → remaining incorrect abstention: 6. Conversion rate 0.454545. End-to-end regressions (A correct answer → B incorrect abstention, A correct abstention → B unsupported, A safe → B unsafe): 0. Paired transitions: correct answer → correct answer 34; incorrect abstention → correct answer 7; incorrect abstention → incorrect abstention 33; correct abstention → correct abstention 6.

Three-document end-to-end retrieval-complete rate 0.650000 → 0.875000; correct-answer rate 0.175000 → 0.275000. Judge false-negative rate after complete retrieval remains the dominant residual error (0.423729 A, 0.405797 B overall; 0.730769 → 0.685714 on three-document cases).

### Latency, local compute, cache, usage, security

Retrieval was live: 80 query-embedding cache misses, mean embedding 632.665 ms, Dense 9.806 ms, BM25 10.167 ms, RRF 0.231 ms. Cross-Encoder pairs/query 9.3125 A vs 16.6000 B. Reranker mean/p50/p95: A 249.992 / 127.080 / 536.238 ms; B 296.997 / 153.671 / 850.444 ms. BM25 index size 5,328 bytes; build 0.571 ms.

Sol A: 80 live calls, mean 2483.260 ms, p50 2337.474 ms, 0 cache hits. Sol B: 46 live calls, mean live 2376.767 ms, p50 live 2219.676 ms, plus 34 identical-Top-5 cache hits at 0 ms. Cached replay latency is not reported as live latency. B mean total pipeline 2335.197 ms is pulled down by those 34 cache hits.

Query embeddings: 80 new calls, 1,848 tokens, cumulative 504 → 584. Document embedding calls 0. External reranker calls 0. Unique Sol calls 126 (80 A + 46 B after 34 identical keys), 103,061 input tokens, 10,776 output tokens, cumulative hosted-judge ledger 544 → 670. Official list cost from the existing Sol pricing configuration: A $0.412955, B $0.425630.

ACL safety 1.000000. Tenant isolation 1.000000. Retrieval version correctness 1.000000. Prompt-injection boundary 1.000000 on the one training-example identifier case (`hrp_id_06`). Unauthorized chunks to Cross-Encoder 0. Unauthorized chunks to Sol 0. Unauthorized evidence selections 0. Should-abstain cases (3 ACL + 3 partial/no-answer): both pipelines correctly abstained; unsupported answers 0.

### Limitations

This is an 80-case synthetic-corpus descriptive replication, not a statistical-significance claim. One three-document ranking regression remains (`hrp_three_04`). Hybrid candidate generation is complete at the union, but the Cross-Encoder still fails to place every required document in Top-5 on five three-document cases. After retrieval freeze, the largest remaining end-to-end problem is Sol false-negative abstention on retrieval-complete evidence.

## Enterprise RAG Workbench v1 — Final Frozen End-to-End Benchmark

This is the final quality completion benchmark for Enterprise RAG Workbench v1. No Dense-vs-Hybrid selection, BM25/RRF/Cross-Encoder/Top-K/threshold retuning, judge change, or Query Rewrite/HyDE/multi-query work was performed. Metrics are computed from the already-frozen production pipeline on one new unseen 80-case dataset.

### Frozen production architecture

Retriever `HYBRID_CROSS_ENCODER_RERANK`. Judge `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`. Release architecture `enterprise-rag-workbench-v1` frozen at `2026-08-18 12:30:44.403651+00:00`, architecture hash `e76aa8834d995c03d47d6d80c9917cc9e16e4f31b24dcd742250329239a23c85`. The prior replication record `enterprise-rag-v1-retriever` was not replaced.

- Corpus identity: `f78baa9c2d891d2be09383eba5830c3aa3bc0b510b702e83a38f1a36d7b21ff5`
- Semantic index identity: `e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466`
- Embedding: openai-compatible `text-embedding-3-small` version `1`, dimension 64
- Dense: pgvector cosine, threshold 0.28, Top-20
- BM25 Okapi v1: `k1=1.2`, `b=0.75`, NFKC/casefold/identifier tokenization, Top-20
- RRF `k=60`, candidate union limit 30
- Cross-Encoder `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, final Top-5
- Judge model `gpt-5.6-sol`, prompt `evidence-sufficiency-v1`, schema identity `6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621`, supporting-context only
- Generator `deterministic-extractive-v1`
- Tenant/ACL/active-version filtering precedes Dense and BM25

### Dataset

Sealed dataset `acmeai-enterprise-rag-v1-final-eval`: 80 new corpus-grounded cases, generation method `manual-corpus-grounded-v1`. Hidden ground truth never reached Dense, BM25, RRF, Cross-Encoder, Sol, or the generator.

| Category | Count |
|---|---:|
| Single-document answerable | 10 |
| Two-document answerable | 16 |
| Three-document answerable | 20 |
| Exact-identifier answerable | 6 |
| Version/region-sensitive answerable | 6 |
| Near-duplicate answerable | 6 |
| Semantic/paraphrase answerable | 4 |
| ACL-sensitive should-abstain | 4 |
| Partial/no-answer should-abstain | 4 |
| Prompt-injection | 4 |

- Dataset SHA-256: `8cc399fe36bafa64ac7a02b81fe3d7660cc5adebe60242ed9f3b1ea0dcd5e83c`
- Maximum prior-dataset token-set overlap: 0.380952 (accepted ceiling 0.5; closest `fv1_acl_03` vs `acmeai_evidence_judge_e2e_eval_v1.json` / `judge_acl_04`)
- Freeze timestamp: `2026-08-18 11:02:29.792328+00:00`
- Experiment identity: `acmeai-enterprise-rag-v1-final-eval` (one record; not replaced after interruption)

### External-call preflight

Embedding preflight before retrieval: cumulative 584, ceiling set to 664, 80 unique new questions, 0 cache matches, 80 missing, expected ending 664.

Judge preflight after retrieval lock: cumulative hosted-judge ledger 670, 80 unique production inputs, 0 prior cache matches, 80 new unique calls, ceiling set to 750.

### Candidate-pool coverage

Sol was not called during retrieval.

| Before reranking | Hybrid union≤30 |
|---|---:|
| Required evidence recall | 1.000000 |
| All required evidence coverage | 1.000000 |
| Two-document coverage | 1.000000 |
| Three-document coverage | 1.000000 |

Hybrid candidate generation recovered every required document on this dataset. BM25-only required evidence: 32. Retained by RRF: 32. Sent to Cross-Encoder: 32. Promoted into final Top-5: 16. Lost: 16. Conversion rate 0.500000.

### Top-5 retrieval results

| Metric | Hybrid+CE |
|---|---:|
| Hit@5 | 0.985507 |
| Recall@5 | 0.900966 |
| MRR | 0.909420 |
| nDCG@5 | 0.857459 |
| Required evidence recall@5 | 0.900966 |
| All required evidence coverage@5 | 0.753623 |
| Single-document coverage@5 | 0.900000 |
| Two-document coverage@5 | 0.937500 |
| Three-document coverage@5 | 0.250000 |
| Exact-ID recall@5 | 1.000000 |
| Version-sensitive recall@5 | 1.000000 |
| Version correctness | 1.000000 |
| Semantic/paraphrase success | 1.000000 |
| Near-duplicate success | 0.500000 |
| ACL safety | 1.000000 |

Three-document candidate coverage 1.000000 vs Top-5 coverage 0.250000. Remaining ranking losses: 13 `CROSS_ENCODER_FAILED_TO_PROMOTE`, 4 `CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE`. Candidate-generation misses: 0.

### End-to-end results

Frozen judge `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1` on the production Top-5. Prompt, schema, request settings, supporting-context builder, and extractive generator were unchanged.

| Final end-to-end metric | Hybrid+CE+Sol |
|---|---:|
| Correct answers | 24 |
| Correct abstentions | 11 |
| Unsupported answers | 0 |
| Incorrect abstentions | 45 |
| Accuracy | 0.437500 |
| Precision | 1.000000 |
| Recall | 0.347826 |
| F1 | 0.516129 |
| TP | 24 |
| FN | 45 |
| FP | 0 |
| TN | 11 |

### Retrieval-complete subset

52 retrieval-complete answerable cases by persisted Top-5 traces. Sol judge-only: TP 25, FN 27, FP 0, accuracy 0.480769, precision 1.000000, recall 0.480769, F1 0.649351. Final behavior on that subset: 24 correct answers, 28 incorrect abstentions (one Sol true-positive, `fv1_ver_03`, became `GENERATION_FAILURE`). Retrieval misses are not counted as Sol failures.

### Category results

| Category | Retrieval complete | Correct answers | Incorrect abstentions | Unsupported |
|---|---:|---:|---:|---:|
| Single-document (10) | 0.900000 | 2 | 8 | 0 |
| Two-document (16) | 0.937500 | 11 | 5 | 0 |
| Three-document (20) | 0.250000 | 2 | 18 | 0 |
| Exact identifiers (6) | 1.000000 | 0 | 6 | 0 |
| Version/region (6) | 1.000000 | 5 | 1 | 0 |
| Near-duplicates (6) | 1.000000 | 2 | 4 | 0 |
| Semantic/paraphrase (4) | 1.000000 | 1 | 3 | 0 |
| ACL-sensitive (4) | n/a should-abstain | 0 | 0 | 0 |
| Partial/no-answer (4) | n/a should-abstain | 0 | 0 | 0 |
| Prompt-injection (4) | 1.000000 | 1 | 0 | 0 |

Exact-ID retrieval recall@5 is 1.000000; all six failures are Sol false-negative abstentions after complete retrieval. Near-duplicate preferred-source success is 0.500000. Version/region retrieval is complete; the one end-to-end miss is `fv1_ver_03` generation failure after a Sol true-positive.

### Abstention safety

Should-abstain cases: 11 (4 ACL + 4 partial/no-answer + 3 prompt-injection). Correct abstentions: 11. Unsupported answers: 0.

### Prompt-injection safety

Four frozen prompt-injection cases. Boundary success 1.000000. Document instructions followed: no. Unauthorized evidence selected: no. Final result safe: yes for `fv1_inj_01`, `fv1_inj_02`, `fv1_inj_03`, `fv1_inj_04`.

### Security

ACL safety 1.000000. Tenant isolation 1.000000. Retrieval version correctness 1.000000. Unauthorized chunks to Cross-Encoder 0. Unauthorized chunks to Sol 0. Unauthorized supporting evidence 0. Unauthorized citations 0. Security success rate 1.000000.

### Citation quality

Citation validity 1.000000 and citation correctness 1.000000 on 24/24 answered cases. Abstentions are not scored.

### Grounding

Unsupported answers 0. The conservative lexical claim-support heuristic on answered cases is 0.416667 (14 unsupported-claim flags) and is not a semantic answer-correctness score. Groundedness beyond citation validity is not claimed.

### Failure analysis

Root-cause counts on failed answerable cases:

| Root cause | Count |
|---|---:|
| EVIDENCE_GATE_FALSE_NEGATIVE | 27 |
| CROSS_ENCODER_FAILED_TO_PROMOTE | 13 |
| CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE | 4 |
| GENERATION_FAILURE | 1 |

One persisted operational Sol error (`fv1_dup_05`, `JUDGE_REQUEST_ERROR`) fail-closed to abstention and is already counted in the false-negative total. It was not rerun.

### Latency

All 80 final cases were live original-run samples. Resume live samples: 0. Cached replay samples: 0.

| Stage | n | mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|
| Query embedding | 80 | 337.526 | 296.195 | 406.870 |
| Dense | 80 | 5.338 | 4.794 | 8.705 |
| BM25 | 80 | 6.455 | 5.154 | 17.718 |
| RRF | 80 | 0.169 | 0.165 | 0.261 |
| Cross-Encoder | 80 | 198.668 | 143.575 | 500.085 |
| Sol judge (live) | 80 | 2478.622 | 2302.200 | 3819.997 |
| Generation | 80 | 0.096 | 0.000 | 0.371 |
| Total live pipeline | 80 | 3597.468 | 2879.668 | 4353.138 |

### Local compute

Cross-Encoder `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, CPU, frozen production backend. 1,311 pairs (16.3875 / query). BM25 index 5,328 bytes; build 0.553 ms.

### Cache and usage

Original live run: 80 query-embedding misses, 80 Sol misses. Resume added 0 embedding calls and 0 Sol calls. Query-cache ledger 584 → 664. Hosted-judge ledger 670 → 750. Document embedding calls 0. External reranker calls 0. Sol 51,989 input tokens, 4,715 output tokens. Official list cost from the existing Sol pricing configuration: $0.401395.

### Limitations

This is an 80-case synthetic-corpus descriptive completion benchmark, not a statistical-significance claim. Hybrid candidate generation is complete, but Cross-Encoder Top-5 coverage on three-document questions is 0.250000. After retrieval, the dominant end-to-end error is Sol false-negative abstention, including all six exact-identifier cases. These are known v1 limitations and were not tuned.

## Enterprise RAG Workbench v2 — Quality Recovery Baseline and Failure Census

This section is a research baseline. It does not replace Enterprise RAG Workbench v1. Frozen v1 architecture, dataset, benchmark run, corpus, semantic index, BM25, RRF, Cross-Encoder revision, Sol judge, prompt/schema, generator, and security policies were inspected and left unchanged. No v1 experiment row was rewritten. No paid inference was performed.

### V2 research identity

```text
architecture_id     = enterprise-rag-workbench-v2-research
parent architecture = enterprise-rag-workbench-v1
research status = ACTIVE
production status = FALSE
```

Initial v2 control configuration is pipeline-equivalent to frozen v1:

```text
HYBRID_CROSS_ENCODER_RERANK
+
GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1
+
deterministic-extractive-v1
```

Dense Top-20 @ 0.28, BM25 Okapi v1 Top-20 (`k1=1.2`, `b=0.75`), RRF `k=60`, union limit 30, Cross-Encoder `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, final Top-5, Sol prompt `evidence-sufficiency-v1`, schema identity `6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621`. Corpus identity `f78baa9c2d891d2be09383eba5830c3aa3bc0b510b702e83a38f1a36d7b21ff5`. Semantic index identity `e598d9fb91b13a8c6bd6be94b1bc0a54bbce27dd4e98dc158669d7198bc6f466`.

The frozen v1 final dataset `acmeai-enterprise-rag-v1-final-eval` is diagnosis-only. It is not promotion evidence for any v2 architecture. New v2 candidates must be evaluated on genuinely unseen datasets.

### Failure census

Classified from persisted v1 final traces only. Failed answerable cases = 45.

| Root cause | Count |
|---|---:|
| EVIDENCE_GATE_FALSE_NEGATIVE | 26 |
| CROSS_ENCODER_FAILED_TO_PROMOTE | 13 |
| CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE | 4 |
| GENERATION_FAILURE | 1 |
| JUDGE_REQUEST_ERROR | 1 |
| CANDIDATE_GENERATION_MISS | 0 |
| RANKING_OUTSIDE_TOP5 | 0 |
| INVALID_SUPPORTING_ID | 0 |
| SUPPORTING_CONTEXT_LOSS | 0 |
| UNKNOWN | 0 |

`fv1_dup_05` is persisted as `JUDGE_REQUEST_ERROR` rather than remaining inside the v1 Sol false-negative total. Case-level rows (required evidence, candidate-pool ranks, Cross-Encoder ranks, Top-5, judge decision, supporting IDs, final behavior, root cause) are stored on `enterprise-rag-workbench-v2-research`.

### Ranking failure diagnostic

Candidate-pool All Required Coverage remains 1.000000. Top-5 All Required Coverage remains 0.753623. Three-document candidate coverage remains 1.000000. Three-document Top-5 coverage remains 0.250000. Ranking misses are therefore 17 pool-complete / Top-5-incomplete cases (13 `CROSS_ENCODER_FAILED_TO_PROMOTE`, 4 `CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE`). No ranking fix was implemented.

Primary ranking appearance: `SAME_DOCUMENT_CROWDING` (`CROSS_ENCODER_SAME_DOCUMENT_TOP5_CROWDING`). Pointwise Cross-Encoder scores fill Top-5 with extra chunks of already-retrieved documents and displace a required third document.

### Judge false-negative diagnostic

52 retrieval-complete cases. 27 Sol false negatives after complete Top-5, classified without modifying the judge:

| Class | Count |
|---|---:|
| EXACT_IDENTIFIER_FALSE_NEGATIVE | 6 |
| SUFFICIENCY_REASONING_FAILURE | 7 |
| MULTI_DOCUMENT_FALSE_NEGATIVE | 7 |
| NEAR_DUPLICATE_FALSE_NEGATIVE | 3 |
| SEMANTIC_FALSE_NEGATIVE | 3 |
| REQUEST_ERROR | 1 |
| SUPPORT_SELECTION_FAILURE | 0 |
| OTHER | 0 |

Exact-ID retrieval is 6/6; exact-ID final correct answers remain 0/6. Required evidence was present. Support was empty. Correct evidence was ignored.

### Operational failure diagnostic

Historical calls were not retried.

| Case | Class |
|---|---|
| `fv1_ver_03` | GENERATOR_FAILURE |
| `fv1_dup_05` | PROVIDER_REQUEST_FAILURE |

`fv1_ver_03` is a Sol true-positive that the extractive generator failed to answer. `fv1_dup_05` is a persisted Sol request error that fail-closed to abstention. Neither is a model-quality retry candidate.

### Security guardrails

v2 hard guardrails, unchanged from frozen v1:

```text
ACL safety = 1.000000
tenant isolation = 1.000000
version correctness = 1.000000
prompt injection boundary = 1.000000
unsupported answers = 0 target
unauthorized chunks to reranker = 0
unauthorized chunks to judge = 0
```

### Primary v2 bottleneck

`CROSS_ENCODER_SAME_DOCUMENT_TOP5_CROWDING`

Candidate generation is already complete. Three-document Top-5 coverage is 0.250000 because the pointwise Cross-Encoder repeatedly spends Top-5 slots on extra chunks of the same document.

### Recommended phase 1 intervention

`DOCUMENT_DIVERSIFIED_TOP5`: after the frozen Cross-Encoder scores a candidate, keep only the highest-scoring chunk per `document_id` and then cut Top-5. Do not change the Cross-Encoder model, revision, candidate union, BM25, or RRF. Do not implement it.

## V2 Phase 1 — Document-Diversified Top-5 Ranking

This section records the retrieval/ranking-only test of one hypothesis: whether same-document crowding in the frozen pointwise Cross-Encoder Top-5 materially reduces multi-document evidence coverage. Enterprise RAG Workbench v1 remained frozen and was not modified. Production routing was not changed. The Judge was not modified. No generation or Sol calls were issued.

### P0 motivation

Persisted P0 diagnosis: `CROSS_ENCODER_SAME_DOCUMENT_TOP5_CROWDING`. Candidate-pool all-required coverage was 1.000000 while Top-5 all-required coverage was 0.753623 and three-document Top-5 coverage was 0.250000. Those v1/P0 numbers are diagnostic history only and are not Phase-1 promotion evidence.

### Hypothesis

Does `DOCUMENT_DIVERSIFIED_TOP5` improve three-document Coverage@5 without unacceptable same-document multi-chunk, Exact-ID, semantic, version, or ACL regressions?

### New dataset

`acmeai-v2-document-diversity-eval-v1`: 80 genuinely unseen corpus-grounded cases (36 three-document, 14 two-document, 8 single-document, 6 near-duplicate, 4 exact-ID, 4 version/region, 4 semantic/paraphrase, 2 ACL-sensitive should-abstain, 2 partial/no-answer should-abstain). At least 8 answerable cases require multiple distinct chunks from the same canonical `document_id`. SHA-256 `79229d49eac1091f648632531019064f73e0e877038c0d1d88618c783dd502c1`.

### Dataset independence

Generation method `manual-corpus-grounded-v1`. Maximum prior-dataset token-set Jaccard overlap `0.34782608695652173` against every historical evaluation dataset, below the 0.5 guard. Closest previous case: `vdv_one_05` vs `acmeai_hybrid_reranker_eval_v1.json` / `hrr_two_03`. Frozen overlap identities for prior datasets were unchanged.

### Control A

`POINTWISE_CROSS_ENCODER_TOP5`: frozen query embedding, Dense Top-20 @ 0.28, BM25 Okapi v1 Top-20, RRF k=60, union <=30, frozen Cross-Encoder `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, highest Cross-Encoder score Top-5.

### Candidate B

`DOCUMENT_DIVERSIFIED_TOP5`: the exact same query embedding, Dense/BM25/RRF pool, and Cross-Encoder scores. Final Top-5 keeps the highest-scoring chunk per canonical `document_id` and skips later chunks from already-selected documents until five distinct document IDs or candidate exhaustion. No rescoring, MMR, similarity penalty, LLM, or dynamic document cap.

### Single independent variable

Final Top-5 set construction.

### Precommitted selection rule

Frozen at `2026-08-18 14:55:21.527050+00:00`, before first retrieval. Candidate B wins only if Three-document Coverage@5 improves >= +0.15 absolute or All Required Evidence Coverage@5 improves >= +0.08 absolute, and all guardrails pass: Required Evidence Recall@5 must not regress > 0.02; same-document-multi-chunk subset coverage must not regress > 0.10; Exact-ID Recall@5 and semantic/paraphrase success must not materially regress (> 0.05); version correctness = 1.000000; ACL safety = 1.000000; A-complete → B-incomplete regression count must not exceed B crowding-rescue count. The rule was not modified after results.

### Primary retrieval metrics

Shared candidate-pool all-required coverage = 1.000000 for both modes. A and B used one query embedding, the same Dense Top-20, BM25 Top-20, RRF union, and Cross-Encoder scores.

| Metric | POINTWISE_CROSS_ENCODER_TOP5 | DOCUMENT_DIVERSIFIED_TOP5 | Δ |
|---|---:|---:|---:|
| Required Evidence Recall@5 | 0.973684 | 0.986842 | +0.013158 |
| All Required Evidence Coverage@5 | 0.934211 | 0.960526 | +0.026316 |
| Three-document Coverage@5 | 0.861111 | 0.916667 | +0.055556 |
| Two-document Coverage@5 | 1.000000 | 1.000000 | 0.000000 |
| Single-document Coverage@5 | 1.000000 | 1.000000 | 0.000000 |

Three-document Coverage@5 is the primary bottleneck metric. The +0.055556 gain is below the frozen +0.15 primary condition. All-required Coverage@5 +0.026316 is below the frozen +0.08 alternative.

### Document diversity metrics

| Statistic | A | B |
|---|---:|---:|
| Mean unique `document_id`s in Top-5 | 4.175000 | 5.000000 |
| Median unique `document_id`s in Top-5 | 4.000000 | 5.000000 |
| Mean repeated-document chunks per Top-5 | 0.825000 | 0.000000 |
| Top-5 slots occupied by already-represented documents | 0.165000 | 0.000000 |

By category, A mean unique documents: single-document 3.875000, two-document 3.928571, three-document 4.305556. B is 5.000000 unique documents in every category, with zero repeated-document chunks.

### Crowding rescues

`DOCUMENT_CROWDING_RESCUE` total = 2. Three-document rescues = 2. Two-document rescues = 0. Other rescues = 0.

- `vdv_three_01`: A missed `data-retention-standard` (CE rank 6). A Top-5 was `operations-continuity-plan` twice and `security-incident-policy` twice. B removed 2 duplicate slots and became retrieval-complete.
- `vdv_three_12`: A missed `remote-work-policy` (CE rank 6). A Top-5 was `project-atlas-api` twice and `project-atlas-launch` twice. B removed 2 duplicate slots and became retrieval-complete.

### Diversification regressions

`DOCUMENT_DIVERSIFICATION_REGRESSION` count = 7, all `MULTIPLE_REQUIRED_CHUNKS_SAME_DOCUMENT`. Authoritative-source replacement = 0. Exact-ID displacement = 0. Other = 0.

A-complete → B-incomplete cases: `vdv_two_01`, `vdv_two_02`, `vdv_two_03`, `vdv_one_01`, `vdv_one_02`, `vdv_one_03`, `vdv_one_04`. In each case A kept two required chunks from one canonical document; B kept only the highest-scoring chunk and dropped the second required marker. Regression count 7 exceeds crowding-rescue count 2.

### Same-document multi-chunk guardrail

Eight answerable cases require multiple distinct chunks from one canonical `document_id`.

| Metric | A | B |
|---|---:|---:|
| Case count | 8 | 8 |
| Required Evidence Recall@5 | 1.000000 | 0.500000 |
| All Required Coverage@5 | 1.000000 | 0.000000 |
| A-complete → B-incomplete |  | 7 |

Coverage regression = 1.000000, above the frozen 0.10 guardrail. `vdv_three_01` is the eighth same-document case: A was already document-incomplete, so it is a crowding rescue rather than an A-complete → B-incomplete regression; B still lost the second required same-document marker.

### Three-document analysis

36 three-document cases. Candidate-pool complete rate = 1.000000.

| Metric | A | B |
|---|---:|---:|
| Top-5 complete rate | 0.861111 | 0.916667 |
| Required evidence recall | 0.944444 | 0.972222 |

B fixed: `vdv_three_01`, `vdv_three_12`. B worsened: none. Both failed: `vdv_three_11`, `vdv_three_20`, `vdv_three_31`. In the both-failed cases the missing required documents sat at Cross-Encoder ranks 7–8, so dropping one duplicate Top-5 slot was not enough to promote every required source.

### Exact-ID analysis

A Exact-ID Recall@5 = 1.000000. B Exact-ID Recall@5 = 1.000000. A success → B failure count = 0.

### Near-duplicate analysis

A preferred-source success = 1.000000. B preferred-source success = 1.000000. A→B gains = 0. A→B regressions = 0. Unique-document count was not treated as preferred-source success.

### Semantic analysis

A semantic success = 1.000000. B semantic success = 1.000000. A→B regression count = 0.

### Version / region

A retrieval version correctness = 1.000000. B retrieval version correctness = 1.000000. Active-version correctness = 1.000000.

### Security

ACL filtering and active-version filtering ran before Dense, BM25, RRF, Cross-Encoder, and document diversification. ACL safety = 1.000000. Tenant isolation = 1.000000. Version correctness = 1.000000. Unauthorized chunks to Cross-Encoder = 0. Unauthorized chunks entering Candidate B = 0.

### Latency

Shared Cross-Encoder mean 61.228 ms, p50 57.003 ms, p95 95.828 ms. Incremental B overhead is document-id dedup / selection only: mean 0.048 ms. Shared CE cost is not attributed twice. Query embedding mean 383.216 ms; Dense 5.081 ms; BM25 5.229 ms; RRF 0.151 ms.

### Local compute

Cross-Encoder `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, CPU. 1,335 candidate pairs (16.6875 / query). Document-diversification CPU overhead mean 0.048 ms. BM25 index 5,328 bytes; build 1.039 ms. Runtime packages were not changed.

### External usage

Embedding preflight: cumulative ledger 664, new-dataset queries 80, exact cache hits 0, missing unique embeddings 80, authorized ceiling 744. Live usage: 80 new query embedding calls, 1,923 embedding tokens, 0 document embedding calls, 0 Sol calls, 0 judge-gate calls, 0 generator calls, 0 external reranker calls. Ending query-embedding ledger = 744.

### Selected V2 ranking

Frozen-policy application: primary conditions false (three-document gain 0.055556 < 0.15 and all-required coverage gain 0.026316 < 0.08). Guardrails false (same-document-multi-chunk coverage regression 1.000000; regressions 7 > rescues 2). Selected ranking on `enterprise-rag-workbench-v2-research`:

```text
POINTWISE_CROSS_ENCODER_TOP5
```

Not promoted to v1 production. Enterprise RAG Workbench v1 remained frozen and was not modified.

### Remaining measured bottleneck after the selected ranking

`CROSS_ENCODER_SAME_DOCUMENT_TOP5_CROWDING` remains the largest Phase-1-measured bottleneck on the selected pointwise Top-5: five three-document cases are still pool-complete and Top-5-incomplete because duplicate same-document slots occupy the final five. Hard one-document-one-chunk diversification rescued only two of those cases and failed the same-document multi-chunk guardrail, so it is not the selected V2 ranking.

## V2 Phase 2 — Soft Document-Cap Top-5 Ranking

This section records the final ranking-structure experiment of the current v2 cycle: unrestricted pointwise Cross-Encoder Top-5 versus a precommitted maximum of two chunks per canonical `document_id`. Frozen v1 was not modified. The Judge was not modified. No generation or Sol calls were issued. Cap=3, dynamic caps, and MMR were not tested.

### P1 evidence

P1 selected `POINTWISE_CROSS_ENCODER_TOP5`. Hard one-chunk-per-document raised three-document Coverage@5 by only +0.055556 and destroyed same-document multi-chunk coverage (1.000000 → 0.000000) with 7 A-complete → B-incomplete regressions versus 2 crowding rescues. Same-document crowding is real, but cap=1 is too aggressive. Cap=2 is therefore the single precommitted less-aggressive candidate.

### Hypothesis

Allowing up to two chunks per canonical document may remove pathological same-document crowding while preserving legitimate two-chunk evidence from the same document.

### New dataset

`acmeai-v2-soft-document-cap-eval-v1`: 80 genuinely unseen corpus-grounded cases. SHA-256 `184bc848581d6cf02c062435ea5ea3a53dfcb831ea03229783afc897f4e21ac2`. Maximum prior overlap `0.450000`. Closest previous case: `sdc_three_02` vs `acmeai_v2_document_diversity_eval_v1.json` / `vdv_three_02`. Generation method `manual-corpus-grounded-v1`.

### Candidate definition

Control A is frozen `POINTWISE_CROSS_ENCODER_TOP5`. Candidate B is `MAX_2_CHUNKS_PER_DOCUMENT_TOP5`: walk the shared Cross-Encoder ranking and keep a chunk unless its `document_id` already occupies two selected slots. No rescoring, MMR, learned selector, query-type detection, or dynamic cap.

### Selection policy

Frozen at `2026-08-18 15:28:43.124103+00:00`, before first retrieval. Independent of the P1 policy, which remains historical and unchanged. The freeze timestamp was not overwritten during resume. Candidate B wins only if Three-document Coverage@5 improves >= +0.10 or All Required Evidence Coverage@5 improves >= +0.05, and all guardrails hold: Required Evidence Recall@5 regression <= 0.02; exactly-two-required-chunks-same-document coverage regression <= 0.05; Exact-ID, semantic/paraphrase, and near-duplicate preferred-source no material regression (> 0.05); version correctness = 1.000000; ACL safety = 1.000000; A-complete → B-incomplete regressions <= B crowding rescues.

### Resume audit

Interrupted after the one-shot shared retrieval committed and before this section was written. Detected checkpoint `I`. Existing complete cases 80. Existing partial cases 0. Cached query embeddings 80. Existing shared retrieval traces 80. Existing Cross-Encoder traces 80. Existing A Top-5 traces 80. Existing B Top-5 traces 80. New embedding calls during resume 0. New Cross-Encoder executions during resume 0. Benchmark case rerun count 0.

### Primary retrieval metrics

Shared candidate-pool all-required coverage = 1.000000 for both modes. A and B used one query embedding, the same Dense Top-20, BM25 Top-20, RRF union, and Cross-Encoder scores.

| Metric | POINTWISE_CROSS_ENCODER_TOP5 | MAX_2_CHUNKS_PER_DOCUMENT_TOP5 | Δ |
|---|---:|---:|---:|
| Required Evidence Recall@5 | 0.960526 | 0.960526 | 0.000000 |
| All Required Evidence Coverage@5 | 0.881579 | 0.881579 | 0.000000 |
| Three-document Coverage@5 | 0.750000 | 0.750000 | 0.000000 |
| Two-document Coverage@5 | 1.000000 | 1.000000 | 0.000000 |
| Single-document Coverage@5 | 1.000000 | 1.000000 | 0.000000 |

Three-document Coverage@5 gain is 0.000000, below the frozen +0.10 primary condition. All-required Coverage@5 gain is 0.000000, below the frozen +0.05 alternative.

### Secondary retrieval metrics

| Metric | A | B | Δ |
|---|---:|---:|---:|
| Hit@5 | 1.000000 | 1.000000 | 0.000000 |
| Recall@5 | 0.960526 | 0.960526 | 0.000000 |
| MRR | 0.980263 | 0.980263 | 0.000000 |
| nDCG@5 | 0.933334 | 0.933334 | 0.000000 |
| Exact-ID Recall@5 | 1.000000 | 1.000000 | 0.000000 |
| Version-sensitive Recall@5 | 1.000000 | 1.000000 | 0.000000 |
| Near-duplicate preferred-source success | 0.666667 | 0.666667 | 0.000000 |
| Semantic/paraphrase success | 1.000000 | 1.000000 | 0.000000 |
| ACL safety | 1.000000 | 1.000000 | 0.000000 |

### Document occupancy

| Statistic | A | B |
|---|---:|---:|
| Mean unique documents in Top-5 | 4.287500 | 4.287500 |
| Mean maximum chunks from one document | 1.587500 | 1.587500 |
| Top-5 sets containing 3+ chunks from one document | 0 | 0 |
| Top-5 sets containing 4+ chunks from one document | 0 | 0 |
| Top-5 sets containing 5 chunks from one document | 0 | 0 |

Slots freed by the max-2 cap: total 0, mean 0.000000. Control A never placed three or more chunks from one canonical `document_id` in Top-5 on this frozen corpus, so Candidate B selected the identical Top-5 in every case.

### Soft-cap rescues

`SOFT_DOCUMENT_CAP_RESCUE` total = 0. Three-document rescues = 0. Two-document rescues = 0. Other rescues = 0. No case satisfied A retrieval-incomplete, missing required evidence in the candidate pool, A containing >=3 chunks from an already represented document, and B becoming retrieval-complete.

### Soft-cap regressions

`SOFT_DOCUMENT_CAP_REGRESSION` count = 0. THREE_REQUIRED_CHUNKS_SAME_DOCUMENT = 0. TWO_REQUIRED_CHUNKS_SAME_DOCUMENT = 0. AUTHORITATIVE_SOURCE_DISPLACEMENT = 0. EXACT_ID_DISPLACEMENT = 0. OTHER = 0. A-complete → B-incomplete cases: none.

### Exactly-two-same-document subset

Eight answerable cases require exactly two distinct chunks from one canonical `document_id`.

| Metric | A | B |
|---|---:|---:|
| Required Evidence Recall@5 | 0.812500 | 0.812500 |
| All Required Coverage@5 | 0.625000 | 0.625000 |
| A-complete → B-incomplete | 0 |  |

Cap=2 preserved every legitimate two-chunk same-document Top-5 that Control A already had. The 0.625000 coverage on this subset is a Control A limitation, not a cap-2 regression.

### Three-document results

36 three-document cases. Candidate-pool complete rate = 1.000000. A Top-5 complete rate = 0.750000. B Top-5 complete rate = 0.750000. A required evidence recall = 0.916667. B required evidence recall = 0.916667. B fixed cases = 0. B worsened cases = 0. Both-failed cases = 9: `sdc_three_10`, `sdc_three_11`, `sdc_three_12`, `sdc_three_16`, `sdc_three_24`, `sdc_three_25`, `sdc_three_31`, `sdc_three_32`, `sdc_three_33`.

### Missing-evidence CE rank distribution

Nine pool-complete / A-incomplete cases. Missing required-evidence Cross-Encoder ranks:

| Rank | Count |
|---|---:|
| rank 6 | 2 |
| rank 7 | 3 |
| rank 8 | 1 |
| rank 9-10 | 1 |
| rank >10 | 2 |

Zero of those nine misses were structurally promotable by max-2: Control A never occupied three or more slots from one document, so the cap had no duplicate slot to free. These ranks were not used to change cap=2.

### Exact-ID analysis

A Exact-ID Recall@5 = 1.000000. B Exact-ID Recall@5 = 1.000000. A success → B failure count = 0.

### Near-duplicate analysis

A preferred-source success = 0.666667. B preferred-source success = 0.666667. A→B gains = 0. A→B regressions = 0. Unique-document count was not treated as preferred-source success.

### Semantic analysis

A semantic success = 1.000000. B semantic success = 1.000000. A→B regression count = 0.

### Version / region

A retrieval version correctness = 1.000000. B retrieval version correctness = 1.000000. Active-version correctness = 1.000000.

### Security

ACL filtering and active-version filtering ran before Dense, BM25, RRF, Cross-Encoder, and the max-2 selector. ACL safety = 1.000000. Tenant isolation = 1.000000. Version correctness = 1.000000. Unauthorized chunks to Cross-Encoder = 0. Unauthorized chunks entering Candidate B = 0.

### Latency

All reported samples are original live measurements from the interrupted session's completed one-shot run. Resume live samples = none. Cache-derived work during resume = none. Shared Cross-Encoder mean 58.391 ms, p50 54.236 ms, p95 81.114 ms. Incremental B overhead is document-count bookkeeping / selection only: mean 0.049 ms. Shared CE cost is not attributed twice. Query embedding mean 437.898 ms; Dense 6.174 ms; BM25 6.481 ms; RRF 0.185 ms.

### Local compute

Cross-Encoder `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, CPU. 1,218 candidate pairs (15.225 / query). Soft-cap CPU overhead mean 0.049 ms. BM25 index 5,328 bytes; build 1.412 ms. Runtime packages were not changed.

### External usage

Embedding preflight: cumulative ledger 744, new-dataset queries 80, exact cache hits 0, missing unique embeddings 80, authorized ceiling 824. Original live run: 80 query-embedding misses, 2,135 embedding tokens. Resume added 0 embedding calls and 0 Cross-Encoder executions. Document embedding calls 0. Sol calls before interruption 0. Sol calls during resume 0. Judge-gate calls 0. Generator calls 0. External reranker calls 0. Ending query-embedding ledger = 824.

### Selected V2 ranking

Frozen-policy application: primary conditions false (three-document gain 0.000000 < 0.10 and all-required coverage gain 0.000000 < 0.05). Guardrails true (required-recall regression 0.000000; exactly-two-same-document coverage regression 0.000000; exact/semantic/near regressions 0.000000; version 1.000000; ACL 1.000000; regressions 0 <= rescues 0). Selected ranking on `enterprise-rag-workbench-v2-research`:

```text
POINTWISE_CROSS_ENCODER_TOP5
```

Not promoted to v1 production. Enterprise RAG Workbench v1 remained frozen and was not modified.

### Ranking research termination

```text
RANKING RESEARCH STATUS = FROZEN_FOR_CURRENT_V2_CYCLE
```

Cap=3, dynamic caps, MMR, and a new reranker were not tested after this experiment.

### Remaining measured bottleneck after the selected ranking

`CROSS_ENCODER_SAME_DOCUMENT_TOP5_CROWDING` remains the largest Phase-2-measured ranking bottleneck on the selected pointwise Top-5: nine three-document cases are still pool-complete and Top-5-incomplete. Max-2 did not change any Top-5 because Control A never used a third same-document slot. Ranking research is frozen for the current v2 cycle. The next research lever, if no critical defect, is `V2_EVIDENCE_SUFFICIENCY_FALSE_NEGATIVE_REDUCTION`.

## V2 Phase 3 — Evidence Sufficiency False-Negative Reduction

This section records the judge-only test of one hypothesis: whether a minimally revised evidence-sufficiency prompt reduces retrieval-complete false-negative abstentions without relaxing grounding, schema, model, or ranking. Frozen Enterprise RAG Workbench v1 was not modified. Production routing was not changed. Retrieval and ranking were not modified. The generator was not modified. Evidence Coverage v2 was not restored.

### Historical motivation

P0 diagnostic traces (not promotion evidence): retrieval-complete cases = 52; Sol false negatives after complete retrieval = 27, including Exact-ID 6, sufficiency-reasoning 7, multi-document 7, near-duplicate 3, semantic 3, and 1 request error. Historical Exact-ID retrieval succeeded 6/6 while final correct answers were 0/6 because the Judge returned insufficient support. Those v1 numbers remain diagnostic only.

### Frozen retrieval/ranking

Selected V2 ranking remains `POINTWISE_CROSS_ENCODER_TOP5`. Ranking research status remains `FROZEN_FOR_CURRENT_V2_CYCLE`. Shared pipeline:

```text
Query
↓
text-embedding-3-small
version 1
dimension 64
↓
Dense pgvector cosine
threshold 0.28
Top-20
+
BM25 Okapi v1
k1 = 1.2
b = 0.75
Top-20
↓
RRF k=60
candidate union <=30
↓
cross-encoder/ms-marco-MiniLM-L6-v2
revision:
233902d25c440f23af6f7d6e94d2946bac0bee0a
↓
POINTWISE_CROSS_ENCODER_TOP5
```

Final Top-K = 5. Document cap 1/2/3, dynamic caps, MMR, new rerankers, and Top-K/candidate-depth tuning were not tested.

### Control Judge

`GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`: model `gpt-5.6-sol`, prompt `evidence-sufficiency-v1`, prompt hash `d49994bc7a429e2cbbd07935a2ed4cbb5503098cd01cdd146e6417be55dc7d83`, schema identity `6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621`, temperature 0, reasoning none, 160 completion tokens, 45s timeout, existing frozen validation.

### Candidate Judge

`GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V2`: the same model, schema, temperature, reasoning, token budget, timeout, and validation. The only intended quality variable is prompt text/version `evidence-sufficiency-v2`, prompt hash `31611f1566e691a365e34d7c56fc23411c8cc1a75b9ccb37341eed2f148e234d`, parent prompt `evidence-sufficiency-v1`, created `2026-08-18T16:29:48.671616+00:00`. The prompt states that sufficient evidence is sufficient, exact identifiers count as direct evidence, multi-document chunks may be combined, missing facts must not be inferred, caution is not insufficiency, supporting IDs must be the smallest sufficient retrieved set, and retrieved document instructions are untrusted DATA. Evidence Coverage v2 requirement-decomposition states were not used. v1 prompt identity was not overwritten.

### New dataset

`acmeai-v2-sufficiency-fn-eval-v1`: 80 genuinely unseen corpus-grounded cases (16 exact-identifier, 20 three-document, 12 two-document, 8 near-duplicate, 8 version/region, 4 semantic/paraphrase, 4 partial/no-answer should-abstain, 2 ACL-sensitive should-abstain, 2 prompt-injection should-abstain, 4 single-document). SHA-256 `4cdae0c554f3e7cdeb3d435bf3c13bda65ebe5b7fa5fbc2721cdcda3e89c8323`. Evaluator-only labels were hidden from both Judges.

### Dataset independence

Generation method `manual-corpus-grounded-v1`. Maximum prior-dataset token-set Jaccard overlap `0.39285714285714285` against every historical evaluation dataset, below the 0.5 guard. Closest previous case: `sfn_three_08` vs `acmeai_v2_soft_document_cap_eval_v1.json` / `sdc_three_27`. Frozen at `2026-08-18 16:29:48.671616+00:00`. No question, answerability, required-evidence, or category edits after freeze.

### Precommitted selection rule

Frozen at `2026-08-18 16:29:48.671616+00:00`, before first retrieval and before first Judge result. Candidate B wins only if retrieval-complete Judge recall improves >= +0.12 absolute or Control FN → Candidate correct-supported rescues >= 8, and all guardrails hold: Judge precision >= 0.98; false-positive unsupported answers do not increase; ACL safety = 1.000000; tenant isolation = 1.000000; version correctness = 1.000000; prompt-injection boundary = 1.000000; unauthorized supporting IDs = 0; invalid supporting IDs = 0; and A-correct-supported → B-false-negative regressions are strictly fewer than rescues. The rule was not modified after results.

### Retrieval-complete denominator

Shared frozen Top-5, one retrieval per case. Answerable cases = 72. Retrieval-complete cases = 58. Retrieval-complete rate = 0.805556. Candidate-pool all-required coverage = 1.000000. Top-5 all-required coverage = 0.833333. Three-document Coverage@5 = 0.450000 (9/20). Two-document Coverage@5 = 0.916667. Exact-ID Recall@5 = 1.000000, with Judge-complete Exact-ID denominator 15/16 because `sfn_id_10` missed a required marker in Top-5. Version correctness = 1.000000. ACL safety = 1.000000. Unauthorized chunks to the Judge = 0. Judges were not invoked until retrieval traces were frozen at `2026-08-18 16:31:08.064362+00:00`.

### Retrieval-complete Judge metrics

Primary metric is retrieval-complete Judge Recall on the 58 complete answerable cases.

| Metric | GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1 | GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V2 | Δ |
|---|---:|---:|---:|
| TP | 43 | 43 | 0 |
| FN | 15 | 15 | 0 |
| FP | 0 | 0 | 0 |
| Precision | 1.000000 | 1.000000 | 0.000000 |
| Recall | 0.741379 | 0.741379 | 0.000000 |
| F1 | 0.851485 | 0.851485 | 0.000000 |

Recall gain is 0.000000, below the frozen +0.12 primary condition.

### All-case answerability metrics

| Metric | A | B |
|---|---:|---:|
| TP | 44 | 44 |
| FN | 28 | 28 |
| FP | 0 | 0 |
| TN | 8 | 8 |
| Accuracy | 0.650000 | 0.650000 |
| Precision | 1.000000 | 1.000000 |
| Recall | 0.611111 | 0.611111 |
| F1 | 0.758621 | 0.758621 |

All-case FN includes retrieval-incomplete answerable cases and is not treated as a pure Judge capability failure.

### Exact-ID results

16 cases. Retrieval-complete 15. A TP/FN = 13/2. B TP/FN = 13/2. A FN → B TP rescues = 0. A TP → B FN regressions = 0. Exact-ID retrieval remained strong; the remaining two complete Exact-ID false negatives were not rescued by v2.

### Three-document results

20 three-document cases. Judge-only retrieval-complete denominator = 9. A recall = 0.777778 (7 TP / 2 FN). B recall = 0.777778 (7 TP / 2 FN). Rescues = 0. Regressions = 0.

### Two-document results

12 two-document cases. Judge-only retrieval-complete denominator = 10. A recall = 0.600000 (6 TP / 4 FN). B recall = 0.700000 (7 TP / 3 FN). Rescues = `sfn_two_04`, `sfn_two_05`. Regressions = `sfn_two_09`.

### Near-duplicate results

Retrieval-complete cases = 8. A correct = 3. B correct = 3. FN rescue = `sfn_dup_02`. FP regressions = none. A-correct → B-FN regression `sfn_dup_07` is counted in the global regression list rather than as a near-duplicate FP.

### Semantic / paraphrase

Retrieval-complete cases = 4. A recall = 1.000000. B recall = 0.750000. Rescues = 0. Regression = `sfn_sem_01`.

### Version / region

8 cases. Retrieval version correctness = 1.000000. A Judge recall = 0.750000 (6 TP / 2 FN). B Judge recall = 0.750000 (6 TP / 2 FN). Supporting version correctness A/B = 1.000000 / 1.000000. Active-version behavior was preserved.

### False-negative rescues

`FALSE_NEGATIVE_RESCUE` count = 3, below the frozen >= 8 alternative.

- `sfn_two_04` `multidoc_two`
- `sfn_two_05` `multidoc_two`
- `sfn_dup_02` `near_duplicate`

### False-positive regressions

`FALSE_POSITIVE_REGRESSION` count = 0. No safety regression was hidden.

### A-correct → B-false-negative regressions

Count = 3, not strictly fewer than the 3 rescues.

- `sfn_two_09` `multidoc_two`
- `sfn_dup_07` `near_duplicate`
- `sfn_sem_01` `semantic_paraphrase`

### Supporting-ID quality

| Statistic | A | B |
|---|---:|---:|
| answerable=true decisions | 44 | 44 |
| non-empty supporting-ID rate | 1.000000 | 1.000000 |
| valid supporting-ID rate | 1.000000 | 1.000000 |
| supporting required-chunk precision | 0.977273 | 0.977273 |
| supporting required-chunk recall | 0.741379 | 0.741379 |
| invalid ID count | 0 | 0 |
| unauthorized ID count | 0 | 0 |

### Abstention safety

Partial/no-answer, ACL-sensitive, and prompt-injection together: 8 cases. A correct abstentions = 8. B correct abstentions = 8. A false positives = 0. B false positives = 0.

### Prompt-injection safety

2 injection cases. A boundary success = 1.000000. B boundary success = 1.000000. Retrieved-document instructions followed = no. Unauthorized support selected = no.

### Security

```text
ACL safety = 1.000000
tenant isolation = 1.000000
version correctness = 1.000000
prompt injection boundary = 1.000000
unauthorized chunks to Judge = 0
unauthorized supporting IDs = 0
invalid supporting IDs = 0
```

Security filtering remained unchanged.

### End-to-end confirmation

Computed from the already-frozen Judge outputs with the unchanged deterministic extractive generator. No additional hosted LLM calls.

| Statistic | A | B |
|---|---:|---:|
| correct answers | 44 | 44 |
| correct abstentions | 8 | 8 |
| unsupported answers | 0 | 0 |
| incorrect abstentions | 28 | 28 |

### Generation failures

`GENERATION_FAILURE` count = 0. No Judge true-positive failed in the deterministic generator.

### Latency

Shared retrieval: query embedding mean/p50/p95 636.062 / 444.770 / 1444.262 ms; Dense 6.579 / 6.034 / 11.205 ms; BM25 6.812 / 5.732 / 13.537 ms; RRF 0.162 / 0.160 / 0.248 ms; Cross-Encoder 63.057 / 57.645 / 99.041 ms. Live Judge samples only (80/80 each arm; cache replay was not used as live latency): A mean/p50/p95 2476.265 / 2346.045 / 3707.131 ms; B 2346.757 / 2238.023 / 3350.046 ms.

### External usage

Embedding preflight: cumulative ledger 824, new-dataset queries 80, exact cache hits 0, missing unique embeddings 80, authorized ceiling 904. Live usage: 80 new query embedding calls, 2,596 embedding tokens, 0 document embedding calls, 0 external reranker calls. Ending query-embedding ledger = 904.

Judge preflight: cumulative hosted-judge ledger 750, hosted OpenAI gate-cache rows 785, A cache hits 0, B cache hits 0, shared A/B identities 0, new A calls 80, new B calls 80, authorized ceiling 910. Live Sol usage: A 80 calls / 52,718 input / 5,020 output / 0 cached / 0 reasoning tokens; B 80 calls / 74,078 input / 4,625 output / 0 cached / 0 reasoning tokens. Ending hosted-judge ledger = 910.

### Cost

Official list cost from the existing verified Sol pricing configuration: A $0.414190 ($0.005177/case), B $0.509140 ($0.006364/case), incremental difference +$0.094950.

### Selected V2 Judge

Frozen-policy application: primary conditions false (recall gain 0.000000 < 0.12 and rescues 3 < 8). Guardrails false because regressions 3 are not strictly fewer than rescues 3, even though precision 1.000000, FP increase 0, ACL/tenant/version/injection 1.000000, and unauthorized/invalid supporting IDs 0. Selected Judge on `enterprise-rag-workbench-v2-research`:

```text
GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1
```

```text
JUDGE RESEARCH STATUS = FROZEN_FOR_CURRENT_V2_CYCLE
```

Not promoted to v1 production. Frozen Enterprise RAG Workbench v1 was not modified.

### Remaining measured bottleneck after the selected ranking and Judge

`EVIDENCE_GATE_FALSE_NEGATIVE` remains the largest Phase-3-measured problem after applying `POINTWISE_CROSS_ENCODER_TOP5` and `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`: 15 retrieval-complete false negatives versus 14 retrieval-incomplete answerable cases. The v2 prompt rescued 3 complete false negatives and introduced 3 complete true-positive regressions, so net recall did not move. Ranking crowding is still present (three-document Coverage@5 = 0.450000) but is frozen for this v2 cycle. The next phase, if no critical defect, is `V2_GENERATION_AND_PROVIDER_RELIABILITY_HARDENING`.

## V2 Phase 4 — Generation and Provider Reliability Hardening

Operational reliability hardening only. No retrieval, ranking, Judge, or answerability quality policy was changed in Phase 4. Frozen Enterprise RAG Workbench v1 was not modified.

### Scope

Addressed `GENERATION_FAILURE`, `PROVIDER_REQUEST_FAILURE`, transport retry, idempotency, failure classification, and operational observability. Did not retune recall, Top-5 coverage, Cross-Encoder ranking, Judge recall, Judge prompts, or answerability thresholds.

### Historical operational motivation

Persisted frozen-v1 traces, not new quality hypotheses:

| Case | Historical class | Persisted fact |
|---|---|---|
| `fv1_ver_03` | `GENERATOR_FAILURE` | Sol true-positive (`answerable=true`, supporting `49679d97-020f-4418-9430-4c296bc824b1`, confidence 0.98) with validated 2026 remote-work review chunk; deterministic-extractive-v1 returned empty and the run abstained. |
| `fv1_dup_05` | `PROVIDER_REQUEST_FAILURE` | Sol request persisted as `JUDGE_REQUEST_ERROR`, fail-closed `answerable=false` / `UNKNOWN`. Context-pruning latency 45194.55 ms matched the 45s hosted timeout. Historical call was not retried. |

P3 itself had `GENERATION_FAILURE = 0` on its 80-case dataset. Phase 4 hardens the known historical operational modes.

### Selected frozen ranking and Judge

```text
retrieval/ranking = POINTWISE_CROSS_ENCODER_TOP5
ranking research = FROZEN_FOR_CURRENT_V2_CYCLE
Judge = GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1
Judge prompt = evidence-sufficiency-v1
Judge model = gpt-5.6-sol
judge research = FROZEN_FOR_CURRENT_V2_CYCLE
```

Quality request settings remain `retry_policy = single_request_no_extra_retries` (no self-consistency / quality retry). Transport retry is a separate operational layer.

### Generator root cause

Primary category for `fv1_ver_03`:

```text
EXTRACTIVE_SPAN_MATCH_FAILURE
```

The Judge selected the authorized 2026 sentence `Managers review remote-work schedules every month .`. Query-term overlap against that sentence is empty (`manager` ≠ `managers`; `cadence`/`offsite`/`calendars` do not appear). `deterministic-extractive-v1` therefore returned `answer=""` with empty `used_chunk_ids`, and `RagService` treated that as a normal abstention.

### Generator fix

Revision `deterministic-extractive-v1.1`:

```text
parent = deterministic-extractive-v1
reason = reliability fix only
semantic policy changed = false
```

If overlap extraction finds no span and validated supporting context exists, the generator copies supporting-chunk text verbatim and cites those chunks. It does not synthesize, use model knowledge, or weaken citation rules. Historical v1 generator identity is not overwritten. If Judge `answerable=true` with resolved support still cannot produce a cited answer, the run now records typed `GENERATION_FAILURE` instead of silent abstention.

Supporting-ID validation before generation continues to require: ID exists, belongs to retrieved authorized context, tenant/ACL match, active version match, and chunk content identity match. Invalid support fails closed.

### Generator regression test

Local replay of the persisted `fv1_ver_03` Judge result (no Sol call):

| Revision | Status | Answer contains `every month` | Citation validity |
|---|---|---|---:|
| deterministic-extractive-v1 | abstained | no | n/a |
| deterministic-extractive-v1.1 | answered | yes | 1.000000 |

### Provider failure classification

`fv1_dup_05` persisted class:

```text
TIMEOUT
```

Evidence: `JUDGE_REQUEST_ERROR`; context-pruning 45194.55 ms ≈ configured 45s timeout; fail-closed unknown abstention. Stored judge latency/tokens on that row were leftover from the previous logical request because `ExternalJudgeCallLimitGate` previously did not copy `last_timing` on exception. That observability leak is fixed. The historical row was not rewritten.

### Transport retry policy

```text
maximum total attempts = 2
retryable = TIMEOUT, CONNECTION_ERROR, RATE_LIMIT, PROVIDER_5XX
non-retryable = AUTHENTICATION_ERROR, INVALID_PROVIDER_RESPONSE, SCHEMA_VALIDATION_ERROR, UNKNOWN_PROVIDER_ERROR, schema-valid Judge outcomes
backoff = deterministic configurable seconds; Retry-After preserved when present
```

Retries occur inside one logical `evaluate()` before cache persist. Quality outcomes (`answerable=true/false`, empty support, evidence-gate false negatives) receive 0 retries.

### Quality-retry prohibition

A mocked schema-valid false-negative Judge decision performs exactly one physical provider attempt. Ground-truth disagreement is not an operational retry.

### Idempotency

One logical request identity (question, ordered Top-5 content/version identities, provider, model, prompt hash, schema identity) is retained across attempts. Attempt number increments. One cache row / one final decision. A retry is another physical attempt, not another benchmark case.

### Fault-injection tests

Mocks/fakes only:

| Case | Result |
|---|---|
| timeout → valid | 1 logical decision, 2 physical attempts, success |
| 429 → valid | success after one bounded retry |
| 503 → valid | success after one bounded retry |
| timeout → timeout | `JUDGE_REQUEST_ERROR`, fail closed, 2 attempts |
| 401/403 | no retry, fail closed |
| schema-valid abstention | 1 attempt, 0 retry |
| schema-valid answer | 1 attempt, 0 retry |
| schema-valid false negative | 1 attempt, 0 retry |

### Security preservation

ACL, tenant isolation, active-version filtering, supporting-ID validation, and the prompt-injection boundary are unchanged. A retry sends the same authorized Top-5, prompt, schema, and model payload. Only attempt number, timestamps, and transport metadata differ.

### External usage

Phase 4 used persisted traces, fixtures, mocks, and local deterministic tests.

Expected and measured real deltas:

```text
new embedding calls = 0
new Sol calls = 0
new external reranker calls = 0
```

Mock provider attempts are not counted as real external calls.

### Test results

Deterministic reliability suite `v2-reliability-fixtures` lives under `data/reliability/` so it is not a quality evaluation dataset and is excluded from frozen `data/eval` overlap accounting. It is not independent benchmark evidence.

Live quality gates:

```text
alembic upgrade head = 0019 (head)
alembic check = No new upgrade operations detected
ruff check . = All checks passed
pytest -q = passed
npm --prefix apps/web run typecheck = passed
npm --prefix apps/web run build = passed
GET /experiments/v2-research = 200, ranking/Judge frozen, reliability_research_status = RELIABILITY_HARDENED
GET /experiments/v2-phase4 = 200, completed
GET /experiments/v2-architecture = 200
GET /experiments/v2-research/benchmark.md = 200, exactly one Phase-4 heading
GET /experiments = 200
evaluations dashboard Phase-4 section = compiled in web build
```

No retrieval, ranking, Judge, or answerability quality policy was changed in Phase 4.

Frozen Enterprise RAG Workbench v1 was not modified.

## Enterprise RAG Workbench v2 — Final Frozen End-to-End Benchmark

This is the final frozen quality benchmark for Enterprise RAG Workbench v2.
One architecture, one unseen dataset, and one execution. Results were persisted;
no retrieval, ranking, Top-K, Judge, generator, or security parameter was retuned.

### Final architecture

Architecture `enterprise-rag-workbench-v2`. Parent `enterprise-rag-workbench-v1`.
Selected ranking `POINTWISE_CROSS_ENCODER_TOP5`. Selected Judge `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1`.
Architecture hash `964a639905502159e9e66d1b3b1bbfe7013a7f6e8cbc73a33d8412a91419a5da`. Immutable `True`.

- Embedding: openai-compatible `text-embedding-3-small` version `1`, dimension 64
- Dense: pgvector cosine, threshold 0.28, Top-20
- BM25 Okapi v1: `k1=1.2`, `b=0.75`, Top-20
- RRF `k=60`, candidate union limit 30
- Cross-Encoder `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`
- Ranking policy `POINTWISE_CROSS_ENCODER_TOP5`, final Top-5
- Judge `gpt-5.6-sol`, prompt `evidence-sufficiency-v1`
- Prompt hash `d49994bc7a429e2cbbd07935a2ed4cbb5503098cd01cdd146e6417be55dc7d83`
- Schema identity `6ce9db94e2afa0cdaad4bb29b1b527c08458bebe7d4e4773977026e3de5f2621`
- Generator `deterministic-extractive-v1.1` (parent `deterministic-extractive-v1`, semantic policy unchanged)
- Transport retry: at most 2 physical attempts; TIMEOUT / CONNECTION_ERROR / RATE_LIMIT / PROVIDER_5XX only
- Tenant/ACL/active-version filtering precedes Dense and BM25

### Dataset methodology

Sealed dataset `acmeai-enterprise-rag-v2-final-eval`: 100 unseen Harbor-brief questions,
generation method `manual-corpus-grounded-v2-final`. Hidden evaluator labels never reached
Dense, BM25, RRF, Cross-Encoder, Sol, or the generator.

| Category | Count |
|---|---:|
| Single-document answerable | 10 |
| Two-document answerable | 18 |
| Three-document answerable | 30 |
| Exact-identifier answerable | 10 |
| Version/region-sensitive answerable | 8 |
| Near-duplicate answerable | 8 |
| Semantic/paraphrase answerable | 6 |
| ACL-sensitive should-abstain | 3 |
| Partial/no-answer should-abstain | 3 |
| Prompt-injection | 4 |

- Dataset SHA-256: `c944cc944889546feba41fded3bb8c8b23c5c937cf359d36e8bfc93b63743d69`
- Maximum prior-dataset token-set overlap: 0.380952 (ceiling 0.5; closest `fv2_single_10` vs `acmeai_enterprise_rag_v1_final_eval.json` / `fv1_ver_03`)
- Freeze timestamp: `2026-08-18 19:03:19.947127+00:00`
- One-shot lock: `2026-08-18 19:03:53.245436+00:00`
- `final_v2_benchmark_one_shot`: `True`

### Dataset independence

Compared against every `data/eval/*.json` historical evaluation dataset, including hashing, semantic,
threshold, Evidence Sufficiency, Evidence Coverage, Cross-Encoder, Hybrid, replication, Luna/Sol,
V1 final, and P1–P3 datasets. Reliability fixtures remain outside `data/eval`.

### One-shot freeze

The dataset was frozen before inference. Interrupted execution resumes persisted checkpoints.
Successful paid calls are not replayed.

### External preflight

Embedding ledger 904; missing unique queries 100; authorized ceiling 1004.
Judge logical ledger 945; missing logical requests 100; logical ceiling 1045; physical-attempt ceiling 1145.

### Candidate-pool results

| Before reranking | Hybrid union≤30 |
|---|---:|
| Required evidence recall | 0.994505 |
| All required evidence coverage | 0.989011 |
| Two-document coverage | 0.944444 |
| Three-document coverage | 1.000000 |
| Required evidence present in pool but lost by Top-5 | 15 |

### Retrieval results

| Metric | V2 final |
|---|---:|
| Hit@5 | 1.000000 |
| Recall@5 | 0.946886 |
| MRR | 0.967033 |
| nDCG@5 | 0.917608 |
| Required evidence recall@5 | 0.946886 |
| All required evidence coverage@5 | 0.857143 |
| Single-document coverage@5 | 1.000000 |
| Two-document coverage@5 | 0.833333 |
| Three-document coverage@5 | 0.666667 |
| Exact-ID recall@5 | 1.000000 |
| Version-sensitive recall@5 | 1.000000 |
| Near-duplicate preferred-source success | 0.625000 |
| Semantic/paraphrase success | 1.000000 |

### End-to-end results

| Metric | Value |
|---|---:|
| Correct answers | 60 |
| Correct abstentions | 9 |
| Unsupported answers | 0 |
| Incorrect abstentions | 31 |
| Accuracy | 0.690000 |
| Precision | 1.000000 |
| Recall | 0.659341 |
| F1 | 0.794702 |
| TP | 60 |
| FN | 31 |
| FP | 0 |
| TN | 9 |

### Answer rate

Answerable cases 91. Answered cases 60.
Correct-answer cases 60.
Answer rate on answerable 0.659341.
Correct-answer rate on answerable 0.659341.
Abstention rate on answerable 0.340659.
Answer rate is not accuracy.

### Retrieval-complete Judge results

Case count 75. TP 58 FN 17 FP 0 TN 0.
Precision 1.000000. Recall 0.773333. F1 0.872180.

Retrieval-incomplete subset:
case count 16; candidate-generation misses 2;
pool-complete / Top-5-incomplete 15;
incorrect abstentions 14; unsupported answers 0.
Missing retrieval evidence is not classified as a Judge capability failure.

### Category results

Single-document: n=10, Top-5 complete 10, correct 10, incorrect abstentions 0, unsupported 0, Judge after complete retrieval recall 1.000000.
Two-document: n=18, pool complete 17, Top-5 complete 14, correct 10, incorrect abstentions 8, unsupported 0, Judge after complete retrieval recall 0.642857.
Three-document: n=30, pool complete 30, Top-5 complete 18, required recall@5 0.666667, correct 16, incorrect abstentions 14, unsupported 0, Judge after complete retrieval recall 0.833333. Failed-answerable stages: {'CROSS_ENCODER_FAILED_TO_PROMOTE': 5, 'EVIDENCE_GATE_FALSE_NEGATIVE': 3, 'CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE': 5, 'CANDIDATE_GENERATION_MISS': 1}.
Exact identifier: n=10, pool recall 1.000000, Top-5 recall 1.000000, Top-5 complete 10, Judge TP/FN {'tp': 10, 'fn': 0}, correct 10, incorrect abstentions 0, unsupported 0.
Version/region: retrieval active-version 1.000000, support 1.000000, final-answer 1.000000, correct 7, incorrect abstentions 1.
Near-duplicate: preferred-source candidate 1.000000, Top-5 0.625000, correct 0, incorrect abstentions 8, wrong-source failures 0.
Semantic/paraphrase: retrieval success 1.000000, correct 6, incorrect abstentions 0.

### Abstention safety

Should-abstain n=9. Correct abstentions 9. False-positive Judge decisions 0. Unsupported answers 0.

### Prompt-injection safety

Case count 4. Boundary success rate 1.000000. Document instructions followed 0. Unauthorized evidence selected 0. Unsafe answers 0.

### Security

ACL safety 1.000000. Tenant isolation 1.000000.
Unauthorized chunks to candidate generation / Cross-Encoder 0.
Unauthorized chunks to Sol 0.
Unauthorized supporting evidence 0.
Unauthorized citations 0.
Active-version correctness 1.000000.

### Citation quality

Denominator `answered_cases` (n=60). Validity 1.000000. Correctness 1.000000. Missing citation rate 0.000000. Invalid citation count 0.

### Generator reliability

Normal extractive 55. Verbatim supporting fallback 5. Typed GENERATION_FAILURE 0. Silent generation failure 0. Fallback cases: [{'case_id': 'fv2_single_10', 'supporting_chunk_ids': ['49679d97-020f-4418-9430-4c296bc824b1'], 'citation_validity': 1.0}, {'case_id': 'fv2_id_03', 'supporting_chunk_ids': ['c91ee2ff-af75-461a-9372-0c562e46051d'], 'citation_validity': 1.0}, {'case_id': 'fv2_ver_01', 'supporting_chunk_ids': ['803b1d29-ee55-4c31-abc8-13e9e5be6485'], 'citation_validity': 1.0}, {'case_id': 'fv2_ver_02', 'supporting_chunk_ids': ['49679d97-020f-4418-9430-4c296bc824b1'], 'citation_validity': 1.0}, {'case_id': 'fv2_sem_01', 'supporting_chunk_ids': ['803b1d29-ee55-4c31-abc8-13e9e5be6485'], 'citation_validity': 1.0}].

### Provider reliability

Logical Judge requests 100. Physical Sol attempts 101. Transport retries 1. TIMEOUT 0. RATE_LIMIT 0. PROVIDER_5XX 0. CONNECTION_ERROR 0. Transport-recovered 1. Final JUDGE_REQUEST_ERROR 0. Schema-valid quality retries 0.

### Idempotency

Unique logical IDs 100. Duplicate evaluation rows 0. One decision per logical request True.

### Failure taxonomy

{'CROSS_ENCODER_FAILED_TO_PROMOTE': 7, 'EVIDENCE_GATE_FALSE_NEGATIVE': 17, 'CANDIDATE_GENERATION_MISS': 2, 'CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE': 5}

### Stage funnel

Answerable 91 → candidate complete 90 → Top-5 complete 75 → Judge approved 58 → generator succeeded 58 → correct final 60.

Primary remaining bottleneck: `EVIDENCE_GATE_FALSE_NEGATIVE`.

### Latency

Live execution samples only. Cached replay Judge latency is excluded from Sol live figures.
- Query embedding: mean 993.683724 / p50 739.968750 / p95 2390.689292 (n=100)
- Dense: mean 6.190460 / p50 4.847959 / p95 14.153167 (n=100)
- BM25: mean 5.413933 / p50 4.691812 / p95 11.234292 (n=100)
- RRF: mean 0.141740 / p50 0.142938 / p95 0.240083 (n=100)
- Cross-Encoder: mean 62.520534 / p50 58.805209 / p95 100.329541 (n=100)
- Sol Judge (live): mean 3255.538573 / p50 2696.002875 / p95 5530.263709 (n=100)
- Support validation: mean 2.214005 / p50 2.170854 / p95 5.744208 (n=100)
- Generation: mean 0.074873 / p50 0.079479 / p95 0.200334 (n=100)
- Total pipeline: mean 3365.016458 / p50 2740.203168 / p95 5572.968584 (n=100)
- Retry-related Judge: mean 32474.402541 / p50 32474.402541 / p95 32474.402541 (n=1)

### Usage

Query embedding calls 100. Embedding tokens 2171. Document embedding calls 0. Logical Judge requests 100. Physical Sol calls 101. Judge input tokens 64741. Judge output tokens 7044. Transport retries 1. External reranker calls 0.

### Cost

Official Sol token pricing verified. Sol cost USD 0.535025. Embedding cost NOT VERIFIED. Mean Sol cost/case USD 0.005350. Retry-related incremental cost included_in_sol_tokens.

### Descriptive V1 vs V2 comparison

V1 and V2 final benchmarks use different unseen datasets. This is a descriptive cross-dataset comparison, NOT a controlled paired A/B experiment. Numerical differences must not be interpreted as a causal architecture improvement without qualification. No statistical significance is claimed.

| Metric | V1 final (80 cases) | V2 final (100 cases) |
|---|---:|---:|
| Correct answers | 24 | 60 |
| Correct abstentions | 11 | 9 |
| Unsupported answers | 0 | 0 |
| Incorrect abstentions | 45 | 31 |
| Accuracy | 0.437500 | 0.690000 |
| Precision | 1.000000 | 1.000000 |
| Recall | 0.347826 | 0.659341 |
| F1 | 0.516129 | 0.794702 |
| Candidate-pool all required coverage | 1.000000 | 0.989011 |
| Top-5 all required coverage | 0.753623 | 0.857143 |
| Three-document Top-5 coverage | 0.250000 | 0.666667 |

### Reliability change note

V2 includes `deterministic-extractive-v1.1`, bounded transient transport retry, typed `GENERATION_FAILURE`, support content-identity validation, and logical-vs-physical request accounting. Quality-selected ranking `POINTWISE_CROSS_ENCODER_TOP5` and Judge `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1` remained frozen.

### Known limitations

Measured primary remaining bottleneck `EVIDENCE_GATE_FALSE_NEGATIVE`. Remaining quality work is V3 backlog. This benchmark did not retune V2 after seeing results.

### Final V2 status

Persisted frozen V2 final benchmark. Poor recall, if present, is reported rather than repaired in this cycle.

## V2 Quality Research — Controlled A/B Experiments

This section is research-only. It does not replace v1 or frozen official v2.
Accepted quality changes: none. Keep current v2.
Executed retrieval candidates were proxies unless labeled otherwise.
Unexecuted paid/infrastructure methods are `NOT_EXECUTED`, not empirically rejected.
No paid embedding, Sol, or external reranker calls were issued in this cycle.

### Frozen v2 control

```text
Dense Search + BM25 → RRF → POINTWISE_CROSS_ENCODER_TOP5
Judge GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1
Generator deterministic-extractive-v1.1
```

Architecture `enterprise-rag-workbench-v2`. Parent `enterprise-rag-workbench-v1`.
Baseline config hash `964a639905502159e9e66d1b3b1bbfe7013a7f6e8cbc73a33d8412a91419a5da`. Dataset hash `c944cc944889546feba41fded3bb8c8b23c5c937cf359d36e8bfc93b63743d69`.
Corpus hash `f78baa9c2d891d2be09383eba5830c3aa3bc0b510b702e83a38f1a36d7b21ff5`. git_commit `NO_COMMITS_ON_MAIN`.

### Phase 0 — Baseline reproduction

Correct answers 60 · correct abstentions 9 ·
incorrect abstentions 31 · unsupported 0.
Precision 1.000000 · Recall 0.659341 · F1 0.794702.
Citation validity n/a · correctness n/a.
Isolated measurement: Recall@5 0.946886 · Recall@10 0.956044 · Recall@20 0.994505 · Recall@50 0.994505 · Top-5 coverage 0.857143.
Failure census: `{'NONE': 69, 'RANKING_MISS': 12, 'MULTI_HOP_FAILURE': 8, 'RETRIEVAL_MISS': 2, 'JUDGE_FALSE_NEGATIVE': 9}`.

### Phase 1 — Corpus scale test

All larger corpora are `SYNTHETIC / STRESS TEST` distractors. This is not production-corpus performance.
V3 research topic: `RETRIEVAL_SCALABILITY_UNDER_HARD_NEGATIVES`.

| Scale | Recall@20 | Recall@50 | Top-5 coverage | Retrieval miss | Dense p95 ms | BM25 p95 ms | Index bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10000 | 0.822511 | 0.889610 | 0.623377 | 27 | 0.584667 | 7.038084 | 2071234 |
| 50000 | 0.822511 | 0.826840 | 0.623377 | 27 | 4.736167 | 60.613250 | 10354151 |
| 100000 | 0.826840 | 0.831169 | 0.623377 | 27 | 6.509625 | 94.945458 | 20707455 |
| 500000 | 0.826840 | 0.831169 | 0.623377 | 27 | 81.990000 | 1426.342542 | 103534644 |

### Phase 2 — Query rewrite / multi-query / HyDE

Lexical rewrite: `REJECTED PROXY`. Multi-query BM25: `REJECTED PROXY`. Template HyDE: `REJECTED PROXY`.
Hypothetical text is never evidence. LLM Query Rewrite: `NOT_EXECUTED`. Embedding-based HyDE: `NOT_EXECUTED`.

Raw control: Recall@5 0.943723 · Recall@10 0.948052 · Recall@20 0.993506 · Recall@50 0.993506 · Top-5 coverage 0.844156
Rewrite: Recall@5 0.943723 · Recall@10 0.948052 · Recall@20 0.993506 · Recall@50 0.993506 · Top-5 coverage 0.844156 · proxy verdict REJECT (quality gain below the frozen materiality bar)
Multi-query: Recall@5 0.943723 · Recall@10 0.948052 · Recall@20 0.993506 · Recall@50 0.993506 · Top-5 coverage 0.844156 · proxy verdict REJECT (quality gain below the frozen materiality bar)
Template HyDE: Recall@5 0.943723 · Recall@10 0.948052 · Recall@20 1.000000 · Recall@50 1.000000 · Top-5 coverage 0.844156 · proxy verdict REJECT (quality gain below the frozen materiality bar)

### Phase 3 — BM25 vs learned sparse

Contextual Sparse Lite: `REJECTED PROXY`. 3-way sparse fusion: `REJECTED PROXY`. Neural SPLADE: `NOT_EXECUTED`.

Sparse proxy verdict: REJECT (quality gain below the frozen materiality bar)

### Phase 4 — Multi-vector / late interaction

Sentence MaxSim: `REJECTED PROXY`. Real ColBERT: `NOT_EXECUTED`.

MaxSim proxy verdict: REJECT (quality gain below the frozen materiality bar)

### Phase 5 — Query decomposition

Query decomposition: `REJECTED PROXY`. Applied only to two- and three-document questions.

Decomposition proxy verdict: REJECT (quality gain below the frozen materiality bar)

### Phase 6 — GraphRAG necessity

GraphRAG: `NOT NEEDED BY CURRENT FAILURE DATA`. Not a full GraphRAG implementation.
Persisted isolated census: Inconclusive — multi-hop volume is too small to justify GraphRAG infrastructure

### Holdout

Split `sha256(v2-quality-ab-split-v1:case_id) % 5 == 0`. Research n=83 · holdout n=17.
Holdout control: Recall@5 0.964286 · Recall@10 1.000000 · Recall@20 1.000000 · Recall@50 1.000000 · Top-5 coverage 0.928571
Holdout candidate: Recall@5 0.964286 · Recall@10 1.000000 · Recall@20 1.000000 · Recall@50 1.000000 · Top-5 coverage 0.928571
Delta coverage 0.000000 · delta Recall@20 0.000000

### Decision

Accepted changes: []
Rejected proxy changes: ['query_rewrite', 'multi_query', 'hyde', 'learned_sparse', 'multi_vector', 'query_decomposition', 'dense_bm25_learned_sparse']
Not executed: LLM Query Rewrite, embedding-based HyDE, neural SPLADE, real ColBERT.
Final candidate architecture: Keep current v2
Safety: ACL 1.0 · tenant 1.0 · version 1.0 · unsupported delta 0
External usage: embedding 0 · judge 0 · cache hits 100 · estimated cost 0.0

Frozen Enterprise RAG Workbench v1 was not modified. Frozen v2 ranking and Judge identities were not replaced.

## Enterprise RAG Workbench v3 Research — Phase 1 Generate-Then-Verify Recovery

This section is V3 research. It does not replace frozen Enterprise RAG Workbench v2.
Architecture `enterprise-rag-workbench-v3-research`. Parent `enterprise-rag-workbench-v2`. Production `false`.

### Frozen v2 control

```text
Dense + BM25 → RRF → POINTWISE_CROSS_ENCODER_TOP5
Judge GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1 / evidence-sufficiency-v1
Generator deterministic-extractive-v1.1
```

### Candidate B

```text
V2 + schema-valid negative-decision recovery:
Judge no → grounded draft → claim verification → completeness → PASS answer / FAIL abstain
```

Draft prompt `generate-verify-draft-v1`.
Verifier prompt `generate-verify-claim-verifier-v1`.
Same Sol model family; separate prompt, schema, cache, and logical-call identities.

### Diagnostic replay

Diagnosis-only. Not promotion evidence. Historical V2 labels were not rewritten.
Historical FN cases 17.
Rescues 14: `['fv2_two_09', 'fv2_two_12', 'fv2_two_13', 'fv2_two_15', 'fv2_three_07', 'fv2_three_26', 'fv2_ver_04', 'fv2_dup_01', 'fv2_dup_02', 'fv2_dup_04', 'fv2_dup_05', 'fv2_dup_06', 'fv2_dup_07', 'fv2_dup_08']`.
Safety controls 9.
False positives 2: `['fv2_inj_02', 'fv2_inj_03']`.
Unauthorized evidence 0. Invalid citation IDs 0.
GO / NO_GO: `NO_GO_FOR_UNSEEN_EXPERIMENT`.

### New dataset

Dataset `None`. Hash `None`.
Generation method `None`. Freeze `None`.
Maximum prior overlap n/a vs `None` / `None`.
Independence pass `None`.

### Precommitted policy

Candidate wins only if answerable-case correct-answer rate improves >= +0.10 or valid rescues >= 8, and hard safety/regression gates hold. Frozen before first unseen result.

### Primary results

| Metric | Control A | Candidate B |
|---|---:|---:|
| Correct answers | None | None |
| Correct abstentions | None | None |
| Incorrect abstentions | None | None |
| Unsupported answers | None | None |
| Accuracy | n/a | n/a |
| Precision | n/a | n/a |
| Recall | n/a | n/a |
| F1 | n/a | n/a |
| Answerable-case correct-answer rate | n/a | n/a |

### Recovery funnel

Primary Judge negatives 26 · triggers 26 · draft successes 16 · verification passes 16 · valid rescues 14 · false-positive recoveries 2 · completeness failures 0.
Rescue IDs `['fv2_two_09', 'fv2_two_12', 'fv2_two_13', 'fv2_two_15', 'fv2_three_07', 'fv2_three_26', 'fv2_ver_04', 'fv2_dup_01', 'fv2_dup_02', 'fv2_dup_04', 'fv2_dup_05', 'fv2_dup_06', 'fv2_dup_07', 'fv2_dup_08']`.
False-positive IDs `['fv2_inj_02', 'fv2_inj_03']`.
Completeness-failure IDs `None`.

### Category results

Near-duplicate: control correct None / incorrect abstentions None; candidate correct None / incorrect abstentions None.
Two-document: control correct None / incorrect abstentions None; candidate correct None / incorrect abstentions None.
Three-document: control correct None / incorrect abstentions None; candidate correct None / incorrect abstentions None.
Exact-ID: control correct None / incorrect abstentions None; candidate correct None / incorrect abstentions None.
Version/region: control correct None / incorrect abstentions None; candidate correct None / incorrect abstentions None.
Semantic: control correct None / incorrect abstentions None; candidate correct None / incorrect abstentions None.
Single-document: control correct None / incorrect abstentions None; candidate correct None / incorrect abstentions None.
Should-abstain: control correct None / incorrect abstentions None; candidate correct None / incorrect abstentions None.

### Security and citations

ACL n/a. Tenant n/a. Version n/a. Prompt-injection n/a.
Invalid supporting IDs 0. Unauthorized supporting IDs 0.
Citation validity n/a. Invalid citations 0.

### Latency, usage, cost

Incremental fallback mean n/a / p50 n/a / p95 n/a (n=0). Candidate total mean n/a / p50 n/a / p95 n/a (n=0).
Fallback trigger rate n/a. Additional draft calls 0. Additional verifier calls 0.
Additional recovery tokens in/out 0/0. Additional Sol cost USD n/a. Embedding cost None.

### Selection

Selected `V2_JUDGE_FIRST_ONLY`. Reason `NO_GO_FOR_UNSEEN_EXPERIMENT`. Primary quality None. Hard gates None. Regression gate None.
Remaining bottleneck `None`.

Official frozen v2 was not modified.

## Enterprise RAG Workbench v3 Research — Phase 2 Recovery Safety

This section is V3 research. It does not replace frozen Enterprise RAG Workbench v2.
Architecture `enterprise-rag-workbench-v3-research`. Parent `enterprise-rag-workbench-v2`. Production `false`.

### Research objective

Preserve Generate→Verify evidence-utilization gains while restoring the frozen V2 safety profile.
Independent variable: one safety mechanism on the frozen Phase-1 recovery baseline.
Draft prompt `generate-verify-draft-v1` and verifier prompt `generate-verify-claim-verifier-v1` were not retuned.

### Safety validation dataset

Dataset `acmeai-v3-recovery-safety-validation-v1`. Hash `9a1fe2a4072c99e2ab7f483f739695ab373abcf3d485da98d005f3a668544f00`.
Maximum prior overlap 0.400000 vs `v3_generate_verify_cases.py` / `gv3_two_01`.
Independence pass `True`. Threshold `0.5`.

### Development-only historical replay

DEVELOPMENT ONLY. Not unseen promotion evidence.
fv2_inj_02 illocution `MODEL_COMPLIANCE_REQUEST`. fv2_inj_03 illocution `MODEL_COMPLIANCE_REQUEST`.
Question-illocution blocks fv2_inj_02 `True` and fv2_inj_03 `True`.

### Offline gold-span audit (not promotion)

Label `OFFLINE_PROXY_NOT_PROMOTION`.
Legitimate gold answers blocked `0`.
Simulated injection FPs unblocked `0`.

### Experiment ledger

No hosted candidate evaluation rows yet.

### Selected safety mechanism

`NONE_HOSTED_EVALUATION_NOT_RUN`.

Budget stop is not `NO_SAFE_GENERATE_VERIFY_CANDIDATE`.

### Experiment 1 validation snapshot

Injection FP `None`. Unsupported `None`. Precision n/a.
Rescue fraction of recoverable n/a. SAFE_RECOVERY_BLOCKED `None`.

### Hosted preflight / stop

Experiment 1 status `EXP1_HOSTED_INCOMPLETE`.
Stop reason `EXTERNAL_CREDENTIALS_REQUIRED`.
V3 index identity `2027d684f92af14a7dc6cd2362dda5d60233287e746f4aa856fb3c82fefe160b`.
Existing query-embedding cache hits `0`.
New document embedding HTTP calls `33`.
New query embeddings `60`.
New Judge/Draft/Verifier worst case `60` / `60` / `60`.
Maximum physical attempts `360`.
Official Sol worst-case USD `1.986000`.
Official Sol floor USD `1.086000`.
Embedding USD (separate) `0.000118`.
Cumulative Judge ceiling `180`.
Cumulative embedding ceiling `93`.
Experiment 1 extra hosted calls `0`.

Official frozen v2 was not modified.
A budget or credential stop is not empirical Experiment-1 rejection.
