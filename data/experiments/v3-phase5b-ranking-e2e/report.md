# V3 Phase 5B — Three-Arm Ranking E2E Benchmark

**Experiment ID:** `v3-phase5b-frozen-ranking-e2e`
**Lock ID:** `v3-phase5b-ranking-e2e`

## Arms

- **Reference R:** `V2_JUDGE_FIRST_ONLY` — Frozen stable V2 (pointwise CE Top-5 → Sol Judge → V2 answer/abstain)
- **Control A:** `V3_GENERATE_VERIFY_WITH_EVIDENCE_INSTRUCTION_BOUNDARY` — V3 research pipeline (pointwise CE Top-5 → Sol Judge → recovery on negative)
- **Candidate B:** `V3_PAIRWISE_COMPLEMENTARITY_RANKING_WITH_GENERATE_VERIFY` — Same as A but pairwise complementarity Top-5

## Dataset

- Dataset ID: `acmeai-enterprise-rag-v3-ranking-e2e-final-v2`
- Hash: `58ac25869720e094ada10717f4b10999a03382f4a7b45f95d88371730a63acdf`
- Cases: 120
- Distribution: {"acl_sensitive": 4, "exact_identifier": 10, "multidoc_three": 28, "multidoc_two": 20, "near_duplicate": 14, "partial_no_answer": 4, "prompt_injection": 10, "semantic_paraphrase": 8, "single_document": 12, "version_region": 10}

## End-to-End Metrics

| Metric | Reference R | Control A | Candidate B |
| ------ | ----------: | --------: | ----------: |
| Correct answers | 59 | 64 | 67 |
| Incorrect abstentions | 43 | 38 | 35 |
| Unsupported answers | 0 | 0 | 1 |
| Precision | 1.0 | 1.0 | 0.9852941176470589 |
| Recall | 0.5784313725490197 | 0.6274509803921569 | 0.6568627450980392 |
| F1 | 0.732919254658385 | 0.7710843373493976 | 0.7882352941176471 |
| Answerable correct rate | 0.5784313725490197 | 0.6274509803921569 | 0.6568627450980392 |

## Promotion Decision

- **Decision:** `KEEP_CURRENT_V3_RESEARCH_ARCHITECTURE`
- **V3 Status:** `V3_CANDIDATE_REJECTED`
- Quality gate (B vs A): False
- Quality gate (B vs R): False
- Safety gate: False
- Regression gate: False
- Ranking gate: True

## Ranking Gates

- Exact-ID Recall: Control=0.0, Candidate=0.0, Pass=True
- Version Correctness: Control=1.0, Candidate=1.0, Pass=True
- Same-doc degradation: 0.0, Pass=True
- Unauthorized downstream evidence: 0

**Completed at:** 2026-08-19 13:33:48.786720+00:00
