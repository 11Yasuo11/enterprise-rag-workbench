export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Citation = {
  document_id: string;
  chunk_id: string;
  source: string;
  title: string;
  version: string;
  page: number | null;
  section: string | null;
};

export type RetrievalResult = {
  chunk_id: string;
  document_id: string;
  text: string;
  rank: number;
  score: number;
  source: string;
  source_type: string;
  title: string;
  version: string;
  page: number | null;
  section: string | null;
  metadata: Record<string, unknown>;
};

export type ExperimentSummary = {
  id: string;
  name: string;
  status: string;
  started_at: string;
  completed_at: string | null;
  config: {
    config_hash: string;
    index_identity: string;
    corpus_version: string;
    evaluation_dataset_version: string;
    ingestion: Record<string, unknown>;
    retrieval: Record<string, unknown>;
    generation: Record<string, unknown>;
    answerability_gate: Record<string, unknown> | null;
  };
  aggregate_metrics: Record<string, number | null>;
  category_metrics: Record<string, Record<string, number | null>>;
  latency: {
    retrieval_ms: number | null;
    query_embedding_ms: number | null;
    embedding_cache_lookup_ms: number | null;
    vector_search_ms: number | null;
    acl_filter_ms: number | null;
    context_construction_ms: number | null;
    answerability_gate_cache_lookup_ms: number | null;
    answerability_judge_ms: number | null;
    context_pruning_ms: number | null;
    generation_ms: number | null;
    total_ms: number | null;
  };
  usage: {
    input_tokens: number | null;
    output_tokens: number | null;
    embedding_tokens: number | null;
    query_embedding_cache_hits: number | null;
    query_embedding_cache_misses: number | null;
    external_embedding_calls: number | null;
    gate_cache_hits: number | null;
    gate_cache_misses: number | null;
    external_judge_calls: number | null;
    local_judge_calls: number | null;
    judge_prompt_tokens: number | null;
    judge_completion_tokens: number | null;
  };
  failure_counts: Record<string, number>;
  error: string | null;
};

export type ExperimentCase = {
  id: string;
  case_id: string;
  category: string;
  question: string;
  expected_answer: string | null;
  expected_document_ids: string[];
  expected_chunk_ids: string[];
  forbidden_document_ids: string[];
  expected_versions: Record<string, string>;
  principal: { principal_id: string; tenant_id: string; permission_groups: string[] };
  retrieved_document_ids: string[];
  retrieved_chunk_ids: string[];
  supporting_chunk_ids: string[];
  generation_context_chunk_ids: string[];
  answerability_result: {
    answerable: boolean;
    supporting_chunk_ids: string[];
    confidence: number | null;
    reason_code: string;
    requirements?: Array<{
      requirement_id: string;
      description: string;
      status: string;
      supporting_chunk_ids: string[];
    }>;
  } | null;
  retrieval_trace: Array<{ chunk_id: string; document_id: string; version: string; rank: number; score: number; text: string }>;
  answer: string | null;
  citations: Array<Record<string, unknown>>;
  expected_abstain: boolean;
  abstained: boolean | null;
  metrics: Record<string, number | null>;
  failure_type: string | null;
  failure_types: string[];
  failure_details: Record<string, unknown>;
  security_passed: boolean | null;
  latency: ExperimentSummary["latency"];
  query_embedding_cache_hit: boolean | null;
  gate_cache_hit: boolean | null;
  external_judge_calls: number | null;
  local_judge_calls: number | null;
  answerability_operational_error: string | null;
  error: string | null;
  rag_run_id: string | null;
};

export type ExperimentDetail = ExperimentSummary & { cases: ExperimentCase[] };

export type EvidenceJudgeBenchmark = {
  split_identity: string;
  selected_candidate: string | null;
  selection_reason?: string;
  locked_at?: string;
  calibration?: Record<string, ExperimentDetail>;
  holdout_started: boolean;
  holdout_completed: boolean;
  holdout?: Record<string, ExperimentDetail>;
};

export type MultiDocumentBenchmark = {
  dataset_id: string;
  dataset_hash: string;
  split_identity: string;
  split_seed: number;
  selected_candidate: "C2" | "D" | null;
  selection_reason?: string;
  locked_at?: string;
  holdout_started: boolean;
  holdout_completed: boolean;
  calibration?: Record<"C2" | "D", ExperimentDetail>;
  holdout?: Record<string, ExperimentDetail>;
};

export type RetrievalBenchmarkRun = {
  id: string;
  metrics: Record<string, number>;
  category_metrics: Record<string, Record<string, number>>;
  branch_contribution: Record<string, number | string[]>;
  latency: Record<string, number>;
  usage: Record<string, number>;
};

export type RetrievalBenchmark = {
  dataset_id: string;
  dataset_hash: string;
  case_count: number;
  category_distribution: Record<string, number>;
  split_seed: number;
  split_identity: string;
  selected_retrieval_mode: "DENSE" | "HYBRID_RRF" | null;
  selection_reason?: string;
  holdout_started: boolean;
  holdout_completed: boolean;
  calibration?: Record<"DENSE" | "LEXICAL_BM25" | "HYBRID_RRF", RetrievalBenchmarkRun>;
  holdout?: Record<"DENSE" | "LEXICAL_BM25" | "HYBRID_RRF", RetrievalBenchmarkRun>;
};

export type HybridRerankerBenchmarkRun = {
  metrics: Record<string, number>;
  category_metrics: Record<string, Record<string, number>>;
  latency: Record<string, number>;
  usage: Record<string, number>;
  branch_contribution?: {
    candidate_pool?: Record<string, number>;
    lexical_rescues?: Record<string, number>;
    cross_encoder_conversion?: Record<string, number | null>;
    three_document?: Record<string, number>;
  };
};

export type HybridRerankerBenchmark = {
  dataset_id: string;
  dataset_hash: string;
  case_count: number;
  category_distribution: Record<string, number>;
  split_seed: number;
  split_identity: string;
  selected_mode: "DENSE_CROSS_ENCODER_RERANK" | "HYBRID_CROSS_ENCODER_RERANK" | null;
  selection_reason?: string;
  locked: boolean;
  holdout_started: boolean;
  holdout_completed: boolean;
  e2e_completed: boolean;
  retrieval_completed?: boolean;
  architecture_frozen?: boolean;
  production_retriever_status?: "DENSE_CROSS_ENCODER_RERANK" | "HYBRID_CROSS_ENCODER_RERANK";
  production_judge_status?: string;
  calibration_metrics?: {
    candidate_pools?: Record<string, Record<string, number>>;
    lexical_rescues?: Record<string, number>;
    cross_encoder_conversion?: Record<string, number | null>;
  };
  replication_metrics?: Record<string, unknown>;
  conversion_analysis?: Record<string, number | string[] | null | Record<string, unknown>>;
  calibration?: Record<"DENSE_CROSS_ENCODER_RERANK" | "HYBRID_CROSS_ENCODER_RERANK", HybridRerankerBenchmarkRun>;
  holdout?: Record<"DENSE_CROSS_ENCODER_RERANK" | "HYBRID_CROSS_ENCODER_RERANK", HybridRerankerBenchmarkRun>;
  retrieval?: Record<"DENSE_CROSS_ENCODER_RERANK" | "HYBRID_CROSS_ENCODER_RERANK", HybridRerankerBenchmarkRun>;
  end_to_end?: Record<"DENSE_CROSS_ENCODER_RERANK" | "HYBRID_CROSS_ENCODER_RERANK", RerankerEndToEndRun>;
};

export type V2Phase1Experiment = {
  dataset_id: string;
  dataset_hash: string;
  architecture_id: string;
  production_status: boolean;
  initialized: boolean;
  completed: boolean;
  selected_ranking?: "POINTWISE_CROSS_ENCODER_TOP5" | "DOCUMENT_DIVERSIFIED_TOP5" | "MAX_2_CHUNKS_PER_DOCUMENT_TOP5" | null;
  selected_v2_ranking?: "POINTWISE_CROSS_ENCODER_TOP5" | "DOCUMENT_DIVERSIFIED_TOP5" | "MAX_2_CHUNKS_PER_DOCUMENT_TOP5" | null;
  maximum_prior_overlap?: number;
  control_metrics?: { metrics?: Record<string, number> };
  candidate_metrics?: { metrics?: Record<string, number> };
  crowding_rescues?: { total?: number; three_document?: number; two_document?: number };
  diversification_regressions?: { count?: number };
  same_document_multichunk?: Record<string, number | string[]>;
  three_document_analysis?: Record<string, number | string[] | Record<string, unknown>[]>;
  security?: Record<string, number>;
  usage?: Record<string, number>;
  selection?: { reason?: string; selected_ranking?: string };
  primary_remaining_bottleneck?: string;
};

export type V2Phase2Experiment = {
  dataset_id: string;
  dataset_hash: string;
  architecture_id: string;
  production_status: boolean;
  initialized: boolean;
  completed: boolean;
  selected_ranking?: "POINTWISE_CROSS_ENCODER_TOP5" | "MAX_2_CHUNKS_PER_DOCUMENT_TOP5" | null;
  selected_v2_ranking?: "POINTWISE_CROSS_ENCODER_TOP5" | "DOCUMENT_DIVERSIFIED_TOP5" | "MAX_2_CHUNKS_PER_DOCUMENT_TOP5" | null;
  ranking_research_status?: string | null;
  maximum_prior_overlap?: number;
  control_metrics?: { metrics?: Record<string, number> };
  candidate_metrics?: { metrics?: Record<string, number> };
  occupancy_metrics?: {
    control?: Record<string, number>;
    candidate?: Record<string, number>;
    slots_freed_by_max_2_cap_total?: number;
  };
  crowding_rescues?: { total?: number; three_document?: number; two_document?: number };
  diversification_regressions?: { count?: number };
  same_document_two_chunk?: Record<string, number | string[]>;
  three_document_analysis?: Record<string, number | string[] | Record<string, unknown>[]>;
  security?: Record<string, number>;
  usage?: Record<string, number>;
  selection?: { reason?: string; selected_ranking?: string };
  primary_remaining_bottleneck?: string;
};

export type V2Phase3Experiment = {
  dataset_id: string;
  dataset_hash: string;
  architecture_id: string;
  production_status: boolean;
  initialized: boolean;
  completed: boolean;
  selected_ranking?: "POINTWISE_CROSS_ENCODER_TOP5" | null;
  selected_v2_ranking?: "POINTWISE_CROSS_ENCODER_TOP5" | "DOCUMENT_DIVERSIFIED_TOP5" | "MAX_2_CHUNKS_PER_DOCUMENT_TOP5" | null;
  selected_judge?: "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1" | "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V2" | null;
  selected_v2_judge?: "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1" | "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V2" | null;
  ranking_research_status?: string | null;
  judge_research_status?: string | null;
  maximum_prior_overlap?: number;
  retrieval_complete_judge?: {
    control?: { tp?: number; fn?: number; fp?: number; precision?: number | null; recall?: number | null; f1?: number | null };
    candidate?: { tp?: number; fn?: number; fp?: number; precision?: number | null; recall?: number | null; f1?: number | null };
  };
  false_negative_rescues?: { count?: number };
  false_positive_regressions?: { count?: number };
  security?: Record<string, number>;
  usage?: Record<string, number>;
  selection?: { reason?: string; selected_judge?: string };
  primary_remaining_bottleneck?: string;
};

export type V2Phase4Experiment = {
  suite_id: string;
  suite_kind?: string;
  not_a_quality_dataset?: boolean;
  architecture_id: string;
  production_status: boolean;
  initialized: boolean;
  completed: boolean;
  selected_v2_ranking?: "POINTWISE_CROSS_ENCODER_TOP5" | "DOCUMENT_DIVERSIFIED_TOP5" | "MAX_2_CHUNKS_PER_DOCUMENT_TOP5" | null;
  selected_v2_judge?: "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1" | "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V2" | null;
  ranking_research_status?: string | null;
  judge_research_status?: string | null;
  generator_root_cause?: string | null;
  generator_revision?: { id?: string; parent?: string; reason?: string; semantic_policy_changed?: boolean } | string | null;
  provider_failure?: { classification?: string; historical_retry?: boolean };
  transport_retry_policy?: {
    max_total_attempts?: number;
    retryable_classes?: string[];
    non_retryable_classes?: string[];
  };
  usage?: {
    new_embedding_calls?: number;
    new_sol_calls?: number;
    new_external_reranker_calls?: number;
    preflight?: Record<string, number>;
    postflight?: Record<string, number>;
  };
  reliability_status?: string | null;
};

export type V2FinalBenchmark = {
  dataset_id: string;
  dataset_hash?: string;
  architecture_id: string;
  parent_architecture?: string;
  architecture_hash?: string;
  initialized: boolean;
  completed: boolean;
  dataset_frozen?: boolean;
  one_shot?: boolean;
  final_v2_benchmark_one_shot?: boolean;
  selected_v2_ranking?: "POINTWISE_CROSS_ENCODER_TOP5" | "DOCUMENT_DIVERSIFIED_TOP5" | "MAX_2_CHUNKS_PER_DOCUMENT_TOP5" | null;
  selected_v2_judge?: "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1" | "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V2" | null;
  ranking_research_status?: string | null;
  judge_research_status?: string | null;
  case_count?: number;
  category_distribution?: Record<string, number>;
  maximum_prior_overlap?: number;
  closest_previous_case?: { final_case_id?: string; prior_dataset?: string; prior_case_id?: string; overlap?: number } | null;
  freeze_timestamp?: string | null;
  one_shot_locked_at?: string | null;
  retrieval?: {
    metrics?: Record<string, number | null>;
    pool?: Record<string, number | null>;
    required_evidence_present_in_pool_lost_by_top5?: number;
  };
  end_to_end?: Record<string, number | Record<string, number | null> | null>;
  stage_funnel?: Record<string, number>;
  category_results?: Record<string, Record<string, unknown>>;
  generator_reliability?: Record<string, unknown>;
  provider_reliability?: Record<string, unknown>;
  security?: Record<string, number | null>;
  citations?: Record<string, number | string | null>;
  usage?: Record<string, number | null>;
  cost?: Record<string, number | string | null>;
  v1_comparison?: {
    note?: string;
    v1?: Record<string, number | string>;
    v2?: Record<string, number | string>;
  };
  primary_remaining_bottleneck?: string | null;
};

export type V2ResearchBaseline = {
  architecture_id: string;
  parent_architecture: string;
  research_status: "ACTIVE";
  production_status: boolean;
  diagnosis_only: boolean;
  initialized: boolean;
  selected_v2_ranking?: "POINTWISE_CROSS_ENCODER_TOP5" | "DOCUMENT_DIVERSIFIED_TOP5" | "MAX_2_CHUNKS_PER_DOCUMENT_TOP5" | null;
  ranking_research_status?: string | null;
  selected_v2_judge?: "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1" | "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V2" | null;
  judge_research_status?: string | null;
  reliability_research_status?: string | null;
  phase4_lock_id?: string | null;
  control_equivalence_hash?: string;
  failure_census?: {
    failed_answerable_case_count?: number;
    counts?: Record<string, number>;
  };
  ranking_diagnostic?: {
    pool_complete_top5_incomplete_count?: number;
    primary_appearance?: string;
    appearance_counts?: Record<string, number>;
  };
  judge_false_negative_diagnostic?: {
    retrieval_complete_sol_false_negative_count?: number;
    class_counts?: Record<string, number>;
  };
  operational_diagnostic?: {
    class_counts?: Record<string, number>;
    historical_calls_retried?: boolean;
  };
  security_guardrails?: Record<string, number>;
  primary_bottleneck?: string;
  recommended_ranking_intervention?: string;
  phase1?: V2Phase1Experiment;
  phase2?: V2Phase2Experiment;
  phase3?: V2Phase3Experiment;
  phase4?: V2Phase4Experiment;
  final_v2?: V2FinalBenchmark;
};

export type V2QualityAbExperiment = {
  lock_id?: string;
  architecture_id: string;
  parent_architecture?: string;
  initialized: boolean;
  completed: boolean;
  git_commit?: string | null;
  baseline_config_hash?: string;
  dataset_hash?: string;
  corpus_hash?: string;
  baseline?: Record<string, unknown>;
  verdicts?: Record<string, { verdict?: string; reason?: string; coverage_gain?: number; recall_at_20_gain?: number }>;
  final_candidate?: { architecture?: string; accepted_changes?: string[]; rejected_changes?: string[] };
  holdout?: { research_cases?: number; holdout_cases?: number; delta_coverage?: number; delta_recall_at_20?: number };
  failure_census?: Record<string, unknown>;
  usage?: Record<string, number | string | null>;
  selected_v2_ranking?: string;
  selected_v2_judge?: string;
};

export type V3Phase1Experiment = {
  architecture_id: string;
  parent_architecture?: string;
  production_status?: boolean;
  initialized: boolean;
  completed: boolean;
  go_nogo?: string | null;
  selected_strategy?: string | null;
  dataset_id?: string | null;
  dataset_hash?: string | null;
  diagnostic?: {
    historical_fn_count?: number;
    rescue_count?: number;
    rescues?: string[];
    false_positive_count?: number;
    go_nogo?: string;
  };
  control_metrics?: Record<string, number>;
  candidate_metrics?: Record<string, number>;
  recovery_funnel?: Record<string, number>;
  primary_remaining_bottleneck?: string | null;
};

export type FinalV1Benchmark = {
  dataset_id: string;
  dataset_hash: string;
  case_count: number;
  category_distribution: Record<string, number>;
  completed: boolean;
  prepared: boolean;
  freeze_timestamp?: string | null;
  final_v1_retriever?: "DENSE_CROSS_ENCODER_RERANK" | "HYBRID_CROSS_ENCODER_RERANK" | null;
  production_retriever_status?: "DENSE_CROSS_ENCODER_RERANK" | "HYBRID_CROSS_ENCODER_RERANK";
  production_judge_status?: string;
  maximum_prior_overlap?: number;
  retrieval?: HybridRerankerBenchmarkRun;
  end_to_end?: {
    metrics: Record<string, number>;
    retrieval_metrics: Record<string, number>;
    category_metrics: Record<string, Record<string, number>>;
    latency: Record<string, number | { mean_ms?: number; p50_ms?: number; p95_ms?: number; count?: number } | null>;
    usage: Record<string, number>;
  };
  conversion_analysis?: Record<string, unknown>;
  release_architecture?: RetrievalArchitecture;
};

export type RetrievalArchitecture = {
  architecture_id: string;
  selected_retriever: "DENSE_CROSS_ENCODER_RERANK" | "HYBRID_CROSS_ENCODER_RERANK";
  configuration: Record<string, unknown>;
  frozen_at: string;
  immutable: boolean;
};

export type RerankingBenchmarkRun = {
  id: string;
  metrics: Record<string, number>;
  category_metrics: Record<string, Record<string, number>>;
  candidate_pool_metrics: Record<string, number>;
  rerankable_subset_metrics: Record<string, number>;
  movement_metrics: Record<string, number>;
  latency: Record<string, number>;
  usage: Record<string, number>;
};

export type RerankingBenchmark = {
  dataset_id: string;
  dataset_hash: string;
  case_count: number;
  category_distribution: Record<string, number>;
  split_seed: number;
  split_identity: string;
  selected_mode: "DENSE" | "DENSE_CROSS_ENCODER_RERANK" | null;
  selection_reason?: string;
  holdout_started: boolean;
  holdout_completed: boolean;
  calibration?: Record<"DENSE" | "DENSE_CROSS_ENCODER_RERANK", RerankingBenchmarkRun>;
  holdout?: Record<"DENSE" | "DENSE_CROSS_ENCODER_RERANK", RerankingBenchmarkRun>;
};

export type RerankerEndToEndRun = {
  id: string;
  metrics: Record<string, number>;
  retrieval_metrics: Record<string, number>;
  category_metrics: Record<string, Record<string, number>>;
  latency: Record<string, number>;
  usage: Record<string, number>;
};

export type RerankerEndToEndBenchmark = {
  dataset_id: string;
  dataset_hash: string;
  case_count: number;
  category_distribution: Record<string, number>;
  prepared: boolean;
  execution_started: boolean;
  completed: boolean;
  production_retriever_status?: "DENSE" | "DENSE_CROSS_ENCODER_RERANK";
  paired_transitions?: Record<string, number>;
  conversion_analysis?: Record<string, number>;
  regression_analysis?: Record<string, number>;
  results?: Record<"DENSE_C2" | "DENSE_CROSS_ENCODER_RERANK_C2", RerankerEndToEndRun>;
};

export type JudgeEndToEndRun = {
  id: string;
  metrics: Record<string, number>;
  subset_metrics: Record<string, Record<string, number> | number>;
  category_metrics: Record<string, Record<string, number>>;
  latency: Record<string, number>;
  usage: Record<string, number>;
};

export type JudgeEndToEndBenchmark = {
  dataset_id: string;
  case_count: number;
  completed: boolean;
  prepared: boolean;
  production_judge_status?: "EVIDENCE_SUFFICIENCY_V1" | "EVIDENCE_COVERAGE_V2";
  paired_transitions?: Record<string, number>;
  analysis?: {
    false_negative_rescues?: number;
    net_false_negative_improvement?: number;
    requirement_diagnostics?: Record<string, number>;
  };
  false_positive_regressions?: { count?: number };
  results?: Record<"EVIDENCE_SUFFICIENCY_V1" | "EVIDENCE_COVERAGE_V2", JudgeEndToEndRun>;
};

export type SolJudgeEndToEndBenchmark = {
  dataset_id: string;
  case_count: number;
  completed: boolean;
  prepared: boolean;
  production_retriever_status?: "DENSE_CROSS_ENCODER_RERANK";
  production_judge_status?: "GPT_5_6_LUNA_EVIDENCE_SUFFICIENCY_V1" | "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1";
  paired_transitions?: Record<string, number>;
  analysis?: {
    false_negative_rescues?: number;
    net_false_negative_improvement?: number;
    luna_correct_to_sol_false_abstention?: number;
  };
  false_positive_regressions?: { count?: number };
  results?: Record<"GPT_5_6_LUNA_EVIDENCE_SUFFICIENCY_V1" | "GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1", JudgeEndToEndRun>;
};

export type ExperimentComparisonRun = {
  id: string;
  name: string;
  status: string;
  config: Record<string, Record<string, unknown>>;
  total_latency_ms: number | null;
  latency: {
    query_embedding_ms: number | null;
    vector_search_ms: number | null;
    acl_filter_ms: number | null;
    context_construction_ms: number | null;
    generation_ms: number | null;
    total_ms: number | null;
  };
  cache: { hits: number | null; misses: number | null };
};

export type ExperimentComparison = {
  baseline: ExperimentComparisonRun;
  candidate: ExperimentComparisonRun;
  metrics: Record<string, { baseline: number | null; candidate: number | null; delta: number | null }>;
  latency_delta_ms: number | null;
  regressions: Array<{ metric: string; reason: string; security_critical: boolean }>;
  category_metrics: Record<string, Record<string, { baseline: number | null; candidate: number | null; delta: number | null }>>;
  configuration_differences: Array<{ field: string; baseline: unknown; candidate: unknown }>;
  hard_constraints: { acl_safety_required: number; version_accuracy_required: number; candidate_acceptable: boolean };
  passed: boolean;
  threshold_policy: string;
};

export type ExperimentAnalysisRun = {
  id: string;
  name: string;
  metrics: Record<string, number | null>;
  latency_ms: number | null;
  config: { ingestion: Record<string, unknown>; retrieval: Record<string, unknown> };
};

export type ExperimentAnalysis = {
  policy: string;
  acceptable_run_count: number;
  rejected_run_count: number;
  roles: {
    highest_recall: ExperimentAnalysisRun | null;
    best_abstention: ExperimentAnalysisRun | null;
    best_balanced: ExperimentAnalysisRun | null;
    lowest_latency: ExperimentAnalysisRun | null;
  };
  pareto_frontier: ExperimentAnalysisRun[];
};

export async function apiRequest<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `API request failed (${response.status})`);
  }
  return (await response.json()) as T;
}
