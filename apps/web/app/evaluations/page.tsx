"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { ErrorMessage } from "@/components/error-message";
import { PageHeading } from "@/components/page-heading";
import { apiRequest, EvidenceJudgeBenchmark, ExperimentAnalysis, ExperimentSummary, FinalV1Benchmark, HybridRerankerBenchmark, JudgeEndToEndBenchmark, MultiDocumentBenchmark, RerankerEndToEndBenchmark, RerankingBenchmark, RetrievalArchitecture, RetrievalBenchmark, SolJudgeEndToEndBenchmark, V2FinalBenchmark, V2Phase1Experiment, V2Phase2Experiment, V2Phase3Experiment, V2Phase4Experiment, V2QualityAbExperiment, V2ResearchBaseline, V3Phase1Experiment } from "@/lib/api";

const baselineConfig = {
  name: "baseline-hashing-eval-v1",
  identity: { corpus_version: "acmeai-v1", evaluation_dataset_version: "acmeai-eval-v1" },
  ingestion: {
    chunk_strategy: "fixed_token", chunk_size: 180, chunk_overlap: 30,
    embedding_provider: "hashing", embedding_model: "local-hashing-64",
    embedding_dimension: 64, embedding_version: "1",
  },
  retrieval: { top_k: 5, score_threshold: 0.2, distance_metric: "cosine" },
  generation: {
    llm_provider: "extractive", llm_model: "deterministic-extractive-v1",
    prompt_version: "baseline-v1", temperature: 0, context_budget: 1200,
  },
};

function metric(value: number | null | undefined) {
  return value == null ? "pending" : value.toFixed(3);
}

function latencyP50(
  value: number | { mean_ms?: number; p50_ms?: number; p95_ms?: number; count?: number } | null | undefined,
  fallback?: number | null,
) {
  if (value && typeof value === "object") {
    return metric(value.p50_ms ?? fallback);
  }
  return metric(typeof value === "number" ? value : fallback);
}

export default function EvaluationsPage() {
  const [runs, setRuns] = useState<ExperimentSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [analysis, setAnalysis] = useState<ExperimentAnalysis | null>(null);
  const [judgeBenchmark, setJudgeBenchmark] = useState<EvidenceJudgeBenchmark | null>(null);
  const [multidocBenchmark, setMultidocBenchmark] = useState<MultiDocumentBenchmark | null>(null);
  const [retrievalBenchmark, setRetrievalBenchmark] = useState<RetrievalBenchmark | null>(null);
  const [rerankingBenchmark, setRerankingBenchmark] = useState<RerankingBenchmark | null>(null);
  const [rerankerEndToEnd, setRerankerEndToEnd] = useState<RerankerEndToEndBenchmark | null>(null);
  const [judgeEndToEnd, setJudgeEndToEnd] = useState<JudgeEndToEndBenchmark | null>(null);
  const [solJudgeEndToEnd, setSolJudgeEndToEnd] = useState<SolJudgeEndToEndBenchmark | null>(null);
  const [hybridReranker, setHybridReranker] = useState<HybridRerankerBenchmark | null>(null);
  const [hybridReplication, setHybridReplication] = useState<HybridRerankerBenchmark | null>(null);
  const [architecture, setArchitecture] = useState<RetrievalArchitecture | null>(null);
  const [finalV1, setFinalV1] = useState<FinalV1Benchmark | null>(null);
  const [v2Research, setV2Research] = useState<V2ResearchBaseline | null>(null);
  const [v2Phase1, setV2Phase1] = useState<V2Phase1Experiment | null>(null);
  const [v2Phase2, setV2Phase2] = useState<V2Phase2Experiment | null>(null);
  const [v2Phase3, setV2Phase3] = useState<V2Phase3Experiment | null>(null);
  const [v2Phase4, setV2Phase4] = useState<V2Phase4Experiment | null>(null);
  const [v2Final, setV2Final] = useState<V2FinalBenchmark | null>(null);
  const [v2QualityAb, setV2QualityAb] = useState<V2QualityAbExperiment | null>(null);
  const [v3Phase1, setV3Phase1] = useState<V3Phase1Experiment | null>(null);

  const loadRuns = useCallback(async () => {
    try {
      const [items, evidence, judges, multidoc, retrieval, reranking, rerankerE2E, judgeE2E, solJudgeE2E, hybridRerank, hybridRepl, architectureRecord, finalBenchmark, v2Baseline, phase1, phase2, phase3, phase4, v2FinalBenchmark, qualityAb, v3Phase1Status] = await Promise.all([
        apiRequest<ExperimentSummary[]>("/experiments"),
        apiRequest<ExperimentAnalysis>("/experiments/analysis"),
        apiRequest<EvidenceJudgeBenchmark>("/experiments/evidence-judge"),
        apiRequest<MultiDocumentBenchmark>("/experiments/multidoc-coverage"),
        apiRequest<RetrievalBenchmark>("/experiments/retrieval-comparison"),
        apiRequest<RerankingBenchmark>("/experiments/reranking-comparison"),
        apiRequest<RerankerEndToEndBenchmark>("/experiments/reranker-e2e-comparison"),
        apiRequest<JudgeEndToEndBenchmark>("/experiments/judge-e2e-comparison"),
        apiRequest<SolJudgeEndToEndBenchmark>("/experiments/sol-judge-e2e-comparison"),
        apiRequest<HybridRerankerBenchmark>("/experiments/hybrid-reranker-comparison"),
        apiRequest<HybridRerankerBenchmark>("/experiments/hybrid-reranker-replication"),
        apiRequest<RetrievalArchitecture>("/experiments/retrieval-architecture").catch(() => null),
        apiRequest<FinalV1Benchmark>("/experiments/final-v1-benchmark").catch(() => null),
        apiRequest<V2ResearchBaseline>("/experiments/v2-research").catch(() => null),
        apiRequest<V2Phase1Experiment>("/experiments/v2-phase1").catch(() => null),
        apiRequest<V2Phase2Experiment>("/experiments/v2-phase2").catch(() => null),
        apiRequest<V2Phase3Experiment>("/experiments/v2-phase3").catch(() => null),
        apiRequest<V2Phase4Experiment>("/experiments/v2-phase4").catch(() => null),
        apiRequest<V2FinalBenchmark>("/experiments/v2-final-benchmark").catch(() => null),
        apiRequest<V2QualityAbExperiment>("/experiments/v2-quality-ab").catch(() => null),
        apiRequest<V3Phase1Experiment>("/experiments/v3-phase1").catch(() => null),
      ]);
      setRuns(items); setAnalysis(evidence); setJudgeBenchmark(judges); setMultidocBenchmark(multidoc); setRetrievalBenchmark(retrieval); setRerankingBenchmark(reranking); setRerankerEndToEnd(rerankerE2E); setJudgeEndToEnd(judgeE2E); setSolJudgeEndToEnd(solJudgeE2E); setHybridReranker(hybridRerank); setHybridReplication(hybridRepl); setArchitecture(architectureRecord); setFinalV1(finalBenchmark); setV2Research(v2Baseline); setV2Phase1(phase1); setV2Phase2(phase2); setV2Phase3(phase3); setV2Phase4(phase4); setV2Final(v2FinalBenchmark); setV2QualityAb(qualityAb); setV3Phase1(v3Phase1Status);
    }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Unknown request error"); }
  }, []);

  useEffect(() => { void loadRuns(); }, [loadRuns]);

  async function runBaseline() {
    setPending(true); setError(null);
    try {
      await apiRequest<ExperimentSummary>("/experiments/run", {
        config: baselineConfig,
        dataset: "eval_v1.json",
        options: { offline_only: true, max_cases: null, max_configs: 20, confirm_external_calls: false },
      });
      await loadRuns();
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Unknown request error"); }
    finally { setPending(false); }
  }

  const latest = runs[0];
  const hashingBaseline = runs.find((run) => run.name === "baseline-hashing-eval-v1");
  const semanticBaseline = runs.find((run) => run.name === "semantic-baseline-v1");
  const tunedSemantic = runs.find((run) => run.name === "semantic-threshold-028");
  return (
    <div className="page">
      <PageHeading eyebrow="Quality loop" title="Evaluation overview" description="Persistent, configuration-aware experiments keep retrieval, generation, safety, and latency evidence inspectable." />
      <div className="actions">
        <button className="button primary" onClick={runBaseline} disabled={pending}>{pending ? "Running 100 cases…" : "Run offline baseline"}</button>
        <Link className="button" href="/evaluations/compare">Compare experiments</Link>
      </div>
      <ErrorMessage message={error} />
      <section className="sectionBlock"><h2>Dense retrieval evidence</h2>
        <div className="evidenceGrid">
          {[{ label: "Hashing baseline", run: hashingBaseline }, { label: "Semantic baseline", run: semanticBaseline }, { label: "Selected semantic threshold", run: tunedSemantic }].map(({ label, run }) => <article className="panel" key={label}><span className="eyebrow">{label}</span><h3>{run?.name ?? "Pending"}</h3><p>Recall {metric(run?.aggregate_metrics.recall_at_k)} · Abstention F1 {metric(run?.aggregate_metrics.abstention_f1)}</p><p>Unsupported answers {metric(run?.aggregate_metrics.unsupported_answer_count)} · threshold {String(run?.config.retrieval.score_threshold ?? "pending")}</p></article>)}
        </div>
        <p className="note">{analysis?.policy ?? "Experiment recommendation is pending measured runs."}</p>
      </section>
      <section className="sectionBlock"><h2>Evidence sufficiency</h2>
        <div className="evidenceGrid">
          <article className="panel"><span className="eyebrow">Baseline vs gate</span><h3>{latest?.config.answerability_gate ? "Answerability gate enabled" : "Threshold-only baseline"}</h3><p>Unsupported {metric(latest?.aggregate_metrics.unsupported_answer_count)} · incorrect abstentions {metric(latest?.aggregate_metrics.incorrect_abstention_count)}</p></article>
          <article className="panel"><span className="eyebrow">Confusion matrix</span><h3>Answer behavior</h3><p>Correct answers {metric(latest?.aggregate_metrics.correct_answer_count)} · correct abstentions {metric(latest?.aggregate_metrics.correct_abstention_count)}</p></article>
          <article className="panel"><span className="eyebrow">Grounding</span><h3>Citation support</h3><p>Validity {metric(latest?.aggregate_metrics.citation_validity)} · support {metric(latest?.aggregate_metrics.claim_support_rate)}</p></article>
        </div>
        {judgeBenchmark?.calibration ? <div className="tableWrap"><table><thead><tr><th>Candidate</th><th>Answerability F1</th><th>Unsupported</th><th>Incorrect abstentions</th><th>Judge latency</th><th>Cache H/M</th><th>Calls</th></tr></thead><tbody>
          {Object.entries(judgeBenchmark.calibration).map(([candidate, run]) => <tr key={candidate}><td>{candidate}{candidate === judgeBenchmark.selected_candidate ? " · selected" : ""}</td><td>{metric(run.aggregate_metrics.answerability_f1)}</td><td>{metric(run.aggregate_metrics.unsupported_answer_count)}</td><td>{metric(run.aggregate_metrics.incorrect_abstention_count)}</td><td>{metric(run.latency.answerability_judge_ms)} ms</td><td>{run.usage.gate_cache_hits ?? 0} / {run.usage.gate_cache_misses ?? 0}</td><td>{(run.usage.external_judge_calls ?? 0) + (run.usage.local_judge_calls ?? 0)}</td></tr>)}
        </tbody></table></div> : <p className="note">Five-way judge calibration has not been locked.</p>}
        {judgeBenchmark?.selection_reason ? <p className="note">Selected {judgeBenchmark.selected_candidate}: {judgeBenchmark.selection_reason}</p> : null}
        {judgeBenchmark?.holdout ? <div className="tableWrap"><table><thead><tr><th>Holdout</th><th>Answerability F1</th><th>Unsupported</th><th>Incorrect abstentions</th><th>Multi-document</th><th>Exact identifiers</th></tr></thead><tbody>
          {Object.entries(judgeBenchmark.holdout).map(([candidate, run]) => <tr key={candidate}><td>{candidate}</td><td>{metric(run.aggregate_metrics.answerability_f1)}</td><td>{metric(run.aggregate_metrics.unsupported_answer_count)}</td><td>{metric(run.aggregate_metrics.incorrect_abstention_count)}</td><td>{metric(run.aggregate_metrics.multi_document_answer_success)}</td><td>{metric(run.aggregate_metrics.exact_identifier_answer_success)}</td></tr>)}
        </tbody></table></div> : null}
      </section>
      <section className="sectionBlock"><h2>Multi-document evidence coverage v2</h2>
        {multidocBenchmark?.calibration ? <>
          <div className="tableWrap"><table><thead><tr><th>Calibration</th><th>Required recall</th><th>Required precision</th><th>All evidence</th><th>Context loss</th><th>False abstention</th><th>Unsupported</th></tr></thead><tbody>
            {Object.entries(multidocBenchmark.calibration).map(([candidate, run]) => <tr key={candidate}><td>{candidate}{candidate === multidocBenchmark.selected_candidate ? " · selected" : ""}</td><td>{metric(run.aggregate_metrics.required_evidence_recall)}</td><td>{metric(run.aggregate_metrics.required_evidence_precision)}</td><td>{metric(run.aggregate_metrics.all_required_evidence_coverage_rate)}</td><td>{metric(run.aggregate_metrics.supporting_context_loss_rate)}</td><td>{metric(run.aggregate_metrics.multi_document_false_abstention_rate)}</td><td>{metric(run.aggregate_metrics.unsupported_answer_count)}</td></tr>)}
          </tbody></table></div>
          <p className="note">Selected {multidocBenchmark.selected_candidate}: {multidocBenchmark.selection_reason}</p>
          {multidocBenchmark.holdout ? <div className="tableWrap"><table><thead><tr><th>New holdout</th><th>Required recall</th><th>Required precision</th><th>All evidence</th><th>Context loss</th><th>False abstention</th><th>Unsupported</th></tr></thead><tbody>
            {Object.entries(multidocBenchmark.holdout).map(([candidate, run]) => <tr key={candidate}><td>{candidate}</td><td>{metric(run.aggregate_metrics.required_evidence_recall)}</td><td>{metric(run.aggregate_metrics.required_evidence_precision)}</td><td>{metric(run.aggregate_metrics.all_required_evidence_coverage_rate)}</td><td>{metric(run.aggregate_metrics.supporting_context_loss_rate)}</td><td>{metric(run.aggregate_metrics.multi_document_false_abstention_rate)}</td><td>{metric(run.aggregate_metrics.unsupported_answer_count)}</td></tr>)}
          </tbody></table></div> : null}
          {multidocBenchmark.calibration.D.cases.some((item) => item.answerability_result?.requirements?.length) ? <details className="failureCard"><summary><span>Requirement → chunk mappings</span><strong>Evidence Coverage v2</strong></summary>
            {multidocBenchmark.calibration.D.cases.filter((item) => item.answerability_result?.requirements?.length).map((item) => <div key={item.id}><p><b>{item.case_id}</b> — {item.question}</p><ul>{item.answerability_result?.requirements?.map((requirement) => <li key={requirement.requirement_id}>{requirement.description} · {requirement.status} → {requirement.supporting_chunk_ids.join(", ") || "none"}</li>)}</ul></div>)}
          </details> : null}
        </> : <p className="note">The new multi-document benchmark has not been initialized.</p>}
      </section>
      <section className="sectionBlock"><h2>Dense vs BM25 vs Hybrid retrieval</h2>
        {retrievalBenchmark?.calibration ? <>
          <div className="tableWrap"><table><thead><tr><th>Calibration</th><th>All required @5</th><th>Required recall @5</th><th>Recall @5</th><th>MRR</th><th>nDCG</th><th>2-doc</th><th>3-doc</th><th>Exact ID</th><th>Total latency</th></tr></thead><tbody>
            {Object.entries(retrievalBenchmark.calibration).map(([mode, run]) => <tr key={mode}><td>{mode}{mode === retrievalBenchmark.selected_retrieval_mode ? " · selected" : ""}</td><td>{metric(run.metrics.all_required_evidence_coverage_at_5)}</td><td>{metric(run.metrics.required_evidence_recall_at_5)}</td><td>{metric(run.metrics.recall_at_5)}</td><td>{metric(run.metrics.mrr)}</td><td>{metric(run.metrics.ndcg_at_5)}</td><td>{metric(run.metrics.two_document_coverage_at_5)}</td><td>{metric(run.metrics.three_document_coverage_at_5)}</td><td>{metric(run.metrics.exact_identifier_recall_at_5)}</td><td>{metric(run.latency[mode === "DENSE" ? "dense_total_ms" : mode === "LEXICAL_BM25" ? "bm25_total_ms" : "hybrid_total_ms"])} ms</td></tr>)}
          </tbody></table></div>
          <p className="note">Selected {retrievalBenchmark.selected_retrieval_mode ?? "pending"}: {retrievalBenchmark.selection_reason ?? "calibration pending"}</p>
          {retrievalBenchmark.calibration.HYBRID_RRF ? <p className="note">BM25 rescues {String(retrievalBenchmark.calibration.HYBRID_RRF.branch_contribution.bm25_rescued_required_evidence_missed_by_dense ?? 0)} required evidence items · Dense rescues {String(retrievalBenchmark.calibration.HYBRID_RRF.branch_contribution.dense_rescued_required_evidence_missed_by_bm25 ?? 0)} · found by both {String(retrievalBenchmark.calibration.HYBRID_RRF.branch_contribution.found_by_both_required_evidence ?? 0)}</p> : null}
          {retrievalBenchmark.holdout ? <div className="tableWrap"><table><thead><tr><th>Holdout</th><th>All required @5</th><th>Required recall @5</th><th>Recall @5</th><th>MRR</th><th>nDCG</th><th>2-doc</th><th>3-doc</th><th>Exact ID</th></tr></thead><tbody>
            {Object.entries(retrievalBenchmark.holdout).map(([mode, run]) => <tr key={mode}><td>{mode}</td><td>{metric(run.metrics.all_required_evidence_coverage_at_5)}</td><td>{metric(run.metrics.required_evidence_recall_at_5)}</td><td>{metric(run.metrics.recall_at_5)}</td><td>{metric(run.metrics.mrr)}</td><td>{metric(run.metrics.ndcg_at_5)}</td><td>{metric(run.metrics.two_document_coverage_at_5)}</td><td>{metric(run.metrics.three_document_coverage_at_5)}</td><td>{metric(run.metrics.exact_identifier_recall_at_5)}</td></tr>)}
          </tbody></table></div> : null}
        </> : <p className="note">The retrieval-only calibration has not been executed.</p>}
      </section>
      <section className="sectionBlock"><h2>Dense Top-5 vs Cross-Encoder reranking</h2>
        {rerankingBenchmark?.calibration ? <>
          <div className="tableWrap"><table><thead><tr><th>Calibration</th><th>All required @5</th><th>Required recall @5</th><th>3-doc @5</th><th>Pool all required @20</th><th>Rerankable all required @5</th><th>Promoted</th><th>Demoted</th><th>Total latency</th></tr></thead><tbody>
            {Object.entries(rerankingBenchmark.calibration).map(([mode, run]) => <tr key={mode}><td>{mode}{mode === rerankingBenchmark.selected_mode ? " · selected" : ""}</td><td>{metric(run.metrics.all_required_evidence_coverage_at_5)}</td><td>{metric(run.metrics.required_evidence_recall_at_5)}</td><td>{metric(run.metrics.three_document_coverage_at_5)}</td><td>{metric(run.candidate_pool_metrics.all_required_evidence_coverage_at_20)}</td><td>{metric(run.rerankable_subset_metrics.all_required_evidence_coverage_at_5)}</td><td>{run.movement_metrics.required_evidence_promoted_into_top5 ?? 0}</td><td>{run.movement_metrics.required_evidence_demoted_out_of_top5 ?? 0}</td><td>{metric(run.latency[mode === "DENSE" ? "dense_total_ms" : "reranked_total_ms"])} ms</td></tr>)}
          </tbody></table></div>
          <p className="note">Selected {rerankingBenchmark.selected_mode ?? "pending"}: {rerankingBenchmark.selection_reason ?? "calibration pending"}</p>
          {rerankingBenchmark.holdout ? <div className="tableWrap"><table><thead><tr><th>Holdout</th><th>All required @5</th><th>Required recall @5</th><th>Recall @5</th><th>MRR</th><th>nDCG</th><th>2-doc</th><th>3-doc</th><th>Exact ID</th></tr></thead><tbody>
            {Object.entries(rerankingBenchmark.holdout).map(([mode, run]) => <tr key={mode}><td>{mode}</td><td>{metric(run.metrics.all_required_evidence_coverage_at_5)}</td><td>{metric(run.metrics.required_evidence_recall_at_5)}</td><td>{metric(run.metrics.recall_at_5)}</td><td>{metric(run.metrics.mrr)}</td><td>{metric(run.metrics.ndcg_at_5)}</td><td>{metric(run.metrics.two_document_coverage_at_5)}</td><td>{metric(run.metrics.three_document_coverage_at_5)}</td><td>{metric(run.metrics.exact_identifier_recall_at_5)}</td></tr>)}
          </tbody></table></div> : null}
        </> : <p className="note">The local Cross-Encoder calibration has not been executed.</p>}
      </section>
      <section className="sectionBlock"><h2>Dense+CE vs Hybrid+CE</h2>
        {hybridReranker?.calibration ? <>
          <div className="tableWrap"><table><thead><tr><th>Calibration</th><th>Pool coverage</th><th>All required @5</th><th>Required recall @5</th><th>3-doc @5</th><th>Exact ID</th><th>Pairs/query</th><th>Rerank mean</th></tr></thead><tbody>
            {Object.entries(hybridReranker.calibration).map(([mode, run]) => <tr key={mode}><td>{mode}{mode === hybridReranker.selected_mode ? " · selected" : ""}</td><td>{metric(run.branch_contribution?.candidate_pool?.all_required_evidence_coverage)}</td><td>{metric(run.metrics.all_required_evidence_coverage_at_5)}</td><td>{metric(run.metrics.required_evidence_recall_at_5)}</td><td>{metric(run.metrics.three_document_coverage_at_5)}</td><td>{metric(run.metrics.exact_identifier_recall_at_5)}</td><td>{metric(run.latency.pairs_per_query)}</td><td>{metric(run.latency.reranker_mean_ms)} ms</td></tr>)}
          </tbody></table></div>
          <p className="note">Selected {hybridReranker.selected_mode ?? "pending"}: {hybridReranker.selection_reason ?? "calibration pending"} · Sol was not used for retrieval configuration selection.</p>
          {hybridReranker.calibration_metrics?.lexical_rescues ? <p className="note">BM25-only required evidence {String(hybridReranker.calibration_metrics.lexical_rescues.bm25_only_required_evidence ?? 0)} · retained by RRF {String(hybridReranker.calibration_metrics.lexical_rescues.retained_by_rrf ?? 0)} · sent to Cross-Encoder {String(hybridReranker.calibration_metrics.lexical_rescues.sent_to_cross_encoder ?? 0)} · promoted to Top-5 {String(hybridReranker.calibration_metrics.lexical_rescues.promoted_to_top5 ?? 0)} · lost {String(hybridReranker.calibration_metrics.lexical_rescues.lost ?? 0)} · CE conversion {metric(hybridReranker.calibration_metrics.cross_encoder_conversion?.conversion_rate as number | undefined)}</p> : null}
          {hybridReranker.holdout ? <div className="tableWrap"><table><thead><tr><th>Holdout retrieval</th><th>All required @5</th><th>Required recall @5</th><th>3-doc @5</th><th>Exact ID</th><th>Version</th><th>ACL</th></tr></thead><tbody>
            {Object.entries(hybridReranker.holdout).map(([mode, run]) => <tr key={mode}><td>{mode}</td><td>{metric(run.metrics.all_required_evidence_coverage_at_5)}</td><td>{metric(run.metrics.required_evidence_recall_at_5)}</td><td>{metric(run.metrics.three_document_coverage_at_5)}</td><td>{metric(run.metrics.exact_identifier_recall_at_5)}</td><td>{metric(run.metrics.version_correctness)}</td><td>{metric(run.metrics.acl_safety)}</td></tr>)}
          </tbody></table></div> : null}
          {hybridReranker.end_to_end ? <div className="tableWrap"><table><thead><tr><th>Holdout Sol</th><th>Correct</th><th>Correct abstain</th><th>Incorrect abstain</th><th>Unsupported</th><th>F1</th><th>Total p50</th><th>Sol calls</th></tr></thead><tbody>
            {Object.entries(hybridReranker.end_to_end).map(([mode, run]) => <tr key={mode}><td>{mode}</td><td>{metric(run.metrics.correct_answer_count)}</td><td>{metric(run.metrics.correct_abstention_count)}</td><td>{metric(run.metrics.incorrect_abstention_count)}</td><td>{metric(run.metrics.unsupported_answer_count)}</td><td>{metric(run.metrics.answerability_f1)}</td><td>{metric(run.latency.total_p50_ms)} ms</td><td>{run.usage.new_sol_judge_calls ?? 0}</td></tr>)}
          </tbody></table></div> : null}
          {hybridReranker.conversion_analysis ? <p className="note">Hybrid retrieval rescues {String(hybridReranker.conversion_analysis.hybrid_retrieval_rescues ?? 0)} · converted to correct answers {String(hybridReranker.conversion_analysis.rescues_to_correct_answer ?? 0)} · remaining abstentions {String(hybridReranker.conversion_analysis.rescues_to_incorrect_abstention ?? 0)} · production retriever {hybridReranker.production_retriever_status}</p> : null}
        </> : <p className="note">The Dense+CE vs Hybrid+CE calibration has not been executed.</p>}
      </section>
      <section className="sectionBlock"><h2>Final Hybrid+Cross-Encoder replication</h2>
        {hybridReplication?.retrieval || hybridReplication?.replication_metrics ? <>
          <div className="tableWrap"><table><thead><tr><th>Replication</th><th>Pool coverage</th><th>All required @5</th><th>Required recall @5</th><th>3-doc @5</th><th>Exact ID</th><th>Pairs/query</th><th>Rerank mean</th></tr></thead><tbody>
            {Object.entries(hybridReplication.retrieval ?? hybridReplication.calibration ?? {}).map(([mode, run]) => <tr key={mode}><td>{mode}{mode === hybridReplication.selected_mode ? " · selected" : ""}</td><td>{metric(run.branch_contribution?.candidate_pool?.all_required_evidence_coverage)}</td><td>{metric(run.metrics.all_required_evidence_coverage_at_5)}</td><td>{metric(run.metrics.required_evidence_recall_at_5)}</td><td>{metric(run.metrics.three_document_coverage_at_5)}</td><td>{metric(run.metrics.exact_identifier_recall_at_5)}</td><td>{metric(run.latency.pairs_per_query)}</td><td>{metric(run.latency.reranker_mean_ms)} ms</td></tr>)}
          </tbody></table></div>
          <p className="note">Selected {hybridReplication.selected_mode ?? "pending"}: {hybridReplication.selection_reason ?? "replication pending"} · frozen v1 retriever {architecture?.selected_retriever ?? hybridReplication.production_retriever_status ?? "pending"}</p>
          {hybridReplication.end_to_end ? <div className="tableWrap"><table><thead><tr><th>Replication Sol</th><th>Correct</th><th>Correct abstain</th><th>Incorrect abstain</th><th>Unsupported</th><th>F1</th><th>Live judge p50</th><th>Cached replay</th><th>Sol calls</th></tr></thead><tbody>
            {Object.entries(hybridReplication.end_to_end).map(([mode, run]) => <tr key={mode}><td>{mode}</td><td>{metric(run.metrics.correct_answer_count)}</td><td>{metric(run.metrics.correct_abstention_count)}</td><td>{metric(run.metrics.incorrect_abstention_count)}</td><td>{metric(run.metrics.unsupported_answer_count)}</td><td>{metric(run.metrics.answerability_f1)}</td><td>{metric(run.latency.live_judge_p50_ms ?? run.latency.judge_mean_ms)} ms</td><td>{metric(run.latency.cached_replay_judge_count)}</td><td>{run.usage.new_sol_judge_calls ?? 0}</td></tr>)}
          </tbody></table></div> : null}
          {hybridReplication.conversion_analysis ? <p className="note">Hybrid retrieval rescues {String(hybridReplication.conversion_analysis.hybrid_retrieval_rescues ?? 0)} · converted to correct answers {String(hybridReplication.conversion_analysis.rescues_to_correct_answer ?? 0)} · remaining abstentions {String(hybridReplication.conversion_analysis.rescues_to_incorrect_abstention ?? 0)} · live Sol latency is reported separately from cached replay</p> : null}
        </> : <p className="note">The final Dense-vs-Hybrid replication has not been executed.</p>}
      </section>
      <section className="sectionBlock"><h2>Enterprise RAG Workbench v1 — Final frozen end-to-end benchmark</h2>
        {finalV1?.end_to_end ? <>
          <p className="note">Frozen retriever {finalV1.final_v1_retriever ?? finalV1.production_retriever_status ?? "pending"} · judge {finalV1.production_judge_status} · dataset {finalV1.dataset_id} · overlap {metric(finalV1.maximum_prior_overlap)}</p>
          <div className="tableWrap"><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
            <tr><td>Correct answers</td><td>{metric(finalV1.end_to_end.metrics.correct_answer_count)}</td></tr>
            <tr><td>Correct abstentions</td><td>{metric(finalV1.end_to_end.metrics.correct_abstention_count)}</td></tr>
            <tr><td>Incorrect abstentions</td><td>{metric(finalV1.end_to_end.metrics.incorrect_abstention_count)}</td></tr>
            <tr><td>Unsupported answers</td><td>{metric(finalV1.end_to_end.metrics.unsupported_answer_count)}</td></tr>
            <tr><td>Answerability F1</td><td>{metric(finalV1.end_to_end.metrics.answerability_f1)}</td></tr>
            <tr><td>All required @5</td><td>{metric(finalV1.retrieval?.metrics.all_required_evidence_coverage_at_5 ?? finalV1.end_to_end.retrieval_metrics.all_required_evidence_coverage_at_5)}</td></tr>
            <tr><td>Live total p50</td><td>{latencyP50(finalV1.end_to_end.latency.total, typeof finalV1.end_to_end.latency.total_p50_ms === "number" ? finalV1.end_to_end.latency.total_p50_ms : null)} ms</td></tr>
          </tbody></table></div>
          <p className="note">Release architecture {finalV1.release_architecture?.architecture_id ?? "pending"} · frozen {String(finalV1.release_architecture?.immutable ?? false)}</p>
        </> : <p className="note">The final frozen v1 benchmark is {finalV1?.prepared ? "prepared and awaiting execution" : "not yet executed"}.</p>}
      </section>
      <section className="sectionBlock"><h2>Enterprise RAG Workbench v2 — research baseline</h2>
        {v2Research?.initialized ? <>
          <p className="note">{v2Research.architecture_id} · parent {v2Research.parent_architecture} · research {v2Research.research_status} · production {String(v2Research.production_status)} · diagnosis-only {String(v2Research.diagnosis_only)}</p>
          <div className="tableWrap"><table><thead><tr><th>Census</th><th>Count</th></tr></thead><tbody>
            {Object.entries(v2Research.failure_census?.counts ?? {}).map(([name, count]) => <tr key={name}><td>{name}</td><td>{count}</td></tr>)}
          </tbody></table></div>
          <p className="note">Failed answerable {v2Research.failure_census?.failed_answerable_case_count ?? 0} · ranking misses {v2Research.ranking_diagnostic?.pool_complete_top5_incomplete_count ?? 0} · appearance {v2Research.ranking_diagnostic?.primary_appearance ?? "pending"} · Sol FN {v2Research.judge_false_negative_diagnostic?.retrieval_complete_sol_false_negative_count ?? 0}</p>
          <p className="note">Primary bottleneck {v2Research.primary_bottleneck} · {v2Research.recommended_ranking_intervention}</p>
        </> : <p className="note">The v2 research identity has not been initialized. Frozen v1 remains the production architecture.</p>}
      </section>
      <section className="sectionBlock"><h2>V2 Phase 1 — Document-Diversified Top-5 Ranking</h2>
        {(v2Phase1 ?? v2Research?.phase1)?.completed ? <>
          <p className="note">{(v2Phase1 ?? v2Research?.phase1)?.architecture_id} · dataset {(v2Phase1 ?? v2Research?.phase1)?.dataset_id} · production {String((v2Phase1 ?? v2Research?.phase1)?.production_status)} · selected {(v2Phase1 ?? v2Research?.phase1)?.selected_ranking ?? (v2Phase1 ?? v2Research?.phase1)?.selected_v2_ranking}</p>
          <div className="tableWrap"><table><thead><tr><th>Ranking</th><th>Required recall @5</th><th>All required @5</th><th>3-doc @5</th><th>2-doc @5</th></tr></thead><tbody>
            <tr><td>POINTWISE_CROSS_ENCODER_TOP5</td><td>{metric((v2Phase1 ?? v2Research?.phase1)?.control_metrics?.metrics?.required_evidence_recall_at_5)}</td><td>{metric((v2Phase1 ?? v2Research?.phase1)?.control_metrics?.metrics?.all_required_evidence_coverage_at_5)}</td><td>{metric((v2Phase1 ?? v2Research?.phase1)?.control_metrics?.metrics?.three_document_coverage_at_5)}</td><td>{metric((v2Phase1 ?? v2Research?.phase1)?.control_metrics?.metrics?.two_document_coverage_at_5)}</td></tr>
            <tr><td>DOCUMENT_DIVERSIFIED_TOP5</td><td>{metric((v2Phase1 ?? v2Research?.phase1)?.candidate_metrics?.metrics?.required_evidence_recall_at_5)}</td><td>{metric((v2Phase1 ?? v2Research?.phase1)?.candidate_metrics?.metrics?.all_required_evidence_coverage_at_5)}</td><td>{metric((v2Phase1 ?? v2Research?.phase1)?.candidate_metrics?.metrics?.three_document_coverage_at_5)}</td><td>{metric((v2Phase1 ?? v2Research?.phase1)?.candidate_metrics?.metrics?.two_document_coverage_at_5)}</td></tr>
          </tbody></table></div>
          <p className="note">Crowding rescues {(v2Phase1 ?? v2Research?.phase1)?.crowding_rescues?.total ?? 0} · regressions {(v2Phase1 ?? v2Research?.phase1)?.diversification_regressions?.count ?? 0} · remaining bottleneck {(v2Phase1 ?? v2Research?.phase1)?.primary_remaining_bottleneck} · {(v2Phase1 ?? v2Research?.phase1)?.selection?.reason}</p>
        </> : <p className="note">The Phase 1 document-diversity ranking experiment has not been completed. Frozen v1 remains unchanged.</p>}
      </section>
      <section className="sectionBlock"><h2>V2 Phase 2 — Soft Document-Cap Top-5 Ranking</h2>
        {(v2Phase2 ?? v2Research?.phase2)?.completed ? <>
          <p className="note">{(v2Phase2 ?? v2Research?.phase2)?.architecture_id} · dataset {(v2Phase2 ?? v2Research?.phase2)?.dataset_id} · production {String((v2Phase2 ?? v2Research?.phase2)?.production_status)} · selected {(v2Phase2 ?? v2Research?.phase2)?.selected_ranking ?? (v2Phase2 ?? v2Research?.phase2)?.selected_v2_ranking} · ranking {(v2Phase2 ?? v2Research?.phase2)?.ranking_research_status}</p>
          <div className="tableWrap"><table><thead><tr><th>Ranking</th><th>Required recall @5</th><th>All required @5</th><th>3-doc @5</th><th>2-doc @5</th></tr></thead><tbody>
            <tr><td>POINTWISE_CROSS_ENCODER_TOP5</td><td>{metric((v2Phase2 ?? v2Research?.phase2)?.control_metrics?.metrics?.required_evidence_recall_at_5)}</td><td>{metric((v2Phase2 ?? v2Research?.phase2)?.control_metrics?.metrics?.all_required_evidence_coverage_at_5)}</td><td>{metric((v2Phase2 ?? v2Research?.phase2)?.control_metrics?.metrics?.three_document_coverage_at_5)}</td><td>{metric((v2Phase2 ?? v2Research?.phase2)?.control_metrics?.metrics?.two_document_coverage_at_5)}</td></tr>
            <tr><td>MAX_2_CHUNKS_PER_DOCUMENT_TOP5</td><td>{metric((v2Phase2 ?? v2Research?.phase2)?.candidate_metrics?.metrics?.required_evidence_recall_at_5)}</td><td>{metric((v2Phase2 ?? v2Research?.phase2)?.candidate_metrics?.metrics?.all_required_evidence_coverage_at_5)}</td><td>{metric((v2Phase2 ?? v2Research?.phase2)?.candidate_metrics?.metrics?.three_document_coverage_at_5)}</td><td>{metric((v2Phase2 ?? v2Research?.phase2)?.candidate_metrics?.metrics?.two_document_coverage_at_5)}</td></tr>
          </tbody></table></div>
          <p className="note">Soft-cap rescues {(v2Phase2 ?? v2Research?.phase2)?.crowding_rescues?.total ?? 0} · regressions {(v2Phase2 ?? v2Research?.phase2)?.diversification_regressions?.count ?? 0} · slots freed {(v2Phase2 ?? v2Research?.phase2)?.occupancy_metrics?.slots_freed_by_max_2_cap_total ?? 0} · remaining bottleneck {(v2Phase2 ?? v2Research?.phase2)?.primary_remaining_bottleneck} · {(v2Phase2 ?? v2Research?.phase2)?.selection?.reason}</p>
        </> : <p className="note">The Phase 2 soft document-cap ranking experiment has not been completed. Frozen v1 remains unchanged.</p>}
      </section>
      <section className="sectionBlock"><h2>V2 Phase 3 — Evidence Sufficiency False-Negative Reduction</h2>
        {(v2Phase3 ?? v2Research?.phase3)?.completed ? <>
          <p className="note">{(v2Phase3 ?? v2Research?.phase3)?.architecture_id} · dataset {(v2Phase3 ?? v2Research?.phase3)?.dataset_id} · production {String((v2Phase3 ?? v2Research?.phase3)?.production_status)} · ranking {(v2Phase3 ?? v2Research?.phase3)?.selected_ranking ?? (v2Phase3 ?? v2Research?.phase3)?.selected_v2_ranking} · selected judge {(v2Phase3 ?? v2Research?.phase3)?.selected_judge ?? (v2Phase3 ?? v2Research?.phase3)?.selected_v2_judge} · judge {(v2Phase3 ?? v2Research?.phase3)?.judge_research_status}</p>
          <div className="tableWrap"><table><thead><tr><th>Judge</th><th>TP</th><th>FN</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>
            <tr><td>GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1</td><td>{metric((v2Phase3 ?? v2Research?.phase3)?.retrieval_complete_judge?.control?.tp)}</td><td>{metric((v2Phase3 ?? v2Research?.phase3)?.retrieval_complete_judge?.control?.fn)}</td><td>{metric((v2Phase3 ?? v2Research?.phase3)?.retrieval_complete_judge?.control?.precision)}</td><td>{metric((v2Phase3 ?? v2Research?.phase3)?.retrieval_complete_judge?.control?.recall)}</td><td>{metric((v2Phase3 ?? v2Research?.phase3)?.retrieval_complete_judge?.control?.f1)}</td></tr>
            <tr><td>GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V2</td><td>{metric((v2Phase3 ?? v2Research?.phase3)?.retrieval_complete_judge?.candidate?.tp)}</td><td>{metric((v2Phase3 ?? v2Research?.phase3)?.retrieval_complete_judge?.candidate?.fn)}</td><td>{metric((v2Phase3 ?? v2Research?.phase3)?.retrieval_complete_judge?.candidate?.precision)}</td><td>{metric((v2Phase3 ?? v2Research?.phase3)?.retrieval_complete_judge?.candidate?.recall)}</td><td>{metric((v2Phase3 ?? v2Research?.phase3)?.retrieval_complete_judge?.candidate?.f1)}</td></tr>
          </tbody></table></div>
          <p className="note">FN rescues {(v2Phase3 ?? v2Research?.phase3)?.false_negative_rescues?.count ?? 0} · FP regressions {(v2Phase3 ?? v2Research?.phase3)?.false_positive_regressions?.count ?? 0} · remaining bottleneck {(v2Phase3 ?? v2Research?.phase3)?.primary_remaining_bottleneck} · {(v2Phase3 ?? v2Research?.phase3)?.selection?.reason}</p>
        </> : <p className="note">The Phase 3 evidence-sufficiency false-negative experiment has not been completed. Frozen v1 remains unchanged.</p>}
      </section>
      <section className="sectionBlock"><h2>V2 Phase 4 — Generation and Provider Reliability Hardening</h2>
        {(v2Phase4 ?? v2Research?.phase4)?.completed ? <>
          <p className="note">{(v2Phase4 ?? v2Research?.phase4)?.architecture_id} · ranking {(v2Phase4 ?? v2Research?.phase4)?.selected_v2_ranking} · judge {(v2Phase4 ?? v2Research?.phase4)?.selected_v2_judge} · generator root cause {(v2Phase4 ?? v2Research?.phase4)?.generator_root_cause} · provider {(v2Phase4 ?? v2Research?.phase4)?.provider_failure?.classification} · status {(v2Phase4 ?? v2Research?.phase4)?.reliability_status}</p>
          <div className="tableWrap"><table><thead><tr><th>Reliability surface</th><th>Value</th></tr></thead><tbody>
            <tr><td>Max transport attempts</td><td>{metric((v2Phase4 ?? v2Research?.phase4)?.transport_retry_policy?.max_total_attempts)}</td></tr>
            <tr><td>New embedding calls</td><td>{metric((v2Phase4 ?? v2Research?.phase4)?.usage?.new_embedding_calls)}</td></tr>
            <tr><td>New Sol calls</td><td>{metric((v2Phase4 ?? v2Research?.phase4)?.usage?.new_sol_calls)}</td></tr>
            <tr><td>New external reranker calls</td><td>{metric((v2Phase4 ?? v2Research?.phase4)?.usage?.new_external_reranker_calls)}</td></tr>
          </tbody></table></div>
          <p className="note">Not a quality dataset. Historical Sol calls were not retried. Frozen v1 remains unchanged.</p>
        </> : <p className="note">The Phase 4 generation and provider reliability hardening has not been completed. Frozen v1 remains unchanged.</p>}
      </section>
      <section className="sectionBlock"><h2>Enterprise RAG Workbench v2 — Final frozen end-to-end benchmark</h2>
        {v2Final?.completed ? <>
          <p className="note">{v2Final.architecture_id} · parent {v2Final.parent_architecture} · dataset {v2Final.dataset_id} · ranking {v2Final.selected_v2_ranking} · judge {v2Final.selected_v2_judge} · one-shot {String(v2Final.final_v2_benchmark_one_shot)} · overlap {metric(v2Final.maximum_prior_overlap)}</p>
          <div className="tableWrap"><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
            <tr><td>Correct answers</td><td>{metric((v2Final.end_to_end as Record<string, number> | undefined)?.correct_answers)}</td></tr>
            <tr><td>Correct abstentions</td><td>{metric((v2Final.end_to_end as Record<string, number> | undefined)?.correct_abstentions)}</td></tr>
            <tr><td>Incorrect abstentions</td><td>{metric((v2Final.end_to_end as Record<string, number> | undefined)?.incorrect_abstentions)}</td></tr>
            <tr><td>Unsupported answers</td><td>{metric((v2Final.end_to_end as Record<string, number> | undefined)?.unsupported_answers)}</td></tr>
            <tr><td>Accuracy</td><td>{metric((v2Final.end_to_end as Record<string, number> | undefined)?.accuracy)}</td></tr>
            <tr><td>All required @5</td><td>{metric(v2Final.retrieval?.metrics?.all_required_evidence_coverage_at_5)}</td></tr>
            <tr><td>Three-document @5</td><td>{metric(v2Final.retrieval?.metrics?.three_document_coverage_at_5)}</td></tr>
          </tbody></table></div>
          <p className="note">Funnel answerable {v2Final.stage_funnel?.answerable ?? 0} → candidate {v2Final.stage_funnel?.candidate_complete ?? 0} → Top-5 {v2Final.stage_funnel?.top5_complete ?? 0} → Judge {v2Final.stage_funnel?.judge_approved ?? 0} → generator {v2Final.stage_funnel?.generator_succeeded ?? 0} → correct {v2Final.stage_funnel?.correct_final ?? 0} · remaining bottleneck {v2Final.primary_remaining_bottleneck}</p>
          <p className="note">Descriptive V1 vs V2 comparison only. The datasets differ and this is not a controlled paired A/B experiment.</p>
        </> : <p className="note">The final frozen v2 benchmark is {v2Final?.initialized ? "initialized and awaiting or executing" : "not yet initialized"}. Frozen v1 remains unchanged.</p>}
      </section>
      <section className="sectionBlock"><h2>V2 Quality Research — Controlled A/B Experiments</h2>
        {v2QualityAb?.completed ? <>
          <p className="note">{v2QualityAb.architecture_id} · ranking {v2QualityAb.selected_v2_ranking} · judge {v2QualityAb.selected_v2_judge} · candidate {v2QualityAb.final_candidate?.architecture} · git {v2QualityAb.git_commit}</p>
          <div className="tableWrap"><table><thead><tr><th>Experiment</th><th>Verdict</th></tr></thead><tbody>
            {Object.entries(v2QualityAb.verdicts ?? {}).map(([name, item]) => <tr key={name}><td>{name}</td><td>{item.verdict}</td></tr>)}
          </tbody></table></div>
          <p className="note">Holdout coverage delta {metric(v2QualityAb.holdout?.delta_coverage)} · Recall@20 delta {metric(v2QualityAb.holdout?.delta_recall_at_20)} · accepted {(v2QualityAb.final_candidate?.accepted_changes ?? []).join(", ") || "none"} · new embedding calls {String(v2QualityAb.usage?.new_embedding_calls ?? 0)} · new judge calls {String(v2QualityAb.usage?.new_judge_calls ?? 0)}</p>
          <p className="note">Frozen v1 remains unchanged. Frozen v2 ranking/Judge were not replaced.</p>
        </> : <p className="note">The v2 quality A/B research cycle is {v2QualityAb?.initialized ? "initialized" : "not yet completed"}. Frozen v1 remains unchanged.</p>}
      </section>
      <section className="sectionBlock"><h2>Enterprise RAG Workbench v3 Research — Phase 1 Generate-Then-Verify Recovery</h2>
        {v3Phase1 && (v3Phase1.completed || v3Phase1.diagnostic) ? <>
          <p className="note">{v3Phase1.architecture_id} · parent {v3Phase1.parent_architecture} · production {String(v3Phase1.production_status)} · GO/NO-GO {v3Phase1.go_nogo ?? "pending"} · selected {v3Phase1.selected_strategy ?? "pending"}</p>
          <div className="tableWrap"><table><thead><tr><th>Metric</th><th>Control A</th><th>Candidate B</th></tr></thead><tbody>
            <tr><td>Correct answers</td><td>{metric(v3Phase1.control_metrics?.correct_answers)}</td><td>{metric(v3Phase1.candidate_metrics?.correct_answers)}</td></tr>
            <tr><td>Incorrect abstentions</td><td>{metric(v3Phase1.control_metrics?.incorrect_abstentions)}</td><td>{metric(v3Phase1.candidate_metrics?.incorrect_abstentions)}</td></tr>
            <tr><td>Unsupported answers</td><td>{metric(v3Phase1.control_metrics?.unsupported_answers)}</td><td>{metric(v3Phase1.candidate_metrics?.unsupported_answers)}</td></tr>
            <tr><td>Recall</td><td>{metric(v3Phase1.control_metrics?.recall)}</td><td>{metric(v3Phase1.candidate_metrics?.recall)}</td></tr>
          </tbody></table></div>
          <p className="note">Diagnostic FN {v3Phase1.diagnostic?.historical_fn_count ?? 0} · rescues {v3Phase1.diagnostic?.rescue_count ?? 0} · false positives {v3Phase1.diagnostic?.false_positive_count ?? 0} · recovery triggers {v3Phase1.recovery_funnel?.recovery_triggers ?? 0} · valid rescues {v3Phase1.recovery_funnel?.valid_rescues ?? 0} · remaining bottleneck {v3Phase1.primary_remaining_bottleneck ?? "pending"}</p>
          <p className="note">Official frozen v2 was not modified. This is v3 research only.</p>
        </> : <p className="note">The v3 generate-then-verify research cycle is {v3Phase1?.initialized ? "initialized" : "not yet initialized"}. Frozen v2 remains unchanged.</p>}
      </section>
      <section className="sectionBlock"><h2>Frozen Dense + C2 vs Reranked Dense + C2</h2>
        {rerankerEndToEnd?.results ? <>
          <div className="tableWrap"><table><thead><tr><th>Pipeline</th><th>Correct</th><th>Correct abstain</th><th>Incorrect abstain</th><th>Unsupported</th><th>Answerability F1</th><th>All required @5</th><th>3-doc correct</th><th>3-doc incorrect abstain</th><th>Total p50</th><th>Judge calls</th></tr></thead><tbody>
            {Object.entries(rerankerEndToEnd.results).map(([mode, run]) => <tr key={mode}><td>{mode}</td><td>{metric(run.metrics.correct_answer_count)}</td><td>{metric(run.metrics.correct_abstention_count)}</td><td>{metric(run.metrics.incorrect_abstention_count)}</td><td>{metric(run.metrics.unsupported_answer_count)}</td><td>{metric(run.metrics.answerability_f1)}</td><td>{metric(run.retrieval_metrics.all_required_evidence_coverage_at_5)}</td><td>{metric(run.category_metrics.multidoc_three?.correct_answer_count)}</td><td>{metric(run.category_metrics.multidoc_three?.incorrect_abstention_count)}</td><td>{metric(run.latency.total_p50_ms)} ms</td><td>{run.usage.new_luna_judge_calls ?? 0}</td></tr>)}
          </tbody></table></div>
          <p className="note">Production retriever status: {rerankerEndToEnd.production_retriever_status ?? "pending"} · retrieval rescues converted to answers: {String(rerankerEndToEnd.conversion_analysis?.retrieval_rescue_to_correct_answer_count ?? 0)} · new incorrect abstentions: {String(rerankerEndToEnd.regression_analysis?.new_incorrect_abstentions ?? 0)}</p>
          <p className="note">Paired transitions: {Object.entries(rerankerEndToEnd.paired_transitions ?? {}).map(([transition, count]) => `${transition} ${count}`).join(" · ")} · local Cross-Encoder pairs: {rerankerEndToEnd.results.DENSE_CROSS_ENCODER_RERANK_C2.usage.cross_encoder_pairs ?? 0}</p>
        </> : <p className="note">The sealed paired end-to-end benchmark is {rerankerEndToEnd?.prepared ? "prepared and awaiting execution" : "not yet prepared"}.</p>}
      </section>
      <section className="sectionBlock"><h2>Frozen Evidence Sufficiency v1 vs Evidence Coverage v2</h2>
        {judgeEndToEnd?.results ? <>
          <div className="tableWrap"><table><thead><tr><th>Judge</th><th>Correct</th><th>Incorrect abstain</th><th>Unsupported</th><th>Final F1</th><th>Complete judge F1</th><th>3-doc correct</th><th>Judge p50</th><th>Total p50</th><th>Calls</th><th>Output tokens</th></tr></thead><tbody>
            {Object.entries(judgeEndToEnd.results).map(([mode, run]) => <tr key={mode}><td>{mode}{mode === judgeEndToEnd.production_judge_status ? " · selected" : ""}</td><td>{metric(run.metrics.correct_answer_count)}</td><td>{metric(run.metrics.incorrect_abstention_count)}</td><td>{metric(run.metrics.unsupported_answer_count)}</td><td>{metric(run.metrics.answerability_f1)}</td><td>{metric((run.subset_metrics.judge_decision_retrieval_complete as Record<string, number>).f1)}</td><td>{metric(run.category_metrics.multidoc_three?.correct_answer_count)}</td><td>{metric(run.latency.judge_p50_ms)} ms</td><td>{metric(run.latency.total_p50_ms)} ms</td><td>{run.usage.new_luna_judge_calls ?? 0}</td><td>{run.usage.luna_output_tokens ?? 0}</td></tr>)}
          </tbody></table></div>
          <p className="note">False-negative rescues {judgeEndToEnd.analysis?.false_negative_rescues ?? 0} · net improvement {judgeEndToEnd.analysis?.net_false_negative_improvement ?? 0} · false-positive regressions {judgeEndToEnd.false_positive_regressions?.count ?? 0}</p>
          <p className="note">Transitions: {Object.entries(judgeEndToEnd.paired_transitions ?? {}).map(([transition, count]) => `${transition} ${count}`).join(" · ")} · v2 requirements/question {metric(judgeEndToEnd.analysis?.requirement_diagnostics?.mean_requirements_per_question)} · supported {metric(judgeEndToEnd.analysis?.requirement_diagnostics?.SUPPORTED)} · partial {metric(judgeEndToEnd.analysis?.requirement_diagnostics?.PARTIAL)} · missing {metric(judgeEndToEnd.analysis?.requirement_diagnostics?.MISSING)} · conflicting {metric(judgeEndToEnd.analysis?.requirement_diagnostics?.CONFLICTING)}</p>
        </> : <p className="note">The sealed frozen-judge comparison is {judgeEndToEnd?.prepared ? "prepared and awaiting execution" : "not yet prepared"}.</p>}
      </section>
      <section className="sectionBlock"><h2>GPT-5.6 Luna vs GPT-5.6 Sol evidence judge</h2>
        {solJudgeEndToEnd?.results ? <>
          <div className="tableWrap"><table><thead><tr><th>Judge</th><th>Correct</th><th>Incorrect abstain</th><th>Unsupported</th><th>Final F1</th><th>Complete judge F1</th><th>3-doc correct</th><th>Judge p50</th><th>Total p50</th><th>Calls</th><th>Tokens in/out</th><th>Official cost</th></tr></thead><tbody>
            {Object.entries(solJudgeEndToEnd.results).map(([mode, run]) => <tr key={mode}><td>{mode}{mode === solJudgeEndToEnd.production_judge_status ? " · selected" : ""}</td><td>{metric(run.metrics.correct_answer_count)}</td><td>{metric(run.metrics.incorrect_abstention_count)}</td><td>{metric(run.metrics.unsupported_answer_count)}</td><td>{metric(run.metrics.answerability_f1)}</td><td>{metric((run.subset_metrics.judge_decision_retrieval_complete as Record<string, number>).f1)}</td><td>{metric(run.category_metrics.multidoc_three?.correct_answer_count)}</td><td>{metric(run.latency.judge_p50_ms)} ms</td><td>{metric(run.latency.total_p50_ms)} ms</td><td>{(run.usage.new_luna_judge_calls ?? 0) + (run.usage.new_sol_judge_calls ?? 0)}</td><td>{(run.usage.luna_input_tokens ?? run.usage.sol_input_tokens ?? 0)} / {(run.usage.luna_output_tokens ?? run.usage.sol_output_tokens ?? 0)}</td><td>{metric(run.usage.official_measured_cost_usd)}</td></tr>)}
          </tbody></table></div>
          <p className="note">False-negative rescues {solJudgeEndToEnd.analysis?.false_negative_rescues ?? 0} · Luna correct → Sol FN {solJudgeEndToEnd.analysis?.luna_correct_to_sol_false_abstention ?? 0} · net improvement {solJudgeEndToEnd.analysis?.net_false_negative_improvement ?? 0} · false-positive regressions {solJudgeEndToEnd.false_positive_regressions?.count ?? 0}</p>
          <p className="note">Transitions: {Object.entries(solJudgeEndToEnd.paired_transitions ?? {}).map(([transition, count]) => `${transition} ${count}`).join(" · ")} · production retriever {solJudgeEndToEnd.production_retriever_status ?? "DENSE_CROSS_ENCODER_RERANK"}</p>
        </> : <p className="note">The sealed Luna vs Sol comparison is {solJudgeEndToEnd?.prepared ? "prepared and awaiting execution" : "not yet prepared"}.</p>}
      </section>
      {latest ? (
        <section className="metricGrid" aria-label="Latest experiment metrics">
          {["recall_at_k", "reciprocal_rank", "ndcg_at_k", "abstention_f1", "security_success_rate"].map((name) => (
            <article key={name}><span>{name.replaceAll("_", " ")}</span><strong>{metric(latest.aggregate_metrics[name])}</strong></article>
          ))}
        </section>
      ) : <p className="note">No persistent experiments yet.</p>}
      <section className="sectionBlock">
        <h2>Experiment list</h2>
        <div className="tableWrap"><table><thead><tr><th>Name</th><th>Status</th><th>Dataset</th><th>Recall</th><th>MRR</th><th>Latency</th></tr></thead>
          <tbody>{runs.map((run) => <tr key={run.id}>
            <td><Link className="textLink" href={`/evaluations/${run.id}`}>{run.name}</Link><small>{run.id}</small></td>
            <td>{run.status}</td><td>{run.config.evaluation_dataset_version}</td>
            <td>{metric(run.aggregate_metrics.recall_at_k)}</td><td>{metric(run.aggregate_metrics.reciprocal_rank)}</td>
            <td>{run.latency.total_ms == null ? "—" : `${run.latency.total_ms.toFixed(1)} ms`}</td>
          </tr>)}</tbody>
        </table></div>
      </section>
    </div>
  );
}
