"use client";

import { useCallback, useEffect, useState } from "react";

import { ErrorMessage } from "@/components/error-message";
import { PageHeading } from "@/components/page-heading";
import { apiRequest, ExperimentComparison, ExperimentSummary } from "@/lib/api";

function value(item: number | null) { return item == null ? "pending" : item.toFixed(3); }

export default function ExperimentComparisonPage() {
  const [runs, setRuns] = useState<ExperimentSummary[]>([]);
  const [baseline, setBaseline] = useState("");
  const [candidate, setCandidate] = useState("");
  const [comparison, setComparison] = useState<ExperimentComparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(async () => {
    try {
      const items = await apiRequest<ExperimentSummary[]>("/experiments"); setRuns(items);
      if (items[1]) setBaseline(items[1].id); if (items[0]) setCandidate(items[0].id);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Unknown request error"); }
  }, []);
  useEffect(() => { void load(); }, [load]);
  async function compare() {
    setError(null);
    try { setComparison(await apiRequest<ExperimentComparison>(`/experiments/compare?baseline_id=${encodeURIComponent(baseline)}&candidate_id=${encodeURIComponent(candidate)}`)); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Unknown request error"); }
  }
  return <div className="page">
    <PageHeading eyebrow="Experiment comparison" title="Configuration A vs B" description="Compare measured quality, safety, and latency deltas under documented project benchmark criteria." />
    <div className="compareControls"><label>Baseline<select value={baseline} onChange={(event) => setBaseline(event.target.value)}>{runs.map((run) => <option key={run.id} value={run.id}>{run.name}</option>)}</select></label><label>Candidate<select value={candidate} onChange={(event) => setCandidate(event.target.value)}>{runs.map((run) => <option key={run.id} value={run.id}>{run.name}</option>)}</select></label><button className="button primary" onClick={compare} disabled={!baseline || !candidate}>Compare</button></div>
    <ErrorMessage message={error} />
    {comparison ? <>
      <p className={`status ${comparison.passed ? "answered" : "abstained"}`}>{comparison.passed ? "No configured regressions" : `${comparison.regressions.length} regression(s)`}</p>
      <p className={`status ${comparison.hard_constraints.candidate_acceptable ? "answered" : "abstained"}`}>{comparison.hard_constraints.candidate_acceptable ? "Security constraints satisfied" : "Candidate is unacceptable"}</p>
      <p className="note">Candidate threshold: {String(comparison.candidate.config.retrieval.score_threshold)} · query cache: {comparison.candidate.cache.hits ?? 0} hits / {comparison.candidate.cache.misses ?? 0} misses</p>
      <section className="sectionBlock"><h2>Configuration differences</h2>
        {comparison.configuration_differences.length ? <div className="tableWrap"><table><thead><tr><th>Field</th><th>Baseline</th><th>Candidate</th></tr></thead><tbody>
          {comparison.configuration_differences.map((item) => <tr key={item.field}><td>{item.field}</td><td>{String(item.baseline)}</td><td>{String(item.candidate)}</td></tr>)}
        </tbody></table></div> : <p className="note">Configurations are identical.</p>}
      </section>
      <div className="tableWrap"><table><thead><tr><th>Metric</th><th>{comparison.baseline.name}</th><th>{comparison.candidate.name}</th><th>Delta</th></tr></thead><tbody>
        {Object.entries(comparison.metrics).map(([name, metric]) => <tr key={name}><td>{name.replaceAll("_", " ")}</td><td>{value(metric.baseline)}</td><td>{value(metric.candidate)}</td><td>{metric.delta == null ? "pending" : `${metric.delta >= 0 ? "+" : ""}${metric.delta.toFixed(3)}`}</td></tr>)}
        <tr><td>mean total latency (ms)</td><td>{value(comparison.baseline.total_latency_ms)}</td><td>{value(comparison.candidate.total_latency_ms)}</td><td>{value(comparison.latency_delta_ms)}</td></tr>
        <tr><td>query embedding latency (ms)</td><td>{value(comparison.baseline.latency.query_embedding_ms)}</td><td>{value(comparison.candidate.latency.query_embedding_ms)}</td><td>—</td></tr>
        <tr><td>vector search latency (ms)</td><td>{value(comparison.baseline.latency.vector_search_ms)}</td><td>{value(comparison.candidate.latency.vector_search_ms)}</td><td>—</td></tr>
      </tbody></table></div>
      <section className="sectionBlock"><h2>Category deltas</h2><div className="tableWrap"><table><thead><tr><th>Category</th><th>Recall Δ</th><th>MRR Δ</th><th>nDCG Δ</th><th>Abstention F1 Δ</th><th>Safety Δ</th></tr></thead><tbody>
        {Object.entries(comparison.category_metrics).map(([category, metrics]) => <tr key={category}><td>{category.replaceAll("_", " ")}</td>{["recall_at_k", "reciprocal_rank", "ndcg_at_k", "abstention_f1", "security_success_rate"].map((metric) => <td key={metric}>{value(metrics[metric]?.delta ?? null)}</td>)}</tr>)}
      </tbody></table></div></section>
      {comparison.regressions.map((item) => <p className="error" key={item.metric}>{item.security_critical ? "Security: " : ""}{item.metric}: {item.reason}</p>)}
      <p className="note">{comparison.threshold_policy}</p>
    </> : null}
  </div>;
}
