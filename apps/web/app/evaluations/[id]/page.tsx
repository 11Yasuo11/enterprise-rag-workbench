"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { ErrorMessage } from "@/components/error-message";
import { PageHeading } from "@/components/page-heading";
import { apiRequest, ExperimentDetail } from "@/lib/api";

function value(item: number | null | undefined) { return item == null ? "pending" : item.toFixed(3); }

export default function ExperimentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [run, setRun] = useState<ExperimentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    void apiRequest<ExperimentDetail>(`/experiments/${id}`).then(setRun).catch((caught) => setError(caught instanceof Error ? caught.message : "Unknown request error"));
  }, [id]);

  return <div className="page">
    <PageHeading eyebrow="Experiment detail" title={run?.name ?? "Loading experiment…"} description="Configuration, category metrics, latency, and retrieval-versus-generation failures for one measured run." />
    <ErrorMessage message={error} />
    {run ? <>
      <section className="metricGrid">
        {Object.entries(run.aggregate_metrics).slice(0, 10).map(([name, item]) => <article key={name}><span>{name.replaceAll("_", " ")}</span><strong>{value(item)}</strong></article>)}
      </section>
      <section className="splitGrid">
        <article className="panel"><h2>Configuration</h2><pre>{JSON.stringify(run.config, null, 2)}</pre></article>
        <article className="panel"><h2>Latency and usage</h2><p>Query embedding: {value(run.latency.query_embedding_ms)} ms</p><p>Vector search: {value(run.latency.vector_search_ms)} ms</p><p>ACL filter: {value(run.latency.acl_filter_ms)} ms</p><p>Gate cache: {value(run.latency.answerability_gate_cache_lookup_ms)} ms</p><p>Answerability judge: {value(run.latency.answerability_judge_ms)} ms</p><p>Context pruning: {value(run.latency.context_pruning_ms)} ms</p><p>Context construction: {value(run.latency.context_construction_ms)} ms</p><p>Generation: {value(run.latency.generation_ms)} ms</p><p>Total: {value(run.latency.total_ms)} ms</p><pre>{JSON.stringify(run.usage, null, 2)}</pre></article>
      </section>
      <section className="sectionBlock"><h2>Metrics by category</h2><div className="tableWrap"><table><thead><tr><th>Category</th><th>Recall</th><th>MRR</th><th>nDCG</th><th>Abstention F1</th><th>Safety</th></tr></thead>
        <tbody>{Object.entries(run.category_metrics).map(([category, metrics]) => <tr key={category}><td>{category}</td><td>{value(metrics.recall_at_k)}</td><td>{value(metrics.reciprocal_rank)}</td><td>{value(metrics.ndcg_at_k)}</td><td>{value(metrics.abstention_f1)}</td><td>{value(metrics.security_success_rate)}</td></tr>)}</tbody>
      </table></div></section>
      <section className="sectionBlock"><h2>Failure and judge evidence</h2>
        <p className="note">{Object.entries(run.failure_counts).map(([name, count]) => `${name}: ${count}`).join(" · ") || "No classified failures."}</p>
        {run.cases.filter((item) => item.failure_type || item.error || item.answerability_result).map((item) => <details className="failureCard" key={item.id}>
          <summary><span>{item.case_id}</span><strong>{item.failure_types.join(", ") || item.failure_type || "UNKNOWN"}</strong></summary>
          <p><b>Question:</b> {item.question}</p><p><b>Category:</b> {item.category}</p>
          <p><b>Expected documents:</b> {item.expected_document_ids.join(", ") || "none"}</p>
          <p><b>Forbidden documents:</b> {item.forbidden_document_ids.join(", ") || "none"}</p>
          <p><b>Principal:</b> {item.principal.principal_id} · {item.principal.permission_groups.join(", ")}</p>
          <p><b>Actual documents:</b> {item.retrieved_document_ids.join(", ") || "none"}</p>
          <p><b>Gate decision:</b> {item.answerability_result ? `${item.answerability_result.answerable ? "SUFFICIENT" : "INSUFFICIENT"} · ${item.answerability_result.reason_code}` : "baseline / no gate"}</p>
          <p><b>Judge operational error:</b> {item.answerability_operational_error ?? "none"}</p>
          <p><b>Supporting chunks:</b> {item.supporting_chunk_ids.join(", ") || "none"}</p>
          <p><b>Generation context:</b> {item.generation_context_chunk_ids.join(", ") || "none"}</p>
          <p><b>Expected answer:</b> {item.expected_answer ?? "abstain"}</p><p><b>Actual answer:</b> {item.answer ?? "abstained"}</p>
          <div className="tableWrap"><table><thead><tr><th>Rank</th><th>Document</th><th>Version</th><th>Score</th><th>Chunk</th></tr></thead><tbody>
            {item.retrieval_trace.map((trace) => <tr key={trace.chunk_id}><td>{trace.rank}</td><td>{trace.document_id}</td><td>{trace.version}</td><td>{trace.score.toFixed(4)}</td><td><code>{trace.chunk_id}</code></td></tr>)}
          </tbody></table></div>
          <pre>{JSON.stringify(item.citations, null, 2)}</pre>
          <pre>{JSON.stringify(item.failure_details, null, 2)}</pre>
          {item.rag_run_id ? <Link className="textLink" href={`/retrieval?run_id=${item.rag_run_id}`}>Open retrieval trace</Link> : null}
        </details>)}
      </section>
    </> : null}
  </div>;
}
