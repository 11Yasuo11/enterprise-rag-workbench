"use client";

import { FormEvent, useEffect, useState } from "react";

import { ErrorMessage } from "@/components/error-message";
import { PageHeading } from "@/components/page-heading";
import { apiRequest, RetrievalResult } from "@/lib/api";

export default function RetrievalPage() {
  const [query, setQuery] = useState("current severity-one reporting deadline");
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [savedTrace, setSavedTrace] = useState<Array<{ chunk_id: string; rank: number; score: number; included_in_context: boolean }>>([]);

  useEffect(() => {
    const runId = new URLSearchParams(window.location.search).get("run_id");
    if (!runId) return;
    void apiRequest<{ query: string; retrieval_results: Array<{ chunk_id: string; rank: number; score: number; included_in_context: boolean }> }>(`/runs/${runId}`)
      .then((run) => { setQuery(run.query); setSavedTrace(run.retrieval_results); })
      .catch((caught) => setError(caught instanceof Error ? caught.message : "Unknown request error"));
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setPending(true); setError(null);
    try {
      setResults(await apiRequest<RetrievalResult[]>("/retrieve", { query, top_k: 5, score_threshold: 0.2 }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unknown request error");
    } finally { setPending(false); }
  }

  return (
    <div className="page narrow">
      <PageHeading eyebrow="Developer mode" title="Retrieval Debug" description="Inspect rank, score, source metadata, and exact text before generation." />
      <form className="queryForm inline" onSubmit={submit}>
        <label htmlFor="retrieval-query">Query</label>
        <input id="retrieval-query" value={query} onChange={(event) => setQuery(event.target.value)} required />
        <button className="button primary" disabled={pending}>{pending ? "Searching…" : "Retrieve"}</button>
      </form>
      <ErrorMessage message={error} />
      {savedTrace.length ? <section className="sectionBlock"><h2>Saved experiment trace</h2><div className="tableWrap"><table><thead><tr><th>Rank</th><th>Score</th><th>Context</th><th>Chunk</th></tr></thead><tbody>{savedTrace.map((item) => <tr key={item.chunk_id}><td>{item.rank}</td><td>{item.score.toFixed(4)}</td><td>{item.included_in_context ? "included" : "excluded"}</td><td><code>{item.chunk_id}</code></td></tr>)}</tbody></table></div></section> : null}
      <section className="retrievalList" aria-live="polite">
        {results.map((result) => (
          <article className="retrievalCard" key={result.chunk_id}>
            <div className="rank">#{result.rank}</div>
            <div><div className="resultHeader"><h2>{result.title}</h2><strong>{result.score.toFixed(3)}</strong></div>
              <p className="meta">{result.document_id} · v{result.version} · {result.section ?? `page ${result.page ?? "n/a"}`}</p>
              <p>{result.text}</p><code>{result.chunk_id}</code></div>
          </article>
        ))}
      </section>
    </div>
  );
}
