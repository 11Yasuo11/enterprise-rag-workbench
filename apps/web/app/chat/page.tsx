"use client";

import { FormEvent, useState } from "react";

import { ErrorMessage } from "@/components/error-message";
import { PageHeading } from "@/components/page-heading";
import { apiRequest, Citation } from "@/lib/api";

type RagResponse = {
  request_id?: string;
  run_id: string;
  status: "answer" | "abstain" | "unavailable" | "answered" | "abstained";
  answer: string | null;
  citations: Citation[];
  route?: "deterministic" | "luna" | "sol" | "abstain";
  error_class?: string | null;
  requirements?: Array<{ requirement_id: string; requirement_text: string; status: string }>;
};

export default function ChatPage() {
  const [query, setQuery] = useState("What is the priority-one support escalation identifier?");
  const [result, setResult] = useState<RagResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPending(true);
    setError(null);
    try {
      setResult(await apiRequest<RagResponse>("/rag/query", { query }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unknown request error");
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="page narrow">
      <PageHeading eyebrow="Grounded answer" title="Chat" description="Answers are returned only when retrieved evidence can be cited." />
      <form className="queryForm" onSubmit={submit}>
        <label htmlFor="chat-query">Question</label>
        <textarea id="chat-query" value={query} onChange={(event) => setQuery(event.target.value)} rows={4} required />
        <button className="button primary" disabled={pending}>{pending ? "Running…" : "Ask"}</button>
      </form>
      <ErrorMessage message={error} />
      {result ? (
        <section className="resultPanel" aria-live="polite">
          <div className="resultHeader">
            <h2>Result</h2>
            <span className={`status ${result.status === "answer" || result.status === "answered" ? "answered" : "abstained"}`}>
              {result.status}
            </span>
          </div>
          <p className="answer">{result.answer ?? "The corpus does not contain sufficient evidence to answer."}</p>
          {result.route ? <small>Route {result.route}</small> : null}
          {result.citations.length > 0 ? <h3>Citations</h3> : null}
          {result.citations.map((citation) => (
            <article className="citation" key={citation.chunk_id}>
              <strong>{citation.title}</strong>
              <span>v{citation.version} · {citation.section ?? (citation.page ? `page ${citation.page}` : "document")}</span>
              <code>{citation.chunk_id}</code>
            </article>
          ))}
          <small>Run {result.run_id}</small>
        </section>
      ) : null}
    </div>
  );
}

