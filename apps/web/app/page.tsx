import Link from "next/link";

export default function HomePage() {
  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">Enterprise RAG evaluation workbench</p>
        <h1>Inspect the evidence chain, not just the answer.</h1>
        <p>
          A local evaluation workbench for hybrid retrieval, Cross-Encoder ranking,
          evidence-sufficiency gating, citations, safe abstention, and reproducible benchmarks.
        </p>
        <div className="actions">
          <Link className="button primary" href="/chat">Ask the corpus</Link>
          <Link className="button" href="/retrieval">Debug retrieval</Link>
        </div>
      </header>
      <section className="stageGrid" aria-label="RAG stages">
        {[
          ["01", "Ingestion", "Parse, clean, deduplicate, version, chunk, embed."],
          ["02", "Retrieval", "Authorize first, then rank inspectable chunks."],
          ["03", "Generation", "Budget context, delimit data, map real citations."],
          ["04", "Evaluation", "Measure retrieval and abstention independently."],
        ].map(([number, title, copy]) => (
          <article className="stage" key={number}>
            <span>{number}</span><h2>{title}</h2><p>{copy}</p>
          </article>
        ))}
      </section>
    </div>
  );
}

