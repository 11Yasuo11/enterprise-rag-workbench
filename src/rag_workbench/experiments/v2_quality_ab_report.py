# ruff: noqa: E501
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

QUALITY_AB_BENCHMARK_HEADING = "## V2 Quality Research — Controlled A/B Experiments"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _metric_block(metrics: dict[str, Any] | None) -> str:
    values = metrics or {}
    return (
        f"Recall@5 {_fmt(values.get('recall_at_5'))} · Recall@10 {_fmt(values.get('recall_at_10'))} · "
        f"Recall@20 {_fmt(values.get('recall_at_20'))} · Recall@50 {_fmt(values.get('recall_at_50'))} · "
        f"Top-5 coverage {_fmt(values.get('top5_evidence_coverage'))}"
    )


def render_quality_ab_markdown(status: dict[str, Any]) -> str:
    baseline = status.get("baseline") or {}
    experiments = status.get("experiments") or {}
    verdicts = status.get("verdicts") or {}
    holdout = status.get("holdout") or {}
    graph = (
        (status.get("failure_census") or {}).get("graph_rag") or experiments.get("graphrag") or {}
    )
    candidate = status.get("final_candidate") or {}
    usage = status.get("usage") or {}
    safety = status.get("safety") or {}
    end = baseline
    rewrite = verdicts.get("query_rewrite") or {}
    multi = verdicts.get("multi_query") or {}
    hyde = verdicts.get("hyde") or {}
    sparse = verdicts.get("learned_sparse") or {}
    multivec = verdicts.get("multi_vector") or {}
    decomp = verdicts.get("query_decomposition") or {}
    scale = experiments.get("corpus_scale") or []
    scale_lines = [
        (
            f"| {item.get('scale')} | {_fmt((item.get('metrics') or {}).get('recall_at_20'))} | "
            f"{_fmt((item.get('metrics') or {}).get('recall_at_50'))} | "
            f"{_fmt((item.get('metrics') or {}).get('top5_evidence_coverage'))} | "
            f"{item.get('retrieval_miss')} | "
            f"{_fmt(((item.get('latency') or {}).get('dense') or {}).get('p95_ms'))} | "
            f"{_fmt(((item.get('latency') or {}).get('lexical') or {}).get('p95_ms'))} | "
            f"{item.get('index_size_bytes')} |"
        )
        for item in scale
    ]
    return "\n".join(
        [
            QUALITY_AB_BENCHMARK_HEADING,
            "",
            "This section is research-only. It does not replace v1 or frozen official v2.",
            "Accepted quality changes: none. Keep current v2.",
            "Executed retrieval candidates were proxies unless labeled otherwise.",
            "Unexecuted paid/infrastructure methods are `NOT_EXECUTED`, not empirically rejected.",
            "No paid embedding, Sol, or external reranker calls were issued in this cycle.",
            "",
            "### Frozen v2 control",
            "",
            "```text",
            "Dense Search + BM25 → RRF → POINTWISE_CROSS_ENCODER_TOP5",
            "Judge GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1",
            "Generator deterministic-extractive-v1.1",
            "```",
            "",
            f"Architecture `{status.get('architecture_id')}`. Parent `{status.get('parent_architecture')}`.",
            f"Baseline config hash `{status.get('baseline_config_hash')}`. Dataset hash `{status.get('dataset_hash')}`.",
            f"Corpus hash `{status.get('corpus_hash')}`. git_commit `{status.get('git_commit')}`.",
            "",
            "### Phase 0 — Baseline reproduction",
            "",
            f"Correct answers {end.get('correct_answers')} · correct abstentions {end.get('correct_abstentions')} ·",
            f"incorrect abstentions {end.get('incorrect_abstentions')} · unsupported {end.get('unsupported_answers')}.",
            f"Precision {_fmt(end.get('precision'))} · Recall {_fmt(end.get('recall'))} · F1 {_fmt(end.get('f1'))}.",
            f"Citation validity {_fmt(end.get('citation_validity'))} · correctness {_fmt(end.get('citation_correctness'))}.",
            f"Isolated measurement: {_metric_block(end.get('isolated_metrics'))}.",
            f"Failure census: `{end.get('failure_census')}`.",
            "",
            "### Phase 1 — Corpus scale test",
            "",
            "All larger corpora are `SYNTHETIC / STRESS TEST` distractors. This is not production-corpus performance.",
            "V3 research topic: `RETRIEVAL_SCALABILITY_UNDER_HARD_NEGATIVES`.",
            "",
            "| Scale | Recall@20 | Recall@50 | Top-5 coverage | Retrieval miss | Dense p95 ms | BM25 p95 ms | Index bytes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *scale_lines,
            "",
            "### Phase 2 — Query rewrite / multi-query / HyDE",
            "",
            "Lexical rewrite: `REJECTED PROXY`. Multi-query BM25: `REJECTED PROXY`. Template HyDE: `REJECTED PROXY`.",
            "Hypothetical text is never evidence. LLM Query Rewrite: `NOT_EXECUTED`. Embedding-based HyDE: `NOT_EXECUTED`.",
            "",
            f"Raw control: {_metric_block((experiments.get('control') or {}).get('metrics'))}",
            f"Rewrite: {_metric_block((experiments.get('query_rewrite') or {}).get('metrics'))} · proxy verdict {rewrite.get('verdict')} ({rewrite.get('reason')})",
            f"Multi-query: {_metric_block((experiments.get('multi_query') or {}).get('metrics'))} · proxy verdict {multi.get('verdict')} ({multi.get('reason')})",
            f"Template HyDE: {_metric_block((experiments.get('hyde') or {}).get('metrics'))} · proxy verdict {hyde.get('verdict')} ({hyde.get('reason')})",
            "",
            "### Phase 3 — BM25 vs learned sparse",
            "",
            "Contextual Sparse Lite: `REJECTED PROXY`. 3-way sparse fusion: `REJECTED PROXY`. Neural SPLADE: `NOT_EXECUTED`.",
            "",
            f"Sparse proxy verdict: {sparse.get('verdict')} ({sparse.get('reason')})",
            "",
            "### Phase 4 — Multi-vector / late interaction",
            "",
            "Sentence MaxSim: `REJECTED PROXY`. Real ColBERT: `NOT_EXECUTED`.",
            "",
            f"MaxSim proxy verdict: {multivec.get('verdict')} ({multivec.get('reason')})",
            "",
            "### Phase 5 — Query decomposition",
            "",
            "Query decomposition: `REJECTED PROXY`. Applied only to two- and three-document questions.",
            "",
            f"Decomposition proxy verdict: {decomp.get('verdict')} ({decomp.get('reason')})",
            "",
            "### Phase 6 — GraphRAG necessity",
            "",
            "GraphRAG: `NOT NEEDED BY CURRENT FAILURE DATA`. Not a full GraphRAG implementation.",
            f"Persisted isolated census: {graph.get('decision')} — {graph.get('reason')}",
            "",
            "### Holdout",
            "",
            f"Split `{holdout.get('split')}`. Research n={holdout.get('research_cases')} · holdout n={holdout.get('holdout_cases')}.",
            f"Holdout control: {_metric_block(holdout.get('control'))}",
            f"Holdout candidate: {_metric_block(holdout.get('candidate'))}",
            f"Delta coverage {_fmt(holdout.get('delta_coverage'))} · delta Recall@20 {_fmt(holdout.get('delta_recall_at_20'))}",
            "",
            "### Decision",
            "",
            f"Accepted changes: {candidate.get('accepted_changes')}",
            f"Rejected proxy changes: {candidate.get('rejected_changes')}",
            "Not executed: LLM Query Rewrite, embedding-based HyDE, neural SPLADE, real ColBERT.",
            f"Final candidate architecture: {candidate.get('architecture')}",
            f"Safety: ACL {safety.get('acl_safety')} · tenant {safety.get('tenant_isolation')} · version {safety.get('version_correctness')} · unsupported delta {safety.get('unsupported_answers_delta')}",
            f"External usage: embedding {usage.get('new_embedding_calls')} · judge {usage.get('new_judge_calls')} · cache hits {usage.get('cache_hits')} · estimated cost {usage.get('estimated_cost')}",
            "",
            "Frozen Enterprise RAG Workbench v1 was not modified. Frozen v2 ranking and Judge identities were not replaced.",
        ]
    )


def persist_quality_ab_markdown(status: dict[str, Any], *, replace: bool = True) -> None:
    path = Path("BENCHMARK.md")
    text = path.read_text()
    count = text.count(QUALITY_AB_BENCHMARK_HEADING)
    section = render_quality_ab_markdown(status)
    if count > 1:
        raise ValueError("duplicate quality A/B heading")
    if count == 1:
        if not replace:
            return
        prefix = text.split(QUALITY_AB_BENCHMARK_HEADING, 1)[0]
        path.write_text(prefix.rstrip() + "\n\n" + section + "\n")
        return
    path.write_text(text.rstrip() + "\n\n" + section + "\n")


def write_experiment_artifacts(status: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(status, default=str))
    (directory / "summary.json").write_text(json.dumps(serializable, indent=2) + "\n")
    rows = [
        {
            "experiment_id": "phase0-baseline",
            "experiment_name": "baseline",
            **{
                key: (status.get("baseline") or {}).get(key)
                for key in (
                    "git_commit",
                    "baseline_config_hash",
                    "dataset_hash",
                    "corpus_hash",
                    "precision",
                    "recall",
                    "f1",
                    "correct_answers",
                    "incorrect_answers",
                    "correct_abstentions",
                    "incorrect_abstentions",
                    "unsupported_answers",
                    "citation_validity",
                    "citation_correctness",
                    "p50_latency",
                    "p95_latency",
                    "embedding_calls",
                    "judge_calls",
                    "reranker_calls",
                    "estimated_cost",
                    "verdict",
                )
            },
        }
    ]
    for name, verdict in (status.get("verdicts") or {}).items():
        rows.append(
            {
                "experiment_id": name,
                "experiment_name": name,
                "verdict": verdict.get("verdict"),
                "coverage_gain": verdict.get("coverage_gain"),
                "recall_at_20_gain": verdict.get("recall_at_20_gain"),
                "reason": verdict.get("reason"),
            }
        )
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "failure_census.json").write_text(
        json.dumps(status.get("failure_census") or {}, indent=2, default=str) + "\n"
    )
