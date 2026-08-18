from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.db.models import ExperimentCaseResultRecord, ExperimentRunRecord


def experiment_to_dict(
    session: Session, run: ExperimentRunRecord, *, include_cases: bool
) -> dict[str, Any]:
    cases = session.scalars(
        select(ExperimentCaseResultRecord).where(
            ExperimentCaseResultRecord.experiment_run_id == run.id
        )
    ).all()
    failure_counts = Counter(
        failure
        for case in cases
        for failure in (case.failure_types or ([case.failure_type] if case.failure_type else []))
    )
    payload: dict[str, Any] = {
        "id": run.id,
        "name": run.name,
        "status": run.status,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "config": {
            "id": run.config.id,
            "config_hash": run.config.config_hash,
            "index_identity": run.config.index_identity,
            "corpus_version": run.config.corpus_version,
            "evaluation_dataset_version": run.config.evaluation_dataset_version,
            "ingestion": run.config.ingestion_config,
            "retrieval": run.config.retrieval_config,
            "generation": run.config.generation_config,
            "answerability_gate": run.config.gate_config,
        },
        "aggregate_metrics": run.aggregate_metrics,
        "category_metrics": run.category_metrics,
        "latency": {
            "retrieval_ms": run.retrieval_latency_ms,
            "query_embedding_ms": run.query_embedding_latency_ms,
            "embedding_cache_lookup_ms": run.embedding_cache_lookup_latency_ms,
            "vector_search_ms": run.vector_search_latency_ms,
            "acl_filter_ms": run.acl_filter_latency_ms,
            "context_construction_ms": run.context_construction_latency_ms,
            "answerability_gate_cache_lookup_ms": (
                run.answerability_gate_cache_lookup_latency_ms
            ),
            "answerability_judge_ms": run.answerability_judge_latency_ms,
            "context_pruning_ms": run.context_pruning_latency_ms,
            "generation_ms": run.generation_latency_ms,
            "total_ms": run.total_latency_ms,
        },
        "usage": {
            "input_tokens": run.input_tokens,
            "output_tokens": run.output_tokens,
            "embedding_tokens": run.embedding_tokens,
            "query_embedding_cache_hits": run.query_embedding_cache_hits,
            "query_embedding_cache_misses": run.query_embedding_cache_misses,
            "external_embedding_calls": run.external_embedding_calls,
            "gate_cache_hits": run.gate_cache_hits,
            "gate_cache_misses": run.gate_cache_misses,
            "external_judge_calls": run.external_judge_calls,
            "local_judge_calls": run.local_judge_calls,
            "judge_prompt_tokens": run.judge_prompt_tokens,
            "judge_completion_tokens": run.judge_completion_tokens,
        },
        "error": run.error,
        "failure_counts": dict(sorted(failure_counts.items())),
    }
    if include_cases:
        payload["cases"] = [
            case_to_dict(case) for case in sorted(cases, key=lambda item: item.eval_case_id)
        ]
    return payload


def case_to_dict(case: ExperimentCaseResultRecord) -> dict[str, Any]:
    return {
        "id": case.id,
        "case_id": case.eval_case_id,
        "category": case.category,
        "question": case.question,
        "expected_answer": case.expected_answer,
        "expected_document_ids": case.expected_document_ids,
        "expected_chunk_ids": case.expected_chunk_ids,
        "forbidden_document_ids": case.forbidden_document_ids,
        "expected_versions": case.expected_versions,
        "principal": case.principal,
        "retrieved_document_ids": case.retrieved_document_ids,
        "retrieved_chunk_ids": case.retrieved_chunk_ids,
        "supporting_chunk_ids": case.supporting_chunk_ids,
        "generation_context_chunk_ids": case.generation_context_chunk_ids,
        "answerability_result": case.answerability_result,
        "answerability_operational_error": case.answerability_operational_error,
        "retrieval_trace": case.retrieval_trace,
        "answer": case.answer,
        "citations": case.citations,
        "expected_abstain": case.expected_abstain,
        "abstained": case.abstained,
        "metrics": case.metrics,
        "failure_type": case.failure_type,
        "failure_types": case.failure_types,
        "failure_details": case.failure_details,
        "security_passed": case.security_passed,
        "latency": {
            "retrieval_ms": case.retrieval_latency_ms,
            "query_embedding_ms": case.query_embedding_latency_ms,
            "embedding_cache_lookup_ms": case.embedding_cache_lookup_latency_ms,
            "vector_search_ms": case.vector_search_latency_ms,
            "acl_filter_ms": case.acl_filter_latency_ms,
            "context_construction_ms": case.context_construction_latency_ms,
            "answerability_gate_cache_lookup_ms": (
                case.answerability_gate_cache_lookup_latency_ms
            ),
            "answerability_judge_ms": case.answerability_judge_latency_ms,
            "context_pruning_ms": case.context_pruning_latency_ms,
            "generation_ms": case.generation_latency_ms,
            "total_ms": case.total_latency_ms,
        },
        "query_embedding_cache_hit": case.query_embedding_cache_hit,
        "gate_cache_hit": case.gate_cache_hit,
        "external_judge_calls": case.external_judge_calls,
        "local_judge_calls": case.local_judge_calls,
        "judge_prompt_tokens": case.judge_prompt_tokens,
        "judge_completion_tokens": case.judge_completion_tokens,
        "error": case.error,
        "rag_run_id": case.rag_run_id,
    }


def export_benchmark_markdown(session: Session, run_id: str, destination: Path) -> Path:
    destination.write_text(render_benchmark_markdown(session, run_id), encoding="utf-8")
    return destination


def render_benchmark_markdown(session: Session, run_id: str) -> str:
    run = session.get(ExperimentRunRecord, run_id)
    if run is None:
        raise ValueError("experiment run was not found")
    lines = [
        f"# Benchmark: {run.name}",
        "",
        "> Measured synthetic benchmark results; not a production performance claim.",
        "",
        f"- Run ID: `{run.id}`",
        f"- Status: `{run.status}`",
        f"- Corpus version: `{run.config.corpus_version}`",
        f"- Evaluation dataset version: `{run.config.evaluation_dataset_version}`",
        f"- Configuration hash: `{run.config.config_hash}`",
        f"- Index identity: `{run.config.index_identity}`",
        "",
        "## Configuration",
        "",
        "```json",
        _json_config(run),
        "```",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {_value(value)} |" for name, value in sorted(run.aggregate_metrics.items())
    )
    lines.extend(["", "## Category metrics", ""])
    for category, metrics in sorted(run.category_metrics.items()):
        lines.extend([f"### {category}", "", "| Metric | Value |", "|---|---:|"])
        lines.extend(f"| {name} | {_value(value)} |" for name, value in sorted(metrics.items()))
        lines.append("")
    lines.extend(
        [
            "## Latency",
            "",
            f"- Mean retrieval: {_value(run.retrieval_latency_ms)} ms",
            f"- Mean query embedding: {_value(run.query_embedding_latency_ms)} ms",
            f"- Mean embedding cache lookup: {_value(run.embedding_cache_lookup_latency_ms)} ms",
            f"- Mean vector search: {_value(run.vector_search_latency_ms)} ms",
            f"- Mean ACL predicate construction: {_value(run.acl_filter_latency_ms)} ms",
            f"- Mean context construction: {_value(run.context_construction_latency_ms)} ms",
            f"- Mean gate cache lookup: "
            f"{_value(run.answerability_gate_cache_lookup_latency_ms)} ms",
            f"- Mean answerability judge: {_value(run.answerability_judge_latency_ms)} ms",
            f"- Mean context pruning: {_value(run.context_pruning_latency_ms)} ms",
            f"- Mean generation: {_value(run.generation_latency_ms)} ms",
            f"- Mean total: {_value(run.total_latency_ms)} ms",
            "",
            "## Known limitations",
            "",
            "- Synthetic enterprise corpus and deterministic offline generation.",
            "- Hashing embeddings are an offline regression baseline, not semantic "
            "production embeddings.",
            "- Semantic answer correctness, groundedness, and completeness are pending.",
            "- Monetary cost is not estimated; provider token usage is stored when available.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_config(run: ExperimentRunRecord) -> str:
    import json

    return json.dumps(
        {
            "ingestion": run.config.ingestion_config,
            "retrieval": run.config.retrieval_config,
            "generation": run.config.generation_config,
            "answerability_gate": run.config.gate_config,
        },
        indent=2,
        sort_keys=True,
    )


def _value(value: object) -> str:
    return (
        "pending" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
    )
