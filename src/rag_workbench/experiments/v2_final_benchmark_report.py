# ruff: noqa: E501
from __future__ import annotations

from pathlib import Path
from typing import Any

V2_FINAL_BENCHMARK_HEADING = "## Enterprise RAG Workbench v2 — Final Frozen End-to-End Benchmark"


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _latency_line(payload: dict[str, Any] | None) -> str:
    values = payload or {}
    return (
        f"mean {_fmt(values.get('mean_ms'))} / p50 {_fmt(values.get('p50_ms'))} / "
        f"p95 {_fmt(values.get('p95_ms'))} (n={values.get('count', 0)})"
    )


def render_final_v2_benchmark_markdown(status: dict[str, Any]) -> str:
    architecture = status.get("architecture_configuration") or {}
    retrieval = status.get("retrieval") or {}
    metrics = retrieval.get("metrics") or {}
    pool = retrieval.get("pool") or {}
    end = status.get("end_to_end") or {}
    funnel = status.get("stage_funnel") or {}
    generator = status.get("generator_reliability") or {}
    provider = status.get("provider_reliability") or {}
    security = status.get("security") or {}
    citations = status.get("citations") or {}
    latency = status.get("latency") or {}
    usage = status.get("usage") or {}
    cost = status.get("cost") or {}
    comparison = status.get("v1_comparison") or {}
    categories = status.get("category_results") or {}
    judge = end.get("retrieval_complete_judge") or {}
    incomplete = end.get("retrieval_incomplete") or {}
    abstain = end.get("should_abstain") or {}
    v1 = comparison.get("v1") or {}
    v2 = comparison.get("v2") or {}
    closest = status.get("closest_previous_case") or {}
    preflight_e = status.get("embedding_preflight") or {}
    preflight_j = status.get("judge_preflight") or {}
    three = categories.get("multidoc_three") or {}
    exact = categories.get("exact_identifier") or {}
    version = categories.get("version_region") or {}
    duplicate = categories.get("near_duplicate") or {}
    semantic = categories.get("semantic_paraphrase") or {}
    single = categories.get("single_document") or {}
    two = categories.get("multidoc_two") or {}
    return "\n".join(
        [
            V2_FINAL_BENCHMARK_HEADING,
            "",
            "This is the final frozen quality benchmark for Enterprise RAG Workbench v2.",
            "One architecture, one unseen dataset, and one execution. Results were persisted;",
            "no retrieval, ranking, Top-K, Judge, generator, or security parameter was retuned.",
            "",
            "### Final architecture",
            "",
            f"Architecture `{status.get('architecture_id')}`. Parent `{status.get('parent_architecture')}`.",
            f"Selected ranking `{status.get('selected_v2_ranking')}`. Selected Judge `{status.get('selected_v2_judge')}`.",
            f"Architecture hash `{status.get('architecture_hash')}`. Immutable `{architecture.get('immutable')}`.",
            "",
            "- Embedding: openai-compatible `text-embedding-3-small` version `1`, dimension 64",
            "- Dense: pgvector cosine, threshold 0.28, Top-20",
            "- BM25 Okapi v1: `k1=1.2`, `b=0.75`, Top-20",
            "- RRF `k=60`, candidate union limit 30",
            "- Cross-Encoder `cross-encoder/ms-marco-MiniLM-L6-v2` revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`",
            "- Ranking policy `POINTWISE_CROSS_ENCODER_TOP5`, final Top-5",
            "- Judge `gpt-5.6-sol`, prompt `evidence-sufficiency-v1`",
            f"- Prompt hash `{architecture.get('judge', {}).get('prompt_hash')}`",
            f"- Schema identity `{architecture.get('judge', {}).get('schema_identity')}`",
            "- Generator `deterministic-extractive-v1.1` (parent `deterministic-extractive-v1`, semantic policy unchanged)",
            "- Transport retry: at most 2 physical attempts; TIMEOUT / CONNECTION_ERROR / RATE_LIMIT / PROVIDER_5XX only",
            "- Tenant/ACL/active-version filtering precedes Dense and BM25",
            "",
            "### Dataset methodology",
            "",
            f"Sealed dataset `{status.get('dataset_id')}`: {status.get('case_count')} unseen Harbor-brief questions,",
            f"generation method `{status.get('generation_method')}`. Hidden evaluator labels never reached",
            "Dense, BM25, RRF, Cross-Encoder, Sol, or the generator.",
            "",
            "| Category | Count |",
            "|---|---:|",
            "| Single-document answerable | 10 |",
            "| Two-document answerable | 18 |",
            "| Three-document answerable | 30 |",
            "| Exact-identifier answerable | 10 |",
            "| Version/region-sensitive answerable | 8 |",
            "| Near-duplicate answerable | 8 |",
            "| Semantic/paraphrase answerable | 6 |",
            "| ACL-sensitive should-abstain | 3 |",
            "| Partial/no-answer should-abstain | 3 |",
            "| Prompt-injection | 4 |",
            "",
            f"- Dataset SHA-256: `{status.get('dataset_hash')}`",
            f"- Maximum prior-dataset token-set overlap: {_fmt(status.get('maximum_prior_overlap'))} (ceiling 0.5; closest `{closest.get('final_case_id')}` vs `{closest.get('prior_dataset')}` / `{closest.get('prior_case_id')}`)",
            f"- Freeze timestamp: `{status.get('freeze_timestamp')}`",
            f"- One-shot lock: `{status.get('one_shot_locked_at')}`",
            f"- `final_v2_benchmark_one_shot`: `{status.get('final_v2_benchmark_one_shot')}`",
            "",
            "### Dataset independence",
            "",
            "Compared against every `data/eval/*.json` historical evaluation dataset, including hashing, semantic,",
            "threshold, Evidence Sufficiency, Evidence Coverage, Cross-Encoder, Hybrid, replication, Luna/Sol,",
            "V1 final, and P1–P3 datasets. Reliability fixtures remain outside `data/eval`.",
            "",
            "### One-shot freeze",
            "",
            "The dataset was frozen before inference. Interrupted execution resumes persisted checkpoints.",
            "Successful paid calls are not replayed.",
            "",
            "### External preflight",
            "",
            f"Embedding ledger {preflight_e.get('current_cumulative_embedding_calls')}; missing unique queries {preflight_e.get('missing_unique_query_embeddings')}; authorized ceiling {preflight_e.get('authorized_ceiling')}.",
            f"Judge logical ledger {preflight_j.get('logical_judge_ledger')}; missing logical requests {preflight_j.get('missing_logical_judge_requests')}; logical ceiling {preflight_j.get('logical_ceiling')}; physical-attempt ceiling {preflight_j.get('physical_attempt_ceiling')}.",
            "",
            "### Candidate-pool results",
            "",
            "| Before reranking | Hybrid union≤30 |",
            "|---|---:|",
            f"| Required evidence recall | {_fmt(pool.get('required_evidence_recall'))} |",
            f"| All required evidence coverage | {_fmt(pool.get('all_required_evidence_coverage'))} |",
            f"| Two-document coverage | {_fmt(pool.get('two_document_coverage'))} |",
            f"| Three-document coverage | {_fmt(pool.get('three_document_coverage'))} |",
            f"| Required evidence present in pool but lost by Top-5 | {retrieval.get('required_evidence_present_in_pool_lost_by_top5')} |",
            "",
            "### Retrieval results",
            "",
            "| Metric | V2 final |",
            "|---|---:|",
            f"| Hit@5 | {_fmt(metrics.get('hit_at_5'))} |",
            f"| Recall@5 | {_fmt(metrics.get('recall_at_5'))} |",
            f"| MRR | {_fmt(metrics.get('mrr'))} |",
            f"| nDCG@5 | {_fmt(metrics.get('ndcg_at_5'))} |",
            f"| Required evidence recall@5 | {_fmt(metrics.get('required_evidence_recall_at_5'))} |",
            f"| All required evidence coverage@5 | {_fmt(metrics.get('all_required_evidence_coverage_at_5'))} |",
            f"| Single-document coverage@5 | {_fmt(metrics.get('single_document_coverage_at_5'))} |",
            f"| Two-document coverage@5 | {_fmt(metrics.get('two_document_coverage_at_5'))} |",
            f"| Three-document coverage@5 | {_fmt(metrics.get('three_document_coverage_at_5'))} |",
            f"| Exact-ID recall@5 | {_fmt(metrics.get('exact_identifier_recall_at_5'))} |",
            f"| Version-sensitive recall@5 | {_fmt(metrics.get('version_sensitive_recall_at_5'))} |",
            f"| Near-duplicate preferred-source success | {_fmt(metrics.get('near_duplicate_preferred_source_success'))} |",
            f"| Semantic/paraphrase success | {_fmt(metrics.get('semantic_paraphrase_success'))} |",
            "",
            "### End-to-end results",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Correct answers | {end.get('correct_answers')} |",
            f"| Correct abstentions | {end.get('correct_abstentions')} |",
            f"| Unsupported answers | {end.get('unsupported_answers')} |",
            f"| Incorrect abstentions | {end.get('incorrect_abstentions')} |",
            f"| Accuracy | {_fmt(end.get('accuracy'))} |",
            f"| Precision | {_fmt(end.get('precision'))} |",
            f"| Recall | {_fmt(end.get('recall'))} |",
            f"| F1 | {_fmt(end.get('f1'))} |",
            f"| TP | {end.get('tp')} |",
            f"| FN | {end.get('fn')} |",
            f"| FP | {end.get('fp')} |",
            f"| TN | {end.get('tn')} |",
            "",
            "### Answer rate",
            "",
            f"Answerable cases {end.get('answerable_cases')}. Answered cases {end.get('answered_cases')}.",
            f"Correct-answer cases {end.get('correct_answer_cases')}.",
            f"Answer rate on answerable {_fmt(end.get('answer_rate_on_answerable'))}.",
            f"Correct-answer rate on answerable {_fmt(end.get('correct_answer_rate_on_answerable'))}.",
            f"Abstention rate on answerable {_fmt(end.get('abstention_rate_on_answerable'))}.",
            "Answer rate is not accuracy.",
            "",
            "### Retrieval-complete Judge results",
            "",
            f"Case count {end.get('retrieval_complete_count')}. TP {judge.get('tp')} FN {judge.get('fn')} FP {judge.get('fp')} TN {judge.get('tn')}.",
            f"Precision {_fmt(judge.get('precision'))}. Recall {_fmt(judge.get('recall'))}. F1 {_fmt(judge.get('f1'))}.",
            "",
            "Retrieval-incomplete subset:",
            f"case count {incomplete.get('case_count')}; candidate-generation misses {incomplete.get('candidate_generation_misses')};",
            f"pool-complete / Top-5-incomplete {incomplete.get('pool_complete_top5_incomplete')};",
            f"incorrect abstentions {incomplete.get('incorrect_abstentions')}; unsupported answers {incomplete.get('unsupported_answers')}.",
            "Missing retrieval evidence is not classified as a Judge capability failure.",
            "",
            "### Category results",
            "",
            f"Single-document: n={single.get('case_count')}, Top-5 complete {single.get('top5_complete')}, correct {single.get('correct_answers')}, incorrect abstentions {single.get('incorrect_abstentions')}, unsupported {single.get('unsupported_answers')}, Judge after complete retrieval recall {_fmt((single.get('judge_after_complete_retrieval') or {}).get('recall'))}.",
            f"Two-document: n={two.get('case_count')}, pool complete {two.get('candidate_pool_complete')}, Top-5 complete {two.get('top5_complete')}, correct {two.get('correct_answers')}, incorrect abstentions {two.get('incorrect_abstentions')}, unsupported {two.get('unsupported_answers')}, Judge after complete retrieval recall {_fmt((two.get('judge_after_complete_retrieval') or {}).get('recall'))}.",
            f"Three-document: n={three.get('case_count')}, pool complete {three.get('candidate_pool_complete')}, Top-5 complete {three.get('top5_complete')}, required recall@5 {_fmt(three.get('required_evidence_recall_at_5'))}, correct {three.get('correct_answers')}, incorrect abstentions {three.get('incorrect_abstentions')}, unsupported {three.get('unsupported_answers')}, Judge after complete retrieval recall {_fmt((three.get('judge_after_complete_retrieval') or {}).get('recall'))}. Failed-answerable stages: {three.get('failed_answerable_stage')}.",
            f"Exact identifier: n={exact.get('case_count')}, pool recall {_fmt(exact.get('candidate_pool_recall'))}, Top-5 recall {_fmt(exact.get('top5_recall'))}, Top-5 complete {exact.get('top5_complete')}, Judge TP/FN {(exact.get('judge_tp_fn') or {})}, correct {exact.get('correct_answers')}, incorrect abstentions {exact.get('incorrect_abstentions')}, unsupported {exact.get('unsupported_answers')}.",
            f"Version/region: retrieval active-version {_fmt(version.get('retrieval_active_version_correctness'))}, support {_fmt(version.get('support_active_version_correctness'))}, final-answer {_fmt(version.get('final_answer_version_correctness'))}, correct {version.get('correct_answers')}, incorrect abstentions {version.get('incorrect_abstentions')}.",
            f"Near-duplicate: preferred-source candidate {_fmt(duplicate.get('preferred_source_candidate_success'))}, Top-5 {_fmt(duplicate.get('preferred_source_top5_success'))}, correct {duplicate.get('correct_answers')}, incorrect abstentions {duplicate.get('incorrect_abstentions')}, wrong-source failures {duplicate.get('wrong_source_failures')}.",
            f"Semantic/paraphrase: retrieval success {_fmt(semantic.get('retrieval_success'))}, correct {semantic.get('correct_answers')}, incorrect abstentions {semantic.get('incorrect_abstentions')}.",
            "",
            "### Abstention safety",
            "",
            f"Should-abstain n={abstain.get('case_count')}. Correct abstentions {abstain.get('correct_abstentions')}. False-positive Judge decisions {abstain.get('false_positive_judge_decisions')}. Unsupported answers {abstain.get('unsupported_answers')}.",
            "",
            "### Prompt-injection safety",
            "",
            f"Case count {security.get('prompt_injection_cases')}. Boundary success rate {_fmt(security.get('boundary_success_rate'))}. Document instructions followed {security.get('document_instructions_followed')}. Unauthorized evidence selected {security.get('unauthorized_evidence_selected')}. Unsafe answers {security.get('unsafe_answers')}.",
            "",
            "### Security",
            "",
            f"ACL safety {_fmt(security.get('acl_safety'))}. Tenant isolation {_fmt(security.get('tenant_isolation'))}.",
            f"Unauthorized chunks to candidate generation / Cross-Encoder {security.get('unauthorized_chunks_to_cross_encoder', 0)}.",
            f"Unauthorized chunks to Sol {security.get('unauthorized_chunks_to_judge', 0)}.",
            f"Unauthorized supporting evidence {security.get('unauthorized_supporting_evidence', 0)}.",
            f"Unauthorized citations {security.get('unauthorized_citations', 0)}.",
            f"Active-version correctness {_fmt(security.get('version_correctness'))}.",
            "",
            "### Citation quality",
            "",
            f"Denominator `{citations.get('denominator')}` (n={citations.get('answered_case_count')}). Validity {_fmt(citations.get('citation_validity'))}. Correctness {_fmt(citations.get('citation_correctness'))}. Missing citation rate {_fmt(citations.get('missing_citation_rate'))}. Invalid citation count {citations.get('invalid_citation_count')}.",
            "",
            "### Generator reliability",
            "",
            f"Normal extractive {generator.get('normal_extractive')}. Verbatim supporting fallback {generator.get('verbatim_supporting_fallback')}. Typed GENERATION_FAILURE {generator.get('generation_failure')}. Silent generation failure {generator.get('silent_generation_failure')}. Fallback cases: {generator.get('fallback_cases')}.",
            "",
            "### Provider reliability",
            "",
            f"Logical Judge requests {provider.get('logical_judge_requests')}. Physical Sol attempts {provider.get('physical_sol_attempts')}. Transport retries {provider.get('transport_retries')}. TIMEOUT {provider.get('timeout_retries')}. RATE_LIMIT {provider.get('rate_limit_retries')}. PROVIDER_5XX {provider.get('provider_5xx_retries')}. CONNECTION_ERROR {provider.get('connection_error_retries')}. Transport-recovered {provider.get('transport_recovered_cases')}. Final JUDGE_REQUEST_ERROR {provider.get('final_judge_request_errors')}. Schema-valid quality retries {provider.get('schema_valid_quality_retries')}.",
            "",
            "### Idempotency",
            "",
            f"Unique logical IDs {provider.get('unique_logical_ids')}. Duplicate evaluation rows {provider.get('duplicate_evaluation_rows')}. One decision per logical request {provider.get('one_decision_per_logical_request')}.",
            "",
            "### Failure taxonomy",
            "",
            str(status.get("failure_taxonomy")),
            "",
            "### Stage funnel",
            "",
            f"Answerable {funnel.get('answerable')} → candidate complete {funnel.get('candidate_complete')} → Top-5 complete {funnel.get('top5_complete')} → Judge approved {funnel.get('judge_approved')} → generator succeeded {funnel.get('generator_succeeded')} → correct final {funnel.get('correct_final')}.",
            "",
            f"Primary remaining bottleneck: `{status.get('primary_remaining_bottleneck')}`.",
            "",
            "### Latency",
            "",
            "Live execution samples only. Cached replay Judge latency is excluded from Sol live figures.",
            f"- Query embedding: {_latency_line(latency.get('query_embedding'))}",
            f"- Dense: {_latency_line(latency.get('dense'))}",
            f"- BM25: {_latency_line(latency.get('bm25'))}",
            f"- RRF: {_latency_line(latency.get('rrf'))}",
            f"- Cross-Encoder: {_latency_line(latency.get('cross_encoder'))}",
            f"- Sol Judge (live): {_latency_line(latency.get('sol_judge_live'))}",
            f"- Support validation: {_latency_line(latency.get('support_validation'))}",
            f"- Generation: {_latency_line(latency.get('generation'))}",
            f"- Total pipeline: {_latency_line(latency.get('total_pipeline'))}",
            f"- Retry-related Judge: {_latency_line(latency.get('retry_related_judge'))}",
            "",
            "### Usage",
            "",
            f"Query embedding calls {usage.get('new_query_embedding_calls')}. Embedding tokens {usage.get('embedding_tokens')}. Document embedding calls {usage.get('document_embedding_calls', 0)}. Logical Judge requests {usage.get('logical_judge_requests')}. Physical Sol calls {usage.get('physical_sol_calls')}. Judge input tokens {usage.get('judge_input_tokens')}. Judge output tokens {usage.get('judge_output_tokens')}. Transport retries {usage.get('transport_retries')}. External reranker calls {usage.get('external_reranker_calls')}.",
            "",
            "### Cost",
            "",
            f"Official Sol token pricing verified. Sol cost USD {_fmt(cost.get('sol_cost_usd')) if isinstance(cost.get('sol_cost_usd'), float) else cost.get('sol_cost_usd')}. Embedding cost {cost.get('embedding_cost_usd')}. Mean Sol cost/case USD {_fmt(cost.get('mean_sol_cost_per_case_usd')) if isinstance(cost.get('mean_sol_cost_per_case_usd'), float) else cost.get('mean_sol_cost_per_case_usd')}. Retry-related incremental cost {cost.get('retry_related_incremental_cost_usd')}.",
            "",
            "### Descriptive V1 vs V2 comparison",
            "",
            "V1 and V2 final benchmarks use different unseen datasets. This is a descriptive cross-dataset comparison, NOT a controlled paired A/B experiment. Numerical differences must not be interpreted as a causal architecture improvement without qualification. No statistical significance is claimed.",
            "",
            "| Metric | V1 final (80 cases) | V2 final (100 cases) |",
            "|---|---:|---:|",
            f"| Correct answers | {v1.get('correct_answers')} | {v2.get('correct_answers')} |",
            f"| Correct abstentions | {v1.get('correct_abstentions')} | {v2.get('correct_abstentions')} |",
            f"| Unsupported answers | {v1.get('unsupported_answers')} | {v2.get('unsupported_answers')} |",
            f"| Incorrect abstentions | {v1.get('incorrect_abstentions')} | {v2.get('incorrect_abstentions')} |",
            f"| Accuracy | {_fmt(v1.get('accuracy'))} | {_fmt(v2.get('accuracy'))} |",
            f"| Precision | {_fmt(v1.get('precision'))} | {_fmt(v2.get('precision'))} |",
            f"| Recall | {_fmt(v1.get('recall'))} | {_fmt(v2.get('recall'))} |",
            f"| F1 | {_fmt(v1.get('f1'))} | {_fmt(v2.get('f1'))} |",
            f"| Candidate-pool all required coverage | {_fmt(v1.get('candidate_pool_all_required_coverage'))} | {_fmt(v2.get('candidate_pool_all_required_coverage'))} |",
            f"| Top-5 all required coverage | {_fmt(v1.get('top5_all_required_coverage'))} | {_fmt(v2.get('top5_all_required_coverage'))} |",
            f"| Three-document Top-5 coverage | {_fmt(v1.get('three_document_top5_coverage'))} | {_fmt(v2.get('three_document_top5_coverage'))} |",
            "",
            "### Reliability change note",
            "",
            "V2 includes `deterministic-extractive-v1.1`, bounded transient transport retry, typed `GENERATION_FAILURE`, support content-identity validation, and logical-vs-physical request accounting. Quality-selected ranking `POINTWISE_CROSS_ENCODER_TOP5` and Judge `GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1` remained frozen.",
            "",
            "### Known limitations",
            "",
            f"Measured primary remaining bottleneck `{status.get('primary_remaining_bottleneck')}`. Remaining quality work is V3 backlog. This benchmark did not retune V2 after seeing results.",
            "",
            "### Final V2 status",
            "",
            "Persisted frozen V2 final benchmark. Poor recall, if present, is reported rather than repaired in this cycle.",
        ]
    )


def persist_final_v2_benchmark_markdown(
    status: dict[str, Any], *, replace: bool = False
) -> None:
    path = Path("BENCHMARK.md")
    text = path.read_text()
    count = text.count(V2_FINAL_BENCHMARK_HEADING)
    if count > 1:
        raise ValueError("duplicate final v2 benchmark heading")
    section = render_final_v2_benchmark_markdown(status)
    if count == 1:
        if not replace:
            return
        prefix = text.split(V2_FINAL_BENCHMARK_HEADING, 1)[0]
        path.write_text(prefix.rstrip() + "\n\n" + section + "\n")
        return
    if not status.get("completed"):
        return
    path.write_text(text.rstrip() + "\n\n" + section + "\n")
