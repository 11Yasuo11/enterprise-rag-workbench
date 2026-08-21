"""End-to-end, cache-only Phase-5 experiment orchestration.

The production retrieval/reranking components are reused through a thin adapter. External
embeddings and semantic Judge calls fail closed; existing caches may be read but are never filled.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.answerability.cache import gate_cache_key
from rag_workbench.db.models import (
    AnswerabilityGateCacheRecord,
    Chunk,
    Document,
    DocumentPermission,
    DocumentVersion,
)
from rag_workbench.evaluation.eval_loop import (
    DEFAULT_KS,
    compare_results,
    evaluate,
    load_frozen_cases,
    load_gate,
)
from rag_workbench.evaluation.final_e2e_scorer_v2 import ScorerInput, score_case
from rag_workbench.experiments.reranker_e2e_benchmark import RERANKER_REVISION
from rag_workbench.experiments.v2_document_diversity import ranking_candidate
from rag_workbench.providers.embeddings.base import EmbeddingUsage
from rag_workbench.providers.llm.base import GenerationContext, GenerationRequest
from rag_workbench.providers.llm.extractive import ExtractiveGenerationProvider
from rag_workbench.reranking import CrossEncoderReranker
from rag_workbench.reranking.document_diversity import (
    select_document_diversified_top5,
    select_max_2_chunks_per_document_top5,
)
from rag_workbench.retrieval.bm25 import BM25Config, BM25Retriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.retriever import Retriever
from rag_workbench.safety.answerability_constraint_guard_v2 import (
    should_abstain_due_to_answerability_constraint,
)
from rag_workbench.safety.question_injection_guard_v2 import is_question_injection_v2
from rag_workbench.security.permissions import Principal

PAID_REQUIRED = "PAID_SEMANTIC_EVAL_REQUIRED"


class RagCaseRunner(Protocol):
    def run_rag_case(self, case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]: ...


class CacheOnlyEmbeddingProvider:
    """Correct cache identity with a hard network-call prohibition."""

    provider_name = "openai-compatible"
    model_name = "text-embedding-3-small"
    version = "1"
    dimension = 64
    is_external = True

    @property
    def usage(self) -> EmbeddingUsage:
        return EmbeddingUsage()

    def embed_query(self, text: str) -> list[float]:
        del text
        raise RuntimeError("EMBEDDING_CACHE_MISS_EXTERNAL_CALL_FORBIDDEN")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise RuntimeError("EMBEDDING_CACHE_MISS_EXTERNAL_CALL_FORBIDDEN")


@dataclass
class Phase5CachedRunner:
    session: Session
    config: dict[str, Any]
    baseline_rows: dict[str, dict[str, Any]]

    def __post_init__(self) -> None:
        retrieval = self.config["retrieval"]
        self.index_identity = self.config["identity"]["index_identity"]
        self.embedding = CacheOnlyEmbeddingProvider()
        self.dense = Retriever(self.session, self.embedding, self.index_identity)
        self.bm25 = BM25Retriever(
            self.session,
            index_identity=self.index_identity,
            embedding_provider=self.embedding.provider_name,
            embedding_model=self.embedding.model_name,
            embedding_version=self.embedding.version,
            embedding_dimension=self.embedding.dimension,
            config=BM25Config(),
        )
        configured_path = self.config["reranking"].get("model_path")
        if configured_path:
            model_path = Path(configured_path)
        else:
            from huggingface_hub.constants import HF_HUB_CACHE

            cache_name = "models--" + self.config["reranking"]["model"].replace("/", "--")
            model_path = (
                Path(HF_HUB_CACHE)
                / cache_name
                / "snapshots"
                / self.config["reranking"]["model_revision"]
            )
        if not model_path.is_dir():
            raise ValueError(f"local cross-encoder snapshot unavailable: {model_path}")
        self.reranker = CrossEncoderReranker(
            device="cpu",
            model_path=str(model_path),
            resolved_revision=RERANKER_REVISION,
        )
        self.dense_k = int(retrieval["dense_k"])
        self.bm25_k = int(retrieval["bm25_k"])
        self.union_k = int(retrieval["union_k"])
        self.rrf_k = int(retrieval["rrf_k"])
        self.dense_threshold = float(retrieval["dense_threshold"])
        self.top_k = int(self.config["reranking"]["top_k"])

    def run_rag_case(self, case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        del config
        started = time.perf_counter()
        principal = _principal(case)
        filters = self._filter_trace(principal)
        try:
            embedding = self.dense.query_embedding_cache.get_or_embed(case["question"])
            dense = self.dense.retrieve_with_embedding(
                embedding,
                top_k=self.dense_k,
                score_threshold=self.dense_threshold,
                principal=principal,
            )
            bm25 = self.bm25.retrieve(case["question"], top_k=self.bm25_k, principal=principal)
            union = reciprocal_rank_fusion(dense, bm25, top_k=10_000, rrf_k=self.rrf_k)[
                : self.union_k
            ]
            reranked = self.reranker.rerank(case["question"], union)
            selected = self._select(reranked)
        except Exception as exc:
            return self._execution_failure(case, filters, exc, started)

        dense_rows = [_retrieval_item(item) for item in dense]
        bm25_rows = [_retrieval_item(item) for item in bm25]
        union_rows = [_retrieval_item(item) for item in union]
        reranked_rows = [
            ranking_candidate(item)
            | {
                "rank": item.reranked_rank,
                "score": item.reranker_score,
                "input_rank": item.result.rank,
            }
            for item in reranked
        ]
        top_rows = [
            ranking_candidate(item) | {"rank": rank, "score": item.reranker_score}
            for rank, item in enumerate(selected, start=1)
        ]
        outcome = self._outcome(case, top_rows, principal)
        stages = {
            "filters": filters,
            "dense": {"items": dense_rows, "chunk_ids": _ids(dense_rows)},
            "bm25": {"items": bm25_rows, "chunk_ids": _ids(bm25_rows)},
            "rrf": {
                "items": union_rows,
                "chunk_ids": _ids(union_rows),
                "input_ids": sorted(set(_ids(dense_rows) + _ids(bm25_rows))),
            },
            "reranker": {
                "items": reranked_rows,
                "input_ids": _ids(union_rows),
                "output_ids": _ids(reranked_rows),
            },
            "top_k": {"items": top_rows, "chunk_ids": _ids(top_rows)},
            "judge": outcome["judge"],
            "generator": outcome["generator"],
            "citation": {"chunk_ids": outcome["citation_ids"]},
        }
        return {
            "case_id": case["case_id"],
            "candidate_ids": _ids(union_rows),
            "retrieved_candidate_document_ids": _docs(union_rows),
            "reranked_ids": _ids(reranked_rows),
            "reranked_document_ids": _docs(reranked_rows),
            "top_k_ids": _ids(top_rows),
            "topk_document_ids": _docs(top_rows),
            "final_answer": outcome["answer"],
            "abstained": outcome["abstained"],
            "citation_ids": outcome["citation_ids"],
            "cited_document_ids": [
                item["document_id"]
                for item in top_rows
                if item["chunk_id"] in set(outcome["citation_ids"])
            ],
            "behavior": outcome["behavior"],
            "answer_correct": outcome["answer_correct"],
            "citation_validity_pass": outcome["citation_valid"],
            "facts_missing": outcome["facts_missing"],
            "judge_answerable": outcome["judge"]["decision"],
            "evidence_sufficient": _evidence_complete(case, top_rows),
            "stage_trace": stages,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "token_usage": outcome["token_usage"],
            "api_calls": 0,
            "execution_error": outcome["execution_error"],
            "cache": {
                "embedding": embedding.cache_hit,
                "judge": outcome["judge"].get("cache_hit"),
            },
        }

    def _select(self, reranked: list[Any]) -> list[Any]:
        strategy = self.config["reranking"]["selection_strategy"]
        if strategy == "pointwise":
            return list(reranked[: self.top_k])
        if strategy == "one_chunk_per_document":
            return list(select_document_diversified_top5(reranked, top_k=self.top_k))
        if strategy == "max_two_chunks_per_document":
            return list(
                select_max_2_chunks_per_document_top5(
                    reranked,
                    top_k=self.top_k,
                    max_chunks_per_document=2,
                )
            )
        raise ValueError(f"unsupported selection_strategy: {strategy}")

    def _outcome(
        self, case: dict[str, Any], top_rows: list[dict[str, Any]], principal: Principal
    ) -> dict[str, Any]:
        baseline = self.baseline_rows.get(case["case_id"])
        baseline_ids = list((baseline or {}).get("retrieved_top_k_ids") or [])
        if baseline and baseline_ids == _ids(top_rows):
            return _historical_outcome(baseline)
        if is_question_injection_v2(case["question"]):
            return _score_outcome(case, top_rows, None, [], True, judge=None)
        evidence = tuple(
            GateEvidence(
                chunk_id=item["chunk_id"],
                document_id=item["document_id"],
                document_version_id=item["document_version_id"],
                version=item["version"],
                text=item["text"],
                index_identity=self.index_identity,
            )
            for item in top_rows
        )
        key, _ = gate_cache_key(
            case["question"],
            evidence,
            provider="openai",
            model="gpt-5.6-sol",
            gate_version="1",
            prompt_version="evidence-sufficiency-v1",
        )
        record = self.session.get(AnswerabilityGateCacheRecord, key)
        if record is None:
            return {
                "answer": None,
                "abstained": True,
                "citation_ids": [],
                "behavior": None,
                "answer_correct": None,
                "citation_valid": None,
                "facts_missing": [],
                "judge": {"decision": None, "cache_hit": False},
                "generator": {"answer": None, "status": "not_run"},
                "token_usage": {},
                "execution_error": PAID_REQUIRED,
            }
        judge_answerable = bool(record.result.get("answerable"))
        supporting_ids = list(record.result.get("supporting_chunk_ids") or [])
        answer: str | None = None
        citations: list[str] = []
        guard_triggered = False
        token_usage: dict[str, int] = {
            key: int(value)
            for key, value in {
                "judge_prompt": record.prompt_tokens,
                "judge_completion": record.completion_tokens,
            }.items()
            if value is not None
        }
        if judge_answerable:
            supporting = [item for item in top_rows if item["chunk_id"] in supporting_ids]
            guard_triggered = should_abstain_due_to_answerability_constraint(
                question=case["question"],
                evidence_texts=[item["text"] for item in supporting],
            )
            if not guard_triggered:
                contexts = tuple(
                    GenerationContext(
                        chunk_id=item["chunk_id"],
                        citation_label=f"C{i}",
                        text=item["text"],
                    )
                    for i, item in enumerate(supporting, start=1)
                )
                generated = ExtractiveGenerationProvider(
                    revision=self.config["generation"]["revision"]
                ).generate(
                    GenerationRequest(question=case["question"], prompt="", contexts=contexts)
                )
                if generated.answer and generated.used_chunk_ids:
                    answer = generated.answer
                    citations = list(generated.used_chunk_ids)
                token_usage.update(
                    {
                        "generation_prompt": generated.prompt_tokens or 0,
                        "generation_completion": generated.completion_tokens or 0,
                    }
                )
        return _score_outcome(
            case,
            top_rows,
            answer,
            citations,
            guard_triggered or not bool(answer),
            judge={
                "decision": judge_answerable,
                "cache_hit": True,
                "supporting_chunk_ids": supporting_ids,
            },
            token_usage=token_usage,
        )

    def _filter_trace(self, principal: Principal) -> dict[str, Any]:
        rows = self.session.execute(
            select(Chunk, Document, DocumentVersion)
            .join(Document, Chunk.document_fk == Document.id)
            .join(DocumentVersion, Chunk.document_version_id == DocumentVersion.id)
            .where(
                Chunk.index_identity == self.index_identity,
                Chunk.embedding_provider == self.embedding.provider_name,
                Chunk.embedding_model == self.embedding.model_name,
                Chunk.embedding_version == self.embedding.version,
                Chunk.embedding_dimension == self.embedding.dimension,
            )
            .order_by(Chunk.id)
        ).all()
        document_fks = {document.id for _, document, _ in rows}
        permissions: dict[str, set[str]] = {}
        if document_fks:
            for document_fk, group in self.session.execute(
                select(
                    DocumentPermission.document_fk,
                    DocumentPermission.permission_group,
                ).where(DocumentPermission.document_fk.in_(document_fks))
            ):
                permissions.setdefault(document_fk, set()).add(group)
        output: list[str] = []
        output_docs: list[str] = []
        reasons: dict[str, list[str]] = {}
        for chunk, document, version in rows:
            removed: list[str] = []
            if document.tenant_id != principal.tenant_id:
                removed.append("TENANT_MISMATCH")
            allowed_groups = permissions.get(document.id, set())
            if document.visibility != "public" and not (
                allowed_groups & set(principal.permission_groups)
            ):
                removed.append("ACL_DENIED")
            if not version.is_active:
                removed.append("INACTIVE_VERSION")
            if removed:
                reasons[chunk.id] = removed
            else:
                output.append(chunk.id)
                output_docs.append(document.document_id)
        return {
            "input_ids": [chunk.id for chunk, _, _ in rows],
            "input_document_ids": _unique([document.document_id for _, document, _ in rows]),
            "output_ids": output,
            "output_document_ids": _unique(output_docs),
            "removed_ids": sorted(reasons),
            "reasons": reasons,
        }

    def _execution_failure(
        self,
        case: dict[str, Any],
        filters: dict[str, Any],
        exc: Exception,
        started: float,
    ) -> dict[str, Any]:
        reason = str(exc) or type(exc).__name__
        return {
            "case_id": case["case_id"],
            "candidate_ids": [],
            "retrieved_candidate_document_ids": [],
            "reranked_ids": [],
            "reranked_document_ids": [],
            "top_k_ids": [],
            "topk_document_ids": [],
            "final_answer": None,
            "abstained": True,
            "citation_ids": [],
            "answer_correct": None,
            "stage_trace": {"filters": filters},
            "latency_ms": (time.perf_counter() - started) * 1000,
            "token_usage": {},
            "api_calls": 0,
            "execution_error": f"RAG_EXECUTION_FAILURE:{reason}",
        }


def _historical_outcome(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": row.get("final_answer"),
        "abstained": row.get("status") == "abstained",
        "citation_ids": list(row.get("citation_ids") or []),
        "behavior": row.get("behavior"),
        "answer_correct": row.get("behavior") in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"},
        "citation_valid": row.get("citation_validity_pass"),
        "facts_missing": list(row.get("facts_missing") or []),
        "judge": {
            "decision": row.get("judge_answerable"),
            "cache_hit": True,
            "supporting_chunk_ids": list(row.get("judge_supporting_chunk_ids") or []),
            "source": "phase5kr_frozen_artifact",
        },
        "generator": {
            "answer": row.get("final_answer"),
            "status": row.get("status"),
            "source": row.get("generation_path"),
        },
        "token_usage": {},
        "execution_error": None,
    }


def _score_outcome(
    case: dict[str, Any],
    top_rows: list[dict[str, Any]],
    answer: str | None,
    citations: list[str],
    abstained: bool,
    *,
    judge: dict[str, Any] | None,
    token_usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    cited = {item["chunk_id"]: item for item in top_rows if item["chunk_id"] in citations}
    scored = score_case(
        ScorerInput(
            arm="CANDIDATE",
            query_id=case["case_id"],
            question=case["question"],
            category=case["category"],
            should_abstain=case["answerable"] is False,
            expected_answerable=case["answerable"] is True,
            required_facts=tuple(case["required_facts"]),
            required_document_ids=tuple(case["gold_document_ids"]),
            final_answer=answer,
            final_answer_present=not abstained and bool(answer),
            citation_ids=tuple(citations),
            cited_document_ids=tuple(item["document_id"] for item in cited.values()),
            cited_chunk_texts={key: item["text"] for key, item in cited.items()},
            retrieved_top_k_ids=tuple(_ids(top_rows)),
            authorized_citation_ids=tuple(citations),
        )
    )
    return {
        "answer": answer,
        "abstained": abstained,
        "citation_ids": citations,
        "behavior": scored.behavior,
        "answer_correct": scored.behavior in {"CORRECT_COMPLETE_ANSWER", "CORRECT_ABSTENTION"},
        "citation_valid": scored.citation_validity_pass,
        "facts_missing": scored.facts_missing,
        "judge": judge or {"decision": None, "cache_hit": None},
        "generator": {"answer": answer, "status": "abstained" if abstained else "answered"},
        "token_usage": token_usage or {},
        "execution_error": None,
    }


def _retrieval_item(item: Any) -> dict[str, Any]:
    return {
        "chunk_id": item.chunk_id,
        "document_id": item.document_id,
        "document_version_id": item.document_version_id,
        "version": item.version,
        "rank": item.rank,
        "score": item.score,
        "dense_score": item.dense_score,
        "lexical_score": item.lexical_score,
        "fusion_score": item.fusion_score,
        "found_by_dense": item.found_by_dense,
        "found_by_bm25": item.found_by_bm25,
    }


def _principal(case: dict[str, Any]) -> Principal:
    raw = case["raw_metadata"].get("principal") or {}
    return Principal(
        principal_id=raw.get("principal_id", "evaluation-user"),
        tenant_id=raw.get("tenant_id", "acmeai"),
        permission_groups=frozenset(raw.get("permission_groups", ["employees"])),
    )


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [row["chunk_id"] for row in rows]


def _docs(rows: list[dict[str, Any]]) -> list[str]:
    return [row["document_id"] for row in rows]


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _evidence_complete(case: dict[str, Any], rows: list[dict[str, Any]]) -> bool | None:
    if case["answerable"] is not True:
        return None
    documents_complete = set(case["gold_document_ids"]) <= set(_docs(rows))
    facts = case["required_facts"]
    facts_complete = all(
        any(fact.casefold() in str(row.get("text") or "").casefold() for row in rows)
        for fact in facts
    )
    return documents_complete and facts_complete


def load_experiment_config(path: Path) -> tuple[dict[str, Any], str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("semantic_judge", {}).get("enabled"):
        raise ValueError("semantic_judge must remain disabled for cache-only experiments")
    required = {"experiment_id", "identity", "dataset", "retrieval", "reranking", "generation"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"experiment config missing fields: {missing}")
    return config, hashlib.sha256(path.read_bytes()).hexdigest()


def load_historical_rows(path: Path, arm: str = "FINAL_R") -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["query_id"]: row for row in rows if row.get("arm") == arm}


def run_cases_checkpointed(
    *,
    cases: list[dict[str, Any]],
    runner: RagCaseRunner,
    config: dict[str, Any],
    path: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if resume and path.exists():
        completed = {
            row["case_id"]: row
            for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        }
    mode = "a" if resume and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for case in cases:
            if case["case_id"] in completed:
                continue
            try:
                row = runner.run_rag_case(case, config)
            except Exception as exc:
                row = {
                    "case_id": case["case_id"],
                    "execution_error": f"RAG_EXECUTION_FAILURE:{type(exc).__name__}:{exc}",
                    "candidate_ids": [],
                    "reranked_ids": [],
                    "top_k_ids": [],
                    "abstained": True,
                }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed[case["case_id"]] = row
    return [completed[case["case_id"]] for case in cases]


def analyze_root_causes(traces: list[dict[str, Any]]) -> dict[str, Any]:
    three = [trace for trace in traces if trace["category"] == "three_document"]
    version = [trace for trace in traces if trace["category"] == "version_sensitive"]
    return {
        "three_document": _root_counts(three),
        "version_sensitive": _root_counts(version),
        "overall_failures": dict(Counter(t["primary_error"] for t in traces)),
    }


def _root_counts(traces: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(trace["primary_error"] for trace in traces)
    return {
        "total": len(traces),
        "retrieval_failures": counts["RETRIEVAL_FAILURE"],
        "filter_failures": counts["FILTER_FAILURE"],
        "ranking_failures": counts["RANKING_FAILURE"],
        "judge_failures": counts["JUDGE_FALSE_NEGATIVE"] + counts["JUDGE_FALSE_POSITIVE"],
        "generator_failures": counts["GENERATOR_INCOMPLETE"] + counts["GENERATOR_INCORRECT"],
        "unresolved": counts["UNRESOLVED"] + counts["OTHER"],
    }


def code_identifier() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "worktree_dirty": None}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def execute_experiment(
    *,
    config_path: Path,
    baseline_path: Path,
    output_root: Path,
    gate_path: Path,
    session: Session,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    config, config_hash = load_experiment_config(config_path)
    dataset_path = Path(config["dataset"]["path"])
    cases, dataset_hash = load_frozen_cases(dataset_path)
    expected_hash = config["dataset"].get("sha256")
    if expected_hash and dataset_hash != expected_hash:
        raise ValueError("frozen dataset hash mismatch")
    output_dir = output_root / config["experiment_id"]
    if output_dir.exists() and not resume:
        raise FileExistsError(f"experiment artifact already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = load_historical_rows(Path(config["identity"]["phase5kr_results"]))
    runner = Phase5CachedRunner(session, config, baseline_rows)
    raw_path = output_dir / "raw_results.jsonl"
    raw = run_cases_checkpointed(
        cases=cases,
        runner=runner,
        config=config,
        path=raw_path,
        resume=resume,
    )
    traces, metrics, census, slices = evaluate(cases, raw, DEFAULT_KS)
    metrics["acl_accuracy"] = _slice_metric(slices, "acl", "accuracy")
    metrics["tenant_accuracy"] = _slice_metric(slices, "tenant", "accuracy")
    metrics["unanswerable_accuracy"] = _slice_metric(slices, "unanswerable", "accuracy")
    incomplete = sum(trace["answer_correct"] is None for trace in traces)
    if incomplete:
        for name in (
            "precision",
            "recall",
            "f1",
            "accuracy",
            "answer_rate",
            "abstention_rate",
            "citation_validity",
            "unsupported_answers",
        ):
            metrics[name] = None
    metrics["execution_complete_cases"] = len(cases) - incomplete
    metrics["execution_incomplete_cases"] = incomplete
    baseline_results = json.loads(
        ((baseline_path / "results.json") if baseline_path.is_dir() else baseline_path).read_text(
            encoding="utf-8"
        )
    )
    candidate_results = {
        "experiment_id": config["experiment_id"],
        "dataset_hash": dataset_hash,
        "config_hash": config_hash,
        "code_identifier": code_identifier(),
        "metrics": metrics,
        "failure_census": census,
        "slice_analysis": slices,
        "traces": traces,
    }
    regression = compare_results(baseline_results, candidate_results, load_gate(gate_path)).payload
    if incomplete and regression["decision"] != "REJECT":
        regression["decision"] = "MANUAL_REVIEW"
        regression["reasons"].append(f"{incomplete} cases require paid/cached semantic outcomes")
    regression["promotion_status"] = (
        "PROMOTION_ELIGIBLE" if regression["decision"] == "PROMOTE" else "NOT_ELIGIBLE"
    )
    root_causes = analyze_root_causes(traces)
    manifest = {
        "experiment_id": config["experiment_id"],
        "dataset_path": str(dataset_path),
        "dataset_hash": dataset_hash,
        "case_count": len(cases),
        "config_hash": config_hash,
        "code_identifier": candidate_results["code_identifier"],
        "semantic_judge": "off-cache-only",
    }
    (output_dir / "experiment_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    write_json(output_dir / "dataset_manifest.json", manifest)
    write_jsonl(output_dir / "traces.jsonl", traces)
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "failure_census.json", census)
    write_jsonl(
        output_dir / "failure_evidence.jsonl",
        [
            {
                "case_id": trace["case_id"],
                "primary_error": trace["primary_error"],
                "unresolved_reason": trace["unresolved_reason"],
                "error_evidence": trace["error_evidence"],
            }
            for trace in traces
            if trace["primary_error"] != "PASS"
        ],
    )
    write_json(output_dir / "slice_analysis.json", slices)
    write_json(output_dir / "regression.json", regression)
    write_json(output_dir / "root_cause_analysis.json", root_causes)
    hypothesis = config.get("hypothesis") or {}
    (output_dir / "hypothesis.md").write_text(_render_hypothesis(hypothesis), encoding="utf-8")
    (output_dir / "report.md").write_text(
        _render_report(manifest, metrics, census, root_causes, regression),
        encoding="utf-8",
    )
    write_json(output_dir / "results.json", candidate_results)
    return output_dir, regression


def _render_hypothesis(hypothesis: dict[str, Any]) -> str:
    fields = (
        "observation",
        "root_cause",
        "change",
        "expected_effect",
        "metrics_to_watch",
        "regression_risks",
    )
    return (
        "# Hypothesis\n\n"
        + "\n\n".join(
            f"## {field.replace('_', ' ').title()}\n\n{hypothesis.get(field, 'UNSPECIFIED')}"
            for field in fields
        )
        + "\n"
    )


def _slice_metric(slices: dict[str, Any], name: str, metric: str) -> Any:
    return (slices.get("slices", {}).get(name) or {}).get(metric)


def _render_report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    census: dict[str, Any],
    root_causes: dict[str, Any],
    regression: dict[str, Any],
) -> str:
    return "\n".join(
        [
            f"# Experiment {manifest['experiment_id']}",
            "",
            f"Dataset hash: `{manifest['dataset_hash']}`",
            f"Config hash: `{manifest['config_hash']}`",
            "Semantic Judge: `off-cache-only`",
            "Deterministic Eval API calls: `0`",
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(metrics, indent=2, sort_keys=True),
            "```",
            "",
            "## Failure census",
            "",
            "```json",
            json.dumps(census["counts"], indent=2, sort_keys=True),
            "```",
            "",
            "## Root cause analysis",
            "",
            "```json",
            json.dumps(root_causes, indent=2, sort_keys=True),
            "```",
            "",
            f"## Regression gate: {regression['decision']}",
            "",
            *(f"- {reason}" for reason in regression["reasons"]),
            "",
        ]
    )
