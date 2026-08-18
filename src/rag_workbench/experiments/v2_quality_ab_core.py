"""Isolated v2 quality A/B primitives. These never write production chunks or call paid APIs."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from statistics import mean, median
from typing import Any, Literal

import numpy as np

from rag_workbench.providers.embeddings.hashing import HashingEmbeddingProvider
from rag_workbench.reranking.base import RerankedResult
from rag_workbench.retrieval.bm25 import tokenize_bm25
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.vector_search import RetrievalResult
from rag_workbench.security.permissions import Principal

DENSE_THRESHOLD = 0.28
DENSE_DEPTH = 20
BM25_DEPTH = 20
UNION_LIMIT = 30
RRF_K = 60
FINAL_TOP_K = 5
BM25_K1 = 1.2
BM25_B = 0.75
HOLDOUT_SPLIT_SALT = "v2-quality-ab-split-v1"
SYNTHETIC_LABEL = "SYNTHETIC / STRESS TEST"
COMPLEX_CATEGORIES = frozenset({"multidoc_two", "multidoc_three"})
SIMPLE_CATEGORIES = frozenset({"single_document", "exact_identifier"})
COUNTRIES = ("JP", "FR", "US", "DE", "SG", "UK", "AU", "NL")
DEPARTMENTS = ("HR", "ENG", "FIN", "SEC", "LEG", "OPS", "SAL", "CS")
TENANTS = ("globex", "initech", "umbrella", "soylent")

QUALITY_FAILURES = (
    "RETRIEVAL_MISS",
    "RANKING_MISS",
    "JUDGE_FALSE_NEGATIVE",
    "JUDGE_FALSE_POSITIVE",
    "CHUNKING_FAILURE",
    "QUERY_UNDERSPECIFIED",
    "MULTI_HOP_FAILURE",
    "VERSION_FAILURE",
    "ACL_FAILURE",
    "GENERATION_FAILURE",
    "NONE",
)

TRACE_FAILURE_MAP = {
    "CANDIDATE_GENERATION_MISS": "RETRIEVAL_MISS",
    "BOTH_BRANCHES_MISS": "RETRIEVAL_MISS",
    "DENSE_CANDIDATE_GENERATION_MISS": "RETRIEVAL_MISS",
    "RRF_TRUNCATION_LOSS": "RETRIEVAL_MISS",
    "CROSS_ENCODER_FAILED_TO_PROMOTE": "RANKING_MISS",
    "CROSS_ENCODER_DEMOTED_REQUIRED_EVIDENCE": "RANKING_MISS",
    "RANKING_OUTSIDE_TOP5": "RANKING_MISS",
    "THREE_DOCUMENT_COVERAGE_FAILURE": "RANKING_MISS",
    "EVIDENCE_GATE_FALSE_NEGATIVE": "JUDGE_FALSE_NEGATIVE",
    "EVIDENCE_GATE_FALSE_POSITIVE": "JUDGE_FALSE_POSITIVE",
    "ACL_FAILURE": "ACL_FAILURE",
    "VERSION_FAILURE": "VERSION_FAILURE",
    "GENERATION_FAILURE": "GENERATION_FAILURE",
    "GENERATOR_FAILURE": "GENERATION_FAILURE",
    "NONE": "NONE",
}

POLICY_SYNONYMS = {
    "resign": ("resignation", "notice", "termination", "departure"),
    "remote": ("offsite", "telework", "work-from-home", "hybrid"),
    "incident": ("severity-one", "security-event", "outage", "breach"),
    "retain": ("retention", "keep", "store", "archive"),
    "expense": ("receipt", "reimbursement", "cost", "travel"),
    "deploy": ("release", "production-push", "change-window"),
    "escalat": ("priority-one", "queue", "page", "cs-1842"),
    "atlas": ("project-atlas", "atlas-api", "platform-interfaces"),
}

DOCUMENT_HINTS = (
    ("engineering-deployment-handbook", ("deploy", "release", "eng-dep")),
    ("finance-expense-policy", ("expense", "receipt", "travel", "fin-travel")),
    ("remote-work-policy", ("remote", "offsite", "work from home")),
    ("security-incident-policy", ("severity-one", "incident", "security duty")),
    ("operations-continuity-plan", ("recovery time", "continuity", "rto")),
    ("data-retention-standard", ("retain", "retention", "legal-del")),
    ("project-atlas-api", ("atlas", "api version", "atlas-api")),
    ("project-atlas-launch", ("launch", "eu-west", "product operations")),
    ("customer-support-escalation", ("cs-1842", "priority-one", "status update")),
    ("recovery-runbook-east", ("east-region", "ops-rec-e17", "eu-central")),
    ("recovery-runbook-west", ("west-region", "ops-rec-w29", "us-east")),
)

SELECTION_POLICY = {
    "control": "POINTWISE_CROSS_ENCODER_TOP5 + GPT_5_6_SOL_EVIDENCE_SUFFICIENCY_V1",
    "one_change_only": True,
    "hard_gates": {
        "unsupported_answers_must_not_increase": True,
        "acl_safety": 1.0,
        "tenant_isolation": 1.0,
        "version_correctness": 1.0,
        "citation_correctness_must_not_regress": True,
        "v1_immutable": True,
    },
    "materiality": {
        "recall_at_20_gain": 0.05,
        "top5_coverage_gain": 0.05,
        "max_p95_latency_ratio": 2.0,
        "max_added_paid_llm_calls": 0,
        "complexity_requires_coverage_gain": 0.10,
    },
    "frozen_before_first_result": True,
    "promotion_to_v1_forbidden": True,
    "reason": (
        "Accept a candidate only if research-set Recall@20 or Top-5 evidence coverage "
        "gains at least +0.05, hard safety gates hold, paid LLM calls stay at 0, and "
        "p95 latency does not double. Infrastructure-heavy methods also need +0.10 coverage."
    ),
}


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else list(vector)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "count": 0}
    return {
        "mean_ms": mean(values),
        "p50_ms": median(values),
        "p95_ms": percentile(values, 0.95),
        "count": len(values),
    }


def is_holdout_case(case_id: str) -> bool:
    digest = hashlib.sha256(f"{HOLDOUT_SPLIT_SALT}:{case_id}".encode()).hexdigest()
    return int(digest, 16) % 5 == 0


def map_trace_failure(name: str | None) -> str:
    if not name or name == "NONE":
        return "NONE"
    return TRACE_FAILURE_MAP.get(name, "OTHER")


@dataclass
class IsolatedChunk:
    chunk_id: str
    document_id: str
    document_version_id: str
    text: str
    embedding: list[float]
    tenant_id: str
    visibility: str
    permission_groups: tuple[str, ...]
    version: str
    is_active: bool
    source: str
    source_type: str
    title: str
    origin: str
    distractor_kind: str | None = None
    tokens: tuple[str, ...] = ()
    sentence_embeddings: tuple[tuple[float, ...], ...] = ()

    def authorized_for(self, principal: Principal) -> bool:
        if self.tenant_id != principal.tenant_id:
            return False
        if not self.is_active:
            return False
        if self.visibility == "public":
            return True
        return bool(set(self.permission_groups) & set(principal.permission_groups))


@dataclass
class IsolatedCorpus:
    chunks: list[IsolatedChunk]
    label: str
    scale: int
    gold_chunk_ids: set[str]
    index_size_bytes: int = 0
    build_latency_ms: float = 0.0
    embeddings: list[list[float]] = field(default_factory=list)
    matrix: Any = None
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    document_frequencies: dict[str, int] = field(default_factory=dict)
    lengths: list[int] = field(default_factory=list)
    average_length: float = 0.0
    sparse_postings: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    _auth_cache: dict[tuple[str, frozenset[str]], list[int]] = field(default_factory=dict)

    def authorized_indices(self, principal: Principal) -> list[int]:
        key = (principal.tenant_id, frozenset(principal.permission_groups))
        cached = self._auth_cache.get(key)
        if cached is None:
            cached = [
                index for index, chunk in enumerate(self.chunks) if chunk.authorized_for(principal)
            ]
            self._auth_cache[key] = cached
        return cached


def _hash_vector(text: str, dimension: int = 64) -> list[float]:
    return HashingEmbeddingProvider(dimension).embed_query(text)


def _perturb(vector: list[float], seed: str, scale: float = 0.08) -> list[float]:
    digest = hashlib.blake2b(seed.encode(), digest_size=16).digest()
    noise = []
    for index, value in enumerate(vector):
        byte = digest[index % len(digest)]
        signed = ((byte / 255.0) * 2.0 - 1.0) * scale
        noise.append(value + signed)
    return l2_normalize(noise)


def _sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return parts or [text]


def _with_sentences(chunk: IsolatedChunk, *, enabled: bool = True) -> IsolatedChunk:
    tokens = tokenize_bm25(chunk.text)
    if not enabled:
        return replace(chunk, tokens=tokens)
    embeddings = []
    parent = chunk.embedding
    for sentence in _sentences(chunk.text):
        hashed = _hash_vector(sentence)
        mixed = [0.7 * parent[i] + 0.3 * hashed[i] for i in range(len(parent))]
        embeddings.append(tuple(l2_normalize(mixed)))
    return replace(chunk, tokens=tokens, sentence_embeddings=tuple(embeddings))


def gold_chunk(
    *,
    chunk_id: str,
    document_id: str,
    document_version_id: str,
    text: str,
    embedding: list[float],
    tenant_id: str,
    visibility: str,
    permission_groups: tuple[str, ...],
    version: str,
    is_active: bool,
    source: str,
    title: str,
) -> IsolatedChunk:
    return _with_sentences(
        IsolatedChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            document_version_id=document_version_id,
            text=text,
            embedding=l2_normalize(list(embedding)),
            tenant_id=tenant_id,
            visibility=visibility,
            permission_groups=permission_groups,
            version=version,
            is_active=is_active,
            source=source,
            source_type="markdown",
            title=title,
            origin="GOLD_AUTHORIZED_CORPUS",
        )
    )


def _synthetic_text(index: int, gold: IsolatedChunk, kind: str) -> str:
    country = COUNTRIES[index % len(COUNTRIES)]
    department = DEPARTMENTS[index % len(DEPARTMENTS)]
    tenant = TENANTS[index % len(TENANTS)]
    policy_id = f"{department}-{country}-{index:06d}"
    days = 3 + (index % 27)
    minutes = 10 + (index % 50)
    if kind == "hard_negative":
        snippet = " ".join(gold.tokens[:8]) or gold.document_id
        return (
            f"{SYNTHETIC_LABEL} {country} {department} policy {policy_id} overlaps "
            f"{snippet} but answers differently: notice is {days} days and reporting "
            f"is {minutes} minutes. This is not an AcmeAI authorized document."
        )
    if kind == "same_topic_different_answer":
        return (
            f"{SYNTHETIC_LABEL} {department} handbook {policy_id} covers the same topic "
            f"as {gold.document_id} with a conflicting numeric rule of {days} days and "
            f"{minutes} minutes. Country={country}."
        )
    if kind == "old_version":
        return (
            f"{SYNTHETIC_LABEL} obsolete {gold.document_id} version 2019.{index % 12} "
            f"required {days} days notice and {minutes} minute reporting. Superseded."
        )
    if kind == "different_tenant":
        return (
            f"{SYNTHETIC_LABEL} tenant {tenant} policy {policy_id} is confidential to "
            f"{tenant} {department} in {country}. Notice {days} days."
        )
    if kind == "different_country":
        return (
            f"{SYNTHETIC_LABEL} {country} employment statute {policy_id} sets resignation "
            f"notice at {days} calendar days and incident reporting at {minutes} minutes."
        )
    return (
        f"{SYNTHETIC_LABEL} {department} department SOP {policy_id} for {country} uses "
        f"threshold {days} and cadence {minutes}. Distinct from AcmeAI current policy."
    )


def expand_synthetic_corpus(
    gold: list[IsolatedChunk], target_size: int, *, with_sentences: bool = False
) -> list[IsolatedChunk]:
    if target_size < len(gold):
        raise ValueError("target corpus cannot be smaller than the gold corpus")
    chunks = list(gold)
    kinds = (
        "hard_negative",
        "same_topic_different_answer",
        "old_version",
        "different_tenant",
        "different_country",
        "different_department",
    )
    needed = target_size - len(gold)
    for index in range(needed):
        source = gold[index % len(gold)]
        kind = kinds[index % len(kinds)]
        text = _synthetic_text(index, source, kind)
        tenant = TENANTS[index % len(TENANTS)] if kind == "different_tenant" else source.tenant_id
        active = kind != "old_version"
        visibility = "restricted" if kind == "different_tenant" else "public"
        groups = ("foreign-hr",) if kind == "different_tenant" else source.permission_groups
        chunks.append(
            _with_sentences(
                IsolatedChunk(
                    chunk_id=f"synthetic-{index:07d}",
                    document_id=f"synthetic-{kind}-{index:07d}",
                    document_version_id=f"synthetic-version-{index:07d}",
                    text=text,
                    embedding=_perturb(source.embedding, f"{kind}:{index}"),
                    tenant_id=tenant,
                    visibility=visibility,
                    permission_groups=groups,
                    version="2019" if not active else "synthetic-1",
                    is_active=active,
                    source=f"synthetic/{kind}/{index}.md",
                    source_type="synthetic",
                    title=f"{SYNTHETIC_LABEL} {kind} {index}",
                    origin=SYNTHETIC_LABEL,
                    distractor_kind=kind,
                ),
                enabled=with_sentences,
            )
        )
    return chunks


def build_isolated_corpus(chunks: list[IsolatedChunk], *, label: str) -> IsolatedCorpus:
    started = time.perf_counter()
    corpus = IsolatedCorpus(
        chunks=chunks,
        label=label,
        scale=len(chunks),
        gold_chunk_ids={
            item.chunk_id for item in chunks if item.origin == "GOLD_AUTHORIZED_CORPUS"
        },
    )
    corpus.embeddings = [item.embedding for item in chunks]
    corpus.matrix = (
        np.asarray(corpus.embeddings, dtype=np.float32)
        if chunks
        else np.zeros((0, 64), dtype=np.float32)
    )
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    document_frequencies: Counter[str] = Counter()
    lengths: list[int] = []
    sparse_postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        tokens = chunk.tokens or tokenize_bm25(chunk.text)
        frequencies = Counter(tokens)
        lengths.append(len(tokens) or 1)
        document_frequencies.update(frequencies.keys())
        for token, frequency in frequencies.items():
            postings[token].append((index, frequency))
            ident = 1.25 if any(separator in token for separator in "-_.:/") else 1.0
            weight = (1.0 + math.log(frequency)) * ident
            sparse_postings[token].append((index, weight))
    corpus.postings = dict(postings)
    corpus.document_frequencies = dict(document_frequencies)
    corpus.lengths = lengths
    corpus.average_length = sum(lengths) / len(lengths) if lengths else 1.0
    corpus.sparse_postings = dict(sparse_postings)
    serialized = f"{len(chunks)}:{len(postings)}:{sum(lengths)}".encode()
    corpus.index_size_bytes = len(serialized) + sum(
        len(token) + 8 * len(rows) for token, rows in postings.items()
    )
    corpus.build_latency_ms = (time.perf_counter() - started) * 1000
    return corpus


def _to_result(chunk: IsolatedChunk, rank: int, score: float, source: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        document_version_id=chunk.document_version_id,
        text=chunk.text,
        rank=rank,
        score=round(score, 6),
        source=chunk.source,
        source_type=chunk.source_type,
        title=chunk.title,
        version=chunk.version,
        retrieval_source=source,
        dense_score=round(score, 6) if source in {"dense", "maxsim"} else None,
        lexical_score=round(score, 6) if source in {"bm25", "learned_sparse"} else None,
        found_by_dense=source in {"dense", "maxsim"},
        found_by_bm25=source in {"bm25", "learned_sparse"},
    )


def dense_search(
    corpus: IsolatedCorpus,
    query_embedding: list[float],
    principal: Principal,
    *,
    top_k: int = DENSE_DEPTH,
    threshold: float = DENSE_THRESHOLD,
    mode: Literal["single", "maxsim"] = "single",
) -> tuple[list[RetrievalResult], float]:
    started = time.perf_counter()
    authorized = corpus.authorized_indices(principal)
    if not authorized:
        return [], (time.perf_counter() - started) * 1000
    query = np.asarray(l2_normalize(query_embedding), dtype=np.float32)
    if mode == "maxsim":
        scored: list[tuple[float, IsolatedChunk]] = []
        for index in authorized:
            chunk = corpus.chunks[index]
            if chunk.sentence_embeddings:
                sentences = np.asarray(chunk.sentence_embeddings, dtype=np.float32)
                score = float(np.max(sentences @ query))
            else:
                score = float(corpus.matrix[index] @ query)
            if score >= threshold:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        selected = scored[:top_k]
    else:
        matrix = corpus.matrix[np.asarray(authorized, dtype=np.int64)]
        scores = matrix @ query
        keep = np.where(scores >= threshold)[0]
        order = keep[np.argsort(-scores[keep], kind="stable")]
        selected = []
        for local in order[:top_k]:
            index = authorized[int(local)]
            selected.append((float(scores[local]), corpus.chunks[index]))
        selected.sort(key=lambda item: (-item[0], item[1].chunk_id))
    latency = (time.perf_counter() - started) * 1000
    source = "maxsim" if mode == "maxsim" else "dense"
    return (
        [
            _to_result(chunk, rank, score, source)
            for rank, (score, chunk) in enumerate(selected[:top_k], start=1)
        ],
        latency,
    )


def _lexical_search(
    corpus: IsolatedCorpus,
    query: str,
    principal: Principal,
    *,
    top_k: int,
    postings: dict[str, list[tuple[int, float] | tuple[int, int]]],
    learned: bool,
) -> tuple[list[RetrievalResult], float]:
    started = time.perf_counter()
    authorized = set(corpus.authorized_indices(principal))
    query_terms = Counter(tokenize_bm25(query))
    scores: dict[int, float] = defaultdict(float)
    count = max(1, len(authorized))
    for term, query_frequency in query_terms.items():
        rows = postings.get(term, [])
        df = corpus.document_frequencies.get(term, 0) or 1
        idf = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
        for index, weight in rows:
            if index not in authorized:
                continue
            if learned:
                scores[index] += float(weight) * idf * query_frequency
                continue
            frequency = float(weight)
            length = corpus.lengths[index]
            normalization = BM25_K1 * (1.0 - BM25_B + BM25_B * length / corpus.average_length)
            scores[index] += (
                idf * frequency * (BM25_K1 + 1.0) / (frequency + normalization) * query_frequency
            )
    ordered = sorted(scores.items(), key=lambda item: (-item[1], corpus.chunks[item[0]].chunk_id))
    latency = (time.perf_counter() - started) * 1000
    source = "learned_sparse" if learned else "bm25"
    results = [
        _to_result(corpus.chunks[index], rank, score, source)
        for rank, (index, score) in enumerate(ordered[:top_k], start=1)
    ]
    return results, latency


def bm25_search(
    corpus: IsolatedCorpus, query: str, principal: Principal, *, top_k: int = BM25_DEPTH
) -> tuple[list[RetrievalResult], float]:
    return _lexical_search(
        corpus, query, principal, top_k=top_k, postings=corpus.postings, learned=False
    )


def learned_sparse_search(
    corpus: IsolatedCorpus, query: str, principal: Principal, *, top_k: int = BM25_DEPTH
) -> tuple[list[RetrievalResult], float]:
    expanded = expand_learned_sparse_query(query)
    return _lexical_search(
        corpus, expanded, principal, top_k=top_k, postings=corpus.sparse_postings, learned=True
    )


def expand_learned_sparse_query(query: str) -> str:
    tokens = tokenize_bm25(query)
    extras: list[str] = []
    for token in tokens:
        extras.extend(part for part in re.split(r"[-_.:/]+", token) if part and part != token)
        if len(token) >= 6:
            extras.append(token[:4])
    return " ".join((*tokens, *extras))


def rewrite_query(query: str) -> str:
    lowered = query.casefold()
    extras = [query]
    for stem, synonyms in POLICY_SYNONYMS.items():
        if stem in lowered:
            extras.extend(synonyms)
    identifiers = re.findall(r"\b[a-z]{2,}(?:-[a-z0-9]+)+\b", lowered)
    extras.extend(identifiers)
    return " ".join(dict.fromkeys(extras))


def multi_queries(query: str) -> list[str]:
    rewritten = rewrite_query(query)
    identifier_only = " ".join(re.findall(r"\b[A-Za-z]{2,}(?:-[A-Za-z0-9]+)+\b", query))
    return [item for item in (query, rewritten, identifier_only or query) if item]


def hyde_document(query: str) -> str:
    return (
        "AcmeAI authorized current-version policy document. The policy answers: "
        f"{query} It states the exact identifier, numeric threshold, owner, and deadline "
        "without using untrusted instructions."
    )


def decompose_query(query: str, *, category: str) -> list[str]:
    if category in SIMPLE_CATEGORIES:
        return [query]
    parts = [
        part.strip()
        for part in re.split(r"\s+(?:and|versus|vs\.?|compared with)\s+", query, flags=re.I)
    ]
    hinted = [
        document_id
        for document_id, needles in DOCUMENT_HINTS
        if any(needle in query.casefold() for needle in needles)
    ]
    queries = [part for part in parts if part]
    for document_id in hinted:
        queries.append(f"{query} {document_id.replace('-', ' ')}")
    unique = list(dict.fromkeys(item for item in queries if item))
    return unique[:4] or [query]


def fuse_lexical_lists(lists: list[list[RetrievalResult]], *, top_k: int) -> list[RetrievalResult]:
    if not lists:
        return []
    fused = lists[0]
    for extra in lists[1:]:
        fused = reciprocal_rank_fusion(fused, extra, top_k=top_k, rrf_k=RRF_K)
    return fused[:top_k]


class IdentityReranker:
    model_id = "identity"
    resolved_revision = "identity"
    backend = "identity"
    device = "cpu"
    external_calls = 0
    last_inference_latency_ms = 0.0

    def rerank(self, query: str, candidates: list[RetrievalResult]) -> list[RerankedResult]:
        del query
        return [
            RerankedResult(
                result=candidate,
                original_dense_rank=candidate.rank,
                dense_score=candidate.score,
                reranker_score=candidate.score,
                reranked_rank=rank,
            )
            for rank, candidate in enumerate(candidates, start=1)
        ]


def pointwise_top5(
    question: str, union: list[RetrievalResult], reranker: Any
) -> tuple[list[RetrievalResult], float]:
    started = time.perf_counter()
    ranked = reranker.rerank(question, union)
    latency = getattr(reranker, "last_inference_latency_ms", None)
    if not latency:
        latency = (time.perf_counter() - started) * 1000
    top5 = [
        replace(
            item.result,
            rank=item.reranked_rank,
            score=float(item.reranker_score),
            retrieval_source="pointwise_cross_encoder_top5",
        )
        for item in ranked[:FINAL_TOP_K]
    ]
    return top5, float(latency)


@dataclass
class RetrievalArm:
    name: str
    bm25_query: str | None = None
    extra_bm25_queries: tuple[str, ...] = ()
    sparse: Literal["bm25", "learned", "both"] = "bm25"
    dense_mode: Literal["single", "maxsim"] = "single"
    decompose: bool = False
    hyde_lexical: bool = False


def retrieve_arm(
    corpus: IsolatedCorpus,
    *,
    question: str,
    query_embedding: list[float],
    principal: Principal,
    category: str,
    arm: RetrievalArm,
    dense_depth: int = DENSE_DEPTH,
    lexical_depth: int = BM25_DEPTH,
    union_limit: int = UNION_LIMIT,
) -> dict[str, Any]:
    dense, dense_ms = dense_search(
        corpus, query_embedding, principal, top_k=dense_depth, mode=arm.dense_mode
    )
    lexical_queries = []
    if arm.decompose:
        lexical_queries.extend(decompose_query(question, category=category))
    elif arm.hyde_lexical:
        lexical_queries.append(hyde_document(question))
    elif arm.extra_bm25_queries:
        lexical_queries.extend(arm.extra_bm25_queries)
    else:
        lexical_queries.append(arm.bm25_query or question)
    lexical_lists: list[list[RetrievalResult]] = []
    lexical_ms = 0.0
    for lexical_query in lexical_queries:
        if arm.sparse in {"bm25", "both"}:
            rows, latency = bm25_search(corpus, lexical_query, principal, top_k=lexical_depth)
            lexical_lists.append(rows)
            lexical_ms += latency
        if arm.sparse in {"learned", "both"}:
            rows, latency = learned_sparse_search(
                corpus, lexical_query, principal, top_k=lexical_depth
            )
            lexical_lists.append(rows)
            lexical_ms += latency
    started = time.perf_counter()
    lexical = fuse_lexical_lists(lexical_lists, top_k=lexical_depth)
    fused_all = reciprocal_rank_fusion(dense, lexical, top_k=10_000, rrf_k=RRF_K)
    union = fused_all[:union_limit]
    rrf_ms = (time.perf_counter() - started) * 1000
    by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    unauthorized = [
        item.document_id
        for item in [*dense, *lexical, *union]
        if not by_id[item.chunk_id].authorized_for(principal)
    ]
    if unauthorized:
        raise RuntimeError(f"ACL/tenant leak in isolated retrieval: {unauthorized[:3]}")
    if any(
        "AcmeAI authorized current-version policy document. The policy answers:" in chunk.text
        for chunk in corpus.chunks
    ):
        raise RuntimeError("HyDE template leaked into the evidence corpus")
    return {
        "dense": dense,
        "lexical": lexical,
        "union": union,
        "timing": {
            "dense_ms": dense_ms,
            "lexical_ms": lexical_ms,
            "rrf_ms": rrf_ms,
        },
        "lexical_queries": lexical_queries,
        "hyde_used_as_evidence": False,
        "fused_all": fused_all,
    }


def required_documents_at_k(results: list[RetrievalResult], required: set[str], k: int) -> set[str]:
    return required & {item.document_id for item in results[:k]}


def retrieval_metrics_for_case(
    *,
    required_document_ids: tuple[str, ...],
    expected_answerability: bool,
    union: list[RetrievalResult],
    top5: list[RetrievalResult],
    expected_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    required = set(required_document_ids)
    if not expected_answerability or not required:
        return {
            "recall_at_5": None,
            "recall_at_10": None,
            "recall_at_20": None,
            "recall_at_50": None,
            "top5_evidence_coverage": None,
            "pool_coverage": None,
            "version_correct": 1.0,
        }
    union_docs = [item.document_id for item in union]
    top5_docs = [item.document_id for item in top5]
    version_correct = all(
        any(item.document_id == document_id and item.version == version for item in top5)
        for document_id, version in (expected_versions or {}).items()
    )

    def recall(k: int) -> float:
        return len(required & set(union_docs[:k])) / len(required)

    return {
        "recall_at_5": len(required & set(top5_docs[:5])) / len(required),
        "recall_at_10": recall(10),
        "recall_at_20": recall(20),
        "recall_at_50": recall(50),
        "top5_evidence_coverage": float(required <= set(top5_docs[:5])),
        "pool_coverage": float(required <= set(union_docs)),
        "version_correct": float(version_correct),
    }


def classify_retrieval_failure(
    *,
    expected_answerability: bool,
    required_document_ids: tuple[str, ...],
    union: list[RetrievalResult],
    top5: list[RetrievalResult],
    gold_in_corpus: bool,
) -> str:
    if not expected_answerability:
        return "NONE"
    required = set(required_document_ids)
    if required <= {item.document_id for item in top5[:5]}:
        return "NONE"
    if gold_in_corpus and not required <= {item.document_id for item in union}:
        return "RETRIEVAL_MISS"
    if required <= {item.document_id for item in union}:
        return "RANKING_MISS"
    return "RETRIEVAL_MISS"


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row.get("expected_answerability")]

    def avg(name: str) -> float:
        values = [
            row["metrics"][name] for row in answerable if row["metrics"].get(name) is not None
        ]
        return mean(values) if values else 0.0

    failures = Counter(row.get("failure") or "NONE" for row in rows)
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        categories[str(row.get("category") or "unknown")].append(row)
    return {
        "n": len(rows),
        "answerable": len(answerable),
        "recall_at_5": avg("recall_at_5"),
        "recall_at_10": avg("recall_at_10"),
        "recall_at_20": avg("recall_at_20"),
        "recall_at_50": avg("recall_at_50"),
        "top5_evidence_coverage": avg("top5_evidence_coverage"),
        "pool_coverage": avg("pool_coverage"),
        "version_correctness": avg("version_correct") if answerable else 1.0,
        "retrieval_miss": failures.get("RETRIEVAL_MISS", 0),
        "ranking_miss": failures.get("RANKING_MISS", 0),
        "failures": dict(failures),
        "by_category": {
            category: {
                "n": len(items),
                "top5_evidence_coverage": mean(
                    [
                        item["metrics"]["top5_evidence_coverage"]
                        for item in items
                        if item["metrics"].get("top5_evidence_coverage") is not None
                    ]
                    or [0.0]
                ),
                "recall_at_20": mean(
                    [
                        item["metrics"]["recall_at_20"]
                        for item in items
                        if item["metrics"].get("recall_at_20") is not None
                    ]
                    or [0.0]
                ),
            }
            for category, items in categories.items()
        },
    }


def compare_failure_census(control: list[str], candidate: list[str]) -> dict[str, Any]:
    improved = 0
    worsened = 0
    unchanged = 0
    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"improved": 0, "worsened": 0})
    for left, right in zip(control, candidate, strict=True):
        if left == right:
            unchanged += 1
            continue
        if left != "NONE" and right == "NONE":
            improved += 1
            by_class[left]["improved"] += 1
        elif left == "NONE" and right != "NONE":
            worsened += 1
            by_class[right]["worsened"] += 1
        else:
            by_class[left]["improved"] += 1
            by_class[right]["worsened"] += 1
            improved += 1
            worsened += 1
    return {
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "by_class": dict(by_class),
    }


def apply_selection_policy(
    *,
    name: str,
    control: dict[str, Any],
    candidate: dict[str, Any],
    safety: dict[str, Any],
    added_paid_llm_calls: int,
    complexity: Literal["low", "high"],
    p95_control: float,
    p95_candidate: float,
) -> dict[str, Any]:
    coverage_gain = candidate["top5_evidence_coverage"] - control["top5_evidence_coverage"]
    recall_gain = candidate["recall_at_20"] - control["recall_at_20"]
    latency_ratio = (p95_candidate / p95_control) if p95_control else 1.0
    hard = (
        safety.get("unsupported_answers_delta", 0) <= 0
        and safety.get("acl_safety", 1.0) == 1.0
        and safety.get("tenant_isolation", 1.0) == 1.0
        and safety.get("version_correctness", 1.0) == 1.0
        and safety.get("citation_correctness_delta", 0.0) >= 0.0
        and added_paid_llm_calls <= SELECTION_POLICY["materiality"]["max_added_paid_llm_calls"]
    )
    material = (
        coverage_gain >= SELECTION_POLICY["materiality"]["top5_coverage_gain"]
        or recall_gain >= SELECTION_POLICY["materiality"]["recall_at_20_gain"]
    )
    if complexity == "high":
        material = (
            coverage_gain >= SELECTION_POLICY["materiality"]["complexity_requires_coverage_gain"]
        )
    latency_ok = latency_ratio <= SELECTION_POLICY["materiality"]["max_p95_latency_ratio"]
    accepted = bool(hard and material and latency_ok)
    if not hard:
        verdict = "REJECT"
        reason = "hard safety, cost, or citation gate failed"
    elif not material:
        verdict = "REJECT"
        reason = "quality gain below the frozen materiality bar"
    elif not latency_ok:
        verdict = "REJECT"
        reason = "p95 latency more than doubled"
    else:
        verdict = "ACCEPT CANDIDATE"
        reason = "material quality gain with safety, cost, and latency gates held"
    return {
        "experiment_name": name,
        "verdict": verdict,
        "accepted": accepted,
        "reason": reason,
        "coverage_gain": coverage_gain,
        "recall_at_20_gain": recall_gain,
        "latency_ratio": latency_ratio,
        "added_paid_llm_calls": added_paid_llm_calls,
        "hard_gates": hard,
        "material": material,
    }


def graph_rag_decision(census: dict[str, int]) -> dict[str, Any]:
    retrieval_miss = census.get("RETRIEVAL_MISS", 0)
    ranking_miss = census.get("RANKING_MISS", 0)
    judge_fn = census.get("JUDGE_FALSE_NEGATIVE", 0)
    multi_hop = census.get("MULTI_HOP_FAILURE", 0)
    needed = multi_hop >= 8 and retrieval_miss >= 8
    if needed:
        decision = "Needed"
        reason = "failure data show widespread cross-document relationship misses"
    elif judge_fn >= ranking_miss and retrieval_miss <= 4:
        decision = "Not Needed"
        reason = (
            "dominant residuals are evidence-gate false negatives and Top-5 ranking, "
            "not global entity-graph retrieval"
        )
    else:
        decision = "Inconclusive"
        reason = "multi-hop volume is too small to justify GraphRAG infrastructure"
    return {
        "decision": decision,
        "reason": reason,
        "retrieval_miss": retrieval_miss,
        "ranking_miss": ranking_miss,
        "judge_false_negative": judge_fn,
        "multi_hop_failure": multi_hop,
    }
