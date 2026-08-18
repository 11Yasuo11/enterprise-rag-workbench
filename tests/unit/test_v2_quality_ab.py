from pathlib import Path

from rag_workbench.experiments.v2_final_benchmark_report import V2_FINAL_BENCHMARK_HEADING
from rag_workbench.experiments.v2_quality_ab_core import (
    IdentityReranker,
    IsolatedChunk,
    RetrievalArm,
    apply_selection_policy,
    build_isolated_corpus,
    decompose_query,
    expand_synthetic_corpus,
    gold_chunk,
    graph_rag_decision,
    hyde_document,
    is_holdout_case,
    l2_normalize,
    multi_queries,
    retrieve_arm,
    rewrite_query,
)
from rag_workbench.experiments.v2_quality_ab_report import QUALITY_AB_BENCHMARK_HEADING
from rag_workbench.experiments.v2_quality_recovery import (
    FROZEN_V1_ARCHITECTURE_HASH,
    V1_BENCHMARK_HEADING,
)
from rag_workbench.security.permissions import Principal

EMPLOYEE = Principal("eval", "acmeai", frozenset({"employees"}))
FOREIGN = Principal("spy", "globex", frozenset({"employees"}))


def _gold(n: int = 3) -> list[IsolatedChunk]:
    chunks = []
    for index in range(n):
        vector = l2_normalize([float(index + 1)] * 64)
        chunks.append(
            gold_chunk(
                chunk_id=f"gold-{index}",
                document_id=f"doc-{index}",
                document_version_id=f"ver-{index}",
                text=f"ENG-DEP-17 production releases occur on Tuesdays. Rule {index}.",
                embedding=vector,
                tenant_id="acmeai",
                visibility="public",
                permission_groups=(),
                version="2026",
                is_active=True,
                source="gold.md",
                title="Gold",
            )
        )
    return chunks


def test_holdout_split_is_frozen_and_not_empty() -> None:
    ids = [f"fv2_single_{index:02d}" for index in range(1, 21)]
    holdout = [item for item in ids if is_holdout_case(item)]
    research = [item for item in ids if not is_holdout_case(item)]
    assert holdout
    assert research
    assert set(holdout).isdisjoint(research)
    assert is_holdout_case("fv2_single_01") == is_holdout_case("fv2_single_01")


def test_synthetic_corpus_is_labeled_and_not_copied() -> None:
    gold = _gold()
    expanded = expand_synthetic_corpus(gold, 12)
    synthetic = [item for item in expanded if item.origin != "GOLD_AUTHORIZED_CORPUS"]
    assert len(expanded) == 12
    assert all("SYNTHETIC / STRESS TEST" in item.text for item in synthetic)
    assert len({item.text for item in synthetic}) == len(synthetic)
    assert not any(gold[0].text == item.text for item in synthetic)
    kinds = {item.distractor_kind for item in synthetic}
    assert "hard_negative" in kinds
    assert "different_tenant" in kinds
    assert "old_version" in kinds


def test_acl_and_tenant_isolation_on_isolated_corpus() -> None:
    gold = _gold()
    expanded = expand_synthetic_corpus(gold, 24)
    corpus = build_isolated_corpus(expanded, label="test")
    embedding = gold[0].embedding
    retrieved = retrieve_arm(
        corpus,
        question="ENG-DEP-17 production releases",
        query_embedding=embedding,
        principal=EMPLOYEE,
        category="single_document",
        arm=RetrievalArm(name="raw"),
    )
    tenants = {
        item.document_id.startswith("synthetic-different_tenant") for item in retrieved["union"]
    }
    assert True not in tenants or all(
        not item.document_id.startswith("synthetic-different_tenant") for item in retrieved["union"]
    )
    foreign = retrieve_arm(
        corpus,
        question="ENG-DEP-17 production releases",
        query_embedding=embedding,
        principal=FOREIGN,
        category="single_document",
        arm=RetrievalArm(name="raw"),
    )
    assert all(not item.document_id.startswith("doc-") for item in foreign["union"])


def test_hyde_is_query_only_and_not_evidence() -> None:
    gold = _gold()
    corpus = build_isolated_corpus(gold, label="gold")
    hypo = hyde_document("When are production releases?")
    assert "authorized current-version policy" in hypo
    retrieved = retrieve_arm(
        corpus,
        question="When are production releases?",
        query_embedding=gold[0].embedding,
        principal=EMPLOYEE,
        category="single_document",
        arm=RetrievalArm(name="hyde", hyde_lexical=True),
    )
    assert retrieved["hyde_used_as_evidence"] is False
    assert all(item.origin != "HYDE" for item in corpus.chunks)
    assert all(hypo not in item.text for item in corpus.chunks)


def test_decomposition_skips_simple_facts() -> None:
    query = "When are production releases?"
    assert decompose_query(query, category="single_document") == [query]
    assert decompose_query(query, category="exact_identifier") == [query]
    complex_query = "What is the Atlas API identifier and the retention deletion workflow?"
    parts = decompose_query(complex_query, category="multidoc_two")
    assert len(parts) >= 2


def test_rewrite_and_multi_query_keep_the_raw_query() -> None:
    raw = "退職するとき何日前？ ENG-DEP-17 remote"
    rewritten = rewrite_query(raw)
    assert raw in rewritten or "ENG-DEP-17" in rewritten or "remote" in rewritten.casefold()
    queries = multi_queries(raw)
    assert raw in queries
    assert len(queries) >= 2


def test_selection_policy_rejects_tiny_gains_and_complexity() -> None:
    control = {"top5_evidence_coverage": 0.80, "recall_at_20": 0.90}
    tiny = {"top5_evidence_coverage": 0.801, "recall_at_20": 0.901}
    safety = {
        "unsupported_answers_delta": 0,
        "acl_safety": 1.0,
        "tenant_isolation": 1.0,
        "version_correctness": 1.0,
        "citation_correctness_delta": 0.0,
    }
    rejected = apply_selection_policy(
        name="tiny",
        control=control,
        candidate=tiny,
        safety=safety,
        added_paid_llm_calls=0,
        complexity="low",
        p95_control=10.0,
        p95_candidate=11.0,
    )
    assert rejected["verdict"] == "REJECT"
    unsafe = apply_selection_policy(
        name="unsafe",
        control=control,
        candidate={"top5_evidence_coverage": 0.99, "recall_at_20": 0.99},
        safety={**safety, "unsupported_answers_delta": 1},
        added_paid_llm_calls=0,
        complexity="low",
        p95_control=10.0,
        p95_candidate=11.0,
    )
    assert unsafe["verdict"] == "REJECT"
    paid = apply_selection_policy(
        name="paid",
        control=control,
        candidate={"top5_evidence_coverage": 0.99, "recall_at_20": 0.99},
        safety=safety,
        added_paid_llm_calls=12,
        complexity="low",
        p95_control=10.0,
        p95_candidate=11.0,
    )
    assert paid["verdict"] == "REJECT"
    heavy = apply_selection_policy(
        name="splade",
        control=control,
        candidate={"top5_evidence_coverage": 0.86, "recall_at_20": 0.96},
        safety=safety,
        added_paid_llm_calls=0,
        complexity="high",
        p95_control=10.0,
        p95_candidate=12.0,
    )
    assert heavy["verdict"] == "REJECT"


def test_graphrag_not_needed_when_judge_fn_dominates() -> None:
    decision = graph_rag_decision(
        {
            "RETRIEVAL_MISS": 2,
            "RANKING_MISS": 12,
            "JUDGE_FALSE_NEGATIVE": 17,
            "MULTI_HOP_FAILURE": 3,
        }
    )
    assert decision["decision"] == "Not Needed"


def test_identity_reranker_preserves_order() -> None:
    gold = _gold()
    corpus = build_isolated_corpus(gold, label="gold")
    retrieved = retrieve_arm(
        corpus,
        question="ENG-DEP-17",
        query_embedding=gold[0].embedding,
        principal=EMPLOYEE,
        category="exact_identifier",
        arm=RetrievalArm(name="raw"),
    )
    ranked = IdentityReranker().rerank("ENG-DEP-17", retrieved["union"])
    assert [item.result.chunk_id for item in ranked] == [
        item.chunk_id for item in retrieved["union"]
    ]


def test_v1_and_v2_final_headings_remain_unique() -> None:
    text = Path("BENCHMARK.md").read_text()
    assert text.count(V1_BENCHMARK_HEADING) == 1
    assert FROZEN_V1_ARCHITECTURE_HASH in text
    assert text.count(V2_FINAL_BENCHMARK_HEADING) == 1
    assert text.count(QUALITY_AB_BENCHMARK_HEADING) == 1
    assert "NOT_EXECUTED" in text
    assert "REJECTED PROXY" in text
    assert "NOT NEEDED BY CURRENT FAILURE DATA" in text
    assert "SYNTHETIC / STRESS TEST" in text
    assert "RETRIEVAL_SCALABILITY_UNDER_HARD_NEGATIVES" in text
