# Interview Technical Explanation

## 2-minute explanation (English)

This project is an enterprise RAG workbench for **grounded** answers over a synthetic company corpus. The hard problem is not “chat with documents” — it is answering only when evidence is authorized, version-correct, complete across requirements, and citable — otherwise **abstain**.

**Vector search alone is not enough.** Dense retrieval is strong on paraphrase but weak on exact codes, dates, and rare lexical anchors. Multi-part questions also need more than a single nearest-neighbor list.

So the system uses **hybrid retrieval**: Dense Top-20 + BM25 Top-20, fused with **RRF (k=60)**, capped to ≤30 candidates. A local **Cross-Encoder** then re-ranks that pool into an internal **Top-15 evidence pool**. Top-15 is a working set for planning and support checks — not a user-facing “dump five chunks into a prompt” design.

Before trust, the runtime enforces **ACL / tenant / region** and a **prompt-injection precheck**. Temporal planning selects the right **document editions**. The question is decomposed into an **Atomic Requirement Plan** (R1, R2, …). **CanonicalEvidenceMapping** and **Requirement Evidence Packets** bind spans to requirements with identity/version/auth checks.

**Deterministic Support** answers when evidence literally supports the requirements — no generative model required. If support is ambiguous, **Luna** verifies evidence; **Sol** escalates only for suspicious abstention cases. Final answer text is produced by **deterministic-requirement-assembler-v3**, after local validation and a **Universal Completeness Gate**. Insufficient, unauthorized, or injected queries yield **safe abstention**.

Hallucinations are reduced by construction: no free-form final-generation LLM in production mode, citation authorization, completeness refusal, and fail-closed external verifiers.

Web `/rag/query` and eval `/eval/run` share **CanonicalRagRuntime**. Latest scored result: **60/60 on the final V2 regression suite** (regression after iterative fixes — not unseen 100%). Architecture is integrated and validated; **not externally deployed**.

---

## 2分説明（日本語・学習用）

このプロジェクトは、企業向け RAG を「チャット」ではなく、**根拠付き・認可付き・版正しい・要件完備・安全な棄権**まで含めて設計したワークベンチです。

ベクトル検索だけでは不十分です。意味近い文書は取れても、政策 ID・年号・固有フレーズを落としやすく、複数要件の抜け漏れも検知しにくいからです。そこで Dense Top-20 と BM25 Top-20 を **RRF (k=60)** で融合し、候補を ≤30 に抑え、**Cross-Encoder** で内部証拠プール **Top-15** を作ります。

その前段で ACL / テナント / リージョンとプロンプトインジェクション検査を行い、時間・版の計画で正しい edition を選びます。質問は **Atomic Requirement** に分解し、**CanonicalEvidenceMapping** と要件パケットで証拠を紐づけます。リテラルに支えられるなら **Deterministic Support**、曖昧なら **Luna**、疑わしい棄権のみ **Sol** へエスカレーション。最終文は **assembler-v3** が組み立て、完備ゲートを通らない場合は棄権します。

最終回答用の自由生成 LLM は本番構成では無効です。Web と Eval は同じ `CanonicalRagRuntime` です。直近の公式スコアは **V2 回帰 60/60**（反復後の回帰セットであり、未見 100% ではありません）。本番サービング構成は統合・検証済みですが、**外部デプロイはしていません**。

---

## Component cheat sheet

| Component | What it does | Why it was added | Failure it fixed |
|---|---|---|---|
| Temporal Planning | Interprets year/edition constraints | Historical vs current policy questions | Wrong-edition answers |
| Question Planning | Structures the query for retrieval/support | Stable planning before ranking | Ad-hoc prompt-only planning |
| Dense Retrieval | Semantic Top-20 | Paraphrase / meaning match | Keyword-only misses |
| BM25 | Lexical Top-20 | Exact IDs, dates, rare tokens | Dense misses lexical anchors |
| RRF (k=60) | Fuses ranked lists without score calibration | Hybrid merge that is stable/cheap | Incompatible dense vs BM25 scores |
| Cross-Encoder | Pointwise query–chunk re-rank | Promote required spans into working set | Pool-complete / top-incomplete misses |
| Top-15 | Internal evidence pool | Room for multi-req / temporal evidence | Too-narrow Top-5 working windows |
| Version / Auth Validation | Keep only allowed, correct editions | Compliance + tenancy | Unauthorized or obsolete evidence |
| Atomic Requirement Plan | R1/R2/R3… contracts | Detect incompleteness | Multi-doc partial answers |
| CanonicalEvidenceMapping | Bind spans ↔ requirements | Auditable support | Untethered “vibes-based” citations |
| Requirement Evidence Packets | Per-requirement evidence bundles | Localize support checks | Global bag-of-chunks confusion |
| Deterministic Support | Answer when literally supported | Avoid unnecessary LLM paths | False “need a model” routing |
| Luna | Evidence verifier on ambiguity | Cheap/precise ambiguity check | Over-abstain or under-verify |
| Sol | Escalation-only verifier | Suspicious abstention review | Missing second opinion on hard cases |
| Local Validation | Post-route consistency checks | Catch assembly/auth bugs | Invalid citation / support slips |
| Universal Completeness Gate | Refuse incomplete assemblies | Multi-requirement honesty | Partial answers sold as complete |
| assembler-v3 | Deterministic final text + cites | No free-form final LLM | Generative hallucination risk |
| Safe Abstention | No substantive answer when unsafe/incomplete | Enterprise fail-closed | Unsupported / injection / ACL leaks |
