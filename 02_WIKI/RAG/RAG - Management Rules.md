# RAG - Management Rules

RAG Knowledge Base の管理ルール。新規ファイル作成・整理時はこの Note を参照する。

Hub: [[RAG]]

---

## Folder Layout

```text
02_WIKI/RAG/
├── RAG.md                    ← 入口 Hub / MOC（必ず更新）
├── RAG - Management Rules.md ← 本 Note
├── Concepts/                 ← 恒久的 Concept / Reference
├── Architecture/             ← Production / Pipeline 設計
├── Research/                 ← V1/V2/V3, Phase, Experiment 記録
└── Interview/                ← 復習・面接準備（Research と混同しない）
```

原則: **3 階層以内**。深いサブフォルダは作らない。

---

## 新しい Concept Note をどこに置くか

- 場所: `02_WIKI/RAG/Concepts/`
- 命名: `RAG - <Topic>.md`（例: `RAG - Hybrid Retrieval.md`）
- 1 テーマ 1 Canonical Note。詳細実験は Research/ へリンク
- 数値・結論の正本は [`BENCHMARK.md`](../../BENCHMARK.md) または [`README.md`](../../README.md)。Concept Note は要約 + リンクのみ

---

## 新しい Experiment をどこに置くか

- 場所: `02_WIKI/RAG/Research/`
- 命名:
  - Version: `RAG V2 - Phase 5B Experiment.md`
  - Final: `RAG V2 - Final Evaluation.md`
- **Research Record と Study Note を分離する**
  - Research → `Research/`（凍結数値・Decision を含む）
  - Study / Interview → `Interview/`（理解・復習用。数値の正本にしない）

---

## Version / Phase の命名方法

| 種別 | 形式 | 例 |
|---|---|---|
| Version | `RAG V{n} - <Title>.md` | `RAG V2 - Final Evaluation.md` |
| Phase | `RAG V{n} - Phase {id} <Title>.md` | `RAG V2 - Phase 1 Document Diversity.md` |
| Post-version research | `RAG V{n} - Post-V2 Quality AB.md` | |

- V1 / V2 の凍結記録は **上書きしない**。V3 は新セクションとして追加
- Phase 名・Version 名は実験記録と一致させる

---

## Canonical Note の決め方

各主要テーマについて **「まずこの Note を見る」** を 1 つ決める。

| Theme | Canonical Note |
|---|---|
| RAG 全体 | [[RAG]] |
| Production | [[RAG - Production Architecture V2]] |
| Hybrid Retrieval | [[RAG - Hybrid Retrieval]] |
| Reranking | [[RAG - Cross Encoder Reranking]] |
| Evaluation | [[RAG - Evaluation Metrics]] |
| Abstention | [[RAG - Abstention Strategy]] |
| V2 Final | [[RAG V2 - Final Evaluation]] |

- 既存の repo 正本（`README.md`, `BENCHMARK.md`, `docs/*`）がある場合、**新しい Source of Truth を作らない**
- Wiki Note はナビゲーション + 要約。正本へのリンクを必ず含める

---

## いつ Merge するか

| 状況 | Action |
|---|---|
| 完全同一（数値・結論・本文すべて一致） | MERGE 候補 |
| 部分重複（一方が superset） | Canonical に整理し、重複側は Archive または redirect リンク |
| 同テーマだが目的が異なる（Concept vs Experiment vs Interview） | **別ファイル維持** |
| 旧 Version の Research 記録 | MERGE しない。Archive も削除しない |

Merge 前に必ず内容比較。実験結果・Failure・Decision は失わない。

---

## いつ Archive するか

Archive 対象:

- 現在利用しない
- Knowledge として Canonical ではない
- Historical record としてのみ必要

Archive しても [[RAG]] Hub から辿れるリンクを維持する。

**削除はしない。** Archive = 参照導線の整理のみ。

---

## Study Note と Research Record の分離

| | Research Record | Study Note |
|---|---|---|
| 場所 | `Research/` | `Interview/` |
| 内容 | 凍結数値、Decision、Failure | 概念説明、Q&A、復習 |
| 正本 | `BENCHMARK.md` 等 | なし（Research へリンク） |
| frontmatter `document_type` | `Experiment`, `Evaluation`, `Decision` | `Guide`, `Interview Prep` |

Interview Note に独自の数値を書かない。必ず Research / BENCHMARK へリンク。

---

## 避けるファイル名

```text
rag-notes.md, rag2.md, new-rag.md, test.md, research.md,
memo.md, phase.md, final.md, final2.md, latest.md
```

---

## 内部リンク

- Wiki 内: `[[RAG - Hybrid Retrieval]]`
- Repo 正本: `` [`BENCHMARK.md`](../../BENCHMARK.md) ``
- rename / move 後は broken link を確認する

---

## Related

- [[RAG]]
- [[RAG - Inventory Audit]]
