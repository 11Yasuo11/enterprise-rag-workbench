# RAG - Inventory Audit

Phase 1 監査結果。2026-08-19 時点。

> **Note:** 本リポジトリは Enterprise RAG Workbench コードベース。Obsidian Vault 全体はこの workspace 外。RAG Knowledge の正本は repo 内 Markdown 4 件 + 新規 Wiki 32 件。

---

## Summary

| Category | Count |
|---|---:|
| Pre-existing RAG markdown（repo 正本） | 4 |
| Synthetic corpus（RAG KB 対象外） | 17 |
| New Wiki notes created | 32 |
| Files moved | 0 |
| Files merged | 0 |
| Files archived | 0 |

---

## Pre-Existing RAG Files

### README.md

| Field | Value |
|---|---|
| Current Path | `/README.md` |
| Main Topic | Production V2 overview, architecture, metrics |
| Document Type | Architecture + Guide |
| Version / Phase | V2 Production |
| Currentness | Current — Production SOT |
| Duplicate Candidates | Partial overlap with wiki Architecture notes |
| Related Files | BENCHMARK.md, docs/*, 02_WIKI/RAG/* |
| Recommended Action | **KEEP**（canonical SOT, do not move） |

### BENCHMARK.md

| Field | Value |
|---|---|
| Current Path | `/BENCHMARK.md` |
| Main Topic | Full measured research log（206KB） |
| Document Type | Research Log + Evaluation |
| Version / Phase | V1, V2, V3 all phases |
| Currentness | Historical frozen records |
| Duplicate Candidates | Partial overlap with Research/ wiki notes |
| Related Files | docs/EXPERIMENTS.md, 02_WIKI/RAG/Research/* |
| Recommended Action | **KEEP**（canonical full log, do not split） |

### docs/ENGINEERING_DECISIONS.md

| Field | Value |
|---|---|
| Current Path | `/docs/ENGINEERING_DECISIONS.md` |
| Main Topic | Why each design decision was made |
| Document Type | Decision |
| Version / Phase | V2 frozen |
| Currentness | Current |
| Duplicate Candidates | Partial overlap with Concepts/ |
| Related Files | docs/EXPERIMENTS.md, README.md |
| Recommended Action | **KEEP** |

### docs/EXPERIMENTS.md

| Field | Value |
|---|---|
| Current Path | `/docs/EXPERIMENTS.md` |
| Main Topic | Experiment chronology（13 sections） |
| Document Type | Research Log |
| Version / Phase | V1–V3 |
| Currentness | Current chronology |
| Duplicate Candidates | Partial overlap with Research/ wiki notes |
| Related Files | BENCHMARK.md, 02_WIKI/RAG/Research/* |
| Recommended Action | **KEEP** |

---

## Non-RAG Markdown（Corpus — Out of Scope）

`data/synthetic_company/*.md`（17 files）— searchable test corpus, not RAG knowledge. **KEEP in place.**

---

## Duplicate Classification

| Pair | Classification | Action |
|---|---|---|
| README ↔ RAG - Production Architecture V2 | Partial overlap, different purpose | KEEP both; wiki links to README as SOT |
| BENCHMARK ↔ Research/ notes | Partial overlap; BENCHMARK is superset | KEEP both; Research notes are index/summary |
| ENGINEERING_DECISIONS ↔ Concepts/ | Same theme, different purpose（Decision vs Concept） | KEEP both |
| EXPERIMENTS ↔ Research/ | Same theme, different purpose（Chronology vs Phase index） | KEEP both |

No complete duplicates found. No merges performed.

---

## New Wiki Structure

```text
02_WIKI/RAG/
├── RAG.md                              Hub / MOC
├── RAG - Management Rules.md
├── RAG - Inventory Audit.md            ← 本 Note
├── Concepts/                           14 notes
├── Architecture/                       6 notes
├── Research/                           10 notes
└── Interview/                          1 note
```

---

## Related

- [[RAG]]
- [[RAG - Management Rules]]
