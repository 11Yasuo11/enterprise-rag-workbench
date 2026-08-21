# Interview Q&A

High-probability technical questions for Enterprise RAG Workbench.

**Global disclaimer:** Prefer “60/60 on the final V2 regression suite” over any unseen/production-100% wording. Serving is integrated/validated, **not externally deployed**.

---

## 1. Tell me about this project.

**Short:** Enterprise RAG workbench focused on grounded, authorized, version-correct, complete answers with safe abstention — improved by evaluation loops.

**Deeper:** Hybrid retrieval → CE Top-15 → atomic requirements → evidence packets → deterministic support / Luna / Sol → completeness gate → assembler-v3. Web and Eval share `CanonicalRagRuntime`.

**Avoid:** “Production SaaS already serving customers.”

## 2. Why did you build your own RAG pipeline?

**Short:** Off-the-shelf chat-RAG did not enforce ACL, versioning, multi-requirement completeness, or fail-closed abstention as first-class metrics.

**Deeper:** Failures were measured (ranking misses, incomplete multi-doc, injection, incorrect abstentions) and each architecture piece maps to a causal failure class.

**Avoid:** “I reinvented LangChain for fun.”

## 3. Why Dense + BM25?

**Short:** Dense catches paraphrase; BM25 catches exact identifiers, years, and rare tokens.

**Deeper:** Controlled hybrid work beat dense-only on multi-doc coverage in sealed experiments.

**Avoid:** “BM25 is obsolete / dense is always enough.”

## 4. What is RRF?

**Short:** Reciprocal Rank Fusion merges ranked lists using ranks, not raw scores.

**Deeper:** Dense cosine and BM25 scores are not commensurate; RRF avoids brittle score calibration. Configured with **k=60**, then a fused candidate cap ≤30.

**Avoid:** Claiming RRF “learns” optimal weights.

## 5. Why k=60?

**Short:** Standard RRF smoothing constant used in this stack; kept fixed for comparability.

**Deeper:** Interview focus is the hybrid+CE system effect, not a claim that k was exhaustively grid-searched as the hero knob.

**Avoid:** “k=60 is universally optimal.”

## 6. Why use a Cross-Encoder?

**Short:** Bi-encoder retrieval is fast but approximate; CE scores query–chunk pairs to promote required evidence.

**Deeper:** Pool can be nearly complete while early ranks still miss required spans; CE addresses ranking miss after hybrid fusion.

**Avoid:** “CE replaces hybrid retrieval.”

## 7. Why Top-15 instead of Top-5?

**Short:** Top-15 is an **internal evidence pool** for requirements/packets/support — not a chat context dump of five chunks.

**Deeper:** Multi-requirement and temporal cases need a wider working set after CE; final user text is assembled from validated support, not “whatever is in Top-5.”

**Avoid:** Equating Top-15 with “we send 15 chunks to an LLM to write the answer.”

## 8. How do you handle multi-document questions?

**Short:** Decompose into atomic requirements; map evidence per requirement; completeness gate refuses partial assemblies.

**Deeper:** Hybrid+CE raise the chance each required doc reaches the pool; packets make missing R2/R3 visible.

**Avoid:** “The LLM just reads everything and figures it out.”

## 9. What is the Atomic Requirement Plan?

**Short:** Explicit R1/R2/R3… contract derived from the question.

**Deeper:** Turns vague completeness into checkable obligations used by mapping, support, and the assembler.

**Avoid:** Saying requirements come from gold labels at serving time.

## 10. How do you handle document versions?

**Short:** Temporal planning + version resolution before packets; historical year constraints select editions.

**Deeper:** Active-version filters alone are insufficient for “using only 2025 editions…” style questions — V14 work includes calendar-year packet terms for temporal editions.

**Avoid:** “We only ever search the latest version, so versions are solved.”

## 11. How do you enforce ACL?

**Short:** Principal-scoped filters (tenant / ACL / region) so unauthorized chunks never become trusted evidence for packets/Luna/Sol/citations.

**Deeper:** Authorization is enforced on candidates before ranking trust; citation validity checks authorized chunk IDs.

**Avoid:** “The prompt tells the model not to reveal private docs.”

## 12. How do you prevent hallucinations?

**Short:** Deterministic support + authorized citations + completeness gate + no final-generation LLM in production mode.

**Deeper:** Ambiguity goes to verifiers; failure modes prefer abstention over unsupported answers.

**Avoid:** “Temperature=0 fixes hallucinations.”

## 13. What happens when evidence is insufficient?

**Short:** Safe abstention — no substantive answer.

**Deeper:** Triggered by incomplete requirements, auth failure, injection precheck, validation failure, or verifier fail-closed.

**Avoid:** “We always guess the best effort answer.”

## 14. What is Luna?

**Short:** Evidence verifier used when deterministic support is ambiguous.

**Deeper:** Not the final answer writer; request-scoped; fails closed when unavailable.

**Avoid:** “Luna is our chat model.”

## 15. What is Sol?

**Short:** Escalation-only verifier for suspicious abstention / hard ambiguity.

**Deeper:** Not a default path; not the final generator. Hierarchy: Deterministic → Luna → Sol.

**Avoid:** “Sol generates the customer-facing answer.”

## 16. Why don't you use an LLM for final answer generation?

**Short:** Free-form generation increases unsupported wording risk; assembler-v3 builds cited text from validated requirement support.

**Deeper:** LLMs may still appear as **verifiers** (Luna/Sol), not as unconstrained final authors in production mode.

**Avoid:** “LLMs are useless for RAG.”

## 17. How do you handle prompt injection?

**Short:** Hard precheck before answer routing; plus packet-level guards from the V13/V14 safety work.

**Deeper:** Injection cases 055/056/057 abstain on the regression suite; broader holdouts historically showed residual risk — do not claim perfect security.

**Avoid:** “Injection is 100% solved.”

## 18. How did you evaluate the system?

**Short:** Frozen datasets, failure census, causal fix, regression, and fresh holdouts when appropriate.

**Deeper:** Web and Eval share the same runtime so serving claims match measured behavior.

**Avoid:** “We vibe-checked a few prompts.”

## 19. What does 60/60 actually mean?

**Short:** **60/60 on the final V2 regression suite** after V14 — including unsupported=0, incorrect abstention=0, injection 3/3.

**Deeper:** Cases were used during iterative development; this is **not** an unseen generalization score. Earlier historical holdouts (e.g., frozen V2 100-case unseen, V3 Phase-5KR) are separate and lower.

**Avoid:** “100% accuracy on all future enterprise questions.”

## 20. What would you improve next?

**Short:** Deployment review hardening, broader injection holdouts, and fresh holdouts after any new change — without breaking the no-gold runtime contract.

**Deeper:** Operational concerns: monitoring of abstention rates, CE latency, auth audit logs, and cost controls for Luna/Sol.

**Avoid:** Promising a redesign of RRF/CE knobs in the interview without evidence.

---

## Evaluation-driven development story

```text
Evaluate
  → Failure census
  → Find first causal failure
  → Implement general fix
  → Regression test
  → Fresh holdout when appropriate
```

Architecture features were added because **measured failure modes** justified them — not because techniques were popular.

Repository-supported examples of that discipline:

- Hybrid + CE work targeting ranking / coverage failures (see experiments docs / BENCHMARK log).
- Atomic requirements + packets + completeness addressing multi-requirement omissions.
- V13 injection guard + V14 calendar-year packet terms fixing specific regression failures while preserving prior safety.
- V3 research closed when final unseen quality did **not** improve enough to promote (`V3_QUALITY_IMPROVEMENT_NOT_CONFIRMED`).

Latest formal regression artifact: `data/experiments/rag-release-pipeline-v14/` (`FRESH_V2_REGRESSION_PASSED`, 60/60).

When speaking:

1. Name the failure class first.
2. Name the general fix second.
3. Name the regression/holdout evidence third.
4. Never convert a regression score into an unseen claim.
