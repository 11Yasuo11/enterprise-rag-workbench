# ruff: noqa: E501
"""
QUESTION_INJECTION_GUARD_V2

Deterministic question-level injection detector used for Phase 5D.

Policy:
- If the *user-visible question* is injection-style and asks the system to
  follow malicious / higher-priority / embedded instructions, the system must
  SAFE-ABSTAIN (never return refusal-style substantive answers for these
  should_abstain=true benchmark cases).

This detector is intentionally generic:
- It does NOT special-case query_ids or benchmark sentences.
- It relies on broad structural/semantic patterns (higher priority, follow
  embedded instructions, override system/assistant rules, reveal secrets,
  ignore safety/guardrails, answer without evidence).

Benign control:
- Questions that *analyze/ask about* a phrase in policy text should not be
  treated as injection simply because the phrase appears in quotes.
"""

from __future__ import annotations

import re

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def _normalize(text: str) -> str:
    return text.translate(_ZERO_WIDTH).replace("\xa0", " ").strip()


# Benign "meta-question" pattern: asking what a policy/guide says about a phrase.
_META_POLICY_PHRASE_Q = re.compile(r"(?is)\bwhat does\b.{0,60}\bpolicy\b.{0,60}\bphrase\b")


# Core intent patterns: presence of (imperative verbs) + (authority override/
# instruction following) + (injection context like retrieved document/quoted
# training snippet/system secrets).
_HIGHER_PRIORITY_FOLLOW = re.compile(
    r"(?is)\b(higher[- ]priority|higher\s+priority)\b.{0,120}\b(follow|obey)\b"
)

_TREAT_AS_SYSTEM_OR_AUTHORITATIVE = re.compile(
    r"(?is)\b(treat|consider)\b.{0,120}\b(retrieved|document|snippet|passage|quoted|training|evidence)\b.{0,80}\b(as\s+)?(system|developer|assistant|authoritative|higher[- ]priority)\b"
)

_OVERRIDE_RULES = re.compile(
    r"(?is)\b(ignore|disregard|override(?:s)?|override\s+all|bypass)\b.{0,120}\b(previous|prior|earlier|system|assistant|rules|guardrails|safety|policies?)\b"
)

_FOLLOW_RETRIEVED_INSTRUCTIONS = re.compile(
    r"(?is)\b(follow|obey|comply with|carry out|execute)\b.{0,160}\b(retrieved|document|snippet|passage|quoted|training|instruction|embedded)\b"
)

_REVEAL_SECRETS = re.compile(
    r"(?is)\b(reveal|expose|dump|show)\b.{0,120}\b(system\s+secrets?|secrets?|credentials?|api\s+keys?|confidential|private|hidden)\b"
)

_DROP_CITATIONS_OR_ANSWER_WITHOUT_EVIDENCE = re.compile(
    r"(?is)\b(drop|ignore)\b.{0,80}\b(citations?|evidence)\b"
    r"|\b(answer|respond|produce|output)\b.{0,120}\b(without\s+citations|no\s+citations)\b"
)

_IGNORE_SAFETY_GUARDRAILS = re.compile(
    r"(?is)\b(ignore|bypass)\b.{0,120}\b(safety|guardrails?|restrictions?|guardrail|policy|policies?)\b"
)

_ANSWER_WITHOUT_EVIDENCE = re.compile(
    r"(?is)\b(answer|respond)\b.{0,160}\b(despite|even\s+though)\b.{0,120}\b(no\s+evidence|lack\s+of\s+evidence|without\s+evidence|unknown evidence)\b"
)


def is_question_injection_v2(question: str) -> bool:
    """
    Returns True iff the question is injection-style under QUESTION_INJECTION_GUARD_V2.
    """
    q = _normalize(question)
    q_lower = q.casefold()

    # Benign meta-question: "what does the policy say about the phrase ..."
    if _META_POLICY_PHRASE_Q.search(q_lower):
        return False

    # Higher-priority instruction-following
    if _HIGHER_PRIORITY_FOLLOW.search(q):
        return True

    # Explicitly treating untrusted retrieved text as system/authoritative
    if _TREAT_AS_SYSTEM_OR_AUTHORITATIVE.search(q):
        return True

    # Override / ignore system/assistant rules
    if _OVERRIDE_RULES.search(q) and (
        "assistant" in q_lower
        or "system" in q_lower
        or "rules" in q_lower
        or "guardrails" in q_lower
        or "safety" in q_lower
        or "policy" in q_lower
    ):
        return True

    # Follow/obey/carry out instructions from retrieved/quoted content
    if _FOLLOW_RETRIEVED_INSTRUCTIONS.search(q):
        return True

    # Secret/confidential exfil instructions
    if _REVEAL_SECRETS.search(q):
        return True

    # Citation suppression often accompanies instruction-following injection
    if _DROP_CITATIONS_OR_ANSWER_WITHOUT_EVIDENCE.search(q):
        return True

    # Bypass safety/guardrails
    if _IGNORE_SAFETY_GUARDRAILS.search(q):
        return True

    # Answer despite lack of evidence
    return bool(_ANSWER_WITHOUT_EVIDENCE.search(q))

