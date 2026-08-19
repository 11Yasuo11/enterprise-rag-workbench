# ruff: noqa: E501, SIM102, SIM103
"""
QUESTION_INJECTION_GUARD_V3

Intent-focused question-level injection detector.
It detects instruction-hierarchy override and hostile compliance intent
without relying on benchmark sentence matching.
"""

from __future__ import annotations

import re

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def _normalize(text: str) -> str:
    return text.translate(_ZERO_WIDTH).replace("\xa0", " ").strip()


_BENIGN_SECURITY_ANALYSIS = [
    re.compile(r"(?is)\bwhat does\b.{0,80}\bpolicy\b.{0,80}\b(?:say|mean|define)\b"),
    re.compile(r"(?is)\b(?:explain|analyze|describe|compare)\b.{0,120}\bprompt injection\b"),
    re.compile(r"(?is)\bhow (?:should|does) (?:the|our) (?:system|assistant|policy)\b.{0,120}\b(?:handle|detect|block)\b"),
    re.compile(r"(?is)\bsecurity (?:review|analysis|training)\b"),
]

_AUTHORITY_OVERRIDE = re.compile(
    r"(?is)\b(higher[- ]priority|override|disregard|ignore|bypass)\b.{0,140}\b(system|assistant|rules?|policy|guardrails?|restrictions?)\b"
)
_FORCED_COMPLIANCE = re.compile(
    r"(?is)\b(comply|follow|obey|execute|carry out)\b.{0,180}\b(command|instruction|directive|message)\b"
)
_UNTRUSTED_SOURCE_CONTEXT = re.compile(
    r"(?is)\b(retrieved|document|passage|quote[sd]?|quoted|snippet|embedded|training text|source text)\b"
)
_ROLE_OVERRIDE = re.compile(
    r"(?is)\bpretend you are\b.{0,80}\b(admin|administrator|system|developer)\b"
)
_SECRETS_EXFIL = re.compile(
    r"(?is)\b(reveal|dump|expose|print|show)\b.{0,120}\b(system prompt|secret|credentials?|api keys?|confidential)\b"
)
_UNGROUNDED_OUTPUT = re.compile(
    r"(?is)\b(answer|respond|output)\b.{0,120}\bwithout\b.{0,80}\b(citations?|evidence)\b"
)


def _is_benign_analysis(q: str) -> bool:
    ql = q.casefold()
    if any(p.search(ql) for p in _BENIGN_SECURITY_ANALYSIS) and not _FORCED_COMPLIANCE.search(
        ql
    ):
        # Benign if it asks to analyze policy behavior and does not carry
        # imperative execution language against untrusted instructions.
        return True
    return False


def is_question_injection_v3(question: str) -> bool:
    q = _normalize(question)
    ql = q.casefold()

    if _is_benign_analysis(q):
        return False

    if _ROLE_OVERRIDE.search(ql):
        return True
    if _SECRETS_EXFIL.search(ql):
        return True
    if _UNGROUNDED_OUTPUT.search(ql):
        return True

    # Core hierarchy-override attack: authority override + compliance intent.
    if _AUTHORITY_OVERRIDE.search(ql) and (
        _FORCED_COMPLIANCE.search(ql) or _UNTRUSTED_SOURCE_CONTEXT.search(ql)
    ):
        return True

    # Untrusted-source instruction following even without explicit "with".
    return bool(_FORCED_COMPLIANCE.search(ql) and _UNTRUSTED_SOURCE_CONTEXT.search(ql))

