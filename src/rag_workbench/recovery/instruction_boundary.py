# ruff: noqa: E501
"""Deterministic evidence/instruction boundary.

Retrieved chunks remain untrusted DATA. Model-directed instructions inside them
must never acquire runtime authority. This is not a keyword blacklist: employee-
directed policy language that happens to contain words such as instruction,
assistant, system, ignore, prompt, model, tool, or override is allowed when the
addressee is a human role and the span is not a model-directed construction.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import hashlib
import json
import re
from dataclasses import dataclass

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.recovery.contracts import RecoveryDraft, RecoveryVerification

STAGE_INSTRUCTION_BOUNDARY = "INSTRUCTION_BOUNDARY"
INSTRUCTION_BOUNDARY_VERSION = "evidence-instruction-boundary-v1"
BOUNDARY_PASS = "PASS"
BOUNDARY_FAIL = "FAIL"
UNTRUSTED_INSTRUCTION_EVIDENCE = "UNTRUSTED_INSTRUCTION_EVIDENCE"

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)
ROLE_MARKUP = re.compile(
    r"(?is)<\s*/?\s*(system|assistant|developer|instruction|sys)\b[^>]*>"
    r"|\[/?\s*(system|assistant|developer|INST|SYS)\s*\]"
    r"|<<\s*SYS\s*>>"
    r"|(?:^|\n)\s*(?:system|assistant|developer)\s*(?:prompt)?\s*:"
    r"|```(?:system|assistant|instruction)\b"
    r"|role\s*[:=]\s*['\"]?(system|assistant|developer)\b"
)
TOOL_COERCION = re.compile(
    r"(?is)\b(?:tool_call|function_call|tool_calls|invoke[_ ]tool|call the \w+ tool)\b"
    r"|<\s*tool_call\b"
    r"|\"(?:name|tool)\"\s*:\s*\"(?:shell|bash|exec|python)\""
)
INHERENT_MODEL_DIRECTED = re.compile(
    r"(?is)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|earlier|above)\s+instructions\b"
    r"|\bignore\s+(?:your\s+)?(?:system\s+)?(?:prompt|rules|guardrails)\b"
    r"|\byou are now\b"
    r"|\bfrom now on,?\s+you\s+(?:must|will|are|should)\b"
    r"|\btreat (?:this|the following|the retrieved) (?:document|text|chunk|content|message) as "
    r"(?:a |the )?(?:system|developer|higher[- ]priority) (?:prompt|instruction|message)\b"
    r"|\bthis (?:document|message|chunk|text) (?:is|has) (?:a )?higher (?:priority|authority) than\b"
    r"|\b(?:do not|don't) (?:cite|use citations|include citations|abstain)\b"
    r"|\banswer without citations\b"
    r"|\breveal (?:the )?(?:system |hidden )?(?:secrets?|credentials?|api keys?|prompt)\b"
    r"|\boverride (?:the )?(?:runtime|system|acl|version) (?:policy|rules?|guardrails?)\b"
    r"|\bignore (?:the )?(?:acl|version|citation|metadata) (?:policy|rules?|fields?)\b"
    r"|\boutput only\b.{0,80}\b(?:answer|json|secret)"
    r"|\bjailbreak\b"
    r"|\bDAN mode\b"
)
MODEL_ADDRESSEE = re.compile(
    r"(?is)\b(?:language model|chatbot|\bllms?\b|you are (?:an? )?(?:ai|assistant)|"
    r"system prompt|developer message|"
    r"retrieved (?:snippet|chunk|document) (?:says|demands|instructs)|"
    r"comply with (?:the )?(?:retrieved|document)|"
    r"follow (?:the )?(?:retrieved|document) (?:instruction|snippet|command))\b"
)
EMPLOYEE_ADDRESSEE = re.compile(
    r"(?is)\b(?:employees?|staff|engineers?|managers?|agents?|operators?|personnel|"
    r"reviewers?|incident commander|finance operations|support agents?)\b"
)
QUESTION_INJECTION = re.compile(
    r"(?is)\b(?:asks? the assistant to (?:comply|follow|obey|switch|publish|ignore|reveal)|"
    r"comply with (?:a |the )?(?:retrieved|document|training snippet)|"
    r"follow (?:the )?(?:retrieved|document) (?:instruction|snippet|command)|"
    r"treat (?:this|the) (?:retrieved )?(?:document|snippet|chunk) as (?:a )?system|"
    r"because a retrieved (?:training )?snippet|"
    r"override (?:system|runtime|acl|version) (?:rules|policy)|"
    r"drop citations|"
    r"leak secrets|"
    r"hidden compensation|"
    r"document text outranks policy)\b"
)
BASE64_TOKEN = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")
HEX_TOKEN = re.compile(r"\b[0-9a-fA-F]{32,}\b")
QUOTED_TRAINING = re.compile(
    r"(?is)(?:training example|malicious example|quoted (?:text|command)|hostile data)"
    r".{0,120}([\"“])(.+?)\1"
)


@dataclass(frozen=True)
class BoundaryDecision:
    verdict: str
    code: str | None
    reason: str
    relied_chunk_ids: tuple[str, ...]
    live_instruction_chunk_ids: tuple[str, ...]
    question_injection: bool
    answer_enacts_instruction: bool

    @property
    def passed(self) -> bool:
        return self.verdict == BOUNDARY_PASS


def instruction_boundary_identity() -> str:
    payload = {
        "version": INSTRUCTION_BOUNDARY_VERSION,
        "stage": STAGE_INSTRUCTION_BOUNDARY,
        "policy": "fail_closed_live_model_directed_or_question_injection_or_enactment",
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    return text.translate(ZERO_WIDTH).replace("\xa0", " ")


def _strip_quoted_training(text: str) -> str:
    return QUOTED_TRAINING.sub(" QUOTED_TRAINING_EXAMPLE ", text)


def _decoded_payloads(text: str) -> tuple[str, ...]:
    found: list[str] = []
    for token in BASE64_TOKEN.findall(text):
        padded = token + "=" * ((4 - len(token) % 4) % 4)
        try:
            raw = base64.b64decode(padded, validate=False)
            decoded = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if any(ch.isalpha() for ch in decoded):
            found.append(decoded)
    for token in HEX_TOKEN.findall(text):
        if len(token) % 2:
            continue
        try:
            decoded = bytes.fromhex(token).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if any(ch.isalpha() for ch in decoded):
            found.append(decoded)
    for comment in HTML_COMMENT.findall(text):
        inner = comment.strip()
        if inner:
            found.append(inner)
    compact = re.sub(r"[^A-Za-z]", "", text)
    if 20 <= len(compact) <= 400:
        rotated = codecs.decode(compact, "rot_13")
        if INHERENT_MODEL_DIRECTED.search(rotated) or MODEL_ADDRESSEE.search(rotated):
            found.append(rotated)
    return tuple(dict.fromkeys(found))


def span_is_live_model_directed(text: str, *, _depth: int = 0) -> bool:
    normalized = _strip_quoted_training(_normalize(text))
    if ROLE_MARKUP.search(normalized) or TOOL_COERCION.search(normalized):
        return True
    if INHERENT_MODEL_DIRECTED.search(normalized):
        return True
    if MODEL_ADDRESSEE.search(normalized) and re.search(
        r"(?is)\b(?:ignore|disregard|override|comply|obey|you must|do not cite|"
        r"reveal|follow these instructions|treat this)\b",
        normalized,
    ):
        return True
    if _depth >= 2:
        return False
    for payload in _decoded_payloads(normalized):
        if payload == text:
            continue
        if span_is_live_model_directed(payload, _depth=_depth + 1):
            return True
        if INHERENT_MODEL_DIRECTED.search(payload) or ROLE_MARKUP.search(payload):
            return True
    return False


def question_is_instruction_injection(question: str) -> bool:
    return QUESTION_INJECTION.search(_normalize(question)) is not None


def _instruction_payloads(chunks: tuple[GateEvidence, ...]) -> tuple[str, ...]:
    payloads: list[str] = []
    for chunk in chunks:
        text = _normalize(chunk.text)
        payloads.append(text)
        payloads.extend(_decoded_payloads(text))
        quoted = QUOTED_TRAINING.findall(text)
        payloads.extend(item[1] for item in quoted)
    return tuple(payloads)


def answer_enacts_instruction(answer: str, chunks: tuple[GateEvidence, ...]) -> bool:
    if not answer.strip():
        return False
    folded = _normalize(answer).casefold()
    if INHERENT_MODEL_DIRECTED.search(folded):
        return True
    for payload in _instruction_payloads(chunks):
        live = _strip_quoted_training(payload)
        forced = re.search(
            r"(?is)(?:answer (?:with|that|:)|the (?:only )?answer is|output)\s+(.{8,80})",
            live,
        )
        if forced:
            snippet = _normalize(forced.group(1)).casefold().strip(" .\"'")
            if len(snippet) >= 8 and snippet in folded:
                return True
        for decoded in _decoded_payloads(payload):
            target = _normalize(decoded).casefold()
            if 12 <= len(target) <= 160 and target in folded:
                return True
    return "reveal system secrets" in folded or "answer without citations" in folded


def evaluate_instruction_boundary(
    *,
    question: str,
    chunks: tuple[GateEvidence, ...],
    answer: str | None,
    citation_ids: tuple[str, ...],
    supporting_ids: tuple[str, ...],
) -> BoundaryDecision:
    relied = tuple(dict.fromkeys([*citation_ids, *supporting_ids]))
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    live_ids = tuple(
        chunk.chunk_id for chunk in chunks if span_is_live_model_directed(chunk.text)
    )
    question_injection = question_is_instruction_injection(question)
    enact = answer_enacts_instruction(answer or "", chunks)
    relied_live = tuple(item for item in relied if item in set(live_ids))
    if question_injection:
        return BoundaryDecision(
            BOUNDARY_FAIL,
            UNTRUSTED_INSTRUCTION_EVIDENCE,
            "user question asks the model to follow retrieved instructions",
            relied,
            live_ids,
            True,
            enact,
        )
    if enact:
        return BoundaryDecision(
            BOUNDARY_FAIL,
            UNTRUSTED_INSTRUCTION_EVIDENCE,
            "candidate answer enacts a model-directed instruction from retrieved text",
            relied,
            live_ids,
            False,
            True,
        )
    if relied_live:
        return BoundaryDecision(
            BOUNDARY_FAIL,
            UNTRUSTED_INSTRUCTION_EVIDENCE,
            "candidate answer relies on a chunk containing a live model-directed instruction",
            relied,
            relied_live,
            False,
            False,
        )
    missing = [item for item in relied if item not in by_id]
    if missing:
        return BoundaryDecision(
            BOUNDARY_FAIL,
            UNTRUSTED_INSTRUCTION_EVIDENCE,
            "candidate answer cites a chunk outside the authorized evidence window",
            relied,
            live_ids,
            False,
            False,
        )
    return BoundaryDecision(
        BOUNDARY_PASS,
        None,
        "relied-upon evidence is not live model-directed instruction",
        relied,
        live_ids,
        False,
        False,
    )


def apply_instruction_boundary(
    *,
    question: str,
    chunks: tuple[GateEvidence, ...],
    draft: RecoveryDraft,
    verification: RecoveryVerification,
) -> BoundaryDecision:
    citations = tuple(dict.fromkeys(item.chunk_id for item in draft.citations))
    supporting = tuple(
        dict.fromkeys(
            chunk_id
            for item in verification.claim_results
            for chunk_id in item.supporting_chunk_ids
        )
    )
    return evaluate_instruction_boundary(
        question=question,
        chunks=chunks,
        answer=draft.candidate_answer,
        citation_ids=citations,
        supporting_ids=supporting,
    )
