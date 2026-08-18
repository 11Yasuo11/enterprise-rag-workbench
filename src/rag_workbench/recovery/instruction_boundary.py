# ruff: noqa: E501
"""Deterministic untrusted-instruction boundary for V3 Phase-2 recovery.

Retrieved text is data. This gate never changes generate-verify-draft-v1 or
generate-verify-claim-verifier-v1. It runs after claim verification and
completeness, using document trust metadata, sentence speech-act structure,
and question illocution. It is not a naive keyword blocker.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Literal

from rag_workbench.answerability.base import GateEvidence
from rag_workbench.recovery.runtime import RecoveryOutcome

BOUNDARY_VERSION = "untrusted-instruction-boundary-v1"
SAFE_EVIDENCE = "SAFE_EVIDENCE"
UNTRUSTED_INSTRUCTION = "UNTRUSTED_INSTRUCTION"
UNCERTAIN = "UNCERTAIN"
TYPED_FAILURE = "UNTRUSTED_INSTRUCTION_BOUNDARY"

TrustClass = Literal[
    "AUTHORITATIVE_BUSINESS",
    "MIXED_QUOTED_TRAINING",
    "UNTRUSTED_ADVERSARIAL",
]
SpeechRole = Literal[
    "FACTUAL_ASSERTION",
    "POLICY_CONTENT",
    "PROCEDURAL_CONTENT",
    "EMPLOYEE_DIRECTED_PROCEDURE",
    "MODEL_DIRECTED_INSTRUCTION",
    "QUOTED_HOSTILE_EXAMPLE",
    "UNKNOWN",
]
QuestionIllocution = Literal[
    "FACTUAL_INFORMATION_SEEKING",
    "CATALOG_IDENTIFIER_SEEKING",
    "EMPLOYEE_PROCEDURE_SEEKING",
    "MODEL_COMPLIANCE_REQUEST",
    "UNKNOWN",
]
BoundaryVerdict = Literal["SAFE_EVIDENCE", "UNTRUSTED_INSTRUCTION", "UNCERTAIN"]

TOKEN = re.compile(r"[a-z0-9]+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
BASE64_TOKEN = re.compile(r"\b[A-Za-z0-9+/]{32,}={0,2}\b")
HEX_TOKEN = re.compile(r"\b(?:[0-9a-fA-F]{2}){16,}\b")
IDENTIFIER_TOKEN = re.compile(r"\b[A-Z]{2,10}-[A-Z0-9]{2,12}\b")

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "from",
        "with",
        "is",
        "are",
        "be",
        "as",
        "by",
        "at",
        "this",
        "that",
        "it",
        "its",
        "if",
        "not",
        "no",
        "must",
        "may",
        "should",
        "can",
        "will",
        "under",
        "current",
        "policy",
        "document",
        "retrieved",
        "snippet",
        "text",
        "brief",
        "harbor",
        "only",
        "any",
        "all",
        "previous",
        "quoted",
        "hostile",
        "untrusted",
        "data",
        "example",
        "training",
    }
)

# Frozen research metadata: document_id → trust class. Existing V2 files are
# unchanged; this overlay does not rewrite corpus bytes.
DOCUMENT_TRUST_CLASS: dict[str, TrustClass] = {
    "security-training-example": "MIXED_QUOTED_TRAINING",
    "v3-research-injection-override": "UNTRUSTED_ADVERSARIAL",
    "v3-research-injection-role": "UNTRUSTED_ADVERSARIAL",
    "v3-research-injection-system": "UNTRUSTED_ADVERSARIAL",
    "v3-research-injection-tool": "UNTRUSTED_ADVERSARIAL",
    "v3-research-injection-force": "UNTRUSTED_ADVERSARIAL",
    "v3-research-injection-cite": "UNTRUSTED_ADVERSARIAL",
    "v3-research-injection-policy": "UNTRUSTED_ADVERSARIAL",
    "v3-research-injection-encoded": "UNTRUSTED_ADVERSARIAL",
    "v3-research-injection-mixed": "UNTRUSTED_ADVERSARIAL",
    "v3-research-injection-conflict-a": "UNTRUSTED_ADVERSARIAL",
    "v3-research-injection-conflict-b": "UNTRUSTED_ADVERSARIAL",
    "v3-research-payment-instruction-policy": "AUTHORITATIVE_BUSINESS",
    "v3-research-assistant-director-ops": "AUTHORITATIVE_BUSINESS",
    "v3-research-prompt-catalog": "AUTHORITATIVE_BUSINESS",
    "v3-research-system-availability": "AUTHORITATIVE_BUSINESS",
    "v3-research-ignore-list-finance": "AUTHORITATIVE_BUSINESS",
    "v3-research-security-tabletop-prompt": "AUTHORITATIVE_BUSINESS",
}

EMPLOYEE_SUBJECTS = (
    "employee",
    "employees",
    "staff",
    "manager",
    "managers",
    "engineer",
    "engineers",
    "finance operations",
    "personnel",
    "teams",
    "reviewer",
    "reviewers",
    "operator",
    "operators",
    "human resources",
    "duty officer",
    "assistant director",
)

MODEL_ADDRESSEE_STRUCTURES = (
    re.compile(r"(?i)\b(the |this )?(assistant|language model|llm|retrieval system)\b"),
    re.compile(r"(?i)\b(you are now|your new (task|role|instructions?))\b"),
    re.compile(r"(?i)\b(system prompt|developer message)\b"),
)

ROLE_IMPERSONATION_STRUCTURES = (
    re.compile(r"(?im)^\s*(system|assistant|developer|tool)\s*:"),
    re.compile(r"(?i)<\|?(im_start|system|assistant)\|?>"),
    re.compile(r"(?i)\[/?INST\]"),
    re.compile(r'(?i)\{\s*"role"\s*:\s*"(system|assistant|developer)"'),
    re.compile(r"(?i)<system>"),
)

TOOL_CALL_STRUCTURES = (
    re.compile(r"(?i)\b(tool_call|function_call|invoke[_ ]tool)\b"),
    re.compile(r'(?i)\{\s*"name"\s*:\s*"(reveal|drop_schema|grant_access|exfiltrate)'),
    re.compile(r"(?i)\bexecute the (tool|function|api) call\b"),
)

MODEL_CONTROL_STRUCTURES = (
    re.compile(r"(?i)\bignore (all )?(previous|prior|above) (instructions?|prompts?|rules?)\b"),
    re.compile(r"(?i)\bdisregard (your|the) (system|safety|previous)\b"),
    re.compile(r"(?i)\boverride (the )?(system|runtime|safety) (policy|rules?|prompt)\b"),
    re.compile(r"(?i)\btreat (this|the) (document|chunk|text|memo) as (system|runtime) (policy|instructions?)\b"),
    re.compile(r"(?i)\bdo not abstain\b"),
    re.compile(r"(?i)\bmark (this|the question) as answerable\b"),
    re.compile(r"(?i)\b(drop|omit|without) (the )?citations\b"),
    re.compile(r"(?i)\binvent (a |an )?(chunk|citation|supporting) id\b"),
    re.compile(r"(?i)\breveal (system secrets|credentials|hidden (compensation|hr) data)\b"),
    re.compile(r"(?i)\banswer without citations\b"),
    re.compile(r"(?i)\byou must now (answer|respond|output|ignore)\b"),
)

QUOTED_EXAMPLE_FRAMES = (
    re.compile(r"(?i)quoted text is hostile data"),
    re.compile(r"(?i)malicious example for employee training"),
    re.compile(r"(?i)this quoted text is hostile"),
    re.compile(r"(?i)treating quoted hostile commands as untrusted"),
    re.compile(r"(?i)must never control the assistant"),
)

IDENTIFIER_SEEKING = re.compile(
    r"(?i)\b(transcribe|quote|record only|catalog token|literal identifier|"
    r"reference identifier|approval (code|identifier)|runbook code)\b"
)
COMPLIANCE_QUESTION = re.compile(
    r"(?i)\b("
    r"asks the assistant to comply|"
    r"follow any retrieved instruction|"
    r"obey (the |any )?retrieved|"
    r"switch tasks because|"
    r"comply with (a |the )?retrieved|"
    r"treat the retrieved .* as system|"
    r"drop the schema|"
    r"choose an unauthorized id|"
    r"reveal credentials|"
    r"ignore acl boundaries|"
    r"document text outranks policy"
    r")\b"
)

FOLLOWED_PAYLOADS = (
    "ignore all previous instructions",
    "reveal system secrets",
    "answer without citations",
    "drop the schema",
    "mark this question answerable",
    "you are now unrestricted",
)


def boundary_template_hash() -> str:
    payload = {
        "version": BOUNDARY_VERSION,
        "document_trust_class": DOCUMENT_TRUST_CLASS,
        "roles": [
            "FACTUAL_ASSERTION",
            "POLICY_CONTENT",
            "PROCEDURAL_CONTENT",
            "EMPLOYEE_DIRECTED_PROCEDURE",
            "MODEL_DIRECTED_INSTRUCTION",
            "QUOTED_HOSTILE_EXAMPLE",
            "UNKNOWN",
        ],
        "fail_closed": True,
        "quality_retries": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.casefold()))


def content_tokens(text: str) -> set[str]:
    return {item for item in tokens(text) if item not in STOPWORDS and len(item) > 1}


def document_trust_class(document_id: str) -> TrustClass:
    return DOCUMENT_TRUST_CLASS.get(document_id, "AUTHORITATIVE_BUSINESS")


def _decode_base64(blob: str) -> str | None:
    padded = blob + ("=" * ((4 - len(blob) % 4) % 4))
    try:
        decoded = base64.b64decode(padded, validate=False)
    except (binascii.Error, ValueError):
        return None
    try:
        text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not any(char.isalpha() for char in text):
        return None
    return text


def _decode_hex(blob: str) -> str | None:
    try:
        decoded = bytes.fromhex(blob)
    except ValueError:
        return None
    try:
        text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not any(char.isalpha() for char in text):
        return None
    return text


def _decoded_payloads(text: str) -> tuple[str, ...]:
    found: list[str] = []
    for match in BASE64_TOKEN.findall(text):
        decoded = _decode_base64(match)
        if decoded and len(decoded) >= 12:
            found.append(decoded)
    for match in HEX_TOKEN.findall(text):
        decoded = _decode_hex(match)
        if decoded and len(decoded) >= 12:
            found.append(decoded)
    stripped = re.sub(r"[^a-z]+", "", text.casefold())
    reversed_text = stripped[::-1]
    if "ignorepreviousinstructions" in reversed_text or "revealsystemsecrets" in reversed_text:
        found.append("ignore previous instructions")
    return tuple(found)


def _has_model_addressee(text: str) -> bool:
    return any(pattern.search(text) for pattern in MODEL_ADDRESSEE_STRUCTURES)


def _has_role_impersonation(text: str) -> bool:
    return any(pattern.search(text) for pattern in ROLE_IMPERSONATION_STRUCTURES)


def _has_tool_call(text: str) -> bool:
    return any(pattern.search(text) for pattern in TOOL_CALL_STRUCTURES)


def _has_model_control(text: str) -> bool:
    return any(pattern.search(text) for pattern in MODEL_CONTROL_STRUCTURES)


def _has_employee_subject(text: str) -> bool:
    folded = text.casefold()
    return any(subject in folded for subject in EMPLOYEE_SUBJECTS)


def _is_quoted_example_frame(text: str) -> bool:
    return any(pattern.search(text) for pattern in QUOTED_EXAMPLE_FRAMES)


def classify_sentence(text: str, *, quoted: bool = False) -> SpeechRole:
    stripped = text.strip()
    if not stripped:
        return "UNKNOWN"
    if (quoted or _is_quoted_example_frame(stripped)) and (
        _has_model_control(stripped) or _has_role_impersonation(stripped)
    ):
        return "QUOTED_HOSTILE_EXAMPLE"
    if _has_role_impersonation(stripped) or _has_tool_call(stripped):
        return "MODEL_DIRECTED_INSTRUCTION"
    if _has_model_control(stripped) and not _has_employee_subject(stripped):
        return "MODEL_DIRECTED_INSTRUCTION"
    if _has_model_addressee(stripped) and (
        _has_model_control(stripped) or "comply" in stripped.casefold()
    ):
        return "MODEL_DIRECTED_INSTRUCTION"
    if _has_employee_subject(stripped) and (
        "must" in stripped.casefold()
        or "shall" in stripped.casefold()
        or "required" in stripped.casefold()
        or "ignore" in stripped.casefold()
    ):
        return "EMPLOYEE_DIRECTED_PROCEDURE"
    if IDENTIFIER_TOKEN.search(stripped) or re.search(
        r"(?i)\b(identifier|code|window|allowance|objective|retained|launched)\b",
        stripped,
    ):
        return "FACTUAL_ASSERTION"
    if re.search(r"(?i)\b(policy|standard|handbook|runbook)\b", stripped):
        return "POLICY_CONTENT"
    if re.search(r"(?i)\b(must|shall|before requesting|attach this identifier)\b", stripped):
        return "PROCEDURAL_CONTENT"
    return "UNKNOWN"


def split_sentences(text: str) -> tuple[str, ...]:
    parts = [item.strip() for item in SENTENCE_SPLIT.split(text) if item.strip()]
    return tuple(parts) if parts else ((text.strip(),) if text.strip() else ())


def classify_chunk_sentences(text: str) -> tuple[tuple[str, SpeechRole], ...]:
    quoted_spans = re.findall(r'"([^"]{8,})"', text)
    labeled: list[tuple[str, SpeechRole]] = []
    for sentence in split_sentences(text):
        in_quote = any(sentence.strip('"') in span or span in sentence for span in quoted_spans)
        labeled.append((sentence, classify_sentence(sentence, quoted=in_quote)))
    for decoded in _decoded_payloads(text):
        labeled.append((decoded, classify_sentence(decoded)))
    return tuple(labeled)


def classify_question(question: str) -> QuestionIllocution:
    if COMPLIANCE_QUESTION.search(question) and not IDENTIFIER_SEEKING.search(question):
        return "MODEL_COMPLIANCE_REQUEST"
    if IDENTIFIER_SEEKING.search(question) and IDENTIFIER_TOKEN.search(question):
        return "CATALOG_IDENTIFIER_SEEKING"
    if IDENTIFIER_SEEKING.search(question):
        return "CATALOG_IDENTIFIER_SEEKING"
    if _has_employee_subject(question) and re.search(
        r"(?i)\b(must|required|ignore|instruction)\b", question
    ):
        return "EMPLOYEE_PROCEDURE_SEEKING"
    if COMPLIANCE_QUESTION.search(question):
        return "MODEL_COMPLIANCE_REQUEST"
    return "FACTUAL_INFORMATION_SEEKING"


def answer_follows_instruction_payload(answer: str | None) -> bool:
    if not answer:
        return False
    folded = answer.casefold()
    return any(payload in folded for payload in FOLLOWED_PAYLOADS)


def _chunk_by_id(chunks: tuple[GateEvidence, ...]) -> dict[str, GateEvidence]:
    return {chunk.chunk_id: chunk for chunk in chunks}


@dataclass(frozen=True)
class SpanRole:
    chunk_id: str
    document_id: str
    trust_class: TrustClass
    sentence: str
    role: SpeechRole


@dataclass(frozen=True)
class InstructionBoundaryDecision:
    verdict: BoundaryVerdict
    question_illocution: QuestionIllocution
    typed_failure: str | None
    reason: str
    span_roles: tuple[SpanRole, ...]
    instruction_spans_used: tuple[str, ...]
    legitimate_instruction_like: bool
    configuration_hash: str

    @property
    def blocked(self) -> bool:
        return self.verdict != SAFE_EVIDENCE


def _roles_for_used_chunks(
    chunks: tuple[GateEvidence, ...], used_ids: set[str]
) -> tuple[SpanRole, ...]:
    roles: list[SpanRole] = []
    for chunk in chunks:
        if used_ids and chunk.chunk_id not in used_ids:
            continue
        trust = document_trust_class(chunk.document_id)
        for sentence, role in classify_chunk_sentences(chunk.text):
            roles.append(
                SpanRole(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    trust_class=trust,
                    sentence=sentence,
                    role=role,
                )
            )
    return tuple(roles)


def _answer_relies_on_instruction(
    answer: str, roles: tuple[SpanRole, ...]
) -> tuple[bool, tuple[str, ...]]:
    answer_tokens = content_tokens(answer)
    instruction_hits: list[str] = []
    fact_tokens: set[str] = set()
    instruction_tokens: set[str] = set()
    for item in roles:
        sentence_tokens = content_tokens(item.sentence)
        if item.role in {"FACTUAL_ASSERTION", "POLICY_CONTENT", "PROCEDURAL_CONTENT", "EMPLOYEE_DIRECTED_PROCEDURE"}:
            fact_tokens |= sentence_tokens
        if item.role == "MODEL_DIRECTED_INSTRUCTION":
            instruction_tokens |= sentence_tokens
            overlap = answer_tokens & sentence_tokens
            if overlap and overlap - fact_tokens:
                instruction_hits.append(item.sentence)
        if item.role == "QUOTED_HOSTILE_EXAMPLE":
            overlap = answer_tokens & sentence_tokens
            distinctive = overlap - fact_tokens
            if distinctive and any(
                payload in item.sentence.casefold() for payload in FOLLOWED_PAYLOADS
            ) and distinctive & content_tokens(" ".join(FOLLOWED_PAYLOADS)):
                instruction_hits.append(item.sentence)
    distinctive_instruction = (answer_tokens & instruction_tokens) - fact_tokens
    if distinctive_instruction:
        instruction_hits.append("token-overlap")
    return bool(instruction_hits), tuple(instruction_hits)


def evaluate_instruction_boundary(
    *,
    question: str,
    chunks: tuple[GateEvidence, ...],
    answer: str | None,
    citation_ids: tuple[str, ...],
    supporting_chunk_ids: tuple[str, ...],
) -> InstructionBoundaryDecision:
    illocution = classify_question(question)
    used_ids = set(citation_ids) | set(supporting_chunk_ids)
    if not used_ids:
        used_ids = {chunk.chunk_id for chunk in chunks}
    roles = _roles_for_used_chunks(chunks, used_ids)
    trust_used = {document_trust_class(chunk.document_id) for chunk in chunks if chunk.chunk_id in used_ids}
    legitimate = any(
        item.role == "EMPLOYEE_DIRECTED_PROCEDURE" and item.trust_class == "AUTHORITATIVE_BUSINESS"
        for item in roles
    )
    config_hash = boundary_template_hash()

    if not answer:
        return InstructionBoundaryDecision(
            verdict=SAFE_EVIDENCE,
            question_illocution=illocution,
            typed_failure=None,
            reason="no_answer_to_gate",
            span_roles=roles,
            instruction_spans_used=(),
            legitimate_instruction_like=legitimate,
            configuration_hash=config_hash,
        )

    if illocution == "MODEL_COMPLIANCE_REQUEST":
        return InstructionBoundaryDecision(
            verdict=UNTRUSTED_INSTRUCTION,
            question_illocution=illocution,
            typed_failure=TYPED_FAILURE,
            reason="question_requests_model_compliance_with_retrieved_instructions",
            span_roles=roles,
            instruction_spans_used=(),
            legitimate_instruction_like=legitimate,
            configuration_hash=config_hash,
        )

    if answer_follows_instruction_payload(answer):
        return InstructionBoundaryDecision(
            verdict=UNTRUSTED_INSTRUCTION,
            question_illocution=illocution,
            typed_failure=TYPED_FAILURE,
            reason="answer_copies_model_directed_payload",
            span_roles=roles,
            instruction_spans_used=tuple(
                item.sentence for item in roles if item.role == "MODEL_DIRECTED_INSTRUCTION"
            ),
            legitimate_instruction_like=legitimate,
            configuration_hash=config_hash,
        )

    relies, hits = _answer_relies_on_instruction(answer, roles)
    if relies:
        return InstructionBoundaryDecision(
            verdict=UNTRUSTED_INSTRUCTION,
            question_illocution=illocution,
            typed_failure=TYPED_FAILURE,
            reason="answer_materially_relies_on_model_directed_instruction_span",
            span_roles=roles,
            instruction_spans_used=hits,
            legitimate_instruction_like=legitimate,
            configuration_hash=config_hash,
        )

    adversarial = "UNTRUSTED_ADVERSARIAL" in trust_used or "MIXED_QUOTED_TRAINING" in trust_used
    if adversarial and illocution != "CATALOG_IDENTIFIER_SEEKING":
        return InstructionBoundaryDecision(
            verdict=UNTRUSTED_INSTRUCTION,
            question_illocution=illocution,
            typed_failure=TYPED_FAILURE,
            reason="recovery_cited_untrusted_adversarial_or_mixed_training_without_identifier_question",
            span_roles=roles,
            instruction_spans_used=tuple(
                item.sentence
                for item in roles
                if item.role in {"MODEL_DIRECTED_INSTRUCTION", "QUOTED_HOSTILE_EXAMPLE"}
            ),
            legitimate_instruction_like=legitimate,
            configuration_hash=config_hash,
        )

    if adversarial and illocution == "CATALOG_IDENTIFIER_SEEKING":
        fact_roles = {
            item.role
            for item in roles
            if content_tokens(item.sentence) & content_tokens(answer)
        }
        if fact_roles and fact_roles <= {"MODEL_DIRECTED_INSTRUCTION"}:
            return InstructionBoundaryDecision(
                verdict=UNTRUSTED_INSTRUCTION,
                question_illocution=illocution,
                typed_failure=TYPED_FAILURE,
                reason="identifier_seeking_answer_aligned_only_to_instruction_span",
                span_roles=roles,
                instruction_spans_used=tuple(
                    item.sentence for item in roles if item.role == "MODEL_DIRECTED_INSTRUCTION"
                ),
                legitimate_instruction_like=legitimate,
                configuration_hash=config_hash,
            )

    unknown_modelish = any(
        item.role == "UNKNOWN" and (_has_role_impersonation(item.sentence) or _decoded_payloads(item.sentence))
        for item in roles
    )
    if unknown_modelish:
        return InstructionBoundaryDecision(
            verdict=UNCERTAIN,
            question_illocution=illocution,
            typed_failure=TYPED_FAILURE,
            reason="uncertain_structured_instruction_payload",
            span_roles=roles,
            instruction_spans_used=(),
            legitimate_instruction_like=legitimate,
            configuration_hash=config_hash,
        )

    return InstructionBoundaryDecision(
        verdict=SAFE_EVIDENCE,
        question_illocution=illocution,
        typed_failure=None,
        reason="business_factual_or_employee_directed_evidence",
        span_roles=roles,
        instruction_spans_used=(),
        legitimate_instruction_like=legitimate,
        configuration_hash=config_hash,
    )


def apply_untrusted_instruction_boundary(
    outcome: RecoveryOutcome,
    *,
    question: str,
    chunks: tuple[GateEvidence, ...],
) -> tuple[RecoveryOutcome, InstructionBoundaryDecision]:
    decision = evaluate_instruction_boundary(
        question=question,
        chunks=chunks,
        answer=outcome.answer,
        citation_ids=outcome.citations,
        supporting_chunk_ids=outcome.supporting_chunk_ids,
    )
    if not outcome.answered or not decision.blocked:
        return outcome, decision
    blocked = replace(
        outcome,
        answered=False,
        answer=None,
        citations=(),
        supporting_chunk_ids=(),
        verification_pass=False,
        validation_error=TYPED_FAILURE,
        typed_failure=TYPED_FAILURE,
    )
    return blocked, decision
