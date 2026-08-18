import re

from rag_workbench.providers.llm.base import (
    GenerationContext,
    GenerationRequest,
    GenerationResult,
)

WORD = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", re.IGNORECASE)
SENTENCE = re.compile(r"(?<=[.!?])\s+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "what",
    "which",
    "who",
}

EXTRACTIVE_V1_MODEL = "deterministic-extractive-v1"
EXTRACTIVE_V1_1_MODEL = "deterministic-extractive-v1.1"
EXTRACTIVE_REVISION = {
    "id": EXTRACTIVE_V1_1_MODEL,
    "parent": EXTRACTIVE_V1_MODEL,
    "reason": "reliability fix only",
    "semantic_policy_changed": False,
}


def query_terms(question: str) -> set[str]:
    return {word.lower() for word in WORD.findall(question)} - STOP_WORDS


def overlap_candidates(
    request: GenerationRequest,
) -> list[tuple[int, int, str, str, str]]:
    terms = query_terms(request.question)
    candidates: list[tuple[int, int, str, str, str]] = []
    for context_index, context in enumerate(request.contexts):
        for sentence in SENTENCE.split(context.text):
            overlap = len(terms & {word.lower() for word in WORD.findall(sentence)})
            if overlap:
                candidates.append(
                    (
                        overlap,
                        -context_index,
                        sentence.strip(),
                        context.chunk_id,
                        context.citation_label,
                    )
                )
    candidates.sort(reverse=True)
    return candidates[:2]


def result_from_selected(
    request: GenerationRequest, selected: list[tuple[int, int, str, str, str]]
) -> GenerationResult:
    answer = " ".join(f"{sentence} [{label}]" for _, _, sentence, _, label in selected)
    chunk_ids = tuple(dict.fromkeys(chunk_id for _, _, _, chunk_id, _ in selected))
    return GenerationResult(
        answer=answer,
        used_chunk_ids=chunk_ids,
        prompt_tokens=len(request.prompt.split()),
        completion_tokens=len(answer.split()),
        metadata={"extractive_path": "query_overlap"},
    )


def _sentences(text: str) -> tuple[str, ...]:
    parts = tuple(sentence.strip() for sentence in SENTENCE.split(text) if sentence.strip())
    if parts:
        return parts
    stripped = text.strip()
    return (stripped,) if stripped else ()


def verbatim_supporting(request: GenerationRequest) -> GenerationResult:
    """Copy authorized supporting chunk text. Does not synthesize or use model knowledge."""
    parts: list[str] = []
    used: list[str] = []
    for context in request.contexts:
        sentences = _sentences(context.text)
        if not sentences:
            continue
        for sentence in sentences:
            parts.append(f"{sentence} [{context.citation_label}]")
        used.append(context.chunk_id)
    if not parts:
        return GenerationResult(answer="", used_chunk_ids=())
    answer = " ".join(parts)
    return GenerationResult(
        answer=answer,
        used_chunk_ids=tuple(dict.fromkeys(used)),
        prompt_tokens=len(request.prompt.split()),
        completion_tokens=len(answer.split()),
        metadata={"extractive_path": "verbatim_supporting_fallback"},
    )


def generate_extractive(
    request: GenerationRequest, *, allow_verbatim_fallback: bool
) -> GenerationResult:
    selected = overlap_candidates(request)
    if selected:
        return result_from_selected(request, selected)
    if allow_verbatim_fallback and request.contexts:
        return verbatim_supporting(request)
    return GenerationResult(answer="", used_chunk_ids=())


def generate_extractive_v1(request: GenerationRequest) -> GenerationResult:
    return generate_extractive(request, allow_verbatim_fallback=False)


def generate_extractive_v1_1(request: GenerationRequest) -> GenerationResult:
    return generate_extractive(request, allow_verbatim_fallback=True)


class ExtractiveGenerationProvider:
    """Offline baseline generator. It is intentionally not represented as a general-purpose LLM."""

    def __init__(self, *, revision: str = EXTRACTIVE_V1_1_MODEL) -> None:
        if revision not in {EXTRACTIVE_V1_MODEL, EXTRACTIVE_V1_1_MODEL}:
            raise ValueError(f"unknown extractive revision: {revision}")
        self._revision = revision

    @property
    def provider_name(self) -> str:
        return "local-extractive"

    @property
    def model_name(self) -> str:
        return self._revision

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return generate_extractive(
            request, allow_verbatim_fallback=self._revision == EXTRACTIVE_V1_1_MODEL
        )


def supporting_contexts_from_texts(
    items: tuple[tuple[str, str, str], ...],
) -> tuple[GenerationContext, ...]:
    return tuple(
        GenerationContext(chunk_id=chunk_id, citation_label=label, text=text)
        for chunk_id, label, text in items
    )
