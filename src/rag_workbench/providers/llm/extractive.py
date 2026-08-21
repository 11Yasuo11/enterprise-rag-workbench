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
EXTRACTIVE_V2_MODEL = "deterministic-extractive-v2"
EXTRACTIVE_V3_MODEL = "deterministic-extractive-v3"
EXTRACTIVE_REVISION = {
    "id": EXTRACTIVE_V1_1_MODEL,
    "parent": EXTRACTIVE_V1_MODEL,
    "reason": "reliability fix only",
    "semantic_policy_changed": False,
}
GENERATOR_COMPLETENESS_V2_REVISION = {
    "id": EXTRACTIVE_V2_MODEL,
    "parent": EXTRACTIVE_V1_1_MODEL,
    "reason": "per-chunk diversity selection to fix multi-document completeness",
    "semantic_policy_changed": True,
    "algorithm_id": "GENERATOR_COMPLETENESS_V2",
}
GENERATOR_COMPLETENESS_V3_REVISION = {
    "id": EXTRACTIVE_V3_MODEL,
    "parent": EXTRACTIVE_V2_MODEL,
    "reason": "identifier-aware sentence selection for endpoint/code completeness",
    "semantic_policy_changed": True,
    "algorithm_id": "GENERATOR_COMPLETENESS_V3",
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


def overlap_candidates_complete(
    request: GenerationRequest,
) -> list[tuple[int, int, str, str, str]]:
    """GENERATOR_COMPLETENESS_V2: per-chunk-diverse sentence selection.

    Ensures at least one sentence is selected from each chunk present in the
    generation context before filling remaining slots globally. This prevents
    multi-document answers from omitting an entire document's required facts.
    """
    terms = query_terms(request.question)
    per_chunk: dict[str, list[tuple[int, int, str, str, str]]] = {}
    for context_index, context in enumerate(request.contexts):
        for sentence in SENTENCE.split(context.text):
            overlap = len(terms & {word.lower() for word in WORD.findall(sentence)})
            if overlap:
                entry = (
                    overlap,
                    -context_index,
                    sentence.strip(),
                    context.chunk_id,
                    context.citation_label,
                )
                per_chunk.setdefault(context.chunk_id, []).append(entry)

    for entries in per_chunk.values():
        entries.sort(reverse=True)

    if not per_chunk:
        return []

    # Phase 1: best sentence from each chunk (document diversity)
    selected: list[tuple[int, int, str, str, str]] = []
    for chunk_id in per_chunk:
        selected.append(per_chunk[chunk_id][0])

    # Phase 2: if only one chunk contributed, add the global second-best
    # from a different chunk if available (preserves original 2-sentence limit
    # while maximizing document coverage)
    if len(selected) == 1:
        all_candidates = []
        for entries in per_chunk.values():
            all_candidates.extend(entries)
        all_candidates.sort(reverse=True)
        if len(all_candidates) > 1:
            selected.append(all_candidates[1])

    # Cap at number of contexts (one fact per chunk maximum)
    max_sentences = max(len(per_chunk), 2)
    selected.sort(reverse=True)
    return selected[:max_sentences]


def overlap_candidates_complete_v3(
    request: GenerationRequest,
) -> list[tuple[int, int, str, str, str]]:
    """GENERATOR_COMPLETENESS_V3: V2 + identifier-aware sentence preference."""
    terms = query_terms(request.question)
    wants_identifier = bool(
        re.search(r"(?is)\b(identifier|id|code|endpoint)\b", request.question)
    )
    per_chunk: dict[str, list[tuple[int, int, str, str, str]]] = {}
    for context_index, context in enumerate(request.contexts):
        for sentence in SENTENCE.split(context.text):
            sentence_terms = {word.lower() for word in WORD.findall(sentence)}
            overlap = len(terms & sentence_terms)
            if not overlap:
                continue
            bonus = 0
            if wants_identifier and re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b", sentence):
                bonus = 2
            entry = (
                overlap + bonus,
                -context_index,
                sentence.strip(),
                context.chunk_id,
                context.citation_label,
            )
            per_chunk.setdefault(context.chunk_id, []).append(entry)
    for entries in per_chunk.values():
        entries.sort(reverse=True)
    if not per_chunk:
        return []
    selected: list[tuple[int, int, str, str, str]] = []
    for chunk_id in per_chunk:
        selected.append(per_chunk[chunk_id][0])
    if len(selected) == 1:
        all_candidates = []
        for entries in per_chunk.values():
            all_candidates.extend(entries)
        all_candidates.sort(reverse=True)
        if len(all_candidates) > 1:
            selected.append(all_candidates[1])
    max_sentences = max(len(per_chunk), 2)
    selected.sort(reverse=True)
    return selected[:max_sentences]


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


def generate_extractive_v2(request: GenerationRequest) -> GenerationResult:
    """GENERATOR_COMPLETENESS_V2: uses per-chunk diverse selection."""
    selected = overlap_candidates_complete(request)
    if selected:
        return result_from_selected(request, selected)
    if request.contexts:
        return verbatim_supporting(request)
    return GenerationResult(answer="", used_chunk_ids=())


def generate_extractive_v3(request: GenerationRequest) -> GenerationResult:
    """GENERATOR_COMPLETENESS_V3: identifier-aware diverse selection."""
    selected = overlap_candidates_complete_v3(request)
    if selected:
        return result_from_selected(request, selected)
    if request.contexts:
        return verbatim_supporting(request)
    return GenerationResult(answer="", used_chunk_ids=())


def generate_extractive_v1(request: GenerationRequest) -> GenerationResult:
    return generate_extractive(request, allow_verbatim_fallback=False)


def generate_extractive_v1_1(request: GenerationRequest) -> GenerationResult:
    return generate_extractive(request, allow_verbatim_fallback=True)


class ExtractiveGenerationProvider:
    """Offline baseline generator. It is intentionally not represented as a general-purpose LLM."""

    def __init__(self, *, revision: str = EXTRACTIVE_V1_1_MODEL) -> None:
        if revision not in {
            EXTRACTIVE_V1_MODEL,
            EXTRACTIVE_V1_1_MODEL,
            EXTRACTIVE_V2_MODEL,
            EXTRACTIVE_V3_MODEL,
        }:
            raise ValueError(f"unknown extractive revision: {revision}")
        self._revision = revision

    @property
    def provider_name(self) -> str:
        return "local-extractive"

    @property
    def model_name(self) -> str:
        return self._revision

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._revision == EXTRACTIVE_V3_MODEL:
            return generate_extractive_v3(request)
        if self._revision == EXTRACTIVE_V2_MODEL:
            return generate_extractive_v2(request)
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
