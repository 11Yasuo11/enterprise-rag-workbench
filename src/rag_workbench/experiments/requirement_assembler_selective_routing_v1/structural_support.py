"""Conservative literal support patterns for deterministically solvable queries."""

from __future__ import annotations

import re

from rag_workbench.experiments.requirement_assembler_selective_routing_v1.assembler import (
    EvidenceChunk,
    VerifiedRequirement,
)


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]


def extract_structural_support(
    question: str, chunks: tuple[EvidenceChunk, ...]
) -> tuple[VerifiedRequirement, ...]:
    request = question.rsplit(". ", 1)[-1]
    lower = request.casefold()
    region = "east" if "east-region" in lower else "west" if "west-region" in lower else None
    predicates: list[tuple[str, str]] = []
    if "production deployment approval identifier" in lower:
        predicates.append(
            (
                "production deployment approval identifier",
                "production deployment approval identifier",
            )
        )
    elif "who maintains the atlas api" in lower:
        predicates.append(("Atlas API maintainer", "atlas api is maintained by"))
    elif "runbook code" in lower and region:
        predicates.append((f"{region}-region recovery runbook code", "runbook code"))
    elif "standby cluster" in lower and region:
        predicates.append((f"{region}-region standby cluster", "standby cluster"))
    else:
        return ()
    requirements: list[VerifiedRequirement] = []
    for index, (label, needle) in enumerate(predicates, 1):
        matches: list[tuple[EvidenceChunk, str]] = []
        for chunk in chunks:
            if region and region not in chunk.document_id.casefold():
                continue
            for sentence in _sentences(chunk.text):
                if needle in sentence.casefold():
                    matches.append((chunk, sentence))
        if len(matches) != 1:
            return ()
        chunk, sentence = matches[0]
        requirements.append(
            VerifiedRequirement(f"R{index}", label, chunk.chunk_id, chunk.document_id, (sentence,))
        )
    return tuple(requirements)
