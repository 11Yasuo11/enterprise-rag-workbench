"""Corpus roots used by ingestion and dataset validation.

Frozen V1/V2 evaluation continues to read only `data/synthetic_company`.
V3 research documents live in `data/v3_research_corpus` and are attached only
when the corpus version is the V3 research extension. This does not mutate
the frozen V2 semantic index identity.
"""

from __future__ import annotations

from pathlib import Path

FROZEN_SYNTHETIC_CORPUS = Path("data/synthetic_company")
V3_RESEARCH_CORPUS = Path("data/v3_research_corpus")
V3_RESEARCH_CORPUS_VERSION = "acmeai-v0.1-v3-research-extension"
CORPUS_MANIFEST_ROOT = Path("data/corpus_manifests")
CORPUS_SUFFIXES = {".md", ".markdown", ".pdf"}


def corpus_roots_for_version(
    corpus_version: str,
    *,
    data_root: Path | None = None,
    frozen_root: Path | None = None,
) -> list[Path]:
    data_root = data_root or Path("data")
    frozen = frozen_root or data_root / "synthetic_company"
    roots = [frozen]
    if corpus_version == V3_RESEARCH_CORPUS_VERSION:
        extra = data_root / "v3_research_corpus"
        if extra not in roots:
            roots.append(extra)
    return roots


def collect_corpus_paths(
    corpus_version: str,
    corpus_root: Path,
    *,
    require_manifest_complete: bool = False,
) -> list[Path]:
    roots = corpus_roots_for_version(corpus_version, frozen_root=corpus_root)
    paths: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in CORPUS_SUFFIXES:
                continue
            if path.name in seen:
                raise ValueError(f"duplicate corpus filename across roots: {path.name}")
            seen.add(path.name)
            paths.append(path)
    manifest = CORPUS_MANIFEST_ROOT / f"{corpus_version}.txt"
    if manifest.is_file():
        allowed = {
            line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        paths = [path for path in paths if path.name in allowed]
        if require_manifest_complete:
            missing = sorted(allowed - {path.name for path in paths})
            if missing:
                raise ValueError(
                    f"Corpus manifest references missing files: {', '.join(missing)}"
                )
    return paths
