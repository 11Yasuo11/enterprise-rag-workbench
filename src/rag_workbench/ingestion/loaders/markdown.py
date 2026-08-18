import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from rag_workbench.domain.documents import CanonicalDocument, DocumentSegment

FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _date(value: Any, default: datetime | None = None) -> datetime | None:
    if value is None:
        return default
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _segments(content: str) -> tuple[DocumentSegment, ...]:
    matches = list(HEADING.finditer(content))
    if not matches:
        return (DocumentSegment(text=content.strip()),)
    segments: list[DocumentSegment] = []
    prefix = content[: matches[0].start()].strip()
    if prefix:
        segments.append(DocumentSegment(text=prefix))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        body = content[match.end() : end].strip()
        if body:
            segments.append(DocumentSegment(text=body, section=match.group(2).strip()))
    return tuple(segments)


def load_markdown(path: Path, tenant_id: str = "acmeai") -> CanonicalDocument:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER.match(raw)
    metadata: dict[str, Any] = yaml.safe_load(match.group(1)) or {} if match else {}
    content = raw[match.end() :] if match else raw
    stat_time = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    document_id = str(metadata.pop("document_id", path.stem))
    title = str(metadata.pop("title", document_id.replace("-", " ").title()))
    version = str(metadata.pop("version", "1"))
    groups = tuple(str(item) for item in metadata.pop("permission_groups", []) or [])
    visibility = str(metadata.pop("visibility", "restricted" if groups else "public"))
    return CanonicalDocument(
        document_id=document_id,
        title=title,
        content=content.strip(),
        source=str(path),
        source_type="markdown",
        source_url=metadata.pop("source_url", None),
        version=version,
        tenant_id=str(metadata.pop("tenant_id", tenant_id)),
        created_at=_date(metadata.pop("created_at", None), stat_time) or stat_time,
        updated_at=_date(metadata.pop("updated_at", None), stat_time) or stat_time,
        effective_at=_date(metadata.pop("effective_at", None)),
        visibility=visibility,
        permission_groups=groups,
        metadata=metadata,
        segments=_segments(content),
    )
