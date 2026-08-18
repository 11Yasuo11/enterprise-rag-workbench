import re
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader

from rag_workbench.domain.documents import CanonicalDocument, DocumentSegment


def load_pdf(path: Path, tenant_id: str = "acmeai") -> CanonicalDocument:
    reader = PdfReader(path)
    segments = tuple(
        DocumentSegment(text=text.strip(), page=index, metadata={"pdf_page": index})
        for index, page in enumerate(reader.pages, start=1)
        if (text := page.extract_text()) and text.strip()
    )
    if not segments:
        raise ValueError(f"PDF contains no extractable text (OCR is not supported): {path}")
    title = str(reader.metadata.title) if reader.metadata and reader.metadata.title else path.stem
    document_id = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return CanonicalDocument(
        document_id=document_id,
        title=title,
        content="\n\n".join(segment.text for segment in segments),
        source=str(path),
        source_type="pdf",
        version="1",
        tenant_id=tenant_id,
        created_at=modified,
        updated_at=modified,
        segments=segments,
    )
