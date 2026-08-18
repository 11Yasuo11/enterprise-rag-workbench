from pathlib import Path

from rag_workbench.domain.documents import CanonicalDocument
from rag_workbench.ingestion.loaders.markdown import load_markdown
from rag_workbench.ingestion.loaders.pdf import load_pdf


def load_document(path: Path, tenant_id: str = "acmeai") -> CanonicalDocument:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return load_markdown(path, tenant_id)
    if suffix == ".pdf":
        return load_pdf(path, tenant_id)
    raise ValueError(f"Unsupported document type: {suffix}")
