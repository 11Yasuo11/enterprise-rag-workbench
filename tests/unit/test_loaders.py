from pathlib import Path

from reportlab.pdfgen.canvas import Canvas

from rag_workbench.ingestion.loaders.markdown import load_markdown
from rag_workbench.ingestion.loaders.pdf import load_pdf


def test_markdown_loader_normalizes_frontmatter_and_sections(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text(
        """---
document_id: stable-policy
title: Stable Policy
version: "2"
effective_at: "2026-01-01T00:00:00Z"
visibility: restricted
permission_groups: [security]
---
# Scope

The policy applies to production.
""",
        encoding="utf-8",
    )
    document = load_markdown(path)
    assert document.document_id == "stable-policy"
    assert document.version == "2"
    assert document.visibility == "restricted"
    assert document.permission_groups == ("security",)
    assert document.segments[0].section == "Scope"
    assert len(document.content_hash) == 64


def test_text_pdf_loader_preserves_page_number(tmp_path: Path) -> None:
    path = tmp_path / "runbook.pdf"
    canvas = Canvas(str(path))
    canvas.drawString(72, 720, "Recovery identifier OPS-71")
    canvas.save()
    document = load_pdf(path)
    assert "OPS-71" in document.content
    assert document.segments[0].page == 1
    assert document.source_type == "pdf"
