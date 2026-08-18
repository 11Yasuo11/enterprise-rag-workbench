import pytest

from rag_workbench.domain.documents import CanonicalDocument
from rag_workbench.ingestion.chunkers import FixedTokenChunker, FixedTokenConfig


def test_fixed_token_chunking_has_configured_overlap() -> None:
    document = CanonicalDocument(
        document_id="doc",
        title="Doc",
        content="one two three four five six seven eight",
        source="memory",
        source_type="markdown",
        version="1",
    )
    chunks = FixedTokenChunker(FixedTokenConfig(chunk_size=5, chunk_overlap=2)).chunk(document)
    assert [chunk.token_count for chunk in chunks] == [5, 5]
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert chunks[0].metadata["chunk_size"] == 5


def test_chunk_overlap_must_be_less_than_size() -> None:
    with pytest.raises(ValueError):
        FixedTokenConfig(chunk_size=10, chunk_overlap=10)
