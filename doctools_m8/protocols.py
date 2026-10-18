"""
C-11 protocols: the seams a deploying service implements or calls.

``doctools_m8`` ships no network, database or credential code. Parsing and
chunking are local CPU work and are synchronous; OCR, embedding, reranking and
metadata extraction are provider capabilities reached through the gateway by
the deploying service, so those protocols are asynchronous.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from doctools_m8.models import Chunk, ExtractionResult, Identifier, Page, ParsedDocument

__all__ = [
    "Chunker",
    "DocumentParser",
    "EmbeddingProvider",
    "MetadataExtractor",
    "OcrPageResult",
    "OcrProvider",
    "ProviderExecution",
    "RerankProvider",
]


class ProviderExecution(BaseModel):
    """The provider identity one gateway call resolved to (C-07)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    provider_id: Identifier
    external_model_id: str = Field(min_length=1, max_length=255)
    model_revision: str = Field(min_length=1, max_length=255)


class OcrPageResult(BaseModel):
    """One OCR'd page and the execution identity that produced it."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    page: Page
    execution: ProviderExecution


@runtime_checkable
class DocumentParser(Protocol):
    """
    Turns one source file into a :class:`ParsedDocument`.

    ``name``, ``version`` and ``config`` are the parser inputs of
    ``parse_revision`` (C-07).
    """

    name: str
    version: str
    config: Mapping[str, object]

    def parse(self, path: str | PathLike[str], *, media_type: str) -> ParsedDocument:
        """Parse the file at ``path`` declared as ``media_type``."""


@runtime_checkable
class OcrProvider(Protocol):
    """Recognizes the text of one rendered page."""

    async def recognize(
        self, image: bytes, *, media_type: str, page_number: int
    ) -> OcrPageResult:
        """Return the page's text with the execution identity that produced it."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embeds texts into one fixed embedding space."""

    dimensions: int

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one vector of ``dimensions`` floats per text, in order."""


@runtime_checkable
class RerankProvider(Protocol):
    """Scores candidate texts against a query."""

    async def rerank(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        """Return one relevance score per document, in order."""


@runtime_checkable
class MetadataExtractor(Protocol):
    """Extracts typed fields, each with provenance, from a document."""

    async def extract(
        self, document: ParsedDocument, chunks: Sequence[Chunk]
    ) -> ExtractionResult:
        """Return the extracted fields of ``document``."""


@runtime_checkable
class Chunker(Protocol):
    """
    Cuts a document into retrieval chunks.

    ``name``, ``version``, ``tokenizer_version`` and ``config`` are the chunker
    inputs of ``chunk_revision`` (C-07).
    """

    name: str
    version: str
    tokenizer_version: str
    config: Mapping[str, object]

    def chunk(self, document: ParsedDocument) -> Sequence[Chunk]:
        """Return the chunks of ``document`` in ordinal order."""
