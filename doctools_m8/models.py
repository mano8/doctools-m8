"""
C-11 document models: the parsed document, chunks and extraction results.

Every model is frozen, strict and refuses unknown fields, so a value read back
from an artifact is exactly the value that was written. Chunk and block kinds
are honest (D-28): ``table``, ``list``, ``code`` and ``caption`` are reserved
for parsers that emitted structural tags; a text-layer parser uses ``text`` or
``unknown`` only. Provenance never records guessed structure.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    Strict,
    StringConstraints,
    model_validator,
)

__all__ = [
    "BBox",
    "Block",
    "BlockKind",
    "Chunk",
    "ChunkKind",
    "ExtractedField",
    "ExtractionResult",
    "FieldValue",
    "Identifier",
    "Page",
    "ParsedDocument",
    "Provenance",
    "ProvenanceSource",
    "TextSource",
]

Identifier = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
]
"""An opaque identifier: block, chunk, field and component names."""

BlockKind = Literal["text", "heading", "table", "list", "code", "caption", "unknown"]
"""Block kinds; anything but ``text``/``unknown`` needs parser-emitted structure."""

ChunkKind = Literal["text", "table", "list", "code", "caption", "unknown"]
"""Chunk kinds (D-28); anything but ``text``/``unknown`` needs structural tags."""

TextSource = Literal["text_layer", "ocr"]
"""Where a block's text came from."""

ProvenanceSource = Literal["text_layer", "ocr", "extractor"]
"""Where an extracted field's value came from."""

type Items[T] = Annotated[tuple[T, ...], Strict(False)]
"""An immutable sequence that also accepts a decoded JSON array (a list)."""

FieldValue = str | int | float | bool | Items[str] | None
"""The value of one extracted field."""

PageNumber = Annotated[int, Field(ge=1)]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class BBox(_Model):
    """A rectangle in page coordinates (PDF points, origin bottom-left)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.x0 > self.x1 or self.y0 > self.y1:
            raise ValueError("bbox corners must be ordered (x0 <= x1, y0 <= y1)")
        return self


class Block(_Model):
    """One contiguous run of text on one page."""

    block_id: Identifier
    page_number: PageNumber
    kind: BlockKind = "unknown"
    source: TextSource = "text_layer"
    text: str
    bbox: BBox | None = None


class Page(_Model):
    """One page with its blocks in reading order."""

    page_number: PageNumber
    width: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)
    ocr: bool = False
    """True when this page's text came from OCR rather than a text layer."""
    blocks: Items[Block] = ()

    @model_validator(mode="after")
    def _blocks_on_this_page(self) -> Self:
        for block in self.blocks:
            if block.page_number != self.page_number:
                raise ValueError("every block must carry its page's page_number")
        return self


class ParsedDocument(_Model):
    """The normalized content of one source object (C-11)."""

    media_type: Annotated[str, StringConstraints(min_length=1, max_length=255)]
    pages: Items[Page] = ()
    metadata: dict[Identifier, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        seen: set[str] = set()
        for index, page in enumerate(self.pages, start=1):
            if page.page_number != index:
                raise ValueError("pages must be numbered 1..n in order")
            for block in page.blocks:
                if block.block_id in seen:
                    raise ValueError(f"duplicate block_id {block.block_id!r}")
                seen.add(block.block_id)
        return self

    @property
    def block_count(self) -> int:
        """The number of blocks over every page."""
        return sum(len(page.blocks) for page in self.pages)

    @property
    def text_bytes(self) -> int:
        """The UTF-8 byte length of every block's text."""
        return sum(
            len(block.text.encode("utf-8"))
            for page in self.pages
            for block in page.blocks
        )

    @property
    def ocr_pages(self) -> int:
        """The number of pages whose text came from OCR."""
        return sum(1 for page in self.pages if page.ocr)


class Chunk(_Model):
    """One retrieval unit cut from a document."""

    chunk_id: Identifier
    ordinal: Annotated[int, Field(ge=0)]
    kind: ChunkKind = "unknown"
    text: str
    page_numbers: Items[PageNumber] = ()
    block_ids: Items[Identifier] = ()
    token_count: Annotated[int, Field(ge=0)] | None = None


class Provenance(_Model):
    """Where one extracted value came from in the document."""

    source: ProvenanceSource
    page_numbers: Items[PageNumber] = ()
    block_ids: Items[Identifier] = ()
    chunk_ids: Items[Identifier] = ()
    quote: str | None = None


class ExtractedField(_Model):
    """One extracted value with its provenance."""

    value: FieldValue
    confidence: Annotated[float, Field(ge=0, le=1)] | None = None
    provenance: Items[Provenance] = ()


class ExtractionResult(_Model):
    """The fields a metadata extractor produced, each with its own provenance."""

    extractor: Identifier
    extractor_version: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    fields: dict[Identifier, ExtractedField] = Field(default_factory=dict)
