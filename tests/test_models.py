"""C-11 models and protocols."""

from collections.abc import Mapping, Sequence
from os import PathLike

import pytest
from pydantic import ValidationError

from doctools_m8.models import (
    BBox,
    Block,
    Chunk,
    ExtractedField,
    ExtractionResult,
    Page,
    ParsedDocument,
    Provenance,
)
from doctools_m8.protocols import (
    Chunker,
    DocumentParser,
    EmbeddingProvider,
    MetadataExtractor,
    OcrPageResult,
    OcrProvider,
    ProviderExecution,
    RerankProvider,
)


def _document() -> ParsedDocument:
    return ParsedDocument.model_validate(
        {
            "media_type": "application/pdf",
            "pages": [
                {
                    "page_number": 1,
                    "width": 612,
                    "height": 792.0,
                    "blocks": [
                        {
                            "block_id": "p1-b1",
                            "page_number": 1,
                            "kind": "text",
                            "text": "héllo",
                            "bbox": {"x0": 0, "y0": 0, "x1": 10.5, "y1": 3},
                        }
                    ],
                },
                {"page_number": 2, "ocr": True, "blocks": []},
            ],
        }
    )


def test_decoded_json_arrays_validate_into_tuples() -> None:
    document = _document()
    assert isinstance(document.pages, tuple)
    assert document.pages[0].width == 612.0
    assert document.block_count == 1
    assert document.text_bytes == len("héllo".encode())
    assert document.ocr_pages == 1


@pytest.mark.parametrize(
    "patch",
    [
        {"media_type": 1},
        {"pages": [{"page_number": "1"}]},
        {"pages": [{"page_number": 1, "ocr": 1}]},
        {"unknown": True},
        {"pages": [{"page_number": 2}]},
        {"metadata": {"bad key": "x"}},
    ],
)
def test_models_are_strict_and_closed(patch: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ParsedDocument.model_validate({"media_type": "text/plain", **patch})


def test_block_must_sit_on_its_page() -> None:
    with pytest.raises(ValidationError, match="page_number"):
        Page(page_number=1, blocks=(Block(block_id="b", page_number=2, text=""),))


def test_block_ids_are_unique_across_pages() -> None:
    page = Page(page_number=1, blocks=(Block(block_id="b", page_number=1, text=""),))
    other = Page(page_number=2, blocks=(Block(block_id="b", page_number=2, text=""),))
    with pytest.raises(ValidationError, match="duplicate block_id"):
        ParsedDocument(media_type="text/plain", pages=(page, other))


def test_bbox_corners_are_ordered() -> None:
    with pytest.raises(ValidationError, match="ordered"):
        BBox(x0=2, y0=0, x1=1, y1=1)


def test_models_are_frozen() -> None:
    document = _document()
    with pytest.raises(ValidationError):
        document.media_type = "text/plain"


def test_chunk_and_extraction_models() -> None:
    chunk = Chunk.model_validate(
        {
            "chunk_id": "c0",
            "ordinal": 0,
            "kind": "text",
            "text": "x",
            "page_numbers": [1],
        }
    )
    assert chunk.page_numbers == (1,)
    with pytest.raises(ValidationError):
        Chunk(chunk_id="c0", ordinal=0, kind="paragraph", text="x")  # type: ignore[arg-type]
    result = ExtractionResult.model_validate(
        {
            "extractor": "meta",
            "extractor_version": "1",
            "fields": {
                "title": {
                    "value": "A title",
                    "confidence": 0.9,
                    "provenance": [{"source": "text_layer", "page_numbers": [1]}],
                },
                "authors": {"value": ["a", "b"]},
            },
        }
    )
    assert result.fields["authors"].value == ("a", "b")
    assert result.fields["title"].provenance == (
        Provenance(source="text_layer", page_numbers=(1,)),
    )
    with pytest.raises(ValidationError):
        ExtractedField(value="x", confidence=1.5)


class _Parser:
    name = "plain"
    version = "1"
    config: Mapping[str, object] = {}

    def parse(self, path: str | PathLike[str], *, media_type: str) -> ParsedDocument:
        return ParsedDocument(media_type=media_type)


class _Chunker:
    name = "window"
    version = "1"
    tokenizer_version = "t"
    config: Mapping[str, object] = {}

    def chunk(self, document: ParsedDocument) -> Sequence[Chunk]:
        return ()


class _Providers:
    dimensions = 2

    async def recognize(
        self, image: bytes, *, media_type: str, page_number: int
    ) -> OcrPageResult:
        return OcrPageResult(
            page=Page(page_number=page_number, ocr=True),
            execution=ProviderExecution(
                provider_id="p", external_model_id="m", model_revision="r"
            ),
        )

    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[0.0, 1.0] for _ in texts]

    async def rerank(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        return [0.0 for _ in documents]

    async def extract(
        self, document: ParsedDocument, chunks: Sequence[Chunk]
    ) -> ExtractionResult:
        return ExtractionResult(extractor="e", extractor_version="1")


def test_protocols_are_structural() -> None:
    assert isinstance(_Parser(), DocumentParser)
    assert isinstance(_Chunker(), Chunker)
    providers = _Providers()
    for protocol in (OcrProvider, EmbeddingProvider, RerankProvider, MetadataExtractor):
        assert isinstance(providers, protocol)
    assert not isinstance(object(), DocumentParser)
