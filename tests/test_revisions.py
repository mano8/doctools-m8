"""C-07 revision identities."""

import hashlib
import json

import pytest

from doctools_m8 import revisions
from doctools_m8.models import Block, Page, ParsedDocument

SOURCE = "ab" * 32


def _h(*parts: object) -> str:
    text = json.dumps(list(parts), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def test_revision_hash_is_the_32_hex_prefix_of_sha256_jcs() -> None:
    assert revisions.revision_hash("a", 1, {"z": 1, "a": [True, None]}) == _h(
        "a", 1, {"a": [True, None], "z": 1}
    )
    assert len(revisions.revision_hash()) == revisions.REVISION_HEX_LENGTH


def test_config_order_and_whitespace_do_not_change_revisions() -> None:
    one = json.loads('{"dpi": 150, "lang": ["en", "it"], "x": {"b": 1, "a": 2}}')
    two = json.loads('{ "x" : {"a":2,"b":1},\n "lang":["en","it"], "dpi":150 }')
    assert revisions.config_hash(one) == revisions.config_hash(two)
    assert revisions.parse_revision(
        SOURCE, "pdfium", "4.30", one
    ) == revisions.parse_revision(SOURCE, "pdfium", "4.30", two)


def test_each_input_changes_the_parse_revision() -> None:
    base = revisions.parse_revision(SOURCE, "pdfium", "4.30", {"dpi": 150})
    assert base == _h(SOURCE, "pdfium", "4.30", {"dpi": 150})
    variants = {
        revisions.parse_revision("cd" * 32, "pdfium", "4.30", {"dpi": 150}),
        revisions.parse_revision(SOURCE, "plain", "4.30", {"dpi": 150}),
        revisions.parse_revision(SOURCE, "pdfium", "4.31", {"dpi": 150}),
        revisions.parse_revision(SOURCE, "pdfium", "4.30", {"dpi": 300}),
    }
    assert base not in variants
    assert len(variants) == 4


def test_document_sha256_is_sha256_of_the_jcs_document() -> None:
    document = ParsedDocument(
        media_type="text/plain",
        pages=(
            Page(
                page_number=1, blocks=(Block(block_id="b1", page_number=1, text="hi"),)
            ),
        ),
    )
    expected = hashlib.sha256(
        json.dumps(
            document.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    assert revisions.document_sha256(document) == expected


def test_normalization_revision_with_and_without_ocr() -> None:
    parse = revisions.parse_revision(SOURCE, "pdfium", "4.30", {})
    doc = "ef" * 32
    identity = revisions.ocr_execution_identity("prov-1", "model-x", "rev-7")
    assert identity == _h("prov-1", "model-x", "rev-7")
    assert revisions.normalization_revision(parse, doc) == _h(parse, "no-ocr", doc)
    assert revisions.normalization_revision(
        parse,
        doc,
        ocr_profile_key="ocr-default",
        ocr_execution=identity,
        ocr_config={"lang": "en"},
    ) == _h(parse, "ocr-default", identity, {"lang": "en"}, doc)


def test_normalization_revision_needs_every_ocr_input() -> None:
    with pytest.raises(ValueError, match="together"):
        revisions.normalization_revision("a" * 32, "b" * 64, ocr_profile_key="ocr")


def test_execution_identity_is_pinned_for_the_run() -> None:
    first = revisions.pin_execution_identity(None, "a" * 32)
    assert revisions.pin_execution_identity(first, "a" * 32) == first
    with pytest.raises(revisions.ExecutionIdentityChangedError) as caught:
        revisions.pin_execution_identity(first, "b" * 32)
    assert caught.value.code == "execution_identity_changed"


def test_chunk_index_and_embedding_space_revisions() -> None:
    norm = "1" * 32
    chunk = revisions.chunk_revision(norm, "window", "1.0", "tok-3", {"size": 512})
    assert chunk == _h(norm, "window", "1.0", "tok-3", {"size": 512})
    space = revisions.embedding_space_id("prov-1", "embed-x", "rev-2", 768, "2" * 32)
    assert space == _h("prov-1", "embed-x", "rev-2", 768, "2" * 32)
    assert revisions.index_revision(chunk, "embed-default", space) == _h(
        chunk, "embed-default", space
    )
