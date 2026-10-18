"""The C-16 artifact envelope: deterministic writing and the S-14 reader."""

import gzip
import hashlib
import io
import json
import struct
import tracemalloc
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from doctools_m8 import artifact
from doctools_m8._jcs import canonicalize
from doctools_m8._jsonstream import JsonStreamDecoder
from doctools_m8.artifact import (
    ArtifactIdentity,
    ArtifactReadError,
    ChunkProducer,
    ChunksHeaderFields,
    EnvelopeHeader,
    OcrExecution,
    Producer,
    ReaderLimits,
    read_chunks,
    read_envelope,
    write_chunks,
    write_envelope,
)
from doctools_m8.models import BBox, Block, Chunk, Page, ParsedDocument
from doctools_m8.revisions import document_sha256

PACKAGE = Path(__file__).resolve().parents[1] / "doctools_m8"
SOURCE = "5" * 64
PARSE = "1" * 32
NORMALIZATION = "2" * 32
CHUNK_REVISION = "3" * 32


def _document(pages: int = 2, blocks: int = 2) -> ParsedDocument:
    return ParsedDocument(
        media_type="application/pdf",
        metadata={"title": "Report", "lang": "it"},
        pages=tuple(
            Page(
                page_number=p,
                width=612.0,
                height=792.5,
                ocr=p == pages,
                blocks=tuple(
                    Block(
                        block_id=f"p{p}-b{b}",
                        page_number=p,
                        kind="text",
                        text=f"pagina {p} blocco {b} — caffè",
                        bbox=BBox(x0=0, y0=0.25, x1=100.5, y1=1e-7 + 3),
                    )
                    for b in range(blocks)
                ),
            )
            for p in range(1, pages + 1)
        ),
    )


def _header(**overrides: object) -> EnvelopeHeader:
    fields: dict[str, object] = {
        "object_id": "obj-1",
        "run_id": "run-1",
        "attempt": 1,
        "source_sha256": SOURCE,
        "parse_revision": PARSE,
        "normalization_revision": NORMALIZATION,
        "producer": Producer(
            doctools_version="0.1.0",
            parser="pdfium",
            parser_version="4.30",
            config_hash="4" * 32,
        ),
        "ocr_execution": OcrExecution(
            profile_key="ocr-default",
            execution_identity="6" * 32,
            provider_id="prov-1",
            model="ocr-model",
            model_revision="r1",
            config_hash="7" * 32,
        ),
        "produced_at": "2026-10-18T12:00:00Z",
    }
    fields.update(overrides)
    return EnvelopeHeader.model_validate(fields)


IDENTITY = ArtifactIdentity(
    object_id="obj-1",
    run_id="run-1",
    attempt=1,
    source_sha256=SOURCE,
    parse_revision=PARSE,
    normalization_revision=NORMALIZATION,
)


def _write(
    document: ParsedDocument | None = None, header: EnvelopeHeader | None = None
) -> bytes:
    out = io.BytesIO()
    write_envelope(document or _document(), out, header=header or _header())
    return out.getvalue()


def _read(
    data: bytes,
    *,
    size: int | None = None,
    sha256: str | None = None,
    limits: ReaderLimits | None = None,
    identity: ArtifactIdentity = IDENTITY,
    clock: Callable[[], float] | None = None,
) -> artifact.Envelope:
    extra = {} if clock is None else {"clock": clock}
    return read_envelope(
        io.BytesIO(data),
        expected_size=len(data) if size is None else size,
        expected_sha256=hashlib.sha256(data).hexdigest() if sha256 is None else sha256,
        limits=limits or ReaderLimits(),
        expected_identity=identity,
        **extra,
    )


def _failure(data: bytes, **kwargs: object) -> ArtifactReadError:
    with pytest.raises(ArtifactReadError) as caught:
        _read(data, **kwargs)  # type: ignore[arg-type]
    return caught.value


def _gzip(payload: bytes) -> bytes:
    return gzip.compress(payload, mtime=0)


def _payload(data: bytes) -> dict[str, object]:
    value = json.loads(gzip.decompress(data))
    assert isinstance(value, dict)
    return value


def _repack(value: object) -> bytes:
    return _gzip(json.dumps(value).encode())


# -- writing --------------------------------------------------------------------


def test_round_trip_and_reported_reference() -> None:
    document = _document()
    out = io.BytesIO()
    written = write_envelope(document, out, header=_header())
    data = out.getvalue()
    assert written.size_bytes == len(data)
    assert written.sha256 == hashlib.sha256(data).hexdigest()
    assert written.document_sha256 == document_sha256(document)
    envelope = _read(data)
    assert envelope.document == document
    assert envelope.schema_version == 1
    assert envelope.counts.pages == 2
    assert envelope.counts.blocks == 4
    assert envelope.counts.ocr_pages == 1
    assert envelope.counts.text_bytes == document.text_bytes
    assert envelope.ocr_execution == _header().ocr_execution


def test_documents_larger_than_the_write_buffer_round_trip() -> None:
    document = _document(pages=40, blocks=40)
    data = _write(document)
    assert len(gzip.decompress(data)) > 4 * 64 * 1024
    assert _read(data).document == document
    assert data == _write(document)


def test_two_writes_with_the_same_produced_at_are_byte_identical() -> None:
    assert _write() == _write()
    assert _write() != _write(header=_header(produced_at="2026-10-18T12:00:01Z"))


def test_key_order_and_whitespace_do_not_change_the_bytes() -> None:
    plain = _document(pages=1, blocks=1).model_dump(mode="json")
    shuffled = json.loads(json.dumps(plain, sort_keys=True, indent=3))
    reordered = dict(reversed(list(shuffled.items())))
    other = ParsedDocument.model_validate(reordered)
    assert document_sha256(other) == document_sha256(_document(pages=1, blocks=1))
    assert _write(other) == _write(_document(pages=1, blocks=1))


def test_gzip_header_is_deterministic_even_for_named_files(tmp_path: Path) -> None:
    target = tmp_path / "document.v1.json.gz"
    with target.open("wb") as fh:
        write_envelope(_document(), fh, header=_header())
    data = target.read_bytes()
    assert data == _write()
    magic, method, flags, mtime, _, _ = struct.unpack("<2sBBIBB", data[:10])
    assert (magic, method, flags, mtime) == (b"\x1f\x8b", 8, 0, 0)


def test_the_envelope_is_jcs_json() -> None:
    raw = gzip.decompress(_write())
    assert raw == canonicalize(json.loads(raw))


def test_written_envelopes_decode_like_json_loads() -> None:
    raw = gzip.decompress(_write(_document(pages=3, blocks=5)))
    decoder = JsonStreamDecoder(ReaderLimits()._decoder())
    assert decoder.feed(raw) + decoder.close() == [json.loads(raw)]


# -- S-14 reader fixtures -----------------------------------------------------------


def test_truncated_artifacts_fail() -> None:
    data = _write()
    assert _failure(data[:-10], size=len(data)).code == "size_mismatch"
    assert _failure(data[:-10]).code == "truncated"
    assert _failure(data[:-3]).code == "truncated"
    assert _failure(data[:5]).code == "truncated"


def test_a_larger_artifact_than_its_reference_fails_while_streaming() -> None:
    data = _write() + b"\0" * 10
    assert _failure(data, size=len(data) - 10).code == "size_mismatch"


def test_bit_flips_fail_integrity_first() -> None:
    data = bytearray(_write())
    original = hashlib.sha256(data).hexdigest()
    data[len(data) // 2] ^= 0x10
    assert _failure(bytes(data), sha256=original).code == "sha256_mismatch"
    assert _failure(bytes(data)).code in {"gzip_invalid", "json_invalid"}


def test_a_flipped_trailer_fails_the_crc_check() -> None:
    data = bytearray(_write())
    data[-8] ^= 0x01
    assert _failure(bytes(data)).code == "gzip_invalid"
    data = bytearray(_write())
    data[-1] ^= 0x01
    assert _failure(bytes(data)).code == "gzip_invalid"


@pytest.mark.parametrize("version", [2, 0, "1", True, None])
def test_unsupported_schema_versions_fail(version: object) -> None:
    value = _payload(_write())
    value["schema_version"] = version
    assert _failure(_repack(value)).code == "artifact_unsupported"
    del value["schema_version"]
    assert _failure(_repack(value)).code == "artifact_unsupported"


def test_a_non_object_artifact_is_schema_invalid() -> None:
    assert _failure(_gzip(b"[1]")).code == "schema_invalid"


def test_unknown_fields_are_schema_invalid() -> None:
    value = _payload(_write())
    value["extra"] = 1
    error = _failure(_repack(value))
    assert error.code == "schema_invalid"
    assert "extra" in error.detail


def test_a_correctly_hashed_compression_bomb_stops_at_the_ceiling() -> None:
    compressor = zlib.compressobj(9, zlib.DEFLATED, 31)
    block = b"a" * (1 << 20)
    parts = [compressor.compress(b'{"document":"')]
    parts += [compressor.compress(block) for _ in range(256)]
    parts.append(compressor.compress(b'"}') + compressor.flush())
    bomb = b"".join(parts)
    bomb = bomb[:4] + b"\0\0\0\0" + bomb[8:]  # mtime = 0, as the reader requires
    assert len(bomb) < 512 * 1024
    limits = ReaderLimits(max_decompressed_bytes=4 << 20, max_string_bytes=1 << 30)
    tracemalloc.start()
    try:
        error = _failure(bomb, limits=limits)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert (error.code, error.limit) == ("limit_exceeded", "max_decompressed_bytes")
    assert peak < (4 << 20) * 2 + (1 << 20)
    default = _failure(bomb)
    assert (default.code, default.limit) == ("limit_exceeded", "max_string_bytes")


def test_correctly_hashed_deep_nesting_fails() -> None:
    error = _failure(_gzip(b"[" * 100_000))
    assert (error.code, error.limit) == ("limit_exceeded", "max_depth")


@pytest.mark.parametrize(
    ("field", "value"),
    [("pages", 3), ("blocks", 5), ("text_bytes", 1), ("ocr_pages", 0)],
)
def test_dishonest_counts_fail(field: str, value: int) -> None:
    payload = _payload(_write())
    counts = payload["counts"]
    assert isinstance(counts, dict)
    counts[field] = value
    assert _failure(_repack(payload)).code == "counts_mismatch"


def test_a_dishonest_document_sha256_fails() -> None:
    payload = _payload(_write())
    payload["document_sha256"] = "0" * 64
    assert _failure(_repack(payload)).code == "document_sha256_mismatch"


def test_concatenated_gzip_members_fail() -> None:
    data = _write()
    assert _failure(data + data).code == "multiple_members"
    assert _failure(data + b"\0").code == "multiple_members"
    small = ReaderLimits(chunk_bytes=7)
    assert _failure(data + data, limits=small).code == "multiple_members"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("object_id", "obj-2"),
        ("run_id", "run-2"),
        ("attempt", 2),
        ("source_sha256", "9" * 64),
        ("parse_revision", "9" * 32),
        ("normalization_revision", None),
    ],
)
def test_another_objects_valid_artifact_fails(field: str, value: object) -> None:
    foreign = _write(header=_header(**{field: value}))
    error = _failure(foreign)
    assert error.code == "identity_mismatch"
    assert field in error.detail


@pytest.mark.parametrize(
    ("offset", "byte"),
    [(0, 0x00), (2, 0x07), (3, 0x08), (4, 0x01)],
    ids=["magic", "method", "fname-flag", "mtime"],
)
def test_foreign_gzip_headers_fail(offset: int, byte: int) -> None:
    data = bytearray(_write())
    data[offset] = byte
    assert _failure(bytes(data)).code == "gzip_invalid"


def test_corrupt_deflate_data_fails() -> None:
    data = _write()
    corrupt = data[:10] + b"\xff" * 32 + data[42:]
    assert _failure(corrupt).code == "gzip_invalid"


def test_the_compressed_ceiling_applies_before_reading() -> None:
    error = _failure(_write(), limits=ReaderLimits(max_compressed_bytes=16))
    assert (error.code, error.limit) == ("limit_exceeded", "max_compressed_bytes")


@pytest.mark.parametrize("chunk_bytes", [1, 2, 9, 11, 64])
def test_tiny_read_and_inflate_steps_round_trip(chunk_bytes: int) -> None:
    data = _write()
    envelope = _read(data, limits=ReaderLimits(chunk_bytes=chunk_bytes))
    assert envelope.document == _document()


@pytest.mark.parametrize(
    ("limits", "limit"),
    [
        (ReaderLimits(max_pages=1), "max_pages"),
        (ReaderLimits(max_blocks=3), "max_blocks"),
        (ReaderLimits(max_string_bytes=8), "max_string_bytes"),
        (ReaderLimits(max_depth=4), "max_depth"),
        (ReaderLimits(max_object_members=5), "max_object_members"),
        (ReaderLimits(max_array_items=1), "max_array_items"),
        (ReaderLimits(max_total_keys=20), "max_total_keys"),
        (ReaderLimits(max_text_bytes=100), "max_text_bytes"),
        (ReaderLimits(max_number_length=2), "max_number_length"),
    ],
)
def test_page_block_and_lexical_counts_one_past_their_limits(
    limits: ReaderLimits, limit: str
) -> None:
    error = _failure(_write(), limits=limits)
    assert (error.code, error.limit) == ("limit_exceeded", limit)


def test_counts_at_their_limits_pass() -> None:
    assert _read(_write(), limits=ReaderLimits(max_pages=2, max_blocks=4)).document


def test_the_wall_clock_bound_stops_a_stalled_read() -> None:
    ticks = iter(range(1_000_000))

    def clock() -> float:
        return float(next(ticks))

    error = _failure(_write(_document(pages=4, blocks=8)), clock=clock)
    assert error.code == "deadline_exceeded"


def test_the_number_length_ceiling_stays_parseable() -> None:
    with pytest.raises(ValidationError):
        ReaderLimits(max_number_length=4001)


# -- chunks.v1.jsonl.gz ---------------------------------------------------------------


def _chunks(count: int = 3) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{i}",
            ordinal=i,
            kind="text",
            text=f"chunk {i} è",
            page_numbers=(1,),
            block_ids=(f"p1-b{i}",),
            token_count=4,
        )
        for i in range(count)
    ]


def _chunks_header(**overrides: object) -> ChunksHeaderFields:
    fields: dict[str, object] = {
        "object_id": "obj-1",
        "run_id": "run-1",
        "attempt": 1,
        "source_sha256": SOURCE,
        "parse_revision": PARSE,
        "normalization_revision": NORMALIZATION,
        "producer": ChunkProducer(
            doctools_version="0.1.0",
            chunker="window",
            chunker_version="1.0",
            tokenizer_version="tok-3",
            config_hash="8" * 32,
        ),
        "document_sha256": "a" * 64,
        "chunk_revision": CHUNK_REVISION,
        "produced_at": "2026-10-18T12:00:00.5Z",
    }
    fields.update(overrides)
    return ChunksHeaderFields.model_validate(fields)


def _write_chunks(chunks: list[Chunk] | None = None, **overrides: object) -> bytes:
    out = io.BytesIO()
    written = write_chunks(
        _chunks() if chunks is None else chunks, out, header=_chunks_header(**overrides)
    )
    assert written.document_sha256 == "a" * 64
    return out.getvalue()


def _read_chunks(
    data: bytes,
    *,
    limits: ReaderLimits | None = None,
    chunk_revision: str = CHUNK_REVISION,
) -> artifact.ChunksArtifact:
    return read_chunks(
        io.BytesIO(data),
        expected_size=len(data),
        expected_sha256=hashlib.sha256(data).hexdigest(),
        limits=limits or ReaderLimits(),
        expected_identity=IDENTITY,
        expected_chunk_revision=chunk_revision,
    )


def _chunks_failure(data: bytes, **kwargs: object) -> ArtifactReadError:
    with pytest.raises(ArtifactReadError) as caught:
        _read_chunks(data, **kwargs)  # type: ignore[arg-type]
    return caught.value


def test_chunks_round_trip_deterministically() -> None:
    data = _write_chunks()
    assert data == _write_chunks()
    result = _read_chunks(data)
    assert result.chunks == tuple(_chunks())
    assert result.header.counts.chunks == 3
    assert result.header.chunk_revision == CHUNK_REVISION
    lines = gzip.decompress(data).split(b"\n")
    assert len(lines) == 5 and lines[-1] == b""
    assert _read_chunks(_write_chunks([])).chunks == ()


def test_chunk_count_one_past_its_limit_fails() -> None:
    error = _chunks_failure(_write_chunks(), limits=ReaderLimits(max_chunks=2))
    assert (error.code, error.limit) == ("limit_exceeded", "max_chunks")
    assert _read_chunks(_write_chunks(), limits=ReaderLimits(max_chunks=3)).chunks


def _chunk_lines(data: bytes) -> list[dict[str, object]]:
    return [json.loads(line) for line in gzip.decompress(data).splitlines()]


def _repack_lines(lines: list[dict[str, object]]) -> bytes:
    return _gzip(b"".join(json.dumps(line).encode() + b"\n" for line in lines))


def test_chunks_dishonest_counts_and_order_fail() -> None:
    lines = _chunk_lines(_write_chunks())
    counts = lines[0]["counts"]
    assert isinstance(counts, dict)
    counts["chunks"] = 2
    assert _chunks_failure(_repack_lines(lines)).code == "counts_mismatch"
    reordered = _write_chunks([_chunks()[1], _chunks()[0]])
    assert _chunks_failure(reordered).code == "schema_invalid"
    duplicate = [_chunks()[0], _chunks()[1].model_copy(update={"chunk_id": "c0"})]
    assert _chunks_failure(_write_chunks(duplicate)).code == "schema_invalid"


def test_chunks_identity_and_version_fail() -> None:
    error = _chunks_failure(_write_chunks(), chunk_revision="9" * 32)
    assert (error.code, error.detail) == ("identity_mismatch", "chunk_revision differs")
    foreign = _write_chunks(object_id="obj-2")
    assert _chunks_failure(foreign).code == "identity_mismatch"
    lines = _chunk_lines(_write_chunks())
    lines[0]["schema_version"] = 2
    assert _chunks_failure(_repack_lines(lines)).code == "artifact_unsupported"


def test_chunks_without_a_header_or_with_a_bad_line_fail() -> None:
    assert _chunks_failure(_gzip(b"")).code == "truncated"
    lines = _chunk_lines(_write_chunks())
    lines[1]["kind"] = "paragraph"
    assert _chunks_failure(_repack_lines(lines)).code == "schema_invalid"
    lines = _chunk_lines(_write_chunks())
    no_newline = _gzip(b"\n".join(json.dumps(line).encode() for line in lines))
    assert _chunks_failure(no_newline).code == "truncated"


# -- fuzzing and structure -------------------------------------------------------------

VALID = _write(_document(pages=1, blocks=2))


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(st.integers(0, len(VALID) - 1), st.integers(1, 255)),
        min_size=1,
        max_size=4,
        unique_by=lambda flip: flip[0],
    ),
    st.booleans(),
    st.integers(1, 64),
)
def test_mutated_artifacts_only_raise_the_typed_error(
    flips: list[tuple[int, int]], rehash: bool, chunk_bytes: int
) -> None:
    data = bytearray(VALID)
    for index, mask in flips:
        data[index] ^= mask
    sha256 = None if rehash else hashlib.sha256(VALID).hexdigest()
    with pytest.raises(ArtifactReadError):
        _read(bytes(data), sha256=sha256, limits=ReaderLimits(chunk_bytes=chunk_bytes))


@settings(max_examples=200, deadline=None)
@given(st.binary(max_size=300))
def test_arbitrary_bytes_only_raise_the_typed_error(data: bytes) -> None:
    with pytest.raises(ArtifactReadError):
        _read(data)


@settings(max_examples=100, deadline=None)
@given(st.binary(max_size=200))
def test_arbitrary_gzip_payloads_only_raise_the_typed_error(payload: bytes) -> None:
    with pytest.raises(ArtifactReadError):
        _read(_gzip(payload))


def _module_sources() -> Iterator[tuple[str, str]]:
    for name in ("artifact.py", "_jsonstream.py"):
        yield name, (PACKAGE / name).read_text(encoding="utf-8")


def test_no_json_load_touches_artifact_bytes() -> None:
    for name, source in _module_sources():
        assert "json.load" not in source, name
        assert "import json" not in source, name
        assert "from json" not in source, name
