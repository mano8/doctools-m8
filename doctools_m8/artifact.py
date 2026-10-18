"""
C-16 artifact envelope: ``document.v1.json.gz`` and ``chunks.v1.jsonl.gz``.

**Writing** is deterministic: one gzip member with header ``mtime = 0``, no
file name and a fixed compression level, carrying JCS JSON; ``produced_at``
comes from the caller (the producing invocation's manifest), never from a
clock. Identical inputs yield byte-identical artifacts.

**Reading** bounds everything before trusting anything (D-34, ``S-14``): the
compressed bytes are counted and hashed while streaming; the gzip header and
trailer are checked by this module; inflation runs through
``zlib.decompressobj`` with ``max_length`` per call, so no call produces more
than one bounded chunk, and a second gzip member is refused; decoding runs
through :mod:`doctools_m8._jsonstream`, whose limits trip before the governed
token is allocated. Only after the size and SHA-256 match is
``schema_version`` checked, then the schema, then the declared counts against
measured values, then the identity fields against the caller's expectation.
Every failure raises :class:`ArtifactReadError` and no partial data is ever
returned.
"""

from __future__ import annotations

import gzip
import hashlib
import struct
import time
import zlib
from collections.abc import Callable, Iterable, Iterator
from typing import IO, Annotated, Literal, NoReturn, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    ValidationError,
    model_validator,
)

from doctools_m8._jcs import iter_canonical
from doctools_m8._jsonstream import (
    DecoderLimits,
    JsonStreamDecoder,
    JsonStreamError,
    PathCap,
)
from doctools_m8.models import Chunk, Identifier, Items, ParsedDocument
from doctools_m8.revisions import document_sha256

__all__ = [
    "CHUNKS_FILENAME",
    "COMPRESSION_LEVEL",
    "DOCUMENT_FILENAME",
    "MAX_DOCUMENT_JSON_BYTES",
    "SCHEMA_VERSION",
    "ArtifactIdentity",
    "ArtifactReadError",
    "ChunkCounts",
    "ChunkProducer",
    "ChunksArtifact",
    "ChunksHeader",
    "ChunksHeaderFields",
    "DocumentCounts",
    "Envelope",
    "EnvelopeHeader",
    "OcrExecution",
    "Producer",
    "ReaderLimits",
    "WrittenArtifact",
    "read_chunks",
    "read_envelope",
    "write_chunks",
    "write_envelope",
]

SCHEMA_VERSION: Literal[1] = 1
DOCUMENT_FILENAME = "document.v1.json.gz"
CHUNKS_FILENAME = "chunks.v1.jsonl.gz"
COMPRESSION_LEVEL = 6
"""The fixed gzip level of every artifact (part of the deterministic encoding)."""
MAX_DOCUMENT_JSON_BYTES = 256 * 1024 * 1024
"""The default ceiling on an artifact's decompressed bytes."""

_GZIP_HEADER = struct.Struct("<2sBBIBB")
_GZIP_TRAILER = struct.Struct("<II")
_WRITE_BUFFER = 64 * 1024

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
Label = Annotated[str, StringConstraints(min_length=1, max_length=255)]
Timestamp = Annotated[
    str,
    StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$"),
]
NonNegative = Annotated[int, Field(ge=0)]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ArtifactReadError(ValueError):
    """
    An artifact failed verification; nothing it contains may be used.

    ``code`` is one of ``size_mismatch``, ``sha256_mismatch``, ``gzip_invalid``,
    ``multiple_members``, ``truncated``, ``json_invalid``, ``limit_exceeded``,
    ``deadline_exceeded``, ``artifact_unsupported``, ``schema_invalid``,
    ``counts_mismatch``, ``document_sha256_mismatch`` and ``identity_mismatch``;
    ``limit`` names the tripped ceiling for ``limit_exceeded``.
    """

    def __init__(self, code: str, detail: str, *, limit: str | None = None) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.limit = limit


# -- models --------------------------------------------------------------------


class Producer(_Model):
    """The software that produced a document envelope."""

    doctools_version: Label
    parser: Identifier
    parser_version: Label
    config_hash: Revision


class OcrExecution(_Model):
    """The pinned OCR execution of a run (C-07), or absent when none ran."""

    profile_key: Identifier
    execution_identity: Revision
    provider_id: Identifier
    model: Label
    model_revision: Label
    config_hash: Revision


class DocumentCounts(_Model):
    """Declared counts; the reader compares them with measured values."""

    pages: NonNegative
    blocks: NonNegative
    text_bytes: NonNegative
    ocr_pages: NonNegative


class _Identity(_Model):
    object_id: Identifier
    run_id: Identifier
    attempt: PositiveInt
    source_sha256: Sha256
    parse_revision: Revision
    normalization_revision: Revision | None


class ArtifactIdentity(_Identity):
    """The identity an artifact must carry, taken from the database rows."""


class EnvelopeHeader(_Identity):
    """The producer-supplied fields of a document envelope."""

    producer: Producer
    ocr_execution: OcrExecution | None = None
    produced_at: Timestamp
    partial: bool = False


class Envelope(EnvelopeHeader):
    """A decoded ``document.v1.json.gz``."""

    schema_version: Literal[1]
    document_sha256: Sha256
    counts: DocumentCounts
    document: ParsedDocument


class ChunkProducer(_Model):
    """The software that produced a chunks artifact."""

    doctools_version: Label
    chunker: Identifier
    chunker_version: Label
    tokenizer_version: Label
    config_hash: Revision


class ChunkCounts(_Model):
    """Declared counts of a chunks artifact."""

    chunks: NonNegative
    text_bytes: NonNegative


class ChunksHeaderFields(_Identity):
    """The producer-supplied fields of a chunks artifact's header line."""

    normalization_revision: Revision
    producer: ChunkProducer
    ocr_execution: OcrExecution | None = None
    document_sha256: Sha256
    chunk_revision: Revision
    produced_at: Timestamp
    partial: bool = False


class ChunksHeader(ChunksHeaderFields):
    """The decoded header line of ``chunks.v1.jsonl.gz``."""

    schema_version: Literal[1]
    counts: ChunkCounts


class ChunksArtifact(_Model):
    """A decoded ``chunks.v1.jsonl.gz``."""

    header: ChunksHeader
    chunks: Items[Chunk]


class WrittenArtifact(_Model):
    """What a writer reports: the ``DerivedArtifactRef`` values and the content hash."""

    size_bytes: NonNegative
    sha256: Sha256
    document_sha256: Sha256


class ReaderLimits(_Model):
    """Reader-owned ceilings (D-34); the defaults suit V1 documents."""

    max_compressed_bytes: PositiveInt = 64 * 1024 * 1024
    max_decompressed_bytes: PositiveInt = MAX_DOCUMENT_JSON_BYTES
    chunk_bytes: PositiveInt = 64 * 1024
    """Bytes read per call, and the ``max_length`` of every inflate call."""
    max_depth: PositiveInt = 32
    max_string_bytes: PositiveInt = 16 * 1024 * 1024
    max_number_length: PositiveInt = 32
    max_object_members: PositiveInt = 1024
    max_array_items: PositiveInt = 1_000_000
    max_total_keys: PositiveInt = 16_000_000
    max_text_bytes: PositiveInt = MAX_DOCUMENT_JSON_BYTES
    max_pages: PositiveInt = 10_000
    max_blocks: PositiveInt = 1_000_000
    max_chunks: PositiveInt = 1_000_000
    max_seconds: PositiveFloat = 60.0

    @model_validator(mode="after")
    def _number_length_parses(self) -> Self:
        if self.max_number_length > 4000:
            raise ValueError("max_number_length must stay below the int digit cap")
        return self

    def _decoder(self, *caps: PathCap) -> DecoderLimits:
        return DecoderLimits(
            max_depth=self.max_depth,
            max_string_bytes=self.max_string_bytes,
            max_number_length=self.max_number_length,
            max_object_members=self.max_object_members,
            max_array_items=self.max_array_items,
            max_total_keys=self.max_total_keys,
            max_text_bytes=self.max_text_bytes,
            path_caps=caps,
        )


# -- writing -------------------------------------------------------------------


class _CountingSink:
    """A write-only file object that hashes and counts what passes through."""

    def __init__(self, fh: IO[bytes]) -> None:
        self._fh = fh
        self.size = 0
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        self._fh.write(data)
        self.digest.update(data)
        self.size += len(data)
        return len(data)

    def flush(self) -> None:
        self._fh.flush()


def _write_gzip(pieces: Iterable[str], fh: IO[bytes]) -> _CountingSink:
    sink = _CountingSink(fh)
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=COMPRESSION_LEVEL,
        fileobj=sink,
        mtime=0,
    ) as stream:
        buffer: list[str] = []
        size = 0
        for piece in pieces:
            buffer.append(piece)
            size += len(piece)
            if size >= _WRITE_BUFFER:
                stream.write("".join(buffer).encode("utf-8"))
                buffer, size = [], 0
        stream.write("".join(buffer).encode("utf-8"))
    sink.flush()
    return sink


def write_envelope(
    document: ParsedDocument, fh: IO[bytes], *, header: EnvelopeHeader
) -> WrittenArtifact:
    """
    Stream ``document`` as a deterministic ``document.v1.json.gz`` into ``fh``.

    The counts and ``document_sha256`` are computed here, never taken from the
    caller. Returns the ``DerivedArtifactRef`` size and SHA-256 of the
    compressed bytes and the document's content hash.
    """
    content_hash = document_sha256(document)
    envelope = Envelope(
        **dict(header),
        schema_version=SCHEMA_VERSION,
        document_sha256=content_hash,
        counts=DocumentCounts(
            pages=len(document.pages),
            blocks=document.block_count,
            text_bytes=document.text_bytes,
            ocr_pages=document.ocr_pages,
        ),
        document=document,
    )
    sink = _write_gzip(iter_canonical(envelope.model_dump(mode="json")), fh)
    return WrittenArtifact(
        size_bytes=sink.size,
        sha256=sink.digest.hexdigest(),
        document_sha256=content_hash,
    )


def write_chunks(
    chunks: Iterable[Chunk], fh: IO[bytes], *, header: ChunksHeaderFields
) -> WrittenArtifact:
    """Write a deterministic ``chunks.v1.jsonl.gz``: a header line, then one chunk per line."""
    chunk_list = list(chunks)
    full_header = ChunksHeader(
        **dict(header),
        schema_version=SCHEMA_VERSION,
        counts=ChunkCounts(
            chunks=len(chunk_list),
            text_bytes=sum(len(chunk.text.encode("utf-8")) for chunk in chunk_list),
        ),
    )

    def lines() -> Iterator[str]:
        for value in (full_header, *chunk_list):
            yield from iter_canonical(value.model_dump(mode="json"))
            yield "\n"

    sink = _write_gzip(lines(), fh)
    return WrittenArtifact(
        size_bytes=sink.size,
        sha256=sink.digest.hexdigest(),
        document_sha256=header.document_sha256,
    )


# -- reading -------------------------------------------------------------------


class _GzipMember:
    """Inflate exactly one gzip member in bounded steps, checking header and trailer."""

    def __init__(self, limits: ReaderLimits) -> None:
        self._limits = limits
        self._header = b""
        self._inflater = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
        self._trailer = b""
        self._crc = 0
        self.size = 0

    def feed(self, data: bytes) -> Iterator[bytes]:
        """Yield the decompressed pieces of ``data``, each at most ``chunk_bytes``."""
        if len(self._header) < _GZIP_HEADER.size:
            need = _GZIP_HEADER.size - len(self._header)
            self._header += data[:need]
            data = data[need:]
            if len(self._header) == _GZIP_HEADER.size:
                self._check_header()
        if self._inflater.eof:
            self._add_trailer(data)
            return
        yield from self._inflate(data)

    def _check_header(self) -> None:
        magic, method, flags, mtime, _, _ = _GZIP_HEADER.unpack(self._header)
        if magic != b"\x1f\x8b" or method != 8:
            raise ArtifactReadError("gzip_invalid", "not a gzip deflate stream")
        if flags or mtime:
            raise ArtifactReadError(
                "gzip_invalid", "gzip header carries flags or a modification time"
            )

    def _inflate(self, data: bytes) -> Iterator[bytes]:
        step = self._limits.chunk_bytes
        while True:
            try:
                out = self._inflater.decompress(data, step)
            except zlib.error as exc:
                raise ArtifactReadError("gzip_invalid", "corrupt deflate data") from exc
            if self.size + len(out) > self._limits.max_decompressed_bytes:
                raise _limit("max_decompressed_bytes")
            self.size += len(out)
            self._crc = zlib.crc32(out, self._crc)
            if out:
                yield out
            data = self._inflater.unconsumed_tail
            if self._inflater.eof:
                self._add_trailer(self._inflater.unused_data)
                return
            if not data and len(out) < step:
                return

    def _add_trailer(self, data: bytes) -> None:
        self._trailer += data
        if len(self._trailer) > _GZIP_TRAILER.size:
            raise ArtifactReadError(
                "multiple_members", "data follows the first gzip member"
            )

    def finish(self) -> None:
        """Check that the member ended and its CRC-32 and length match."""
        if not self._inflater.eof or len(self._trailer) != _GZIP_TRAILER.size:
            raise ArtifactReadError("truncated", "the gzip member is incomplete")
        crc, size = _GZIP_TRAILER.unpack(self._trailer)
        if crc != self._crc or size != self.size & 0xFFFFFFFF:
            raise ArtifactReadError("gzip_invalid", "gzip trailer mismatch")


def _limit(name: str) -> ArtifactReadError:
    return ArtifactReadError("limit_exceeded", f"{name} exceeded", limit=name)


class _Integrity:
    """Count and hash the compressed bytes against the expected reference."""

    def __init__(self, fh: IO[bytes], size: int, sha256: str, step: int) -> None:
        self._fh = fh
        self._expected_size = size
        self._expected_sha256 = sha256
        self._step = step
        self._digest = hashlib.sha256()
        self._size = 0

    def chunks(self) -> Iterator[bytes]:
        """Yield the compressed bytes, refusing to read past the expected size."""
        while True:
            want = min(self._step, self._expected_size - self._size + 1)
            data = self._fh.read(want)
            if not data:
                return
            self._size += len(data)
            if self._size > self._expected_size:
                raise ArtifactReadError(
                    "size_mismatch", "artifact is larger than expected"
                )
            self._digest.update(data)
            yield data

    def verify(self) -> None:
        """Drain what is left (bounded by the expected size) and compare."""
        for _ in self.chunks():
            pass
        if self._size != self._expected_size:
            raise ArtifactReadError("size_mismatch", "artifact size differs")
        if self._digest.hexdigest() != self._expected_sha256:
            raise ArtifactReadError("sha256_mismatch", "artifact SHA-256 differs")


def _decode(
    fh: IO[bytes],
    *,
    expected_size: int,
    expected_sha256: str,
    limits: ReaderLimits,
    decoder_limits: DecoderLimits,
    sequence: bool,
    clock: Callable[[], float],
) -> list[object]:
    """Return every decoded top-level value once integrity has been verified."""
    if expected_size > limits.max_compressed_bytes:
        raise _limit("max_compressed_bytes")
    deadline = clock() + limits.max_seconds
    integrity = _Integrity(fh, expected_size, expected_sha256, limits.chunk_bytes)
    member = _GzipMember(limits)
    decoder = JsonStreamDecoder(
        decoder_limits, sequence=sequence, deadline=deadline, clock=clock
    )
    values: list[object] = []
    try:
        for compressed in integrity.chunks():
            for piece in member.feed(compressed):
                values.extend(decoder.feed(piece))
        member.finish()
        values.extend(decoder.close())
    except JsonStreamError as exc:
        _fail_after_integrity(
            integrity, ArtifactReadError(exc.code, exc.detail, limit=exc.limit)
        )
    except ArtifactReadError as exc:
        _fail_after_integrity(integrity, exc)
    integrity.verify()
    return values


def _fail_after_integrity(integrity: _Integrity, error: ArtifactReadError) -> NoReturn:
    """Report an integrity failure in preference to a decoding one."""
    if error.code not in ("deadline_exceeded", "size_mismatch"):
        integrity.verify()
    raise error


def _check_schema_version(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactReadError("schema_invalid", "the artifact is not a JSON object")
    version = value.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ArtifactReadError(
            "artifact_unsupported", f"unsupported schema_version {version!r}"
        )
    return value


def _validate[M: BaseModel](model: type[M], value: object) -> M:
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        first = exc.errors(include_input=False, include_url=False)[0]
        location = ".".join(map(str, first["loc"]))
        raise ArtifactReadError(
            "schema_invalid", f"{exc.error_count()} error(s); first at {location}"
        ) from None


def _check_identity(
    found: _Identity, expected: ArtifactIdentity, *extra: tuple[str, object, object]
) -> None:
    for name in ArtifactIdentity.model_fields:
        if getattr(found, name) != getattr(expected, name):
            raise ArtifactReadError("identity_mismatch", f"{name} differs")
    for name, got, want in extra:
        if got != want:
            raise ArtifactReadError("identity_mismatch", f"{name} differs")


def read_envelope(
    fh: IO[bytes],
    *,
    expected_size: int,
    expected_sha256: str,
    limits: ReaderLimits,
    expected_identity: ArtifactIdentity,
    clock: Callable[[], float] = time.monotonic,
) -> Envelope:
    """Read and fully verify a ``document.v1.json.gz`` (D-26, D-34, ``S-14``)."""
    caps = (
        PathCap(("document", "pages"), limits.max_pages, "max_pages"),
        PathCap(("document", "pages", "*", "blocks"), limits.max_blocks, "max_blocks"),
    )
    (value,) = _decode(
        fh,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        limits=limits,
        decoder_limits=limits._decoder(*caps),
        sequence=False,
        clock=clock,
    )
    envelope = _validate(Envelope, _check_schema_version(value))
    document = envelope.document
    measured = DocumentCounts(
        pages=len(document.pages),
        blocks=document.block_count,
        text_bytes=document.text_bytes,
        ocr_pages=document.ocr_pages,
    )
    if envelope.counts != measured:
        raise ArtifactReadError(
            "counts_mismatch", "declared counts differ from content"
        )
    if document_sha256(document) != envelope.document_sha256:
        raise ArtifactReadError(
            "document_sha256_mismatch", "document_sha256 differs from the content"
        )
    _check_identity(envelope, expected_identity)
    return envelope


def read_chunks(
    fh: IO[bytes],
    *,
    expected_size: int,
    expected_sha256: str,
    limits: ReaderLimits,
    expected_identity: ArtifactIdentity,
    expected_chunk_revision: str,
    clock: Callable[[], float] = time.monotonic,
) -> ChunksArtifact:
    """Read and fully verify a ``chunks.v1.jsonl.gz`` under the same ceilings."""
    values = _decode(
        fh,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        limits=limits,
        decoder_limits=limits._decoder(
            PathCap((), limits.max_chunks + 1, "max_chunks")
        ),
        sequence=True,
        clock=clock,
    )
    if not values:
        raise ArtifactReadError("truncated", "the chunks artifact has no header line")
    header = _validate(ChunksHeader, _check_schema_version(values[0]))
    chunks = tuple(_validate(Chunk, value) for value in values[1:])
    _check_chunks(header, chunks)
    _check_identity(
        header,
        expected_identity,
        ("chunk_revision", header.chunk_revision, expected_chunk_revision),
    )
    return ChunksArtifact(header=header, chunks=chunks)


def _check_chunks(header: ChunksHeader, chunks: tuple[Chunk, ...]) -> None:
    measured = ChunkCounts(
        chunks=len(chunks),
        text_bytes=sum(len(chunk.text.encode("utf-8")) for chunk in chunks),
    )
    if header.counts != measured:
        raise ArtifactReadError(
            "counts_mismatch", "declared counts differ from content"
        )
    if [chunk.ordinal for chunk in chunks] != list(range(len(chunks))):
        raise ArtifactReadError("schema_invalid", "chunk ordinals must be 0..n-1")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ArtifactReadError("schema_invalid", "duplicate chunk_id")
