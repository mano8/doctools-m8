"""
Bounded incremental JSON decoder (D-34).

The decoder is fed byte chunks and builds Python objects only from tokens it
has accepted. Every lexical limit is checked **before** the token it governs
is allocated: a container is refused before its ``dict``/``list`` exists, an
object member before its key is read, an array item before it starts, and a
string or number literal before the bytes that would carry it past its limit
are buffered. Peak memory is therefore a function of the limits, not of the
input. The wall-clock deadline is checked between tokens and between chunks,
so the overshoot is bounded by one chunk's work.

The grammar is RFC 8259 JSON with three deliberate refusals: duplicate object
keys, lone surrogates and non-finite numbers. In sequence mode (JSON Lines)
every top-level value must be followed by a newline.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

__all__ = ["DecoderLimits", "JsonStreamDecoder", "JsonStreamError", "PathCap"]

_WS = re.compile(rb"[ \t\n\r]*")
_LINE_WS = re.compile(rb"[ \t\r]*")
_STRING_STOP = re.compile(rb'["\\\x00-\x1f]')
_NUMBER_CHARS = re.compile(rb"[0-9eE+\-.]*")
_LITERAL_CHARS = re.compile(rb"[a-z]*")
_NUMBER = re.compile(rb"-?(?:0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?")
_ESCAPE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|(["\\/bfnrt])|(.?))', re.DOTALL)
_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_LITERALS: dict[bytes, object] = {b"true": True, b"false": False, b"null": None}
_MAX_LITERAL = 5
_NUMBER_START = frozenset(b"-0123456789")
_LITERAL_START = frozenset(b"tfn")

# Parser states.
_VALUE = 0  # a value is required
_ARRAY_FIRST = 1  # a value or "]"
_OBJECT_FIRST = 2  # a key or "}"
_OBJECT_KEY = 3  # a key
_COLON = 4
_AFTER = 5  # "," or the container's closing bracket
_STRING = 6
_NUMBER_TOKEN = 7
_LITERAL_TOKEN = 8
_TOP_AFTER = 9  # after a top-level value


class JsonStreamError(ValueError):
    """
    The input is not acceptable JSON under the configured limits.

    ``code`` is one of ``json_invalid``, ``truncated``, ``limit_exceeded`` and
    ``deadline_exceeded``; ``limit`` names the tripped limit.
    """

    def __init__(self, code: str, detail: str, *, limit: str | None = None) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.limit = limit


@dataclass(frozen=True)
class PathCap:
    """
    An aggregate cap on the items of every array found at ``path``.

    ``path`` lists the object keys from the top-level value down to the array,
    with ``"*"`` for each enclosing array; the counter is shared by every array
    at that path (e.g. ``("document", "pages", "*", "blocks")`` caps the blocks
    of all pages together). The empty path caps the number of top-level values
    in sequence mode.
    """

    path: tuple[str, ...]
    limit: int
    name: str


@dataclass(frozen=True)
class DecoderLimits:
    """The lexical limits of one decode."""

    max_depth: int
    max_string_bytes: int
    """Raw bytes of one string token between its quotes, before unescaping."""
    max_number_length: int
    max_object_members: int
    max_array_items: int
    max_total_keys: int
    max_text_bytes: int
    """Raw bytes of every string token (keys and values) together."""
    path_caps: tuple[PathCap, ...] = ()


@dataclass
class _Frame:
    container: dict[str, object] | list[object]
    is_object: bool
    count: int = 0
    key: str = ""
    cap: int | None = None


@dataclass
class _Token:
    buffer: bytearray = field(default_factory=bytearray)
    is_key: bool = False
    escape: bool = False


class JsonStreamDecoder:
    """Decode one JSON value, or a JSON Lines sequence, from byte chunks."""

    def __init__(
        self,
        limits: DecoderLimits,
        *,
        sequence: bool = False,
        deadline: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Prepare a decoder; ``deadline`` is a ``clock()`` value."""
        self._limits = limits
        self._sequence = sequence
        self._deadline = deadline
        self._clock = clock
        self._caps: Mapping[tuple[str, ...], int] = {
            cap.path: index for index, cap in enumerate(limits.path_caps)
        }
        self._cap_counts = [0] * len(limits.path_caps)
        self._frames: list[_Frame] = []
        self._state = _VALUE
        self._token = _Token()
        self._values: list[object] = []
        self._value_count = 0
        self._total_keys = 0
        self._text_bytes = 0
        self._handlers: dict[int, Callable[[bytes, int], int]] = {
            _VALUE: self._on_value,
            _ARRAY_FIRST: self._on_array_first,
            _OBJECT_FIRST: self._on_object_first,
            _OBJECT_KEY: self._on_object_key,
            _COLON: self._on_colon,
            _AFTER: self._on_after,
            _STRING: self._on_string,
            _NUMBER_TOKEN: self._on_number,
            _LITERAL_TOKEN: self._on_literal,
            _TOP_AFTER: self._on_top_after,
        }

    # -- public API ---------------------------------------------------------

    def feed(self, data: bytes) -> list[object]:
        """Consume ``data`` and return the top-level values it completed."""
        pos, end = 0, len(data)
        while pos < end:
            self._check_deadline()
            pos = self._handlers[self._state](data, pos)
        self._check_deadline()
        return self._take_values()

    def close(self) -> list[object]:
        """Signal end of input and return the values that completed."""
        self._check_deadline()
        if self._state in (_NUMBER_TOKEN, _LITERAL_TOKEN):
            self._finish_scalar()
        if self._frames or self._state not in (_VALUE, _TOP_AFTER):
            raise JsonStreamError("truncated", "input ended inside a value")
        if self._sequence and self._state == _TOP_AFTER:
            raise JsonStreamError("truncated", "the last line has no newline")
        if not self._sequence and self._value_count != 1:
            raise JsonStreamError("truncated", "no JSON value")
        return self._take_values()

    # -- bookkeeping --------------------------------------------------------

    def _take_values(self) -> list[object]:
        values, self._values = self._values, []
        return values

    def _check_deadline(self) -> None:
        if self._deadline is not None and self._clock() > self._deadline:
            raise JsonStreamError("deadline_exceeded", "decode wall-clock exceeded")

    def _exceeded(self, limit: str) -> JsonStreamError:
        return JsonStreamError("limit_exceeded", f"{limit} exceeded", limit=limit)

    def _bump_cap(self, cap: int | None) -> None:
        if cap is None:
            return
        if self._cap_counts[cap] >= self._limits.path_caps[cap].limit:
            raise self._exceeded(self._limits.path_caps[cap].name)
        self._cap_counts[cap] += 1

    def _begin_value(self) -> None:
        """Count a value that is about to start in an array or at the top."""
        if not self._frames:
            self._bump_cap(self._caps.get(()))
            self._value_count += 1
            return
        frame = self._frames[-1]
        if not frame.is_object:
            if frame.count >= self._limits.max_array_items:
                raise self._exceeded("max_array_items")
            self._bump_cap(frame.cap)
            frame.count += 1

    def _finish_value(self, value: object) -> None:
        if not self._frames:
            self._values.append(value)
            self._state = _TOP_AFTER
            return
        frame = self._frames[-1]
        if isinstance(frame.container, dict):
            if frame.key in frame.container:
                raise JsonStreamError("json_invalid", f"duplicate key {frame.key!r}")
            frame.container[frame.key] = value
        else:
            frame.container.append(value)
        self._state = _AFTER

    # -- structural states --------------------------------------------------

    def _on_value(self, buf: bytes, pos: int) -> int:
        pos = _skip(_WS, buf, pos)
        if pos == len(buf):
            return pos
        self._begin_value()
        return self._start_value(buf, pos)

    def _start_value(self, buf: bytes, pos: int) -> int:
        char = buf[pos]
        if char in (0x7B, 0x5B):  # "{" "["
            self._open(is_object=char == 0x7B)
            return pos + 1
        if char == 0x22:  # '"'
            self._token = _Token()
            self._state = _STRING
            return pos + 1
        if char in _NUMBER_START:
            self._token = _Token()
            self._state = _NUMBER_TOKEN
            return pos
        if char in _LITERAL_START:
            self._token = _Token()
            self._state = _LITERAL_TOKEN
            return pos
        raise JsonStreamError("json_invalid", f"unexpected byte {char:#04x}")

    def _open(self, *, is_object: bool) -> None:
        if len(self._frames) >= self._limits.max_depth:
            raise self._exceeded("max_depth")
        cap = None if is_object else self._caps.get(self._path())
        container: dict[str, object] | list[object] = {} if is_object else []
        self._frames.append(_Frame(container, is_object, cap=cap))
        self._state = _OBJECT_FIRST if is_object else _ARRAY_FIRST

    def _path(self) -> tuple[str, ...]:
        return tuple(frame.key if frame.is_object else "*" for frame in self._frames)

    def _close(self, buf: bytes, pos: int) -> int:
        frame = self._frames.pop()
        self._finish_value(frame.container)
        return pos + 1

    def _on_array_first(self, buf: bytes, pos: int) -> int:
        pos = _skip(_WS, buf, pos)
        if pos == len(buf):
            return pos
        if buf[pos] == 0x5D:  # "]"
            return self._close(buf, pos)
        self._begin_value()
        return self._start_value(buf, pos)

    def _on_object_first(self, buf: bytes, pos: int) -> int:
        pos = _skip(_WS, buf, pos)
        if pos == len(buf):
            return pos
        if buf[pos] == 0x7D:  # "}"
            return self._close(buf, pos)
        return self._begin_key(buf, pos)

    def _on_object_key(self, buf: bytes, pos: int) -> int:
        pos = _skip(_WS, buf, pos)
        if pos == len(buf):
            return pos
        return self._begin_key(buf, pos)

    def _begin_key(self, buf: bytes, pos: int) -> int:
        if buf[pos] != 0x22:
            raise JsonStreamError("json_invalid", "expected an object key")
        frame = self._frames[-1]
        if frame.count >= self._limits.max_object_members:
            raise self._exceeded("max_object_members")
        if self._total_keys >= self._limits.max_total_keys:
            raise self._exceeded("max_total_keys")
        frame.count += 1
        self._total_keys += 1
        self._token = _Token(is_key=True)
        self._state = _STRING
        return pos + 1

    def _on_colon(self, buf: bytes, pos: int) -> int:
        pos = _skip(_WS, buf, pos)
        if pos == len(buf):
            return pos
        if buf[pos] != 0x3A:  # ":"
            raise JsonStreamError("json_invalid", "expected ':'")
        self._state = _VALUE
        return pos + 1

    def _on_after(self, buf: bytes, pos: int) -> int:
        pos = _skip(_WS, buf, pos)
        if pos == len(buf):
            return pos
        char, frame = buf[pos], self._frames[-1]
        if char == 0x2C:  # ","
            self._state = _OBJECT_KEY if frame.is_object else _VALUE
            return pos + 1
        if char == (0x7D if frame.is_object else 0x5D):
            return self._close(buf, pos)
        raise JsonStreamError("json_invalid", "expected ',' or a closing bracket")

    def _on_top_after(self, buf: bytes, pos: int) -> int:
        if not self._sequence:
            pos = _skip(_WS, buf, pos)
            if pos < len(buf):
                raise JsonStreamError("json_invalid", "data after the JSON value")
            return pos
        pos = _skip(_LINE_WS, buf, pos)
        if pos == len(buf):
            return pos
        if buf[pos] != 0x0A:
            raise JsonStreamError("json_invalid", "a JSON Lines value must end a line")
        self._state = _VALUE
        return pos + 1

    # -- strings ------------------------------------------------------------

    def _on_string(self, buf: bytes, pos: int) -> int:
        token = self._token
        if token.escape:
            self._append(buf[pos : pos + 1])
            token.escape = False
            return pos + 1
        match = _STRING_STOP.search(buf, pos)
        stop = len(buf) if match is None else match.start()
        self._append(buf[pos:stop])
        if stop == len(buf):
            return stop
        char = buf[stop]
        if char == 0x22:
            self._finish_string()
            return stop + 1
        if char == 0x5C:
            self._append(b"\\")
            token.escape = True
            return stop + 1
        raise JsonStreamError("json_invalid", "unescaped control character in string")

    def _append(self, segment: bytes) -> None:
        """Buffer string bytes after checking both string limits."""
        size = len(segment)
        if len(self._token.buffer) + size > self._limits.max_string_bytes:
            raise self._exceeded("max_string_bytes")
        if self._text_bytes + size > self._limits.max_text_bytes:
            raise self._exceeded("max_text_bytes")
        self._text_bytes += size
        self._token.buffer += segment

    def _finish_string(self) -> None:
        text = _decode_string(bytes(self._token.buffer))
        if self._token.is_key:
            self._frames[-1].key = text
            self._state = _COLON
        else:
            self._finish_value(text)

    # -- numbers and literals ----------------------------------------------

    def _on_number(self, buf: bytes, pos: int) -> int:
        return self._scan_scalar(
            buf, pos, _NUMBER_CHARS, self._limits.max_number_length, "max_number_length"
        )

    def _on_literal(self, buf: bytes, pos: int) -> int:
        return self._scan_scalar(buf, pos, _LITERAL_CHARS, _MAX_LITERAL, "literal")

    def _scan_scalar(
        self, buf: bytes, pos: int, chars: re.Pattern[bytes], limit: int, name: str
    ) -> int:
        end = _skip(chars, buf, pos)
        if len(self._token.buffer) + end - pos > limit:
            if name == "literal":
                raise JsonStreamError("json_invalid", "invalid literal")
            raise self._exceeded(name)
        self._token.buffer += buf[pos:end]
        if end < len(buf):
            self._finish_scalar()
        return end

    def _finish_scalar(self) -> None:
        raw = bytes(self._token.buffer)
        if self._state == _LITERAL_TOKEN:
            if raw not in _LITERALS:
                raise JsonStreamError("json_invalid", "invalid literal")
            self._finish_value(_LITERALS[raw])
        else:
            self._finish_value(_decode_number(raw))


def _skip(pattern: re.Pattern[bytes], buf: bytes, pos: int) -> int:
    """Return the end of ``pattern``'s match at ``pos``; it matches the empty string."""
    return cast(re.Match[bytes], pattern.match(buf, pos)).end()


def _decode_number(raw: bytes) -> int | float:
    match = _NUMBER.fullmatch(raw)
    if match is None:
        raise JsonStreamError("json_invalid", "invalid number")
    if match.group(1) is None and match.group(2) is None:
        try:
            return int(raw)
        except ValueError as exc:  # beyond sys.get_int_max_str_digits()
            raise JsonStreamError("json_invalid", "integer too long") from exc
    value = float(raw)
    if value in (float("inf"), float("-inf")):
        raise JsonStreamError("json_invalid", "number out of range")
    return value


def _decode_string(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JsonStreamError("json_invalid", "string is not valid UTF-8") from exc
    if "\\" not in text:
        return text
    text = _ESCAPE.sub(_unescape, text)
    try:
        return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le")
    except UnicodeDecodeError as exc:
        raise JsonStreamError("json_invalid", "lone surrogate in string") from exc


def _unescape(match: re.Match[str]) -> str:
    hex_digits, simple, _ = match.groups()
    if hex_digits is not None:
        return chr(int(hex_digits, 16))
    if simple is not None:
        return _SIMPLE_ESCAPES[simple]
    raise JsonStreamError("json_invalid", "invalid escape in string")
