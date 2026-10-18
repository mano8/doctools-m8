"""
RFC 8785 JSON Canonicalization Scheme (JCS) for the revision and artifact hashes.

Only JSON values are accepted: ``dict`` with ``str`` keys, ``list``/``tuple``,
``str``, ``int``, ``float``, ``bool`` and ``None``. JCS serializes every number
as an ECMAScript double, so floats must be finite and an integer beyond 2**53
must already be the canonical text of its nearest double. Strings must be valid
Unicode (no lone surrogates). Anything else raises :class:`ValueError` or
:class:`TypeError`.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

_MAX_SAFE_INTEGER = 2**53

_ESCAPES = {
    0x08: "\\b",
    0x09: "\\t",
    0x0A: "\\n",
    0x0C: "\\f",
    0x0D: "\\r",
    0x22: '\\"',
    0x5C: "\\\\",
}
_ESCAPE_TABLE = {
    code: _ESCAPES.get(code, f"\\u{code:04x}") for code in [*range(0x20), 0x22, 0x5C]
}


def canonicalize(value: object) -> bytes:
    """Return the JCS UTF-8 encoding of a JSON value."""
    return "".join(iter_canonical(value)).encode("utf-8")


def iter_canonical(value: object) -> Iterator[str]:
    """Yield the JCS text of a JSON value in pieces, for streaming writers."""
    if isinstance(value, dict):
        yield from _iter_object(value)
    elif isinstance(value, list | tuple):
        yield from _iter_array(value)
    else:
        yield _scalar(value)


def _iter_object(value: dict[object, object]) -> Iterator[str]:
    keys: list[str] = []
    for key in value:
        if not isinstance(key, str):
            raise TypeError("JCS object keys must be strings")
        keys.append(key)
    yield "{"
    for index, key in enumerate(sorted(keys, key=_utf16_key)):
        if index:
            yield ","
        yield _string(key)
        yield ":"
        yield from iter_canonical(value[key])
    yield "}"


def _iter_array(value: list[object] | tuple[object, ...]) -> Iterator[str]:
    yield "["
    for index, item in enumerate(value):
        if index:
            yield ","
        yield from iter_canonical(item)
    yield "]"


def _utf16_key(key: str) -> bytes:
    return _check_unicode(key).encode("utf-16-be")


def _scalar(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, int):
        return _integer(value)
    if isinstance(value, float):
        return format_number(value)
    raise TypeError(f"not a JSON value: {type(value).__name__}")


def _check_unicode(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("JCS strings must be valid Unicode") from exc
    return value


def _string(value: str) -> str:
    return '"' + _check_unicode(value).translate(_ESCAPE_TABLE) + '"'


def _integer(value: int) -> str:
    text = str(value)
    if abs(value) <= _MAX_SAFE_INTEGER:
        return text
    try:
        nearest = format_number(float(value))
    except OverflowError:
        nearest = ""
    if nearest != text:
        raise ValueError("JCS integers must be the canonical text of a double")
    return text


def format_number(value: float) -> str:
    """Serialize a finite float as ECMAScript ``Number.prototype.toString``."""
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("JCS numbers must be finite")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    digits, n = _shortest_digits(abs(value))
    return sign + _es_layout(digits, n)


def _shortest_digits(value: float) -> tuple[str, int]:
    """Return ``(digits, n)`` with ``value == 0.digits * 10**n``, digits minimal."""
    _, digit_tuple, exponent = Decimal(repr(value)).as_tuple()
    digits = "".join(map(str, digit_tuple)).lstrip("0")
    stripped = digits.rstrip("0")
    assert isinstance(exponent, int)  # finite, so never 'n'/'N'/'F'
    exponent += len(digits) - len(stripped)
    return stripped, exponent + len(stripped)


def _es_layout(digits: str, n: int) -> str:
    k = len(digits)
    if k <= n <= 21:
        return digits + "0" * (n - k)
    if 0 < n <= 21:
        return digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return "0." + "0" * -n + digits
    exponent = n - 1
    mark = "+" if exponent >= 0 else "-"
    mantissa = digits if k == 1 else digits[0] + "." + digits[1:]
    return f"{mantissa}e{mark}{abs(exponent)}"
