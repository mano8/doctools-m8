"""The bounded incremental JSON decoder (D-34)."""

import json
import tracemalloc
from collections.abc import Callable, Iterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from doctools_m8._jsonstream import (
    DecoderLimits,
    JsonStreamDecoder,
    JsonStreamError,
    PathCap,
)

GENEROUS = DecoderLimits(
    max_depth=64,
    max_string_bytes=1 << 20,
    max_number_length=64,
    max_object_members=10_000,
    max_array_items=100_000,
    max_total_keys=100_000,
    max_text_bytes=1 << 24,
)

CORPUS = [
    b"{}",
    b"[]",
    b'""',
    b"0",
    b"-0",
    b"-0.0",
    b"true",
    b"null",
    b"123456789012345678901234567890",
    b" [1, -2.5, 3e2, 4E-2, 0.5e+3, false, null, true] ",
    b'{"a": {"b": [[], {}, [[[]]]]}, "c": "d"}',
    b'"\\u00e9\\u20ac\\ud83d\\ude00 \\" \\\\ \\/ \\b\\f\\n\\r\\t"',
    '"héllo 世界 \U0001f600"'.encode(),
    b'{"document": {"pages": [{"page_number": 1, "blocks": []}]}, "x": 1.5}',
    b'\n\t\r {"k":\n[1\n,\n2]\r\n}\n',
    b'{ "a" : 1 , "b" : [ ] }',
]


def _decode(
    data: bytes,
    limits: DecoderLimits = GENEROUS,
    *,
    step: int | None = None,
    sequence: bool = False,
) -> list[object]:
    decoder = JsonStreamDecoder(limits, sequence=sequence)
    size = step or max(len(data), 1)
    values: list[object] = []
    for start in range(0, len(data), size):
        values.extend(decoder.feed(data[start : start + size]))
    values.extend(decoder.close())
    return values


@pytest.mark.parametrize("text", CORPUS)
def test_differential_against_json_loads(text: bytes) -> None:
    expected = json.loads(text)
    for step in (1, 2, 3, 7, None):
        assert _decode(text, step=step) == [expected]


@pytest.mark.parametrize(
    ("text", "code"),
    [
        (b"", "truncated"),
        (b"   ", "truncated"),
        (b"[1", "truncated"),
        (b'"abc', "truncated"),
        (b'{"a":', "truncated"),
        (b"[1,]", "json_invalid"),
        (b"{,}", "json_invalid"),
        (b'{"a",1}', "json_invalid"),
        (b'{"a":1 "b":2}', "json_invalid"),
        (b"[1 2]", "json_invalid"),
        (b"[1}", "json_invalid"),
        (b"01", "json_invalid"),
        (b"-", "json_invalid"),
        (b"1.", "json_invalid"),
        (b"1e", "json_invalid"),
        (b"1e400", "json_invalid"),
        (b"tru", "json_invalid"),
        (b"truex", "json_invalid"),
        (b"nullnull", "json_invalid"),
        (b'"\x01"', "json_invalid"),
        (b'"\\x"', "json_invalid"),
        (b'"\\', "truncated"),
        (b'"\\ud800"', "json_invalid"),
        (b'"\\udc00\\ud800"', "json_invalid"),
        (b'"\xff"', "json_invalid"),
        (b'{"a":1,"a":2}', "json_invalid"),
        (b"[1] 2", "json_invalid"),
        (b"@", "json_invalid"),
        (b"1" * 4301, "json_invalid"),
    ],
)
def test_malformed_input_raises_the_typed_error(text: bytes, code: str) -> None:
    limits = DecoderLimits(**{**GENEROUS.__dict__, "max_number_length": 5000})
    for step in (1, None):
        with pytest.raises(JsonStreamError) as caught:
            _decode(text, limits, step=step)
        assert caught.value.code == code


def test_json_lines_sequence() -> None:
    text = b'{"a": 1}\n[2]\n3\n"x" \r\n\n'
    assert _decode(text, sequence=True) == [{"a": 1}, [2], 3, "x"]
    assert _decode(text, sequence=True, step=1) == [{"a": 1}, [2], 3, "x"]
    assert _decode(b"", sequence=True) == []


@pytest.mark.parametrize(
    ("text", "code"),
    [
        (b'{"a": 1} {"b": 2}\n', "json_invalid"),
        (b'{"a": 1}', "truncated"),
        (b"1", "truncated"),
    ],
)
def test_json_lines_values_end_their_line(text: bytes, code: str) -> None:
    with pytest.raises(JsonStreamError) as caught:
        _decode(text, sequence=True)
    assert caught.value.code == code


def _limits(**overrides: object) -> DecoderLimits:
    return DecoderLimits(**{**GENEROUS.__dict__, **overrides})


@pytest.mark.parametrize(
    ("text", "overrides", "limit"),
    [
        (b'"' + b"a" * 9 + b'"', {"max_string_bytes": 8}, "max_string_bytes"),
        (b'{"' + b"k" * 9 + b'":1}', {"max_string_bytes": 8}, "max_string_bytes"),
        (b"1" * 9, {"max_number_length": 8}, "max_number_length"),
        (b"[" * 9 + b"]" * 9, {"max_depth": 8}, "max_depth"),
        (b'{"a":1,"b":2,"c":3}', {"max_object_members": 2}, "max_object_members"),
        (b"[1,2,3]", {"max_array_items": 2}, "max_array_items"),
        (b'[{"a":1},{"b":1},{"c":1}]', {"max_total_keys": 2}, "max_total_keys"),
        (b'["abcd","abcd","abcd"]', {"max_text_bytes": 8}, "max_text_bytes"),
        (
            b'{"d":{"pages":[1,2,3]}}',
            {"path_caps": (PathCap(("d", "pages"), 2, "max_pages"),)},
            "max_pages",
        ),
        (
            b'{"p":[{"b":[1]},{"b":[2,3]}]}',
            {"path_caps": (PathCap(("p", "*", "b"), 2, "max_blocks"),)},
            "max_blocks",
        ),
    ],
)
def test_each_limit_trips_one_past_it(
    text: bytes, overrides: dict[str, object], limit: str
) -> None:
    limits = _limits(**overrides)
    for step in (1, None):
        with pytest.raises(JsonStreamError) as caught:
            _decode(text, limits, step=step)
        assert (caught.value.code, caught.value.limit) == ("limit_exceeded", limit)


def test_values_at_their_limits_are_accepted() -> None:
    limits = _limits(
        max_string_bytes=4,
        max_number_length=3,
        max_depth=3,
        max_object_members=2,
        max_array_items=2,
        max_total_keys=3,
        max_text_bytes=10,
        path_caps=(PathCap(("a",), 2, "max_a"),),
    )
    assert _decode(b'{"a":[123,"wxyz"],"b":[[]]}', limits) == [
        {"a": [123, "wxyz"], "b": [[]]}
    ]


def test_top_level_value_cap_in_sequence_mode() -> None:
    limits = _limits(path_caps=(PathCap((), 2, "max_chunks"),))
    assert _decode(b"1\n2\n", limits, sequence=True) == [1, 2]
    with pytest.raises(JsonStreamError) as caught:
        _decode(b"1\n2\n3\n", limits, sequence=True)
    assert caught.value.limit == "max_chunks"


# -- every limit trips before allocation (peak memory under the limit) ----------

PIECE = 64 * 1024
TOTAL = 32 * 1024 * 1024  # 512 times larger than any limit below
SMALL = 64 * 1024
BUDGET = 1024 * 1024  # the limit-sized buffers plus a fixed overhead


def _repeat(head: bytes, unit: bytes) -> Callable[[], Iterator[bytes]]:
    body = unit * max(1, PIECE // len(unit))

    def stream() -> Iterator[bytes]:
        yield head
        for _ in range(TOTAL // len(body)):
            yield body

    return stream


def _unique_keys() -> Iterator[bytes]:
    yield b"{"
    for block in range(TOTAL // PIECE):
        yield b"".join(b'"k%d_%d":0,' % (block, i) for i in range(PIECE // 16))


PAST_LIMIT = [
    ("one huge string", _repeat(b'"', b"a"), {"max_string_bytes": SMALL}),
    ("one huge number", _repeat(b"[1", b"0"), {"max_number_length": 32}),
    ("a wide object", _unique_keys, {"max_object_members": 1000}),
    ("a long array", _repeat(b"[", b"0,"), {"max_array_items": 1000}),
    ("deep nesting", _repeat(b"", b"["), {"max_depth": 64}),
    ("many keys", _repeat(b"[", b'{"a":0},'), {"max_total_keys": 1000}),
    ("many short strings", _repeat(b"[", b'"ab",'), {"max_text_bytes": 8192}),
    (
        "page count",
        _repeat(b'{"document":{"pages":[', b"{},"),
        {"path_caps": (PathCap(("document", "pages"), 1000, "max_pages"),)},
    ),
    (
        "block count",
        _repeat(b'{"document":{"pages":[', b'{"blocks":[0,0]},'),
        {
            "path_caps": (
                PathCap(("document", "pages", "*", "blocks"), 1000, "max_blocks"),
            )
        },
    ),
]


@pytest.mark.parametrize(
    ("name", "stream", "overrides"), PAST_LIMIT, ids=[case[0] for case in PAST_LIMIT]
)
def test_peak_memory_is_bounded_by_the_limits(
    name: str, stream: Callable[[], Iterator[bytes]], overrides: dict[str, object]
) -> None:
    decoder = JsonStreamDecoder(_limits(**overrides))
    chunks = stream()
    tracemalloc.start()
    try:
        with pytest.raises(JsonStreamError) as caught:
            for chunk in chunks:
                decoder.feed(chunk)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert caught.value.code == "limit_exceeded"
    assert peak < BUDGET, f"{name}: peak {peak} bytes"


def test_chunk_count_trips_before_allocation() -> None:
    limits = _limits(path_caps=(PathCap((), 1001, "max_chunks"),))
    decoder = JsonStreamDecoder(limits, sequence=True)
    chunks = _repeat(b"", b'{"text":"x"}\n')()
    tracemalloc.start()
    try:
        with pytest.raises(JsonStreamError) as caught:
            for chunk in chunks:
                decoder.feed(chunk)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert caught.value.limit == "max_chunks"
    assert peak < BUDGET


# -- wall clock -------------------------------------------------------------------


def test_deadline_is_checked_between_chunks() -> None:
    now = [0.0]
    decoder = JsonStreamDecoder(GENEROUS, deadline=10.0, clock=lambda: now[0])
    decoder.feed(b"[1,")
    now[0] = 11.0
    with pytest.raises(JsonStreamError) as caught:
        decoder.feed(b"2]")
    assert caught.value.code == "deadline_exceeded"
    with pytest.raises(JsonStreamError):
        decoder.close()


def test_a_stalled_decode_stops_at_the_first_token_past_the_deadline() -> None:
    calls = [0]
    crossing = 500

    def clock() -> float:
        calls[0] += 1
        return 0.0 if calls[0] <= crossing else 100.0

    decoder = JsonStreamDecoder(GENEROUS, deadline=50.0, clock=clock)
    with pytest.raises(JsonStreamError) as caught:
        decoder.feed(b"[" + b"0," * 50_000)
    assert caught.value.code == "deadline_exceeded"
    assert calls[0] == crossing + 1


# -- property-based ----------------------------------------------------------------

text_strategy = st.text(alphabet=st.characters(exclude_categories=["Cs"]), max_size=20)
json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-(10**30), 10**30)
    | st.floats(allow_nan=False, allow_infinity=False)
    | text_strategy,
    lambda children: (
        st.lists(children, max_size=5)
        | st.dictionaries(text_strategy, children, max_size=5)
    ),
    max_leaves=30,
)


@given(
    json_values,
    st.sampled_from([None, 0, 2]),
    st.booleans(),
    st.integers(1, 17),
)
def test_valid_json_decodes_like_json_loads(
    value: object, indent: int | None, ensure_ascii: bool, step: int
) -> None:
    text = json.dumps(value, indent=indent, ensure_ascii=ensure_ascii).encode()
    assert _decode(text, step=step) == [json.loads(text)]


def _mutate(data: bytes, edits: list[tuple[int, int, int]]) -> bytes:
    buffer = bytearray(data)
    for kind, where, byte in edits:
        if not buffer:
            buffer.append(byte)
            continue
        index = where % len(buffer)
        if kind == 0:
            buffer[index] = byte
        elif kind == 1:
            buffer.insert(index, byte)
        else:
            del buffer[index]
    return bytes(buffer)


@settings(max_examples=300)
@given(
    json_values,
    st.lists(
        st.tuples(st.integers(0, 2), st.integers(0, 1 << 16), st.integers(0, 255)),
        max_size=4,
    ),
    st.integers(1, 9),
    st.booleans(),
)
def test_malformed_input_never_raises_anything_else(
    value: object, edits: list[tuple[int, int, int]], step: int, sequence: bool
) -> None:
    data = _mutate(json.dumps(value).encode(), edits)
    try:
        _decode(data, step=step, sequence=sequence)
    except JsonStreamError:
        pass


@given(st.binary(max_size=200), st.integers(1, 9))
def test_random_bytes_never_raise_anything_else(data: bytes, step: int) -> None:
    try:
        _decode(data, step=step)
    except JsonStreamError:
        pass


@given(
    json_values,
    st.integers(1, 6),
    st.integers(1, 12),
    st.integers(1, 6),
    st.integers(1, 6),
    st.integers(1, 6),
)
def test_limit_edges_either_decode_exactly_or_trip(
    value: object,
    depth: int,
    string_bytes: int,
    members: int,
    items: int,
    keys: int,
) -> None:
    limits = DecoderLimits(
        max_depth=depth,
        max_string_bytes=string_bytes,
        max_number_length=string_bytes + 8,
        max_object_members=members,
        max_array_items=items,
        max_total_keys=keys,
        max_text_bytes=string_bytes * 4,
    )
    text = json.dumps(value).encode()
    try:
        result = _decode(text, limits)
    except JsonStreamError as exc:
        assert exc.code == "limit_exceeded"
    else:
        assert result == [json.loads(text)]
