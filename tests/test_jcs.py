"""RFC 8785 canonicalization: number layout, string escaping and key order."""

import json
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from doctools_m8._jcs import canonicalize, format_number, iter_canonical


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # RFC 8785 appendix B and ECMAScript Number.prototype.toString vectors.
        (0.0, "0"),
        (-0.0, "0"),
        (5e-324, "5e-324"),
        (-5e-324, "-5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
        (-1.7976931348623157e308, "-1.7976931348623157e+308"),
        (9007199254740992.0, "9007199254740992"),
        (-9007199254740992.0, "-9007199254740992"),
        (295147905179352830000.0, "295147905179352830000"),
        (1e21, "1e+21"),
        (1e20, "100000000000000000000"),
        (1e23, "1e+23"),
        (333333333.3333333, "333333333.3333333"),
        (0.000001, "0.000001"),
        (1e-7, "1e-7"),
        (1.5e-7, "1.5e-7"),
        (123.456, "123.456"),
        (0.5, "0.5"),
        (1.0, "1"),
        (100.0, "100"),
        (4.5e15, "4500000000000000"),
    ],
)
def test_number_layout_matches_ecmascript(value: float, expected: str) -> None:
    assert format_number(value) == expected


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_refused(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        canonicalize(value)


def test_integers_and_literals() -> None:
    assert canonicalize([0, -1, 2**53, -(2**53), True, False, None]) == (
        b"[0,-1,9007199254740992,-9007199254740992,true,false,null]"
    )


def test_integers_beyond_the_safe_range_must_be_canonical_doubles() -> None:
    assert canonicalize(2**53 + 2) == b"9007199254740994"
    assert canonicalize(36560667464224070) == b"36560667464224070"
    for value in (2**53 + 1, 10**21, 10**400):
        with pytest.raises(ValueError, match="canonical text of a double"):
            canonicalize(value)


def test_string_escaping() -> None:
    value = "".join(map(chr, [0x20AC, 0x24, 0x0F, 0x0A, 0x41, 0x27, 0x22, 0x5C, 0x2F]))
    expected = "".join(map(chr, [0x22, 0x20AC, 0x24])) + r"\u000f\nA'" + r'\"\\/"'
    assert canonicalize(value) == expected.encode()
    controls = "".join(map(chr, [0x08, 0x09, 0x0A, 0x0C, 0x0D, 0x1F, 0x7F]))
    assert canonicalize(controls) == rb'"\b\t\n\f\r\u001f' + b'\x7f"'


def test_key_order_is_utf16_code_unit_order() -> None:
    value = {
        "€": "Euro Sign",
        "\r": "Carriage Return",
        "דּ": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "\U0001f600": "Emoji: Grinning Face",
        "\u0080": "Control",
        "ö": "Latin Small Letter O With Diaeresis",
    }
    expected = (
        '{"\\r":"Carriage Return","1":"One","\u0080":"Control",'
        '"ö":"Latin Small Letter O With Diaeresis","€":"Euro Sign",'
        '"\U0001f600":"Emoji: Grinning Face",'
        '"דּ":"Hebrew Letter Dalet With Dagesh"}'
    )
    assert canonicalize(value) == expected.encode()


def test_nested_containers_and_tuples() -> None:
    assert canonicalize({"b": [1, (2, {})], "a": {}}) == b'{"a":{},"b":[1,[2,{}]]}'
    assert "".join(iter_canonical([])) == "[]"


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ("\ud800", ValueError),
        ({"\udc00": 1}, ValueError),
        ({1: "x"}, TypeError),
        (b"bytes", TypeError),
        ({1.5}, TypeError),
    ],
)
def test_non_json_values_are_refused(value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        canonicalize(value)


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_numbers_round_trip_exactly(value: float) -> None:
    assert float(format_number(value)) == value


json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-(2**53), 2**53)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(alphabet=st.characters(exclude_categories=["Cs"])),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(
            st.text(alphabet=st.characters(exclude_categories=["Cs"])),
            children,
            max_size=4,
        )
    ),
    max_leaves=20,
)


@given(json_values)
def test_canonical_text_is_json_and_idempotent(value: object) -> None:
    encoded = canonicalize(value)
    assert canonicalize(json.loads(encoded)) == encoded
