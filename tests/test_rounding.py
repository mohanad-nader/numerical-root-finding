

import math
from fractions import Fraction

import pytest

import numcomp as nc

README_TABLE = [
    (2.675, 3, "round", 2.68),
    (9.995, 3, "round", 10.0),
    (0.0072684, 3, "round", 0.00727),
    (150.23, 1, "round", 200.0),
    (-2.5, 1, "round", -3.0),
    (3.1415926535, 7, "chop", 3.141592),
    (-2.59, 2, "chop", -2.5),
]

TIES_AND_CARRIES = [
    (2.5, 1, "round", 3.0),
    (2.5, 1, "chop", 2.0),
    (0.25, 1, "round", 0.3),
    (0.35, 1, "round", 0.4),
    (-0.25, 1, "round", -0.3),
    (1234.5, 4, "round", 1235.0),
    (-1234.5, 4, "round", -1235.0),
    (1234.5, 4, "chop", 1234.0),
    (999.5, 3, "round", 1000.0),
    (999.9, 3, "chop", 999.0),
    (0.0999999, 3, "round", 0.1),
    (123456, 3, "round", 123000.0),
    (123456, 3, "chop", 123000.0),
    (123500, 3, "round", 124000.0),
    (123500, 3, "chop", 123000.0),
    (5, 1, "chop", 5.0),
    (7, 12, "round", 7.0),
    (0.1234567890125, 12, "round", 0.123456789013),
    (0.1234567890125, 12, "chop", 0.123456789012),
    (1e-300, 3, "round", 1e-300),
]


@pytest.mark.parametrize("value, n, mode, expected", README_TABLE + TIES_AND_CARRIES)
def test_known_values(value, n, mode, expected):
    assert nc.round_sig(value, n, mode) == expected


@pytest.mark.parametrize("n", range(1, 13))
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("mode", ["round", "chop"])
def test_every_n_on_pi_digits(n, sign, mode):
    digits = "314159265358979"  # 3.14159265358979
    kept = int(digits[:n])
    if mode == "round" and digits[n] >= "5":
        kept += 1
    expected = float(Fraction(kept, 10 ** (n - 1)))
    assert nc.round_sig(sign * 3.14159265358979, n, mode) == sign * expected


def test_default_mode_is_round():
    assert nc.round_sig(2.675, 3) == 2.68


def test_result_is_always_float():
    assert isinstance(nc.round_sig(7, 3, "round"), float)
    assert isinstance(nc.round_sig(Fraction(1, 3), 3, "round"), float)


def test_fraction_input():
    assert nc.round_sig(Fraction(1, 3), 3, "round") == 0.333
    assert nc.round_sig(Fraction(2, 3), 3, "chop") == 0.666
    assert nc.round_sig(Fraction(2, 3), 3, "round") == 0.667


def test_float_is_read_as_its_decimal_text_not_its_binary_value():
    # Fraction(2.675) is 2.67499999..., which would round down to 2.67.
    assert nc.round_sig(2.675, 3, "round") == 2.68
    assert nc.round_sig(1.005, 3, "round") == 1.01


def test_n_none_only_converts_to_float():
    assert nc.round_sig(2.675, None) == 2.675
    r = nc.round_sig(3, None)
    assert r == 3.0 and isinstance(r, float)


@pytest.mark.parametrize("value", [0, 0.0, -0.0])
@pytest.mark.parametrize("mode", ["round", "chop"])
def test_zero_is_positive_zero(value, mode):
    r = nc.round_sig(value, 3, mode)
    assert r == 0.0 and math.copysign(1.0, r) == 1.0


def test_underflow_gives_positive_zero_not_negative_zero():
    r = nc.round_sig(Fraction(-1, 10**400), 3, "round")
    assert r == 0.0 and math.copysign(1.0, r) == 1.0
    r = nc.round_sig(Fraction(1, 10**400), 3, "chop")
    assert r == 0.0 and math.copysign(1.0, r) == 1.0


@pytest.mark.parametrize(
    "args",
    [
        (1, 0, "round"),
        (1, 13, "round"),
        (1, -1, "round"),
        (1, 2.0, "round"),
        (1, "3", "round"),
        (1, 3, "up"),
        (1, 3, None),
        (True, 3, "round"),
        (False, 3, "round"),
        ("1", 3, "round"),
        (None, 3, "round"),
        ([1], 3, "round"),
    ],
)
def test_invalid_arguments_raise_value_error(args):
    with pytest.raises(ValueError):
        nc.round_sig(*args)


@pytest.mark.parametrize(
    "value",
    [float("inf"), float("-inf"), float("nan"), 10**400, -(10**400)],
)
def test_non_finite_or_too_large_raises_divergence(value):
    with pytest.raises(nc.DivergenceError):
        nc.round_sig(value, 3, "round")


def test_rounding_up_past_float_max_raises_divergence():
    with pytest.raises(nc.DivergenceError):
        nc.round_sig(1.7976931348623157e308, 1, "round")