

import math

import pytest

import numcomp as nc


def test_readme_worked_example_step_by_step():
    m = nc.Machine(3, "chop")
    x = m.enter(2.37)
    assert x == 2.37
    x2 = m.mul(x, x)
    assert x2 == 5.61
    x3 = m.mul(x2, x)
    assert x3 == 13.2
    five_x2 = m.mul(m.enter(5), x2)
    assert five_x2 == 28.0
    seven_x = m.mul(m.enter(7), x)
    assert seven_x == 16.5
    a = m.sub(x3, five_x2)
    assert a == -14.8
    b = m.add(a, seven_x)
    assert b == 1.7
    assert m.sub(b, m.enter(0.25)) == 1.45


def test_exact_decimal_addition_not_float_addition():
    assert nc.Machine(3, "chop").add(-14.8, 16.5) == 1.70


def test_readme_worked_example_as_expression():
    m = nc.Machine(3, "chop")
    f = nc.parse("x**3 - 5*x**2 + 7*x - 0.25")
    assert f.evaluate(m, m.enter(2.37)) == 1.45


def test_nested_form_gives_a_different_answer():
    m = nc.Machine(3, "chop")
    f = nc.parse("((x - 5)*x + 7)*x - 0.25")
    assert f.evaluate(m, m.enter(2.37)) == 1.57


@pytest.mark.parametrize(
    "mode, a, b, expected",
    [
        ("round", 1, 3, 0.333),
        ("chop", 1, 3, 0.333),
        ("round", 2, 3, 0.667),
        ("chop", 2, 3, 0.666),
        ("round", -2, 3, -0.667),
        ("chop", -2, 3, -0.666),
        ("round", 0, 5, 0.0),
    ],
)
def test_div_rounds_the_exact_quotient_once(mode, a, b, expected):
    assert nc.Machine(3, mode).div(a, b) == expected


def test_div_by_zero():
    with pytest.raises(ZeroDivisionError):
        nc.Machine(3, "round").div(1, 0)
    with pytest.raises(ZeroDivisionError):
        nc.Machine(None).div(1, 0)


def test_mul_round_vs_chop():
    assert nc.Machine(3, "round").mul(1.23, 4.56) == 5.61  # exact 5.6088
    assert nc.Machine(3, "chop").mul(1.23, 4.56) == 5.6


def test_add_loses_small_terms():
    assert nc.Machine(3, "round").add(1000, 0.4) == 1000.0
    assert nc.Machine(3, "round").add(1005, 1) == 1010.0
    assert nc.Machine(3, "chop").add(1005, 1) == 1000.0


def test_subtraction_of_close_numbers_is_exact_when_it_fits():
    assert nc.Machine(3, "round").sub(1.0, 0.999) == 0.001


def test_zero_results_are_positive_zero():
    m = nc.Machine(3, "round")
    for r in (m.sub(1, 1), m.mul(0, -5), m.div(0, -5), m.add(-2, 2)):
        assert r == 0.0 and math.copysign(1.0, r) == 1.0


def test_pow_int_multiplies_step_by_step():
    m = nc.Machine(2, "round")
    # 1.5*1.5 = 2.25 -> 2.3 ; 2.3*1.5 = 3.45 -> 3.5 (exact cube 3.375 would give 3.4)
    assert m.pow_int(1.5, 3) == 3.5
    assert nc.Machine(3, "round").pow_int(1.9, 3) == 6.86
    assert nc.Machine(3, "chop").pow_int(1.9, 3) == 6.85


def test_pow_int_zero_one_and_negative():
    m = nc.Machine(3, "round")
    assert m.pow_int(5, 0) == 1.0
    assert m.pow_int(1.7, 1) == 1.7
    assert m.pow_int(2, -2) == 0.25
    assert m.pow_int(3, -1) == 0.333


@pytest.mark.parametrize(
    "mode, name, arg, expected",
    [
        ("round", "sqrt", 2, 1.414),
        ("round", "exp", 1, 2.718),
        ("round", "ln", 10, 2.303),
        ("chop", "ln", 10, 2.302),
        ("round", "sin", 1, 0.8415),
        ("chop", "sin", 1, 0.8414),
    ],
)
def test_call_rounds_the_library_result_once(mode, name, arg, expected):
    assert nc.Machine(4, mode).call(name, arg) == expected


def test_call_domain_error_propagates_as_value_error():
    with pytest.raises(ValueError):
        nc.Machine(4).call("sqrt", -1)
    with pytest.raises(ValueError):
        nc.Machine(4).call("ln", 0)


def test_call_overflow_is_divergence():
    with pytest.raises(nc.DivergenceError):
        nc.Machine(3).call("exp", 1000)


def test_enter_rounds_outside_numbers_once():
    assert nc.Machine(3, "round").enter(2.675) == 2.68
    assert nc.Machine(3, "chop").enter(2.675) == 2.67
    assert nc.Machine(None).enter(2.675) == 2.675


@pytest.mark.parametrize("bad", [True, "1", None, [1]])
def test_enter_rejects_bad_types(bad):
    with pytest.raises(ValueError):
        nc.Machine(3).enter(bad)


@pytest.mark.parametrize("args", [(0,), (13,), (2.0,), ("3",), (3, "up")])
def test_constructor_validation(args):
    with pytest.raises(ValueError):
        nc.Machine(*args)


def test_double_precision_machine_is_plain_float_math():
    m = nc.Machine(None)
    assert m.add(0.1, 0.2) == 0.1 + 0.2
    assert m.mul(1.1, 1.1) == 1.1 * 1.1
    assert m.div(1, 3) == 1 / 3
    assert m.pow_int(1.1, 3) == (1.1 * 1.1) * 1.1


def test_expression_double_precision_values():
    m = nc.Machine(None)
    assert nc.parse("x**2 - 2").evaluate(m, 3.0) == 7.0
    assert nc.parse("x^2 - 2").evaluate(m, 3.0) == 7.0
    assert nc.parse("2*x + 1").evaluate(m, 4.0) == 9.0
    assert nc.parse("sqrt(x)").evaluate(m, 9.0) == 3.0
    assert nc.parse("abs(x)").evaluate(m, -4.5) == 4.5
    assert nc.parse("x**-2").evaluate(m, 2.0) == 0.25


def test_derivative_of_the_readme_polynomial():
    m = nc.Machine(None)
    f = nc.parse("x**3 - 5*x**2 + 7*x - 0.25")
    assert f.derivative().evaluate(m, 3.0) == 4.0  # 3x^2 - 10x + 7 at 3


def test_expression_errors():
    m = nc.Machine(None)
    with pytest.raises(nc.ExpressionError):
        nc.parse("2x")
    with pytest.raises(nc.ExpressionError):
        nc.parse("sqrt(x)").evaluate(m, -1.0)
    with pytest.raises(nc.ExpressionError):
        nc.parse("ln(x)").evaluate(m, 0.0)