"""Extra tests for argument validation, validation order and exact exception types."""

import pytest

import numcomp as nc

BRACKET_METHODS = [nc.bisection, nc.false_position]


@pytest.mark.parametrize("method", BRACKET_METHODS)
def test_no_bracket(method):
    with pytest.raises(nc.BracketError) as info:
        method("x**2 + 1", 0, 1)
    assert type(info.value) is nc.BracketError
    assert str(info.value)


@pytest.mark.parametrize("method", BRACKET_METHODS)
@pytest.mark.parametrize("xl, xu", [(2, 1), (1, 1), (1.0, 1.0)])
def test_interval_must_be_increasing(method, xl, xu):
    with pytest.raises(ValueError):
        method("x**2 - 2", xl, xu)


@pytest.mark.parametrize("method", BRACKET_METHODS)
def test_interval_is_checked_after_entering_on_the_machine(method):
    with pytest.raises(ValueError):
        method("x - 1", 1.001, 1.002, sig_figs=3)


def test_newton_zero_derivative():
    with pytest.raises(nc.ZeroDerivativeError) as info:
        nc.newton("x**2 - 1", 0)
    assert type(info.value) is nc.ZeroDerivativeError
    assert len(info.value.steps) == 0


def test_secant_equal_function_values():
    with pytest.raises(nc.ZeroDerivativeError):
        nc.secant("x**2", -1, 1)


def test_modified_newton_zero_denominator():
    # f = exp(x): f' = f'' = f, so f'*f' - f*f'' is exactly 0 for every x.
    with pytest.raises(nc.ZeroDerivativeError):
        nc.modified_newton("exp(x)", 0)


def test_newton_zero_derivative_on_a_positive_function():
    with pytest.raises(nc.ZeroDerivativeError):
        nc.newton("x**2 + 1", 0)


def test_fixed_point_divergence_on_overflow():
    with pytest.raises(nc.DivergenceError) as info:
        nc.fixed_point("2*x + 1", 1, sig_figs=3, max_iter=5000)
    assert type(info.value) is nc.DivergenceError
    assert len(info.value.steps) > 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sig_figs": 0},
        {"sig_figs": 13},
        {"sig_figs": 2.0},
        {"mode": "up"},
        {"eps": 0},
        {"eps": -1},
        {"eps": float("inf")},
        {"eps": float("nan")},
        {"max_iter": 0},
        {"max_iter": -3},
        {"max_iter": 2.5},
        {"true_root": float("inf")},
        {"true_root": float("nan")},
    ],
)
@pytest.mark.parametrize(
    "call",
    [
        lambda **kw: nc.bisection("x**2 - 2", 1, 2, **kw),
        lambda **kw: nc.false_position("x**2 - 2", 1, 2, **kw),
        lambda **kw: nc.fixed_point("cos(x)", 1, **kw),
        lambda **kw: nc.newton("x**2 - 2", 1, **kw),
        lambda **kw: nc.modified_newton("x**2 - 2", 1, **kw),
        lambda **kw: nc.secant("x**2 - 2", 1, 2, **kw),
    ],
)
def test_invalid_options_raise_value_error(call, kwargs):
    with pytest.raises(ValueError):
        call(**kwargs)


@pytest.mark.parametrize("bad", [True, False, "1", None, float("inf"), float("nan"), 10**400])
def test_invalid_starting_values(bad):
    with pytest.raises(ValueError):
        nc.newton("x**2 - 2", bad)
    with pytest.raises(ValueError):
        nc.bisection("x**2 - 2", bad, 2)
    with pytest.raises(ValueError):
        nc.secant("x**2 - 2", 1, bad)


@pytest.mark.parametrize("expr", ["2x", "x +", "foo(x)"])
def test_invalid_expressions(expr):
    with pytest.raises(nc.ExpressionError):
        nc.bisection(expr, 1, 2)


def test_options_are_validated_before_the_expression_is_parsed():
    with pytest.raises(ValueError):
        nc.bisection("2x", 0, 1, sig_figs=0)


def test_expression_is_parsed_before_starting_values_are_checked():
    with pytest.raises(nc.ExpressionError):
        nc.bisection("2x", 2, 1)


def test_newton_builds_the_derivative_before_checking_starting_values():
    with pytest.raises(nc.ExpressionError):
        nc.newton("abs(x)", "bad")


def test_domain_error_inside_a_method_is_an_expression_error():
    with pytest.raises(nc.ExpressionError):
        nc.newton("ln(x)", -1)