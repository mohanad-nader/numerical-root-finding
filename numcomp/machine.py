"""Simulated n-significant-figure decimal machine.  YOU WRITE THIS FILE.

Read README.md, section 2 "The machine", before starting.
Rules: no `decimal`, no built-in `round()`, no `Fraction.__round__`.
`fractions.Fraction` and `math` are allowed.
"""

from __future__ import annotations

import math
from fractions import Fraction

from .errors import DivergenceError  # noqa: F401  (you will raise it)

MODES = ("round", "chop")
MAX_SIG_FIGS = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_n(n):
    """Validate sig_figs: must be an int in 1..12 or None; bool is never ok."""
    if n is None:
        return
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"sig_figs must be an int in 1..{MAX_SIG_FIGS} or None, got {n!r}")
    if not (1 <= n <= MAX_SIG_FIGS):
        raise ValueError(f"sig_figs must be an int in 1..{MAX_SIG_FIGS} or None, got {n}")


def _validate_mode(mode):
    """Validate rounding mode."""
    if mode not in MODES:
        raise ValueError(f"mode must be 'round' or 'chop', got {mode!r}")


def _validate_value(value):
    """Validate the value type: int, float or Fraction. Bool is never accepted."""
    if isinstance(value, bool):
        raise ValueError("booleans are not numbers here")
    if not isinstance(value, (int, float, Fraction)):
        raise ValueError(f"expected an int, float or Fraction, got {type(value).__name__}")


def _to_fraction(value):
    """Convert value to an exact Fraction using the decimal repr for floats."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    # float → use repr to get the exact decimal the user typed
    return Fraction(repr(float(value)))


def _safe_float(q):
    """Convert a Fraction to float, raising DivergenceError on overflow/inf/nan."""
    try:
        result = float(q)
    except (OverflowError, ValueError):
        raise DivergenceError("value too large for a double")
    if math.isinf(result) or math.isnan(result):
        raise DivergenceError("value too large for a double")
    # Never return -0.0
    if result == 0.0:
        return 0.0
    return result


def _find_exponent(abs_q):
    """Find e such that 10**e <= abs_q < 10**(e+1), using exact Fraction arithmetic.

    abs_q must be a positive Fraction.
    """
    # Start by guessing with float log10, then verify exactly
    try:
        guess = int(math.floor(math.log10(float(abs_q))))
    except (OverflowError, ValueError):
        # Huge or tiny numbers: do a binary search
        guess = 0

    power = Fraction(10) ** guess

    # Adjust upward if too small
    while power * 10 <= abs_q:
        guess += 1
        power *= 10

    # Adjust downward if too large
    while power > abs_q:
        guess -= 1
        power = Fraction(10) ** guess

    return guess


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def round_sig(value, n, mode="round"):
    """Round ``value`` to ``n`` significant figures.

    mode "round": half away from zero (2.675 -> 2.68, -2.5 -> -3 at n=1).
    mode "chop":  toward zero (3.1415926535 -> 3.141592 at n=7).
    n None: no decimal rounding, just convert to float (inputs are still
    validated and the result is never -0.0). Raise ValueError for bad n / mode,
    DivergenceError for inf / nan or a result too large for a float.
    """
    # --- validation in the required order ---
    _validate_n(n)
    _validate_mode(mode)
    _validate_value(value)

    # Check for inf/nan in float inputs
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        raise DivergenceError(f"value is {value}")

    # n=None → just convert to float
    if n is None:
        return _safe_float(_to_fraction(value))

    # Convert to exact Fraction
    q = _to_fraction(value)

    # Zero stays zero
    if q == 0:
        return 0.0

    sign = -1 if q < 0 else 1
    abs_q = abs(q)

    # Find exponent e: 10**e <= abs_q < 10**(e+1)
    e = _find_exponent(abs_q)

    # Shift so we have n digits before the decimal point:
    # shifted = abs_q * 10**(n - 1 - e)
    shift = n - 1 - e
    shifted = abs_q * Fraction(10) ** shift  # exact

    if mode == "chop":
        rounded_int = int(shifted)
    else:
        rounded_int = int(shifted + Fraction(1, 2))

    # Convert back: result = sign * rounded_int * 10**(e - n + 1)
    result_frac = Fraction(sign * rounded_int) * Fraction(10) ** (e - n + 1)

    return _safe_float(result_frac)


class Machine:
    """Every method returns a float holding an exact n-digit decimal."""

    def __init__(self, sig_figs=None, mode="round"):
        _validate_n(sig_figs)
        _validate_mode(mode)
        self._n = sig_figs
        self._mode = mode

    def _round(self, frac):
        """Round a Fraction through round_sig using stored n and mode."""
        if self._n is None:
            return _safe_float(frac)
        return round_sig(frac, self._n, self._mode)

    def enter(self, value):
        """Store an input number (literal, constant, initial guess): one rounding."""
        _validate_value(value)
        if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
            raise DivergenceError(f"value is {value}")
        if self._n is None:
            return _safe_float(_to_fraction(value))
        return round_sig(value, self._n, self._mode)

    def _exact_op(self, a, b, op):
        """Perform an exact operation on two floats via Fraction, then round once."""
        fa = Fraction(repr(a))
        fb = Fraction(repr(b))
        if op == "add":
            exact = fa + fb
        elif op == "sub":
            exact = fa - fb
        elif op == "mul":
            exact = fa * fb
        elif op == "div":
            if fb == 0:
                raise ZeroDivisionError("division by zero")
            exact = fa / fb
        else:
            raise ValueError(f"unknown op {op!r}")
        return self._round(exact)

    def add(self, a, b):
        return self._exact_op(a, b, "add")

    def sub(self, a, b):
        return self._exact_op(a, b, "sub")

    def mul(self, a, b):
        return self._exact_op(a, b, "mul")

    def div(self, a, b):
        """Raise ZeroDivisionError when b == 0."""
        return self._exact_op(a, b, "div")

    def pow_int(self, a, k):
        """a**k by repeated multiplication from the left, rounding each product.
        k == 0 -> 1; k < 0 -> one rounded reciprocal of a**(-k)."""
        if k == 0:
            return self.enter(1)
        if k < 0:
            pos_result = self.pow_int(a, -k)
            return self.div(self.enter(1), pos_result)
        # k > 0: r = a, then r = mul(r, a) k-1 times
        r = a
        for _ in range(k - 1):
            r = self.mul(r, a)
        return r

    def call(self, name, a):
        """exp, sin, cos, tan, ln, sqrt: computed with `math` in double, then
        rounded once. Let math's ValueError (domain error) propagate; turn
        overflow into DivergenceError."""
        func_map = {
            "exp": math.exp,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "ln": math.log,
            "sqrt": math.sqrt,
        }
        if name not in func_map:
            raise ValueError(f"unknown function {name!r}")
        try:
            result = func_map[name](a)
        except OverflowError:
            raise DivergenceError(f"overflow in {name}({a})")
        # The result is a float; enter it (which rounds it)
        return self.enter(result)
