"""Root-finding methods on the simulated machine.  YOU WRITE THIS FILE.

Read README.md, sections 4 "Methods" and 5 "Result and errors", before starting.
Every method takes ``f`` (or ``g``) as a string or a parsed Expression and
returns a RootResult. Use Machine operations for every arithmetic step of the
method's formula, in exactly the order given in the README.
"""

from __future__ import annotations

import math
import time

from .errors import (
    BracketError,
    DivergenceError,
    ExpressionError,
    NumCompError,
    ZeroDerivativeError,
)
from .expr import Expression, parse
from .machine import Machine
from .result import RootResult


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_common(sig_figs, mode, eps, max_iter, true_root):
    """Validate the shared keyword arguments in the required order.

    Returns the resolved eps value.
    """
    # sig_figs and mode are validated by Machine constructor

    # eps: finite and > 0
    if eps is not None:
        if isinstance(eps, bool):
            raise ValueError(f"eps must be a positive number (percent), got {eps!r}")
        if not isinstance(eps, (int, float)):
            raise ValueError(f"eps must be a positive number (percent), got {eps!r}")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"eps must be a positive number (percent), got {eps}")

    # max_iter: positive int
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter!r}")

    # true_root: None or finite number (not bool)
    if true_root is not None:
        if isinstance(true_root, bool):
            raise ValueError(f"true_root must be a finite number, got {true_root!r}")
        if not isinstance(true_root, (int, float)):
            raise ValueError(f"true_root must be a finite number, got {true_root!r}")
        if not math.isfinite(true_root):
            raise ValueError(f"true_root must be a finite number, got {true_root}")


def _default_eps(sig_figs):
    """Compute the default epsilon (percent).

    With finite sig_figs: 0.5 * 10^(2 - n)
    With sig_figs=None: 0.001
    """
    if sig_figs is None:
        return 0.001
    return 0.5 * 10 ** (2 - sig_figs)


def _parse_expr(f):
    """Accept a string or an already-parsed Expression."""
    if isinstance(f, str):
        return parse(f)
    if isinstance(f, Expression):
        return f
    raise ExpressionError("the expression must be a string")


def _validate_starting_value(v, name="value"):
    """Validate that v is a finite int/float (not bool).

    Large ints that cannot become floats are invalid.
    """
    if isinstance(v, bool):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    if not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v}")
    # Check if int is too large for float
    if isinstance(v, int):
        try:
            fv = float(v)
            if not math.isfinite(fv):
                raise DivergenceError(f"starting value {v} overflows to float")
        except OverflowError:
            raise ValueError(f"{name} must be a finite number, got {v!r}")


# ---------------------------------------------------------------------------
# Significant figures computation
# ---------------------------------------------------------------------------

def _sig_figs_digits(e_percent, cap):
    """Compute significant figures from an error percentage.

    digits(E) = None if E is None, cap if 0, 0 if inf,
    otherwise floor(2 - log10(2*E)) clamped to [0, cap].
    """
    if e_percent is None:
        return None
    if e_percent == 0:
        return cap
    if math.isinf(e_percent):
        return 0
    val = 2 - math.log10(2 * e_percent)
    result = int(math.floor(val))
    # Clamp to [0, cap]
    return max(0, min(result, cap))


def _compute_ea(x_new, x_old):
    """Compute approximate relative error in percent: |( x_new - x_old ) / x_new| * 100.

    Done in double precision. If x_new == 0: 0 when x_old == 0, else inf.
    """
    if x_new == 0:
        if x_old == 0:
            return 0.0
        return float('inf')
    return abs((x_new - x_old) / x_new) * 100


def _compute_true_ea(root, true_root):
    """Compute true relative error percent: |(true_root - root)/true_root| * 100."""
    if true_root is None or true_root == 0:
        return None
    return abs((true_root - root) / true_root) * 100


def _sig_figs_cap(sig_figs):
    """Return the cap for significant figures."""
    return sig_figs if sig_figs is not None else 15


def _build_result(method_name, root, steps, ea, sig_figs, true_root, stop_reason,
                  residual, t0, g_prime_x0=None):
    """Construct a RootResult from the iteration data."""
    cap = _sig_figs_cap(sig_figs)

    # estimated_sig_figs: None when stagnated
    if stop_reason == "stagnated":
        estimated_sig_figs = None
    else:
        estimated_sig_figs = _sig_figs_digits(ea, cap)

    # true_sig_figs
    true_ea = _compute_true_ea(root, true_root)
    true_sig_figs = _sig_figs_digits(true_ea, cap)

    return RootResult(
        method=method_name,
        root=root,
        iterations=len(steps),
        ea=ea,
        estimated_sig_figs=estimated_sig_figs,
        true_sig_figs=true_sig_figs,
        stop_reason=stop_reason,
        residual=residual,
        steps=steps,
        time_s=time.perf_counter() - t0,
        g_prime_x0=g_prime_x0,
    )


def _same_sign(a, b):
    """Check if a and b have the same sign (both positive or both negative).

    Zero is not considered to have a sign for bracketing purposes.
    """
    if a == 0 or b == 0:
        return False
    return (a > 0) == (b > 0)


# ---------------------------------------------------------------------------
# Root-finding methods
# ---------------------------------------------------------------------------

def bisection(f, xl, xu, *, sig_figs=None, mode="round", eps=None, max_iter=50, true_root=None):
    """xr = (xl + xu) / 2."""
    t0 = time.perf_counter()

    # 1. Validate
    _validate_common(sig_figs, mode, eps, max_iter, true_root)
    m = Machine(sig_figs, mode)

    # 2. Parse
    expr = _parse_expr(f)

    # 3. Validate and enter starting values
    _validate_starting_value(xl, "xl")
    _validate_starting_value(xu, "xu")
    xl_m = m.enter(xl)
    xu_m = m.enter(xu)

    if not (xl_m < xu_m):
        raise ValueError(f"need xl < xu after rounding, got xl = {xl_m}, xu = {xu_m}")

    # Resolve eps
    if eps is None:
        eps = _default_eps(sig_figs)

    # 4. Pre-loop checks
    f_xl = expr.evaluate(m, xl_m)
    if f_xl == 0:
        return _build_result("bisection", xl_m, [], None, sig_figs, true_root,
                             "computed_zero", 0.0, t0)

    f_xu = expr.evaluate(m, xu_m)
    if f_xu == 0:
        return _build_result("bisection", xu_m, [], None, sig_figs, true_root,
                             "computed_zero", 0.0, t0)

    if _same_sign(f_xl, f_xu):
        raise BracketError(
            f"f(xl) = {f_xl} and f(xu) = {f_xu} have the same sign: "
            f"[{xl_m}, {xu_m}] does not bracket a root"
        )

    # 5. Iteration loop
    steps = []
    two = m.enter(2)
    prev_xr = None

    for i in range(1, max_iter + 1):
        # xr = (xl + xu) / 2
        xr = m.div(m.add(xl_m, xu_m), two)
        f_xr = expr.evaluate(m, xr)

        # ea
        if prev_xr is None:
            ea = None
        else:
            ea = _compute_ea(xr, prev_xr)

        # Record trace row (bracket BEFORE update)
        row = {
            "i": i,
            "xl": xl_m, "f_xl": f_xl,
            "xu": xu_m, "f_xu": f_xu,
            "xr": xr, "f_xr": f_xr,
            "ea": ea,
        }
        steps.append(row)

        # Stop conditions
        if f_xr == 0:
            return _build_result("bisection", xr, steps, ea, sig_figs, true_root,
                                 "computed_zero", f_xr, t0)

        # Stagnated: xr is not strictly between xl and xu
        if not (xl_m < xr < xu_m):
            return _build_result("bisection", xr, steps, ea, sig_figs, true_root,
                                 "stagnated", f_xr, t0)

        if ea is not None and ea <= eps:
            return _build_result("bisection", xr, steps, ea, sig_figs, true_root,
                                 "tolerance", f_xr, t0)

        if i == max_iter:
            return _build_result("bisection", xr, steps, ea, sig_figs, true_root,
                                 "max_iter", f_xr, t0)

        # Update bracket
        if not _same_sign(f_xl, f_xr):
            # root is in [xl, xr]
            xu_m = xr
            f_xu = f_xr
        else:
            xl_m = xr
            f_xl = f_xr

        prev_xr = xr


def false_position(f, xl, xu, *, sig_figs=None, mode="round", eps=None, max_iter=50, true_root=None):
    """xr = xu - (f(xu) * (xl - xu)) / (f(xl) - f(xu))."""
    t0 = time.perf_counter()

    # 1. Validate
    _validate_common(sig_figs, mode, eps, max_iter, true_root)
    m = Machine(sig_figs, mode)

    # 2. Parse
    expr = _parse_expr(f)

    # 3. Validate and enter starting values
    _validate_starting_value(xl, "xl")
    _validate_starting_value(xu, "xu")
    xl_m = m.enter(xl)
    xu_m = m.enter(xu)

    if not (xl_m < xu_m):
        raise ValueError(f"need xl < xu after rounding, got xl = {xl_m}, xu = {xu_m}")

    # Resolve eps
    if eps is None:
        eps = _default_eps(sig_figs)

    # 4. Pre-loop checks
    f_xl = expr.evaluate(m, xl_m)
    if f_xl == 0:
        return _build_result("false_position", xl_m, [], None, sig_figs, true_root,
                             "computed_zero", 0.0, t0)

    f_xu = expr.evaluate(m, xu_m)
    if f_xu == 0:
        return _build_result("false_position", xu_m, [], None, sig_figs, true_root,
                             "computed_zero", 0.0, t0)

    if _same_sign(f_xl, f_xu):
        raise BracketError(
            f"f(xl) = {f_xl} and f(xu) = {f_xu} have the same sign: "
            f"[{xl_m}, {xu_m}] does not bracket a root"
        )

    # 5. Iteration loop
    steps = []
    prev_xr = None

    for i in range(1, max_iter + 1):
        # xr = xu - f(xu)*(xl - xu) / (f(xl) - f(xu))
        # Computed in the order written: inner parentheses first
        try:
            xl_minus_xu = m.sub(xl_m, xu_m)
            f_xu_times_diff = m.mul(f_xu, xl_minus_xu)
            f_xl_minus_f_xu = m.sub(f_xl, f_xu)
            quotient = m.div(f_xu_times_diff, f_xl_minus_f_xu)
            xr = m.sub(xu_m, quotient)
        except DivergenceError as e:
            e.steps = steps
            raise

        f_xr = expr.evaluate(m, xr)

        # ea
        if prev_xr is None:
            ea = None
        else:
            ea = _compute_ea(xr, prev_xr)

        # Record trace row (bracket BEFORE update)
        row = {
            "i": i,
            "xl": xl_m, "f_xl": f_xl,
            "xu": xu_m, "f_xu": f_xu,
            "xr": xr, "f_xr": f_xr,
            "ea": ea,
        }
        steps.append(row)

        # Stop conditions
        if f_xr == 0:
            return _build_result("false_position", xr, steps, ea, sig_figs, true_root,
                                 "computed_zero", f_xr, t0)

        # Stagnated: xr is not strictly between xl and xu
        if not (xl_m < xr < xu_m):
            return _build_result("false_position", xr, steps, ea, sig_figs, true_root,
                                 "stagnated", f_xr, t0)

        if ea is not None and ea <= eps:
            return _build_result("false_position", xr, steps, ea, sig_figs, true_root,
                                 "tolerance", f_xr, t0)

        if i == max_iter:
            return _build_result("false_position", xr, steps, ea, sig_figs, true_root,
                                 "max_iter", f_xr, t0)

        # Update bracket
        if not _same_sign(f_xl, f_xr):
            xu_m = xr
            f_xu = f_xr
        else:
            xl_m = xr
            f_xl = f_xr

        prev_xr = xr


def fixed_point(g, x0, *, sig_figs=None, mode="round", eps=None, max_iter=50, true_root=None):
    """x_{i+1} = g(x_i)."""
    t0 = time.perf_counter()

    # 1. Validate
    _validate_common(sig_figs, mode, eps, max_iter, true_root)
    m = Machine(sig_figs, mode)

    # 2. Parse
    expr = _parse_expr(g)

    # Build g'(x) for diagnostic (before starting value is checked)
    try:
        g_prime_expr = expr.derivative()
    except NumCompError:
        g_prime_expr = None

    # 3. Validate and enter starting value
    _validate_starting_value(x0, "x0")
    x = m.enter(x0)

    # Resolve eps
    if eps is None:
        eps = _default_eps(sig_figs)

    # Compute g'(x0) diagnostic
    g_prime_x0 = None
    if g_prime_expr is not None:
        try:
            g_prime_x0 = g_prime_expr.evaluate(m, x)
        except NumCompError:
            g_prime_x0 = None

    # 5. Iteration loop
    steps = []

    for i in range(1, max_iter + 1):
        try:
            g_x = expr.evaluate(m, x)
        except NumCompError as e:
            e.steps = steps
            raise

        # ea
        ea = _compute_ea(g_x, x)

        # Record trace row
        row = {"i": i, "x": x, "g_x": g_x, "ea": ea}
        steps.append(row)

        # Stop conditions
        # Computed zero: g(x) == x
        if g_x == x:
            # Residual is 0.0 on computed_zero
            return _build_result("fixed_point", g_x, steps, ea, sig_figs, true_root,
                                 "computed_zero", 0.0, t0, g_prime_x0=g_prime_x0)

        # Fixed point never stagnates (per spec)

        if ea <= eps:
            # Compute residual: m.sub(g(root), root) evaluated once after stopping
            try:
                residual = m.sub(expr.evaluate(m, g_x), g_x)
            except NumCompError:
                residual = float('nan')
            return _build_result("fixed_point", g_x, steps, ea, sig_figs, true_root,
                                 "tolerance", residual, t0, g_prime_x0=g_prime_x0)

        if i == max_iter:
            # Compute residual
            try:
                residual = m.sub(expr.evaluate(m, g_x), g_x)
            except NumCompError:
                residual = float('nan')
            return _build_result("fixed_point", g_x, steps, ea, sig_figs, true_root,
                                 "max_iter", residual, t0, g_prime_x0=g_prime_x0)

        # Update
        x = g_x


def newton(f, x0, *, sig_figs=None, mode="round", eps=None, max_iter=50, true_root=None):
    """x_{i+1} = x_i - f(x_i) / f'(x_i)."""
    t0 = time.perf_counter()

    # 1. Validate
    _validate_common(sig_figs, mode, eps, max_iter, true_root)
    m = Machine(sig_figs, mode)

    # 2. Parse and build derivative (before starting values)
    expr = _parse_expr(f)
    df_expr = expr.derivative()

    # 3. Validate and enter starting value
    _validate_starting_value(x0, "x0")
    x = m.enter(x0)

    # Resolve eps
    if eps is None:
        eps = _default_eps(sig_figs)

    # 4. Pre-loop check
    f_x = expr.evaluate(m, x)
    if f_x == 0:
        return _build_result("newton", x, [], None, sig_figs, true_root,
                             "computed_zero", 0.0, t0)

    # 5. Iteration loop
    steps = []

    for i in range(1, max_iter + 1):
        df_x = df_expr.evaluate(m, x)

        if df_x == 0:
            raise ZeroDerivativeError(f"f'(x) = 0 at x = {x}", steps=steps)

        # x_new = x - f(x) / f'(x)
        x_next = m.sub(x, m.div(f_x, df_x))

        # ea
        ea = _compute_ea(x_next, x)

        # Trace row
        row = {"i": i, "x": x, "f_x": f_x, "df_x": df_x, "x_next": x_next, "ea": ea}
        steps.append(row)

        # Reuse f(x_new) as next f(x)
        f_x_next = expr.evaluate(m, x_next)

        # Stop conditions
        if f_x_next == 0:
            return _build_result("newton", x_next, steps, ea, sig_figs, true_root,
                                 "computed_zero", f_x_next, t0)

        if x_next == x:
            return _build_result("newton", x_next, steps, ea, sig_figs, true_root,
                                 "stagnated", f_x_next, t0)

        if ea <= eps:
            return _build_result("newton", x_next, steps, ea, sig_figs, true_root,
                                 "tolerance", f_x_next, t0)

        if i == max_iter:
            return _build_result("newton", x_next, steps, ea, sig_figs, true_root,
                                 "max_iter", f_x_next, t0)

        # Update
        x = x_next
        f_x = f_x_next


def modified_newton(f, x0, *, sig_figs=None, mode="round", eps=None, max_iter=50, true_root=None):
    """x_{i+1} = x_i - (f f') / (f' f' - f f'')."""
    t0 = time.perf_counter()

    # 1. Validate
    _validate_common(sig_figs, mode, eps, max_iter, true_root)
    m = Machine(sig_figs, mode)

    # 2. Parse and build derivatives (f', then f'')
    expr = _parse_expr(f)
    df_expr = expr.derivative()
    d2f_expr = expr.derivative(2)

    # 3. Validate and enter starting value
    _validate_starting_value(x0, "x0")
    x = m.enter(x0)

    # Resolve eps
    if eps is None:
        eps = _default_eps(sig_figs)

    # 4. Pre-loop check
    f_x = expr.evaluate(m, x)
    if f_x == 0:
        return _build_result("modified_newton", x, [], None, sig_figs, true_root,
                             "computed_zero", 0.0, t0)

    # 5. Iteration loop
    steps = []

    for i in range(1, max_iter + 1):
        df_x = df_expr.evaluate(m, x)
        d2f_x = d2f_expr.evaluate(m, x)

        # denominator = f'(x)*f'(x) - f(x)*f''(x)
        df_sq = m.mul(df_x, df_x)
        f_d2f = m.mul(f_x, d2f_x)
        denom = m.sub(df_sq, f_d2f)

        if denom == 0:
            raise ZeroDerivativeError(
                f"f'(x)^2 - f(x)*f''(x) = 0 at x = {x}", steps=steps
            )

        # numerator = f(x) * f'(x)
        numer = m.mul(f_x, df_x)

        # x_next = x - numer / denom
        x_next = m.sub(x, m.div(numer, denom))

        # ea
        ea = _compute_ea(x_next, x)

        # Trace row
        row = {
            "i": i, "x": x, "f_x": f_x, "df_x": df_x, "d2f_x": d2f_x,
            "x_next": x_next, "ea": ea,
        }
        steps.append(row)

        # Reuse f(x_next) as next f(x)
        f_x_next = expr.evaluate(m, x_next)

        # Stop conditions
        if f_x_next == 0:
            return _build_result("modified_newton", x_next, steps, ea, sig_figs, true_root,
                                 "computed_zero", f_x_next, t0)

        if x_next == x:
            return _build_result("modified_newton", x_next, steps, ea, sig_figs, true_root,
                                 "stagnated", f_x_next, t0)

        if ea <= eps:
            return _build_result("modified_newton", x_next, steps, ea, sig_figs, true_root,
                                 "tolerance", f_x_next, t0)

        if i == max_iter:
            return _build_result("modified_newton", x_next, steps, ea, sig_figs, true_root,
                                 "max_iter", f_x_next, t0)

        # Update
        x = x_next
        f_x = f_x_next


def secant(f, x0, x1, *, sig_figs=None, mode="round", eps=None, max_iter=50, true_root=None):
    """x_{i+1} = x_i - (f(x_i) * (x_{i-1} - x_i)) / (f(x_{i-1}) - f(x_i))."""
    t0 = time.perf_counter()

    # 1. Validate
    _validate_common(sig_figs, mode, eps, max_iter, true_root)
    m = Machine(sig_figs, mode)

    # 2. Parse
    expr = _parse_expr(f)

    # 3. Validate and enter starting values (in signature order: x0, then x1)
    _validate_starting_value(x0, "x0")
    _validate_starting_value(x1, "x1")
    x_prev = m.enter(x0)
    x_cur = m.enter(x1)

    # Resolve eps
    if eps is None:
        eps = _default_eps(sig_figs)

    # 4. Pre-loop checks: return x1 if f(x1) == 0, then x0 if f(x0) == 0
    f_cur = expr.evaluate(m, x_cur)
    if f_cur == 0:
        return _build_result("secant", x_cur, [], None, sig_figs, true_root,
                             "computed_zero", 0.0, t0)

    f_prev = expr.evaluate(m, x_prev)
    if f_prev == 0:
        return _build_result("secant", x_prev, [], None, sig_figs, true_root,
                             "computed_zero", 0.0, t0)

    # 5. Iteration loop
    steps = []

    for i in range(1, max_iter + 1):
        # f(x0) - f(x1)
        denom = m.sub(f_prev, f_cur)

        if denom == 0:
            raise ZeroDerivativeError(
                f"f(x_prev) = f(x) = {f_cur}: the secant is horizontal", steps=steps
            )

        # x_new = x1 - f(x1) * (x0 - x1) / (f(x0) - f(x1))
        x0_minus_x1 = m.sub(x_prev, x_cur)
        f_x1_times_diff = m.mul(f_cur, x0_minus_x1)
        quotient = m.div(f_x1_times_diff, denom)
        x_next = m.sub(x_cur, quotient)

        # ea: x_new vs x1
        ea = _compute_ea(x_next, x_cur)

        # Trace row
        row = {
            "i": i,
            "x_prev": x_prev, "x": x_cur,
            "f_prev": f_prev, "f_x": f_cur,
            "x_next": x_next, "ea": ea,
        }
        steps.append(row)

        # Evaluate f at the new point
        f_next = expr.evaluate(m, x_next)

        # Stop conditions
        if f_next == 0:
            return _build_result("secant", x_next, steps, ea, sig_figs, true_root,
                                 "computed_zero", f_next, t0)

        if x_next == x_cur:
            return _build_result("secant", x_next, steps, ea, sig_figs, true_root,
                                 "stagnated", f_next, t0)

        if ea <= eps:
            return _build_result("secant", x_next, steps, ea, sig_figs, true_root,
                                 "tolerance", f_next, t0)

        if i == max_iter:
            return _build_result("secant", x_next, steps, ea, sig_figs, true_root,
                                 "max_iter", f_next, t0)

        # Update: x0, x1 = x1, x_new
        x_prev = x_cur
        f_prev = f_cur
        x_cur = x_next
        f_cur = f_next
