"""Plot f(x), or g(x) against y = x, to pick initial guesses. 

Using matplotlib (numpy is allowed here). figs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt

from .expr import parse, Expression
from .machine import Machine


def plot(expr, a: float, b: float, path, *, fixed_point: bool = False, points: int = 400) -> Path:
    """Save a PNG of expr on [a, b] to ``path`` and return it as a Path.

    fixed_point=False: draw f(x) and the x-axis.
    fixed_point=True:  draw g(x) and the line y = x.
    Evaluate with Machine(None) (double precision); leave gaps where expr is undefined.
    """
    # Validation
    if not isinstance(a, (int, float)) or not isinstance(a, (int, float)):
        raise ValueError(f"a and b must be numbers, got a={a!r}, b={b!r}")
    if a >= b:
        raise ValueError(f"need a < b, got a={a}, b={b}")
    if not isinstance(points, int) or points < 2:
        raise ValueError(f"points must be an integer >= 2, got {points!r}")

    # Parse if string
    if isinstance(expr, str):
        expr = parse(expr)
    elif not isinstance(expr, Expression):
        from .errors import ExpressionError
        raise ExpressionError("the expression must be a string")

    # Create machine (double precision)
    m = Machine(None)

    # Generate x values
    xs = np.linspace(a, b, points)
    ys = []

    for x_val in xs:
        try:
            y_val = expr.evaluate(m, m.enter(float(x_val)))
            ys.append(y_val)
        except Exception:
            ys.append(float('nan'))  # gap where undefined

    ys = np.array(ys, dtype=float)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(xs, ys, 'b-', linewidth=1.5, label=str(expr))

    if fixed_point:
        # Draw y = x
        ax.plot([a, b], [a, b], 'r--', linewidth=1, label='y = x')
        ax.set_ylabel('g(x)')
        ax.set_title(f'Fixed Point: g(x) = {expr}')
    else:
        # Draw x-axis
        ax.axhline(y=0, color='r', linewidth=0.8, linestyle='--', label='y = 0')
        ax.set_ylabel('f(x)')
        ax.set_title(f'f(x) = {expr}')

    ax.set_xlabel('x')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return path
