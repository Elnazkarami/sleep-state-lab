"""Small numerical helpers that have to behave the same on every NumPy.

``np.trapz`` was renamed ``np.trapezoid`` in NumPy 2.0 and the old name now
warns. The package supports both, and integration under the power spectrum is
used in two places, so the choice is made once here rather than in each.
"""

from __future__ import annotations

import numpy as np

_TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Integrate ``y`` over ``x`` by the trapezium rule.

    Trapezoidal rather than a bin sum because a sum makes the answer depend on
    the frequency resolution, so the same signal measured over a 30-second epoch
    and a 20-second one would disagree.
    """
    if np.size(y) < 2:
        return 0.0
    return float(_TRAPEZOID(y, x))
