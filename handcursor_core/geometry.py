"""Small geometry helpers (kept dependency-free beyond NumPy)."""

from __future__ import annotations

import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points of equal dimensionality."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def remap(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    """Linearly map ``value`` from one range to another (no clamping)."""
    if in_hi == in_lo:
        return out_lo
    t = (value - in_lo) / (in_hi - in_lo)
    return out_lo + t * (out_hi - out_lo)
