"""One Euro filter — the right tool for interactive pointing.

A plain moving-average or fixed EMA forces a single trade-off: smooth but
laggy, or responsive but jittery. The One Euro filter (Casiez et al., CHI 2012)
adapts its cutoff to speed: slow motion -> heavy smoothing (kills jitter while
you're trying to hold still on a target), fast motion -> light smoothing (no
lag while you fling across monitors). Two intuitive knobs:

    min_cutoff : baseline smoothing at low speed (lower = smoother)
    beta       : how aggressively to reduce smoothing as speed rises

It's also frame-rate independent (uses real timestamps), which matters because
MediaPipe inference time — and thus our effective FPS — fluctuates.
"""

from __future__ import annotations

import math

import numpy as np


class _LowPass:
    def __init__(self) -> None:
        self._y: float | None = None

    def __call__(self, value: float, alpha: float) -> float:
        if self._y is None:
            self._y = value
        else:
            self._y = alpha * value + (1.0 - alpha) * self._y
        return self._y

    @property
    def has_value(self) -> bool:
        return self._y is not None

    @property
    def last(self) -> float:
        return self._y if self._y is not None else 0.0


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_ts: float | None = None
        self._last_x: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value: float, timestamp: float) -> float:
        if self._last_ts is None:
            self._last_ts = timestamp
            self._last_x = value
            return self._x(value, 1.0)

        dt = timestamp - self._last_ts
        if dt <= 0:
            dt = 1e-3
        self._last_ts = timestamp

        # Derivative of the signal, low-pass filtered.
        dx = (value - (self._last_x if self._last_x is not None else value)) / dt
        self._last_x = value
        edx = self._dx(dx, self._alpha(self.d_cutoff, dt))

        # Speed-adaptive cutoff.
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x(value, self._alpha(cutoff, dt))


class Vector2Filter:
    """Convenience: independent One Euro filters for x and y."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def __call__(self, point: np.ndarray, timestamp: float) -> np.ndarray:
        return np.array([self._fx(float(point[0]), timestamp),
                         self._fy(float(point[1]), timestamp)])
