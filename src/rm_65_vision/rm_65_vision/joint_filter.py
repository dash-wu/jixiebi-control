#!/usr/bin/env python3
"""One Euro filter for 3D landmark smoothing."""

import math
from typing import Dict, Optional, Tuple

import numpy as np


class OneEuroFilter:
    """Adaptive low-pass filter (Casiez et al.)."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, value: float, dt: float) -> float:
        if self._x_prev is None:
            self._x_prev = value
            self._dx_prev = 0.0
            return value

        dx = (value - self._x_prev) / max(dt, 1e-6)
        alpha_d = self._alpha(self._d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * float(self._dx_prev)

        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * value + (1.0 - alpha) * float(self._x_prev)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None


class LandmarkFilter3D:
    """One Euro filter per axis for shoulder / elbow / wrist world coords."""

    JOINTS = ('shoulder', 'elbow', 'wrist')

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._filters = {
            joint: [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(3)]
            for joint in self.JOINTS
        }
        self._last_time: Optional[float] = None

    def reset(self) -> None:
        for axes in self._filters.values():
            for filt in axes:
                filt.reset()
        self._last_time = None

    def filter(
        self,
        points: Dict[str, np.ndarray],
        timestamp_sec: float,
    ) -> Dict[str, np.ndarray]:
        if self._last_time is None:
            dt = 1.0 / 30.0
        else:
            dt = max(timestamp_sec - self._last_time, 1.0 / 120.0)
        self._last_time = timestamp_sec

        out: Dict[str, np.ndarray] = {}
        for joint in self.JOINTS:
            raw = points[joint]
            filtered = np.array([
                self._filters[joint][i].filter(float(raw[i]), dt)
                for i in range(3)
            ], dtype=float)
            out[joint] = filtered
        return out
