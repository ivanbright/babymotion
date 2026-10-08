"""Timestamped sliding windows and their statistics."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class WindowStats:
    count: int
    mean: float
    variance: float
    min: float
    max: float
    latest: float
    velocity: float        # units/second, linear regression slope over window
    acceleration: float    # units/second^2 (slope of first half vs second half)
    trend: float           # -1..1: (latest - mean) / (range or eps)
    duration_s: float      # span of the window actually covered

    @property
    def range(self) -> float:
        return self.max - self.min


class TimedWindow:
    """Deque of (timestamp, value) with timestamp-based trimming."""

    def __init__(self, window_ms: int = 3000, max_samples: int = 600) -> None:
        self.window_s = window_ms / 1000.0
        self._data: collections.deque[tuple[float, float]] = collections.deque(maxlen=max_samples)

    def append(self, timestamp: float, value: float) -> None:
        self._data.append((timestamp, float(value)))
        self._trim(timestamp)

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()

    def extend(self, pairs: list[tuple[float, float]]) -> None:
        for t, v in pairs:
            self.append(t, v)

    def clear(self) -> None:
        self._data.clear()

    @property
    def empty(self) -> bool:
        return not self._data

    def latest(self) -> Optional[float]:
        return self._data[-1][1] if self._data else None

    def latest_time(self) -> Optional[float]:
        return self._data[-1][0] if self._data else None

    def values(self) -> np.ndarray:
        return np.array([v for _, v in self._data], dtype=np.float64)

    def times(self) -> np.ndarray:
        return np.array([t for t, _ in self._data], dtype=np.float64)

    def mean(self) -> float:
        return float(self.values().mean()) if self._data else 0.0

    def stats(self) -> WindowStats:
        if not self._data:
            return WindowStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        t = self.times()
        v = self.values()
        duration = float(t[-1] - t[0])
        mean = float(v.mean())
        var = float(v.var())
        if len(v) >= 2 and duration > 1e-6:
            # least-squares slope: velocity (units per second)
            tc = t - t.mean()
            vc = v - mean
            denom = float((tc ** 2).sum())
            velocity = float((tc * vc).sum() / denom) if denom > 1e-12 else 0.0
            half = max(1, len(v) // 2)
            v_first = float(v[:half].mean())
            v_last = float(v[half:].mean())
            t_first = float(t[:half].mean())
            t_last = float(t[half:].mean())
            dt = t_last - t_first
            accel = float((v_last - v_first) / (dt ** 2)) if dt > 1e-6 else 0.0
        else:
            velocity = 0.0
            accel = 0.0
        rng = float(v.max() - v.min())
        trend = float((v[-1] - mean) / (rng / 2.0)) if rng > 1e-9 else 0.0
        return WindowStats(
            count=len(v),
            mean=mean,
            variance=var,
            min=float(v.min()),
            max=float(v.max()),
            latest=float(v[-1]),
            velocity=velocity,
            acceleration=accel,
            trend=float(max(-1.0, min(1.0, trend))),
            duration_s=duration,
        )

    def duration_above(self, threshold: float, now: float) -> float:
        """Total time the signal spent above threshold within the window."""
        if not self._data:
            return 0.0
        total = 0.0
        items = list(self._data)
        for i, (t, v) in enumerate(items):
            t_next = items[i + 1][0] if i + 1 < len(items) else now
            if v > threshold:
                total += max(0.0, t_next - t)
        return total
