"""Smoothing primitives: EMA, hysteresis, clamping."""

from __future__ import annotations

from typing import Optional


def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


class EMASmoother:
    """Exponential moving average with configurable alpha (0..1, higher = faster)."""

    def __init__(self, alpha: float = 0.35, initial: float = 0.0) -> None:
        self.alpha = clamp(alpha, 0.001, 1.0)
        self.value = initial
        self.initialized = False

    def update(self, sample: float, dt: Optional[float] = None) -> float:
        if not self.initialized:
            self.value = sample
            self.initialized = True
        else:
            a = self.alpha
            if dt is not None and dt > 0:
                # time-aware alpha: convert configured alpha (assumed ~30 Hz) to dt
                a = 1.0 - (1.0 - self.alpha) ** (dt * 30.0)
                a = clamp(a, 0.001, 1.0)
            self.value = self.value * (1.0 - a) + sample * a
        return self.value

    def reset(self, value: float = 0.0) -> None:
        self.value = value
        self.initialized = False


class Hysteresis:
    """Two-threshold state switch: prevents flickering around a boundary."""

    def __init__(self, on_threshold: float, off_threshold: float, initial: bool = False) -> None:
        assert off_threshold <= on_threshold, "off threshold must be <= on threshold"
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.state = initial

    def update(self, value: float) -> bool:
        if not self.state and value >= self.on_threshold:
            self.state = True
        elif self.state and value <= self.off_threshold:
            self.state = False
        return self.state

    def reset(self, state: bool = False) -> None:
        self.state = state


class MinDurationGate:
    """Only reports a change after the new value held for min_duration_s."""

    def __init__(self, min_duration_s: float = 0.25) -> None:
        self.min_duration_s = min_duration_s
        self._candidate: Optional[object] = None
        self._candidate_since: float = 0.0
        self._stable: Optional[object] = None

    def update(self, value: object, now: float) -> tuple[bool, object]:
        """Returns (changed, stable_value)."""
        if self._stable is None:
            self._stable = value
            self._candidate = value
            self._candidate_since = now
            return (False, value)
        if value != self._stable:
            if value != self._candidate:
                self._candidate = value
                self._candidate_since = now
            elif now - self._candidate_since >= self.min_duration_s:
                self._stable = value
                return (True, value)
        else:
            self._candidate = value
            self._candidate_since = now
        return (False, self._stable)
