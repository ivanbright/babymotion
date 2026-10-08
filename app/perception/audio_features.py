"""Audio feature extraction (optional microphone).

V1 computes only measurable signal: RMS energy in the recent window, mapped to
a 0..1 vocalization level with a simple noise-floor estimate. No speech
recognition, no emotion claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AudioFeatures:
    vocalization: float  # 0..1 normalized energy above noise floor
    rms: float
    noise_floor: float
    available: bool

    def to_dict(self) -> dict:
        return {
            "vocalization": round(self.vocalization, 4),
            "rms": round(self.rms, 6),
            "noise_floor": round(self.noise_floor, 6),
            "available": self.available,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AudioFeatures":
        return cls(vocalization=d["vocalization"], rms=d["rms"],
                   noise_floor=d["noise_floor"], available=d["available"])

    @classmethod
    def silent(cls) -> "AudioFeatures":
        return cls(vocalization=0.0, rms=0.0, noise_floor=0.0, available=False)


class AudioFeatureExtractor:
    """Maps mic RMS blocks to a smoothed 0..1 vocalization estimate."""

    def __init__(self, gate: float = 0.02, saturation: float = 0.15) -> None:
        self.gate = gate          # RMS below this = silence
        self.saturation = saturation  # RMS at/above this = full scale
        self._noise_floor = gate
        self._smoothed = 0.0
        self._alpha = 0.4

    def update(self, recent_rms: list[tuple[float, float]]) -> AudioFeatures:
        if not recent_rms:
            return AudioFeatures.silent()
        values = np.array([v for _, v in recent_rms], dtype=np.float64)
        rms = float(values.mean())
        # track a slow noise floor (10th percentile of recent audio)
        floor = float(np.percentile(values, 10))
        self._noise_floor = min(self._noise_floor * 0.95 + floor * 0.05, rms)
        above = max(0.0, rms - self._noise_floor)
        level = min(1.0, above / max(1e-6, self.saturation - self.gate))
        if rms < self.gate:
            level = 0.0
        self._smoothed = self._smoothed * (1 - self._alpha) + level * self._alpha
        return AudioFeatures(vocalization=float(self._smoothed), rms=rms,
                             noise_floor=self._noise_floor, available=True)
