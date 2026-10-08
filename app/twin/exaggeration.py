"""Exaggeration engine.

Pipeline:
    baby measurement -> normalize -> neutral reference -> amplification
                     -> clamp to safe visual limits -> twin parameter

    twin_value = neutral + gain * (baby_value - neutral)

All gains/limits come from configuration (config/defaults.yaml), never
hard-coded call sites.
"""

from __future__ import annotations

from ..config.settings import TwinConfig
from ..state.baby_state import BabyState
from ..temporal.sequence import TemporalFeatures
from ..temporal.smoothing import clamp
from .twin_state import TwinState


class ExaggerationEngine:
    def __init__(self, config: TwinConfig) -> None:
        self.config = config

    def amplify(self, value: float, neutral: float, gain: float,
                limits: tuple[float, float] | list[float]) -> float:
        lo, hi = limits
        out = neutral + gain * (value - neutral)
        return float(clamp(out, lo, hi))

    def map_to_twin(self, temporal: TemporalFeatures, baby: BabyState) -> TwinState:
        """Produce the mirrored+amplified twin parameters for this frame."""
        g = self.config.exaggeration
        n = self.config.neutral
        lim = self.config.limits

        mouth = self.amplify(temporal.mouth_open, n.mouth_open, g.mouth, lim["mouth_open"])

        # smile proxy: mouth width relative to neutral (measurements, not emotion labels)
        smile_raw = 0.0
        # temporal does not carry mouth width; baby state facial activity acts as smile drive
        smile_raw = clamp(baby.facial_activity * 1.2 + max(0.0, temporal.mouth_open - n.mouth_open), 0.0, 1.0)
        smile = self.amplify(smile_raw, 0.0, g.expression, lim["smile"])

        return TwinState(
            mouth_open=mouth,
            smile=smile,
            head_yaw=self.amplify(temporal.head_yaw, 0.0, g.head_rotation, lim["head_yaw"]),
            head_pitch=self.amplify(temporal.head_pitch, 0.0, g.head_rotation, lim["head_pitch"]),
            head_roll=self.amplify(0.0, 0.0, g.head_rotation, lim["head_roll"]),
            eye_scale=self.amplify(
                temporal.eye_open / max(1e-6, n.eye_open), 1.0, g.expression, lim["eye_scale"]),
            expression_intensity=self.amplify(
                temporal.facial_motion, 0.0, g.facial_motion, lim["expression_intensity"]),
            timestamp=temporal.timestamp,
        )

    def amplify_action_overrides(self, target: TwinState) -> TwinState:
        return target
