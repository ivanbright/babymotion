"""Digital twin state - a parameterized, renderer-agnostic representation.

A frontend can consume TwinState directly (e.g. to drive a 3D avatar) and
replace the OpenCV debug renderer without touching the engine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TwinState:
    mouth_open: float = 0.1
    smile: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    eye_scale: float = 1.0
    expression_intensity: float = 0.0
    action: str = "IDLE"
    action_intensity: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "mouth_open": round(self.mouth_open, 3),
            "smile": round(self.smile, 3),
            "head_yaw": round(self.head_yaw, 2),
            "head_pitch": round(self.head_pitch, 2),
            "head_roll": round(self.head_roll, 2),
            "eye_scale": round(self.eye_scale, 3),
            "expression_intensity": round(self.expression_intensity, 3),
            "action": self.action,
            "action_intensity": round(self.action_intensity, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TwinState":
        return cls(
            mouth_open=d.get("mouth_open", 0.1),
            smile=d.get("smile", 0.0),
            head_yaw=d.get("head_yaw", 0.0),
            head_pitch=d.get("head_pitch", 0.0),
            head_roll=d.get("head_roll", 0.0),
            eye_scale=d.get("eye_scale", 1.0),
            expression_intensity=d.get("expression_intensity", 0.0),
            action=d.get("action", "IDLE"),
            action_intensity=d.get("action_intensity", 0.0),
            timestamp=d.get("timestamp", 0.0),
        )
