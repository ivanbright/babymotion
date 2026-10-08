"""BabyState: the continuously updated, smoothed observable state of the infant.

All fields are observable or model-estimated signals. Estimated fields carry a
confidence (see confidence.py). No field asserts what the baby "feels" or
"thinks".
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ..perception.head_pose import HeadOrientation


class BabyStateName(str, enum.Enum):
    NO_FACE = "NO_FACE"
    OBSERVING = "OBSERVING"
    ENGAGED = "ENGAGED"
    RESPONDING = "RESPONDING"
    EXPLORING = "EXPLORING"
    LOW_ENGAGEMENT = "LOW_ENGAGEMENT"
    COOLDOWN = "COOLDOWN"
    SESSION_END = "SESSION_END"


@dataclass
class BabyState:
    face_present: bool = False
    attention: float = 0.0
    attention_confidence: float = 0.0
    movement: float = 0.0
    vocalization: float = 0.0
    facial_activity: float = 0.0
    head_motion: float = 0.0
    mouth_open: float = 0.0
    interaction_duration: float = 0.0
    response_latency: float | None = None  # seconds, last measured response
    current_orientation: HeadOrientation = HeadOrientation.UNKNOWN
    state: BabyStateName = BabyStateName.NO_FACE
    state_duration: float = 0.0
    timestamp: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "state": self.state.value,
            "state_duration": round(self.state_duration, 2),
            "face_present": self.face_present,
            "attention": round(self.attention, 3),
            "attention_confidence": round(self.attention_confidence, 3),
            "movement": round(self.movement, 3),
            "vocalization": round(self.vocalization, 3),
            "facial_activity": round(self.facial_activity, 3),
            "head_motion": round(self.head_motion, 3),
            "mouth_open": round(self.mouth_open, 3),
            "interaction_duration": round(self.interaction_duration, 1),
            "response_latency": self.response_latency,
            "current_orientation": self.current_orientation.value,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BabyState":
        return cls(
            face_present=d.get("face_present", False),
            attention=d.get("attention", 0.0),
            attention_confidence=d.get("attention_confidence", 0.0),
            movement=d.get("movement", 0.0),
            vocalization=d.get("vocalization", 0.0),
            facial_activity=d.get("facial_activity", 0.0),
            head_motion=d.get("head_motion", 0.0),
            mouth_open=d.get("mouth_open", 0.0),
            interaction_duration=d.get("interaction_duration", 0.0),
            response_latency=d.get("response_latency"),
            current_orientation=HeadOrientation(d.get("current_orientation", "UNKNOWN")),
            state=BabyStateName(d.get("state", "NO_FACE")),
            state_duration=d.get("state_duration", 0.0),
            timestamp=d.get("timestamp", 0.0),
            evidence=d.get("evidence", []),
        )
