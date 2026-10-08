"""Twin action vocabulary - the contract between the intelligence engine and
any renderer (Python debug renderer today, frontend team tomorrow).

Actions are plain serializable dicts; the frontend never needs to know what
MediaPipe, OpenCV, or Python internals produced them.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class TwinActionType(str, enum.Enum):
    MIRROR = "MIRROR"
    EXAGGERATE = "EXAGGERATE"
    LOOK_AT_BABY = "LOOK_AT_BABY"
    SMILE = "SMILE"
    OPEN_MOUTH = "OPEN_MOUTH"
    TURN_HEAD = "TURN_HEAD"
    PAUSE = "PAUSE"
    ATTRACT_ATTENTION = "ATTRACT_ATTENTION"
    INTRODUCE_NOVELTY = "INTRODUCE_NOVELTY"
    REPEAT = "REPEAT"
    ENCOURAGE = "ENCOURAGE"
    END_ACTIVITY = "END_ACTIVITY"
    IDLE = "IDLE"


@dataclass
class TwinAction:
    type: TwinActionType
    target: str = ""            # e.g. "mouth", "head", "facial_expression"
    intensity: float = 0.0      # 0..1
    duration_ms: int = 0        # 0 = until superseded
    direction: str = ""         # e.g. "left", "right" for TURN_HEAD
    stimulus_id: str = ""       # links to STIMULUS_STARTED/ENDED events
    reason: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "action": self.type.value,
            "target": self.target,
            "intensity": round(self.intensity, 3),
            "duration_ms": self.duration_ms,
            "direction": self.direction,
            "stimulus_id": self.stimulus_id,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TwinAction":
        return cls(
            type=TwinActionType(d["action"]),
            target=d.get("target", ""),
            intensity=d.get("intensity", 0.0),
            duration_ms=d.get("duration_ms", 0),
            direction=d.get("direction", ""),
            stimulus_id=d.get("stimulus_id", ""),
            reason=d.get("reason", []),
            timestamp=d.get("timestamp", 0.0),
        )

    @classmethod
    def idle(cls, timestamp: float = 0.0) -> "TwinAction":
        return cls(type=TwinActionType.IDLE, reason=["no_action"], timestamp=timestamp)
