"""PerceptionResult: the single structure leaving the perception layer.

Serializable so sessions can record feature data (never video by default) and
be replayed exactly like live input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .audio_features import AudioFeatures
from .face_detector import FaceDetection
from .facial_features import FacialFeatures
from .head_pose import HeadPose, HeadOrientation
from .landmarks import FaceLandmarks, Landmark


@dataclass
class PerceptionResult:
    timestamp: float
    frame_id: int
    face_present: bool
    detection: Optional[FaceDetection] = None
    landmarks: Optional[FaceLandmarks] = None
    head_pose: Optional[HeadPose] = None
    features: Optional[FacialFeatures] = None
    audio: AudioFeatures = field(default_factory=AudioFeatures.silent)
    blendshapes: dict[str, float] = field(default_factory=dict)
    detector: str = "landmark"  # landmark | cascade | none
    latency_ms: float = 0.0

    @property
    def orientation(self) -> HeadOrientation:
        if self.head_pose is None:
            return HeadOrientation.UNKNOWN
        return self.head_pose.orientation

    def to_dict(self, include_landmarks: bool = False) -> dict:
        d = {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "face_present": self.face_present,
            "detector": self.detector,
            "latency_ms": round(self.latency_ms, 2),
            "detection": self.detection.to_dict() if self.detection else None,
            "head_pose": self.head_pose.to_dict() if self.head_pose else None,
            "features": self.features.to_dict() if self.features else None,
            "audio": self.audio.to_dict(),
        }
        if include_landmarks and self.landmarks is not None:
            d["landmarks"] = [lm.to_dict() for lm in self.landmarks.normalized]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PerceptionResult":
        features = FacialFeatures.from_dict(d["features"]) if d.get("features") else None
        detection = FaceDetection.from_dict(d["detection"]) if d.get("detection") else None
        pose_d = d.get("head_pose")
        head_pose = None
        if pose_d:
            head_pose = HeadPose(
                yaw=pose_d["yaw"], pitch=pose_d["pitch"], roll=pose_d["roll"],
                orientation=HeadOrientation(pose_d["orientation"]),
                confidence=pose_d["confidence"],
                reprojection_error_px=pose_d.get("reprojection_error_px", 0.0),
            )
        return cls(
            timestamp=d["timestamp"],
            frame_id=d["frame_id"],
            face_present=d["face_present"],
            detection=detection,
            landmarks=None,  # landmark arrays are not required for replay
            head_pose=head_pose,
            features=features,
            audio=AudioFeatures.from_dict(d["audio"]) if d.get("audio") else AudioFeatures.silent(),
            blendshapes=d.get("blendshapes", {}),
            detector=d.get("detector", "landmark"),
            latency_ms=d.get("latency_ms", 0.0),
        )
