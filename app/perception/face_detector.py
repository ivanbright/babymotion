"""Face detection structures.

V1 performs NO identity recognition. The detector only answers "is a face
present, and where". MediaPipe's FaceLandmarker runs its own internal face
detector; a Haar cascade (OpenCV, bundled model) is available as a secondary /
fallback detector for diagnostics and multi-face candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

BBox = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class FaceDetection:
    bbox: BBox
    confidence: float
    timestamp: float

    @property
    def area(self) -> int:
        x, y, w, h = self.bbox
        return max(0, w) * max(0, h)

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    def to_dict(self) -> dict:
        return {"bbox": list(self.bbox), "confidence": self.confidence, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: dict) -> "FaceDetection":
        return cls(bbox=tuple(d["bbox"]), confidence=d["confidence"], timestamp=d["timestamp"])


class FaceDetector(Protocol):
    def detect(self, image: np.ndarray, timestamp: float) -> list[FaceDetection]:
        ...


class CascadeFaceDetector:
    """Haar frontal-face cascade (bundled with OpenCV).

    Confidence is a coarse proxy (OpenCV cascades do not output calibrated
    probabilities); it is only used for ranking/fallback, never for identity.
    """

    def __init__(self, min_size: tuple[int, int] = (60, 60), scale_factor: float = 1.1,
                 min_neighbors: int = 5) -> None:
        import cv2

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, image: np.ndarray, timestamp: float) -> list[FaceDetection]:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        out: list[FaceDetection] = []
        for (x, y, w, h) in boxes:
            out.append(FaceDetection(bbox=(int(x), int(y), int(w), int(h)),
                                     confidence=0.5, timestamp=timestamp))
        return out


def bbox_from_landmarks(points_xy: np.ndarray, frame_w: int, frame_h: int,
                        padding: float = 0.15) -> BBox:
    """Derive a padded pixel bbox from landmark pixel coordinates."""
    x0 = float(points_xy[:, 0].min())
    x1 = float(points_xy[:, 0].max())
    y0 = float(points_xy[:, 1].min())
    y1 = float(points_xy[:, 1].max())
    w = x1 - x0
    h = y1 - y0
    x0 -= w * padding
    y0 -= h * padding
    x1 += w * padding
    y1 += h * padding
    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    x1 = min(frame_w - 1.0, x1)
    y1 = min(frame_h - 1.0, y1)
    return (int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0)))


def select_primary(detections: Sequence[FaceDetection],
                   frame_w: int, frame_h: int) -> Optional[FaceDetection]:
    """Deterministic primary-face selection: largest area, ties -> most centered.

    V1 assumes one baby in frame; multi-face handling stays behind this seam.
    """
    if not detections:
        return None
    cx, cy = frame_w / 2.0, frame_h / 2.0

    def key(d: FaceDetection) -> tuple[int, float]:
        dx, dy = d.center
        return (-d.area, (dx - cx) ** 2 + (dy - cy) ** 2)

    return sorted(detections, key=key)[0]
