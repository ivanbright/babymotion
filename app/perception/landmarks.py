"""Face landmarks via MediaPipe Tasks FaceLandmarker.

The rest of BabyMotion depends only on the structures defined here
(``Landmark``, ``FaceLandmarks``) - never on MediaPipe object types, so the
perception model can be swapped without touching the engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Landmark indices (MediaPipe FaceMesh 478-point topology).
NOSE_TIP = 4
CHIN = 152
EYE_R_OUTER = 33        # subject's right eye (image left)
EYE_R_INNER = 133
EYE_R_TOP = 159
EYE_R_BOTTOM = 145
EYE_L_OUTER = 263       # subject's left eye (image right)
EYE_L_INNER = 362
EYE_L_TOP = 386
EYE_L_BOTTOM = 374
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_INNER_TOP = 13
MOUTH_INNER_BOTTOM = 14
FACE_LEFT = 234         # subject's right cheek contour (image left)
FACE_RIGHT = 454
FOREHEAD = 10
BROW_R = (70, 105)      # right brow inner/outer-ish pair
BROW_L = (334, 300)

# Rings for the textured-mesh twin: features that must keep full detail
# (lips, eyes, brows) plus the face silhouette; everything else can be
# decimated for the render mesh without losing articulation.
LIPS_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409,
              270, 269, 267, 0, 37, 39, 40, 185)
LIPS_INNER = (78, 191, 80, 192, 81, 82, 13, 312, 311, 310, 415, 308,
              324, 318, 402, 317, 14, 87, 178, 88)
EYE_R_RING = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
              159, 160, 161, 246)
EYE_L_RING = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
              386, 385, 384, 398)
IRIS = tuple(range(468, 478))
BROW_R_RING = (46, 53, 52, 65, 55, 70, 105, 63, 66, 107)
BROW_L_RING = (276, 283, 282, 295, 285, 334, 300, 296, 336, 293)
NOSE = (1, 4, 5, 6, 168, 195, 197)
FACE_OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)

MESH_PRIORITY = tuple(sorted(set(
    (NOSE_TIP, CHIN, FACE_LEFT, FACE_RIGHT, FOREHEAD,
     EYE_R_OUTER, EYE_R_INNER, EYE_R_TOP, EYE_R_BOTTOM,
     EYE_L_OUTER, EYE_L_INNER, EYE_L_TOP, EYE_L_BOTTOM,
     MOUTH_LEFT, MOUTH_RIGHT, MOUTH_INNER_TOP, MOUTH_INNER_BOTTOM)
    + BROW_R + BROW_L
    + LIPS_OUTER + LIPS_INNER + EYE_R_RING + EYE_L_RING + IRIS
    + BROW_R_RING + BROW_L_RING + NOSE + FACE_OVAL
)))


@dataclass
class Landmark:
    x: float  # normalized [0,1] unless noted
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z, "v": self.visibility}


@dataclass
class FaceLandmarks:
    """Landmarks in one face, kept in BOTH normalized and pixel space."""

    normalized: list[Landmark]
    pixels: np.ndarray  # (N, 2) float32 pixel coords
    visibility: np.ndarray  # (N,) float32
    timestamp: float
    depths: Optional[np.ndarray] = None  # (N,) float32, z in pixel units

    @property
    def count(self) -> int:
        return len(self.normalized)

    def pixel(self, index: int) -> tuple[float, float]:
        return float(self.pixels[index, 0]), float(self.pixels[index, 1])

    def normalized_xy(self) -> np.ndarray:
        return np.array([(lm.x, lm.y) for lm in self.normalized], dtype=np.float32)

    def depth_values(self) -> np.ndarray:
        """z per landmark in pixel units (zeros when depth was not captured)."""
        if self.depths is not None:
            return self.depths
        return np.zeros(len(self.normalized), dtype=np.float32)


class MediaPipeLandmarker:
    """Thin wrapper around MediaPipe Tasks FaceLandmarker (VIDEO mode)."""

    def __init__(self, model_path: str | Path, num_faces: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ms = -1
        self.last_blendshapes: dict[str, float] = {}

    def detect(self, image_bgr: np.ndarray, timestamp: float) -> Optional[FaceLandmarks]:
        """Return primary-face landmarks or None if no face is present."""
        rgb = image_bgr[:, :, ::-1].copy()
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(max(0.0, timestamp - self._t0) * 1000.0)
        if ts_ms <= self._last_ms:
            ts_ms = self._last_ms + 1
        self._last_ms = ts_ms
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        self.last_blendshapes = MediaPipeLandmarker.blendshapes(result)
        if not result.face_landmarks:
            return None
        h, w = image_bgr.shape[:2]
        face = result.face_landmarks[0]  # deterministic: first/most confident face
        normalized = [Landmark(x=lm.x, y=lm.y, z=lm.z,
                               visibility=getattr(lm, "visibility", 1.0) or 1.0)
                      for lm in face]
        pixels = np.array([(lm.x * w, lm.y * h) for lm in face], dtype=np.float32)
        visibility = np.array([lm.visibility for lm in normalized], dtype=np.float32)
        # MediaPipe z shares the width-normalized scale of x -> pixel units
        depths = np.array([lm.z * w for lm in face], dtype=np.float32)
        return FaceLandmarks(normalized=normalized, pixels=pixels,
                             visibility=visibility, timestamp=timestamp,
                             depths=depths)

    @staticmethod
    def blendshapes(result) -> dict[str, float]:
        """Flatten MediaPipe blendshape categories to {name: score}."""
        out: dict[str, float] = {}
        if getattr(result, "face_blendshapes", None):
            for cat in result.face_blendshapes[0]:
                out[cat.category_name] = float(cat.score)
        return out

    def close(self) -> None:
        self._landmarker.close()
