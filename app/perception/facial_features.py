"""Deterministic geometric facial features computed from landmarks.

These are MEASUREMENTS, not emotions. All ratios are normalized (0..1-ish)
so they are independent of face size and camera distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import landmarks as L
from .landmarks import FaceLandmarks


def _dist_px(pix: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(pix[a] - pix[b]))


def _clamp01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


@dataclass
class FacialFeatures:
    mouth_open: float      # inner mouth gap / mouth width
    mouth_width: float     # mouth width / face width
    eye_open_left: float   # eye height / eye width (subject's left eye)
    eye_open_right: float
    eyebrow_position: float  # brow-to-eye distance / face height
    face_width: float      # px
    face_height: float     # px
    face_center: tuple[float, float]  # px, in frame
    nose_position: tuple[float, float]  # normalized offset from face center
    face_scale: float      # face diagonal / frame diagonal (approx distance proxy)

    def to_dict(self) -> dict:
        return {
            "mouth_open": round(self.mouth_open, 4),
            "mouth_width": round(self.mouth_width, 4),
            "eye_open_left": round(self.eye_open_left, 4),
            "eye_open_right": round(self.eye_open_right, 4),
            "eyebrow_position": round(self.eyebrow_position, 4),
            "face_width": round(self.face_width, 2),
            "face_height": round(self.face_height, 2),
            "face_center": [round(self.face_center[0], 2), round(self.face_center[1], 2)],
            "nose_position": [round(self.nose_position[0], 4), round(self.nose_position[1], 4)],
            "face_scale": round(self.face_scale, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FacialFeatures":
        return cls(
            mouth_open=d["mouth_open"], mouth_width=d["mouth_width"],
            eye_open_left=d["eye_open_left"], eye_open_right=d["eye_open_right"],
            eyebrow_position=d["eyebrow_position"],
            face_width=d["face_width"], face_height=d["face_height"],
            face_center=tuple(d["face_center"]), nose_position=tuple(d["nose_position"]),
            face_scale=d["face_scale"],
        )


class FacialFeatureExtractor:
    def extract(self, landmarks: FaceLandmarks, frame_w: int, frame_h: int) -> FacialFeatures:
        pix = landmarks.pixels

        face_w = max(1.0, _dist_px(pix, L.FACE_LEFT, L.FACE_RIGHT))
        face_h = max(1.0, _dist_px(pix, L.FOREHEAD, L.CHIN))
        frame_diag = math.hypot(frame_w, frame_h) or 1.0
        face_diag = math.hypot(face_w, face_h)

        mouth_w = max(1.0, _dist_px(pix, L.MOUTH_LEFT, L.MOUTH_RIGHT))
        mouth_gap = _dist_px(pix, L.MOUTH_INNER_TOP, L.MOUTH_INNER_BOTTOM)
        mouth_open = _clamp01(mouth_gap / mouth_w)
        mouth_width = _clamp01(mouth_w / face_w)

        eye_l_h = _dist_px(pix, L.EYE_L_TOP, L.EYE_L_BOTTOM)
        eye_l_w = max(1.0, _dist_px(pix, L.EYE_L_OUTER, L.EYE_L_INNER))
        eye_r_h = _dist_px(pix, L.EYE_R_TOP, L.EYE_R_BOTTOM)
        eye_r_w = max(1.0, _dist_px(pix, L.EYE_R_OUTER, L.EYE_R_INNER))
        eye_left = _clamp01(eye_l_h / eye_l_w)
        eye_right = _clamp01(eye_r_h / eye_r_w)

        brow_r = (pix[L.BROW_R[0]] + pix[L.BROW_R[1]]) / 2.0
        brow_l = (pix[L.BROW_L[0]] + pix[L.BROW_L[1]]) / 2.0
        eye_top_r = pix[L.EYE_R_TOP]
        eye_top_l = pix[L.EYE_L_TOP]
        brow_gap = 0.5 * (np.linalg.norm(brow_r - eye_top_r) + np.linalg.norm(brow_l - eye_top_l))
        eyebrow_position = _clamp01(brow_gap / face_h)

        center = pix.mean(axis=0)
        face_center = (float(center[0]), float(center[1]))
        nose = pix[L.NOSE_TIP]
        nose_position = (
            float((nose[0] - face_center[0]) / face_w),
            float((nose[1] - face_center[1]) / face_h),
        )

        return FacialFeatures(
            mouth_open=mouth_open,
            mouth_width=mouth_width,
            eye_open_left=eye_left,
            eye_open_right=eye_right,
            eyebrow_position=eyebrow_position,
            face_width=face_w,
            face_height=face_h,
            face_center=face_center,
            nose_position=nose_position,
            face_scale=_clamp01(face_diag / frame_diag),
        )
