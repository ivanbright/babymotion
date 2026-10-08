"""Synthetic signal fixtures: no baby or camera required.

Scenarios feed the engine exactly like live perception output, so the whole
temporal/state/policy loop can be tested deterministically.
"""

from __future__ import annotations

import math

import numpy as np

from app.perception import landmarks as L
from app.perception.face_detector import FaceDetection, bbox_from_landmarks
from app.perception.facial_features import FacialFeatureExtractor, FacialFeatures
from app.perception.head_pose import HeadOrientation, HeadPose, HeadPoseEstimator
from app.perception.landmarks import FaceLandmarks, Landmark
from app.perception.perception_result import PerceptionResult

FPS = 30.0
DT = 1.0 / FPS


def make_landmarks(frame_w: int = 1280, frame_h: int = 720,
                   face_cx: float = 0.5, face_cy: float = 0.5,
                   face_w_frac: float = 0.30,
                   mouth_open: float = 0.1,
                   eye_open: float = 0.8,
                   brow_raise: float = 0.5,
                   noise: float = 0.0,
                   seed: int = 0) -> FaceLandmarks:
    """Build a plausible 478-point landmark set with controllable geometry."""
    rng = np.random.default_rng(seed)
    fw = face_w_frac * frame_w
    fh = fw * 1.3
    cx = face_cx * frame_w
    cy = face_cy * frame_h
    x0, x1 = cx - fw / 2, cx + fw / 2
    y0, y1 = cy - fh / 2, cy + fh / 2

    def at(nx: float, ny: float) -> tuple[float, float]:
        # nx/ny in 0..1 within the face box
        x = x0 + nx * fw + (rng.normal(0, noise) if noise else 0.0)
        y = y0 + ny * fh + (rng.normal(0, noise) if noise else 0.0)
        return float(np.clip(x, 0, frame_w - 1)), float(np.clip(y, 0, frame_h - 1))

    pts = np.zeros((478, 2), dtype=np.float32)
    # generic fill so every index has a sane location
    for i in range(478):
        pts[i] = at(0.5 + 0.35 * math.cos(i), 0.5 + 0.45 * math.sin(i * 1.7))

    # structure points
    pts[L.FACE_LEFT] = at(0.02, 0.52)
    pts[L.FACE_RIGHT] = at(0.98, 0.52)
    pts[L.FOREHEAD] = at(0.5, 0.02)
    pts[L.CHIN] = at(0.5, 0.98)
    pts[L.NOSE_TIP] = at(0.5, 0.55)

    # eyes: height controlled by eye_open ratio (v = eye_open * w)
    eye_w = 0.16 * fw
    for outer, inner, top, bot, mirror in (
        (L.EYE_R_OUTER, L.EYE_R_INNER, L.EYE_R_TOP, L.EYE_R_BOTTOM, False),
        (L.EYE_L_OUTER, L.EYE_L_INNER, L.EYE_L_TOP, L.EYE_L_BOTTOM, True),
    ):
        ox, oy = at(0.20 if not mirror else 0.80, 0.40)
        ix = ox + (eye_w if not mirror else -eye_w)
        pts[outer] = (ox, oy)
        pts[inner] = (ix, oy)
        mid = (ox + ix) / 2
        half_v = eye_open * eye_w / 2
        pts[top] = (mid, oy - half_v)
        pts[bot] = (mid, oy + half_v)

    # mouth
    mouth_cx, mouth_cy = at(0.5, 0.76)
    mouth_w = 0.40 * fw
    half = mouth_w / 2
    gap = mouth_open * mouth_w
    pts[L.MOUTH_LEFT] = (mouth_cx - half, mouth_cy)
    pts[L.MOUTH_RIGHT] = (mouth_cx + half, mouth_cy)
    pts[L.MOUTH_INNER_TOP] = (mouth_cx, mouth_cy - gap / 2)
    pts[L.MOUTH_INNER_BOTTOM] = (mouth_cx, mouth_cy + gap / 2)

    # brows (raise moves them up)
    brow_y = 0.30 - 0.06 * brow_raise
    pts[L.BROW_R[0]] = at(0.30, brow_y)
    pts[L.BROW_R[1]] = at(0.42, brow_y)
    pts[L.BROW_L[0]] = at(0.70, brow_y)
    pts[L.BROW_L[1]] = at(0.58, brow_y)

    # depth: shallow dome (center closer = more negative) + nose bump
    dx = (pts[:, 0] - cx) / max(1.0, fw)
    dy = (pts[:, 1] - cy) / max(1.0, fh)
    dome = np.clip(1.0 - (dx / 0.55) ** 2 - (dy / 0.65) ** 2, 0.0, None)
    z_norm = (-0.06 * np.sqrt(dome)).astype(np.float32)
    nose = np.abs(pts[:, 0] - pts[L.NOSE_TIP, 0]) < 0.12 * fw
    z_norm[nose & (dy > -0.1) & (dy < 0.25)] -= 0.02

    normalized = [Landmark(x=float(px / frame_w), y=float(py / frame_h),
                           z=float(zn), visibility=1.0)
                  for (px, py), zn in zip(pts, z_norm)]
    visibility = np.ones(478, dtype=np.float32)
    depths = (z_norm * frame_w).astype(np.float32)
    return FaceLandmarks(normalized=normalized, pixels=pts, visibility=visibility,
                         timestamp=0.0, depths=depths)


def make_perception(
    timestamp: float,
    frame_id: int = 0,
    face_present: bool = True,
    mouth_open: float = 0.1,
    eye_open: float = 0.8,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    face_cx: float = 0.5,
    face_cy: float = 0.5,
    face_scale: float = 0.30,
    vocalization: float = 0.0,
    noise: float = 0.0,
    frame_w: int = 1280,
    frame_h: int = 720,
    seed: int = 0,
    head_confidence: float = 0.9,
) -> PerceptionResult:
    if not face_present:
        return PerceptionResult(
            timestamp=timestamp, frame_id=frame_id, face_present=False,
            detector="none",
        )
    landmarks = make_landmarks(frame_w, frame_h, face_cx, face_cy, face_scale,
                               mouth_open=mouth_open, eye_open=eye_open,
                               noise=noise, seed=seed)
    extractor = FacialFeatureExtractor()
    features = extractor.extract(landmarks, frame_w, frame_h)
    pose = HeadPose(yaw=yaw, pitch=pitch, roll=roll,
                    orientation=_orient(yaw, pitch),
                    confidence=head_confidence)
    detection = FaceDetection(
        bbox=bbox_from_landmarks(landmarks.pixels, frame_w, frame_h),
        confidence=0.9, timestamp=timestamp)
    from app.perception.audio_features import AudioFeatures
    return PerceptionResult(
        timestamp=timestamp, frame_id=frame_id, face_present=True,
        detection=detection, landmarks=landmarks, head_pose=pose,
        features=features,
        audio=AudioFeatures(vocalization=vocalization, rms=vocalization * 0.1,
                            noise_floor=0.01, available=True),
        detector="landmark",
    )


def make_features(**over):
    """TemporalFeatures with sensible defaults (also used by twin/bench)."""
    from app.temporal.sequence import TemporalFeatures

    base = dict(timestamp=1.0, face_present=True, mouth_open=0.6, eye_open=0.8,
                head_yaw=20.0, head_pitch=5.0, facial_motion=0.5, movement=0.5)
    base.update(over)
    return TemporalFeatures(**base)


def make_baby(**over):
    """BabyState with sensible defaults (also used by twin/bench)."""
    from app.state.baby_state import BabyState

    base = dict(face_present=True, attention=0.8, attention_confidence=0.8,
                facial_activity=0.6, movement=0.5, timestamp=1.0)
    base.update(over)
    return BabyState(**base)


def _orient(yaw: float, pitch: float) -> HeadOrientation:
    from app.perception.head_pose import classify_orientation
    return classify_orientation(yaw, pitch)


# -- scenario timelines ---------------------------------------------------
def timeline_no_face(n: int = 60) -> list[PerceptionResult]:
    return [make_perception(i * DT, i, face_present=False) for i in range(n)]


def timeline_stable_face(n: int = 150) -> list[PerceptionResult]:
    return [make_perception(i * DT, i) for i in range(n)]


def timeline_head_turn(n: int = 150) -> list[PerceptionResult]:
    out = []
    for i in range(n):
        yaw = 0.0 if i < 30 else min(45.0, (i - 30) * 1.5)
        out.append(make_perception(i * DT, i, yaw=yaw))
    return out


def timeline_mouth_movement(n: int = 150) -> list[PerceptionResult]:
    out = []
    for i in range(n):
        if i < 40:
            mouth = 0.1
        elif i < 70:
            mouth = 0.1 + (i - 40) * 0.02  # opening ramp -> MOUTH_OPENED
        elif i < 110:
            mouth = 0.7
        else:
            mouth = max(0.1, 0.7 - (i - 110) * 0.04)  # closing -> MOUTH_CLOSED
        out.append(make_perception(i * DT, i, mouth_open=mouth))
    return out


def timeline_rapid_movement(n: int = 120) -> list[PerceptionResult]:
    out = []
    for i in range(n):
        face_cx = 0.5 + 0.12 * math.sin(i * 0.8)
        yaw = 30.0 * math.sin(i * 0.6)
        out.append(make_perception(i * DT, i, face_cx=face_cx, yaw=yaw))
    return out


def timeline_noisy_landmarks(n: int = 150) -> list[PerceptionResult]:
    return [make_perception(i * DT, i, noise=6.0, seed=i) for i in range(n)]


def timeline_tracking_loss(n: int = 180) -> list[PerceptionResult]:
    """face for 60 frames, lost for 10 frames (inside grace), back for the rest"""
    out = []
    for i in range(n):
        face = not (60 <= i < 70)
        out.append(make_perception(i * DT, i, face_present=face))
    return out


def timeline_long_disengagement(n: int = 300) -> list[PerceptionResult]:
    """attention-relevant: face present but tiny/far and motionless"""
    out = []
    for i in range(n):
        out.append(make_perception(i * DT, i, face_scale=0.12, eye_open=0.3))
    return out


def timeline_engaged_interaction(n: int = 300) -> list[PerceptionResult]:
    """stable face -> head movement -> mouth open (smile) -> sustained engagement"""
    out = []
    for i in range(n):
        if i < 40:
            mouth, yaw, cx = 0.1, 0.0, 0.5
        elif i < 90:
            mouth, yaw, cx = 0.1 + (i - 40) * 0.012, min(30.0, (i - 40) * 0.8), 0.5
        elif i < 200:
            mouth, yaw, cx = 0.6, 30.0, 0.52
        else:
            mouth, yaw, cx = 0.5, 10.0, 0.5
        out.append(make_perception(i * DT, i, mouth_open=mouth, yaw=yaw, face_cx=cx))
    return out


SCENARIOS = {
    "no_face": timeline_no_face,
    "stable_face": timeline_stable_face,
    "head_turn": timeline_head_turn,
    "mouth_movement": timeline_mouth_movement,
    "rapid_movement": timeline_rapid_movement,
    "noisy_landmarks": timeline_noisy_landmarks,
    "tracking_loss": timeline_tracking_loss,
    "long_disengagement": timeline_long_disengagement,
    "engaged_interaction": timeline_engaged_interaction,
}
