"""Head pose estimation tests with synthetic projected landmarks."""

import cv2
import numpy as np
import pytest

from app.perception.head_pose import (
    HeadPoseEstimator,
    HeadOrientation,
    _MODEL_POINTS,
    _REFERENCE_UPRIGHT,
)
from app.perception.landmarks import FaceLandmarks, Landmark

W, H = 1280, 720
_FOCAL = float(max(W, H))
_K = np.array([[_FOCAL, 0.0, W / 2.0], [0.0, _FOCAL, H / 2.0], [0.0, 0.0, 1.0]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _compose(yaw_deg, pitch_deg, roll_deg) -> np.ndarray:
    """Matches the estimator's extraction: R = Rz(-roll) Ry(yaw) Rx(pitch)."""
    y, p, r = np.radians([yaw_deg, pitch_deg, roll_deg])
    return _rot_z(-r) @ _rot_y(y) @ _rot_x(p)


def _landmarks_from_pose(yaw, pitch, roll, tvec=(0.0, 0.0, 600.0)) -> FaceLandmarks:
    rrel = _compose(yaw, pitch, roll)
    rsolve = _REFERENCE_UPRIGHT @ rrel
    rvec, _ = cv2.Rodrigues(rsolve)
    t = np.array(tvec, dtype=np.float64).reshape(3, 1)
    projected, _ = cv2.projectPoints(_MODEL_POINTS, rvec, t, _K,
                                     np.zeros((4, 1)))
    pts = projected.reshape(-1, 2).astype(np.float32)
    pixels = np.tile(np.array([[W / 2, H / 2]], dtype=np.float32), (478, 1))
    for i, pt in zip((4, 152, 33, 263, 61, 291), pts):
        pixels[i] = pt
    normalized = [Landmark(x=float(p[0] / W), y=float(p[1] / H)) for p in pixels]
    visibility = np.ones(478, dtype=np.float32)
    return FaceLandmarks(normalized=normalized, pixels=pixels,
                         visibility=visibility, timestamp=0.0)


@pytest.mark.parametrize("yaw,pitch,roll", [
    (0.0, 0.0, 0.0),
    (25.0, 0.0, 0.0),
    (-30.0, 0.0, 0.0),
    (0.0, 20.0, 0.0),
    (0.0, -15.0, 0.0),
    (10.0, -8.0, 12.0),
])
def test_recovers_known_pose(yaw, pitch, roll):
    lm = _landmarks_from_pose(yaw, pitch, roll)
    est = HeadPoseEstimator(W, H)
    pose = est.estimate(lm, W, H)
    assert pose.yaw == pytest.approx(yaw, abs=2.0)
    assert pose.pitch == pytest.approx(pitch, abs=2.0)
    assert pose.roll == pytest.approx(roll, abs=2.0)
    assert pose.confidence > 0.5


def test_frontal_face_is_center():
    lm = _landmarks_from_pose(0, 0, 0)
    pose = HeadPoseEstimator(W, H).estimate(lm, W, H)
    assert pose.orientation == HeadOrientation.LOOKING_CENTER


def test_yaw_sign_matches_viewer_perspective():
    # positive yaw: face oriented toward image right -> LOOKING_RIGHT
    right = HeadPoseEstimator(W, H).estimate(_landmarks_from_pose(30, 0, 0), W, H)
    left = HeadPoseEstimator(W, H).estimate(_landmarks_from_pose(-30, 0, 0), W, H)
    assert right.orientation == HeadOrientation.LOOKING_RIGHT
    assert left.orientation == HeadOrientation.LOOKING_LEFT


def test_unknown_when_landmarks_invisible():
    lm = _landmarks_from_pose(0, 0, 0)
    lm.visibility[4] = 0.0
    pose = HeadPoseEstimator(W, H).estimate(lm, W, H)
    assert pose.orientation == HeadOrientation.UNKNOWN
    assert pose.confidence == 0.0


def test_small_face_penalizes_confidence():
    lm_small = _landmarks_from_pose(0, 0, 0, tvec=(0.0, 0.0, 2400.0))
    lm_big = _landmarks_from_pose(0, 0, 0, tvec=(0.0, 0.0, 500.0))
    est = HeadPoseEstimator(W, H)
    small = est.estimate(lm_small, W, H)
    big = est.estimate(lm_big, W, H)
    assert small.confidence < big.confidence
