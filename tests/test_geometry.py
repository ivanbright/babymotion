"""Geometry tests: mouth/eye ratios, normalization, exaggeration math."""

import numpy as np
import pytest

from app.perception import landmarks as L
from app.perception.facial_features import FacialFeatureExtractor
from app.perception.head_pose import HeadOrientation, classify_orientation
from app.twin.exaggeration import ExaggerationEngine
from synthetic import make_landmarks

FRAME_W, FRAME_H = 1280, 720


def _features(**kwargs):
    lm = make_landmarks(FRAME_W, FRAME_H, **kwargs)
    return FacialFeatureExtractor().extract(lm, FRAME_W, FRAME_H)


def test_mouth_open_ratio_matches_construction():
    f_lo = _features(mouth_open=0.1)
    f_hi = _features(mouth_open=0.7)
    assert f_hi.mouth_open > f_lo.mouth_open + 0.3
    assert f_hi.mouth_open == pytest.approx(0.7, abs=0.05)
    assert 0.0 <= f_lo.mouth_open <= 1.0


def test_eye_open_ratio_matches_construction():
    f = _features(eye_open=0.85)
    assert f.eye_open_left == pytest.approx(0.85, abs=0.05)
    assert f.eye_open_right == pytest.approx(0.85, abs=0.05)
    closed = _features(eye_open=0.05)
    assert closed.eye_open_left < 0.15


def test_normalization_independent_of_position_and_size():
    a = _features(face_cx=0.4, face_cy=0.4, face_w_frac=0.25, mouth_open=0.4)
    b = _features(face_cx=0.6, face_cy=0.6, face_w_frac=0.40, mouth_open=0.4)
    assert a.mouth_open == pytest.approx(b.mouth_open, abs=0.06)
    assert a.mouth_width == pytest.approx(b.mouth_width, abs=0.06)
    assert a.face_width > 0 and b.face_width > 0
    assert b.face_width > a.face_width


def test_face_center_and_scale():
    f = _features(face_cx=0.5, face_cy=0.5, face_w_frac=0.3)
    assert abs(f.face_center[0] - 640) < 40
    assert abs(f.face_center[1] - 360) < 40
    assert 0.1 < f.face_scale < 0.6


def test_brow_raise_increases_eyebrow_position():
    low = _features(brow_raise=0.0)
    high = _features(brow_raise=1.0)
    assert high.eyebrow_position > low.eyebrow_position


def test_orientation_classification_viewer_perspective():
    assert classify_orientation(0.0, 0.0) == HeadOrientation.LOOKING_CENTER
    assert classify_orientation(30.0, 0.0) == HeadOrientation.LOOKING_RIGHT
    assert classify_orientation(-30.0, 0.0) == HeadOrientation.LOOKING_LEFT
    assert classify_orientation(0.0, 30.0) == HeadOrientation.LOOKING_DOWN
    assert classify_orientation(0.0, -30.0) == HeadOrientation.LOOKING_UP


def test_landmark_index_constants_distinct():
    ids = [L.NOSE_TIP, L.CHIN, L.EYE_R_OUTER, L.EYE_L_OUTER, L.MOUTH_LEFT,
           L.MOUTH_RIGHT, L.MOUTH_INNER_TOP, L.MOUTH_INNER_BOTTOM, L.FACE_LEFT,
           L.FACE_RIGHT, L.FOREHEAD]
    assert len(set(ids)) == len(ids)


# -- exaggeration engine --------------------------------------------------
def test_exaggeration_formula_and_clamp(settings):
    eng = ExaggerationEngine(settings.twin)
    # neutral + gain * (value - neutral)
    out = eng.amplify(0.6, neutral=0.1, gain=1.5, limits=(0.0, 1.0))
    assert out == pytest.approx(0.1 + 1.5 * (0.6 - 0.1))
    # clamped at the top
    out = eng.amplify(1.0, neutral=0.0, gain=3.0, limits=(0.0, 1.0))
    assert out == 1.0
    # clamped at the bottom
    out = eng.amplify(0.0, neutral=0.5, gain=3.0, limits=(0.0, 1.0))
    assert out == 0.0


def test_exaggeration_head_rotation_gain(settings):
    eng = ExaggerationEngine(settings.twin)
    yaw = eng.amplify(20.0, neutral=0.0,
                      gain=settings.twin.exaggeration.head_rotation,
                      limits=settings.twin.limits["head_yaw"])
    assert yaw == pytest.approx(20.0 * 1.3, abs=1e-6)
    assert yaw < settings.twin.limits["head_yaw"][1]


def test_gains_come_from_settings_not_hardcoded(settings):
    settings.twin.exaggeration.mouth = 2.5
    eng = ExaggerationEngine(settings.twin)
    out = eng.amplify(0.4, neutral=0.1, gain=2.5, limits=(0.0, 1.0))
    assert out == pytest.approx(0.1 + 2.5 * 0.3)
    # clamp still wins over the gain
    assert eng.amplify(0.9, neutral=0.1, gain=2.5, limits=(0.0, 1.0)) == 1.0
