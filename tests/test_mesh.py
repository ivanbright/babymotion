"""Textured 3D mesh geometry: rigid fit, triangulation, neutral template."""

import numpy as np
import pytest

from app.perception import landmarks as L
from app.twin.mesh import (MeshExaggerator, NeutralTemplate, amplify_rotation,
                           extract_points, rigid_fit, triangulate)
from synthetic import make_landmarks


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_rigid_fit_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(60, 3))
    R_true = _rot_z(0.4)
    s_true, t_true = 1.7, np.array([2.0, -3.0, 0.5])
    dst = s_true * (src @ R_true.T) + t_true
    s, R, t = rigid_fit(src, dst)
    assert s == pytest.approx(s_true, rel=1e-6)
    assert R == pytest.approx(R_true, abs=1e-9)
    assert t == pytest.approx(t_true, abs=1e-9)


def test_rigid_fit_identity_on_identical_points():
    pts = np.random.default_rng(1).normal(size=(50, 3))
    s, R, t = rigid_fit(pts, pts)
    assert s == pytest.approx(1.0, abs=1e-9)
    assert R == pytest.approx(np.eye(3), abs=1e-9)
    assert t == pytest.approx(np.zeros(3), abs=1e-9)


def test_amplify_rotation_scales_angle():
    R = _rot_z(0.3)
    R2 = amplify_rotation(R, 2.0)
    theta = np.arccos(np.clip((np.trace(R2) - 1) / 2, -1, 1))
    assert theta == pytest.approx(0.6, abs=1e-6)
    assert amplify_rotation(R, 1.0) is R


def test_triangulate_deterministic_and_valid():
    xy = make_landmarks().pixels.astype(np.float64)
    t1 = triangulate(xy)
    t2 = triangulate(xy)
    assert np.array_equal(t1, t2)
    assert t1.ndim == 2 and t1.shape[1] == 3
    assert t1.min() >= 0 and t1.max() < len(xy)
    # all triangles have positive area
    tri = xy[t1]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    area2 = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    assert (area2 > 1e-6).all()
    # nearly every landmark participates in the mesh
    coverage = len(np.unique(t1)) / len(xy)
    assert coverage > 0.95


def test_extract_points_shape_and_depth():
    lm = make_landmarks()
    pts = extract_points(lm)
    assert pts.shape == (len(lm.normalized), 3)
    assert np.allclose(pts[:, 0], lm.pixels[:, 0])
    # synthetic face has depth (dome): center closer (more negative) than edge
    assert pts[:, 2].min() < 0.0


def test_neutral_template_bootstrap_and_ema():
    tmpl = NeutralTemplate(update_rate=0.05, mouth_band=0.15,
                           neutral_mouth=0.10, max_facial_activity=0.12)
    assert not tmpl.ready
    pts = extract_points(make_landmarks())
    tmpl.update(pts, mouth_open=0.5, facial_activity=0.9)  # first face bootstraps
    assert tmpl.ready
    assert tmpl.triangles is not None
    assert tmpl.points == pytest.approx(pts)

    # neutral frame -> EMA moves the template
    neutral = tmpl.points.copy()
    shifted = pts + np.array([10.0, 0.0, 0.0])
    tmpl.update(shifted, mouth_open=0.05, facial_activity=0.0)
    expected = 0.95 * neutral + 0.05 * shifted
    assert tmpl.points == pytest.approx(expected)

    # non-neutral frame -> template untouched
    before = tmpl.points.copy()
    tmpl.update(pts * 2, mouth_open=0.9, facial_activity=0.9)
    assert tmpl.points == pytest.approx(before)


def test_neutral_classifier():
    tmpl = NeutralTemplate(0.05, 0.15, 0.10, 0.12)
    assert tmpl.is_neutral(mouth_open=0.12, facial_activity=0.05)
    assert not tmpl.is_neutral(mouth_open=0.40, facial_activity=0.05)
    assert not tmpl.is_neutral(mouth_open=0.10, facial_activity=0.50)


def test_exaggerator_identity_when_current_is_neutral():
    pts = extract_points(make_landmarks())
    ex = MeshExaggerator(head_gain=1.3, expression_gain=1.6, depth_gain=1.4)
    target, fitted = ex.compute(pts, pts)
    assert target == pytest.approx(pts, abs=1e-6)
    assert fitted == pytest.approx(pts, abs=1e-6)


def test_exaggerator_amplifies_mouth_opening():
    neutral = extract_points(make_landmarks(mouth_open=0.1))
    current = extract_points(make_landmarks(mouth_open=0.7))
    ex = MeshExaggerator(head_gain=1.0, expression_gain=1.6, depth_gain=1.4)
    target, _ = ex.compute(neutral, current)

    def opening(pts):
        return float(pts[L.MOUTH_INNER_BOTTOM, 1] - pts[L.MOUTH_INNER_TOP, 1])

    assert opening(target) > opening(current) * 1.1


def test_exaggerator_amplifies_head_rotation():
    neutral = extract_points(make_landmarks())
    center = neutral.mean(axis=0)
    pts2d = make_landmarks().pixels.astype(np.float64)
    pts2d = (pts2d - pts2d.mean(axis=0)) @ _rot_z(0.5)[:2, :2].T + pts2d.mean(axis=0)
    current = neutral.copy()
    current[:, :2] = pts2d
    ex = MeshExaggerator(head_gain=2.0, expression_gain=1.0, depth_gain=1.0)
    target, _ = ex.compute(neutral, current)

    v0 = neutral[0, :2] - center[:2]
    vc = current[0, :2] - center[:2]
    vt = target[0, :2] - center[:2]

    def ang(a, b):
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.arccos(np.clip(cos, -1, 1)))

    assert ang(v0, vt) > ang(v0, vc) + 0.1


def test_exaggerator_head_rotation_direction_not_inverted():
    """Yaw the face to the right: the twin must move further right, never
    to the opposite side (lateral-inversion regression)."""
    neutral = extract_points(make_landmarks())
    center = neutral.mean(axis=0)
    # rotate about the vertical (image-y) axis: x' = x cos(a) + z sin(a);
    # angle chosen so the nose (z closest to the camera, negative) moves +x
    a = np.deg2rad(-12.0)
    c, s = np.cos(a), np.sin(a)
    rel = neutral - center
    yawed = np.stack([rel[:, 0] * c + rel[:, 2] * s,
                      rel[:, 1],
                      -rel[:, 0] * s + rel[:, 2] * c], axis=1) + center
    ex = MeshExaggerator(head_gain=1.5, expression_gain=1.0, depth_gain=1.0)
    target, _ = ex.compute(neutral, yawed)

    nose = int(np.argmin(neutral[:, 2]))  # point closest to the camera
    dx_now = float(yawed[nose, 0] - neutral[nose, 0])
    dx_tgt = float(target[nose, 0] - neutral[nose, 0])
    assert abs(dx_now) > 1e-6
    assert float(np.sign(dx_tgt)) == float(np.sign(dx_now))  # not flipped
    assert abs(dx_tgt) > abs(dx_now)  # amplified
