"""Textured 3D mesh geometry for the digital twin.

The twin is the baby's own face, drawn as a 3D point cloud + triangle mesh
whose expression and head rotation are amplified:

    fitted  = s * R * neutral + t          (rigid match of neutral -> current)
    target  = s * R_amp * (neutral - c_n) + c_cur + amplified(current - fitted)

- ``R_amp``: rotation with its angle multiplied by the head-rotation gain,
  applied about the face centroid ``c_n`` and re-anchored at the current
  head position ``c_cur`` (never about the coordinate origin, which would
  slide the head instead of turning it)
- residual: non-rigid expression displacement, amplified by the expression
  gain (and depth gain on z)

The neutral template adapts slowly (EMA) while the baby is at rest, so the
reference is this baby's own neutral face - not a generic average.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def rigid_fit(src: np.ndarray, dst: np.ndarray,
              with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity fit ``dst ~= s * R @ src + t`` (Kabsch/Umeyama).

    src, dst: (N, 3) point sets. Returns (scale, rotation (3,3), translation).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = max(1, len(src))
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    S = src - mu_src
    D = dst - mu_dst
    C = (D.T @ S) / n
    U, sig, Vt = np.linalg.svd(C)
    d = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
    Dm = np.diag([1.0, 1.0, d])
    R = U @ Dm @ Vt
    if with_scale:
        var_src = float((S ** 2).sum()) / n
        s = float((sig @ np.diag(Dm)) / max(var_src, 1e-12))
        if not np.isfinite(s) or s <= 1e-6:
            s = 1.0
    else:
        s = 1.0
    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def amplify_rotation(R: np.ndarray, gain: float) -> np.ndarray:
    """Return R with its rotation angle multiplied by ``gain``."""
    if abs(gain - 1.0) < 1e-9:
        return R
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
    rvec = rvec * gain
    R_amp, _ = cv2.Rodrigues(rvec)
    return R_amp


def triangulate(xy: np.ndarray) -> np.ndarray:
    """Deterministic triangle connectivity (T, 3) over 2D landmark positions."""
    from scipy.spatial import Delaunay

    simplices = Delaunay(np.asarray(xy, dtype=np.float64)).simplices
    # stable ordering: vertices sorted within a triangle, triangles sorted
    # lexicographically, so the same input always yields the same mesh
    rows = sorted(tuple(sorted(int(v) for v in tri)) for tri in simplices)
    return np.array(rows, dtype=np.int32)


def select_mesh_points(xy: np.ndarray, priority: Optional[np.ndarray],
                       spacing: float) -> np.ndarray:
    """Indices of the points used for the render mesh (boolean -> indices).

    All ``priority`` points are kept (lips/eyes/brows/contour stay sharp);
    remaining points are added greedily with at least ``spacing`` px between
    them, which cuts a dense 478-point face down to a few hundred triangles.
    """
    n = len(xy)
    keep = np.zeros(n, dtype=bool)
    if priority is not None and len(priority):
        pr = np.asarray(priority, dtype=np.int64)
        keep[pr[(pr >= 0) & (pr < n)]] = True
    kept_xy = xy[keep]
    for i in range(n):  # deterministic order: landmark index
        if keep[i]:
            continue
        if len(kept_xy):
            d = np.min(np.abs(kept_xy - xy[i]).max(axis=1))
            if d < spacing:
                continue
        keep[i] = True
        kept_xy = xy[keep]
    return np.flatnonzero(keep)


def extract_points(landmarks) -> np.ndarray:
    """FaceLandmarks -> (N, 3) float64 points in pixel units (x, y, z)."""
    xy = np.asarray(landmarks.pixels, dtype=np.float64)
    z = np.asarray(landmarks.depth_values(), dtype=np.float64)
    if z.shape[0] != xy.shape[0]:
        z = np.zeros(len(xy), dtype=np.float64)
    return np.hstack([xy, z[:, None]])


class NeutralTemplate:
    """Adaptive neutral reference face (this baby, at rest).

    Bootstraps from the first face frame, then drifts with an EMA only on
    frames where the baby's expression is neutral.
    """

    def __init__(self, update_rate: float, mouth_band: float,
                 neutral_mouth: float, max_facial_activity: float,
                 priority: Optional[np.ndarray] = None,
                 spacing_ratio: float = 0.12) -> None:
        self.update_rate = float(update_rate)
        self.mouth_band = float(mouth_band)
        self.neutral_mouth = float(neutral_mouth)
        self.max_facial_activity = float(max_facial_activity)
        self.priority = None if priority is None else np.asarray(priority,
                                                                  dtype=np.int64)
        self.spacing_ratio = float(spacing_ratio)
        self.points: Optional[np.ndarray] = None  # (N, 3)
        self.triangles: Optional[np.ndarray] = None

    @property
    def ready(self) -> bool:
        return self.points is not None

    def is_neutral(self, mouth_open: float, facial_activity: float) -> bool:
        return (abs(mouth_open - self.neutral_mouth) <= self.mouth_band
                and facial_activity <= self.max_facial_activity)

    def update(self, points: np.ndarray, mouth_open: float,
               facial_activity: float) -> bool:
        """Ingest the current face; returns True when the template is ready."""
        if self.points is None:
            self.points = np.array(points, dtype=np.float64, copy=True)
            xy = points[:, :2]
            span = float(max(xy[:, 0].max() - xy[:, 0].min(), 1.0))
            picked = select_mesh_points(xy, self.priority,
                                        self.spacing_ratio * span)
            self.triangles = picked[triangulate(xy[picked])]
        elif self.is_neutral(mouth_open, facial_activity):
            a = self.update_rate
            self.points = (1.0 - a) * self.points + a * np.asarray(points,
                                                                    dtype=np.float64)
        return True


class MeshExaggerator:
    """Computes amplified target points for the twin mesh."""

    def __init__(self, head_gain: float, expression_gain: float,
                 depth_gain: float) -> None:
        self.head_gain = float(head_gain)
        self.expression_gain = float(expression_gain)
        self.depth_gain = float(depth_gain)

    def compute(self, neutral: np.ndarray, current: np.ndarray,
                expression_mult: float = 1.0,
                head_mult: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Return (target_points, rigid_fit_points) both shaped (N, 3)."""
        s, R, t = rigid_fit(neutral, current)
        fitted = s * (neutral @ R.T) + t
        residual = current - fitted
        residual[:, :2] *= self.expression_gain * expression_mult
        residual[:, 2] *= self.depth_gain * expression_mult
        R_amp = amplify_rotation(R, self.head_gain * head_mult)
        c_n = neutral.mean(axis=0)
        c_cur = fitted.mean(axis=0)
        target = s * ((neutral - c_n) @ R_amp.T) + c_cur + residual
        return target, fitted
