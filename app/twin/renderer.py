"""Digital twin renderer: textured 3D mesh of the baby's own face.

The baby's own pixels are warped through an exaggerated 3D point cloud:

    displacement field (current -> amplified target, barycentric over the
    face hull, coarse grid) -> one cv2.remap of the frame -> depth-based
    flat shading -> compose over a blurred background -> cropped panel

Performance notes (this runs inside the debug loop, must stay well under the
camera frame budget):
- background blur is computed at 1/8 resolution and upscaled
- the render mesh is decimated (priority landmarks + spacing); the rigid fit
  still uses ALL 478 points
- everything is vectorized (no per-triangle Python loop) and rasterized at
  ``render_scale`` before upscaling to the panel
- rasterization is throttled to ``min_frame_interval_ms``; geometry (neutral
  template, smoothing) still updates every frame

Rendering is a display-layer concern: it never touches the deterministic
engine loop or anything recorded (sessions store structured signals, no
images).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config.settings import Settings
from ..perception import landmarks as L
from .mesh import MeshExaggerator, NeutralTemplate, extract_points
from .twin_state import TwinState

# twin actions that should amplify the expression residual
_EXPRESSION_ACTIONS = {
    "EXAGGERATE", "SMILE", "OPEN_MOUTH", "INTRODUCE_NOVELTY",
    "ATTRACT_ATTENTION", "ENCOURAGE", "REPEAT",
}
# twin actions that should amplify the head rotation
_HEAD_ACTIONS = {"TURN_HEAD"}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class MeshTwinRenderer:
    """Stateful per-view renderer (owns the neutral template + smoothing)."""

    def __init__(self, settings: Settings) -> None:
        mesh = settings.twin.mesh
        self.mesh = mesh
        head_gain = mesh.head_rotation_gain or settings.twin.exaggeration.head_rotation
        self.exaggerator = MeshExaggerator(
            head_gain=head_gain,
            expression_gain=mesh.expression_gain,
            depth_gain=mesh.depth_gain,
        )
        self._priority = np.array(L.MESH_PRIORITY, dtype=np.int64)
        self._neutral_mouth = float(settings.twin.neutral.mouth_open)
        self._new_template()
        self._maps: dict = {}
        self._oval: Optional[np.ndarray] = None
        self._smoothed: Optional[np.ndarray] = None
        self._last_ts: Optional[float] = None
        self._action: str = ""
        self._intensity: float = 0.0
        self._last_render_ts: Optional[float] = None
        self._last_panel: Optional[np.ndarray] = None

    def _new_template(self) -> None:
        self.template = NeutralTemplate(
            update_rate=self.mesh.neutral_update_rate,
            mouth_band=self.mesh.neutral_mouth_band,
            neutral_mouth=self._neutral_mouth,
            max_facial_activity=self.mesh.neutral_max_facial_activity,
            priority=self._priority,
            spacing_ratio=self.mesh.spacing_ratio,
        )

    def reset(self) -> None:
        """Forget the neutral template (new session/window)."""
        self._neutral_mouth = self.template.neutral_mouth
        self._new_template()
        self._smoothed = None
        self._last_ts = None
        self._last_render_ts = None
        self._last_panel = None

    # -- public -----------------------------------------------------------
    def render(self, frame: np.ndarray, landmarks, features,
               twin_state: TwinState, facial_activity: float = 0.0) -> np.ndarray:
        """Return the twin panel image for this frame."""
        if landmarks is None or getattr(landmarks, "count", 0) < 100:
            return self._placeholder()

        self._action = str(twin_state.action or "IDLE")
        self._intensity = float(twin_state.action_intensity)

        # geometry runs every frame (neutral template + smoothing), even when
        # rasterization below is throttled
        current = extract_points(landmarks)
        if self._oval is None:
            self._oval = np.array([i for i in L.FACE_OVAL if i < len(current)],
                                  dtype=np.int64)
        mouth_open = float(features.mouth_open) if features is not None else 0.0
        self.template.update(current, mouth_open, facial_activity)
        target = self._target_points(current, mouth_open, facial_activity)
        target = self._smooth(target, float(landmarks.timestamp))
        h_frame, w_frame = frame.shape[:2]
        target[:, 0] = np.clip(target[:, 0], 0.0, w_frame - 1.0)
        target[:, 1] = np.clip(target[:, 1], 0.0, h_frame - 1.0)

        # throttle rasterization (geometry above still ran)
        interval = self.mesh.min_frame_interval_ms / 1000.0
        ts = float(landmarks.timestamp)
        delta = ts - (self._last_render_ts if self._last_render_ts is not None
                      else ts)
        if self._last_panel is not None and 0.0 <= delta < interval:
            return self._last_panel
        self._last_render_ts = ts

        if self.template.triangles is None:
            return self._placeholder()
        panel = self._rasterize(frame, current[:, :2], target)
        self._last_panel = panel
        return panel

    # -- geometry ---------------------------------------------------------
    def _target_points(self, current: np.ndarray, mouth_open: float,
                       facial_activity: float) -> np.ndarray:
        if self.template.points is None:
            return current
        expr_mult, head_mult = self._action_multipliers()
        target, _ = self.exaggerator.compute(
            self.template.points, current,
            expression_mult=expr_mult, head_mult=head_mult)
        return target

    def _action_multipliers(self) -> tuple[float, float]:
        action = self._action or ""
        intensity = float(np.clip(self._intensity, 0.0, 1.0))
        boost = self.mesh.action_boost
        expr = 1.0 + (boost - 1.0) * intensity if action in _EXPRESSION_ACTIONS else 1.0
        head = 1.0 + (boost - 1.0) * intensity if action in _HEAD_ACTIONS else 1.0
        return expr, head

    def _smooth(self, target: np.ndarray, timestamp: float) -> np.ndarray:
        tau = max(1.0, self.mesh.vertex_smoothing) / 1000.0
        if self._smoothed is None or self._last_ts is None:
            alpha = 1.0
        else:
            dt = float(np.clip(timestamp - self._last_ts, 0.0, 0.5))
            alpha = 1.0 - float(np.exp(-dt / tau))
        self._last_ts = timestamp
        if self._smoothed is None:
            self._smoothed = target.copy()
        else:
            self._smoothed = self._smoothed + alpha * (target - self._smoothed)
        return self._smoothed

    def _vertex_shading(self, target: np.ndarray) -> np.ndarray:
        """Per-vertex brightness (0.55..1.3) from the mesh normal, or 1s."""
        n_pts = len(target)
        if not self.mesh.shading:
            return np.ones(n_pts, dtype=np.float32)
        tri = self.template.triangles
        pts = target[tri]  # (T, 3, 3)
        n = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        n[n[:, 2] > 0] *= -1.0  # orient toward the camera (camera at -z)
        light = np.array([-0.25, -0.45, -0.85])
        light = light / np.linalg.norm(light)
        tri_shade = np.clip(0.55 + 0.45 * np.clip(n @ light, 0.0, 1.0),
                            0.0, 1.3)
        shade = np.zeros(n_pts, dtype=np.float64)
        counts = np.zeros(n_pts, dtype=np.float64)
        flat = tri.ravel()
        np.add.at(shade, flat, np.repeat(tri_shade, 3))
        np.add.at(counts, flat, 1.0)
        out = np.ones(n_pts, dtype=np.float64)
        np.divide(shade, counts, out=out, where=counts > 0)
        return out.astype(np.float32)

    # -- rasterization ----------------------------------------------------
    def _rasterize(self, frame: np.ndarray, src_xy: np.ndarray,
                   target: np.ndarray) -> np.ndarray:
        """Vectorized textured-mesh warp: displacement field + one remap.

        The baby's pixels are sampled through the piecewise-linear map
        current -> exaggerated target, evaluated on a coarse grid via
        barycentric coordinates of the (target) face hull, then a single
        ``cv2.remap`` applies it to the whole image. No per-triangle Python
        loop - everything is vectorized.
        """
        from scipy.spatial import Delaunay

        scale = float(self.mesh.render_scale or 1.0)
        h_frame, w_frame = frame.shape[:2]
        ws = max(2, int(w_frame * scale))
        hs = max(2, int(h_frame * scale))
        texture = (cv2.resize(frame, (ws, hs), interpolation=cv2.INTER_AREA)
                   if scale != 1.0 else frame)
        background = self._background(frame, ws, hs, scale)

        dst_pts = target[:, :2] * scale                 # (N, 2) work-res
        disp = ((target[:, :2] - src_xy) * scale).astype(np.float32)
        shade_v = self._vertex_shading(target)

        # coarse grid over the canvas; barycentric lookup in the face hull
        step = max(2, int(self.mesh.grid_step))
        xs = np.append(np.arange(0, ws, step, dtype=np.float64), ws - 1.0)
        ys = np.append(np.arange(0, hs, step, dtype=np.float64), hs - 1.0)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        ny, nx = len(ys), len(xs)

        try:
            hull = Delaunay(dst_pts)
        except Exception:
            return self._last_panel or self._placeholder()
        sidx = hull.find_simplex(grid)
        inside = sidx >= 0
        disp_g = np.zeros((len(grid), 2), dtype=np.float32)
        shade_g = np.ones(len(grid), dtype=np.float32)
        if inside.any():
            simp = hull.simplices
            A = dst_pts[simp]                            # (S, 3, 2)
            e1 = A[:, 1] - A[:, 0]
            e2 = A[:, 2] - A[:, 0]
            det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            det = np.where(np.abs(det) < 1e-12, 1e-12, det)
            idx = sidx[inside]
            rel = grid[inside] - A[idx][:, 0, :]
            e1v, e2v, detv = e1[idx], e2[idx], det[idx]
            u0 = (e2v[:, 1] * rel[:, 0] - e2v[:, 0] * rel[:, 1]) / detv
            u1 = (e1v[:, 0] * rel[:, 1] - e1v[:, 1] * rel[:, 0]) / detv
            lam = np.stack([1.0 - u0 - u1, u0, u1], axis=1).astype(np.float32)
            verts = simp[idx]
            disp_g[inside] = np.einsum("mi,mij->mj", lam, disp[verts])
            shade_g[inside] = np.einsum("mi,mi->m", lam, shade_v[verts])

        dmap = cv2.resize(disp_g.reshape(ny, nx, 2), (ws, hs),
                          interpolation=cv2.INTER_LINEAR)
        smap = cv2.resize(shade_g.reshape(ny, nx), (ws, hs),
                          interpolation=cv2.INTER_LINEAR)
        id_x, id_y = self._identity_maps(ws, hs)
        sharp = cv2.remap(texture, id_x - dmap[:, :, 0], id_y - dmap[:, :, 1],
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # compose: sharp warped face inside the face silhouette, blurred behind
        mask = np.zeros((hs, ws), dtype=np.uint8)
        oval = np.round(dst_pts[self._oval]).astype(np.int32)
        cv2.fillConvexPoly(mask, oval, 255, lineType=cv2.LINE_AA)
        canvas = background.copy()
        cv2.copyTo(sharp, mask, canvas)

        sel = smap < 0.999
        if sel.any():
            vals = smap[sel].astype(np.float32)
            canvas[sel] = np.clip(
                canvas[sel].astype(np.float32) * vals[:, None], 0, 255
            ).astype(np.uint8)
        return self._crop_panel(canvas, dst_pts)

    def _identity_maps(self, ws: int, hs: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._maps.get((ws, hs))
        if cached is None:
            cached = (np.tile(np.arange(ws, dtype=np.float32), (hs, 1)),
                      np.tile(np.arange(hs, dtype=np.float32)[:, None],
                              (1, ws)))
            self._maps[(ws, hs)] = cached
        return cached

    def _background(self, frame: np.ndarray, ws: int, hs: int,
                    scale: float) -> np.ndarray:
        blur = self.mesh.background_blur
        if not blur or blur <= 0:
            return cv2.resize(frame, (ws, hs), interpolation=cv2.INTER_AREA)
        # blur at 1/8 resolution: a full-res sigma-21 blur costs ~250 ms
        sw = max(8, ws // 8)
        sh = max(8, hs // 8)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(0.5, blur * scale / 8.0))
        return cv2.resize(small, (ws, hs), interpolation=cv2.INTER_LINEAR)

    def _crop_panel(self, canvas: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
        x0, y0 = dst_xy.min(axis=0)
        x1, y1 = dst_xy.max(axis=0)
        mx = 0.10 * max(1.0, x1 - x0)
        my = 0.10 * max(1.0, y1 - y0)
        h, w = canvas.shape[:2]
        x0 = int(max(0, x0 - mx))
        y0 = int(max(0, y0 - my))
        x1 = int(min(w, x1 + mx))
        y1 = int(min(h, y1 + my))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return self._placeholder()
        crop = canvas[y0:y1, x0:x1]
        ph = int(self.mesh.panel_height)
        pw = max(64, int(ph * (x1 - x0) / max(1, y1 - y0)))
        return cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_AREA)

    def _placeholder(self) -> np.ndarray:
        ph = int(self.mesh.panel_height)
        pw = max(64, int(ph * 0.8))
        panel = np.full((ph, pw, 3), 24, dtype=np.uint8)
        cv2.putText(panel, "TWIN", (16, 34), _FONT, 0.8, (90, 90, 90), 1,
                    cv2.LINE_AA)
        cv2.putText(panel, "waiting", (16, 62), _FONT, 0.55, (70, 70, 70), 1,
                    cv2.LINE_AA)
        cv2.putText(panel, "for face", (16, 84), _FONT, 0.55, (70, 70, 70), 1,
                    cv2.LINE_AA)
        return panel


# backwards-compatible name (the twin API surface used to be a plain class)
TwinRenderer = MeshTwinRenderer
