"""Camera abstraction.

Reuses the proven backend-fallback + quality-hint approach from the existing
repository's ``facetrackingwithidentitylock/face_tracking.py``, rewritten as a
standalone class with no coupling to identity recognition or serial output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .frame import Frame, monotonic_now


@dataclass
class CameraMetadata:
    device: int
    backend: str
    width: int
    height: int
    fps: float
    opened: bool = False


class Camera:
    """OpenCV camera with open/read/release lifecycle and monotonic timestamps.

    ``mirror`` (default on) flips each frame horizontally, selfie-style:
    the baby's own right side appears on image-right, so turning right
    matches what the twin (and any viewer) sees.
    """

    def __init__(
        self,
        device: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 30,
        backend: str = "auto",
        apply_quality: bool = True,
        mirror: bool = True,
    ) -> None:
        self.device = device
        self.requested_width = width
        self.requested_height = height
        self.target_fps = target_fps
        self.backend_name = backend
        self.apply_quality = apply_quality
        self.mirror = mirror
        self._cap: Optional[cv2.VideoCapture] = None
        self.metadata = CameraMetadata(device=device, backend=backend, width=0, height=0, fps=0.0)
        self._frame_id = 0

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> CameraMetadata:
        if self._cap is not None:
            return self.metadata
        devices = [self.device] + [d for d in range(4) if d != self.device]
        last_err = None
        fallback = False
        for device in devices:
            meta, err = self._try_device(device)
            if meta is not None:
                fallback = device != self.device
                self.device = device
                self.metadata = meta
                self.metadata.device = device
                if fallback:
                    self.metadata.backend = f"{self.metadata.backend} (fallback from device {devices[0]})"
                return meta
            last_err = err
        raise RuntimeError(f"could not open camera {self.device}: {last_err or 'no backend worked'}")

    def _try_device(self, device: int) -> tuple[Optional[CameraMetadata], Optional[str]]:
        backends = self._backend_candidates()
        last_err = None
        for name, backend in backends:
            cap = cv2.VideoCapture(device, backend)
            if not cap.isOpened():
                cap.release()
                last_err = f"backend {name} failed to open"
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            if self.apply_quality:
                self._apply_quality(cap)
            # verify we actually get a non-blank frame
            ok, probe = cap.read()
            if not ok or probe is None:
                cap.release()
                last_err = f"backend {name}: read failed"
                continue
            if float(probe.mean()) <= 10.0:
                cap.release()
                last_err = f"backend {name}: blank frame from device {device}"
                continue
            self._cap = cap
            return CameraMetadata(
                device=device,
                backend=name,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=float(cap.get(cv2.CAP_PROP_FPS)) or float(self.target_fps),
                opened=True,
            ), None
        return None, last_err

    def _backend_candidates(self) -> list[tuple[str, int]]:
        named = {
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "default": cv2.CAP_ANY,
        }
        if self.backend_name != "auto" and self.backend_name in named:
            return [(self.backend_name, named[self.backend_name])]
        # Windows: MSMF first (better timestamping), then DSHOW, then default.
        if hasattr(cv2, "CAP_MSMF"):
            return [("msmf", cv2.CAP_MSMF), ("dshow", cv2.CAP_DSHOW), ("default", cv2.CAP_ANY)]
        return [("default", cv2.CAP_ANY)]

    @staticmethod
    def _apply_quality(cap: cv2.VideoCapture) -> None:
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def read(self) -> Frame:
        if self._cap is None:
            raise RuntimeError("camera not opened; call open() first")
        t0 = monotonic_now()
        ok, image = self._cap.read()
        t1 = monotonic_now()
        if not ok or image is None:
            raise RuntimeError("camera read failed")
        if self.mirror:
            image = cv2.flip(image, 1)
        frame = Frame(
            image=image,
            timestamp=t1,
            frame_id=self._frame_id,
            width=image.shape[1],
            height=image.shape[0],
            capture_latency_ms=(t1 - t0) * 1000.0,
        )
        self._frame_id += 1
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self.metadata.opened = False

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def list_cameras(max_index: int = 4) -> list[dict]:
    """Probe camera indices - diagnostic helper."""
    found = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, img = cap.read()
            if ok and img is not None:
                found.append(
                    {
                        "index": idx,
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                    }
                )
        cap.release()
    return found
