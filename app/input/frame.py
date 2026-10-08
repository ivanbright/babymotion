"""Frame container shared by every pipeline stage."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def monotonic_now() -> float:
    """Monotonic timestamp used for every latency / temporal calculation."""
    return time.monotonic()


@dataclass
class Frame:
    image: np.ndarray  # BGR, HxWx3
    timestamp: float  # monotonic seconds
    frame_id: int
    width: int
    height: int
    capture_latency_ms: float = 0.0  # camera read latency if known

    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        frame_id: int,
        timestamp: float | None = None,
        capture_latency_ms: float = 0.0,
    ) -> "Frame":
        h, w = image.shape[:2]
        return cls(
            image=image,
            timestamp=timestamp if timestamp is not None else monotonic_now(),
            frame_id=frame_id,
            width=w,
            height=h,
            capture_latency_ms=capture_latency_ms,
        )
