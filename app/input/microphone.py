"""Optional microphone input.

Audio is optional: if ``sounddevice`` is unavailable or capture fails, the
engine runs video-only with a silent AudioFeatures stream.
"""

from __future__ import annotations

import collections
import time
from typing import Optional

import numpy as np


class Microphone:
    """Ring buffer of recent RMS energy blocks.

    Non-blocking: the audio callback appends small blocks; consumers read a
    rolling window. Degrades to a silent stub when sounddevice is missing.
    """

    def __init__(self, sample_rate: int = 16000, block_ms: int = 100, device: int | None = None):
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.device = device
        self._blocks: collections.deque[tuple[float, float]] = collections.deque(maxlen=200)
        self._stream = None
        self.available = False
        self.error: Optional[str] = None

    def start(self) -> bool:
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:  # pragma: no cover - environment dependent
            self.error = f"sounddevice unavailable: {exc}"
            return False
        try:
            block_samples = int(self.sample_rate * self.block_ms / 1000)

            def callback(indata, frames, time_info, status) -> None:  # pragma: no cover
                mono = indata.mean(axis=1) if indata.ndim > 1 else indata
                rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
                self._blocks.append((time.monotonic(), rms))

            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=block_samples,
                dtype="float32",
                device=self.device,
                callback=callback,
            )
            self._stream.start()
            self.available = True
            return True
        except Exception as exc:  # pragma: no cover
            self.error = f"microphone open failed: {exc}"
            self.available = False
            return False

    def recent_rms(self, window_s: float = 1.0) -> list[tuple[float, float]]:
        cutoff = time.monotonic() - window_s
        return [(t, v) for t, v in self._blocks if t >= cutoff]

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.available = False
