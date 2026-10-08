"""Temporal signal processing.

Consumes one PerceptionResult per frame and produces smoothed temporal
features plus discrete temporal events. Nothing here decides what the twin
does - it only measures and detects.

Motion calibration notes (normalized, dimensionless):
  facial_motion : windowed per-frame feature delta x FACIAL_GAIN, clamped 0..1
  head_motion   : head angular speed / HEAD_SPEED_REF (deg/s), EMA, 0..1
  center_motion : face-center speed / face-width per second / CENTER_REF, 0..1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config.settings import Settings
from ..metrics.events import Event, EventLog, EventType
from ..perception.head_pose import HeadOrientation
from ..perception.perception_result import PerceptionResult
from .smoothing import EMASmoother, Hysteresis, MinDurationGate, clamp
from .window import TimedWindow

FACIAL_GAIN = 10.0
HEAD_SPEED_REF = 60.0     # deg/s -> 1.0
CENTER_REF = 2.0          # face-widths/s -> 1.0

MOUTH_OPEN_ON = 0.35
MOUTH_OPEN_OFF = 0.25
VOCAL_ON = 0.30
VOCAL_OFF = 0.15


@dataclass
class TemporalFeatures:
    timestamp: float
    face_present: bool
    mouth_open: float = 0.0
    mouth_open_velocity: float = 0.0
    eye_open: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_motion: float = 0.0
    facial_motion: float = 0.0
    center_motion: float = 0.0
    movement: float = 0.0
    face_scale: float = 0.0
    vocalization: float = 0.0
    orientation: HeadOrientation = HeadOrientation.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "face_present": self.face_present,
            "mouth_open": round(self.mouth_open, 4),
            "mouth_open_velocity": round(self.mouth_open_velocity, 4),
            "eye_open": round(self.eye_open, 4),
            "head_yaw": round(self.head_yaw, 2),
            "head_pitch": round(self.head_pitch, 2),
            "head_motion": round(self.head_motion, 4),
            "facial_motion": round(self.facial_motion, 4),
            "center_motion": round(self.center_motion, 4),
            "movement": round(self.movement, 4),
            "face_scale": round(self.face_scale, 4),
            "vocalization": round(self.vocalization, 4),
            "orientation": self.orientation.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TemporalFeatures":
        return cls(
            timestamp=d["timestamp"], face_present=d["face_present"],
            mouth_open=d.get("mouth_open", 0.0),
            mouth_open_velocity=d.get("mouth_open_velocity", 0.0),
            eye_open=d.get("eye_open", 0.0),
            head_yaw=d.get("head_yaw", 0.0), head_pitch=d.get("head_pitch", 0.0),
            head_motion=d.get("head_motion", 0.0),
            facial_motion=d.get("facial_motion", 0.0),
            center_motion=d.get("center_motion", 0.0),
            movement=d.get("movement", 0.0),
            face_scale=d.get("face_scale", 0.0),
            vocalization=d.get("vocalization", 0.0),
            orientation=HeadOrientation(d.get("orientation", "UNKNOWN")),
        )


class TemporalEngine:
    """Sliding windows, smoothing, and event detection over perception output."""

    def __init__(self, settings: Settings, event_log: EventLog,
                 session_id: Optional[str] = None) -> None:
        self.settings = settings
        self.events = event_log
        self.session_id = session_id
        t = settings.temporal
        self.win_mouth = TimedWindow(t.window_short_ms)
        self.win_yaw = TimedWindow(t.window_medium_ms)
        self.win_pitch = TimedWindow(t.window_medium_ms)
        self.win_motion = TimedWindow(t.window_medium_ms)
        self.win_attention = TimedWindow(t.window_short_ms)

        alpha = t.ema_alpha
        self.ema_mouth = EMASmoother(alpha)
        self.ema_eye = EMASmoother(alpha)
        self.ema_head_motion = EMASmoother(alpha)
        self.ema_facial = EMASmoother(alpha)
        self.ema_center = EMASmoother(alpha)
        self.ema_vocal = EMASmoother(alpha)

        self.mouth_hyst = Hysteresis(MOUTH_OPEN_ON, MOUTH_OPEN_OFF)
        self.vocal_hyst = Hysteresis(VOCAL_ON, VOCAL_OFF)
        self.facial_hyst = Hysteresis(t.motion_enter, t.motion_exit)
        self.head_hyst = Hysteresis(t.motion_enter, t.motion_exit)
        self.attention_high_hyst = Hysteresis(settings.attention.engage_threshold,
                                              settings.attention.engage_threshold - 0.05)
        self.attention_low_hyst = Hysteresis(settings.attention.disengage_threshold + 0.05,
                                             settings.attention.disengage_threshold)
        self.orient_gate = MinDurationGate(t.event_min_duration_ms / 1000.0)
        self.min_duration_s = t.event_min_duration_ms / 1000.0
        self.face_grace_s = t.face_loss_grace_ms / 1000.0
        self._face_seen_at: Optional[float] = None
        self._face_present = False

        self._prev_features: Optional[dict] = None
        self._prev_center: Optional[tuple[float, float]] = None
        self._prev_face_w = 100.0
        self._prev_ts: Optional[float] = None
        self._mouth_since: Optional[float] = None
        self._vocal_since: Optional[float] = None
        self._facial_since: Optional[float] = None
        self._head_since: Optional[float] = None
        self._attention_state: Optional[bool] = None
        self._attention_since: Optional[float] = None

        self.features = TemporalFeatures(timestamp=0.0, face_present=False)

    # ------------------------------------------------------------------
    def update(self, perception: PerceptionResult) -> tuple[TemporalFeatures, list[Event]]:
        """Update features from one perception result.

        Attention is fed separately via note_attention() once the attention
        estimator has run on these features.
        """
        ts = perception.timestamp
        dt = (ts - self._prev_ts) if self._prev_ts is not None else 0.0
        self._prev_ts = ts
        new_events: list[Event] = []

        # --- face presence with grace period --------------------------
        if perception.face_present:
            if not self._face_present:
                self._face_present = True
                new_events.append(self._emit(EventType.FACE_DETECTED, ts,
                                             detector=perception.detector))
            self._face_seen_at = ts
        elif self._face_present and self._face_seen_at is not None:
            if ts - self._face_seen_at > self.face_grace_s:
                self._face_present = False
                self._reset_tracking()
                new_events.append(self._emit(EventType.FACE_LOST, ts))

        if not self._face_present or perception.features is None:
            # decay motion estimates so state doesn't freeze on last value
            self.ema_facial.update(0.0, dt)
            self.ema_head_motion.update(0.0, dt)
            self.features = TemporalFeatures(timestamp=ts, face_present=False,
                                             vocalization=self.ema_vocal.value,
                                             face_scale=self.features.face_scale)
            self._prev_ts = ts
            return self.features, new_events

        f = perception.features
        pose = perception.head_pose
        feat_d = f.to_dict()
        # constants below are documented calibration knobs (see module docstring)
        # --- per-frame deltas -> raw motion ---------------------------
        expr_delta = 0.0
        if self._prev_features is not None:
            p = self._prev_features
            expr_delta = (
                abs(f.mouth_open - p["mouth_open"])
                + abs(f.eyebrow_position - p["eyebrow_position"])
                + 0.5 * abs(f.eye_open_left - p["eye_open_left"])
                + 0.5 * abs(f.eye_open_right - p["eye_open_right"])
                + 4.0 * abs(f.nose_position[0] - p["nose_position"][0])
                + 4.0 * abs(f.nose_position[1] - p["nose_position"][1])
            )
        raw_facial = clamp(expr_delta * FACIAL_GAIN, 0.0, 1.0)

        center_speed_norm = 0.0
        if self._prev_center is not None and dt > 1e-4:
            dx = f.face_center[0] - self._prev_center[0]
            dy = f.face_center[1] - self._prev_center[1]
            center_speed_norm = clamp((np.hypot(dx, dy) / self._prev_face_w) / CENTER_REF, 0.0, 1.0)
        self._prev_center = f.face_center
        self._prev_face_w = max(1.0, f.face_width)

        yaw = pose.yaw if pose else 0.0
        pitch = pose.pitch if pose else 0.0
        ang_speed = 0.0
        if self.win_yaw.latest() is not None and dt > 1e-4:
            dyaw = abs(yaw - self.win_yaw.latest())
            dpitch = abs(pitch - self.win_pitch.latest())
            ang_speed = (dyaw + dpitch) / dt
        head_motion = clamp(ang_speed / HEAD_SPEED_REF, 0.0, 1.0)

        # --- windows --------------------------------------------------
        self.win_mouth.append(ts, f.mouth_open)
        self.win_yaw.append(ts, yaw)
        self.win_pitch.append(ts, pitch)
        self.win_motion.append(ts, raw_facial)

        mouth_stats = self.win_mouth.stats()
        # --- smoothed features ---------------------------------------
        mouth = self.ema_mouth.update(f.mouth_open, dt)
        eye = self.ema_eye.update((f.eye_open_left + f.eye_open_right) / 2.0, dt)
        h_motion = self.ema_head_motion.update(head_motion, dt)
        facial = self.ema_facial.update(raw_facial, dt)
        center = self.ema_center.update(center_speed_norm, dt)
        vocal = self.ema_vocal.update(perception.audio.vocalization, dt)
        movement = clamp(max(facial, h_motion, center), 0.0, 1.0)

        orientation = pose.orientation if pose else HeadOrientation.UNKNOWN
        _, stable_orient = self.orient_gate.update(orientation, ts)

        self.features = TemporalFeatures(
            timestamp=ts, face_present=True,
            mouth_open=mouth, mouth_open_velocity=mouth_stats.velocity,
            eye_open=eye, head_yaw=yaw, head_pitch=pitch,
            head_motion=h_motion, facial_motion=facial, center_motion=center,
            movement=movement, face_scale=f.face_scale, vocalization=vocal,
            orientation=stable_orient,
        )

        # --- event detection (hysteresis + min duration) --------------
        new_events += self._detect_events(ts)
        self._prev_features = feat_d
        return self.features, new_events

    # ------------------------------------------------------------------
    def _detect_events(self, ts: float) -> list[Event]:
        out: list[Event] = []
        f = self.features

        if self.mouth_hyst.update(f.mouth_open):
            if self._mouth_since is None:
                self._mouth_since = ts
            elif ts - self._mouth_since >= self.min_duration_s and self._last_mouth != "open":
                self._last_mouth = "open"
                out.append(self._emit(EventType.MOUTH_OPENED, ts,
                                      value=round(f.mouth_open, 3),
                                      velocity=round(f.mouth_open_velocity, 3)))
        else:
            if self._mouth_since is not None and self._last_mouth == "open":
                self._last_mouth = "closed"
                out.append(self._emit(EventType.MOUTH_CLOSED, ts, value=round(f.mouth_open, 3)))
            self._mouth_since = None

        if self.facial_hyst.update(f.facial_motion):
            if self._facial_since is None:
                self._facial_since = ts
            elif ts - self._facial_since >= self.min_duration_s and not self._last_facial:
                self._last_facial = True
                out.append(self._emit(EventType.FACIAL_MOVEMENT, ts,
                                      value=round(f.facial_motion, 3)))
        else:
            self._facial_since = None
            self._last_facial = False

        if self.head_hyst.update(f.head_motion):
            if self._head_since is None:
                self._head_since = ts
            elif ts - self._head_since >= self.min_duration_s and not self._last_head:
                self._last_head = True
                out.append(self._emit(EventType.HEAD_MOVED, ts,
                                      yaw=round(f.head_yaw, 2), pitch=round(f.head_pitch, 2),
                                      motion=round(f.head_motion, 3)))
        else:
            self._head_since = None
            self._last_head = False

        if self._last_orientation is None:
            self._last_orientation = f.orientation
        elif f.orientation != self._last_orientation and f.orientation != HeadOrientation.UNKNOWN:
            out.append(self._emit(EventType.ORIENTATION_CHANGED, ts,
                                  from_orientation=self._last_orientation.value,
                                  to_orientation=f.orientation.value))
            self._last_orientation = f.orientation

        if self.vocal_hyst.update(f.vocalization):
            if self._vocal_since is None:
                self._vocal_since = ts
            elif ts - self._vocal_since >= self.min_duration_s and not self._last_vocal:
                self._last_vocal = True
                out.append(self._emit(EventType.VOCALIZATION_STARTED, ts,
                                      value=round(f.vocalization, 3)))
        else:
            if self._last_vocal:
                self._last_vocal = False
                out.append(self._emit(EventType.VOCALIZATION_STOPPED, ts))
            self._vocal_since = None

        # attention threshold crossings are detected in note_attention()

        return out

    def note_attention(self, value: float, now: float) -> list[Event]:
        """Feed the attention estimate and detect sustained band crossings."""
        self.win_attention.append(now, value)
        if not self._face_present:
            return []
        out: list[Event] = []
        if self._attention_state is None:
            self._attention_state = value >= self.settings.attention.engage_threshold
            return out
        if not self._attention_state and self.attention_high_hyst.update(value):
            self._attention_state = True
            out.append(self._emit(EventType.ATTENTION_INCREASED, now, value=round(value, 3)))
        elif self._attention_state and not self.attention_low_hyst.update(value):
            self._attention_state = False
            out.append(self._emit(EventType.ATTENTION_DECREASED, now, value=round(value, 3)))
        return out

    def _reset_tracking(self) -> None:
        self._prev_features = None
        self._prev_center = None
        self._mouth_since = None
        self._vocal_since = None
        self._facial_since = None
        self._head_since = None
        self._last_mouth = "closed"
        self._last_facial = False
        self._last_head = False
        self._last_vocal = False
        self.mouth_hyst.reset()
        self.facial_hyst.reset()
        self.head_hyst.reset()
        self.vocal_hyst.reset()

    _last_mouth = "closed"
    _last_facial = False
    _last_head = False
    _last_vocal = False
    _last_orientation: Optional[HeadOrientation] = None

    def _emit(self, etype: EventType, ts: float, **payload) -> Event:
        ev = Event(event=etype, timestamp=ts, session_id=self.session_id, payload=payload)
        self.events.append(ev)
        return ev
