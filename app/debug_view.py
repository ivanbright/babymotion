"""Developer visualization: an engineering instrument, not a product UI.

Overlays on the camera feed: face box, landmarks, head pose, state,
attention, confidence, activity, twin action, FPS, latency, event stream,
plus the rendered digital twin.

The overlay is drawn at screen resolution (upscaled from the camera frame)
so the window can run fullscreen without unreadable text.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np

from .config.settings import Settings
from .metrics.events import Event
from .perception.perception_result import PerceptionResult
from .twin.renderer import MeshTwinRenderer
from .twin.twin_state import TwinState

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_MAX_UI_SCALE = 2.5


def _screen_size() -> tuple[int, int]:
    """Primary-monitor size in pixels (Windows; safe fallback)."""
    try:
        import ctypes
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        w = int(user32.GetSystemMetrics(0))
        h = int(user32.GetSystemMetrics(1))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1920, 1080


class DebugView:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.window_name = settings.debug.window_name
        self.renderer = MeshTwinRenderer(settings)
        self.events: deque[str] = deque(maxlen=8)
        self._last_event_key: Optional[tuple] = None
        self._visible = True
        self._window_ready = False

    def _ui_scale(self, h: int, w: int) -> float:
        if not self.settings.debug.fullscreen:
            return 1.0
        sw, sh = _screen_size()
        s = min(sw / w, sh / h)
        return float(max(1.0, min(s, _MAX_UI_SCALE)))

    def draw(self, frame: np.ndarray, perception: Optional[PerceptionResult],
             snapshot: dict) -> np.ndarray:
        s = self._ui_scale(frame.shape[0], frame.shape[1])
        if s != 1.0:
            img = cv2.resize(frame, None, fx=s, fy=s,
                             interpolation=cv2.INTER_CUBIC)
        else:
            img = frame.copy()
        h, w = img.shape[:2]

        def th(v: float) -> int:
            return max(1, int(round(v * s)))

        def px(v: float) -> int:
            return int(round(v * s))

        # face box + landmarks
        if perception is not None and perception.detection is not None:
            x, y, bw, bh = perception.detection.bbox
            x, y, bw, bh = px(x), px(y), px(bw), px(bh)
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (80, 220, 120), th(2))
            cv2.putText(img, f"conf {perception.detection.confidence:.2f}",
                        (x, max(px(16), y - px(6))), _FONT, 0.5 * s,
                        (80, 220, 120), th(1), cv2.LINE_AA)
        if (perception is not None and perception.landmarks is not None
                and self.settings.debug.draw_landmarks):
            pix = perception.landmarks.pixels
            r = th(1.6)
            for i in range(0, len(pix), 3):
                cv2.circle(img, (px(pix[i, 0]), px(pix[i, 1])), r,
                           (0, 200, 255), -1)

        # twin avatar (top-right): textured 3D mesh of the baby's own face
        twin_d = snapshot.get("twin", {})
        twin_state = TwinState.from_dict(twin_d) if twin_d else TwinState()
        facial_activity = float(snapshot.get("state", {}).get("facial_activity", 0.0))
        landmarks = perception.landmarks if perception is not None else None
        features = perception.features if perception is not None else None
        panel = self.renderer.render(frame, landmarks, features, twin_state,
                                     facial_activity)
        if s != 1.0:
            panel = cv2.resize(panel, (px(panel.shape[1]), px(panel.shape[0])),
                               interpolation=cv2.INTER_CUBIC)
        ph, pw = panel.shape[:2]
        x0, y0 = w - pw - px(10), px(10)
        if x0 >= 0 and y0 + ph <= h:
            img[y0:y0 + ph, x0:x0 + pw] = panel
            cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (200, 200, 200),
                          th(1))
            cv2.putText(img, "TWIN", (x0 + px(8), y0 + px(22)), _FONT, 0.55 * s,
                        (240, 240, 240), th(1), cv2.LINE_AA)

        # HUD panel (bottom-left)
        # event stream (dedup: only append when the newest event changes)
        for ev in snapshot.get("recent_events", []):
            label = ev.get("event", "")
            detail = ev.get("payload", {})
            if "state" in detail:
                label += f" -> {detail['state']}"
            elif "stimulus_id" in detail:
                label += f" {detail['stimulus_id']}"
            key = (ev.get("timestamp"), label)
            if key != self._last_event_key:
                self._last_event_key = key
                self.events.append(label)
        lines = self._hud_lines(snapshot)
        line_h = px(22)
        panel_h = line_h * len(lines) + px(12)
        overlay = img.copy()
        cv2.rectangle(overlay, (px(8), h - panel_h - px(8)),
                      (px(430), h - px(8)), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        yy = h - panel_h + px(8)
        for line in lines:
            cv2.putText(img, line, (px(16), yy), _FONT, 0.52 * s,
                        (240, 240, 240), th(1), cv2.LINE_AA)
            yy += line_h

        # event stream (top-left)
        yy = px(24)
        cv2.putText(img, "EVENTS", (px(12), yy), _FONT, 0.55 * s,
                    (120, 220, 255), th(1), cv2.LINE_AA)
        yy += px(20)
        for line in self.events:
            cv2.putText(img, line, (px(12), yy), _FONT, 0.48 * s,
                        (200, 200, 200), th(1), cv2.LINE_AA)
            yy += px(18)
        return img

    def _hud_lines(self, snapshot: dict) -> list[str]:
        state = snapshot.get("state", {})
        att = snapshot.get("attention", {})
        twin = snapshot.get("twin", {})
        action = snapshot.get("twin_action", {})
        perf = snapshot.get("performance", {})
        metrics = snapshot.get("metrics", {})
        activity = snapshot.get("activity")
        pose = snapshot.get("head_pose") or {}
        lines = [
            "BABYMOTION DEBUG",
            f"State: {state.get('state', '-')}  "
            f"({state.get('state_duration', 0):.1f}s)",
            f"Attention: {att.get('value', 0):.2f}  "
            f"Conf: {att.get('confidence', 0):.2f}",
            f"Head yaw {pose.get('yaw', 0):.1f}  pitch {pose.get('pitch', 0):.1f}  "
            f"orient {state.get('current_orientation', '-')}",
            f"Facial motion: {state.get('facial_activity', 0):.2f}  "
            f"movement: {state.get('movement', 0):.2f}  "
            f"mouth: {state.get('mouth_open', 0):.2f}",
            f"Twin: {action.get('action', '-')} "
            f"tgt={action.get('target', '-')} I={action.get('intensity', 0):.2f} "
            f"| open={twin.get('mouth_open', 0):.2f} smile={twin.get('smile', 0):.2f}",
            f"Activity: {activity['id'] if activity else '-'} "
            f"({activity['progress'] * 100:.0f}%)" if activity else
            "Activity: none",
            f"FPS: {perf.get('fps', 0):.1f}  perc {perf.get('perception_latency_ms', 0):.0f}ms  "
            f"dec {perf.get('decision_latency_ms', 0):.1f}ms",
            f"Responses: {metrics.get('responses', 0)}/{metrics.get('stimuli', 0)}  "
            f"mean att {metrics.get('mean_attention') or 0:.2f}",
        ]
        return lines

    def show(self, image: np.ndarray) -> bool:
        """Display and return False if the user requested quit (q/ESC)."""
        try:
            if not self._window_ready:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                if self.settings.debug.fullscreen:
                    cv2.setWindowProperty(
                        self.window_name, cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN)
                self._window_ready = True
            cv2.imshow(self.window_name, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return False
        except cv2.error:
            self._visible = False
        return True

    @staticmethod
    def close() -> None:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def format_console_report(snapshot: dict) -> str:
    """The demo console block from the spec (section 38)."""
    state = snapshot.get("state", {})
    att = snapshot.get("attention", {})
    action = snapshot.get("twin_action", {})
    metrics = snapshot.get("metrics", {})
    activity = snapshot.get("activity")
    latency = metrics.get("last_response_latency_ms")
    detected = "YES" if metrics.get("responses", 0) > 0 else "NO"
    return f"""
BABYMOTION

Interaction:
{activity['name'].upper() if activity else 'NONE'}

Baby:
attention       {att.get('value', 0):.2f}
facial_motion   {state.get('facial_activity', 0):.2f}
head_motion     {state.get('head_motion', 0):.2f}

Twin:
action          {action.get('action', '-')}
target          {action.get('target', '-')}

Response:
detected        {detected}
latency         {f"{latency:.0f}ms" if latency is not None else 'n/a'}
"""


def format_session_report(payload: dict) -> str:
    """The end-of-session console block from the spec (section 38)."""
    record = payload["session"]
    latency = record.get("median_response_latency_ms")
    rate = record.get("response_rate")
    return f"""
SESSION REPORT

Duration: {record.get('duration_seconds', 0):.0f} sec
Stimuli: {record.get('stimuli', 0)}
Responses: {record.get('responses', 0)}
Response rate: {f"{rate * 100:.0f}%" if rate is not None else 'n/a'}
Median response latency: {f"{latency / 1000:.2f} sec" if latency is not None else 'n/a'}
Mean attention estimate: {record.get('mean_attention') or 0:.2f}

Data saved:
{payload.get('saved_to', '(not saved)')}
"""
