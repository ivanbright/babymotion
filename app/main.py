"""BabyMotion engine: the closed loop.

CAMERA/AUDIO -> PERCEPTION -> TEMPORAL STATE -> INTERACTION POLICY ->
DIGITAL TWIN RESPONSE -> BABY RESPONSE -> MEASUREMENT -> STATE UPDATE.

``process_perception()`` is the single loop step; live camera and session
replay both go through it, so replay reproduces live behavior exactly.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from .config.settings import Settings, resolve_output_dir
from .input.camera import Camera
from .input.frame import Frame
from .input.microphone import Microphone
from .interaction.activities import ActivityInstance, Stimulus, build_registry
from .interaction.actions import TwinAction, TwinActionType
from .interaction.feedback import ResponseMonitor, ResponseRecord
from .interaction.policy import InteractionPolicy, create_policy
from .interaction.scheduler import ActivityScheduler
from .metrics.events import Event, EventLog, EventType
from .metrics.session import SessionRecorder
from .metrics import statistics as stats
from .perception.perception_result import PerceptionResult
from .perception.pipeline import PerceptionPipeline
from .state.baby_state import BabyState, BabyStateName
from .state.estimator import StateEstimator
from .storage.repositories import SessionRepository
from .temporal.sequence import TemporalEngine, TemporalFeatures
from .twin.controller import TwinController
from .twin.exaggeration import ExaggerationEngine
from .twin.twin_state import TwinState

FREE_STIMULUS_TYPES = {
    TwinActionType.EXAGGERATE, TwinActionType.SMILE, TwinActionType.OPEN_MOUTH,
    TwinActionType.ATTRACT_ATTENTION, TwinActionType.INTRODUCE_NOVELTY,
    TwinActionType.REPEAT,
}
FREE_STIMULUS_MIN_INTERVAL_S = 2.0
COOLDOWN_AFTER_LOW_S = 8.0
HISTORY_LIMIT = 400


class BabyMotionEngine:
    def __init__(self, settings: Settings, session_id: Optional[str] = None,
                 enable_perception: bool = True,
                 repository: Optional[SessionRepository] = None) -> None:
        self.settings = settings
        self.enable_perception = enable_perception
        self.repository = repository or SessionRepository(
            json_dir=resolve_output_dir(settings))
        self.event_log = EventLog()
        self.perception: Optional[PerceptionPipeline] = (
            PerceptionPipeline(settings) if enable_perception else None)
        self.camera: Optional[Camera] = None
        self.microphone: Optional[Microphone] = None

        self._fixed_session_id = session_id
        self.recorder: Optional[SessionRecorder] = None
        self.activity_starts: list[dict] = []
        self._build_components()

        self.snapshot: dict = self._empty_snapshot()
        self._last_ts: Optional[float] = None
        self._session_started_at: Optional[float] = None
        self._attention_series: list[tuple[float, float]] = []
        self._stimulus_queue: list[Stimulus] = []
        self._started_stimulus_ids: set[str] = set()
        self._last_free_stimulus_at: Optional[float] = None
        self._last_logged_action: Optional[tuple] = None
        self._low_attention_since: Optional[float] = None
        self._stimuli_count = 0
        self._fps = 0.0
        self._perception_latency_ms = 0.0
        self._decision_latency_ms = 0.0
        self.session_expired = False
        self.on_response: Optional[Callable[[ResponseRecord], None]] = None
        self.monitor.on_response = self._handle_response

    # -- component construction (also used to reset between sessions) -----
    def _build_components(self) -> None:
        sid = self._fixed_session_id
        settings = self.settings
        self.temporal = TemporalEngine(settings, self.event_log, sid)
        self.state_estimator = StateEstimator(settings, self.event_log, sid)
        self.policy: InteractionPolicy = create_policy(settings)
        self.exaggeration = ExaggerationEngine(settings.twin)
        self.twin = TwinController(self.exaggeration, settings.twin.response_speed_ms)
        self.registry = build_registry(settings)
        self.scheduler = ActivityScheduler(self.registry, self.event_log, sid)
        self.monitor = ResponseMonitor(settings, self.event_log, sid)
        self.monitor.on_response = self._handle_response

    # -- session lifecycle ------------------------------------------------
    def start_session(self, session_id: Optional[str] = None,
                      now: Optional[float] = None) -> str:
        now = now if now is not None else time.monotonic()
        self._fixed_session_id = session_id or self._fixed_session_id
        self._build_components()
        self.recorder = SessionRecorder(session_id=self._fixed_session_id)
        self._fixed_session_id = self.recorder.session_id
        self.event_log = EventLog()  # fresh log bound to this session
        # rebind log to components created above
        self.temporal.events = self.event_log
        self.state_estimator.events = self.event_log
        self.scheduler.events = self.event_log
        self.monitor.events = self.event_log

        self._session_started_at = now
        self._last_ts = None
        self._attention_series = []
        self._stimulus_queue = []
        self._started_stimulus_ids = set()
        self._last_free_stimulus_at = None
        self._last_logged_action = None
        self._low_attention_since = None
        self._stimuli_count = 0
        self.session_expired = False
        self.activity_starts = []
        self._emit(EventType.SESSION_STARTED, now,
                   camera=self.settings.camera.device,
                   activities=list(self.registry.keys()))
        return self.recorder.session_id

    def stop_session(self, now: Optional[float] = None) -> Optional[dict]:
        if self.recorder is None or self.recorder.ended:
            return None
        now = now if now is not None else (self._last_ts or time.monotonic())
        if self.scheduler.current is not None and self.scheduler.current.running:
            self.scheduler.finish(now, reason="session_ended", aborted=True)
        self.state_estimator.force_state(BabyStateName.SESSION_END, now)
        self._emit(EventType.SESSION_ENDED, now,
                   responses=self.monitor.response_count,
                   stimuli=self._stimuli_count)
        # every event in the log belongs to this session (components append
        # directly to event_log), so sync them here before finalizing
        self.recorder.events = self.event_log.events
        # duration is measured on the engine's monotonic timeline (wall clock
        # duration is meaningless when replaying faster than real time)
        if self._session_started_at is not None:
            self.recorder.start_time = time.time() - max(0.0, now - self._session_started_at)
        record = self.recorder.finalize(
            responses=self.monitor.records,
            stimuli=self._stimuli_count,
            attention_series=self._attention_series,
            sustained_attention_s=stats.sustained_attention(
                self._attention_series, self.settings.attention.engage_threshold),
            attention_threshold=self.settings.attention.engage_threshold,
        )
        payload = self.recorder.to_payload()
        # monotonic timeline bounds: replay needs them to reproduce the
        # session/activity boundaries exactly (wall clock is meaningless there)
        payload["session"]["start_timestamp"] = self._session_started_at
        payload["session"]["end_timestamp"] = now
        payload["activity_starts"] = self.activity_starts
        payload["settings"] = self.settings.model_dump()
        path = self.repository.save(payload)
        payload["saved_to"] = str(path) if path else None
        return payload

    @property
    def session_id(self) -> Optional[str]:
        return self.recorder.session_id if self.recorder else None

    def start_activity(self, activity_id: str,
                       now: Optional[float] = None) -> ActivityInstance:
        """Start an activity, record it, and remember its start timestamp
        (needed so replay reproduces the same schedule)."""
        now = now if now is not None else time.monotonic()
        if self.recorder is None or self.recorder.ended:
            self.start_session(now=now)
        assert self.recorder is not None
        instance = self.scheduler.start(activity_id, now=now)
        self.recorder.add_activity(activity_id)
        self.activity_starts.append({"activity_id": activity_id, "at": now})
        return instance

    # -- the loop ---------------------------------------------------------
    def process_frame(self, frame: Frame) -> tuple[PerceptionResult, dict]:
        """Live camera step: perception (incl. optional audio) -> loop."""
        if self.perception is None:
            raise RuntimeError("engine created with enable_perception=False")
        t0 = time.perf_counter()
        perception = self.perception.process(frame)
        self._perception_latency_ms = perception.latency_ms
        if self.microphone is not None and self.microphone.available:
            perception.audio = self.perception.audio.update(
                self.microphone.recent_rms(1.0))
        snap = self.process_perception(perception)
        snap["total_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        return perception, snap

    def process_perception(self, perception: PerceptionResult) -> dict:
        """One deterministic loop step. Used by live capture AND replay."""
        # Canonicalize: the loop consumes exactly the representation that is
        # recorded (rounded fields, no landmarks), so replaying the recorded
        # timeline through this same code reproduces identical outputs.
        perception = PerceptionResult.from_dict(perception.to_dict())
        now = perception.timestamp
        if self.recorder is None:
            self.start_session(now=now)
        assert self.recorder is not None
        if self._session_started_at is None:
            self._session_started_at = now

        dt = 1.0 / 30.0 if self._last_ts is None else max(0.0, now - self._last_ts)
        self._last_ts = now
        if dt > 0:
            self._fps = self._fps * 0.9 + (1.0 / dt) * 0.1 if self._fps else 1.0 / dt

        # 1. temporal features
        features, events = self.temporal.update(perception)

        # 2. attention estimate (on the just-updated features)
        detection_conf = perception.detection.confidence if perception.detection else 0.0
        attention = self.state_estimator.attention_estimator.estimate(
            features, detection_conf, now)
        events += self.temporal.note_attention(attention.value, now)

        # 3. baby state + state machine
        interaction_duration = max(0.0, now - self._session_started_at)
        activity_id = self.scheduler.current_id
        decision_t0 = time.perf_counter()
        baby, state_events = self.state_estimator.update(
            temporal=features,
            detection_confidence=detection_conf,
            interaction_duration=interaction_duration,
            response_latency=self.monitor.last_latency_s,
            activity_id=activity_id,
            now=now,
        )
        events += state_events
        self._attention_series.append((now, attention.value))

        # 4. activity scheduling (defer stimuli while face absent / low attention)
        face_ready = baby.face_present and attention.value >= self.settings.attention.disengage_threshold
        due = self.scheduler.tick(now, collect_stimuli=face_ready)
        if due:
            self._stimulus_queue.extend(due)
        scheduled = self._stimulus_queue.pop(0) if self._stimulus_queue else None
        low_term = self.scheduler.check_termination(
            now, baby.face_present,
            attention_low=attention.value <= self.settings.attention.disengage_threshold)
        if low_term:
            self.scheduler.finish(now, reason=low_term, aborted=True)

        # 5. policy decision
        history = self.event_log.events[-HISTORY_LIMIT:]
        action = self.policy.decide(baby, self.scheduler.current, history, now, scheduled)
        self._decision_latency_ms = (time.perf_counter() - decision_t0) * 1000.0

        # 6. stimulus bookkeeping + response measurement
        if self._maybe_start_stimulus(action, now, activity_id or ""):
            self._stimuli_count += 1
        events += self.monitor.tick(now, events, activity_id or "")

        # 7. twin update (exaggeration + smoothing)
        twin_state = self.twin.update(features, baby, action, dt)

        # 8. event logging for decisions/actions
        action_key = (action.type, action.target, round(action.intensity, 1))
        if action_key != self._last_logged_action:
            self._last_logged_action = action_key
            self._emit(EventType.TWIN_ACTION, now, **action.to_dict())
            self._emit(EventType.POLICY_DECISION, now,
                       decision=action.type.value, reason=action.reason)

        # 9. low-attention cooldown forcing
        self._update_cooldown(now, baby, attention.value)

        # 10. record the full-rate feature timeline (replay input)
        self.recorder.add_features({
            "timestamp": now,
            "perception": perception.to_dict(include_landmarks=False),
            "temporal": features.to_dict(),
            "baby_state": baby.to_dict(),
        })

        self.snapshot = self._build_snapshot(now, perception, features, baby,
                                             attention.value, attention.confidence,
                                             twin_state, action)
        # session duration limit
        limit = self.settings.session.default_duration_seconds
        if limit and interaction_duration >= limit:
            self.session_expired = True
        return self.snapshot

    # -- helpers ----------------------------------------------------------
    def _maybe_start_stimulus(self, action: TwinAction, now: float,
                              activity_id: str) -> bool:
        scheduled = bool(action.stimulus_id)
        is_free = action.type in FREE_STIMULUS_TYPES
        if not scheduled and not is_free:
            return False
        if scheduled:
            if action.stimulus_id in self._started_stimulus_ids:
                return False
            self._started_stimulus_ids.add(action.stimulus_id)
        else:
            pending = self.monitor.pending
            if pending is not None and now < pending.ends_at():
                return False
            if (self._last_free_stimulus_at is not None
                    and now - self._last_free_stimulus_at < FREE_STIMULUS_MIN_INTERVAL_S):
                return False
            if not action.stimulus_id:
                action.stimulus_id = (f"free_{action.type.value.lower()}"
                                      f"_{int(now * 1000)}")
            self._last_free_stimulus_at = now
        self.monitor.start_stimulus(action, now, activity_id)
        return True

    def _update_cooldown(self, now: float, baby: BabyState, attention: float) -> None:
        low = baby.state == BabyStateName.LOW_ENGAGEMENT and \
            attention <= self.settings.attention.disengage_threshold
        if low:
            if self._low_attention_since is None:
                self._low_attention_since = now
            elif now - self._low_attention_since >= COOLDOWN_AFTER_LOW_S:
                self.state_estimator.request_policy_state(BabyStateName.COOLDOWN)
        else:
            self._low_attention_since = None
            self.state_estimator.request_policy_state(None)

    def _handle_response(self, record: ResponseRecord) -> None:
        self.state_estimator.note_baby_response()
        if self.on_response is not None:
            self.on_response(record)

    def _emit(self, etype: EventType, ts: float, **payload: Any) -> Event:
        ev = Event(event=etype, timestamp=ts,
                   session_id=self.recorder.session_id if self.recorder else None,
                   payload=payload)
        self.event_log.append(ev)
        return ev

    def _build_snapshot(self, now: float, perception: PerceptionResult,
                        features: TemporalFeatures, baby: BabyState,
                        attention: float, attention_conf: float,
                        twin_state: TwinState, action: TwinAction) -> dict:
        current: Optional[ActivityInstance] = self.scheduler.current
        return {
            "timestamp": now,
            "session_id": self.recorder.session_id if self.recorder else None,
            "state": baby.to_dict(),
            "temporal": features.to_dict(),
            "attention": {"value": round(attention, 3),
                          "confidence": round(attention_conf, 3),
                          "evidence": baby.evidence},
            "head_pose": perception.head_pose.to_dict() if perception.head_pose else None,
            "twin": twin_state.to_dict(),
            "twin_action": action.to_dict(),
            "activity": current.to_dict(now) if current else None,
            "metrics": self.metrics(),
            "recent_events": [
                {"event": e.event.value, "timestamp": e.timestamp, "payload": e.payload}
                for e in self.event_log.recent(8)
            ],
            "performance": {
                "fps": round(self._fps, 1),
                "perception_latency_ms": round(self._perception_latency_ms, 1),
                "decision_latency_ms": round(self._decision_latency_ms, 1),
            },
        }

    def metrics(self) -> dict:
        latencies = [r.latency_s * 1000.0 for r in self.monitor.records]
        duration = 0.0
        if self._session_started_at is not None and self._last_ts is not None:
            duration = max(0.0, self._last_ts - self._session_started_at)
        return {
            "responses": self.monitor.response_count,
            "stimuli": self._stimuli_count,
            "response_rate": stats.response_rate(self.monitor.response_count,
                                                 self._stimuli_count),
            "median_response_latency_ms": stats.median_response_latency_ms(self.monitor.records),
            "last_response_latency_ms": round(latencies[-1], 1) if latencies else None,
            "mean_attention": stats.mean_attention(self._attention_series),
            "responses_per_minute": stats.responses_per_minute(
                self.monitor.response_count, duration),
            "events": len(self.event_log),
        }

    def _empty_snapshot(self) -> dict:
        return {"timestamp": 0.0, "state": BabyState().to_dict(),
                "attention": {"value": 0.0, "confidence": 0.0, "evidence": []},
                "twin": TwinState().to_dict(),
                "twin_action": TwinAction.idle().to_dict(),
                "activity": None, "metrics": {}, "performance": {}}

    # -- camera convenience ---------------------------------------------
    def open_camera(self) -> Camera:
        c = self.settings.camera
        self.camera = Camera(device=c.device, width=c.width, height=c.height,
                             target_fps=c.target_fps, backend=c.backend,
                             apply_quality=c.apply_quality, mirror=c.mirror)
        self.camera.open()
        return self.camera

    def open_microphone(self) -> Optional[Microphone]:
        m = self.settings.microphone
        if not m.enabled:
            return None
        self.microphone = Microphone(sample_rate=m.sample_rate, block_ms=m.block_ms,
                                     device=m.device)
        self.microphone.start()
        return self.microphone

    def close(self) -> None:
        if self.camera is not None:
            self.camera.release()
        if self.microphone is not None:
            self.microphone.stop()
        if self.perception is not None:
            self.perception.close()
