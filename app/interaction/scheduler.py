"""Activity scheduling: which activity runs, when stimuli fire, when it ends."""

from __future__ import annotations

from typing import Optional

from ..metrics.events import Event, EventLog, EventType
from .activities import Activity, ActivityInstance, ActivityPhase, Stimulus, build_registry


class ActivityScheduler:
    def __init__(self, registry: dict[str, Activity], event_log: EventLog,
                 session_id: Optional[str] = None) -> None:
        self.registry = registry
        self.events = event_log
        self.session_id = session_id
        self.current: Optional[ActivityInstance] = None
        self.completed: list[ActivityInstance] = []
        self._due: list[Stimulus] = []

    @property
    def current_id(self) -> Optional[str]:
        return self.current.id if self.current else None

    def start(self, activity_id: str, now: float,
              duration_override: float | None = None) -> ActivityInstance:
        if activity_id not in self.registry:
            raise KeyError(f"unknown activity '{activity_id}'; known: {sorted(self.registry)}")
        if self.current is not None and self.current.running:
            self.finish(now, reason="superseded", aborted=True)
        activity = self.registry[activity_id]
        if duration_override is not None:
            activity = Activity(**{**activity.__dict__, "duration": duration_override})
        self.current = ActivityInstance(activity, started_at=now)
        self._emit(EventType.ACTIVITY_STARTED, now, activity_id=activity_id,
                   objective=activity.objective, duration=activity.duration)
        return self.current

    def finish(self, now: float, reason: str, aborted: bool = False) -> None:
        if self.current is None or not self.current.running:
            return
        phase = ActivityPhase.ABORTED if aborted else ActivityPhase.COMPLETED
        self.current.terminate(now, reason, phase=phase)
        etype = EventType.ACTIVITY_ABORTED if aborted else EventType.ACTIVITY_COMPLETED
        self._emit(etype, now, activity_id=self.current.id, reason=reason,
                   fired=len(self.current.fired))
        self.completed.append(self.current)

    def tick(self, now: float, collect_stimuli: bool = True) -> list[Stimulus]:
        """Advance the activity timeline.

        collect_stimuli=False still advances duration/completion but does not
        consume due stimuli (used while the baby is absent/low attention so
        stimuli are deferred rather than lost).
        """
        self._due = []
        if self.current is None or not self.current.running:
            return []
        if collect_stimuli:
            self._due = self.current.due_stimuli(now)
        if self.current.is_duration_over(now):
            self.finish(now, reason="duration_elapsed")
        return self._due

    def check_termination(self, now: float, face_present: bool,
                          attention_low: bool) -> Optional[str]:
        if self.current is None or not self.current.running:
            return None
        conditions = set(self.current.activity.termination_conditions)
        if "no_face" in conditions and not face_present:
            return "no_face"
        if "low_attention" in conditions and attention_low:
            return "low_attention"
        return None

    def stop(self, now: float) -> None:
        self.finish(now, reason="session_ended", aborted=True)

    def _emit(self, etype: EventType, ts: float, **payload) -> Event:
        ev = Event(event=etype, timestamp=ts, session_id=self.session_id, payload=payload)
        self.events.append(ev)
        return ev
