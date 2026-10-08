"""Stimulus -> baby-response measurement.

After the policy issues a stimulus, the monitor watches for observable baby
events within a configurable window and records BABY_RESPONSE events with
latency. This is the "measure the response to our own stimulus" half of the
loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config.settings import Settings
from ..metrics.events import Event, EventLog, EventType
from .actions import TwinAction, TwinActionType

BABY_RESPONSE_EVENT_TYPES = {
    EventType.MOUTH_OPENED,
    EventType.MOUTH_CLOSED,
    EventType.HEAD_MOVED,
    EventType.ORIENTATION_CHANGED,
    EventType.ATTENTION_INCREASED,
    EventType.VOCALIZATION_STARTED,
    EventType.FACIAL_MOVEMENT,
}


@dataclass
class PendingStimulus:
    stimulus_id: str
    action: str
    started_at: float
    duration_ms: int
    responded: bool = False
    responses: list[dict] = field(default_factory=list)

    def ends_at(self) -> float:
        return self.started_at + self.duration_ms / 1000.0


@dataclass
class ResponseRecord:
    stimulus_id: str
    response_event: str
    latency_s: float
    timestamp: float
    activity_id: str = ""

    def to_dict(self) -> dict:
        return {
            "stimulus_id": self.stimulus_id,
            "response_event": self.response_event,
            "latency_s": round(self.latency_s, 3),
            "timestamp": self.timestamp,
            "activity_id": self.activity_id,
        }


class ResponseMonitor:
    def __init__(self, settings: Settings, event_log: EventLog,
                 session_id: Optional[str] = None) -> None:
        self.settings = settings
        self.events = event_log
        self.session_id = session_id
        self.pending: Optional[PendingStimulus] = None
        self.records: list[ResponseRecord] = []
        self._last_response_at: Optional[float] = None
        self.on_response = None  # optional callback(ResponseRecord)

    def start_stimulus(self, action: TwinAction, now: float,
                       activity_id: str = "") -> PendingStimulus:
        if self.pending is not None:
            # close the previous stimulus before starting a new one
            self._emit(EventType.STIMULUS_ENDED, now,
                       stimulus_id=self.pending.stimulus_id,
                       action=self.pending.action,
                       responded=self.pending.responded,
                       activity_id=activity_id,
                       superseded=True)
        stimulus_id = action.stimulus_id or f"stim_{int(now * 1000)}"
        pending = PendingStimulus(
            stimulus_id=stimulus_id,
            action=action.type.value,
            started_at=now,
            duration_ms=action.duration_ms or 1000,
        )
        self.pending = pending
        self._emit(EventType.STIMULUS_STARTED, now, stimulus_id=stimulus_id,
                   action=action.type.value, intensity=round(action.intensity, 3),
                   activity_id=activity_id, reason=action.reason)
        return pending

    def tick(self, now: float, frame_events: list[Event],
             activity_id: str = "") -> list[Event]:
        """Close expired stimuli and detect baby responses to the pending one."""
        out: list[Event] = []
        if self.pending is not None and now > self.pending.ends_at():
            self._emit(EventType.STIMULUS_ENDED, now,
                       stimulus_id=self.pending.stimulus_id,
                       action=self.pending.action,
                       responded=self.pending.responded,
                       activity_id=activity_id)
            self.pending = None

        if self.pending is None:
            return out
        window_s = self.settings.interaction.stimulus_window_ms / 1000.0
        gap_s = self.settings.interaction.response_event_gap_ms / 1000.0
        for ev in frame_events:
            if ev.event not in BABY_RESPONSE_EVENT_TYPES:
                continue
            latency = ev.timestamp - self.pending.started_at
            # a response must come strictly after the stimulus (the same frame
            # that triggered the stimulus cannot also be its own response)
            if latency <= 0 or latency > window_s:
                continue
            if self._last_response_at is not None and \
                    ev.timestamp - self._last_response_at < gap_s:
                continue
            self._last_response_at = ev.timestamp
            self.pending.responded = True
            record = ResponseRecord(
                stimulus_id=self.pending.stimulus_id,
                response_event=ev.event.value,
                latency_s=latency,
                timestamp=ev.timestamp,
                activity_id=activity_id,
            )
            self.records.append(record)
            self.pending.responses.append(record.to_dict())
            out.append(self._emit(
                EventType.BABY_RESPONSE, ev.timestamp,
                stimulus_id=record.stimulus_id,
                response=record.response_event,
                latency_ms=round(latency * 1000.0, 1),
                activity_id=activity_id,
            ))
            if self.on_response is not None:
                self.on_response(record)
        return out

    @property
    def response_count(self) -> int:
        return len(self.records)

    @property
    def last_latency_s(self) -> Optional[float]:
        return self.records[-1].latency_s if self.records else None

    def _emit(self, etype: EventType, ts: float, **payload) -> Event:
        ev = Event(event=etype, timestamp=ts, session_id=self.session_id, payload=payload)
        self.events.append(ev)
        return ev
