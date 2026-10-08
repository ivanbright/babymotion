"""Structured event model + in-memory event log.

Every meaningful occurrence in the loop becomes an Event. Events are JSON
serializable, persisted through storage repositories, and are the substrate
for metrics, replay, and policy history.
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional


class EventType(str, enum.Enum):
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_ENDED = "SESSION_ENDED"
    FACE_DETECTED = "FACE_DETECTED"
    FACE_LOST = "FACE_LOST"
    ATTENTION_CHANGED = "ATTENTION_CHANGED"
    STATE_CHANGED = "STATE_CHANGED"
    STIMULUS_STARTED = "STIMULUS_STARTED"
    STIMULUS_ENDED = "STIMULUS_ENDED"
    BABY_RESPONSE = "BABY_RESPONSE"
    TWIN_ACTION = "TWIN_ACTION"
    ACTIVITY_STARTED = "ACTIVITY_STARTED"
    ACTIVITY_COMPLETED = "ACTIVITY_COMPLETED"
    ACTIVITY_ABORTED = "ACTIVITY_ABORTED"
    # temporal-level events
    HEAD_MOVED = "HEAD_MOVED"
    MOUTH_OPENED = "MOUTH_OPENED"
    MOUTH_CLOSED = "MOUTH_CLOSED"
    FACIAL_MOVEMENT = "FACIAL_MOVEMENT"
    ORIENTATION_CHANGED = "ORIENTATION_CHANGED"
    VOCALIZATION_STARTED = "VOCALIZATION_STARTED"
    VOCALIZATION_STOPPED = "VOCALIZATION_STOPPED"
    ATTENTION_INCREASED = "ATTENTION_INCREASED"
    ATTENTION_DECREASED = "ATTENTION_DECREASED"
    POLICY_DECISION = "POLICY_DECISION"
    SYSTEM = "SYSTEM"


@dataclass
class Event:
    event: EventType
    timestamp: float  # monotonic (engine time); wall clock kept separately
    session_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    wall_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "wall_time": self.wall_time,
            "session_id": self.session_id,
            "event": self.event.value,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            event=EventType(d["event"]),
            timestamp=d["timestamp"],
            session_id=d.get("session_id"),
            payload=d.get("payload", {}),
            wall_time=d.get("wall_time", 0.0),
        )


class EventLog:
    """Bounded in-memory event log (append + query + serialize)."""

    def __init__(self, max_events: int = 100_000) -> None:
        self._events: list[Event] = []
        self._max = max_events

    def append(self, event: Event) -> Event:
        self._events.append(event)
        if len(self._events) > self._max:
            del self._events[: len(self._events) - self._max]
        return event

    def emit(self, event_type: EventType, timestamp: float, session_id: str | None = None,
             **payload: Any) -> Event:
        return self.append(Event(event=event_type, timestamp=timestamp,
                                 session_id=session_id, payload=payload))

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def of_type(self, *types: EventType) -> list[Event]:
        wanted = set(types)
        return [e for e in self._events if e.event in wanted]

    def since(self, timestamp: float) -> list[Event]:
        return [e for e in self._events if e.timestamp >= timestamp]

    def recent(self, n: int = 20) -> list[Event]:
        return self._events[-n:]

    def __len__(self) -> int:
        return len(self._events)

    def to_jsonl(self) -> str:
        return "\n".join(e.to_json() for e in self._events)

    @classmethod
    def from_iterable(cls, events: Iterable[Event]) -> "EventLog":
        log = cls()
        for e in events:
            log.append(e)
        return log
