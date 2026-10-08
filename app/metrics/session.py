"""Session data model + recorder.

A session captures the structured interaction: events, feature timeline
(never video by default), responses, activities, and summary statistics.
"""

from __future__ import annotations

import platform
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ..metrics.events import Event, EventLog, EventType


@dataclass
class SessionRecord:
    session_id: str
    start_time: float  # wall clock
    end_time: Optional[float] = None
    duration: float = 0.0
    activity_ids: list[str] = field(default_factory=list)
    event_count: int = 0
    response_count: int = 0
    stimulus_count: int = 0
    mean_attention: Optional[float] = None
    max_attention: Optional[float] = None
    median_response_latency_ms: Optional[float] = None
    sustained_attention_s: Optional[float] = None
    responses_per_minute: Optional[float] = None
    response_rate: Optional[float] = None
    system: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": round(self.duration, 2),
            "activities": self.activity_ids,
            "event_count": self.event_count,
            "responses": self.response_count,
            "stimuli": self.stimulus_count,
            "mean_attention": round(self.mean_attention, 3) if self.mean_attention is not None else None,
            "max_attention": round(self.max_attention, 3) if self.max_attention is not None else None,
            "median_response_latency_ms": (round(self.median_response_latency_ms, 1)
                                           if self.median_response_latency_ms is not None else None),
            "sustained_attention_seconds": (round(self.sustained_attention_s, 2)
                                            if self.sustained_attention_s is not None else None),
            "responses_per_minute": (round(self.responses_per_minute, 2)
                                     if self.responses_per_minute is not None else None),
            "response_rate": (round(self.response_rate, 3)
                              if self.response_rate is not None else None),
            "system": self.system,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionRecord":
        return cls(
            session_id=d["session_id"],
            start_time=d["start_time"],
            end_time=d.get("end_time"),
            duration=d.get("duration_seconds", 0.0),
            activity_ids=list(d.get("activities", [])),
            event_count=d.get("event_count", 0),
            response_count=d.get("responses", 0),
            stimulus_count=d.get("stimuli", 0),
            mean_attention=d.get("mean_attention"),
            max_attention=d.get("max_attention"),
            median_response_latency_ms=d.get("median_response_latency_ms"),
            sustained_attention_s=d.get("sustained_attention_seconds"),
            responses_per_minute=d.get("responses_per_minute"),
            response_rate=d.get("response_rate"),
            system=d.get("system", {}),
        )


def system_metadata(extra: Optional[dict] = None) -> dict:
    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "babymotion_version": "0.1.0",
        "created_at": time.time(),
    }
    if extra:
        meta.update(extra)
    return meta


class SessionRecorder:
    """Accumulates everything for one session and finalizes a record."""

    def __init__(self, session_id: Optional[str] = None,
                 system: Optional[dict] = None) -> None:
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        self.start_time = time.time()
        self.started = True
        self.ended = False
        self.events: list[Event] = []
        self.features: list[dict] = []  # perception/temporal/state timeline
        self.activity_ids: list[str] = []
        self.system = system_metadata(system)
        self.record: Optional[SessionRecord] = None

    def add_event(self, event: Event) -> None:
        self.events.append(event)

    def add_features(self, entry: dict) -> None:
        self.features.append(entry)

    def add_activity(self, activity_id: str) -> None:
        if activity_id not in self.activity_ids:
            self.activity_ids.append(activity_id)

    def finalize(self, responses: list, stimuli: int,
                 attention_series: list[tuple[float, float]],
                 sustained_attention_s: float,
                 attention_threshold: float) -> SessionRecord:
        from .statistics import (
            mean_attention,
            max_attention,
            median_response_latency_ms,
            responses_per_minute,
            response_rate,
        )

        end_time = time.time()
        duration = max(0.0, end_time - self.start_time)
        record = SessionRecord(
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=end_time,
            duration=duration,
            activity_ids=list(self.activity_ids),
            event_count=len(self.events),
            response_count=len(responses),
            stimulus_count=stimuli,
            mean_attention=mean_attention(attention_series),
            max_attention=max_attention(attention_series),
            median_response_latency_ms=median_response_latency_ms(responses),
            sustained_attention_s=sustained_attention_s,
            responses_per_minute=responses_per_minute(len(responses), duration),
            response_rate=response_rate(len(responses), stimuli),
            system=self.system,
        )
        self.record = record
        self.ended = True
        return record

    def to_payload(self) -> dict:
        """Full session payload (record + events + feature timeline)."""
        record = self.record or SessionRecord(
            session_id=self.session_id, start_time=self.start_time,
            event_count=len(self.events), system=self.system)
        return {
            "session": record.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "features": self.features,
        }
