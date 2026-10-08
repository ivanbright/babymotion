"""Session replay: feed recorded structured signals through the engine with no
camera, and reproduce the same state/policy/response outputs.

Replay input is the feature timeline (perception + temporal + state), NOT
video - sessions record structured signals by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config.settings import Settings, load_settings
from .main import BabyMotionEngine
from .perception.perception_result import PerceptionResult
from .storage.repositories import SessionRepository

# events that depend on wall-clock session lifecycle and are not expected to
# match bit-for-bit between live and replay runs
_LIFECYCLE_EVENTS = {"SESSION_STARTED", "SESSION_ENDED"}


@dataclass
class ReplayResult:
    session_id: str
    frames: int
    events_replayed: list[dict]
    mismatches: list[str] = field(default_factory=list)
    payload: Optional[dict] = None

    @property
    def reproducible(self) -> bool:
        return not self.mismatches


def replay_session(payload: dict, settings: Optional[Settings] = None,
                   engine: Optional[BabyMotionEngine] = None) -> ReplayResult:
    features = payload.get("features", [])
    if not features:
        raise ValueError("session payload contains no feature timeline")
    settings = settings or load_settings()
    fresh = engine is None
    # never overwrite the source recording during replay
    engine = engine or BabyMotionEngine(settings, enable_perception=False,
                                        repository=SessionRepository())

    first_ts = features[0]["timestamp"]
    last_ts = features[-1]["timestamp"]
    session_record = payload.get("session", {})
    session_id = session_record.get("session_id", "replay")
    start_ts = session_record.get("start_timestamp", first_ts)
    end_ts = session_record.get("end_timestamp", last_ts + 0.25)
    engine.start_session(session_id=session_id, now=start_ts)

    activity_starts = sorted(payload.get("activity_starts", []),
                             key=lambda a: a["at"])
    queue = list(activity_starts)

    # activity starts that happened before the first recorded frame
    while queue and queue[0]["at"] < first_ts:
        item = queue.pop(0)
        engine.start_activity(item["activity_id"], now=item["at"])

    for entry in features:
        ts = entry["timestamp"]
        # start due activities at their recorded timestamp (not the frame
        # timestamp) so the event stream matches the recording exactly
        while queue and queue[0]["at"] <= ts:
            item = queue.pop(0)
            engine.start_activity(item["activity_id"], now=item["at"])
        perception = PerceptionResult.from_dict(entry["perception"])
        engine.process_perception(perception)

    while queue:
        item = queue.pop(0)
        engine.start_activity(item["activity_id"], now=item["at"])

    # make wall-clock duration match the recorded interaction length
    if engine.recorder is not None:
        import time as _time
        engine.recorder.start_time = _time.time() - max(0.0, end_ts - start_ts)
    final_payload = engine.stop_session(now=end_ts)
    if final_payload is None:
        raise RuntimeError("replay produced no session payload")

    replayed_events = final_payload.get("events", [])
    mismatches = compare_events(payload.get("events", []), replayed_events)
    result = ReplayResult(
        session_id=session_id,
        frames=len(features),
        events_replayed=replayed_events,
        mismatches=mismatches,
        payload=final_payload,
    )
    if fresh:
        engine.close()
    return result


def _event_keys(events: list[dict]) -> list[tuple]:
    return [
        (e["event"], round(float(e["timestamp"]), 3))
        for e in events
        if e["event"] not in _LIFECYCLE_EVENTS
    ]


def compare_events(recorded: list[dict], replayed: list[dict]) -> list[str]:
    """Return human-readable differences between two event streams."""
    a = _event_keys(recorded)
    b = _event_keys(replayed)
    problems: list[str] = []
    if a == b:
        return problems
    # report counts per event type first
    from collections import Counter
    ca, cb = Counter(e for e, _ in a), Counter(e for e, _ in b)
    for etype in sorted(set(ca) | set(cb)):
        if ca.get(etype, 0) != cb.get(etype, 0):
            problems.append(
                f"{etype}: recorded={ca.get(etype, 0)} replayed={cb.get(etype, 0)}")
    if not problems:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                problems.append(f"first divergence at #{i}: recorded={a[i]} replayed={b[i]}")
                break
        if len(a) != len(b):
            problems.append(f"event count: recorded={len(a)} replayed={len(b)}")
    return problems
