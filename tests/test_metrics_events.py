"""Event log, statistics, and response-monitor tests."""

import json

import pytest

from app.config.settings import load_settings
from app.interaction.actions import TwinAction, TwinActionType
from app.interaction.feedback import ResponseMonitor
from app.metrics import statistics as stats
from app.metrics.events import Event, EventLog, EventType


# -- event model ----------------------------------------------------------
def test_event_json_roundtrip():
    ev = Event(event=EventType.MOUTH_OPENED, timestamp=1.25,
               session_id="s1", payload={"open": 0.7})
    d = json.loads(ev.to_json())
    assert d["event"] == "MOUTH_OPENED"
    assert d["timestamp"] == 1.25
    back = Event.from_dict(d)
    assert back.event == EventType.MOUTH_OPENED
    assert back.payload == {"open": 0.7}


def test_event_log_queries():
    log = EventLog()
    log.emit(EventType.FACE_DETECTED, 1.0, session_id="s")
    log.emit(EventType.MOUTH_OPENED, 2.0, open_ratio=0.5)
    log.emit(EventType.MOUTH_CLOSED, 3.0)
    assert len(log) == 3
    assert len(log.of_type(EventType.MOUTH_OPENED)) == 1
    assert len(log.since(2.0)) == 2
    assert log.recent(2)[-1].event == EventType.MOUTH_CLOSED
    assert len(log.events) == 3  # returns a copy


def test_event_log_bounded():
    log = EventLog(max_events=10)
    for i in range(25):
        log.emit(EventType.SYSTEM, float(i))
    assert len(log) == 10
    assert log.events[0].timestamp == 15.0


def test_event_log_jsonl_roundtrip():
    log = EventLog()
    log.emit(EventType.FACE_DETECTED, 1.0)
    log.emit(EventType.STATE_CHANGED, 2.0, state="ENGAGED")
    lines = log.to_jsonl().splitlines()
    assert len(lines) == 2
    restored = EventLog.from_iterable(Event.from_dict(json.loads(l)) for l in lines)
    assert [e.event for e in restored.events] == [EventType.FACE_DETECTED,
                                                  EventType.STATE_CHANGED]
    assert restored.events[1].payload["state"] == "ENGAGED"


# -- statistics -----------------------------------------------------------
def test_response_rate_definition():
    assert stats.response_rate(0, 0) is None
    assert stats.response_rate(2, 4) == pytest.approx(0.5)
    assert stats.response_rate(5, 3) == 1.0  # never exceeds 1


def test_median_response_latency_ms():
    assert stats.median_response_latency_ms([]) is None
    got = stats.median_response_latency_ms([{"latency_s": 0.4},
                                            {"latency_s": 0.6}])
    assert got == pytest.approx(500.0)


def test_mean_attention_is_time_weighted():
    # 0.5 for 10 s then 1.0 for 10 s -> 0.75
    series = [(0.0, 0.5), (10.0, 0.5), (10.0001, 1.0), (20.0, 1.0)]
    assert stats.mean_attention(series) == pytest.approx(0.75, abs=0.01)
    assert stats.mean_attention([]) is None


def test_max_attention_and_stats():
    series = [(0.0, 0.2), (1.0, 0.9), (2.0, 0.5)]
    assert stats.max_attention(series) == pytest.approx(0.9)
    assert stats.max_attention([]) is None
    s = stats.attention_stats(series)
    assert s["max"] == 0.9 and s["samples"] == 3


def test_sustained_attention_counts_seconds_above_threshold():
    # 0.8 for 5 s, 0.2 for 5 s -> ~5 s sustained
    series = [(0.0, 0.8), (5.0, 0.8), (5.0, 0.2), (10.0, 0.2)]
    assert stats.sustained_attention(series, threshold=0.65) == pytest.approx(5.0)
    assert stats.sustained_attention([]) == 0.0


def test_responses_per_minute():
    assert stats.responses_per_minute(0, 0.0) is None
    assert stats.responses_per_minute(6, 120.0) == pytest.approx(3.0)


def test_novelty_and_imitation_ratios():
    assert stats.novelty_response_ratio([], [0.8]) is None
    assert stats.novelty_response_ratio([0.5, 0.5], [1.0, 1.0]) == pytest.approx(2.0)
    assert stats.novelty_response_ratio([0.0], [1.0]) is None
    assert stats.imitation_rate(3, 0) is None
    assert stats.imitation_rate(3, 4) == pytest.approx(0.75)


# -- response monitor -----------------------------------------------------
def make_stimulus(duration_ms=2000):
    return TwinAction(type=TwinActionType.EXAGGERATE, intensity=0.8,
                      duration_ms=duration_ms, stimulus_id="stim_001")


def test_baby_response_recorded_with_latency():
    settings = load_settings()
    log = EventLog()
    mon = ResponseMonitor(settings, log, session_id="s")
    mon.start_stimulus(make_stimulus(), now=10.0, activity_id="mirror")
    frame_events = [Event(event=EventType.MOUTH_OPENED, timestamp=10.4)]
    emitted = mon.tick(now=10.5, frame_events=frame_events, activity_id="mirror")
    assert mon.response_count == 1
    assert mon.last_latency_s == pytest.approx(0.4)
    assert emitted[0].event == EventType.BABY_RESPONSE
    assert emitted[0].payload["latency_ms"] == pytest.approx(400.0, abs=1.0)
    assert emitted[0].payload["stimulus_id"] == "stim_001"
    assert mon.pending.responded is True


def test_response_outside_window_ignored():
    settings = load_settings()
    mon = ResponseMonitor(settings, EventLog())
    mon.start_stimulus(make_stimulus(duration_ms=1000), now=10.0)
    window_s = settings.interaction.stimulus_window_ms / 1000.0
    late = now = 10.0 + 1000 / 1000.0 + window_s + 1.0  # stimulus over + window passed
    emitted = mon.tick(now=late, frame_events=[Event(event=EventType.MOUTH_OPENED,
                                                     timestamp=late)])
    assert mon.response_count == 0
    assert emitted == []
    assert mon.pending is None  # stimulus already expired


def test_stimulus_expiry_emits_ended():
    settings = load_settings()
    log = EventLog()
    mon = ResponseMonitor(settings, log)
    mon.start_stimulus(make_stimulus(duration_ms=1000), now=5.0)
    mon.tick(now=6.5, frame_events=[])
    assert mon.pending is None
    types = [e.event for e in log.events]
    assert types == [EventType.STIMULUS_STARTED, EventType.STIMULUS_ENDED]
    assert log.events[-1].payload["responded"] is False


def test_superseding_stimulus_closes_previous():
    settings = load_settings()
    log = EventLog()
    mon = ResponseMonitor(settings, log)
    mon.start_stimulus(make_stimulus(duration_ms=5000), now=5.0)
    mon.start_stimulus(make_stimulus(duration_ms=5000), now=6.0, activity_id="follow")
    types = [e.event for e in log.events]
    assert types == [EventType.STIMULUS_STARTED, EventType.STIMULUS_ENDED,
                     EventType.STIMULUS_STARTED]
    assert log.events[1].payload["superseded"] is True


def test_response_event_gap_debounce():
    settings = load_settings()
    mon = ResponseMonitor(settings, EventLog())
    mon.start_stimulus(make_stimulus(), now=10.0)
    gap = settings.interaction.response_event_gap_ms / 1000.0
    mon.tick(now=10.2, frame_events=[Event(event=EventType.MOUTH_OPENED,
                                           timestamp=10.2)])
    # second event inside the debounce gap: not counted
    mon.tick(now=10.25, frame_events=[Event(event=EventType.HEAD_MOVED,
                                            timestamp=10.25)])
    assert mon.response_count == 1
    # after the gap it counts again
    mon.tick(now=10.2 + gap + 0.05,
             frame_events=[Event(event=EventType.HEAD_MOVED,
                                 timestamp=10.2 + gap + 0.05)])
    assert mon.response_count == 2


def test_non_response_events_ignored():
    settings = load_settings()
    mon = ResponseMonitor(settings, EventLog())
    mon.start_stimulus(make_stimulus(), now=10.0)
    mon.tick(now=10.3, frame_events=[Event(event=EventType.FACE_DETECTED,
                                           timestamp=10.3)])
    assert mon.response_count == 0


def test_same_frame_event_is_not_its_own_response():
    # the frame that triggered the stimulus must not count as the response
    settings = load_settings()
    mon = ResponseMonitor(settings, EventLog())
    mon.start_stimulus(make_stimulus(), now=10.0)
    emitted = mon.tick(now=10.0, frame_events=[Event(event=EventType.MOUTH_OPENED,
                                                     timestamp=10.0)])
    assert mon.response_count == 0
    assert emitted == []
    # the next frame's event does count
    mon.tick(now=10.1, frame_events=[Event(event=EventType.MOUTH_OPENED,
                                           timestamp=10.1)])
    assert mon.response_count == 1
    assert mon.last_latency_s == pytest.approx(0.1)
