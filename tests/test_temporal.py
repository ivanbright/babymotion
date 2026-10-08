"""Temporal processing tests: smoothing, windows, event detection."""

import pytest

from app.config.settings import load_settings
from app.metrics.events import EventLog, EventType
from app.temporal.sequence import TemporalEngine
from app.temporal.smoothing import EMASmoother, Hysteresis, MinDurationGate, clamp
from app.temporal.window import TimedWindow

from synthetic import (
    DT,
    timeline_head_turn,
    timeline_mouth_movement,
    timeline_no_face,
    timeline_noisy_landmarks,
    timeline_tracking_loss,
    SCENARIOS,
)


def make_engine(settings=None):
    settings = settings or load_settings()
    log = EventLog()
    return TemporalEngine(settings, log, "test_session"), log


def run(engine, timeline, attention=None):
    all_events = []
    features = None
    for i, p in enumerate(timeline):
        features, evs = engine.update(p)
        all_events += evs
        if attention is not None:
            all_events += engine.note_attention(attention, p.timestamp)
    return features, all_events


# -- primitives -----------------------------------------------------------
def test_clamp():
    assert clamp(1.5, 0.0, 1.0) == 1.0
    assert clamp(-0.5, 0.0, 1.0) == 0.0
    assert clamp(0.5, 0.0, 1.0) == 0.5


def test_ema_converges_toward_input():
    ema = EMASmoother(alpha=0.5)
    for _ in range(30):
        ema.update(10.0)
    assert abs(ema.value - 10.0) < 1e-6


def test_hysteresis_prevents_flicker():
    h = Hysteresis(on_threshold=0.65, off_threshold=0.35)
    assert h.update(0.70) is True
    # dips below on-threshold but above off-threshold: stays on
    assert h.update(0.50) is True
    assert h.update(0.30) is False
    # rises above 0.5 but not 0.65: stays off
    assert h.update(0.50) is False
    assert h.update(0.70) is True


def test_min_duration_gate_requires_hold():
    gate = MinDurationGate(min_duration_s=0.5)
    gate.update("A", 0.0)
    changed, stable = gate.update("B", 0.1)
    assert not changed and stable == "A"
    changed, stable = gate.update("B", 0.4)
    assert not changed
    changed, stable = gate.update("B", 0.7)
    assert changed and stable == "B"


def test_window_velocity_and_trend():
    w = TimedWindow(window_ms=5000)
    for i in range(30):
        w.append(i * DT, i * DT * 2.0)  # ramp at 2 units/second
    stats = w.stats()
    assert stats.velocity == pytest.approx(2.0, abs=0.2)
    assert stats.trend > 0
    assert stats.mean > 0


def test_window_trims_old_samples():
    w = TimedWindow(window_ms=1000)
    for i in range(0, 200):
        w.append(i * 0.1, float(i))
    # only the last 1 s should remain
    assert w.times()[0] >= 19.9 - 1.0 - 1e-6
    assert w.stats().count < 15


# -- event detection ------------------------------------------------------
def test_mouth_open_and_close_events():
    engine, log = make_engine()
    run(engine, timeline_mouth_movement())
    types = [e.event for e in log.events]
    assert EventType.MOUTH_OPENED in types
    assert EventType.MOUTH_CLOSED in types
    assert types.index(EventType.MOUTH_OPENED) < types.index(EventType.MOUTH_CLOSED)


def test_face_appear_and_lost_events():
    engine, log = make_engine()
    run(engine, timeline_no_face(30))
    run(engine, SCENARIOS["stable_face"]())  # continues timestamps
    types = [e.event for e in log.events]
    assert EventType.FACE_DETECTED in types


def test_face_loss_grace_period():
    engine, log = make_engine()
    timeline = timeline_tracking_loss()  # 10-frame gap < 700ms grace
    run(engine, timeline)
    types = [e.event for e in log.events]
    assert EventType.FACE_LOST not in types  # brief gap tolerated
    assert types.count(EventType.FACE_DETECTED) == 1


def test_head_turn_produces_head_moved_and_orientation_change():
    engine, log = make_engine()
    run(engine, timeline_head_turn())
    types = [e.event for e in log.events]
    assert EventType.HEAD_MOVED in types
    assert EventType.ORIENTATION_CHANGED in types


def test_noisy_landmarks_do_not_trigger_mouth_open():
    engine, log = make_engine()
    run(engine, timeline_noisy_landmarks())
    types = [e.event for e in log.events]
    assert EventType.MOUTH_OPENED not in types


def test_attention_events_fire_on_threshold_crossing():
    engine, log = make_engine()
    timeline = SCENARIOS["stable_face"]()
    # low attention first, then high
    for p in timeline[:40]:
        engine.update(p)
        engine.note_attention(0.2, p.timestamp)
    for p in timeline[40:]:
        engine.update(p)
        engine.note_attention(0.9, p.timestamp)
    types = [e.event for e in log.events]
    assert EventType.ATTENTION_INCREASED in types


def test_no_face_keeps_features_face_present_false():
    engine, _ = make_engine()
    features, _ = run(engine, timeline_no_face())
    assert features.face_present is False
