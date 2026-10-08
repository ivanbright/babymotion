"""Replay tests: recorded sessions reproduce the same event stream."""

import pytest

from app.replay import compare_events, replay_session
from synthetic import SCENARIOS, DT


def run_timeline(engine, timeline):
    """Record one session from a synthetic perception timeline."""
    engine.start_session(now=timeline[0].timestamp)
    for perception in timeline:
        engine.process_perception(perception)
    payload = engine.stop_session(now=timeline[-1].timestamp + 0.25)
    assert payload is not None
    return payload


def test_session_payload_is_complete(engine_factory):
    payload = run_timeline(engine_factory(), SCENARIOS["engaged_interaction"]())
    assert set(payload) >= {"session", "events", "features", "activity_starts",
                            "settings"}
    assert payload["saved_to"] is not None  # written to the tmp repo
    assert len(payload["features"]) == len(SCENARIOS["engaged_interaction"]())
    entry = payload["features"][0]
    assert set(entry) >= {"timestamp", "perception", "temporal", "baby_state"}
    assert payload["session"]["duration_seconds"] > 0
    # structured events exist
    assert len(payload["events"]) > 5
    assert all("event" in e and "timestamp" in e for e in payload["events"])
    assert "sustained_attention_seconds" in payload["session"]


def test_replay_reproduces_recorded_events(engine_factory):
    payload = run_timeline(engine_factory(), SCENARIOS["engaged_interaction"]())
    result = replay_session(payload)
    assert result.reproducible, result.mismatches
    assert result.frames == len(payload["features"])
    assert result.events_replayed


def test_replay_does_not_overwrite_source_session(engine_factory, tmp_path):
    payload = run_timeline(engine_factory(), SCENARIOS["head_turn"]())
    source = tmp_path / f"{payload['session']['session_id']}.json"
    assert source.exists()
    before = source.read_text(encoding="utf-8")

    result = replay_session(payload)
    assert result.reproducible, result.mismatches
    assert result.payload.get("saved_to") is None  # replay repository writes nothing

    after = source.read_text(encoding="utf-8")
    assert before == after


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_replays_identically(engine_factory, name):
    payload = run_timeline(engine_factory(), SCENARIOS[name]())
    result = replay_session(payload)
    assert result.reproducible, f"{name}: {result.mismatches}"


def test_replay_with_activity_schedule(engine_factory):
    engine = engine_factory()
    engine.start_session(now=0.0)
    engine.start_activity("mirror", now=0.0)
    from synthetic import timeline_stable_face
    timeline = timeline_stable_face(150)
    for p in timeline:
        engine.process_perception(p)
    payload = engine.stop_session(now=timeline[-1].timestamp + 0.25)
    assert payload["activity_starts"] == [{"activity_id": "mirror", "at": 0.0}]

    result = replay_session(payload)
    assert result.reproducible, result.mismatches
    # the replay actually started the activity
    replayed = [e for e in result.events_replayed
                if e["event"] == "ACTIVITY_STARTED"]
    assert len(replayed) == 1
    assert replayed[0]["timestamp"] == pytest.approx(0.0, abs=1e-6)


def test_compare_events_reports_divergence():
    a = [{"event": "MOUTH_OPENED", "timestamp": 1.0},
         {"event": "MOUTH_CLOSED", "timestamp": 2.0}]
    b = [{"event": "MOUTH_OPENED", "timestamp": 1.0},
         {"event": "HEAD_MOVED", "timestamp": 2.0}]
    assert compare_events(a, a) == []
    problems = compare_events(a, b)
    assert problems
    assert any("MOUTH_CLOSED" in p or "HEAD_MOVED" in p for p in problems)


def test_replay_without_features_raises(engine_factory):
    payload = run_timeline(engine_factory(), SCENARIOS["stable_face"]())
    payload["features"] = []
    with pytest.raises(ValueError):
        replay_session(payload)


def test_timeline_timestamps_are_monotonic():
    for name, factory in SCENARIOS.items():
        ts = [p.timestamp for p in factory(30)]
        assert ts == sorted(ts), name
        assert all(ts[i] > ts[i - 1] for i in range(1, len(ts))), name
    assert DT > 0
