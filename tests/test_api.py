"""API tests: FastAPI endpoints + WebSocket state stream (no camera)."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from app.api.server import create_app  # noqa: E402


@pytest.fixture()
def client(engine_factory):
    engine = engine_factory()
    app = create_app(engine, runner=None)
    with TestClient(app) as c:
        yield c, engine


def test_health(client):
    c, engine = client
    r = c.get("/health")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "ok"
    assert d["running"] is False
    assert d["session_id"] is None


def test_state_returns_full_snapshot(client):
    c, engine = client
    d = c.get("/state").json()
    assert set(d) >= {"timestamp", "state", "attention", "twin", "twin_action",
                      "activity", "metrics", "performance"}
    assert d["state"]["face_present"] is False
    assert d["attention"]["value"] == 0.0
    assert "mouth_open" in d["twin"]


def test_session_lifecycle(client):
    c, engine = client
    # current session when idle
    idle = c.get("/session/current").json()
    assert idle["session_id"] is None
    assert idle["running"] is False

    started = c.post("/session/start").json()
    assert started["session_id"]
    assert started["running"] is True

    # second start conflicts
    conflict = c.post("/session/start")
    assert conflict.status_code == 409

    stopped = c.post("/session/stop")
    assert stopped.status_code == 200
    body = stopped.json()
    assert body["event_count"] >= 1
    assert body["session"]["session_id"] == started["session_id"]

    # stop with no active session conflicts
    assert c.post("/session/stop").status_code == 409


def test_activities_endpoint_lists_five(client):
    c, engine = client
    items = c.get("/activities").json()
    assert {a["id"] for a in items} == {"mirror", "exaggerated_mirror", "follow",
                                        "cause_effect", "repetition"}
    for a in items:
        assert a["name"] and a["objective"]
        assert a["duration"] > 0
        assert isinstance(a["stimulus_count"], int)


def test_start_unknown_activity_404(client):
    c, engine = client
    r = c.post("/activities/not_an_activity/start")
    assert r.status_code == 404
    assert "not_an_activity" in r.json()["detail"]


def test_start_activity_records_it(client):
    c, engine = client
    c.post("/session/start")
    r = c.post("/activities/follow/start")
    assert r.status_code == 200
    d = r.json()
    assert d["started"] is True and d["activity_id"] == "follow"
    assert d["detail"]["phase"] == "RUNNING"
    current = c.get("/session/current").json()
    assert "follow" in current["activity_ids"]
    c.post("/session/stop")


def test_websocket_streams_state(client):
    c, engine = client
    with c.websocket_connect("/ws/state") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "state"
    assert "state" in msg and "twin" in msg and "attention" in msg


def test_no_internal_types_leak(client):
    import json
    c, engine = client
    for path in ("/health", "/state", "/session/current", "/activities"):
        raw = c.get(path).content
        json.loads(raw)  # pure JSON, no numpy/enum reprs
        assert b"datetime." not in raw
        assert b"object at 0x" not in raw
