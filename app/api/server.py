"""FastAPI boundary + background engine runner.

Endpoints (all JSON, no internal types leak):
  GET  /health
  GET  /state
  GET  /session/current
  POST /session/start
  POST /session/stop
  GET  /activities
  POST /activities/{id}/start
  WS   /ws/state
"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket
from pydantic import BaseModel

from ..config.settings import Settings
from ..debug_view import DebugView
from ..input.frame import Frame
from ..main import BabyMotionEngine
from . import schemas
from .websocket import ws_state_endpoint


class EngineRunner:
    """Runs the camera->perception->state loop in a background thread."""

    def __init__(self, engine: BabyMotionEngine, show_window: bool = False) -> None:
        self.engine = engine
        self.show_window = show_window
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        self.frames = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._loop, name="babymotion-loop", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        view: Optional[DebugView] = None
        try:
            camera = self.engine.open_camera()
            self.engine.open_microphone()
            if self.engine.recorder is None:
                self.engine.start_session()
            if self.show_window:
                view = DebugView(self.engine.settings)
            while not self._stop.is_set():
                frame = camera.read()
                perception, snap = self.engine.process_frame(frame)
                self.frames += 1
                if view is not None:
                    img = view.draw(frame.image, perception, snap)
                    if not view.show(img):
                        break
                if self.engine.session_expired:
                    break
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.error = f"{exc}\n{traceback.format_exc()}"
        finally:
            try:
                self.engine.stop_session()
            except Exception:
                pass
            self.engine.close()
            if view is not None:
                view.close()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)


def create_app(engine: BabyMotionEngine,
               runner: Optional[EngineRunner] = None) -> FastAPI:
    app = FastAPI(title="BabyMotion V1 API", version="0.1.0")
    app.state.engine = engine
    app.state.runner = runner

    def _require_runner() -> EngineRunner:
        if runner is None:
            raise HTTPException(status_code=409, detail="no engine runner attached")
        return runner

    @app.get("/health", response_model=schemas.HealthResponse)
    def health() -> schemas.HealthResponse:
        perf = engine.snapshot.get("performance", {})
        return schemas.HealthResponse(
            status="ok" if (runner is None or runner.error is None) else "error",
            session_id=engine.session_id,
            running=bool(runner and runner.running),
            fps=float(perf.get("fps", 0.0)),
        )

    @app.get("/state", response_model=schemas.StateResponse)
    def state() -> schemas.StateResponse:
        return schemas.StateResponse(**engine.snapshot)

    @app.get("/session/current", response_model=schemas.SessionInfo)
    def session_current() -> schemas.SessionInfo:
        started_at = engine._session_started_at
        now = engine._last_ts or time.monotonic()
        duration = max(0.0, now - started_at) if started_at else 0.0
        return schemas.SessionInfo(
            session_id=engine.session_id,
            running=engine.recorder is not None and not engine.recorder.ended,
            duration_seconds=duration,
            activity_ids=list(engine.recorder.activity_ids) if engine.recorder else [],
            metrics=engine.metrics(),
        )

    @app.post("/session/start", response_model=schemas.SessionInfo)
    def session_start(req: schemas.StartSessionRequest | None = None) -> schemas.SessionInfo:
        if engine.recorder is not None and not engine.recorder.ended:
            raise HTTPException(status_code=409, detail="session already running")
        sid = engine.start_session(session_id=req.session_id if req else None)
        if req and req.activity_id:
            _start_activity(req.activity_id)
        if runner is not None and not runner.running:
            runner.start()
        return schemas.SessionInfo(session_id=sid, running=True)

    @app.post("/session/stop", response_model=schemas.StopSessionResponse)
    def session_stop() -> schemas.StopSessionResponse:
        if runner is not None and runner.running:
            runner.stop()
        payload = engine.stop_session()
        if payload is None:
            raise HTTPException(status_code=409, detail="no active session")
        return schemas.StopSessionResponse(
            session=payload["session"],
            saved_to=payload.get("saved_to"),
            event_count=len(payload.get("events", [])),
        )

    @app.get("/activities", response_model=list[schemas.ActivityInfo])
    def activities() -> list[schemas.ActivityInfo]:
        return [
            schemas.ActivityInfo(
                id=a.id, name=a.name, objective=a.objective, duration=a.duration,
                termination_conditions=a.termination_conditions,
                stimulus_count=len(a.stimuli),
            )
            for a in engine.registry.values()
        ]

    @app.post("/activities/{activity_id}/start",
              response_model=schemas.ActivityActionResponse)
    def activity_start(activity_id: str) -> schemas.ActivityActionResponse:
        return _start_activity(activity_id)

    def _start_activity(activity_id: str) -> schemas.ActivityActionResponse:
        if activity_id not in engine.registry:
            raise HTTPException(status_code=404, detail=f"unknown activity '{activity_id}'")
        instance = engine.start_activity(activity_id, now=time.monotonic())
        return schemas.ActivityActionResponse(
            started=True, activity_id=activity_id, detail=instance.to_dict(time.monotonic()))

    @app.websocket("/ws/state")
    async def ws_state(websocket: WebSocket) -> None:
        await ws_state_endpoint(websocket, engine, hz=float(
            engine.settings.temporal.state_update_hz))

    return app


def run_server(engine: BabyMotionEngine, runner: Optional[EngineRunner] = None,
               host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    app = create_app(engine, runner)
    uvicorn.run(app, host=host, port=port, log_level="warning")
