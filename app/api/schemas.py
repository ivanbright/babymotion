"""Pydantic schemas for the HTTP/WebSocket boundary.

This API is the contract for a future frontend: it never exposes MediaPipe,
OpenCV, or Python internals - only plain JSON state, actions, and metrics.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    session_id: Optional[str] = None
    running: bool = False
    fps: float = 0.0


class StateResponse(BaseModel):
    timestamp: float
    session_id: Optional[str] = None
    state: dict[str, Any]
    temporal: dict[str, Any] = Field(default_factory=dict)
    attention: dict[str, Any]
    head_pose: Optional[dict[str, Any]] = None
    twin: dict[str, Any]
    twin_action: dict[str, Any]
    activity: Optional[dict[str, Any]] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    performance: dict[str, Any] = Field(default_factory=dict)
    recent_events: list[dict[str, Any]] = Field(default_factory=list)


class SessionInfo(BaseModel):
    session_id: Optional[str] = None
    running: bool = False
    duration_seconds: float = 0.0
    activity_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class StartSessionRequest(BaseModel):
    session_id: Optional[str] = None
    activity_id: Optional[str] = None


class StopSessionResponse(BaseModel):
    session: dict[str, Any]
    saved_to: Optional[str] = None
    event_count: int = 0


class ActivityInfo(BaseModel):
    id: str
    name: str
    objective: str
    duration: float
    termination_conditions: list[str]
    stimulus_count: int = 0


class ActivityActionResponse(BaseModel):
    started: bool
    activity_id: str
    detail: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str


class WsStateMessage(BaseModel):
    type: str = "state"
    data: dict[str, Any]
