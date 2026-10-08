"""Repositories: persist/load session payloads (SQLite + JSON files).

JSON files under data/sessions/ are the canonical replay format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .database import Database


class SessionRepository:
    def __init__(self, database: Optional[Database] = None,
                 json_dir: Optional[Path] = None) -> None:
        self.database = database
        self.json_dir = json_dir
        if json_dir is not None:
            json_dir.mkdir(parents=True, exist_ok=True)

    # -- write ------------------------------------------------------------
    def save(self, payload: dict) -> Optional[Path]:
        """Persist a session payload; returns the JSON path if written."""
        record = payload["session"]
        sid = record["session_id"]
        path: Optional[Path] = None
        if self.json_dir is not None:
            path = self.json_dir / f"{sid}.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
        if self.database is not None:
            db = self.database
            db.execute(
                "INSERT OR REPLACE INTO sessions(session_id, start_time, duration, record_json) "
                "VALUES (?,?,?,?)",
                (sid, record.get("start_time", 0.0), record.get("duration_seconds", 0.0),
                 json.dumps(record)),
            )
            for ev in payload.get("events", []):
                db.execute(
                    "INSERT INTO events(session_id, timestamp, event, wall_time, payload_json) "
                    "VALUES (?,?,?,?,?)",
                    (sid, ev["timestamp"], ev["event"], ev.get("wall_time", 0.0),
                     json.dumps(ev.get("payload", {}))),
                )
            for feat in payload.get("features", []):
                db.execute(
                    "INSERT INTO features(session_id, timestamp, data_json) VALUES (?,?,?)",
                    (sid, feat.get("timestamp", 0.0), json.dumps(feat)),
                )
        return path

    # -- read -------------------------------------------------------------
    def list_sessions(self) -> list[dict]:
        if self.database is None:
            return []
        rows = self.database.query(
            "SELECT session_id, start_time, duration, record_json FROM sessions ORDER BY start_time")
        return [json.loads(r["record_json"]) for r in rows]

    def get(self, session_id: str) -> Optional[dict]:
        if self.database is not None:
            rows = self.database.query(
                "SELECT record_json FROM sessions WHERE session_id = ?", (session_id,))
            if rows:
                record = json.loads(rows[0]["record_json"])
                events = [dict(r) for r in self.database.query(
                    "SELECT timestamp, event, wall_time, payload_json FROM events "
                    "WHERE session_id = ? ORDER BY timestamp", (session_id,))]
                features = [json.loads(r["data_json"]) for r in self.database.query(
                    "SELECT data_json FROM features WHERE session_id = ? ORDER BY timestamp",
                    (session_id,))]
                return {
                    "session": record,
                    "events": [
                        {"timestamp": e["timestamp"], "event": e["event"],
                         "wall_time": e["wall_time"], "session_id": session_id,
                         "payload": json.loads(e["payload_json"])}
                        for e in events
                    ],
                    "features": features,
                }
        if self.json_dir is not None:
            return load_session_json(self.json_dir / f"{session_id}.json")
        return None


def save_session_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return path


def load_session_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
