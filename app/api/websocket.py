"""WebSocket state stream: pushes the latest engine snapshot as JSON.

A future frontend connects to /ws/state and receives everything it needs to
render - it never needs to know what produced the data.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..main import BabyMotionEngine


def snapshot_message(engine: "BabyMotionEngine") -> dict[str, Any]:
    snap = dict(engine.snapshot)
    snap["type"] = "state"
    return snap


async def ws_state_endpoint(websocket, engine: "BabyMotionEngine", hz: float = 10.0) -> None:
    await websocket.accept()
    period = 1.0 / max(1.0, hz)
    try:
        while True:
            await websocket.send_json(snapshot_message(engine))
            await asyncio.sleep(period)
    except Exception:
        # client disconnected or server shutting down
        try:
            await websocket.close()
        except Exception:
            pass
