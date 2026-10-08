"""Replay a recorded session through the engine without a camera.

    python scripts/replay_session.py data/sessions/session_xxx.json
    python scripts/replay_session.py data/sessions/session_xxx.json --verify

--verify compares replayed events against the recorded events and reports
any divergence (reproducibility check).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from app.config.settings import load_settings
from app.debug_view import format_session_report
from app.replay import replay_session
from app.storage.repositories import load_session_json


def main() -> int:
    p = argparse.ArgumentParser(description="BabyMotion session replay")
    p.add_argument("path", help="path to session JSON")
    p.add_argument("--config", default=None, help="YAML config override")
    p.add_argument("--verify", action="store_true",
                   help="fail (exit 1) if replayed events diverge from recording")
    p.add_argument("--json", default=None, help="write replayed payload to this path")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"session file not found: {path}")
        return 2

    payload = load_session_json(path)
    settings = load_settings(args.config)
    result = replay_session(payload, settings=settings)

    print(f"replayed session {result.session_id}: "
          f"{result.frames} frames, {len(result.events_replayed)} events")
    if result.payload:
        print(format_session_report(result.payload))

    if result.mismatches:
        print("\nREPLAY DIVERGENCE (replay does not match recording):")
        for m in result.mismatches:
            print(f"  - {m}")
        return 1 if args.verify else 0
    print("\nreplay matches recorded events")

    if args.json and result.payload:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result.payload, fh, indent=1)
        print(f"replayed payload written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
