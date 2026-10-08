"""BabyMotion V1 demo: camera -> perception -> state -> policy -> twin -> loop.

    python scripts/run_demo.py
    python scripts/run_demo.py --activity follow --camera 1
    python scripts/run_demo.py --api          # also serve REST/WebSocket
"""

from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401

from app.api.server import EngineRunner, run_server
from app.config.settings import load_settings
from app.debug_view import DebugView, format_console_report, format_session_report
from app.input.camera import list_cameras


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BabyMotion V1 camera demo")
    p.add_argument("--config", default=None, help="YAML config override")
    p.add_argument("--camera", type=int, default=None, help="camera device index")
    p.add_argument("--activity", default="exaggerated_mirror",
                   help="activity id to start (see --list-activities)")
    p.add_argument("--no-activity", action="store_true", help="do not auto-start an activity")
    p.add_argument("--duration", type=float, default=None, help="session duration seconds")
    p.add_argument("--no-window", action="store_true", help="disable debug window")
    p.add_argument("--api", action="store_true", help="serve REST + WebSocket API")
    p.add_argument("--api-port", type=int, default=None)
    p.add_argument("--list-activities", action="store_true")
    p.add_argument("--list-cameras", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    if args.camera is not None:
        settings.camera.device = args.camera
    if args.duration is not None:
        settings.session.default_duration_seconds = int(args.duration)

    if args.list_cameras:
        print("cameras:", list_cameras())
        return 0

    from app.main import BabyMotionEngine
    engine = BabyMotionEngine(settings)

    if args.list_activities:
        for a in engine.registry.values():
            print(f"  {a.id:20s} {a.name:20s} {a.duration:>3.0f}s  {a.objective}")
        engine.close()
        return 0

    if args.api:
        runner = EngineRunner(engine, show_window=not args.no_window)
        if not args.no_activity:
            # activity will be started once the runner session exists
            runner.start()
            time.sleep(1.0)
            try:
                engine.start_activity(args.activity)
            except Exception as exc:
                print(f"activity start failed: {exc}")
        else:
            runner.start()
        port = args.api_port or settings.api.port
        print(f"BabyMotion API on http://{settings.api.host}:{port}  (WS /ws/state)")
        try:
            run_server(engine, runner, host=settings.api.host, port=port)
        finally:
            runner.stop()
        _final_report(engine)
        return 0

    return run_interactive(engine, args)


def run_interactive(engine, args: argparse.Namespace) -> int:
    settings = engine.settings
    show_window = not args.no_window
    camera = engine.open_camera()
    engine.open_microphone()
    engine.start_session()
    view = DebugView(settings) if show_window else None
    activity_started = False
    session_started_at = time.monotonic()
    last_report = time.monotonic()

    print(f"BabyMotion V1 running  camera={camera.metadata.device} "
          f"{camera.metadata.width}x{camera.metadata.height} @ {camera.metadata.fps:.0f}fps")
    print("press 'q' in the video window to end the session\n")
    try:
        while True:
            frame = camera.read()
            perception, snapshot = engine.process_frame(frame)

            if (not args.no_activity and not activity_started
                    and snapshot["state"]["face_present"]):
                engine.start_activity(args.activity)
                activity_started = True
                print(f"activity started: {args.activity}")

            if view is not None:
                img = view.draw(frame.image, perception, snapshot)
                if not view.show(img):
                    break
            if engine.session_expired:
                print("session duration limit reached")
                break
            now = time.monotonic()
            if now - last_report >= 5.0:
                print(format_console_report(snapshot))
                last_report = now
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        payload = engine.stop_session()
        engine.close()
        if view is not None:
            view.close()
    if payload:
        print(format_session_report(payload))
    return 0


def _final_report(engine) -> None:
    payload = engine.stop_session()
    if payload:
        print(format_session_report(payload))


if __name__ == "__main__":
    raise SystemExit(main())
