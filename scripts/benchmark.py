"""BabyMotion performance benchmark.

    python scripts/benchmark.py                 # engine + twin + policy + replay
    python scripts/benchmark.py --camera        # + live perception (camera needed)
    python scripts/benchmark.py --json out.json

Everything except --camera is hardware independent and runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from app.config.settings import load_settings  # noqa: E402
from app.interaction.activities import build_registry  # noqa: E402
from app.interaction.policy import RuleBasedPolicy  # noqa: E402
from app.main import BabyMotionEngine  # noqa: E402
from app.replay import replay_session  # noqa: E402
from app.storage.repositories import SessionRepository  # noqa: E402
from synthetic import SCENARIOS  # noqa: E402


def _stats(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)
    return {
        "n": len(ordered),
        "mean_ms": round(statistics.fmean(ordered), 3),
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 3),
        "max_ms": round(ordered[-1], 3),
    }


def bench_engine_loop(settings, frames: int = 600) -> dict:
    """Full deterministic loop step on a synthetic engaged timeline."""
    from synthetic import timeline_engaged_interaction

    engine = BabyMotionEngine(settings, enable_perception=False,
                              repository=SessionRepository())
    timeline = timeline_engaged_interaction(frames)
    engine.start_session(now=timeline[0].timestamp)
    engine.process_perception(timeline[0])  # warm-up (imports/first-touch)
    samples = []
    for perception in timeline[1:]:
        t0 = time.perf_counter()
        engine.process_perception(perception)
        samples.append((time.perf_counter() - t0) * 1000.0)
    engine.stop_session(now=timeline[-1].timestamp + 0.25)
    engine.close()
    stats = _stats(samples)
    stats["fps_equivalent"] = round(1000.0 / max(1e-6, stats["p50_ms"]), 1)
    return stats


def bench_twin_render(settings, frames: int = 600) -> dict:
    """Textured 3D mesh twin render (geometry + warp + shading)."""
    import numpy as np

    from synthetic import make_features, make_landmarks

    from app.twin.renderer import MeshTwinRenderer
    from app.twin.twin_state import TwinState

    renderer = MeshTwinRenderer(settings)
    h, w = settings.camera.height, settings.camera.width
    yy, xx = np.mgrid[0:h, 0:w]
    frame = np.stack([xx % 256, yy % 256, (xx + yy) % 256],
                     axis=-1).astype(np.uint8)

    samples = []
    warmup = 10
    for i in range(frames + warmup):
        landmarks = make_landmarks(w, h,
                                   mouth_open=0.1 + 0.4 * (0 if i % 40 < 20 else 1),
                                   seed=i % 5)
        # spaced so the render throttle never serves a cached panel
        landmarks.timestamp = i * 0.1
        features = make_features(mouth_open=0.3, timestamp=i / 30.0)
        state = TwinState(mouth_open=0.3, action="MIRROR",
                          action_intensity=0.5, timestamp=i / 30.0)
        t0 = time.perf_counter()
        renderer.render(frame, landmarks, features, state, facial_activity=0.1)
        if i >= warmup:
            samples.append((time.perf_counter() - t0) * 1000.0)
    stats = _stats(samples)
    stats["fps_equivalent"] = round(1000.0 / max(1e-6, stats["p50_ms"]), 1)
    return stats


def bench_policy(settings, iterations: int = 5000) -> dict:
    from app.state.baby_state import BabyState

    policy = RuleBasedPolicy(settings)
    state = BabyState(face_present=True, attention=0.8, attention_confidence=0.8,
                      facial_activity=0.6, movement=0.5)
    registry = build_registry(settings)
    from app.interaction.activities import ActivityInstance

    instance = ActivityInstance(registry["cause_effect"], started_at=0.0)
    history = []
    samples = []
    for i in range(iterations):
        t0 = time.perf_counter()
        policy.decide(state, instance, history, now=i / 30.0)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return _stats(samples)


def bench_replay(settings, frames: int = 600) -> dict:
    from synthetic import timeline_engaged_interaction

    engine = BabyMotionEngine(settings, enable_perception=False,
                              repository=SessionRepository())
    timeline = timeline_engaged_interaction(frames)
    engine.start_session(now=timeline[0].timestamp)
    for perception in timeline:
        engine.process_perception(perception)
    payload = engine.stop_session(now=timeline[-1].timestamp + 0.25)
    engine.close()

    t0 = time.perf_counter()
    result = replay_session(payload, settings=settings)
    elapsed = time.perf_counter() - t0
    return {
        "frames": result.frames,
        "seconds": round(elapsed, 3),
        "frames_per_second": round(result.frames / max(1e-6, elapsed), 1),
        "reproducible": result.reproducible,
    }


def bench_perception(settings, frames: int = 150) -> dict:
    """Live camera perception; skipped when no usable camera is present."""
    from app.input.camera import Camera

    camera = Camera(device=settings.camera.device, width=settings.camera.width,
                    height=settings.camera.height,
                    target_fps=settings.camera.target_fps,
                    backend=settings.camera.backend,
                    mirror=settings.camera.mirror)
    camera.open()
    pipeline = None
    try:
        from app.perception.pipeline import PerceptionPipeline

        pipeline = PerceptionPipeline(settings)
        camera.read()  # warm-up frame
        pipeline.process(camera.read())
        samples = []
        for _ in range(frames):
            frame = camera.read()
            t0 = time.perf_counter()
            pipeline.process(frame)
            samples.append((time.perf_counter() - t0) * 1000.0)
        stats = _stats(samples)
        stats["fps_equivalent"] = round(1000.0 / max(1e-6, stats["p50_ms"]), 1)
        return stats
    finally:
        if pipeline is not None:
            pipeline.close()
        camera.release()


def main() -> int:
    p = argparse.ArgumentParser(description="BabyMotion benchmark")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--camera", action="store_true",
                   help="also benchmark live camera perception")
    p.add_argument("--config", default=None)
    p.add_argument("--json", default=None, help="write results to this path")
    args = p.parse_args()

    settings = load_settings(args.config)
    results: dict[str, dict] = {}
    plan = [
        ("engine_loop", lambda: bench_engine_loop(settings, args.frames)),
        ("twin_render", lambda: bench_twin_render(settings, args.frames)),
        ("policy", lambda: bench_policy(settings)),
        ("replay", lambda: bench_replay(settings, args.frames)),
    ]
    if args.camera:
        plan.append(("perception_camera",
                     lambda: bench_perception(settings, max(50, args.frames // 4))))

    print(f"BabyMotion benchmark (frames={args.frames})")
    for name, fn in plan:
        try:
            results[name] = fn()
            print(f"  {name:<18} {json.dumps(results[name])}")
        except Exception as exc:  # camera/audio may be unavailable
            results[name] = {"error": str(exc)}
            print(f"  {name:<18} SKIPPED: {exc}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
