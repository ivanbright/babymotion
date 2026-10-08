"""Perception-only camera test (no state/policy/twin):

    python scripts/run_camera.py --camera 0 --save-features out.json

Prints FPS, detection status, head pose, and landmark count. Press 'q' to quit.
"""

from __future__ import annotations

import argparse
import time

import _bootstrap  # noqa: F401  (caps BLAS threads before cv2/numpy load)

import cv2

from app.config.settings import load_settings
from app.input.camera import Camera, list_cameras


def main() -> int:
    p = argparse.ArgumentParser(description="BabyMotion perception smoke test")
    p.add_argument("--config", default=None)
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=0, help="0 = until q")
    p.add_argument("--list-cameras", action="store_true")
    args = p.parse_args()

    if args.list_cameras:
        print("cameras:", list_cameras())
        return 0

    settings = load_settings(args.config)
    if args.camera is not None:
        settings.camera.device = args.camera

    from app.perception.pipeline import PerceptionPipeline

    pipeline = PerceptionPipeline(settings)
    camera = Camera(device=settings.camera.device, width=settings.camera.width,
                    height=settings.camera.height,
                    target_fps=settings.camera.target_fps,
                    mirror=settings.camera.mirror)
    camera.open()
    print(f"camera: {camera.metadata}")

    frames = 0
    fps_ema = 0.0
    lat_ema = 0.0
    last = time.perf_counter()
    try:
        while True:
            frame = camera.read()
            result = pipeline.process(frame)
            now = time.perf_counter()
            dt = now - last
            last = now
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_ema = fps if frames == 0 else fps_ema * 0.9 + fps * 0.1
            lat_ema = result.latency_ms if frames == 0 else lat_ema * 0.9 + result.latency_ms * 0.1

            info = [f"FPS {fps_ema:5.1f}", f"perc {lat_ema:5.1f}ms",
                    f"face={'YES' if result.face_present else 'no '}",
                    f"det={result.detector}"]
            if result.head_pose is not None:
                info.append(f"yaw {result.head_pose.yaw:6.1f} pitch {result.head_pose.pitch:6.1f} "
                            f"roll {result.head_pose.roll:6.1f} conf {result.head_pose.confidence:.2f}")
            if result.landmarks is not None:
                info.append(f"lm={result.landmarks.count}")
            if result.features is not None:
                info.append(f"mouth {result.features.mouth_open:.2f} "
                            f"eyes {result.features.eye_open_left:.2f}/{result.features.eye_open_right:.2f}")
            print("\r" + "  ".join(info), end="", flush=True)

            if result.detection is not None:
                x, y, w, h = result.detection.bbox
                cv2.rectangle(frame.image, (x, y), (x + w, y + h), (80, 220, 120), 2)
            if result.landmarks is not None:
                for i in range(0, len(result.landmarks.pixels), 4):
                    px, py = result.landmarks.pixels[i]
                    cv2.circle(frame.image, (int(px), int(py)), 1, (0, 200, 255), -1)
            cv2.imshow("BabyMotion Perception", frame.image)
            frames += 1
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nframes={frames} avg_fps={fps_ema:.1f} avg_perception_ms={lat_ema:.1f}")
        camera.release()
        pipeline.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
