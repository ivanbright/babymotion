# BabyMotion V1

An adaptive digital-twin interaction engine: a real-time closed loop that
perceives a baby's visible behavior, estimates engagement state, decides what
the twin should do, renders an exaggerated textured 3D mesh of the baby's own
face, measures the
baby's response to each stimulus, and records everything as structured,
replayable session data.

```
camera / audio
      |
      v
 perception  --->  temporal state  --->  interaction policy  --->  digital twin
 (face, pose,      (smoothing,          (activities,             (exaggerated
  features,         events,              stimuli,                 3D mesh of
  audio)            attention,           feedback)                the baby's
                     baby state)                                   own face)
      ^                                                          |
      |                  response measurement <------------------+
      |                  (stimulus -> latency -> events)
      +------------------------ session data (JSON + SQLite) <---+
```

## Principles

- **No identity recognition.** Faces are detected and measured, never
  identified or enrolled. No embeddings, no gallery, no match scores.
- **No unsupported claims.** Outputs are observable measurements
  (attention estimate, movement, response latency, response rate) with
  explicit definitions - never developmental scores or diagnoses.
- **Deterministic and replayable.** The whole loop consumes one
  `PerceptionResult` per step; recorded sessions replay through the same code
  and reproduce the identical event stream (`--verify` proves it).
- **Config-driven.** Every threshold, gain, schedule, and limit lives in
  `app/config/defaults.yaml` (overridable per run with `--config file.yaml`).

## Install

```bash
pip install -r requirements.txt      # or: pip install -e .[dev]
```

Requires Python >= 3.11 and a webcam. Audio (`sounddevice`) is optional; the
engine degrades gracefully without a microphone. The MediaPipe face landmark
model is auto-discovered (the repo's `models/face_landmarker.task` is used
when present).

## Quick start

```bash
# full demo: camera -> loop -> twin window -> session report
python scripts/run_demo.py --camera 1

# choose an activity and a duration
python scripts/run_demo.py --camera 1 --activity follow --duration 60

# headless perception smoke test
python scripts/run_camera.py --camera 1 --max-frames 120

# serve the REST + WebSocket API alongside the demo
python scripts/run_demo.py --camera 1 --api --api-port 8000

# replay a recorded session (no camera) and verify reproducibility
python scripts/replay_session.py data/sessions/<session_id>.json --verify

# tests + benchmark
python -m pytest
python scripts/benchmark.py --camera --json bench_results.json
```

Camera indices can be listed with `--list-cameras`; activities with
`--list-activities`.

## Modules

| Path | Responsibility |
| --- | --- |
| `app/input/` | camera frames (backend auto-fallback, mirrored selfie view by default), optional microphone |
| `app/perception/` | face detection, 478 landmarks, head pose, facial features, audio features |
| `app/temporal/` | EMA smoothing, timed windows, hysteresis, event detection (mouth/head/orientation/attention) |
| `app/state/` | attention estimation with confidence + evidence, baby state machine (dwell + hysteresis) |
| `app/interaction/` | deterministic activities (5 protocols), rule-based policy, stimulus/response feedback |
| `app/twin/` | exaggeration mapping, response-time smoothing, action overrides, textured 3D mesh renderer |
| `app/metrics/` | event model/log, session records, explicitly defined statistics |
| `app/storage/` | JSON session files + SQLite repository |
| `app/api/` | FastAPI endpoints + WebSocket state stream |
| `app/main.py` | the engine: one deterministic step per perception |
| `app/replay.py` | feed a recorded feature timeline back through the engine |
| `scripts/` | demo, camera smoke test, replay, benchmark |

### Activities

| id | protocol |
| --- | --- |
| `mirror` | twin mirrors the baby; periodic gentle prompts (baseline) |
| `exaggerated_mirror` | amplification bursts at fixed times |
| `follow` | deterministic alternating head turns; observes orientation following |
| `cause_effect` | baby mouth-open triggers a predictable twin response |
| `repetition` | identical stimulus repeated; measures response trend / habituation |

All schedules are fixed timelines (no randomness), so experiments are
reproducible.

## Session data

Each `POST /session/stop` (or demo exit) writes
`data/sessions/<session_id>.json` containing:

- `session` - record: duration, activities, event/response counts,
  attention statistics (mean/max/sustained), response rate, latencies
- `events` - full structured event log (`FACE_DETECTED`, `MOUTH_OPENED`,
  `STATE_CHANGED`, `STIMULUS_STARTED`, `BABY_RESPONSE`, `TWIN_ACTION`, ...)
- `features` - full-rate timeline: per-frame `perception` + `temporal` +
  `baby_state` (this is the replay input - structured signals, not video)
- `activity_starts` - exact activity start times for schedule reproduction
- `settings` - the config snapshot the session ran with

Replay never overwrites the source file.

## API

| endpoint | purpose |
| --- | --- |
| `GET /health` | status, session id, fps |
| `GET /state` | full snapshot: state, attention, twin, action, metrics |
| `GET /session/current` | running session info + metrics |
| `POST /session/start` / `POST /session/stop` | lifecycle |
| `GET /activities` / `POST /activities/{id}/start` | activity registry/control |
| `WS /ws/state` | JSON state stream (~`temporal.state_update_hz` Hz) |

All responses are plain JSON; no internal types leak.

## Configuration

`app/config/defaults.yaml` groups: `camera`, `microphone`, `perception`,
`temporal`, `attention`, `twin` (exaggeration gains + safe limits + neutral
reference), `interaction` (stimulus window, response gap), `activities`
(schedules), `session`, `recording`, `api`, `debug`. Override any subset:

```bash
python scripts/run_demo.py --config my.yaml
```

## Benchmark

`scripts/benchmark.py` reports p50/p95 latency for the engine loop, twin
render, policy, and replay throughput; with `--camera` it adds live
perception. Representative numbers on this machine (600 frames, CPU):
engine loop ~0.2-1 ms/frame, twin render ~11 ms p50 when idle (textured
mesh warp, displacement-field + vectorized remap; whole-process variance
of 25-40 ms appears under heavy machine load), policy ~1-4 us/decision,
replay >900 frames/s (reproducible), live perception ~65-78 ms/frame with
MediaPipe (the binding constraint at 720p).

## Acceptance criteria

`python scripts/run_demo.py` walks the full A-P checklist end to end
(camera loop, state transitions, activities, stimuli, responses, session
file, replay verification, API, metrics). `tests/` covers geometry, head
pose recovery, temporal events, state machine, policy determinism,
exaggeration, activities, metrics, replay, and the API.

## Limitations

- Attention is a **heuristic estimate from observable behavior**, not a
  ground-truth measure of internal state.
- No claims of developmental benefit: the system measures responses to its
  own stimuli; it does not assess development.
- Single face, single session; identity is intentionally out of scope.
- Camera performance depends on the machine; the demo degrades to a lower
  frame rate rather than dropping the loop.
