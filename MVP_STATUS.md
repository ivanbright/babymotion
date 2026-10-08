# BabyMotion V1 — MVP status

A complete, local, deterministic closed loop — camera → perception →
temporal state → policy → exaggerated digital twin → response measurement →
structured session data — with a live demo, REST/WebSocket API, replay,
benchmark, and 127 passing tests.

## What works today

**Perception** (`app/perception/`)
- MediaPipe face mesh (478 points), head pose (yaw/pitch/roll), expression
  features (mouth open, smile, eye scale), face confidence; ROI-tracked,
  cascade fallback when the mesh model is unavailable
- ~65–78 ms/frame @ 720p CPU — the binding constraint, honestly documented

**Temporal + attention** (`app/temporal/`, `app/attention/`)
- Short/long-window features: velocities, head/facial/center motion,
  orientation, optional vocalization energy from the mic
- Heuristic attention estimate + engagement state machine
  (engaged / transitional / disengaged) with hysteresis

**Interaction policy** (`app/interaction/`)
- 5 fixed-schedule activities — `mirror`, `exaggerated_mirror`, `follow`,
  `cause_effect`, `repetition` — all deterministic timelines (no
  randomness), stimulus→response windowing with per-response latencies

**Digital twin** (`app/twin/`)
- The baby's own face as a textured 3D mesh: adaptive neutral template
  (EMA at rest), Kabsch rigid fit, centroid-anchored head-rotation
  exaggeration (gain 1.3×, safe-limited), expression residual
  amplification (1.4–1.6×)
- Vectorized displacement-field renderer (Delaunay barycentric warp +
  Lambert shading): ~11 ms p50, target <33 ms
- Selfie-style camera mirroring (`camera.mirror: true`) so turning right
  matches what the twin shows; rotation exaggeration is direction-locked
  by a regression test

**Data + replay + API**
- Session JSON per run: record, full event log, per-frame feature timeline
  (replay input — signals, not video), activity starts, config snapshot
- Deterministic replay (>900 fps, never overwrites source); FastAPI REST +
  `WS /ws/state`
- Config-driven: every threshold/gain/schedule in `defaults.yaml`,
  overridable per run
- Fullscreen debug HUD; benchmark script; README with principles + explicit
  limitations

**Constraints honored:** no identity recognition, no developmental claims,
100% local, replayable, config-driven.

**Explicitly not in the MVP:** frontend UI (WebSocket is ready), packaging,
real-infant validation, multi-face, GPU inference.

## Quick start

```bash
python scripts/run_demo.py --camera 1 --duration 30   # live demo + twin
python scripts/replay_session.py data/sessions/<id>.json
python -m pytest tests/                                # 127 passed
python scripts/benchmark.py
python scripts/run_demo.py --camera 1 --api --api-port 8017
```

## Prepared Q&A

**Demo**
- *Show me.* `run_demo.py --camera 1 --duration 30` (fullscreen twin);
  `replay_session.py <session.json>`; `pytest` → 127 passed.
- *What am I looking at?* Your own face as the twin, expressions/head
  rotation amplified ~1.3–1.6×, driven live — a warped mesh of your face,
  not an avatar.

**Technical**
- *End-to-end latency?* Perception 65–78 ms + twin render ~11 ms + policy
  µs ≈ ~80–95 ms — under the ~100 ms mirror-perception threshold.
- *Why not 30 fps?* The loop runs as fast as perception allows
  (~13 fps @720p CPU). Render and policy have headroom (11 ms / µs);
  next lever is GPU or lower-res inference.
- *Why a mesh warp, not a neural avatar?* Deterministic, testable,
  real-time on CPU, no training data (no PII), no failure modes from
  unseen faces.
- *Is it really deterministic?* Yes — fixed schedules, no RNG in the loop;
  replay re-runs recorded features through the same pipeline and tests
  assert it.
- *How does it degrade?* Lower fps, mesh→cascade landmark fallback; the
  loop never dies silently.

**Product / safety**
- *Privacy?* 100% local; no identity recognition ever; sessions store
  structured signals and events — no video, no images.
- *Is attention real?* A behavioral heuristic estimate, labeled as such; no
  claims about internal state or development.
- *Developmental claims?* None. It measures responses to its own stimuli;
  it does not assess development.
- *Safe for a baby?* Framed as an interactive mirroring prototype:
  bounded amplification (safe limits in config), local processing, short
  sessions, no medical positioning.

**Roadmap**
- *What's next?* V2 options: frontend on the existing WebSocket,
  GPU/lower-res perception for 30 fps, audio features, packaging, and —
  only with proper ethics protocol — real-infant sessions.
- *Why is the policy simple?* Reproducibility over cleverness; the
  experiment scaffolding (schedules, response windows, session data) is
  the product, not the AI.
