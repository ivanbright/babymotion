"""Configuration system: pydantic models + YAML loading with defaults merge."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

PACKAGE_DIR = Path(__file__).resolve().parent
BABYMOTION_DIR = PACKAGE_DIR.parent.parent  # .../babymotion
REPO_ROOT = BABYMOTION_DIR.parent


class CameraConfig(BaseModel):
    device: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30
    backend: str = "auto"
    apply_quality: bool = True
    mirror: bool = True  # selfie-style horizontal flip: the baby's own right
    # direction appears on image-right, so "turn right" matches the twin


class MicrophoneConfig(BaseModel):
    enabled: bool = False
    device: Optional[int] = None
    sample_rate: int = 16000
    block_ms: int = 100


class PerceptionConfig(BaseModel):
    landmark_model: str = ""
    min_face_confidence: float = 0.5
    min_landmark_visibility: float = 0.5
    use_cascade_fallback: bool = True
    roi_padding: float = 0.15
    max_faces: int = 1


class AttentionConfig(BaseModel):
    smoothing_window_ms: int = 500
    engage_threshold: float = 0.65
    disengage_threshold: float = 0.35
    min_confidence: float = 0.40


class TemporalConfig(BaseModel):
    window_short_ms: int = 1000
    window_medium_ms: int = 3000
    window_long_ms: int = 5000
    state_update_hz: int = 10
    ema_alpha: float = 0.35
    motion_enter: float = 0.08
    motion_exit: float = 0.04
    event_min_duration_ms: int = 250
    face_loss_grace_ms: int = 700


class StateConfig(BaseModel):
    min_dwell_ms: int = 800
    engaged_min_attention: float = 0.60
    low_engagement_attention: float = 0.35
    responding_timeout_ms: int = 4000
    cooldown_ms: int = 3000


class ExaggerationConfig(BaseModel):
    mouth: float = 1.5
    head_rotation: float = 1.3
    facial_motion: float = 1.4
    expression: float = 1.2


class NeutralConfig(BaseModel):
    mouth_open: float = 0.10
    mouth_width: float = 0.45
    eye_open: float = 0.80


class MeshConfig(BaseModel):
    """Textured 3D mesh twin renderer (warps the baby's own pixels)."""

    expression_gain: float = 1.6    # amplifies non-rigid (expression) residual
    depth_gain: float = 1.4         # amplifies depth (z) residual
    head_rotation_gain: float = 0.0  # 0 = use twin.exaggeration.head_rotation
    action_boost: float = 1.6       # extra multiplier during expressive actions
    vertex_smoothing: int = 250     # ms time constant for vertex targets
    neutral_update_rate: float = 0.05  # EMA alpha for neutral template refresh
    neutral_mouth_band: float = 0.15   # |mouth_open - neutral| to count as neutral
    neutral_max_facial_activity: float = 0.12
    shading: bool = True            # depth-based flat shading
    spacing_ratio: float = 0.12     # render-mesh min point spacing / face width
    render_scale: float = 0.5       # rasterize the mesh at this fraction of frame
    grid_step: int = 4              # displacement-field grid (work-res px)
    min_frame_interval_ms: int = 33 # rasterization throttle (geometry still 1:1)
    panel_height: int = 300         # twin panel size in the debug window
    background_blur: int = 21       # blurred camera image behind the mesh


class TwinConfig(BaseModel):
    exaggeration: ExaggerationConfig = Field(default_factory=ExaggerationConfig)
    neutral: NeutralConfig = Field(default_factory=NeutralConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    limits: dict[str, list[float]] = Field(
        default_factory=lambda: {
            "mouth_open": [0.0, 1.0],
            "smile": [0.0, 1.0],
            "head_yaw": [-45.0, 45.0],
            "head_pitch": [-35.0, 35.0],
            "head_roll": [-30.0, 30.0],
            "eye_scale": [0.5, 1.3],
            "expression_intensity": [0.0, 1.0],
        }
    )
    response_speed_ms: int = 250


class InteractionConfig(BaseModel):
    policy: str = "rule_based"
    stimulus_window_ms: int = 3000
    response_event_gap_ms: int = 400
    min_confidence_to_act: float = 0.40
    novelty_interval_s: float = 20.0
    imitation_window_ms: int = 2500


class ActivitySpecConfig(BaseModel):
    duration: int = 30
    move_period_s: float = 4.0
    hold_threshold_s: float = 1.5
    repeat_period_s: float = 3.0
    repeats: int = 8


class ActivitiesConfig(BaseModel):
    default_duration_seconds: int = 30
    mirror: ActivitySpecConfig = Field(default_factory=lambda: ActivitySpecConfig(duration=30))
    exaggerated_mirror: ActivitySpecConfig = Field(default_factory=lambda: ActivitySpecConfig(duration=30))
    follow: ActivitySpecConfig = Field(default_factory=lambda: ActivitySpecConfig(duration=30))
    cause_effect: ActivitySpecConfig = Field(default_factory=lambda: ActivitySpecConfig(duration=30))
    repetition: ActivitySpecConfig = Field(default_factory=lambda: ActivitySpecConfig(duration=40))

    def spec(self, activity_id: str) -> ActivitySpecConfig:
        return getattr(self, activity_id, ActivitySpecConfig())


class SessionConfig(BaseModel):
    default_duration_seconds: int = 120


class RecordingConfig(BaseModel):
    save_video: bool = False
    save_features: bool = True
    output_dir: str = "data/sessions"
    video_dir: str = "recordings"


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    enabled: bool = False


class DebugConfig(BaseModel):
    show_window: bool = True
    window_name: str = "BabyMotion Debug"
    draw_landmarks: bool = True
    fullscreen: bool = True  # scale the overlay to the screen and go fullscreen


class Settings(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    microphone: MicrophoneConfig = Field(default_factory=MicrophoneConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    twin: TwinConfig = Field(default_factory=TwinConfig)
    interaction: InteractionConfig = Field(default_factory=InteractionConfig)
    activities: ActivitiesConfig = Field(default_factory=ActivitiesConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(override_path: str | Path | None = None) -> Settings:
    """Load defaults.yaml, then deep-merge an optional override file."""
    with open(PACKAGE_DIR / "defaults.yaml", "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if override_path is not None:
        with open(override_path, "r", encoding="utf-8") as fh:
            override = yaml.safe_load(fh) or {}
        data = _deep_merge(data, override)
    return Settings.model_validate(data)


LANDMARK_MODEL_CANDIDATES = [
    BABYMOTION_DIR / "models" / "face_landmarker.task",
    REPO_ROOT / "models" / "face_landmarker.task",
]


def resolve_landmark_model(settings: Settings) -> Path:
    """Resolve the MediaPipe FaceLandmarker .task file path."""
    if settings.perception.landmark_model:
        p = Path(settings.perception.landmark_model)
        if not p.is_absolute():
            p = (BABYMOTION_DIR / p).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"landmark model not found: {p}")
    for candidate in LANDMARK_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "face_landmarker.task not found; set perception.landmark_model in config"
    )


def resolve_output_dir(settings: Settings) -> Path:
    p = Path(settings.recording.output_dir)
    if not p.is_absolute():
        p = (BABYMOTION_DIR / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
