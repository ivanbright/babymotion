"""Developmental activities: deterministic, inspectable interaction protocols.

V1 activities are fixed schedules (no randomness) so experiments are
reproducible. Each activity declares its stimulus sequence, expected
observations, and termination conditions.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

from ..config.settings import Settings
from ..interaction.actions import TwinAction, TwinActionType


@dataclass
class Stimulus:
    stimulus_id: str
    action: TwinAction
    start_s: float
    duration_s: float


@dataclass
class Activity:
    id: str
    name: str
    objective: str
    duration: float
    stimuli: list[Stimulus]
    expected_observations: list[str]
    termination_conditions: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "objective": self.objective,
            "duration": self.duration,
            "stimuli": [
                {"stimulus_id": s.stimulus_id, "action": s.action.to_dict(),
                 "start_s": s.start_s, "duration_s": s.duration_s}
                for s in self.stimuli
            ],
            "expected_observations": self.expected_observations,
            "termination_conditions": self.termination_conditions,
        }


def _stim(activity_id: str, idx: int, action_type: TwinActionType, start_s: float,
          duration_s: float = 2.0, intensity: float = 0.7, target: str = "",
          direction: str = "") -> Stimulus:
    return Stimulus(
        stimulus_id=f"{activity_id}_stim_{idx:03d}",
        action=TwinAction(type=action_type, target=target, intensity=intensity,
                          duration_ms=int(duration_s * 1000), direction=direction,
                          stimulus_id=f"{activity_id}_stim_{idx:03d}"),
        start_s=start_s,
        duration_s=duration_s,
    )


def build_registry(settings: Settings) -> dict[str, Activity]:
    """Construct the five V1 activities from configuration."""
    a = settings.activities
    reg: dict[str, Activity] = {}

    # 1. Mirror - twin follows the baby continuously; periodic gentle prompts.
    spec = a.mirror
    reg["mirror"] = Activity(
        id="mirror",
        name="Mirror",
        objective="Twin mirrors baby movement without amplification; measure baseline response.",
        duration=spec.duration,
        stimuli=[_stim("mirror", i, TwinActionType.LOOK_AT_BABY, t, 2.0, 0.4, "head")
                 for i, t in enumerate([5.0, 15.0, 25.0]) if t < spec.duration],
        expected_observations=["baby_movement_mirrored", "attention_estimate"],
        termination_conditions=["no_face", "low_attention", "duration_elapsed"],
    )

    # 2. Exaggerated mirror - amplification bursts at fixed times.
    spec = a.exaggerated_mirror
    reg["exaggerated_mirror"] = Activity(
        id="exaggerated_mirror",
        name="Exaggerated Mirror",
        objective="Twin amplifies baby facial movement; measure response to exaggeration.",
        duration=spec.duration,
        stimuli=[_stim("exaggerated_mirror", i, TwinActionType.EXAGGERATE, t, 2.5, 0.8, "facial_expression")
                 for i, t in enumerate([6.0, 14.0, 22.0]) if t < spec.duration],
        expected_observations=["response_to_exaggeration", "response_latency"],
        termination_conditions=["no_face", "low_attention", "duration_elapsed"],
    )

    # 3. Follow - twin performs slow deterministic head turns; system observes
    # whether the baby's head orientation follows.
    spec = a.follow
    stimuli: list[Stimulus] = []
    t = 4.0
    idx = 0
    direction = "left"
    while t < spec.duration - 2.0:
        stimuli.append(_stim("follow", idx, TwinActionType.TURN_HEAD, t,
                             spec.move_period_s, 0.7, "head", direction))
        direction = "right" if direction == "left" else "left"
        idx += 1
        t += spec.move_period_s
    reg["follow"] = Activity(
        id="follow",
        name="Follow",
        objective="Twin turns slowly; observe whether head orientation follows.",
        duration=spec.duration,
        stimuli=stimuli,
        expected_observations=["orientation_follows_twin", "orientation_change_latency"],
        termination_conditions=["no_face", "duration_elapsed"],
    )

    # 4. Cause/effect - a baby interaction (mouth open) triggers a predictable
    # visual response; plus scheduled demo triggers keep the protocol fixed.
    spec = a.cause_effect
    reg["cause_effect"] = Activity(
        id="cause_effect",
        name="Cause/Effect",
        objective="Baby mouth-open triggers a predictable twin response; measure response rate.",
        duration=spec.duration,
        stimuli=[_stim("cause_effect", i, TwinActionType.ATTRACT_ATTENTION, t, 1.5, 0.6, "expression")
                 for i, t in enumerate([8.0, 18.0, 28.0]) if t < spec.duration],
        expected_observations=["stimulus_cause", "baby_response", "response_latency"],
        termination_conditions=["no_face", "low_attention", "duration_elapsed"],
    )

    # 5. Repetition - identical stimulus repeated; measure habituation.
    spec = a.repetition
    stimuli = [
        _stim("repetition", i, TwinActionType.REPEAT, 3.0 + i * spec.repeat_period_s,
              1.5, 0.7, "expression")
        for i in range(spec.repeats)
        if 3.0 + i * spec.repeat_period_s < spec.duration - 1.0
    ]
    reg["repetition"] = Activity(
        id="repetition",
        name="Repetition",
        objective="Repeat an identical stimulus and measure how the response changes.",
        duration=spec.duration,
        stimuli=stimuli,
        expected_observations=["response_trend_across_repeats", "habituation"],
        termination_conditions=["no_face", "duration_elapsed"],
    )
    return reg


class ActivityPhase(str, enum.Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


class ActivityInstance:
    """Runtime state of one activity execution (deterministic timeline)."""

    def __init__(self, activity: Activity, started_at: float) -> None:
        self.activity = activity
        self.started_at = started_at
        self.phase = ActivityPhase.RUNNING
        self.fired: set[str] = set()
        self.ended_at: Optional[float] = None
        self.end_reason: str = ""

    @property
    def id(self) -> str:
        return self.activity.id

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.started_at)

    def due_stimuli(self, now: float) -> list[Stimulus]:
        """Stimuli whose start time has passed and which have not fired yet."""
        rel = self.elapsed(now)
        due = []
        for s in self.activity.stimuli:
            if s.stimulus_id not in self.fired and rel >= s.start_s:
                due.append(s)
                self.fired.add(s.stimulus_id)
        return due

    def active_stimulus(self, now: float) -> Optional[Stimulus]:
        rel = self.elapsed(now)
        for s in self.activity.stimuli:
            if s.start_s <= rel <= s.start_s + s.duration_s:
                return s
        return None

    def is_duration_over(self, now: float) -> bool:
        return self.elapsed(now) >= self.activity.duration

    def terminate(self, now: float, reason: str,
                  phase: ActivityPhase = ActivityPhase.ABORTED) -> None:
        self.phase = phase
        self.ended_at = now
        self.end_reason = reason

    @property
    def running(self) -> bool:
        return self.phase == ActivityPhase.RUNNING

    def progress(self, now: float) -> float:
        return min(1.0, self.elapsed(now) / max(1e-6, self.activity.duration))

    def to_dict(self, now: float) -> dict:
        d = self.activity.to_dict()
        d.update({
            "phase": self.phase.value,
            "started_at": self.started_at,
            "progress": round(self.progress(now), 3),
            "fired_stimuli": sorted(self.fired),
            "end_reason": self.end_reason,
        })
        return d
