"""Twin controller: turns (baby state + policy action) into a smoothed
TwinState. Owns the exaggeration mapping, action overrides, and the response
time constant. The renderer consumes whatever this produces.
"""

from __future__ import annotations

from ..interaction.actions import TwinAction, TwinActionType
from ..state.baby_state import BabyState
from ..temporal.sequence import TemporalFeatures
from ..temporal.smoothing import clamp
from .exaggeration import ExaggerationEngine
from .twin_state import TwinState

import math


class TwinController:
    def __init__(self, exaggeration: ExaggerationEngine, response_speed_ms: int = 250) -> None:
        self.exaggeration = exaggeration
        self.response_speed_s = max(1.0, response_speed_ms) / 1000.0
        self.state = TwinState()
        self.target = TwinState()
        self.last_action = TwinAction.idle()

    def update(self, temporal: TemporalFeatures, baby: BabyState,
               action: TwinAction, dt: float) -> TwinState:
        target = self.exaggeration.map_to_twin(temporal, baby)
        target = self._apply_action(target, action, temporal, baby)
        target.action = action.type.value
        target.action_intensity = action.intensity
        target.timestamp = temporal.timestamp
        self.target = target
        self.state = self._smooth(self.state, target, dt)
        self.last_action = action
        return self.state

    def _apply_action(self, target: TwinState, action: TwinAction,
                      temporal: TemporalFeatures, baby: BabyState) -> TwinState:
        lim = self.exaggeration.config.limits
        i = clamp(action.intensity, 0.0, 1.0)
        t = action.type
        if t == TwinActionType.IDLE or t == TwinActionType.PAUSE:
            # hold near neutral
            n = self.exaggeration.config.neutral
            target.mouth_open = n.mouth_open
            target.smile = 0.0
            target.expression_intensity *= 0.3
        elif t == TwinActionType.MIRROR:
            pass  # pure mirrored+exaggerated mapping is already applied
        elif t == TwinActionType.EXAGGERATE:
            target.mouth_open = clamp(target.mouth_open * (1.0 + 0.5 * i), *lim["mouth_open"])
            target.expression_intensity = clamp(
                target.expression_intensity * (1.0 + 0.5 * i), *lim["expression_intensity"])
        elif t == TwinActionType.SMILE:
            target.smile = clamp(0.4 + 0.6 * i, *lim["smile"])
        elif t == TwinActionType.OPEN_MOUTH:
            target.mouth_open = clamp(0.3 + 0.7 * i, *lim["mouth_open"])
        elif t == TwinActionType.LOOK_AT_BABY:
            target.head_yaw = clamp(temporal.head_yaw, *lim["head_yaw"])
            target.head_pitch = clamp(temporal.head_pitch, *lim["head_pitch"])
        elif t == TwinActionType.TURN_HEAD:
            direction = action.direction or "left"
            sign = -1.0 if direction == "left" else 1.0
            target.head_yaw = clamp(sign * (15.0 + 25.0 * i), *lim["head_yaw"])
        elif t in (TwinActionType.ATTRACT_ATTENTION, TwinActionType.ENCOURAGE):
            target.expression_intensity = clamp(0.5 + 0.5 * i, *lim["expression_intensity"])
            target.eye_scale = clamp(1.0 + 0.2 * i, *lim["eye_scale"])
            target.smile = clamp(0.3 + 0.4 * i, *lim["smile"])
        elif t == TwinActionType.INTRODUCE_NOVELTY:
            target.expression_intensity = clamp(0.6 + 0.4 * i, *lim["expression_intensity"])
            target.mouth_open = clamp(0.2 + 0.4 * i, *lim["mouth_open"])
            target.head_yaw = clamp((1.0 if direction_of(action) == "right" else -1.0)
                                    * (20.0 + 15.0 * i), *lim["head_yaw"])
        elif t == TwinActionType.REPEAT:
            target.expression_intensity = clamp(0.4 + 0.4 * i, *lim["expression_intensity"])
        elif t == TwinActionType.END_ACTIVITY:
            n = self.exaggeration.config.neutral
            target.mouth_open = n.mouth_open
            target.smile = 0.0
            target.expression_intensity = 0.0
        return target

    def _smooth(self, current: TwinState, target: TwinState, dt: float) -> TwinState:
        """First-order response smoothing; alpha from response_speed_ms."""
        if dt <= 0:
            return current
        alpha = 1.0 - math.exp(-dt / self.response_speed_s)

        def mix(a: float, b: float) -> float:
            return a + (b - a) * alpha

        return TwinState(
            mouth_open=mix(current.mouth_open, target.mouth_open),
            smile=mix(current.smile, target.smile),
            head_yaw=mix(current.head_yaw, target.head_yaw),
            head_pitch=mix(current.head_pitch, target.head_pitch),
            head_roll=mix(current.head_roll, target.head_roll),
            eye_scale=mix(current.eye_scale, target.eye_scale),
            expression_intensity=mix(current.expression_intensity, target.expression_intensity),
            action=target.action,
            action_intensity=target.action_intensity,
            timestamp=target.timestamp,
        )


def direction_of(action: TwinAction) -> str:
    return action.direction or "left"
