"""Interaction policy.

V1 ships a deterministic, fully inspectable ``RuleBasedPolicy``. The
``InteractionPolicy`` interface is the replacement seam for a future
``LearnedPolicy`` - the engine does not care which implementation is active.

Every decision carries a reason list; decisions are logged as
POLICY_DECISION events for debugging and later ML training.
"""

from __future__ import annotations

from typing import Optional, Protocol

from ..config.settings import Settings
from ..metrics.events import Event, EventType
from ..state.baby_state import BabyState, BabyStateName
from ..temporal.smoothing import clamp
from .activities import ActivityInstance, Stimulus
from .actions import TwinAction, TwinActionType


class InteractionPolicy(Protocol):
    def decide(
        self,
        baby_state: BabyState,
        activity_state: Optional[ActivityInstance],
        history: list[Event],
        now: float,
        scheduled: Optional[Stimulus] = None,
    ) -> TwinAction:
        ...


def _recent(history: list[Event], *types: EventType, within_s: float, now: float) -> list[Event]:
    wanted = set(types)
    return [e for e in history
            if e.event in wanted and 0.0 <= now - e.timestamp <= within_s]


class RuleBasedPolicy:
    """Ordered if/then rules over observable signals.

    Rule precedence (first match wins):
      1. face absent / session over        -> PAUSE
      2. unreliable attention estimate     -> PAUSE (do not over-react)
      3. scheduled activity stimulus due   -> that stimulus action
      4. cause/effect trigger (activity)   -> predictable effect
      5. repeated recent responses         -> INTRODUCE_NOVELTY (infrequent)
      6. high attention + mouth movement   -> EXAGGERATE (mouth)
      7. high attention + head movement    -> LOOK_AT_BABY
      8. attention dropped recently        -> ATTRACT_ATTENTION
      9. low attention                     -> PAUSE
     10. otherwise                         -> MIRROR
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._last_novelty_at: Optional[float] = None
        self.decision_count = 0

    def decide(
        self,
        baby_state: BabyState,
        activity_state: Optional[ActivityInstance],
        history: list[Event],
        now: float,
        scheduled: Optional[Stimulus] = None,
    ) -> TwinAction:
        self.decision_count += 1
        cfg = self.settings.interaction
        engage = self.settings.attention.engage_threshold
        disengage = self.settings.attention.disengage_threshold

        def act(t: TwinActionType, target: str = "", intensity: float = 0.0,
                duration_ms: int = 0, direction: str = "", stimulus_id: str = "",
                *reasons: str) -> TwinAction:
            return TwinAction(type=t, target=target, intensity=clamp(intensity, 0.0, 1.0),
                              duration_ms=duration_ms, direction=direction,
                              stimulus_id=stimulus_id, reason=list(reasons), timestamp=now)

        # 1. no face / terminal states
        if not baby_state.face_present or baby_state.state == BabyStateName.NO_FACE:
            return act(TwinActionType.PAUSE, "", 0.0, 0, "", "", "face_absent")
        if baby_state.state in (BabyStateName.SESSION_END, BabyStateName.COOLDOWN):
            return act(TwinActionType.PAUSE, "", 0.0, 0, "", "",
                       f"state_{baby_state.state.value.lower()}")

        # 2. unreliable estimate -> no strong response
        if (baby_state.attention_confidence < cfg.min_confidence_to_act
                and baby_state.attention < engage):
            return act(TwinActionType.PAUSE, "", 0.0, 0, "", "",
                       "attention_confidence_low",
                       f"confidence={baby_state.attention_confidence:.2f}")

        # 3. scheduled activity stimulus (deterministic timeline)
        if scheduled is not None:
            a = scheduled.action
            return TwinAction(
                type=a.type, target=a.target, intensity=a.intensity,
                duration_ms=a.duration_ms, direction=a.direction,
                stimulus_id=a.stimulus_id,
                reason=["scheduled_stimulus", scheduled.stimulus_id,
                        f"activity={activity_state.id if activity_state else 'none'}"],
                timestamp=now,
            )

        activity_id = activity_state.id if activity_state else None

        # 4. cause/effect: observable baby action triggers predictable response
        if activity_id == "cause_effect":
            causes = _recent(history, EventType.MOUTH_OPENED, EventType.FACIAL_MOVEMENT,
                             within_s=1.5, now=now)
            if causes:
                return act(TwinActionType.SMILE, "facial_expression",
                           0.5 + 0.5 * baby_state.mouth_open, 1500, "", "",
                           "cause_effect_triggered",
                           f"trigger={causes[-1].event.value}")

        # 5. novelty: baby responded repeatedly -> small novel variation
        responses = _recent(history, EventType.BABY_RESPONSE, within_s=10.0, now=now)
        if len(responses) >= 2:
            last_novelty_window = _recent(history, EventType.TWIN_ACTION, within_s=cfg.novelty_interval_s,
                                          now=now)
            novelty_actions = [e for e in last_novelty_window
                               if e.payload.get("action") == TwinActionType.INTRODUCE_NOVELTY.value]
            if not novelty_actions:
                direction = "right" if len(responses) % 2 else "left"
                return act(TwinActionType.INTRODUCE_NOVELTY, "expression", 0.7, 2500, direction, "",
                           "repeated_responses", f"responses_last_10s={len(responses)}")

        # 6. high attention + mouth/facial movement -> exaggerated mirror
        mouth_events = _recent(history, EventType.MOUTH_OPENED, EventType.FACIAL_MOVEMENT,
                               within_s=1.5, now=now)
        if baby_state.attention >= engage and baby_state.attention_confidence >= cfg.min_confidence_to_act:
            if mouth_events or baby_state.mouth_open > 0.45:
                intensity = clamp(0.4 + 0.6 * max(baby_state.mouth_open,
                                                   baby_state.facial_activity), 0.0, 1.0)
                return act(TwinActionType.EXAGGERATE, "mouth", intensity, 900, "", "",
                           "face_present", "mouth_motion_detected", "engagement_high",
                           f"attention={baby_state.attention:.2f}")

        # 7. high attention + head movement -> mirror / look at baby
        head_events = _recent(history, EventType.HEAD_MOVED, EventType.ORIENTATION_CHANGED,
                              within_s=2.0, now=now)
        if baby_state.attention >= engage and (head_events or baby_state.head_motion > 0.2):
            return act(TwinActionType.LOOK_AT_BABY, "head",
                       clamp(baby_state.head_motion * 2.0, 0.2, 1.0), 800, "", "",
                       "engagement_high", "head_motion_detected")

        # 8. attention just dropped -> try to re-attract
        dropped = _recent(history, EventType.ATTENTION_DECREASED, within_s=2.5, now=now)
        if dropped:
            return act(TwinActionType.ATTRACT_ATTENTION, "expression", 0.6, 2000, "", "",
                       "attention_decreased")

        # 9. sustained low attention -> pause (change of pace)
        if baby_state.attention <= disengage:
            return act(TwinActionType.PAUSE, "", 0.0, 0, "", "",
                       "attention_low", f"attention={baby_state.attention:.2f}")

        # 10. default: gentle mirror
        return act(TwinActionType.MIRROR, "face",
                   clamp(0.3 + 0.5 * baby_state.movement, 0.0, 1.0), 500, "", "",
                   "baseline_mirror", f"attention={baby_state.attention:.2f}")


def create_policy(settings: Settings) -> InteractionPolicy:
    if settings.interaction.policy == "rule_based":
        return RuleBasedPolicy(settings)
    raise ValueError(f"unknown policy: {settings.interaction.policy}")
