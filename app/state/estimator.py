"""Attention estimation + high-level state machine + BabyState estimation.

Both estimators implement interfaces so they can be replaced by learned
models later (RuleBasedAttentionEstimator -> LearnedAttentionEstimator)
without touching the engine.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Protocol

from ..config.settings import Settings
from ..metrics.events import Event, EventLog, EventType
from ..perception.head_pose import HeadOrientation
from ..temporal.sequence import TemporalFeatures
from ..temporal.smoothing import clamp
from .baby_state import BabyState, BabyStateName
from .confidence import Estimate, combine

ORIENTATION_SCORE = {
    HeadOrientation.LOOKING_CENTER: 1.0,
    HeadOrientation.LOOKING_UP: 0.55,
    HeadOrientation.LOOKING_DOWN: 0.55,
    HeadOrientation.LOOKING_LEFT: 0.45,
    HeadOrientation.LOOKING_RIGHT: 0.45,
    HeadOrientation.UNKNOWN: 0.25,
}


class AttentionEstimator(Protocol):
    def estimate(self, temporal: TemporalFeatures, detection_confidence: float,
                 now: float) -> Estimate:
        ...


class RuleBasedAttentionEstimator:
    """Deterministic, inspectable attention estimate.

    Combines observable signals only: face presence, orientation toward the
    camera, tracking stability, face size (distance proxy), and how long the
    face has been visible. Output is an ESTIMATE of engagement/attention,
    never a claim about conscious focus.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._face_visible_since: Optional[float] = None

    def estimate(self, temporal: TemporalFeatures, detection_confidence: float,
                 now: float) -> Estimate:
        evidence: list[str] = []
        if not temporal.face_present:
            self._face_visible_since = None
            return Estimate(value=0.0, confidence=max(0.1, detection_confidence),
                            evidence=["no_face"])

        if self._face_visible_since is None:
            self._face_visible_since = now
        visible_for = max(0.0, now - self._face_visible_since)
        evidence.append("face_present")

        orient_score = ORIENTATION_SCORE.get(temporal.orientation, 0.25)
        if temporal.orientation == HeadOrientation.LOOKING_CENTER:
            evidence.append("head_oriented_toward_camera")
        else:
            evidence.append(f"head_orientation_{temporal.orientation.value.lower()}")

        stability = 1.0 - temporal.head_motion
        if stability > 0.6:
            evidence.append("stable_tracking")
        else:
            evidence.append("moving")

        # bigger face on frame -> closer / more available for interaction
        size_score = clamp(temporal.face_scale / 0.35, 0.0, 1.0)
        if size_score > 0.5:
            evidence.append("face_large_in_frame")

        persistence = clamp(visible_for / 10.0, 0.0, 1.0)
        if persistence > 0.5:
            evidence.append("sustained_presence")

        value = (
            0.35 * 1.0                       # face present
            + 0.30 * orient_score
            + 0.15 * stability
            + 0.10 * size_score
            + 0.10 * persistence
        )
        confidence = combine(
            detection_confidence,
            0.70 + 0.30 * stability,
            0.55 + 0.45 * size_score,
        )
        # mouth activity slightly raises the estimate (active participation)
        if temporal.mouth_open > 0.3:
            value += 0.05
            evidence.append("mouth_active")

        return Estimate(value=clamp(value, 0.0, 1.0), confidence=confidence,
                        evidence=evidence)

    def reset(self) -> None:
        self._face_visible_since = None


# ----------------------------------------------------------------------
class TransitionContext:
    """Inputs consulted by the state machine guards."""

    def __init__(self, attention: float, attention_confidence: float,
                 movement: float, activity_id: Optional[str],
                 baby_responded: bool, policy_forced: Optional[BabyStateName],
                 now: float) -> None:
        self.attention = attention
        self.attention_confidence = attention_confidence
        self.movement = movement
        self.activity_id = activity_id
        self.baby_responded = baby_responded
        self.policy_forced = policy_forced
        self.now = now


class StateMachine:
    """Explicit, explainable state machine with hysteresis + minimum dwell.

    Every transition returns a reason string so logs can justify behavior.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = BabyStateName.NO_FACE
        self.entered_at: Optional[float] = None
        self._pending: Optional[BabyStateName] = None
        self._pending_since = 0.0

    def _dwell_ok(self, now: float) -> bool:
        if self.entered_at is None:
            return True
        return (now - self.entered_at) * 1000.0 >= self.settings.state.min_dwell_ms

    def step(self, face_present: bool, ctx: TransitionContext) -> Optional[tuple[BabyStateName, str]]:
        """Advance the machine; returns (new_state, reason) or None."""
        if self.entered_at is None:
            self.entered_at = ctx.now
        st = self.settings.state
        target: Optional[BabyStateName] = None
        reason = ""

        # forced transitions first (session end, policy cooldown)
        if ctx.policy_forced is not None and ctx.policy_forced != self.state:
            target, reason = ctx.policy_forced, "forced_by_policy"

        if target is None:
            if self.state == BabyStateName.SESSION_END:
                return None
            if not face_present:
                if self.state != BabyStateName.NO_FACE:
                    target, reason = BabyStateName.NO_FACE, "face_absent"
            elif self.state == BabyStateName.NO_FACE:
                target, reason = BabyStateName.OBSERVING, "face_detected"
            elif self.state == BabyStateName.OBSERVING:
                if ctx.attention >= st.engaged_min_attention and ctx.attention_confidence >= 0.4:
                    target, reason = BabyStateName.ENGAGED, (
                        f"attention {ctx.attention:.2f} >= {st.engaged_min_attention:.2f}")
            elif self.state == BabyStateName.ENGAGED:
                if ctx.baby_responded:
                    target, reason = BabyStateName.RESPONDING, "baby_response_detected"
                elif ctx.attention <= st.low_engagement_attention:
                    target, reason = BabyStateName.LOW_ENGAGEMENT, (
                        f"attention {ctx.attention:.2f} <= {st.low_engagement_attention:.2f}")
                elif (ctx.movement > 0.5
                      and ctx.activity_id in ("follow", "cause_effect", "repetition")):
                    target, reason = BabyStateName.EXPLORING, "active_movement_in_activity"
            elif self.state == BabyStateName.RESPONDING:
                if (ctx.now - (self.entered_at or ctx.now)) * 1000.0 >= st.responding_timeout_ms:
                    target, reason = BabyStateName.ENGAGED, "response_complete"
                elif ctx.attention <= st.low_engagement_attention:
                    target, reason = BabyStateName.LOW_ENGAGEMENT, "attention_dropped"
            elif self.state == BabyStateName.EXPLORING:
                if ctx.baby_responded:
                    target, reason = BabyStateName.RESPONDING, "baby_response_detected"
                elif ctx.attention <= st.low_engagement_attention:
                    target, reason = BabyStateName.LOW_ENGAGEMENT, "attention_dropped"
                elif ctx.movement < 0.15:
                    target, reason = BabyStateName.ENGAGED, "movement_settled"
            elif self.state == BabyStateName.LOW_ENGAGEMENT:
                if ctx.attention >= st.engaged_min_attention and ctx.attention_confidence >= 0.4:
                    target, reason = BabyStateName.OBSERVING, "attention_recovered"
            elif self.state == BabyStateName.COOLDOWN:
                if (ctx.now - (self.entered_at or ctx.now)) * 1000.0 >= st.cooldown_ms:
                    target, reason = BabyStateName.OBSERVING, "cooldown_elapsed"

        if target is None or target == self.state:
            # hysteresis on the ENGAGED <-> LOW_ENGAGEMENT boundary is
            # handled by the asymmetric thresholds above (0.60 vs 0.35).
            self._pending = None
            return None

        # debounce: a new state must be pending across steps, and the current
        # state must have held for min_dwell (unless forced by policy)
        if self._pending != target:
            self._pending = target
            self._pending_since = ctx.now
            return None
        held_pending_ms = (ctx.now - self._pending_since) * 1000.0
        if reason != "forced_by_policy":
            if not self._dwell_ok(ctx.now) or held_pending_ms < self.settings.state.min_dwell_ms:
                return None
        self._pending = None
        self.state = target
        self.entered_at = ctx.now
        return (target, reason)

    def force(self, state: BabyStateName, now: float) -> None:
        self.state = state
        self.entered_at = now
        self._pending = None


class StateEstimator:
    """Builds the BabyState each update and runs the state machine."""

    def __init__(self, settings: Settings, event_log: EventLog,
                 session_id: Optional[str] = None) -> None:
        self.settings = settings
        self.events = event_log
        self.session_id = session_id
        self.attention_estimator: AttentionEstimator = RuleBasedAttentionEstimator(settings)
        self.machine = StateMachine(settings)
        self._baby_responded = False
        self._policy_forced: Optional[BabyStateName] = None
        self._last_attention_band: Optional[str] = None

    def note_baby_response(self) -> None:
        self._baby_responded = True

    def force_state(self, state: BabyStateName, now: float) -> None:
        self.machine.force(state, now)

    def request_policy_state(self, state: Optional[BabyStateName]) -> None:
        self._policy_forced = state

    def update(self, temporal: TemporalFeatures, detection_confidence: float,
               interaction_duration: float, response_latency: Optional[float],
               activity_id: Optional[str], now: float) -> tuple[BabyState, list[Event]]:
        attention = self.attention_estimator.estimate(
            temporal, detection_confidence if temporal.face_present else 0.0, now)
        out: list[Event] = []

        # attention band change event (hysteresis via engage/disengage thresholds)
        band = self._attention_band(attention.value, temporal.face_present)
        if band != self._last_attention_band:
            if self._last_attention_band is not None:
                out.append(self._emit(EventType.ATTENTION_CHANGED, now,
                                      value=round(attention.value, 3),
                                      confidence=round(attention.confidence, 3),
                                      band=band))
            self._last_attention_band = band

        ctx = TransitionContext(
            attention=attention.value, attention_confidence=attention.confidence,
            movement=temporal.movement, activity_id=activity_id,
            baby_responded=self._baby_responded,
            policy_forced=self._policy_forced, now=now,
        )
        prev_state = self.machine.state
        result = self.machine.step(temporal.face_present, ctx)
        self._baby_responded = False
        if result is not None:
            new_state, reason = result
            out.append(self._emit(EventType.STATE_CHANGED, now,
                                  from_state=prev_state.value,
                                  state=new_state.value, reason=reason))

        state_duration = 0.0 if self.machine.entered_at is None else now - self.machine.entered_at
        baby = BabyState(
            face_present=temporal.face_present,
            attention=attention.value,
            attention_confidence=attention.confidence,
            movement=temporal.movement,
            vocalization=temporal.vocalization,
            facial_activity=temporal.facial_motion,
            head_motion=temporal.head_motion,
            mouth_open=temporal.mouth_open,
            interaction_duration=interaction_duration,
            response_latency=response_latency,
            current_orientation=temporal.orientation,
            state=self.machine.state,
            state_duration=state_duration,
            timestamp=now,
            evidence=attention.evidence,
        )
        return baby, out

    def _attention_band(self, value: float, face_present: bool) -> str:
        if not face_present:
            return "none"
        a = self.settings.attention
        if value >= a.engage_threshold:
            return "high"
        if value <= a.disengage_threshold:
            return "low"
        return "medium"

    def _emit(self, etype: EventType, ts: float, **payload) -> Event:
        ev = Event(event=etype, timestamp=ts, session_id=self.session_id, payload=payload)
        self.events.append(ev)
        return ev
