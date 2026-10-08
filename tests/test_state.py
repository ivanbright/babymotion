"""State engine tests: transitions, hysteresis, dwell, confidence gating."""

from app.config.settings import load_settings
from app.metrics.events import EventLog
from app.state.baby_state import BabyStateName
from app.state.estimator import StateEstimator, StateMachine, TransitionContext
from app.temporal.sequence import TemporalEngine, TemporalFeatures

from synthetic import timeline_no_face, timeline_stable_face


def make_estimator(settings=None):
    settings = settings or load_settings()
    log = EventLog()
    return StateEstimator(settings, log, "test"), log


def ctx(attention=0.5, conf=0.9, movement=0.1, activity=None,
        responded=False, forced=None, now=0.0):
    return TransitionContext(attention=attention, attention_confidence=conf,
                             movement=movement, activity_id=activity,
                             baby_responded=responded, policy_forced=forced,
                             now=now)


def test_initial_state_is_no_face():
    sm = StateMachine(load_settings())
    assert sm.state == BabyStateName.NO_FACE


def test_no_face_to_observing_to_engaged():
    sm = StateMachine(load_settings())
    now = 0.0
    # face appears
    r = sm.step(True, ctx(attention=0.3, now=now))
    assert r is None  # pending
    now = 1.0
    r = sm.step(True, ctx(attention=0.3, now=now))
    assert r == (BabyStateName.OBSERVING, "face_detected")
    # attention rises -> ENGAGED (after dwell + pending)
    now = 2.0
    r = sm.step(True, ctx(attention=0.8, now=now))
    assert r is None
    now = 3.5
    r = sm.step(True, ctx(attention=0.8, now=now))
    assert r is not None and r[0] == BabyStateName.ENGAGED
    assert "attention" in r[1]


def test_hysteresis_between_engaged_and_low():
    sm = StateMachine(load_settings())
    now = 0.0
    sm.step(True, ctx(attention=0.3, now=now)); now = 1.0
    sm.step(True, ctx(attention=0.3, now=now)); now = 2.0  # OBSERVING
    sm.step(True, ctx(attention=0.8, now=now)); now = 3.5
    sm.step(True, ctx(attention=0.8, now=now))  # ENGAGED
    assert sm.state == BabyStateName.ENGAGED
    # attention in the ambiguous band (0.35 < a < 0.60): must NOT leave ENGAGED
    for t in (5.0, 6.5, 8.0, 9.5, 11.0):
        r = sm.step(True, ctx(attention=0.5, now=t))
        assert sm.state == BabyStateName.ENGAGED
    # attention clearly low -> LOW_ENGAGEMENT
    sm.step(True, ctx(attention=0.2, now=12.0))
    r = sm.step(True, ctx(attention=0.2, now=13.5))
    assert r is not None and r[0] == BabyStateName.LOW_ENGAGEMENT


def test_baby_response_triggers_responding():
    sm = StateMachine(load_settings())
    now = 0.0
    sm.step(True, ctx(attention=0.3, now=now)); now = 1.0
    sm.step(True, ctx(attention=0.3, now=now)); now = 2.0
    sm.step(True, ctx(attention=0.9, now=now)); now = 3.5
    sm.step(True, ctx(attention=0.9, now=now))
    assert sm.state == BabyStateName.ENGAGED
    sm.step(True, ctx(attention=0.9, responded=True, now=4.0))
    r = sm.step(True, ctx(attention=0.9, responded=True, now=4.9))
    assert r is not None and r[0] == BabyStateName.RESPONDING
    # timeout returns to ENGAGED (debounced over consecutive steps)
    assert sm.step(True, ctx(attention=0.9, now=9.4)) is None
    assert sm.step(True, ctx(attention=0.9, now=9.6)) is None
    r = sm.step(True, ctx(attention=0.9, now=10.5))
    assert r is not None and r[0] == BabyStateName.ENGAGED


def test_face_absent_returns_no_face():
    sm = StateMachine(load_settings())
    sm.step(True, ctx(attention=0.3, now=0.0))
    sm.step(True, ctx(attention=0.3, now=1.0))
    assert sm.state == BabyStateName.OBSERVING
    sm.step(False, ctx(attention=0.0, now=2.0))
    r = sm.step(False, ctx(attention=0.0, now=4.0))
    assert r is not None and r[0] == BabyStateName.NO_FACE


def test_min_dwell_blocks_immediate_chatter():
    sm = StateMachine(load_settings())
    sm.step(True, ctx(attention=0.3, now=0.0))
    sm.step(True, ctx(attention=0.3, now=1.0))  # -> OBSERVING (entered at 1.0)
    # immediately try to engage: dwell (800ms) + pending must both pass
    sm.step(True, ctx(attention=0.9, now=1.1))
    r = sm.step(True, ctx(attention=0.9, now=1.3))
    assert r is None  # not yet
    r = sm.step(True, ctx(attention=0.9, now=2.2))
    assert r is not None


def test_session_end_is_terminal():
    sm = StateMachine(load_settings())
    sm.force(BabyStateName.SESSION_END, now=0.0)
    r = sm.step(True, ctx(attention=0.9, now=1.0))
    assert r is None
    assert sm.state == BabyStateName.SESSION_END


# -- full estimator -------------------------------------------------------
def test_estimator_builds_baby_state_from_timeline():
    settings = load_settings()
    est, log = make_estimator(settings)
    temporal = TemporalEngine(settings, EventLog(), "test")
    baby = None
    now = 0.0
    for p in timeline_stable_face(120):
        features, _ = temporal.update(p)
        est.attention_estimator.estimate(features, 0.9, p.timestamp)
        temporal.note_attention(0.7, p.timestamp)
        baby, evs = est.update(features, 0.9, interaction_duration=p.timestamp,
                               response_latency=None, activity_id=None,
                               now=p.timestamp)
        now = p.timestamp
    assert baby is not None
    assert baby.face_present is True
    assert baby.attention > 0.5
    assert 0.0 <= baby.attention_confidence <= 1.0
    assert baby.state in (BabyStateName.OBSERVING, BabyStateName.ENGAGED,
                          BabyStateName.EXPLORING, BabyStateName.RESPONDING)
    assert baby.current_orientation.value == "LOOKING_CENTER"


def test_estimator_no_face_state():
    settings = load_settings()
    est, log = make_estimator(settings)
    temporal = TemporalEngine(settings, EventLog(), "test")
    baby = None
    for p in timeline_no_face(60):
        features, _ = temporal.update(p)
        baby, evs = est.update(features, 0.0, 0.0, None, None, p.timestamp)
    assert baby.state == BabyStateName.NO_FACE
    assert baby.face_present is False
    assert baby.attention == 0.0


def test_attention_estimate_has_evidence_and_confidence():
    settings = load_settings()
    est, _ = make_estimator(settings)
    temporal = TemporalEngine(settings, EventLog(), "test")
    features = None
    for p in timeline_stable_face(30):
        features, _ = temporal.update(p)
    estimate = est.attention_estimator.estimate(features, detection_confidence=1.0,
                                                now=p.timestamp)
    assert 0.0 < estimate.value <= 1.0
    assert 0.0 < estimate.confidence <= 1.0
    assert "face_present" in estimate.evidence
    d = estimate.to_dict()
    assert set(d) == {"value", "confidence", "evidence"}


def test_low_detection_confidence_lowers_attention_confidence():
    settings = load_settings()
    est, _ = make_estimator(settings)
    temporal = TemporalEngine(settings, EventLog(), "test")
    for p in timeline_stable_face(30):
        features, _ = temporal.update(p)
    good = est.attention_estimator.estimate(features, 1.0, p.timestamp)
    est.attention_estimator.reset()
    for p in timeline_stable_face(30):
        features, _ = temporal.update(p)
    weak = est.attention_estimator.estimate(features, 0.3, p.timestamp)
    assert weak.confidence < good.confidence
