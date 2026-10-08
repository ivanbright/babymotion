"""Policy tests: determinism and expected action for known states."""

from app.config.settings import load_settings
from app.interaction.actions import TwinActionType
from app.interaction.activities import build_registry
from app.interaction.policy import RuleBasedPolicy
from app.metrics.events import Event, EventLog, EventType
from app.state.baby_state import BabyState, BabyStateName


def make_policy(settings=None):
    return RuleBasedPolicy(settings or load_settings())


def engaged_state(**over) -> BabyState:
    base = dict(
        face_present=True, attention=0.85, attention_confidence=0.8,
        movement=0.3, facial_activity=0.72, head_motion=0.2, mouth_open=0.5,
        state=BabyStateName.ENGAGED, current_orientation="LOOKING_CENTER",
    )
    base.update(over)
    from app.perception.head_pose import HeadOrientation
    ori = base.pop("current_orientation")
    s = BabyState(**base)
    s.current_orientation = HeadOrientation(ori)
    return s


def test_spec_example_high_engagement_exaggerates():
    policy = make_policy()
    action = policy.decide(engaged_state(), None, history=[], now=10.0)
    assert action.type.value == "EXAGGERATE"
    assert "engagement_high" in action.reason
    assert "face_present" in action.reason
    assert "mouth_motion_detected" in action.reason


def test_no_face_pauses():
    policy = make_policy()
    action = policy.decide(engaged_state(face_present=False,
                                         state=BabyStateName.NO_FACE),
                           None, [], now=1.0)
    assert action.type == TwinActionType.PAUSE
    assert "face_absent" in action.reason


def test_low_attention_pauses():
    policy = make_policy()
    action = policy.decide(engaged_state(attention=0.2, facial_activity=0.0,
                                         mouth_open=0.05, head_motion=0.0),
                           None, [], now=1.0)
    assert action.type == TwinActionType.PAUSE
    assert any("attention_low" in r for r in action.reason)


def test_low_confidence_blocks_strong_response():
    policy = make_policy()
    action = policy.decide(engaged_state(attention=0.5, attention_confidence=0.2),
                           None, [], now=1.0)
    assert action.type == TwinActionType.PAUSE
    assert any("confidence" in r for r in action.reason)


def test_deterministic_same_input_same_output():
    policy = make_policy()
    history = [
        Event(event=EventType.MOUTH_OPENED, timestamp=9.5, payload={}),
        Event(event=EventType.HEAD_MOVED, timestamp=9.8, payload={}),
    ]
    state = engaged_state()
    a1 = policy.decide(state, None, history, now=10.0)
    a2 = policy.decide(state, None, history, now=10.0)
    assert a1.to_dict() == a2.to_dict()


def test_scheduled_stimulus_takes_precedence():
    policy = make_policy()
    from app.interaction.activities import Stimulus
    from app.interaction.actions import TwinAction
    stim = Stimulus(stimulus_id="mirror_stim_000",
                    action=TwinAction(type=TwinActionType.LOOK_AT_BABY,
                                      target="head", intensity=0.4,
                                      duration_ms=2000,
                                      stimulus_id="mirror_stim_000"),
                    start_s=5.0, duration_s=2.0)
    action = policy.decide(engaged_state(), None, [], now=6.0, scheduled=stim)
    assert action.type == TwinActionType.LOOK_AT_BABY
    assert action.stimulus_id == "mirror_stim_000"
    assert "scheduled_stimulus" in action.reason


def test_attention_drop_requests_attention():
    policy = make_policy()
    history = [Event(event=EventType.ATTENTION_DECREASED, timestamp=9.0, payload={})]
    action = policy.decide(engaged_state(attention=0.45), None, history, now=10.0)
    assert action.type == TwinActionType.ATTRACT_ATTENTION


def test_repeated_responses_introduce_novelty():
    policy = make_policy()
    history = [
        Event(event=EventType.BABY_RESPONSE, timestamp=8.0, payload={}),
        Event(event=EventType.BABY_RESPONSE, timestamp=9.5, payload={}),
    ]
    action = policy.decide(engaged_state(), None, history, now=10.0)
    assert action.type == TwinActionType.INTRODUCE_NOVELTY
    assert "repeated_responses" in action.reason


def test_cause_effect_activity_triggers_smile_on_mouth_open():
    policy = make_policy()
    registry = build_registry(load_settings())
    from app.interaction.activities import ActivityInstance
    instance = ActivityInstance(registry["cause_effect"], started_at=0.0)
    history = [Event(event=EventType.MOUTH_OPENED, timestamp=9.2, payload={})]
    action = policy.decide(engaged_state(), instance, history, now=10.0)
    assert action.type == TwinActionType.SMILE
    assert "cause_effect_triggered" in action.reason


def test_reasons_are_serializable():
    policy = make_policy()
    action = policy.decide(engaged_state(), None, [], now=1.0)
    d = action.to_dict()
    assert isinstance(d["reason"], list)
    import json
    json.dumps(d)  # must not raise
    assert d["action"] in {t.value for t in TwinActionType}
