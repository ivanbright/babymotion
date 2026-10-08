"""Activity tests: deterministic schedules, one-shot firing, termination."""

import json

from app.config.settings import load_settings
from app.interaction.activities import ActivityInstance, ActivityPhase, build_registry
from app.interaction.actions import TwinActionType


def test_registry_has_five_activities():
    reg = build_registry(load_settings())
    assert set(reg) == {"mirror", "exaggerated_mirror", "follow",
                        "cause_effect", "repetition"}


def test_schedules_are_deterministic_and_monotonic():
    reg = build_registry(load_settings())
    for act in reg.values():
        starts = [s.start_s for s in act.stimuli]
        assert starts == sorted(starts)
        ids = [s.stimulus_id for s in act.stimuli]
        assert len(ids) == len(set(ids))
        for s in act.stimuli:
            assert 0 <= s.start_s < act.duration
            assert s.duration_s > 0
            assert s.action.duration_ms > 0
        # rebuilding gives the identical schedule
    again = build_registry(load_settings())
    for key in reg:
        assert reg[key].to_dict() == again[key].to_dict()


def test_activity_dict_is_json_serializable():
    reg = build_registry(load_settings())
    for act in reg.values():
        json.dumps(act.to_dict())
        d = act.to_dict()
        assert d["expected_observations"] and d["termination_conditions"]


def test_follow_turns_alternate_direction():
    reg = build_registry(load_settings())
    directions = [s.action.direction for s in reg["follow"].stimuli]
    assert len(directions) >= 4
    for a, b in zip(directions, directions[1:]):
        assert a != b


def test_repetition_uses_identical_stimuli():
    reg = build_registry(load_settings())
    stimuli = reg["repetition"].stimuli
    assert len(stimuli) >= 3
    ref = stimuli[0].action
    for s in stimuli:
        assert s.action.type == ref.type
        assert s.action.intensity == ref.intensity
        assert s.action.direction == ref.direction
        assert s.action.duration_ms == ref.duration_ms


# -- instance runtime -----------------------------------------------------
def test_due_stimuli_fire_exactly_once():
    act = build_registry(load_settings())["mirror"]
    inst = ActivityInstance(act, started_at=100.0)
    # jump far past every schedule: all fire once
    due = inst.due_stimuli(now=100.0 + 30.0)
    assert len(due) == len(act.stimuli)
    assert inst.due_stimuli(now=100.0 + 30.0) == []
    assert sorted(inst.fired) == sorted(s.stimulus_id for s in act.stimuli)


def test_due_stimuli_respect_schedule_order():
    act = build_registry(load_settings())["mirror"]
    inst = ActivityInstance(act, started_at=0.0)
    # jump only past the first scheduled stimulus (5.0 s)
    first = inst.due_stimuli(now=5.5)
    assert len(first) == 1
    assert first[0].start_s == 5.0
    # next window (15 s) fires only the second
    second = inst.due_stimuli(now=16.0)
    assert len(second) == 1
    assert second[0].start_s == 15.0


def test_active_stimulus_window():
    act = build_registry(load_settings())["mirror"]
    inst = ActivityInstance(act, started_at=0.0)
    assert inst.active_stimulus(4.0) is None
    active = inst.active_stimulus(6.0)
    assert active is not None and active.start_s == 5.0
    assert inst.active_stimulus(7.5) is None  # 2 s stimulus ended


def test_duration_termination_and_progress():
    act = build_registry(load_settings())["mirror"]
    inst = ActivityInstance(act, started_at=0.0)
    assert not inst.is_duration_over(now=act.duration - 1.0)
    assert inst.is_duration_over(now=act.duration)
    assert inst.progress(now=act.duration / 2.0) == 0.5
    assert inst.running
    inst.terminate(now=act.duration, reason="duration_elapsed",
                   phase=ActivityPhase.COMPLETED)
    assert not inst.running
    assert inst.end_reason == "duration_elapsed"
    d = inst.to_dict(now=act.duration)
    assert d["phase"] == "COMPLETED"
    assert d["progress"] == 1.0
    json.dumps(d)


def test_stimulus_action_types_match_protocol():
    reg = build_registry(load_settings())
    assert all(s.action.type == TwinActionType.LOOK_AT_BABY
               for s in reg["mirror"].stimuli)
    assert all(s.action.type == TwinActionType.EXAGGERATE
               for s in reg["exaggerated_mirror"].stimuli)
    assert all(s.action.type == TwinActionType.TURN_HEAD
               for s in reg["follow"].stimuli)
    assert all(s.action.type == TwinActionType.REPEAT
               for s in reg["repetition"].stimuli)
