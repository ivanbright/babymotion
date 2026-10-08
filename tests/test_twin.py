"""Digital twin tests: exaggeration mapping, action overrides, renderer."""

import cv2
import numpy as np
import pytest

from app.interaction.actions import TwinAction, TwinActionType
from app.twin.controller import TwinController
from app.twin.exaggeration import ExaggerationEngine
from app.twin.renderer import MeshTwinRenderer
from app.twin.twin_state import TwinState
from synthetic import make_baby, make_features, make_landmarks


def test_twin_mouth_is_amplified(settings):
    eng = ExaggerationEngine(settings.twin)
    twin = eng.map_to_twin(make_features(mouth_open=0.6), make_baby())
    # neutral 0.10 + 1.5 * (0.6 - 0.1) = 0.85
    assert twin.mouth_open == pytest.approx(0.1 + 1.5 * (0.6 - 0.1), abs=1e-6)


def test_twin_head_rotation_amplified_and_clamped(settings):
    eng = ExaggerationEngine(settings.twin)
    # within limits: pure gain
    twin = eng.map_to_twin(make_features(head_yaw=20.0), make_baby())
    assert twin.head_yaw == pytest.approx(20.0 * 1.3)
    # beyond limits: clamped to the configured safe range
    hi = settings.twin.limits["head_yaw"][1]
    lo = settings.twin.limits["head_yaw"][0]
    assert eng.map_to_twin(make_features(head_yaw=60.0), make_baby()).head_yaw == hi
    assert eng.map_to_twin(make_features(head_yaw=-60.0), make_baby()).head_yaw == lo


def test_twin_neutral_when_baby_neutral(settings):
    eng = ExaggerationEngine(settings.twin)
    twin = eng.map_to_twin(make_features(mouth_open=0.1, eye_open=0.8,
                                         facial_motion=0.0), make_baby(facial_activity=0.0))
    assert twin.mouth_open == pytest.approx(settings.twin.neutral.mouth_open)
    assert twin.expression_intensity == pytest.approx(0.0, abs=1e-6)
    assert twin.eye_scale == pytest.approx(1.0, abs=1e-6)


def test_controller_smooths_toward_target(settings):
    ctrl = TwinController(ExaggerationEngine(settings.twin),
                          response_speed_ms=250)
    twin = None
    for i in range(120):
        twin = ctrl.update(make_features(mouth_open=0.9, timestamp=i / 30.0),
                           make_baby(),
                           TwinAction(type=TwinActionType.MIRROR),
                           dt=1 / 30.0)
    # after 4 s the smoothed value is essentially at target (0.1 + 1.5*0.8)
    assert twin.mouth_open > 0.8


def test_controller_action_overrides(settings):
    ctrl = TwinController(ExaggerationEngine(settings.twin), response_speed_ms=50)
    for _ in range(200):
        state = ctrl.update(make_features(mouth_open=0.1),
                            make_baby(facial_activity=0.0),
                            TwinAction(type=TwinActionType.SMILE, intensity=1.0),
                            dt=1 / 30.0)
    assert state.smile > 0.9
    assert state.action == "SMILE"

    ctrl2 = TwinController(ExaggerationEngine(settings.twin), response_speed_ms=50)
    for _ in range(200):
        state = ctrl2.update(make_features(head_yaw=0.0), make_baby(),
                             TwinAction(type=TwinActionType.TURN_HEAD,
                                        intensity=1.0, direction="left"),
                             dt=1 / 30.0)
    assert state.head_yaw < -20.0


def test_controller_pause_moves_toward_neutral(settings):
    ctrl = TwinController(ExaggerationEngine(settings.twin), response_speed_ms=50)
    for _ in range(200):
        state = ctrl.update(make_features(mouth_open=0.9),
                            make_baby(facial_activity=0.9),
                            TwinAction(type=TwinActionType.PAUSE),
                            dt=1 / 30.0)
    assert state.mouth_open == pytest.approx(settings.twin.neutral.mouth_open,
                                             abs=0.02)


def test_twin_state_serializable_roundtrip():
    s = TwinState(mouth_open=0.5, smile=0.4, head_yaw=10.0, eye_scale=1.1,
                  expression_intensity=0.6, action="EXAGGERATE")
    d = s.to_dict()
    import json
    json.dumps(d)
    s2 = TwinState.from_dict(d)
    assert s2.mouth_open == d["mouth_open"]
    assert s2.action == "EXAGGERATE"


def _flat_frame(w=1280, h=720, value=120):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_mesh_renderer_produces_panel(settings):
    renderer = MeshTwinRenderer(settings)
    landmarks = make_landmarks()
    landmarks.timestamp = 0.0
    panel = renderer.render(_flat_frame(), landmarks,
                            make_features(mouth_open=0.3), TwinState(),
                            facial_activity=0.1)
    assert panel.shape == (settings.twin.mesh.panel_height,
                           panel.shape[1], 3)
    assert panel.dtype == np.uint8
    # shading/texture variation: not a flat block
    assert float(panel.std()) > 1.0


def test_mesh_renderer_grows_the_mouth(settings):
    renderer = MeshTwinRenderer(settings)
    h, w = 720, 1280
    yy, xx = np.mgrid[0:h, 0:w]
    frame = np.stack([xx % 256, yy % 256, (xx // 2 + yy) % 256],
                     axis=-1).astype(np.uint8)
    twin_state = TwinState(action="OPEN_MOUTH", action_intensity=1.0)

    def render_seq(mouth, n=45):
        panel = None
        for i in range(n):
            lm = make_landmarks(mouth_open=mouth)
            lm.timestamp = i / 30.0
            panel = renderer.render(frame, lm, make_features(mouth_open=mouth),
                                    twin_state, facial_activity=0.0)
        return panel

    closed = render_seq(0.05)
    # template bootstrap happened above, so the open-mouth sequence exaggerates
    # relative to the captured neutral
    open_panel = render_seq(0.9)
    # exaggerated opening makes the face taller -> narrower panel at equal height
    assert open_panel.shape[1] < closed.shape[1]
    resized = cv2.resize(open_panel, (closed.shape[1], closed.shape[0]))
    assert np.abs(resized.astype(int) - closed.astype(int)).mean() > 2.0


def test_mesh_renderer_placeholder_without_face(settings):
    renderer = MeshTwinRenderer(settings)
    panel = renderer.render(_flat_frame(), None, None, TwinState())
    assert panel.shape[0] == settings.twin.mesh.panel_height
    assert panel.dtype == np.uint8
