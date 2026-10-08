"""Head pose estimation (yaw / pitch / roll) from 6 facial landmarks.

Method: cv2.solvePnP (EPnP + iterative refinement) against a generic 3D head
model, standard in the literature (e.g. OpenCV head-pose tutorial).

Coordinate convention (documented, stable across BabyMotion):
  - The 3D model is y-up; the image is y-down, so the raw solvePnP rotation is
    conjugated by the upright reference Rx(180) = diag(1,-1,-1) before
    extracting Euler angles (otherwise a frontal face reads pitch ~= 180).
  - yaw   in [-180, 180] degrees: positive = face oriented toward IMAGE right
    (for a baby facing the camera, that is the baby turning toward their own
    left; the orientation labels below are from the VIEWER's perspective)
  - pitch in [-90, 90] degrees:  positive = looking down
  - roll  in [-180, 180] degrees: positive = clockwise in image (right ear down)

This is a FACE/HEAD ORIENTATION estimate, not calibrated gaze tracking.
Orientation labels (LOOKING_LEFT/RIGHT/...) describe image direction:
LOOKING_RIGHT = nose oriented toward the image right edge.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from .landmarks import CHIN, EYE_L_OUTER, EYE_R_OUTER, MOUTH_LEFT, MOUTH_RIGHT, NOSE_TIP, FaceLandmarks


class HeadOrientation(str, enum.Enum):
    LOOKING_LEFT = "LOOKING_LEFT"
    LOOKING_RIGHT = "LOOKING_RIGHT"
    LOOKING_CENTER = "LOOKING_CENTER"
    LOOKING_UP = "LOOKING_UP"
    LOOKING_DOWN = "LOOKING_DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float
    orientation: HeadOrientation
    confidence: float  # 0..1 (from reprojection error + landmark visibility)
    reprojection_error_px: float = 0.0

    def to_dict(self) -> dict:
        return {
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "orientation": self.orientation.value,
            "confidence": round(self.confidence, 3),
            "reprojection_error_px": round(self.reprojection_error_px, 2),
        }

    @classmethod
    def unknown(cls) -> "HeadPose":
        return cls(yaw=0.0, pitch=0.0, roll=0.0,
                   orientation=HeadOrientation.UNKNOWN, confidence=0.0)


# Generic 3D head model points (millimeters), OpenCV head-pose convention
# (y up, z away from face plane toward camera after solvePnP conventions).
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),            # nose tip
        (0.0, -330.0, -65.0),       # chin
        (-225.0, 170.0, -135.0),    # subject's right eye outer corner
        (225.0, 170.0, -135.0),     # subject's left eye outer corner
        (-150.0, -150.0, -125.0),   # subject's right mouth corner
        (150.0, -150.0, -125.0),    # subject's left mouth corner
    ],
    dtype=np.float64,
)
_LANDMARK_IDS = (NOSE_TIP, CHIN, EYE_R_OUTER, EYE_L_OUTER, MOUTH_LEFT, MOUTH_RIGHT)

_CENTER_THRESH = 18.0   # degrees: |yaw| below this = centered horizontally
_VERTICAL_THRESH = 18.0  # degrees: |pitch| below this = not up/down


_REFERENCE_UPRIGHT = np.diag([1.0, -1.0, -1.0])  # model y-up -> image y-down


def classify_orientation(yaw: float, pitch: float) -> HeadOrientation:
    """Coarse face/head orientation (NOT gaze), viewer/image perspective."""
    if abs(pitch) >= _VERTICAL_THRESH and abs(pitch) > abs(yaw):
        return HeadOrientation.LOOKING_UP if pitch < 0 else HeadOrientation.LOOKING_DOWN
    if abs(yaw) < _CENTER_THRESH:
        return HeadOrientation.LOOKING_CENTER
    return HeadOrientation.LOOKING_RIGHT if yaw > 0 else HeadOrientation.LOOKING_LEFT


class HeadPoseEstimator:
    def __init__(self, image_width: int = 1280, image_height: int = 720) -> None:
        self._focal = float(max(image_width, image_height))
        self._center = (image_width / 2.0, image_height / 2.0)
        self._camera_matrix = np.array(
            [
                [self._focal, 0.0, self._center[0]],
                [0.0, self._focal, self._center[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._dist = np.zeros((4, 1), dtype=np.float64)

    def estimate(self, landmarks: FaceLandmarks, frame_w: int, frame_h: int,
                 min_visibility: float = 0.5) -> HeadPose:
        import cv2

        if frame_w != self._center[0] * 2 or frame_h != self._center[1] * 2:
            self.__init__(frame_w, frame_h)  # re-intrinsics on resolution change

        visible = [landmarks.visibility[i] >= min_visibility for i in _LANDMARK_IDS]
        if not all(visible):
            return HeadPose.unknown()

        image_points = np.array([landmarks.pixels[i] for i in _LANDMARK_IDS], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS, image_points, self._camera_matrix, self._dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return HeadPose.unknown()

        projected, _ = cv2.projectPoints(_MODEL_POINTS, rvec, tvec,
                                         self._camera_matrix, self._dist)
        err = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)))

        rot, _ = cv2.Rodrigues(rvec)
        # Express the rotation relative to the upright reference so a frontal
        # face yields angles near 0 instead of ~180 (see module docstring).
        rot = _REFERENCE_UPRIGHT.T @ rot
        sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))   # nod down/up
            yaw = np.degrees(np.arctan2(-rot[2, 0], sy))            # turn left/right
            roll = np.degrees(np.arctan2(-rot[1, 0], rot[0, 0]))    # tilt
        else:
            pitch = np.degrees(np.arctan2(-rot[1, 2], rot[1, 1]))
            yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
            roll = 0.0

        # Confidence from RELATIVE reprojection error (fit quality vs face size)
        # plus a penalty for tiny faces where pose is ill-conditioned.
        face_w = float(landmarks.pixels[:, 0].max() - landmarks.pixels[:, 0].min())
        rel_err = err / max(1.0, face_w)
        confidence = float(np.clip(1.0 - 5.0 * rel_err, 0.0, 1.0))
        if face_w < 80.0:
            confidence *= float(np.clip(face_w / 80.0, 0.2, 1.0))

        orientation = classify_orientation(yaw, pitch) if confidence >= 0.3 else HeadOrientation.UNKNOWN
        return HeadPose(yaw=float(yaw), pitch=float(pitch), roll=float(roll),
                        orientation=orientation, confidence=confidence,
                        reprojection_error_px=err)
