"""Perception pipeline: Frame -> PerceptionResult.

Runs MediaPipe FaceLandmarker (primary) and optionally the Haar cascade
(fallback / diagnostics). No identity recognition is performed anywhere.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..config.settings import Settings, resolve_landmark_model
from ..input.frame import Frame
from .audio_features import AudioFeatureExtractor
from .face_detector import CascadeFaceDetector, FaceDetection, bbox_from_landmarks, select_primary
from .facial_features import FacialFeatureExtractor
from .head_pose import HeadPoseEstimator
from .landmarks import MediaPipeLandmarker
from .perception_result import PerceptionResult


class PerceptionPipeline:
    def __init__(self, settings: Settings, model_path: Path | None = None) -> None:
        self.settings = settings
        path = model_path or resolve_landmark_model(settings)
        self.landmarker = MediaPipeLandmarker(
            model_path=path,
            num_faces=settings.perception.max_faces,
            min_detection_confidence=settings.perception.min_face_confidence,
        )
        self.cascade = (
            CascadeFaceDetector()
            if settings.perception.use_cascade_fallback
            else None
        )
        self.head_pose = HeadPoseEstimator()
        self.features = FacialFeatureExtractor()
        self.audio = AudioFeatureExtractor()

    def process(self, frame: Frame) -> PerceptionResult:
        t0 = time.perf_counter()
        landmarks = self.landmarker.detect(frame.image, frame.timestamp)
        blendshapes = dict(self.landmarker.last_blendshapes)
        detection: Optional[FaceDetection] = None
        detector_name = "none"

        if landmarks is not None:
            detection = FaceDetection(
                bbox=bbox_from_landmarks(landmarks.pixels, frame.width, frame.height,
                                         padding=self.settings.perception.roi_padding),
                confidence=1.0,
                timestamp=frame.timestamp,
            )
            detector_name = "landmark"
            # Secondary cascade run only as a diagnostic cross-check.
            if self.cascade is not None:
                cascade_hits = self.cascade.detect(frame.image, frame.timestamp)
                primary = select_primary(cascade_hits, frame.width, frame.height)
                if primary is not None:
                    detection.confidence = min(1.0, 0.75 + 0.25 * primary.confidence)

        if landmarks is None and self.cascade is not None:
            # Graceful degradation: face likely present but landmark tracking
            # lost -> report detection without geometry.
            hits = self.cascade.detect(frame.image, frame.timestamp)
            primary = select_primary(hits, frame.width, frame.height)
            if primary is not None and primary.confidence >= self.settings.perception.min_face_confidence:
                detection = primary
                detector_name = "cascade"

        face_present = detection is not None
        head_pose = None
        features = None
        if landmarks is not None:
            head_pose = self.head_pose.estimate(
                landmarks, frame.width, frame.height,
                min_visibility=self.settings.perception.min_landmark_visibility,
            )
            features = self.features.extract(landmarks, frame.width, frame.height)

        result = PerceptionResult(
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            face_present=face_present,
            detection=detection,
            landmarks=landmarks,
            head_pose=head_pose,
            features=features,
            audio=self.audio.update([]),  # mic values are injected by the engine
            blendshapes=blendshapes,
            detector=detector_name,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return result

    def close(self) -> None:
        self.landmarker.close()
