import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class LandmarkerSnapshot:
    blendshapes: Any
    landmarks: Any
    transform_matrix: Any


class LandmarkerState:
    def __init__(self):
        self._blendshapes = None
        self._landmarks = None
        self._transform_matrix = None
        self._lock = threading.Lock()

    def update_from_result(self, result, _image, _timestamp_ms):
        with self._lock:
            self._blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
            self._landmarks = result.face_landmarks[0] if result.face_landmarks else None
            self._transform_matrix = (
                result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None
            )

    def snapshot(self) -> LandmarkerSnapshot:
        with self._lock:
            return LandmarkerSnapshot(
                blendshapes=self._blendshapes,
                landmarks=self._landmarks,
                transform_matrix=self._transform_matrix,
            )
