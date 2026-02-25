import cv2
import threading
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
FaceLandmarkerResult = vision.FaceLandmarkerResult
VisionRunningMode = vision.RunningMode

class FaceLandmarkerProcessor:
    def __init__(self, model_path, result_callback):
        self.result_callback = result_callback
        self.lock = threading.Lock()
        self.model_path = model_path
        self.landmarker = None
        self._init_landmarker()

    def _init_landmarker(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._on_result,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def _on_result(self, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if self.result_callback:
            self.result_callback(result, output_image, timestamp_ms)

    def process(self, frame: np.ndarray, timestamp_ms: int):
        #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        with self.lock:
            self.landmarker.detect_async(mp_image, timestamp_ms)

    def close(self):
        if self.landmarker:
            self.landmarker.close()
