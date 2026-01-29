import threading
import numpy as np
from utils.emotion_classification import EmotionRecognizer
import time


class EmotionWorker(threading.Thread):
    def __init__(self, model_path: str, process_every_n: int = 4):
        super().__init__(daemon=True)
        self.recognizer = EmotionRecognizer(model_path)
        self.process_every_n = process_every_n
        self._lock = threading.Lock()
        self._frame_counter = 0

        self.latest_result = None
        self._frame = None
        self._running = True

    def update_frame(self, frame_rgb: np.ndarray):
        """Thread-safe way to update the current frame."""
        with self._lock:
            self._frame_counter += 1
            if self._frame_counter % self.process_every_n == 0:
                self._frame = frame_rgb

    def get_latest_result(self):
        """Access the most recent emotion recognition result."""
        with self._lock:
            return self.latest_result

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            frame_to_process = None # Use a different variable name to avoid confusion
            with self._lock:
                if self._frame is not None:
                    frame_to_process = self._frame
                    self._frame = None  # Consume the frame

            if frame_to_process is not None:
                # Heavy operation done outside the lock
                result = self.recognizer.predict_emotions(frame_to_process)
                with self._lock:
                    self.latest_result = result
            else:
                if self._running:
                    time.sleep(0.01)
