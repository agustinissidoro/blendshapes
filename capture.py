import cv2
import threading
import queue
import time

class VideoCaptureThread:
    def __init__(self, src=0, api_preference=cv2.CAP_ANY, flip_image=False):
        self.cap = cv2.VideoCapture(src, api_preference)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source {src} with backend {api_preference}")
        self.q = queue.Queue(maxsize=1)
        self.running = False
        self._thread = None
        self.flip = flip_image


    def start(self):
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if not self.running:  # Double-check
                    break
                time.sleep(0.02)  # Longer sleep on read failure
                continue

            # Non-blocking queue update
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # Discard the oldest frame
                except queue.Empty:
                    pass
            if self.flip:
                frame = cv2.flip(frame, 1)             
            self.q.put(frame)

    def read(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        # Clear queue to unblock reads
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        # Wait for thread to properly finish
        if self._thread is not None:
            self._thread.join(timeout=1.0)  # Wait for the thread to finish
            if self._thread.is_alive():
                print("Warning: Video capture thread did not stop gracefully.")
        # Release resources
        self.cap.release()
        self._thread = None

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()
