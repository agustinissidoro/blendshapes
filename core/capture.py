import cv2
import threading
import queue
import time


class VideoCaptureThread:
    def __init__(self, src=0, api_preference=cv2.CAP_ANY, flip_image=False,
                 width: int = None, height: int = None, fps: int = None):
        self.cap = cv2.VideoCapture(src, api_preference)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source {src} with backend {api_preference}")
        self.q = queue.Queue(maxsize=1)
        self.running = False
        self._thread = None
        self.flip = flip_image

        if width is not None and height is not None:
            self._configure_resolution(width, height, fps)

    def _configure_resolution(self, desired_w: int, desired_h: int, desired_fps: int = None):
        """Try to set the desired resolution and fps, falling back gracefully."""
        candidates = []
        if desired_fps:
            candidates.append((desired_w, desired_h, desired_fps))
        candidates.append((desired_w, desired_h, 30))
        if desired_w > 1280 or desired_h > 720:
            if desired_fps and desired_fps > 30:
                candidates.append((1280, 720, desired_fps))
            candidates.append((1280, 720, 30))

        for w, h, f in candidates:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self.cap.set(cv2.CAP_PROP_FPS, f)

            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            if actual_w == w and actual_h == h:
                print(f"[VideoCaptureThread] Configured: {actual_w}x{actual_h}@{actual_fps}fps")
                return
            print(f"[VideoCaptureThread] Tried {w}x{h}@{f}fps, got {actual_w}x{actual_h}@{actual_fps}fps. Trying fallback...")

        # Nothing worked — report whatever the camera decided
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"[VideoCaptureThread] Using camera default: {actual_w}x{actual_h}@{actual_fps}fps")

    def start(self):
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if not self.running:
                    break
                time.sleep(0.02)
                continue

            if not self.q.empty():
                try:
                    self.q.get_nowait()
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
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                print("Warning: Video capture thread did not stop gracefully.")
        self.cap.release()
        self._thread = None

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()
