import time

class FrameScheduler:
    def __init__(self, target_fps: int):
        self.target_fps = target_fps
        self.target_interval = 1.0 / target_fps
        self._next_time = time.perf_counter() + self.target_interval

    def wait_for_next_frame(self):
        now = time.perf_counter()
        sleep_time = self._next_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)  # Sleep to maintain the target FPS
        now = time.perf_counter()
        # Advance to the next tick, skipping missed frames if we lag.
        while self._next_time <= now:
            self._next_time += self.target_interval

    def adjust_fps(self, current_fps: int):
        """Dynamically adjust target interval based on real FPS."""
        self.target_fps = current_fps
        self.target_interval = 1.0 / self.target_fps
        self._next_time = time.perf_counter() + self.target_interval
        print(f"Adjusted target FPS to {self.target_fps}")
