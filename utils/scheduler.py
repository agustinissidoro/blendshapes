import time

class FrameScheduler:
    def __init__(self, target_fps: int):
        self.target_fps = target_fps
        self.target_interval = 1.0 / target_fps
        self.last_time = time.time()

    def wait_for_next_frame(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_time
        sleep_time = self.target_interval - elapsed_time
        
        if sleep_time > 0:
            time.sleep(sleep_time)  # Sleep to maintain the target FPS
        else:
            # If we lag, we skip the sleep (drop the frame)
            #print("Warning: Dropped frame due to lag!")
            pass
        
        # Update last_time for the next frame, regardless of sleep
        self.last_time = time.time()

    def adjust_fps(self, current_fps: int):
        """Dynamically adjust target interval based on real FPS."""
        self.target_fps = current_fps
        self.target_interval = 1.0 / self.target_fps
        print(f"Adjusted target FPS to {self.target_fps}")
