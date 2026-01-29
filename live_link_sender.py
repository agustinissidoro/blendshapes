import socket
import threading
from typing import List, Set, Optional, Tuple
import time
import numpy as np
from pylivelinkface import PyLiveLinkFace, FaceBlendShape
from mediapipe.tasks.python.components.containers.category import Category
from utils.scheduler import FrameScheduler


class LiveLinkSender(threading.Thread):
    def __init__(self, ip: str, port: int, swap_left_right: bool = False, target_fps: int = 60, pair_eyelids: bool = True):
        super().__init__()
        self.ip = ip
        self.port = port
        self.swap_left_right = swap_left_right
        self.target_fps = target_fps
        self.pair_eyelids = pair_eyelids
        self.running = False
        self._face_normal = PyLiveLinkFace(fps=target_fps)  # Normal face state
        self._face_override = PyLiveLinkFace(fps=target_fps)  # Neutral/random override state
        # State for facial blendshape reset logic (if used by update_blendshapes)
        self._latest_blendshapes: List[Category] = []
        self._previous_facial_keys: Set[str] = set()
        self._unknown_blendshape_names: Set[str] = set()
        self._dirty_lock = threading.Lock()
        self._dirty = False
        self._mode_lock = threading.Lock()
        self._mode = "normal"  # normal | neutral | random
        self._random_rate_hz = 10.0
        self._last_random_time = 0.0
        self._override_lock = threading.Lock()
        self._blink_right_on = False
        self._tongue_out_on = False

        # Socket setup
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.connect((self.ip, self.port))
            print(f"[LiveLinkSender] Socket targeting {ip}:{port}")
        except socket.error as e:
            print(f"[LiveLinkSender] Socket connect error: {e}")
            self._socket = None # Mark as unusable

    def run(self):
        if not self._socket:
            print("[LiveLinkSender] Cannot run, socket not connected.")
            return
        self.running = True
        print("[LiveLinkSender] Sender thread started.")
        scheduler = FrameScheduler(self.target_fps)
        try:
            while self.running:
                scheduler.wait_for_next_frame()
                with self._mode_lock:
                    mode = self._mode
                    random_rate_hz = self._random_rate_hz

                if mode == "normal":
                    with self._dirty_lock:
                        should_send = self._dirty
                        self._dirty = False
                    if should_send:
                        self._send_data(self._face_normal) # Send only when updated
                elif mode == "neutral":
                    self._set_neutral_pose(self._face_override)
                    self._send_data(self._face_override)
                elif mode == "random":
                    now = time.perf_counter()
                    if now - self._last_random_time >= (1.0 / max(random_rate_hz, 0.1)):
                        self._last_random_time = now
                        self._fill_random_pose(self._face_override)
                    self._send_data(self._face_override)
        except socket.error as se:
             print(f"[LiveLinkSender] Socket error: {se}")  # <-- Debug print
             self.running = False
        except Exception as e:
            print(f"[LiveLinkSender] Error in run loop: {e}")  # <-- Debug print
            self.running = False
        finally:
            if self._socket:
                self._socket.close()
            print("[LiveLinkSender] Sender thread stopped.")

    def stop(self):
        if self.running:
             print("[LiveLinkSender] Stopping sender...")
        self.running = False

    def set_mode(self, mode: str):
        mode = mode.lower().strip()
        if mode not in ("normal", "neutral", "random"):
            print(f"[LiveLinkSender] Unknown mode '{mode}', keeping '{self._mode}'.")
            return
        with self._mode_lock:
            self._mode = mode
        if mode == "normal":
            with self._dirty_lock:
                self._dirty = True

    def set_random_rate(self, rate_hz: float):
        try:
            rate = float(rate_hz)
        except (TypeError, ValueError):
            return
        with self._mode_lock:
            self._random_rate_hz = max(0.1, rate)

    def trigger_blink_right(self, duration_s: float = 0.15):
        with self._override_lock:
            self._blink_right_on = True

    def trigger_tongue_out(self, duration_s: float = 0.6):
        with self._override_lock:
            self._tongue_out_on = True

    def toggle_blink_right(self):
        with self._override_lock:
            self._blink_right_on = not self._blink_right_on

    def toggle_tongue_out(self):
        with self._override_lock:
            self._tongue_out_on = not self._tongue_out_on

    def set_blink_right(self, enabled: bool):
        with self._override_lock:
            self._blink_right_on = bool(enabled)

    def set_tongue_out(self, enabled: bool):
        with self._override_lock:
            self._tongue_out_on = bool(enabled)

    def update_blendshapes(self, blendshapes: Optional[List[Category]]):
        """Updates the facial expression blendshapes (indices 0-51) in the face state."""
        if not self.running: return
        if blendshapes is None:
            return

        current_keys = set()
        blink_left = None
        blink_right = None
        if blendshapes:
            for category in blendshapes:
                name = category.category_name
                if not name: continue
                if self.swap_left_right:
                    name = self._swap_left_right_name(name)
                enum_key_name = name[0].upper() + name[1:]
                try:
                    enum_key = FaceBlendShape[enum_key_name]
                    # Only update facial blendshapes (0-51) here
                    if 0 <= enum_key.value <= 51:
                        score = float(category.score)
                        self._face_normal.set_blendshape(enum_key, score, no_filter=True)
                        current_keys.add(enum_key_name)
                        if enum_key == FaceBlendShape.EyeBlinkLeft:
                            blink_left = score
                        elif enum_key == FaceBlendShape.EyeBlinkRight:
                            blink_right = score
                except KeyError:
                    if enum_key_name not in self._unknown_blendshape_names:
                        self._unknown_blendshape_names.add(enum_key_name)
                        print(f"[LiveLinkSender] Warning: unknown blendshape '{enum_key_name}'")
                    continue # Ignore unmapped keys
                except ValueError: continue # Ignore invalid scores

        # Reset only facial keys previously set by this method
        keys_to_reset = self._previous_facial_keys - current_keys
        for prev_key in keys_to_reset:
             try:
                enum_key = FaceBlendShape[prev_key]
                if 0 <= enum_key.value <= 51:
                      self._face_normal.set_blendshape(enum_key, 0.0, no_filter=True)
             except KeyError: continue

        # Optional eyelid pairing (avoid if we are forcing a right-eye blink)
        if self.pair_eyelids and blink_left is not None and blink_right is not None:
            with self._override_lock:
                blink_right_forced = self._blink_right_on
            if not blink_right_forced:
                avg_blink = (blink_left + blink_right) * 0.5
                self._face_normal.set_blendshape(FaceBlendShape.EyeBlinkLeft, avg_blink, no_filter=True)
                self._face_normal.set_blendshape(FaceBlendShape.EyeBlinkRight, avg_blink, no_filter=True)

        self._previous_facial_keys = current_keys
        with self._dirty_lock:
            self._dirty = True

    def _swap_left_right_name(self, name: str) -> str:
        if name.endswith('Left'):
            return name[:-4] + 'Right'
        if name.endswith('Right'):
            return name[:-5] + 'Left'
        return name

    # --- THIS IS THE CORRECTED, SIMPLIFIED VERSION ---
    def update_head_pose(self, final_pose_angles: Optional[Tuple[float, float, float]]):
        """Updates face state with final processed & normalized head pose angles."""
        if not self.running: return

        # Use 0.0 as default if input tuple is None
        yaw_final, pitch_final, roll_final = final_pose_angles if final_pose_angles else (0.0, 0.0, 0.0)

        # Directly update face object state with the final values
        # Input should already be clamped [-1, 1] by HeadPoseProcessor
        # Add float conversion and clipping as a safety measure
        try:
            yaw_final_safe = np.clip(float(yaw_final), -1.0, 1.0)
            pitch_final_safe = np.clip(float(pitch_final), -1.0, 1.0)
            roll_final_safe = np.clip(float(roll_final), -1.0, 1.0)
        except (ValueError, TypeError):
            yaw_final_safe, pitch_final_safe, roll_final_safe = 0.0, 0.0, 0.0


        self._face_normal.set_blendshape(FaceBlendShape.HeadYaw, yaw_final_safe, no_filter=True)
        self._face_normal.set_blendshape(FaceBlendShape.HeadPitch, pitch_final_safe, no_filter=True)
        self._face_normal.set_blendshape(FaceBlendShape.HeadRoll, roll_final_safe, no_filter=True)
        with self._dirty_lock:
            self._dirty = True
    # --- END OF METHOD ---

    def _send_data(self, face: PyLiveLinkFace):
        """Encodes the provided face state and sends it over UDP."""
        if not self.running or not self._socket:
            return
        try:
            overrides = self._apply_transient_overrides(face)
            payload = face.encode()
            self._socket.sendall(payload)
            if overrides:
                self._restore_overrides(face, overrides)
        except socket.error as se:
            print(f"[LiveLinkSender] Socket error in _send_data: {se}")  # <-- Add this
            self.running = False
        except Exception as e:
            print(f"[LiveLinkSender] Error encoding/sending data: {e}")  # <-- Already present
            self.running = False


    # Keep this method name if run() calls it
    def _send_blendshapes(self):
         self._send_data(self._face_normal)

    def _set_neutral_pose(self, face: PyLiveLinkFace):
        for i in range(61):
            face.set_blendshape(FaceBlendShape(i), 0.0, no_filter=True)

    def _fill_random_pose(self, face: PyLiveLinkFace):
        for i in range(61):
            if i >= FaceBlendShape.HeadYaw.value:
                val = float(np.random.uniform(-1.0, 1.0))
            else:
                val = float(np.random.uniform(0.0, 1.0))
            face.set_blendshape(FaceBlendShape(i), val, no_filter=True)

    def _apply_transient_overrides(self, face: PyLiveLinkFace):
        overrides = []
        with self._override_lock:
            blink_right_active = self._blink_right_on
            tongue_out_active = self._tongue_out_on
        if blink_right_active:
            idx = FaceBlendShape.EyeBlinkRight
            overrides.append((idx, face.get_blendshape(idx)))
            face.set_blendshape(idx, 1.0, no_filter=True)
        if tongue_out_active:
            idx = FaceBlendShape.TongueOut
            overrides.append((idx, face.get_blendshape(idx)))
            face.set_blendshape(idx, 1.0, no_filter=True)
        return overrides

    def _restore_overrides(self, face: PyLiveLinkFace, overrides):
        for idx, value in overrides:
            face.set_blendshape(idx, value, no_filter=True)
