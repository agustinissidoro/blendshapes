import socket
import threading
from typing import List, Set, Optional, Tuple
import time
import numpy as np
from collections import deque
import statistics
import scipy.spatial.transform as R_scipy # Requires scipy

from pylivelinkface import PyLiveLinkFace, FaceBlendShape
from mediapipe.tasks.python.components.containers.category import Category

# --- Constants ---
MAX_YAW_DEGREES = 45.0      # Tune for sensitivity
MAX_PITCH_DEGREES = 20.0    # Tune for sensitivity
MAX_ROLL_DEGREES = 45.0     # Tune for sensitivity
EPSILON = 1e-6
FILTER_WINDOW_SIZE = 7      # Tune for smoothness
# TUNE THIS: Look straight, check FiltDeg-> P value, set this to -(that value)
PITCH_OFFSET_DEGREES = 0.0

class LiveLinkSender(threading.Thread):

    def __init__(self, ip: str, port: int): # Removed target_size
        super().__init__()
        self.ip = ip
        self.port = port
        self.running = False
        self._face = PyLiveLinkFace()
        self._latest_blendshapes: List[Category] = []
        self._previous_keys: Set[str] = set()
        # History for EULER angles derived from matrix
        self._yaw_history = deque(maxlen=FILTER_WINDOW_SIZE)
        self._pitch_history = deque(maxlen=FILTER_WINDOW_SIZE)
        self._roll_history = deque(maxlen=FILTER_WINDOW_SIZE)
        # Previous unwrapped angles
        self._prev_unwrapped_yaw = 0.0
        self._prev_unwrapped_pitch = 0.0
        self._prev_unwrapped_roll = 0.0
        # Socket setup
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.connect((self.ip, self.port))
        except socket.error as e:
            print(f"[LiveLinkSender] Socket connect error: {e}")
            self._socket = None

    def run(self):
        if not self._socket: return
        self.running = True
        try:
            while self.running:
                self._send_data() # Send combined state
                time.sleep(0.01)  # Control send rate
        except socket.error: # Catch socket errors silently in loop if desired
             self.running = False
        except Exception as e:
            print(f"[LiveLinkSender] Error in run loop: {e}")
            self.running = False
        finally:
            if self._socket:
                self._socket.close()

    def stop(self):
        self.running = False

    def update_blendshapes(self, blendshapes: Optional[List[Category]]):
        if not self.running: return
        current_keys = set()
        if blendshapes:
            for category in blendshapes:
                name = category.category_name
                if not name: continue
                enum_key_name = name[0].upper() + name[1:]
                try:
                    enum_key = FaceBlendShape[enum_key_name]
                    if 0 <= enum_key.value <= 51: # Only update facial blendshapes
                        self._face.set_blendshape(enum_key, category.score, no_filter=True)
                        current_keys.add(enum_key_name)
                except KeyError: continue
        # Reset only facial keys previously set by this method
        for prev_key in self._previous_keys - current_keys:
             try:
                 enum_key = FaceBlendShape[prev_key]
                 if 0 <= enum_key.value <= 51:
                      self._face.set_blendshape(enum_key, 0.0, no_filter=True)
             except KeyError: continue
        self._previous_keys = current_keys

    def _unwrap_angle(self, current_angle: float, prev_angle: float) -> float:
         # Simple unwrap logic
         delta = current_angle - prev_angle
         while delta >= 180.0: delta -= 360.0; current_angle -= 360.0
         while delta < -180.0: delta += 360.0; current_angle += 360.0
         return current_angle

    def update_head_pose(self, transformation_matrix: Optional[np.ndarray]):
        if not self.running: return

        raw_yaw_deg, raw_pitch_deg, raw_roll_deg = 0.0, 0.0, 0.0

        if transformation_matrix is not None and transformation_matrix.shape == (4, 4):
            try:
                R_mat = transformation_matrix[:3, :3]
                # Convert matrix to Euler angles (degrees)
                # --- TRY DIFFERENT ORDERS like 'xyz', 'zyx' if 'yxz' doesn't map well ---
                r = R_scipy.Rotation.from_matrix(R_mat)
                euler_angles_deg = r.as_euler('yxz', degrees=True) # Example: Yaw, Pitch, Roll
                raw_yaw_deg = euler_angles_deg[0]
                raw_pitch_deg = euler_angles_deg[1]
                raw_roll_deg = euler_angles_deg[2]

                # --- Pitch Adjustment Test ---
                # Test removing this adjustment first! Add back if needed.
                # raw_pitch_deg = raw_pitch_deg - 180.0
                # --- ---

            except Exception as e:
                print(f"[LiveLinkSender] Error converting matrix to Euler: {e}")
                # Keep raw angles at 0 if conversion fails, proceed to filtering defaults

        # Angle Unwrapping (applied to Euler angles derived from matrix)
        unwrapped_yaw_deg = self._unwrap_angle(raw_yaw_deg, self._prev_unwrapped_yaw)
        unwrapped_pitch_deg = self._unwrap_angle(raw_pitch_deg, self._prev_unwrapped_pitch)
        unwrapped_roll_deg = self._unwrap_angle(raw_roll_deg, self._prev_unwrapped_roll)

        self._prev_unwrapped_yaw = unwrapped_yaw_deg
        self._prev_unwrapped_pitch = unwrapped_pitch_deg
        self._prev_unwrapped_roll = unwrapped_roll_deg

        # Update filter history with unwrapped Euler angles
        self._yaw_history.append(unwrapped_yaw_deg)
        self._pitch_history.append(unwrapped_pitch_deg)
        self._roll_history.append(unwrapped_roll_deg)

        # Calculate filtered (averaged) degrees
        yaw_deg_filtered = statistics.mean(self._yaw_history) if self._yaw_history else 0.0
        pitch_deg_filtered = statistics.mean(self._pitch_history) if self._pitch_history else 0.0
        roll_deg_filtered = statistics.mean(self._roll_history) if self._roll_history else 0.0

        # Apply Pitch Offset Compensation
        pitch_deg_compensated = pitch_deg_filtered + PITCH_OFFSET_DEGREES

        # --- Apply Axis Mapping & Signs ---
        # !!! MUST Re-Test and Adjust this mapping and signs ('-') !!!
        # Start assuming direct mapping from the Euler order you chose ('yxz' -> Yaw, Pitch, Roll)
        final_yaw_deg = yaw_deg_filtered         # Example: Start assuming Yaw -> Yaw
        final_pitch_deg = -pitch_deg_compensated  # Example: Start assuming Pitch -> Pitch
        final_roll_deg = roll_deg_filtered       # Example: Start assuming Roll -> Roll
        # Add '-' signs based on observed direction in Unreal

        # Normalize
        yaw_norm = final_yaw_deg / (MAX_YAW_DEGREES + EPSILON)
        pitch_norm = final_pitch_deg / (MAX_PITCH_DEGREES + EPSILON)
        roll_norm = final_roll_deg / (MAX_ROLL_DEGREES + EPSILON)

        # Clamp
        yaw_clamped = np.clip(yaw_norm, -1.0, 1.0)
        pitch_clamped = np.clip(pitch_norm, -1.0, 1.0)
        roll_clamped = np.clip(roll_norm, -1.0, 1.0)

        # Update face object state
        self._face.set_blendshape(FaceBlendShape.HeadYaw, yaw_clamped, no_filter=True)
        self._face.set_blendshape(FaceBlendShape.HeadPitch, pitch_clamped, no_filter=True)
        self._face.set_blendshape(FaceBlendShape.HeadRoll, roll_clamped, no_filter=True)

    # Sends the *entire* self._face state (blendshapes + pose)
    def _send_data(self):
        if not self.running or not self._socket: return
        try:
            payload = self._face.encode()
            self._socket.sendall(payload)
        except socket.error: pass # Ignore send errors silently
        except Exception as e: print(f"[LiveLinkSender] Error encoding/sending: {e}")

    # Keep this if run() calls it, otherwise remove
    def _send_blendshapes(self):
        self._send_data()