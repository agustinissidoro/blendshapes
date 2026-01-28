# head_pose.py (New or Modified)

import numpy as np
from collections import deque
import statistics
import scipy.spatial.transform as R_scipy # Requires scipy
from typing import Optional, Tuple

# (You can keep the old get_head_pose function if needed elsewhere, or remove it)

class HeadPoseProcessor:
    def __init__(self,
                 filter_window_size: int = 7,
                 max_yaw_deg: float = 45.0,
                 max_pitch_deg: float = 20.0,
                 max_roll_deg: float = 45.0,
                 pitch_offset_deg: float = 0.0,
                 euler_order: str = 'yxz'): # Allow configuring Euler order
        # Configuration
        self.filter_window_size = filter_window_size
        self.max_yaw_deg = max_yaw_deg
        self.max_pitch_deg = max_pitch_deg
        self.max_roll_deg = max_roll_deg
        self.pitch_offset_deg = pitch_offset_deg
        self.euler_order = euler_order # Store the chosen order
        self.epsilon = 1e-6

        # State for filtering and unwrapping
        self._yaw_history = deque(maxlen=self.filter_window_size)
        self._pitch_history = deque(maxlen=self.filter_window_size)
        self._roll_history = deque(maxlen=self.filter_window_size)
        self._prev_unwrapped_yaw = 0.0
        self._prev_unwrapped_pitch = 0.0
        self._prev_unwrapped_roll = 0.0

    def _unwrap_angle(self, current_angle: float, prev_angle: float) -> float:
         delta = current_angle - prev_angle
         while delta >= 180.0: delta -= 360.0; current_angle -= 360.0
         while delta < -180.0: delta += 360.0; current_angle += 360.0
         return current_angle

    def process(self, transformation_matrix: Optional[np.ndarray]) -> Tuple[float, float, float]:
        """
        Processes a 4x4 transformation matrix into filtered, normalized head pose angles.
        Returns: Tuple[float, float, float] (yaw_clamped, pitch_clamped, roll_clamped) in range [-1, 1].
        """
        raw_yaw_deg, raw_pitch_deg, raw_roll_deg = 0.0, 0.0, 0.0

        if transformation_matrix is not None and transformation_matrix.shape == (4, 4):
            try:
                R_mat = transformation_matrix[:3, :3]
                r = R_scipy.Rotation.from_matrix(R_mat)
                # Use the configured Euler order
                euler_angles_deg = r.as_euler(self.euler_order, degrees=True)

                # Assign based on common conventions for 'yxz' order (adjust if order changes)
                if self.euler_order.lower() == 'yxz':
                    raw_yaw_deg = euler_angles_deg[0]
                    raw_pitch_deg = euler_angles_deg[1]
                    raw_roll_deg = euler_angles_deg[2]
                elif self.euler_order.lower() == 'xyz': # Example for different order
                    raw_roll_deg = euler_angles_deg[0] # X often maps to Roll
                    raw_pitch_deg = euler_angles_deg[1]# Y often maps to Pitch
                    raw_yaw_deg = euler_angles_deg[2]  # Z often maps to Yaw
                else:
                     # Default or handle other orders if needed
                     raw_yaw_deg, raw_pitch_deg, raw_roll_deg = euler_angles_deg[:3]


                # Test removing this adjustment (depends on matrix coordinate system)
                # raw_pitch_deg = raw_pitch_deg - 180.0

            except Exception: # Keep raw angles at 0 if conversion fails
                pass

        unwrapped_yaw = self._unwrap_angle(raw_yaw_deg, self._prev_unwrapped_yaw)
        unwrapped_pitch = self._unwrap_angle(raw_pitch_deg, self._prev_unwrapped_pitch)
        unwrapped_roll = self._unwrap_angle(raw_roll_deg, self._prev_unwrapped_roll)
        self._prev_unwrapped_yaw, self._prev_unwrapped_pitch, self._prev_unwrapped_roll = unwrapped_yaw, unwrapped_pitch, unwrapped_roll

        self._yaw_history.append(unwrapped_yaw)
        self._pitch_history.append(unwrapped_pitch)
        self._roll_history.append(unwrapped_roll)

        yaw_filtered = statistics.mean(self._yaw_history) if self._yaw_history else 0.0
        pitch_filtered = statistics.mean(self._pitch_history) if self._pitch_history else 0.0
        roll_filtered = statistics.mean(self._roll_history) if self._roll_history else 0.0

        pitch_compensated = pitch_filtered + self.pitch_offset_deg

        # Apply Final Axis Mapping & Signs (CONFIRM THESE based on testing)
        final_yaw = yaw_filtered         # Example: Yaw -> Yaw
        final_pitch = -pitch_compensated # Example: Pitch -> -Pitch (Compensated, Negated)
        final_roll = roll_filtered       # Example: Roll -> Roll

        # Normalization & Clamping
        yaw_norm = final_yaw / (self.max_yaw_deg + self.epsilon)
        pitch_norm = final_pitch / (self.max_pitch_deg + self.epsilon)
        roll_norm = final_roll / (self.max_roll_deg + self.epsilon)

        yaw_clamped = np.clip(yaw_norm, -1.0, 1.0)
        pitch_clamped = np.clip(pitch_norm, -1.0, 1.0)
        roll_clamped = np.clip(roll_norm, -1.0, 1.0)

        return yaw_clamped, pitch_clamped, roll_clamped