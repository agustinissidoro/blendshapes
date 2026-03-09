import numpy as np
from collections import deque
import statistics
import scipy.spatial.transform as R_scipy
from typing import Optional, Tuple


class _KalmanFilter1D:
    """Minimal scalar Kalman filter (constant-position model).

    Q  — process noise variance: higher = more responsive to fast motion.
    R  — measurement noise variance: higher = more smoothing.
    """
    def __init__(self, q: float = 1e-3, r: float = 1e-2):
        self._q = q
        self._r = r
        self._x = 0.0   # state estimate
        self._p = 1.0   # error covariance

    def update(self, measurement: float) -> float:
        # Predict
        self._p += self._q
        # Update
        k = self._p / (self._p + self._r)
        self._x += k * (measurement - self._x)
        self._p *= (1.0 - k)
        return self._x

    def reset(self, value: float = 0.0) -> None:
        self._x = value
        self._p = 1.0


class HeadPoseProcessor:
    """Converts a MediaPipe 4x4 facial transformation matrix into
    normalised (yaw, pitch, roll) floats in [-1, 1].

    HP_FILTER_TYPE options
    ----------------------
    "none"    — raw unwrapped angle, no temporal smoothing.
    "box"     — rolling mean over HP_FILTER_WINDOW frames (original behaviour).
    "ema"     — exponential moving average. HP_EMA_ALPHA: 1.0 = raw, 0.0 = frozen.
    "kalman"  — scalar Kalman filter per axis. HP_KALMAN_Q / HP_KALMAN_R tune it.
    """

    def __init__(self,
                 filter_window_size: int = 7,
                 max_yaw_deg: float = 45.0,
                 max_pitch_deg: float = 20.0,
                 max_roll_deg: float = 45.0,
                 yaw_offset_deg: float = 0.0,
                 pitch_offset_deg: float = 0.0,
                 euler_order: str = 'yxz',
                 filter_type: str = 'box',
                 ema_alpha: float = 0.5,
                 kalman_q: float = 1e-3,
                 kalman_r: float = 1e-2):

        self.filter_window_size = filter_window_size
        self.max_yaw_deg = max_yaw_deg
        self.max_pitch_deg = max_pitch_deg
        self.max_roll_deg = max_roll_deg
        self._default_yaw_offset_deg = float(yaw_offset_deg)
        self._default_pitch_offset_deg = float(pitch_offset_deg)
        self._default_roll_offset_deg = 0.0
        self._yaw_offset_correction_deg = 0.0
        self._pitch_offset_correction_deg = 0.0
        self._roll_offset_correction_deg = 0.0
        self.yaw_offset_deg = 0.0
        self.pitch_offset_deg = 0.0
        self.roll_offset_deg = 0.0
        self.euler_order = euler_order
        self.calibration_enabled = True
        self.epsilon = 1e-6

        # Filter selection
        self.filter_type = filter_type.lower()
        self.ema_alpha = float(ema_alpha)

        # Angle unwrap state (shared by all filter types)
        self._prev_unwrapped_yaw = 0.0
        self._prev_unwrapped_pitch = 0.0
        self._prev_unwrapped_roll = 0.0

        # Box filter state
        self._yaw_history = deque(maxlen=self.filter_window_size)
        self._pitch_history = deque(maxlen=self.filter_window_size)
        self._roll_history = deque(maxlen=self.filter_window_size)

        # EMA filter state
        self._ema_yaw: Optional[float] = None
        self._ema_pitch: Optional[float] = None
        self._ema_roll: Optional[float] = None

        # Kalman filter state
        self._kalman_yaw = _KalmanFilter1D(kalman_q, kalman_r)
        self._kalman_pitch = _KalmanFilter1D(kalman_q, kalman_r)
        self._kalman_roll = _KalmanFilter1D(kalman_q, kalman_r)

        self._refresh_effective_offsets()

    # ------------------------------------------------------------------
    # Offset management
    # ------------------------------------------------------------------

    def _refresh_effective_offsets(self) -> None:
        self.yaw_offset_deg = self._default_yaw_offset_deg + self._yaw_offset_correction_deg
        self.pitch_offset_deg = self._default_pitch_offset_deg + self._pitch_offset_correction_deg
        self.roll_offset_deg = self._default_roll_offset_deg + self._roll_offset_correction_deg

    def get_offsets(self) -> Tuple[float, float]:
        return float(self.yaw_offset_deg), float(self.pitch_offset_deg)

    def get_default_offsets(self) -> Tuple[float, float]:
        return float(self._default_yaw_offset_deg), float(self._default_pitch_offset_deg)

    def get_offset_corrections(self) -> Tuple[float, float]:
        return float(self._yaw_offset_correction_deg), float(self._pitch_offset_correction_deg)

    def set_offsets(
        self,
        yaw_offset_deg: Optional[float] = None,
        pitch_offset_deg: Optional[float] = None,
        additive: bool = False,
    ) -> Tuple[float, float]:
        if yaw_offset_deg is not None:
            yaw_value = float(yaw_offset_deg)
            self._yaw_offset_correction_deg = (
                self._yaw_offset_correction_deg + yaw_value if additive else yaw_value
            )
        if pitch_offset_deg is not None:
            pitch_value = float(pitch_offset_deg)
            self._pitch_offset_correction_deg = (
                self._pitch_offset_correction_deg + pitch_value if additive else pitch_value
            )
        self._refresh_effective_offsets()
        return self.get_offsets()

    def reset_offsets(self) -> Tuple[float, float]:
        self._yaw_offset_correction_deg = 0.0
        self._pitch_offset_correction_deg = 0.0
        self._roll_offset_correction_deg = 0.0
        self._refresh_effective_offsets()
        return self.get_offsets()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _current_filtered(self) -> Tuple[float, float, float]:
        """Return the current filtered estimate for all three axes."""
        if self.filter_type == "none":
            return self._prev_unwrapped_yaw, self._prev_unwrapped_pitch, self._prev_unwrapped_roll
        elif self.filter_type == "box":
            yaw = statistics.mean(self._yaw_history) if self._yaw_history else 0.0
            pitch = statistics.mean(self._pitch_history) if self._pitch_history else 0.0
            roll = statistics.mean(self._roll_history) if self._roll_history else 0.0
            return yaw, pitch, roll
        elif self.filter_type == "ema":
            return (
                self._ema_yaw if self._ema_yaw is not None else 0.0,
                self._ema_pitch if self._ema_pitch is not None else 0.0,
                self._ema_roll if self._ema_roll is not None else 0.0,
            )
        elif self.filter_type == "kalman":
            return self._kalman_yaw._x, self._kalman_pitch._x, self._kalman_roll._x
        return 0.0, 0.0, 0.0

    def capture_neutral(self) -> Tuple[float, float, float]:
        """Capture the current filtered pose as the zero reference.

        Works regardless of which filter_type is active. The user should
        be in a natural neutral head position when calling this.

        Returns the effective (yaw, pitch, roll) correction offsets after capture.
        """
        yaw, pitch, roll = self._current_filtered()
        self._yaw_offset_correction_deg = -yaw
        self._pitch_offset_correction_deg = -pitch
        self._roll_offset_correction_deg = -roll
        self._refresh_effective_offsets()
        return float(self.yaw_offset_deg), float(self.pitch_offset_deg), float(self.roll_offset_deg)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _unwrap_angle(self, current_angle: float, prev_angle: float) -> float:
        delta = current_angle - prev_angle
        while delta >= 180.0:
            delta -= 360.0
            current_angle -= 360.0
        while delta < -180.0:
            delta += 360.0
            current_angle += 360.0
        return current_angle

    def _apply_filter(self, yaw: float, pitch: float, roll: float) -> Tuple[float, float, float]:
        """Apply the selected temporal filter and return smoothed values."""
        if self.filter_type == "none":
            return yaw, pitch, roll

        elif self.filter_type == "box":
            self._yaw_history.append(yaw)
            self._pitch_history.append(pitch)
            self._roll_history.append(roll)
            return (
                statistics.mean(self._yaw_history),
                statistics.mean(self._pitch_history),
                statistics.mean(self._roll_history),
            )

        elif self.filter_type == "ema":
            a = self.ema_alpha
            self._ema_yaw = yaw if self._ema_yaw is None else a * yaw + (1.0 - a) * self._ema_yaw
            self._ema_pitch = pitch if self._ema_pitch is None else a * pitch + (1.0 - a) * self._ema_pitch
            self._ema_roll = roll if self._ema_roll is None else a * roll + (1.0 - a) * self._ema_roll
            return self._ema_yaw, self._ema_pitch, self._ema_roll

        elif self.filter_type == "kalman":
            return (
                self._kalman_yaw.update(yaw),
                self._kalman_pitch.update(pitch),
                self._kalman_roll.update(roll),
            )

        return yaw, pitch, roll

    def process(self, transformation_matrix: Optional[np.ndarray]) -> Tuple[float, float, float]:
        """Process a 4x4 transformation matrix → normalised (yaw, pitch, roll) in [-1, 1]."""
        raw_yaw_deg, raw_pitch_deg, raw_roll_deg = 0.0, 0.0, 0.0

        if transformation_matrix is not None and transformation_matrix.shape == (4, 4):
            try:
                R_mat = transformation_matrix[:3, :3]
                r = R_scipy.Rotation.from_matrix(R_mat)
                euler_angles_deg = r.as_euler(self.euler_order, degrees=True)

                if self.euler_order.lower() == 'yxz':
                    raw_yaw_deg = euler_angles_deg[0]
                    raw_pitch_deg = euler_angles_deg[1]
                    raw_roll_deg = euler_angles_deg[2]
                elif self.euler_order.lower() == 'xyz':
                    raw_roll_deg = euler_angles_deg[0]
                    raw_pitch_deg = euler_angles_deg[1]
                    raw_yaw_deg = euler_angles_deg[2]
                else:
                    raw_yaw_deg, raw_pitch_deg, raw_roll_deg = euler_angles_deg[:3]

            except Exception:
                pass

        # Angle unwrapping (needed by all filter types to avoid 360° jumps)
        unwrapped_yaw = self._unwrap_angle(raw_yaw_deg, self._prev_unwrapped_yaw)
        unwrapped_pitch = self._unwrap_angle(raw_pitch_deg, self._prev_unwrapped_pitch)
        unwrapped_roll = self._unwrap_angle(raw_roll_deg, self._prev_unwrapped_roll)
        self._prev_unwrapped_yaw = unwrapped_yaw
        self._prev_unwrapped_pitch = unwrapped_pitch
        self._prev_unwrapped_roll = unwrapped_roll

        yaw_f, pitch_f, roll_f = self._apply_filter(unwrapped_yaw, unwrapped_pitch, unwrapped_roll)

        # Apply calibration corrections (only when enabled)
        yaw_corr = self._yaw_offset_correction_deg if self.calibration_enabled else 0.0
        pitch_corr = self._pitch_offset_correction_deg if self.calibration_enabled else 0.0
        roll_corr = self._roll_offset_correction_deg if self.calibration_enabled else 0.0
        yaw_compensated = yaw_f + self._default_yaw_offset_deg + yaw_corr
        pitch_compensated = pitch_f + self._default_pitch_offset_deg + pitch_corr
        roll_compensated = roll_f + self._default_roll_offset_deg + roll_corr

        final_yaw = yaw_compensated
        final_pitch = -pitch_compensated
        final_roll = roll_compensated

        yaw_norm = final_yaw / (self.max_yaw_deg + self.epsilon)
        pitch_norm = final_pitch / (self.max_pitch_deg + self.epsilon)
        roll_norm = final_roll / (self.max_roll_deg + self.epsilon)

        return (
            float(np.clip(yaw_norm, -1.0, 1.0)),
            float(np.clip(pitch_norm, -1.0, 1.0)),
            float(np.clip(roll_norm, -1.0, 1.0)),
        )
