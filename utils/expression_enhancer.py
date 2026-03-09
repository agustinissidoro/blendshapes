# utils/expression_enhancer.py
import json
import numpy as np
from typing import List, Optional
import math

# MediaPipe Category fallback
try:
    from mediapipe.tasks.python.components.containers.category import Category
except ImportError:
    class Category:
        def __init__(self, index: int, score: float, display_name: str, category_name: str):
            self.index = index
            self.score = score
            self.display_name = display_name
            self.category_name = category_name
    print("Warning: Using placeholder for MediaPipe Category type.")

DEFAULT_CONFIG_PATH = "expression_config.json"
DEFAULT_SMOOTHING_ALPHA = 0.7
DEFAULT_GLOBAL_MULTIPLIER = 1.0
DEFAULT_GLOBAL_EXPONENT = 1.0

class BlendshapePostprocessor:
    """
    Applies calibration baseline, global sensitivity, overrides, emotion profiles,
    and smoothing to raw blendshape scores. Supports per-shape clamp, add, and HPF.
    """
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.profiles = {}
        self.global_overrides = {}
        self.global_sensitivity_multiplier = DEFAULT_GLOBAL_MULTIPLIER
        self.global_sensitivity_exponent = DEFAULT_GLOBAL_EXPONENT
        self.smoothing_alpha = DEFAULT_SMOOTHING_ALPHA

        self.apply_emotions = True
        self.apply_global_overrides = True

        self._last_smoothed_scores = {}
        self._last_raw_scores = {}    # HPF: previous input
        self._last_hpf_scores = {}    # HPF: previous output
        self._calibration_baseline = {}  # {category_name: neutral_score}
        self.calibration_enabled = True
        self.load_config()

    def load_config(self) -> bool:
        """Load JSON config for global overrides and emotion profiles."""
        try:
            with open(self.config_path, "r") as f:
                cfg = json.load(f)
            self.smoothing_alpha = float(cfg.get("global_settings", {}).get("smoothing_alpha", self.smoothing_alpha))
            self.global_sensitivity_multiplier = float(cfg.get("global_settings", {}).get("global_sensitivity_multiplier", self.global_sensitivity_multiplier))
            self.global_sensitivity_exponent = float(cfg.get("global_settings", {}).get("global_sensitivity_exponent", self.global_sensitivity_exponent))
            self.global_overrides = cfg.get("global_overrides", {})
            self.profiles = cfg.get("emotion_profiles", {})
            self._last_smoothed_scores.clear()
            self._last_raw_scores.clear()
            self._last_hpf_scores.clear()
            print(f"[BlendshapePostprocessor] Config loaded from {self.config_path}")
            return True
        except Exception as e:
            print(f"[BlendshapePostprocessor] Failed to load config: {e}")
            return False

    def capture_neutral(self, raw_blendshapes) -> None:
        """Store raw blendshape scores as the neutral baseline.

        Pass the raw MediaPipe blendshapes (before any post-processing) so that
        calibration is independent of expression config changes.
        """
        self._calibration_baseline = {
            bs.category_name: float(bs.score)
            for bs in raw_blendshapes
            if bs.category_name
        }
        print(f"[BlendshapePostprocessor] Calibration captured ({len(self._calibration_baseline)} shapes).")

    def clear_calibration(self) -> None:
        self._calibration_baseline.clear()
        print("[BlendshapePostprocessor] Calibration cleared.")

    def save_calibration(self, path: str) -> bool:
        try:
            with open(path, "w") as f:
                json.dump({"blendshape_baseline": self._calibration_baseline}, f, indent=2)
            print(f"[BlendshapePostprocessor] Calibration saved to {path}")
            return True
        except Exception as e:
            print(f"[BlendshapePostprocessor] Failed to save calibration: {e}")
            return False

    def load_calibration(self, path: str) -> bool:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._calibration_baseline = data.get("blendshape_baseline", {})
            print(f"[BlendshapePostprocessor] Calibration loaded from {path} ({len(self._calibration_baseline)} shapes).")
            return True
        except Exception as e:
            print(f"[BlendshapePostprocessor] Failed to load calibration: {e}")
            return False

    def _apply_ops(self, score: float, ops: dict, category_name: str) -> float:
        """Apply multiply/add/power/clamp/high-pass operations."""
        s = float(score)

        # Multiply & add
        if "multiply" in ops:
            s *= float(ops["multiply"])
        if "add" in ops:
            s += float(ops["add"])

        # Power curve
        if "power" in ops and abs(s) > 1e-6:
            try:
                s = math.pow(max(0.0, s), float(ops["power"]))
            except Exception:
                pass

        # Clamp
        if "clamp" in ops and isinstance(ops["clamp"], (list, tuple)) and len(ops["clamp"]) == 2:
            low, high = ops["clamp"]
            s = max(float(low), min(float(high), s))

        # High-pass filter: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        # alpha near 1.0 = pass-through, alpha near 0 = strong HPF
        if "hp_alpha" in ops or "hp_filter" in ops:
            alpha = float(ops.get("hp_alpha", ops.get("hp_filter")))
            prev_raw = self._last_raw_scores.get(category_name, s)
            prev_hpf = self._last_hpf_scores.get(category_name, s)
            hpf_val = alpha * (prev_hpf + s - prev_raw)
            self._last_raw_scores[category_name] = s
            self._last_hpf_scores[category_name] = hpf_val
            s = hpf_val

        return s

    def _apply_global_sensitivity(self, score: float) -> float:
        s = score
        if self.global_sensitivity_multiplier != 1.0:
            s *= self.global_sensitivity_multiplier
        if self.global_sensitivity_exponent != 1.0 and abs(s) > 1e-6:
            try:
                s = math.pow(max(0.0, s), self.global_sensitivity_exponent)
            except Exception:
                pass
        return s

    def _low_pass_filter(self, category_name: str, current_value: float) -> float:
        prev = self._last_smoothed_scores.get(category_name, current_value)
        smoothed = self.smoothing_alpha * current_value + (1 - self.smoothing_alpha) * prev
        self._last_smoothed_scores[category_name] = smoothed
        return smoothed

    def process(self,
                raw_blendshapes: Optional[List[Category]],
                detected_emotion: str
               ) -> List[Category]:
        """Process blendshapes with optional calibration, emotion & global overrides."""
        if not raw_blendshapes:
            return []

        emotion_profile = self.profiles.get(detected_emotion, {})
        output = []

        for bs in raw_blendshapes:
            if not bs.category_name:
                continue

            score = float(bs.score)

            # Subtract neutral baseline (calibration) before any processing
            if self._calibration_baseline and self.calibration_enabled:
                score -= self._calibration_baseline.get(bs.category_name, 0.0)
                score = max(0.0, score)

            score = self._apply_global_sensitivity(score)

            # Global overrides
            if self.apply_global_overrides and bs.category_name in self.global_overrides:
                score = self._apply_ops(score, self.global_overrides[bs.category_name], bs.category_name)

            # Emotion profile
            if self.apply_emotions and bs.category_name in emotion_profile:
                score = self._apply_ops(score, emotion_profile[bs.category_name], bs.category_name)

            # Clamp and smooth
            score = np.clip(score, 0.0, 1.0)
            score = self._low_pass_filter(bs.category_name, score)
            score = np.clip(score, 0.0, 1.0)

            output.append(Category(bs.index, score, bs.display_name, bs.category_name))

        return output

class EyePostProcessor:
    """
    Stabilizes and optionally pairs eye-related blendshapes (iris direction)
    to reduce jitter and bias toward a neutral (forward-looking) orientation.
    Reads configuration from a JSON file (expression_profiles.json).
    """

    DEFAULT_CONFIG_SECTION = "_eye_settings"

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.enabled = True
        self.smoothing = 0.85
        self.bias_strength = 0.6
        self.deadzone = 0.03
        self.pair_eyes = True
        self.previous = {}

        self.eye_pairs = {
            "L": ["eyeLookUpLeft", "eyeLookDownLeft", "eyeLookInLeft", "eyeLookOutLeft"],
            "R": ["eyeLookUpRight", "eyeLookDownRight", "eyeLookInRight", "eyeLookOutRight"]
        }

        self.load_config()

    def load_config(self) -> bool:
        """Load eye settings from the JSON config file."""
        try:
            with open(self.config_path, "r") as f:
                cfg = json.load(f)
            eye_cfg = cfg.get(self.DEFAULT_CONFIG_SECTION, cfg.get("eye_postprocessor", {}))

            self.enabled = eye_cfg.get("enabled", self.enabled)
            self.smoothing = eye_cfg.get("smoothing", self.smoothing)
            self.bias_strength = eye_cfg.get("bias_strength", self.bias_strength)
            self.deadzone = eye_cfg.get("deadzone", self.deadzone)
            self.pair_eyes = eye_cfg.get("pair_eyes", self.pair_eyes)
            return True
        except Exception as e:
            print(f"[EyePostProcessor] Failed to load config: {e}")
            return False

    def process(self, blendshape_dict: dict) -> dict:
        """Post-process eye-related blendshapes for stabilization and pairing."""
        if not self.enabled:
            return blendshape_dict

        # Optionally synchronize eyes (averaging left/right)
        if self.pair_eyes:
            for l_key, r_key in zip(self.eye_pairs["L"], self.eye_pairs["R"]):
                if l_key in blendshape_dict and r_key in blendshape_dict:
                    avg = (blendshape_dict[l_key] + blendshape_dict[r_key]) / 2.0
                    blendshape_dict[l_key] = avg
                    blendshape_dict[r_key] = avg

        # Stabilize each individual movement
        for key_group in [self.eye_pairs["L"], self.eye_pairs["R"]]:
            for key in key_group:
                if key not in blendshape_dict:
                    continue

                val = float(blendshape_dict[key])

                # Apply deadzone (ignore micro-motions)
                if abs(val) < self.deadzone:
                    val = 0.0

                # Pull toward neutral (reduce drift)
                val *= (1.0 - self.bias_strength)

                # Smooth transitions
                prev_val = self.previous.get(key, val)
                val = self.smoothing * prev_val + (1.0 - self.smoothing) * val
                self.previous[key] = val

                # Clamp result
                blendshape_dict[key] = float(np.clip(val, 0.0, 1.0))

        return blendshape_dict
