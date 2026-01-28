# utils/expression_enhancer.py
import json
import numpy as np
from typing import List, Optional, Dict, Any
import math # Using math.pow

# Attempt to import MediaPipe Category, provide fallback
try:
    from mediapipe.tasks.python.components.containers.category import Category
except ImportError:
    class Category: # Simple placeholder if MediaPipe not fully available here
        def __init__(self, index: int, score: float, display_name: str, category_name: str):
            self.index = index
            self.score = score
            self.display_name = display_name
            self.category_name = category_name
    print("Warning: Using placeholder for MediaPipe Category type.")


DEFAULT_CONFIG_PATH = "expression_config.json" # Default config file name

# Default values used if config file is missing or keys are absent
DEFAULT_SMOOTHING_ALPHA = 0.7
DEFAULT_GLOBAL_MULTIPLIER = 1.0
DEFAULT_GLOBAL_EXPONENT = 1.0

class BlendshapePostprocessor:
    """
    Applies global sensitivity, global overrides, emotion-specific profiles,
    and smoothing to raw blendshape scores. Reads configuration from a JSON file.
    Supports 'multiply', 'add', 'power' operations.
    """
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        # Initialize attributes with defaults before loading
        self.profiles: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.global_overrides: Dict[str, Dict[str, float]] = {}
        self.global_sensitivity_multiplier: float = DEFAULT_GLOBAL_MULTIPLIER
        self.global_sensitivity_exponent: float = DEFAULT_GLOBAL_EXPONENT
        self.smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA

        self.load_config() # Initial load

        # Stores the last smoothed value for the EMA filter
        self._last_smoothed_scores: Dict[str, float] = {}

    def load_config(self) -> bool:
        """Loads or reloads configuration from the JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)

            global_settings = config_data.get("global_settings", {})
            # Load global settings, falling back to current value if key missing in file
            self.smoothing_alpha = float(global_settings.get("smoothing_alpha", self.smoothing_alpha))
            self.global_sensitivity_multiplier = float(global_settings.get("global_sensitivity_multiplier", self.global_sensitivity_multiplier))
            self.global_sensitivity_exponent = float(global_settings.get("global_sensitivity_exponent", self.global_sensitivity_exponent))

            # Load overrides and profiles, replacing existing
            self.global_overrides = config_data.get("global_overrides", {})
            self.profiles = config_data.get("emotion_profiles", {})

            print(f"[BlendshapePostprocessor] Config loaded from {self.config_path}")
            # Reset smoothed scores as parameters affecting calculation may have changed
            self._last_smoothed_scores = {}
            return True
        except FileNotFoundError:
            print(f"[BlendshapePostprocessor] ERROR: Config file not found: {self.config_path}. Using previous/default values.")
            self._ensure_defaults_exist()
            return False
        except json.JSONDecodeError as e:
            print(f"[BlendshapePostprocessor] ERROR: Could not decode JSON '{self.config_path}': {e}. Using previous/default values.")
            self._ensure_defaults_exist()
            return False
        except Exception as e:
            print(f"[BlendshapePostprocessor] ERROR: Unexpected error loading config: {e}. Using previous/default values.")
            self._ensure_defaults_exist()
            return False

    def _ensure_defaults_exist(self):
        """Sets internal defaults only if attributes haven't been set yet."""
        if not hasattr(self, 'smoothing_alpha'): self.smoothing_alpha = DEFAULT_SMOOTHING_ALPHA
        if not hasattr(self, 'global_sensitivity_multiplier'): self.global_sensitivity_multiplier = DEFAULT_GLOBAL_MULTIPLIER
        if not hasattr(self, 'global_sensitivity_exponent'): self.global_sensitivity_exponent = DEFAULT_GLOBAL_EXPONENT
        if not hasattr(self, 'profiles'): self.profiles = {}
        if not hasattr(self, 'global_overrides'): self.global_overrides = {}
        if not hasattr(self, '_last_smoothed_scores'): self._last_smoothed_scores = {}

    def _apply_ops(self, score: float, ops: Dict[str, float]) -> float:
        """Applies multiply, add, and power operations."""
        modified_score = score
        # Order: Multiply -> Add -> Power
        if "multiply" in ops:
            modified_score *= float(ops["multiply"])
        if "add" in ops: # <<< "add" operation restored >>>
            modified_score += float(ops["add"])
        if "power" in ops and abs(modified_score) > 1e-6: # Check magnitude threshold
            exponent = float(ops["power"])
            try:
                # Use max(0.0,...) ensures non-negative base for fractional exponents
                base = max(0.0, modified_score)
                modified_score = math.pow(base, exponent)
            except ValueError:
                 pass # Keep score as is if pow fails
        return modified_score

    def _apply_global_sensitivity(self, score: float) -> float:
        """Applies global multiplier and exponent."""
        modified_score = score
        if self.global_sensitivity_multiplier != 1.0:
            modified_score *= self.global_sensitivity_multiplier
        if self.global_sensitivity_exponent != 1.0 and abs(modified_score) > 1e-6:
            try:
                modified_score = math.pow(max(0.0, modified_score), self.global_sensitivity_exponent)
            except ValueError:
                pass
        # Clipping happens later
        return modified_score

    def _low_pass_filter(self, category_name: str, current_value: float) -> float:
        """Applies an Exponential Moving Average (EMA) low-pass filter."""
        current_value = float(current_value)
        prev_value = float(self._last_smoothed_scores.get(category_name, current_value))
        smoothed_value = (self.smoothing_alpha * current_value) + \
                         ((1.0 - self.smoothing_alpha) * prev_value)
        self._last_smoothed_scores[category_name] = smoothed_value
        return smoothed_value

    def process(self,
                raw_blendshapes: Optional[List[Category]],
                detected_emotion: str
               ) -> List[Category]:
        """
        Applies processing pipeline: Global Sensitivity -> Global Overrides ->
        Emotion Profile -> Clamp -> Filter -> Final Clamp.
        Returns a new list of Category objects with modified scores.
        """
        if not raw_blendshapes:
            return []

        # --- Debug Print 1: Show entry and detected emotion ---
        #print(f"[PostProcess] Emotion: {detected_emotion} | ", end="")

        emotion_profile = self.profiles.get(detected_emotion, {})
        output_blendshapes = []
        emotion_mod_count = 0 # Count shapes modified by emotion profile

        for bs in raw_blendshapes:
            original_category_name = bs.category_name
            try: original_score = float(bs.score)
            except (ValueError, TypeError): original_score = 0.0
            current_score = original_score

            if not original_category_name: continue

            # Step 1: Apply Global Sensitivity
            current_score = self._apply_global_sensitivity(current_score)

            # Step 2: Apply Global Overrides
            score_before_emotion_ops = current_score # Store for checking modification count later
            if original_category_name in self.global_overrides:
                global_ops = self.global_overrides[original_category_name]
                current_score = self._apply_ops(current_score, global_ops)
                score_before_emotion_ops = current_score # Update base if global override applied

            # Step 3: Apply Emotion-Specific Profile
            if original_category_name in emotion_profile:
                emotion_ops = emotion_profile[original_category_name]
                current_score = self._apply_ops(current_score, emotion_ops)
                # Count if emotion profile ops made a difference
                if abs(current_score - score_before_emotion_ops) > 0.01:
                    emotion_mod_count +=1

            # Step 4: Clamp BEFORE filtering
            score_to_filter = np.clip(current_score, 0.0, 1.0)

            # Step 5: Apply Low-Pass Filter
            final_score = self._low_pass_filter(original_category_name, score_to_filter)

            # Step 6: Final clamp
            final_score_clamped = np.clip(final_score, 0.0, 1.0)

            # --- Debug Print 2: Show change for a specific blendshape (e.g., noseSneerLeft) ---
            # Change "noseSneerLeft" to monitor a different blendshape if needed
            if original_category_name == "noseSneerLeft" and abs(original_score - final_score_clamped) > 0.05:
                 continue
                 #print(f"NoseL: {original_score:.2f}->{final_score_clamped:.2f} | ", end="")

            output_blendshapes.append(
                Category(index=bs.index, score=final_score_clamped, display_name=bs.display_name, category_name=bs.category_name)
            )

        # Print summary count at the end of processing all blendshapes
        #print(f"EmoMods: {emotion_mod_count}") # Adds newline

        return output_blendshapes