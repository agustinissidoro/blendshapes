import json
from typing import Any, Dict


CONFIG_PATH = "config.json"


DEFAULTS: Dict[str, Any] = {
    "SOURCE": 0,
    "FLIP_IMAGE": False,
    "TARGET_SIZE": 1024,
    "LIVE_LINK_IP": "192.168.100.2",
    "LIVE_LINK_PORT": 11111,
    "LIVE_LINK_CLIENT_NAME": "Python_LiveLinkFace",
    "FACE_MODEL_PATH": "./models/face_landmarker.task",
    "TARGET_FPS": 30,       # Legacy — prefer PROCESS_FPS / SEND_FPS
    "PROCESS_FPS": 60,      # Main-loop / MediaPipe processing rate
    "SEND_FPS": 30,         # LiveLink UDP send rate (can differ from PROCESS_FPS)
    "CAPTURE_WIDTH": 1920,  # Desired camera capture width
    "CAPTURE_HEIGHT": 1080, # Desired camera capture height
    "CAPTURE_FPS": 60,      # Desired camera hardware fps
    "CROP_ENABLED": True,   # Crop center square before MediaPipe; false = pass full frame
    "DISPLAY_VIDEO": True,
    "SHOW_FPS": False,
    "BLENDSHAPE_SWAP_LR": False,
    "HP_FILTER_TYPE": "box",  # Head-pose temporal filter: "none" | "box" | "ema" | "kalman"
    "HP_FILTER_WINDOW": 6,    # "box" only: rolling-mean window size
    "HP_EMA_ALPHA": 0.5,      # "ema" only: 1.0 = raw, 0.0 = no update (lower = smoother)
    "HP_KALMAN_Q": 1e-3,      # "kalman" only: process noise (higher = more responsive)
    "HP_KALMAN_R": 1e-2,      # "kalman" only: measurement noise (higher = smoother)
    "HP_MAX_YAW": 45.0,
    "HP_MAX_PITCH": 20.0,
    "HP_MAX_ROLL": 45.0,
    "HP_YAW_OFFSET": 0.0,
    "HP_PITCH_OFFSET": 0.0,
    "HP_EULER_ORDER": "yxz",
    "EMOTION_MODEL_PATH": "./models/enet_b0_8_best_afew.onnx",
    "EVERY_FPS": 6,
    "EMOTION_RECOGNITION_ENABLED": True,
    "POST_PROCESS_BLENDSHAPES": True,
    "EXPRESSION_CONFIG_PATH": "expression_profiles.json",
    "EYE_POST_PROCESSOR": True,
    "PAIR_EYELIDS": True,
    "UDP_COMMAND_IP": "127.0.0.1",
    "UDP_COMMAND_PORT": 12000,
    "UDP_STATE_IP": "127.0.0.1",
    "UDP_STATE_PORT": 12001,
    "CALIBRATION_PATH": "calibration.json",
    "CALIBRATION_HEADPOSE_ENABLED": True,
    "CALIBRATION_BLENDSHAPES_ENABLED": True,
}


def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            cfg = json.load(file)
    except Exception:
        cfg = {}

    for key, value in DEFAULTS.items():
        cfg.setdefault(key, value)
    return cfg
