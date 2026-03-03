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
    "TARGET_FPS": 30,
    "DISPLAY_VIDEO": True,
    "SHOW_FPS": False,
    "BLENDSHAPE_SWAP_LR": False,
    "HP_FILTER_WINDOW": 6,
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
