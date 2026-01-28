import time
import json
import cv2
import queue
from capture import VideoCaptureThread
from facelandmarks import FaceLandmarkerProcessor
from utils.face_cropper import FaceCropper
from utils.display_blendshapes import draw_overlay
from live_link_sender import LiveLinkSender
from utils.head_pose import HeadPoseProcessor
from utils.scheduler import FrameScheduler
from utils.emotion_worker import EmotionWorker
from utils.expression_enhancer import BlendshapePostprocessor, EyePostProcessor
from utils.input_handler import InputHandler

CONFIG_PATH = "config.json"

# --- Global state ---
latest_blendshapes = None
latest_landmarks = None
latest_transform_matrix = None

def load_config(path=CONFIG_PATH):
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
        print(f"[Config] Loaded configuration from {path}")
    except Exception as e:
        print(f"[Config] Could not load config file ({e}), using defaults.")
        cfg = {}

    # Set defaults if keys are missing
    cfg.setdefault("SOURCE", 0)
    cfg.setdefault("FLIP_IMAGE", False)
    cfg.setdefault("TARGET_SIZE", 1024)
    cfg.setdefault("LIVE_LINK_IP", "192.168.100.2")
    cfg.setdefault("LIVE_LINK_PORT", 11111)
    cfg.setdefault("FACE_MODEL_PATH", "./models/face_landmarker.task")
    cfg.setdefault("EMOTION_MODEL_PATH", "./models/enet_b0_8_best_afew.onnx")
    cfg.setdefault("TARGET_FPS", 30)
    cfg.setdefault("DISPLAY_VIDEO", True)
    cfg.setdefault("SHOW_FPS", False)
    cfg.setdefault("HP_FILTER_WINDOW", 7)
    cfg.setdefault("HP_MAX_YAW", 45.0)
    cfg.setdefault("HP_MAX_PITCH", 20.0)
    cfg.setdefault("HP_MAX_ROLL", 45.0)
    cfg.setdefault("HP_PITCH_OFFSET", 0.0)
    cfg.setdefault("HP_EULER_ORDER", "yxz")
    cfg.setdefault("EVERY_FPS", 6)
    cfg.setdefault("POST_PROCESS_BLENDSHAPES", True)
    cfg.setdefault("EXPRESSION_CONFIG_PATH", "expression_profiles.json")
    cfg.setdefault("MODE", "performance")
    cfg.setdefault("GLOBAL_EXPRESSION_OVERRIDES", True)
    cfg.setdefault("EMOTION_RECOGNITION_WEIGHTS", True)
    return cfg

def handle_landmarker_result(result, image, timestamp_ms):
    global latest_blendshapes, latest_landmarks, latest_transform_matrix
    latest_blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
    latest_landmarks = result.face_landmarks[0] if result.face_landmarks else None
    latest_transform_matrix = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None

def main():
    global latest_blendshapes, latest_landmarks, latest_transform_matrix

    cfg = load_config()

    cropper = FaceCropper(target_size=cfg["TARGET_SIZE"])
    scheduler = FrameScheduler(cfg["TARGET_FPS"])

    emotion_thread = EmotionWorker(cfg["EMOTION_MODEL_PATH"], process_every_n=cfg["EVERY_FPS"])
    emotion_thread.start()

    head_processor = HeadPoseProcessor(
        filter_window_size=cfg["HP_FILTER_WINDOW"],
        max_yaw_deg=cfg["HP_MAX_YAW"],
        max_pitch_deg=cfg["HP_MAX_PITCH"],
        max_roll_deg=cfg["HP_MAX_ROLL"],
        pitch_offset_deg=cfg["HP_PITCH_OFFSET"],
        euler_order=cfg["HP_EULER_ORDER"]
    )

    blendshape_post_processor = BlendshapePostprocessor(config_path=cfg["EXPRESSION_CONFIG_PATH"])
    eye_post_processor = EyePostProcessor(config_path=cfg["EXPRESSION_CONFIG_PATH"])
    sender = LiveLinkSender(cfg["LIVE_LINK_IP"], cfg["LIVE_LINK_PORT"])

    cam = VideoCaptureThread(src=cfg["SOURCE"], flip_image=cfg["FLIP_IMAGE"])
    face_processor = FaceLandmarkerProcessor(model_path=cfg["FACE_MODEL_PATH"], result_callback=handle_landmarker_result)

    action_queue = queue.Queue()
    input_handler = InputHandler(action_queue)
    sender.start()
    cam.start()
    
    if cfg["MODE"] == "setting":
        input_handler.start()

    print("[Main] Processing started...")
    run_main_loop = True
    processed_pose_angles_for_display = None

    try:
        while run_main_loop:
            frame = cam.read(timeout=1.0)
            timestamp_ms = int(time.time() * 1000)

            # --- Check input actions ---
            if cfg["MODE"] == "setting":
                try:
                    action = action_queue.get_nowait()
                    print(f"[Main] Processing action: {action}")

                    if action == "reload_config":
                        print("[Main] Reloading blendshape config...")
                        blendshape_post_processor.load_config()
                    elif action == "toggle_post_processing":
                        cfg["POST_PROCESS_BLENDSHAPES"] = not cfg["POST_PROCESS_BLENDSHAPES"]
                        print(f"[Main] POST_PROCESS_BLENDSHAPES = {cfg['POST_PROCESS_BLENDSHAPES']}")
                    elif action == "toggle_emotions":
                        blendshape_post_processor.apply_emotions = not blendshape_post_processor.apply_emotions
                        print(f"[Main] apply_emotions = {blendshape_post_processor.apply_emotions}")
                    elif action == "toggle_global_overrides":
                        blendshape_post_processor.apply_global_overrides = not blendshape_post_processor.apply_global_overrides
                        print(f"[Main] apply_global_overrides = {blendshape_post_processor.apply_global_overrides}")
                    elif action == "toggle_global_expression_overrides":
                        cfg["GLOBAL_EXPRESSION_OVERRIDES"] = not cfg["GLOBAL_EXPRESSION_OVERRIDES"]
                        print(f"[Main] GLOBAL_EXPRESSION_OVERRIDES = {cfg['GLOBAL_EXPRESSION_OVERRIDES']}")
                    elif action == "toggle_emotion_recognition_weights":
                        cfg["EMOTION_RECOGNITION_WEIGHTS"] = not cfg["EMOTION_RECOGNITION_WEIGHTS"]
                        print(f"[Main] EMOTION_RECOGNITION_WEIGHTS = {cfg['EMOTION_RECOGNITION_WEIGHTS']}")
                    elif action == "quit":
                        print("[Main] Quit requested")
                        run_main_loop = False
                        continue
                except queue.Empty:
                    pass

            if frame is None:
                scheduler.wait_for_next_frame()
                continue

            cropped_frame = cropper.crop_center_square(frame)
            if cropped_frame is not None:
                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                face_processor.process(rgb_frame, timestamp_ms)
                emotion_thread.update_frame(rgb_frame)

            final_pose_angles = None
            if latest_transform_matrix is not None:
                final_pose_angles = head_processor.process(latest_transform_matrix)
                processed_pose_angles_for_display = final_pose_angles

            # --- Emotion Detection ---
            emotion_scores_list = emotion_thread.get_latest_result()
            detected_emotion_label = "Neutral"
            if emotion_scores_list:
                first_face_emotion_scores = emotion_scores_list[0]
                detected_emotion_label = emotion_thread.recognizer.get_likeliest_emotion(
                    first_face_emotion_scores,
                    default_emotion="Neutral"
                )
                if not cfg["EMOTION_RECOGNITION_WEIGHTS"]:
                    detected_emotion_label = "Neutral"
            # --- Blendshape post-processing ---
            blendshapes_to_send = latest_blendshapes
            if cfg["POST_PROCESS_BLENDSHAPES"] and latest_blendshapes is not None:
                # Apply general blendshape post-processing first
                blendshapes_to_send = blendshape_post_processor.process(
                    latest_blendshapes,
                    detected_emotion_label
                )
                if not cfg["GLOBAL_EXPRESSION_OVERRIDES"]:
                    blendshapes_to_send = latest_blendshapes

                # --- Eye stabilization / pairing ---
                if blendshapes_to_send is not None:
                    # Convert from list of Category to dict for eye processor
                    blendshape_dict = {bs.category_name: bs.score for bs in blendshapes_to_send}
                    blendshape_dict = eye_post_processor.process(blendshape_dict)

                    # Convert back to list of Category for LiveLink
                    blendshapes_to_send = [
                        type(blendshapes_to_send[0])(bs.index, blendshape_dict[bs.category_name], bs.display_name, bs.category_name)
                        for bs in blendshapes_to_send
                    ]

            sender.update_head_pose(final_pose_angles)
            sender.update_blendshapes(blendshapes_to_send if blendshapes_to_send is not None else None)

            if cfg["DISPLAY_VIDEO"] and cropped_frame is not None:
                display_frame = draw_overlay(
                    cropped_frame,
                    None,
                    blendshapes_to_send,
                    target_size=cfg["TARGET_SIZE"],
                    landmarks=latest_landmarks,
                    head_pose=processed_pose_angles_for_display,
                    emotions=emotion_scores_list
                )
                cv2.imshow("Webcam Feed", display_frame)
                cv2.waitKey(1)

            if cfg["SHOW_FPS"]:
                elapsed = time.time() - (timestamp_ms / 1000.0)
                print(f"[Main] Frame time: {elapsed:.3f}s")

            scheduler.wait_for_next_frame()

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")
    finally:
        print("[Main] Cleaning up...")
        if cam.is_alive(): cam.stop()
        if hasattr(face_processor, "close"): face_processor.close()
        if sender.is_alive():
            sender.stop()
            sender.join(timeout=1)
        if emotion_thread.is_alive():
            emotion_thread.stop()
            emotion_thread.join(timeout=1)
        if cfg["MODE"] == "setting" and input_handler.is_alive():
            input_handler.stop()
            input_handler.join(timeout=1)
        if cfg["DISPLAY_VIDEO"]:
            cv2.destroyAllWindows()
        print("[Main] Cleanup finished.")

if __name__ == "__main__":
    main()
