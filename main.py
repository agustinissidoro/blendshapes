# main.py
import time
import json
import cv2
import numpy as np
from capture import VideoCaptureThread
from facelandmarks import FaceLandmarkerProcessor
from utils.face_cropper import FaceCropper
from utils.display_blendshapes import draw_overlay # Assuming this exists
from live_link_sender import LiveLinkSender      # Simplified sender
from udp_command_server import UdpCommandServer
from utils.head_pose import HeadPoseProcessor        # New processor class
from utils.scheduler import FrameScheduler
from utils.emotion_worker import EmotionWorker
from utils.expression_enhancer import BlendshapePostprocessor, EyePostProcessor
from utils.input_handler import InputHandler #
import queue 
import threading


# --- Configuration ---
CONFIG_PATH = "config.json"
FACE_MODEL_PATH = "./models/face_landmarker.task"
EMOTION_MODEL_PATH = "./models/enet_b0_8_best_afew.onnx"
EXPRESSION_CONFIG_PATH = "expression_profiles.json"

# Defaults (overridden by config.json)
DEFAULTS = {
    "SOURCE": 0,
    "FLIP_IMAGE": False,
    "TARGET_SIZE": 1024,
    "LIVE_LINK_IP": "192.168.100.2",
    "LIVE_LINK_PORT": 11111,
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
    "EVERY_FPS": 6,
    "EMOTION_RECOGNITION_ENABLED": True,
    "POST_PROCESS_BLENDSHAPES": True,
    "EYE_POST_PROCESSOR": True,
    "PAIR_EYELIDS": True,
    "UDP_COMMAND_IP": "127.0.0.1",
    "UDP_COMMAND_PORT": 12000
}

# --- Global variables for data exchange with MediaPipe callback ---
latest_blendshapes = None
latest_landmarks = None
latest_transform_matrix = None
latest_lock = threading.Lock()


def load_config(path=CONFIG_PATH):
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    for key, value in DEFAULTS.items():
        cfg.setdefault(key, value)
    return cfg


def handle_landmarker_result(result, image, timestamp_ms):
    global latest_blendshapes, latest_landmarks, latest_transform_matrix, latest_lock
    with latest_lock:
        latest_blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
        latest_landmarks = result.face_landmarks[0] if result.face_landmarks else None
        latest_transform_matrix = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None

def main():
    global latest_blendshapes, latest_transform_matrix, latest_landmarks # Ensure all used globals are listed

    cfg = load_config()
    eye_post_enabled = bool(cfg["EYE_POST_PROCESSOR"])
    emotion_enabled = bool(cfg["EMOTION_RECOGNITION_ENABLED"])

    cropper = FaceCropper(target_size=cfg["TARGET_SIZE"])
    scheduler = FrameScheduler(cfg["TARGET_FPS"])
    
    # Initialize EmotionWorker (this loads the EmotionRecognizer internally)
    emotion_thread = None
    if emotion_enabled:
        emotion_thread = EmotionWorker(model_path=EMOTION_MODEL_PATH, process_every_n=cfg["EVERY_FPS"])
        emotion_thread.start()

    head_processor = HeadPoseProcessor(
        filter_window_size=cfg["HP_FILTER_WINDOW"],
        max_yaw_deg=cfg["HP_MAX_YAW"],
        max_pitch_deg=cfg["HP_MAX_PITCH"],
        max_roll_deg=cfg["HP_MAX_ROLL"],
        yaw_offset_deg=cfg["HP_YAW_OFFSET"],
        pitch_offset_deg=cfg["HP_PITCH_OFFSET"],
        euler_order=cfg["HP_EULER_ORDER"]
    )
    
    blendshape_post_processor = BlendshapePostprocessor(config_path=EXPRESSION_CONFIG_PATH)
    eye_post_processor = EyePostProcessor(config_path=EXPRESSION_CONFIG_PATH)
    
    sender = LiveLinkSender(
        cfg["LIVE_LINK_IP"],
        cfg["LIVE_LINK_PORT"],
        swap_left_right=cfg["BLENDSHAPE_SWAP_LR"],
        target_fps=cfg["TARGET_FPS"],
        pair_eyelids=cfg["PAIR_EYELIDS"]
    )

    cam = VideoCaptureThread(src=cfg["SOURCE"], flip_image=cfg["FLIP_IMAGE"])
    
    face_processor = FaceLandmarkerProcessor(
        model_path=FACE_MODEL_PATH,
        result_callback=handle_landmarker_result
    )
    
    # --- Setup and start Input Handler ---
    action_queue = queue.Queue()
    input_handler = InputHandler(action_queue)
    # --- ---

    def handle_udp_command(address, args):
        addr = address.lower()
        if addr in ("/livelink/normal", "/livelink/start"):
            sender.set_mode("normal")
            print("[UDP] LiveLink mode: normal")
        elif addr in ("/livelink/neutral", "/livelink/stop"):
            sender.set_mode("neutral")
            print("[UDP] LiveLink mode: neutral")
        elif addr in ("/livelink/random",):
            sender.set_mode("random")
            if args:
                sender.set_random_rate(args[0])
                print(f"[UDP] LiveLink mode: random (rate={args[0]})")
            else:
                print("[UDP] LiveLink mode: random")
        elif addr in ("/livelink/random_rate",):
            if args:
                sender.set_random_rate(args[0])
                print(f"[UDP] LiveLink random rate set to {args[0]}")
        elif addr in ("/livelink/random_slow",):
            sender.set_mode("random")
            sender.set_random_rate(1.0)
            print("[UDP] LiveLink mode: random (slow)")
        elif addr in ("/livelink/random_fast",):
            sender.set_mode("random")
            sender.set_random_rate(cfg["TARGET_FPS"])
            print("[UDP] LiveLink mode: random (fast)")
        elif addr in ("/livelink/blink_right",):
            if args:
                sender.set_blink_right(bool(args[0]))
                print(f"[UDP] Blink right set to {bool(args[0])}")
            else:
                sender.toggle_blink_right()
                print("[UDP] Blink right toggled")
        elif addr in ("/livelink/tongue_out",):
            if args:
                sender.set_tongue_out(bool(args[0]))
                print(f"[UDP] Tongue out set to {bool(args[0])}")
            else:
                sender.toggle_tongue_out()
                print("[UDP] Tongue out toggled")
        else:
            print(f"[UDP] Unhandled command: {address} {args}")

    udp_server = UdpCommandServer(cfg["UDP_COMMAND_IP"], cfg["UDP_COMMAND_PORT"], handle_udp_command)

    sender.start()
    cam.start()
    input_handler.start()
    udp_server.start()
    print("Processing started...")

    processed_pose_angles_for_display = None # For overlay display
    overlay_canvas = None
    if cfg["DISPLAY_VIDEO"]:
        canvas_width = cfg["TARGET_SIZE"] + 1000
        canvas_height = cfg["TARGET_SIZE"]
        overlay_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    run_main_loop = True
    
    try:
        while run_main_loop:
            frame = cam.read(timeout=1.0)
            frame_processing_start_time = time.time()
            timestamp_ms = int(frame_processing_start_time * 1000)
            
            try:
                action = action_queue.get_nowait() # Check queue without waiting
                if action == "reload_config":
                    print("[Main] Action: Reloading expression config...")
                    blendshape_post_processor.load_config()
                    eye_post_processor.load_config()
                elif action == "quit":
                    print("[Main] Action: Quit requested via keyboard.")
                    run_main_loop = False
                    continue 
            except queue.Empty:
                pass
            # --- End Action Check ---

            with latest_lock:
                current_blendshapes = latest_blendshapes
                current_landmarks = latest_landmarks
                current_transform = latest_transform_matrix

            if frame is None:
                scheduler.wait_for_next_frame() # Ensure scheduler still runs
                continue

            cropped_frame = cropper.crop_center_square(frame)
            if cropped_frame is not None:
                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                face_processor.process(rgb_frame, timestamp_ms) # Triggers handle_landmarker_result
                if emotion_enabled and emotion_thread is not None:
                    emotion_thread.update_frame(rgb_frame) # Give frame to emotion worker

            # --- Head Pose Processing ---
            final_pose_angles = None
            if current_transform is not None:
                final_pose_angles = head_processor.process(current_transform)
                processed_pose_angles_for_display = final_pose_angles # For overlay

            # --- Emotion Detection & Blendshape Post-Processing ---
            emotion_scores_list = emotion_thread.get_latest_result() if (emotion_enabled and emotion_thread is not None) else None
            detected_emotion_label = "Neutral" # Default

            if emotion_scores_list: # Check if list is not None and not empty
                first_face_emotion_scores = emotion_scores_list[0]
                detected_emotion_label = emotion_thread.recognizer.get_likeliest_emotion(
                    first_face_emotion_scores, # Pass the dict for the first face
                    default_emotion="Neutral"
                )

            blendshapes_to_send = current_blendshapes 
            if cfg["POST_PROCESS_BLENDSHAPES"] and current_blendshapes is not None:
                blendshapes_to_send = blendshape_post_processor.process(
                    current_blendshapes,
                    detected_emotion_label 
                )

            if eye_post_enabled and blendshapes_to_send is not None and len(blendshapes_to_send) > 0:
                blendshape_dict = {bs.category_name: bs.score for bs in blendshapes_to_send}
                blendshape_dict = eye_post_processor.process(blendshape_dict)
                blendshapes_to_send = [
                    type(blendshapes_to_send[0])(bs.index, blendshape_dict[bs.category_name], bs.display_name, bs.category_name)
                    for bs in blendshapes_to_send
                ]

            # --- Sending Data ---
            sender.update_head_pose(final_pose_angles)
            sender.update_blendshapes(blendshapes_to_send if blendshapes_to_send is not None else None)

            # --- Display Logic ---
            if cfg["DISPLAY_VIDEO"]:
                display_frame = cropped_frame
                if cropped_frame is not None:
                    # Ensure draw_overlay can handle the emotion string
                    display_frame = draw_overlay(
                        cropped_frame,
                        None, # No raw head pose object here
                        current_blendshapes, # Could pass blendshapes_to_send to see processed ones
                        target_size=cfg["TARGET_SIZE"],
                        landmarks=current_landmarks,
                        head_pose=processed_pose_angles_for_display, # Normalized angles or None
                        emotions=emotion_scores_list, # Pass the detected emotion string
                        canvas=overlay_canvas
                    )
                    overlay_canvas = display_frame
                if display_frame is not None:
                    cv2.imshow("Webcam Feed", display_frame)
                    cv2.waitKey(1)
            # --- End Display ---

            if cfg["SHOW_FPS"]:
                # Measure processing time from after cam.read() to before scheduler.wait()
                elapsed_time = time.time() - frame_processing_start_time
                print(f"Frame processed in {elapsed_time:.4f} seconds.")

            scheduler.wait_for_next_frame()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Cleaning up...")
        if isinstance(cam, VideoCaptureThread) and cam.is_alive(): 
            cam.stop()
        if 'face_processor' in locals() and isinstance(face_processor, FaceLandmarkerProcessor):
             if hasattr(face_processor, 'close') and callable(getattr(face_processor, 'close', None)):
                 face_processor.close()
        if isinstance(sender, LiveLinkSender) and sender.is_alive():
            sender.stop()
            sender.join(timeout=1.0)
        if isinstance(emotion_thread, EmotionWorker) and emotion_thread.is_alive():
            emotion_thread.stop()
            emotion_thread.join(timeout=1.0)
        if isinstance(input_handler, InputHandler) and input_handler.is_alive():
            input_handler.stop()
            input_handler.join(timeout=1)
        if 'udp_server' in locals() and isinstance(udp_server, UdpCommandServer) and udp_server.is_alive():
            udp_server.stop()
            udp_server.join(timeout=1)
        if cfg["DISPLAY_VIDEO"]:
            cv2.destroyAllWindows()
        print("Cleanup finished.")

if __name__ == "__main__":
    main()
