# main.py
import time
import cv2
import numpy as np
from capture import VideoCaptureThread
from facelandmarks import FaceLandmarkerProcessor
from utils.face_cropper import FaceCropper
from utils.display_blendshapes import draw_overlay # Assuming this exists
from live_link_sender import LiveLinkSender      # Simplified sender
from utils.head_pose import HeadPoseProcessor        # New processor class
from utils.scheduler import FrameScheduler
from utils.emotion_worker import EmotionWorker
from utils.emotion_classification import EmotionRecognizer
from utils.expression_enhancer import BlendshapePostprocessor
from utils.input_handler import InputHandler #
import cProfile
import queue 


# --- Configuration ---
TARGET_SIZE = 1024
LIVE_LINK_IP = "192.168.100.2" # Your Unreal target IP
LIVE_LINK_PORT = 11111
FACE_MODEL_PATH = "./models/face_landmarker.task" # Your model path
TARGET_FPS = 30
DISPLAY_VIDEO = False # Set True to show OpenCV window
SHOW_FPS = False    # Set True to print FPS info

# --- Head Pose Processor Configuration ---
# Tune these parameters as needed
HP_FILTER_WINDOW = 7
HP_MAX_YAW = 45.0
HP_MAX_PITCH = 20.0
HP_MAX_ROLL = 45.0
HP_PITCH_OFFSET = 0.0 # Determine by looking straight and checking processor output
HP_EULER_ORDER = 'yxz' # Experiment with 'xyz', 'zyx' - affects axis mapping needs

# --- Emotion Classification ---
EMOTION_MODEL_PATH = "./models/enet_b0_8_best_afew.onnx"
EVERY_FPS = 6

# --- Blendshape post-processing ---
POST_PROCESS_BLENDSHAPES = True
CONFIG_JSON_PATH = "expression_profiles.json"

# --- Global variables for data exchange with MediaPipe callback ---
latest_blendshapes = None
latest_landmarks = None
latest_transform_matrix = None


def handle_landmarker_result(result, image, timestamp_ms):
    global latest_blendshapes, latest_landmarks, latest_transform_matrix
    latest_blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None
    latest_landmarks = result.face_landmarks[0] if result.face_landmarks else None
    latest_transform_matrix = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None

def main():
    global latest_blendshapes, latest_transform_matrix, latest_landmarks # Ensure all used globals are listed

    cropper = FaceCropper(target_size=TARGET_SIZE)
    scheduler = FrameScheduler(TARGET_FPS)
    
    # Initialize EmotionWorker (this loads the EmotionRecognizer internally)
    emotion_thread = EmotionWorker(model_path=EMOTION_MODEL_PATH, process_every_n=EVERY_FPS)
    emotion_thread.start()

    head_processor = HeadPoseProcessor(
        filter_window_size=HP_FILTER_WINDOW,
        max_yaw_deg=HP_MAX_YAW,
        max_pitch_deg=HP_MAX_PITCH,
        max_roll_deg=HP_MAX_ROLL,
        pitch_offset_deg=HP_PITCH_OFFSET,
        euler_order=HP_EULER_ORDER
    )
    
    blendshape_post_processor = BlendshapePostprocessor(config_path=CONFIG_JSON_PATH)
    
    sender = LiveLinkSender(LIVE_LINK_IP, LIVE_LINK_PORT)

    cam = VideoCaptureThread(src=0, flip_image=False)
    
    face_processor = FaceLandmarkerProcessor(
        model_path=FACE_MODEL_PATH,
        result_callback=handle_landmarker_result
    )
    
    # --- Setup and start Input Handler ---
    action_queue = queue.Queue()
    input_handler = InputHandler(action_queue)
    # --- ---

    sender.start()
    cam.start()
    input_handler.start()
    print("Processing started...")

    processed_pose_angles_for_display = None # For overlay display

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
                elif action == "quit":
                    print("[Main] Action: Quit requested via keyboard.")
                    run_main_loop = False
                    continue 
            except queue.Empty:
                pass
            # --- End Action Check ---

            if frame is None:
                scheduler.wait_for_next_frame() # Ensure scheduler still runs
                continue

            cropped_frame = cropper.crop_center_square(frame)
            if cropped_frame is not None:
                rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                face_processor.process(rgb_frame, timestamp_ms) # Triggers handle_landmarker_result
                emotion_thread.update_frame(rgb_frame) # Give frame to emotion worker

            # --- Head Pose Processing ---
            final_pose_angles = None
            if latest_transform_matrix is not None:
                final_pose_angles = head_processor.process(latest_transform_matrix)
                processed_pose_angles_for_display = final_pose_angles # For overlay

            # --- Emotion Detection & Blendshape Post-Processing ---
            emotion_scores_list = emotion_thread.get_latest_result() # List[Dict[str, float]] or None
            detected_emotion_label = "Neutral" # Default

            if emotion_scores_list: # Check if list is not None and not empty
                first_face_emotion_scores = emotion_scores_list[0]
                detected_emotion_label = emotion_thread.recognizer.get_likeliest_emotion(
                    first_face_emotion_scores, # Pass the dict for the first face
                    default_emotion="Neutral"
                )

            blendshapes_to_send = latest_blendshapes 
            if POST_PROCESS_BLENDSHAPES and latest_blendshapes is not None:
                blendshapes_to_send = blendshape_post_processor.process(
                    latest_blendshapes,
                    detected_emotion_label 
                )

            # --- Sending Data ---
            sender.update_head_pose(final_pose_angles)
            sender.update_blendshapes(blendshapes_to_send if blendshapes_to_send is not None else None)

            # --- Display Logic ---
            if DISPLAY_VIDEO:
                display_frame = cropped_frame
                if cropped_frame is not None:
                    # Ensure draw_overlay can handle the emotion string
                    display_frame = draw_overlay(
                        cropped_frame,
                        None, # No raw head pose object here
                        latest_blendshapes, # Could pass blendshapes_to_send to see processed ones
                        target_size=TARGET_SIZE,
                        landmarks=latest_landmarks,
                        head_pose=processed_pose_angles_for_display, # Normalized angles or None
                        emotions=emotion_scores_list # Pass the detected emotion string
                    )
                if display_frame is not None:
                    cv2.imshow("Webcam Feed", display_frame)
                    cv2.waitKey(1)
            # --- End Display ---

            if SHOW_FPS:
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
        if DISPLAY_VIDEO:
            cv2.destroyAllWindows()
        print("Cleanup finished.")

if __name__ == "__main__":
    main()

