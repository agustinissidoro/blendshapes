import time
import cv2
from capture import VideoCaptureThread
from facelandmarks import FaceLandmarkerProcessor
from utils.face_cropper import FaceCropper
from utils.display_blendshapes import draw_overlay
from live_link_sender import LiveLinkSender
from utils.scheduler import FrameScheduler
from utils.head_pose import get_head_pose


target_size = 512
cropper = FaceCropper(target_size=target_size)
latest_blendshapes=[]
latest_landmarks=[]
latest_transform_matrix = None

sender = LiveLinkSender("192.168.100.2", 11111, target_size=target_size)
sender.start()

target_fps = 30
scheduler = FrameScheduler(target_fps)
display = False
show_fps = False

def handle_landmarker_result(result, image, timestamp_ms):
    global latest_blendshapes, latest_landmarks, latest_transform_matrix
    # Get Blendshapes
    if result.face_blendshapes and result.face_blendshapes[0]:
        latest_blendshapes = result.face_blendshapes[0]
    else:
        latest_blendshapes = None
    # Get Landmarks (optional, for display)
    if result.face_landmarks and result.face_landmarks[0]:
        latest_landmarks = result.face_landmarks[0]
    else:
        latest_landmarks = None
    # --- Get Transformation Matrix ---
    if result.facial_transformation_matrixes and result.facial_transformation_matrixes[0]:
        latest_transform_matrix = result.facial_transformation_matrixes[0]
    else:
        latest_transform_matrix = None
    # --- End getting matrix ---
        
def main():
    global target_size
    global latest_blendshapes
    global display
    global show_fps
    cropped = None
    
    cam = VideoCaptureThread(flip_image=False)
    cam.start()

    face_processor = FaceLandmarkerProcessor(
        model_path="./models/face_landmarker.task",
        result_callback=handle_landmarker_result
    )

    try:
        while True:
            start_time = time.time()
            frame = cam.read(timeout=1.0)
            timestamp_ms = int(time.time() * 1000)
            
            if frame is None:
                continue
            
            cropped = cropper.crop_center_square(frame)
            if cropped is not None:
                face_processor.process(cropped, timestamp_ms)
                
            if latest_landmarks is not None:
                head_pose = get_head_pose(latest_landmarks, (target_size, target_size))
                yaw_disp, pitch_disp, roll_disp, _, _, rvec = head_pose
                rotation_vector_to_send = rvec # Get the rotation vector
                head_pose_for_display = (yaw_disp, pitch_disp, roll_disp) # For overlay
                
            if latest_blendshapes is not None:
                sender.update_blendshapes(latest_blendshapes)
                sender.update_head_pose(rotation_vector_to_send)
            
            if display:   
                if latest_blendshapes is not None:
                    display_frame = draw_overlay(cropped, None, latest_blendshapes, target_size=target_size, landmarks=latest_landmarks, head_pose=head_pose_for_display)
                else:
                    display_frame = cropped

                cv2.imshow("Webcam", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            elapsed_time = time.time() - start_time
            if show_fps:
                print(f"Frame processed in {elapsed_time:.4f} seconds.")
            scheduler.wait_for_next_frame()


    except KeyboardInterrupt:
        print("Interrupted by user.")

    # Cleanup
    cam.stop()
    face_processor.close()
    cv2.destroyAllWindows()
    sender.stop()
    sender.join()

if __name__ == "__main__":
    main()
