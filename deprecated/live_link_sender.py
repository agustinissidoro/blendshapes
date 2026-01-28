import socket
import threading
from typing import List, Set, Optional, Tuple
import time
import numpy as np
from pylivelinkface import PyLiveLinkFace, FaceBlendShape
from mediapipe.tasks.python.components.containers.category import Category


class LiveLinkSender(threading.Thread):
    def __init__(self, ip: str, port: int):
        super().__init__()
        self.ip = ip
        self.port = port
        self.running = False
        self._face = PyLiveLinkFace() # Holds the state (blendshapes + pose) to be sent
        # State for facial blendshape reset logic (if used by update_blendshapes)
        self._latest_blendshapes: List[Category] = []
        self._previous_facial_keys: Set[str] = set()

        # Socket setup
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._socket.connect((self.ip, self.port))
            print(f"[LiveLinkSender] Socket targeting {ip}:{port}")
        except socket.error as e:
            print(f"[LiveLinkSender] Socket connect error: {e}")
            self._socket = None # Mark as unusable

    def run(self):
        if not self._socket:
            print("[LiveLinkSender] Cannot run, socket not connected.")
            return
        self.running = True
        print("[LiveLinkSender] Sender thread started.")
        try:
            while self.running:
                self._send_data() # Periodically send the current face state
                time.sleep(0.016) # Adjust rate (~60fps)
        except socket.error as se:
             # Stop thread silently on socket error
             self.running = False
        except Exception as e:
            print(f"[LiveLinkSender] Error in run loop: {e}")
            self.running = False
        finally:
            if self._socket:
                self._socket.close()
            print("[LiveLinkSender] Sender thread stopped.")

    def stop(self):
        if self.running:
             print("[LiveLinkSender] Stopping sender...")
        self.running = False

    def update_blendshapes(self, blendshapes: Optional[List[Category]]):
        """Updates the facial expression blendshapes (indices 0-51) in the face state."""
        if not self.running: return

        current_keys = set()
        if blendshapes:
            for category in blendshapes:
                name = category.category_name
                if not name: continue
                enum_key_name = name[0].upper() + name[1:]
                try:
                    enum_key = FaceBlendShape[enum_key_name]
                    # Only update facial blendshapes (0-51) here
                    if 0 <= enum_key.value <= 51:
                        score = float(category.score)
                        self._face.set_blendshape(enum_key, score, no_filter=True)
                        current_keys.add(enum_key_name)
                except KeyError: continue # Ignore unmapped keys
                except ValueError: continue # Ignore invalid scores

        # Reset only facial keys previously set by this method
        keys_to_reset = self._previous_facial_keys - current_keys
        for prev_key in keys_to_reset:
             try:
                 enum_key = FaceBlendShape[prev_key]
                 if 0 <= enum_key.value <= 51:
                      self._face.set_blendshape(enum_key, 0.0, no_filter=True)
             except KeyError: continue

        self._previous_facial_keys = current_keys

    # --- THIS IS THE CORRECTED, SIMPLIFIED VERSION ---
    def update_head_pose(self, final_pose_angles: Optional[Tuple[float, float, float]]):
        """Updates face state with final processed & normalized head pose angles."""
        if not self.running: return

        # Use 0.0 as default if input tuple is None
        yaw_final, pitch_final, roll_final = final_pose_angles if final_pose_angles else (0.0, 0.0, 0.0)

        # Directly update face object state with the final values
        # Input should already be clamped [-1, 1] by HeadPoseProcessor
        # Add float conversion and clipping as a safety measure
        try:
            yaw_final_safe = np.clip(float(yaw_final), -1.0, 1.0)
            pitch_final_safe = np.clip(float(pitch_final), -1.0, 1.0)
            roll_final_safe = np.clip(float(roll_final), -1.0, 1.0)
        except (ValueError, TypeError):
            yaw_final_safe, pitch_final_safe, roll_final_safe = 0.0, 0.0, 0.0


        self._face.set_blendshape(FaceBlendShape.HeadYaw, yaw_final_safe, no_filter=True)
        self._face.set_blendshape(FaceBlendShape.HeadPitch, pitch_final_safe, no_filter=True)
        self._face.set_blendshape(FaceBlendShape.HeadRoll, roll_final_safe, no_filter=True)
    # --- END OF METHOD ---

    def _send_data(self):
        """Encodes the current self._face state and sends it over UDP."""
        if not self.running or not self._socket: return
        try:
            payload = self._face.encode()
            self._socket.sendall(payload)
        except socket.error:
            # Connection might be lost, stop trying to send
            self.running = False
        except Exception as e:
            print(f"[LiveLinkSender] Error encoding/sending data: {e}")
            self.running = False # Stop on other errors too

    # Keep this method name if run() calls it
    def _send_blendshapes(self):
         self._send_data()