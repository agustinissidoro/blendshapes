
import socket
import time
import random
import threading
import uuid
import sys
import os

# --- Path Setup ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
main_project_dir = os.path.dirname(current_script_dir)
sys.path.insert(0, main_project_dir)

try:
    from network.pylivelinkface import PyLiveLinkFace, FaceBlendShape
except ImportError:
    print("ERROR: Could not import network.pylivelinkface.")
    print("Please ensure the project root is on your Python path.")
    sys.exit(1)

# --- Configuration ---
TARGET_IP = "192.168.100.2"  # Target IP address for Unreal Engine PC
BASE_PORT = 11111           # Starting port number
NUM_CLIENTS = 4             # Number of simulated clients/Metahumans
TARGET_FPS = 30             # Target frames per second
SLEEP_DURATION = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0.016  # Time between sends

# --- Global State ---
running = True
threads = []
mode = 1
mode_lock = threading.Lock()

# --- Keyboard Listener Thread (using pynput) ---
def keyboard_listener():
    from pynput import keyboard

    global mode, running

    print("Keyboard controls: [0]=Zero | [1]=Random/Frame | [2]=Random/Second | [3]=Interpolated/5s | [ESC]=Quit")

    def on_press(key):
        global mode, running
        try:
            if key.char in ['0', '1', '2', '3']:
                with mode_lock:
                    mode = int(key.char)
                print(f">>> Switched to mode {mode}")
        except AttributeError:
            if key == keyboard.Key.esc:
                print(">>> ESC pressed. Stopping...")
                running = False
                return False  # Stop listener

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# --- Worker Thread Function ---
def sender_worker(client_id: int, ip: str, port: int):
    global running
    client_name = f"PythonClient_{client_id:02d}"
    client_uuid = str(uuid.uuid4())
    print(f"Starting client {client_id}: Name='{client_name}', UUID='{client_uuid}', Port={port}")

    py_face = PyLiveLinkFace(name=client_name, uuid=client_uuid)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # For modes 2 and 3
    last_update = 0
    last_random = {}
    r1 = {}
    r2 = {}
    interp_start = time.time()

    try:
        while running:
            start_time = time.perf_counter()
            now = time.time()

            with mode_lock:
                current_mode = mode

            # --- Mode Logic ---
            if current_mode == 0:
                current_values = {b: 0.0 for b in FaceBlendShape}

            elif current_mode == 1:
                current_values = {}
                for b in FaceBlendShape:
                    if 0 <= b.value <= 51:
                        current_values[b] = random.uniform(0.0, 1.0)
                    elif 52 <= b.value <= 60:
                        current_values[b] = random.uniform(-1.0, 1.0)

            elif current_mode == 2:
                if now - last_update > 1.0 or not last_random:
                    last_random = {
                        b: random.uniform(0.0, 1.0) if 0 <= b.value <= 51 else random.uniform(-1.0, 1.0)
                        for b in FaceBlendShape
                    }
                    last_update = now
                current_values = last_random

            elif current_mode == 3:
                interp_duration = 5.0
                if not r1:
                    r1 = {
                        b: random.uniform(0.0, 1.0) if 0 <= b.value <= 51 else random.uniform(-1.0, 1.0)
                        for b in FaceBlendShape
                    }
                    r2 = {
                        b: random.uniform(0.0, 1.0) if 0 <= b.value <= 51 else random.uniform(-1.0, 1.0)
                        for b in FaceBlendShape
                    }
                    interp_start = now

                if now - interp_start >= interp_duration:
                    r1 = r2
                    r2 = {
                        b: random.uniform(0.0, 1.0) if 0 <= b.value <= 51 else random.uniform(-1.0, 1.0)
                        for b in FaceBlendShape
                    }
                    interp_start = now

                t = (now - interp_start) / interp_duration
                t = max(0.0, min(t, 1.0))

                current_values = {
                    b: (1 - t) * r1[b] + t * r2[b]
                    for b in FaceBlendShape
                }

            else:
                current_values = {b: 0.0 for b in FaceBlendShape}

            # --- Set and Send ---
            for blendshape, value in current_values.items():
                py_face.set_blendshape(blendshape, value)

            try:
                payload = py_face.encode()
                sock.sendto(payload, (ip, port))
            except socket.error:
                time.sleep(1)
            except Exception as e:
                print(f"Client {client_id} encoding/other error: {e}")
                break

            elapsed = time.perf_counter() - start_time
            sleep_needed = SLEEP_DURATION - elapsed
            if sleep_needed > 0:
                time.sleep(sleep_needed)

    except Exception as e:
        print(f"Client {client_id} error: {e}")
    finally:
        print(f"Stopping client {client_id}...")
        sock.close()

# --- Main ---
if __name__ == "__main__":
    ports = [BASE_PORT + i for i in range(NUM_CLIENTS)]
    print(f"Starting {NUM_CLIENTS} Live Link test senders.")
    print(f"Target IP: {TARGET_IP}")
    print(f"Target Ports: {ports}")
    print(f"Target FPS: {TARGET_FPS} (~{SLEEP_DURATION*1000:.1f} ms interval)")
    print("Press Ctrl+C or [ESC] to stop.")

    # Start keyboard listener thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()

    # Start sender threads
    for i in range(NUM_CLIENTS):
        port = ports[i]
        thread = threading.Thread(target=sender_worker, args=(i + 1, TARGET_IP, port), daemon=True)
        threads.append(thread)
        thread.start()
        time.sleep(0.05)

    try:
        while running:
            alive = sum(1 for t in threads if t.is_alive())
            if alive < NUM_CLIENTS:
                print(f"Warning: Only {alive}/{NUM_CLIENTS} sender threads are alive.")
            if alive == 0:
                print("All sender threads stopped.")
                running = False
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping...")
        running = False

    print("Waiting for threads to finish...")
    for thread in threads:
        try:
            thread.join(timeout=1.5)
        except Exception as e:
            print(f"Error joining thread: {e}")

    print("All sender threads stopped. Exiting.")
