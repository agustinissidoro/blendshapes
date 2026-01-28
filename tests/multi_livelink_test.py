# multi_livelink_test.py
import socket
import time
import random
import threading
import uuid
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
main_project_dir = os.path.dirname(current_script_dir)
sys.path.insert(0, main_project_dir)

# Assuming pylivelinkface.py is in the same directory or accessible
try:
    from pylivelinkface import PyLiveLinkFace, FaceBlendShape
except ImportError:
    print("ERROR: Could not import pylivelinkface.")
    print("Please ensure pylivelinkface.py is in the same directory or your Python path.")
    sys.exit(1)


# --- Configuration ---
TARGET_IP = "192.168.100.2"  # Target IP address for Unreal Engine PC
BASE_PORT = 11111           # Starting port number
NUM_CLIENTS = 4             # Number of simulated clients/Metahumans
TARGET_FPS = 30             # Target frames per second
SLEEP_DURATION = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0.016 # Time between sends

# --- Threading Control ---
# Using a global flag to signal threads to stop
running = True
threads = []

def sender_worker(client_id: int, ip: str, port: int):
    """Function run by each sender thread."""
    global running
    client_name = f"PythonClient_{client_id:02d}" # e.g., PythonClient_01
    # Generate a unique UUID for each client to distinguish them in Live Link
    client_uuid = str(uuid.uuid4())
    print(f"Starting client {client_id}: Name='{client_name}', UUID='{client_uuid}', Port={port}")

    # Create unique face data instance for this client
    # Use the client_name for the Live Link Subject Name
    py_face = PyLiveLinkFace(name=client_name, uuid=client_uuid)

    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No need to connect for UDP sendto, just specify destination each time
        # sock.connect((ip, port)) # Connect is optional for UDP sending

        while running:
            start_time = time.perf_counter()

            # --- Set random values for ALL blendshapes ---
            for blendshape in FaceBlendShape:
                value = 0.0
                # Facial shapes (Indices 0-51): Range 0.0 to 1.0
                if 0 <= blendshape.value <= 51:
                    value = random.uniform(0.0, 1.0)
                # Head rotation (Indices 52-54): Range -1.0 to 1.0 (matches normalized output)
                elif 52 <= blendshape.value <= 54:
                    value = random.uniform(-1.0, 1.0)
                # Eye rotation (Indices 55-60): Range -1.0 to 1.0 (adjust if different range needed)
                elif 55 <= blendshape.value <= 60:
                    value = random.uniform(-1.0, 1.0)

                py_face.set_blendshape(blendshape, value)
            # --- End setting blendshapes ---

            # Encode and send data
            try:
                payload = py_face.encode()
                # Use sendto for unconnected UDP socket
                sock.sendto(payload, (ip, port))
            except socket.error as e:
                # Don't spam errors if destination is unreachable
                # print(f"Client {client_id} socket error: {e}")
                time.sleep(1) # Wait if send fails
                # Optional: break if error persists?
            except Exception as e:
                 print(f"Client {client_id} encoding/other error: {e}")
                 break # Stop this thread on other errors

            # Calculate sleep time to maintain target FPS
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            sleep_needed = SLEEP_DURATION - elapsed_time
            if sleep_needed > 0:
                time.sleep(sleep_needed)

    except Exception as e:
        print(f"Client {client_id} encountered an unhandled error: {e}")
    finally:
        print(f"Stopping client {client_id}...")
        sock.close()

# --- Main Execution ---
if __name__ == "__main__":
    ports = [BASE_PORT + i for i in range(NUM_CLIENTS)]
    print(f"Starting {NUM_CLIENTS} Live Link test senders.")
    print(f"Target IP: {TARGET_IP}")
    print(f"Target Ports: {ports}")
    print(f"Target FPS: {TARGET_FPS} (~{SLEEP_DURATION*1000:.1f} ms interval)")
    print("Press Ctrl+C to stop.")

    # Create and start threads
    for i in range(NUM_CLIENTS):
        port = ports[i]
        thread = threading.Thread(target=sender_worker, args=(i + 1, TARGET_IP, port), daemon=True)
        threads.append(thread)
        thread.start()
        time.sleep(0.05) # Stagger thread starts slightly

    try:
        # Keep main thread alive while worker threads run
        while running:
            # Optional: Check if threads are alive periodically
            alive_threads = sum(1 for t in threads if t.is_alive())
            if alive_threads < NUM_CLIENTS and running:
                 print(f"Warning: Only {alive_threads}/{NUM_CLIENTS} sender threads are alive.")
                 # You could add logic here to restart threads if desired
                 # For now, we just let the main loop continue until Ctrl+C
            if alive_threads == 0:
                print("All worker threads stopped.")
                running = False # Exit main loop if all workers die
                break
            time.sleep(1.0) # Check thread status every second

    except KeyboardInterrupt:
        print("\nCtrl+C received. Signaling sender threads to stop...")
        running = False # Signal threads using the global flag

    # Wait for threads to finish cleaning up (closing sockets)
    print("Waiting for sender threads to finish...")
    for thread in threads:
        try:
            thread.join(timeout=1.5) # Wait max 1.5 seconds per thread
        except Exception as e:
             print(f"Error joining thread: {e}")


    print("All sender threads stopped. Exiting.")