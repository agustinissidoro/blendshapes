# utils/input_handler.py
import threading
import queue
import time
from pynput import keyboard



class InputHandler(threading.Thread):
    """
    Listens for specific keyboard inputs in a separate thread and queues actions
    for the main thread to process, avoiding blocking.
    """
    def __init__(self, action_queue: queue.Queue):
        # Initialize as a daemon thread so it exits with the main program
        super().__init__(daemon=True, name="InputHandlerThread")
        self._action_queue = action_queue
        self._running = True
        self._listener = None

    def _on_press(self, key):
        """Callback function executed when a key is pressed."""
        if not self._running: # Check if thread should stop
            return False # Stops the pynput listener

        action = None
        # First, try checking for character keys like '5'
        try:
            char = getattr(key, 'char', None)
            if char == '5':
                action = "reload_config"
            # Add other character key actions here using elif char == '...'
            # Removed 'q' key check from here
        except Exception:
             # Ignore errors getting char attribute (happens for special keys)
             pass

        # If no character action was found, check for special keys like Esc
        if not action and hasattr(key, 'name'): # Check PYNPUT_AVAILABLE here too
            if key == keyboard.Key.esc: # Check for Escape key
                 action = "quit"        # Assign the quit action
            # Add other special key actions here if needed
            # elif key == keyboard.Key.f1:
            #     action = "do_something_else"

        # If an action was identified ('5' or Esc), process it
        if action:
            # print(f"[InputHandler] Action detected: {action}") # Optional debug
            self._action_queue.put(action)
            # If the action is to quit, signal the thread/listener to stop
            if action == "quit":
                self._running = False # Signal run loop to exit eventually
                return False # Stops the pynput listener immediately

        # Keep listener running if no stop condition met
        return self._running

    def run(self):
        """Starts the keyboard listener loop."""
        print("[InputHandler] Starting keyboard listener... (Press 'Esc' to quit, '5' to reload profiles)")
        try:
            # Create and start the listener within this thread
            with keyboard.Listener(on_press=self._on_press) as self._listener:
                # listener.join() blocks this thread until the listener stops
                self._listener.join()
        except ImportError:
             print("[InputHandler] pynput check failed inside run.")
        except Exception as e:
             # This often indicates permission issues
             print(f"[InputHandler] Error starting keyboard listener: {e}")
             print("!!! Input handling might require special permissions !!!")
        finally:
             # print("[InputHandler] Keyboard listener stopped.") # Optional debug
             self._running = False # Ensure flag is false on exit

    def stop(self):
        """Signals the thread and listener to stop."""
        if self._running:
             # print("[InputHandler] Stopping request received...") # Optional debug
             pass
        self._running = False
        if self._listener:
            # Stop the listener instance if it exists
            # Use pynput's method - it should handle threading correctly
            try:
                 keyboard.Listener.stop(self._listener)
            except Exception as e:
                 print(f"[InputHandler] Error stopping listener via static method: {e}")
                 # Fallback attempt
                 try:
                     self._listener.stop()
                 except Exception as e2:
                     print(f"[InputHandler] Error stopping listener via instance method: {e2}")