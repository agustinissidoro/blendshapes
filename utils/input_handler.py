# utils/input_handler.py
import threading
import queue
from pynput import keyboard

class InputHandler(threading.Thread):
    """
    Listens for specific keyboard inputs in a separate thread and queues actions
    for the main thread to process, avoiding blocking.
    """
    def __init__(self, action_queue: queue.Queue):
        super().__init__(daemon=True, name="InputHandlerThread")
        self._action_queue = action_queue
        self._running = True
        self._listener = None

    def _on_press(self, key):
        if not self._running:
            return False

        action = None
        # Character keys
        try:
            char = getattr(key, 'char', None)
            if char == '5':
                action = "reload_config"
            elif char == '6':
                action = "toggle_post_processing"
            elif char == '7':
                action = "toggle_emotions"
            elif char == '8':
                action = "toggle_global_overrides"
            elif char == 'h':
                action = "toggle_global_expression_overrides"
            elif char == 'j':
                action = "toggle_emotion_recognition_weights"
        except Exception:
            pass

        # Special keys
        if not action and hasattr(key, 'name'):
            if key == keyboard.Key.esc:
                action = "quit"

        # Queue the action and print it
        if action:
            print(f"[InputHandler] Action detected: {action}")
            self._action_queue.put(action)
            if action == "quit":
                self._running = False
                return False

        return self._running

    def run(self):
        """Starts the keyboard listener loop."""
        print("[InputHandler] Starting keyboard listener... (Press 'Esc' to quit, '5-8', 'h', 'j' for actions)")
        try:
            with keyboard.Listener(on_press=self._on_press) as self._listener:
                self._listener.join()
        except ImportError:
            print("[InputHandler] pynput check failed inside run.")
        except Exception as e:
            print(f"[InputHandler] Error starting keyboard listener: {e}")
            print("!!! Input handling might require special permissions !!!")
        finally:
            self._running = False

    def stop(self):
        """Signals the thread and listener to stop."""
        self._running = False
        if self._listener:
            try:
                keyboard.Listener.stop(self._listener)
            except Exception as e:
                print(f"[InputHandler] Error stopping listener via static method: {e}")
                try:
                    self._listener.stop()
                except Exception as e2:
                    print(f"[InputHandler] Error stopping listener via instance method: {e2}")
