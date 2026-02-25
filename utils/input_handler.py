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
        self._shift_pressed = False

    def _on_press(self, key):
        if not self._running:
            return False

        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._shift_pressed = True
            return self._running

        action = None
        # Character keys
        try:
            char = getattr(key, 'char', None)
            if char == '0':
                action = "reload_config"
            elif char in ("f", "F"):
                action = "toggle_tracking"
        except Exception:
            pass

        # Special keys
        if not action and hasattr(key, 'name'):
            if key == keyboard.Key.esc and self._shift_pressed:
                action = "quit"

        # Queue the action and print it
        if action:
            print(f"[InputHandler] Action detected: {action}")
            self._action_queue.put(action)
            if action == "quit":
                self._running = False
                return False

        return self._running

    def _on_release(self, key):
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self._shift_pressed = False
        return self._running

    def run(self):
        """Starts the keyboard listener loop."""
        print("[InputHandler] Starting keyboard listener... (Press '0' to reload config, 'f' to toggle tracking, 'Shift+Esc' to quit)")
        try:
            with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as self._listener:
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
