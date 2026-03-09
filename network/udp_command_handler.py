import queue
from typing import Any, Callable, List, Mapping, Optional


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "on", "yes", "y"):
            return True
        if normalized in ("0", "false", "off", "no", "n", ""):
            return False
    return bool(value)


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip()
        if normalized == "":
            return None
        try:
            return float(normalized)
        except ValueError:
            return None
    return None


def build_udp_command_handler(
    action_queue: queue.Queue,
    cfg: Mapping[str, Any],
    on_tracking: Optional[Callable[[Optional[bool]], None]] = None,
    on_get_state: Optional[Callable[[], None]] = None,
    on_set_headpose_offsets: Optional[Callable[[Optional[float], Optional[float], bool], None]] = None,
    on_reset_headpose_offsets: Optional[Callable[[], None]] = None,
    on_calibrate: Optional[Callable[[str], None]] = None,
) -> Callable[[str, List[Any]], None]:
    def handle_udp_command(address: str, args: List[Any]):
        addr = address.strip().lower()
        if addr in ("/livelink/normal", "/livelink/start"):
            action_queue.put(("sender_mode", "normal"))
            print("[UDP] LiveLink mode: normal")
        elif addr in ("/livelink/neutral", "/livelink/stop"):
            action_queue.put(("sender_mode", "neutral"))
            print("[UDP] LiveLink mode: neutral")
        elif addr in ("/livelink/random",):
            action_queue.put(("sender_mode", "random"))
            if args:
                action_queue.put(("sender_random_rate", args[0]))
                print(f"[UDP] LiveLink mode: random (rate={args[0]})")
            else:
                print("[UDP] LiveLink mode: random")
        elif addr in ("/livelink/random_rate",):
            if args:
                action_queue.put(("sender_random_rate", args[0]))
                print(f"[UDP] LiveLink random rate set to {args[0]}")
        elif addr in ("/livelink/random_slow",):
            action_queue.put(("sender_mode", "random"))
            action_queue.put(("sender_random_rate", 1.0))
            print("[UDP] LiveLink mode: random (slow)")
        elif addr in ("/livelink/random_fast",):
            action_queue.put(("sender_mode", "random"))
            action_queue.put(("sender_random_rate", cfg.get("SEND_FPS", cfg.get("TARGET_FPS", 30))))
            print("[UDP] LiveLink mode: random (fast)")
        elif addr in ("/livelink/blink_right",):
            if args:
                action_queue.put(("sender_blink_right", bool(args[0])))
                print(f"[UDP] Blink right set to {bool(args[0])}")
            else:
                action_queue.put("sender_blink_right_toggle")
                print("[UDP] Blink right toggled")
        elif addr in ("/livelink/tongue_out",):
            if args:
                action_queue.put(("sender_tongue_out", bool(args[0])))
                print(f"[UDP] Tongue out set to {bool(args[0])}")
            else:
                action_queue.put("sender_tongue_out_toggle")
                print("[UDP] Tongue out toggled")
        elif addr in (
            "/livelink/tracking",
            "livelink/tracking",
            "/livelink/trackiing",
            "livelink/trackiing",
            "/tracking",
            "tracking",
            "/trackiing",
            "trackiing",
            "/facetracking",
            "facetracking",
        ):
            desired_state = _coerce_bool(args[0]) if args else None
            if on_tracking is not None:
                on_tracking(desired_state)
            if desired_state is None:
                print("[UDP] /livelink/tracking toggle requested")
            else:
                print(f"[UDP] /livelink/tracking set requested: {int(desired_state)}")
        elif addr in ("/get_state", "get_state"):
            if on_get_state is not None:
                on_get_state()
            print("[UDP] State requested")
        elif addr in (
            "/livelink/headpose/offset/yaw",
            "livelink/headpose/offset/yaw",
            "/headpose/yaw_offset",
            "headpose/yaw_offset",
            "/livelink/headpose/yaw_offset",
            "livelink/headpose/yaw_offset",
        ):
            yaw_offset = _coerce_float(args[0]) if args else None
            if yaw_offset is None:
                print(f"[UDP] Invalid yaw offset command: {address} {args}")
                return
            if on_set_headpose_offsets is not None:
                on_set_headpose_offsets(yaw_offset, None, False)
            print(f"[UDP] Head pose yaw correction set requested: {yaw_offset:.3f} deg")
        elif addr in (
            "/livelink/headpose/offset/pitch",
            "livelink/headpose/offset/pitch",
            "/headpose/pitch_offset",
            "headpose/pitch_offset",
            "/livelink/headpose/pitch_offset",
            "livelink/headpose/pitch_offset",
        ):
            pitch_offset = _coerce_float(args[0]) if args else None
            if pitch_offset is None:
                print(f"[UDP] Invalid pitch offset command: {address} {args}")
                return
            if on_set_headpose_offsets is not None:
                on_set_headpose_offsets(None, pitch_offset, False)
            print(f"[UDP] Head pose pitch correction set requested: {pitch_offset:.3f} deg")
        elif addr in (
            "/livelink/headpose/offset/set",
            "livelink/headpose/offset/set",
            "/headpose/offsets",
            "headpose/offsets",
            "/livelink/headpose/offsets",
            "livelink/headpose/offsets",
        ):
            yaw_offset = _coerce_float(args[0]) if len(args) >= 1 else None
            pitch_offset = _coerce_float(args[1]) if len(args) >= 2 else None
            if yaw_offset is None and pitch_offset is None:
                print(f"[UDP] Invalid head offsets command: {address} {args}")
                return
            if on_set_headpose_offsets is not None:
                on_set_headpose_offsets(yaw_offset, pitch_offset, False)
            print(
                "[UDP] Head pose corrections set requested: "
                f"yaw={yaw_offset if yaw_offset is not None else 'unchanged'}, "
                f"pitch={pitch_offset if pitch_offset is not None else 'unchanged'}"
            )
        elif addr in (
            "/livelink/headpose/offset/add",
            "livelink/headpose/offset/add",
            "/headpose/offsets/add",
            "headpose/offsets/add",
            "/livelink/headpose/offsets/add",
            "livelink/headpose/offsets/add",
        ):
            yaw_delta = _coerce_float(args[0]) if len(args) >= 1 else None
            pitch_delta = _coerce_float(args[1]) if len(args) >= 2 else None
            if yaw_delta is None and pitch_delta is None:
                print(f"[UDP] Invalid head offset delta command: {address} {args}")
                return
            if on_set_headpose_offsets is not None:
                on_set_headpose_offsets(yaw_delta, pitch_delta, True)
            print(
                "[UDP] Head pose correction delta requested: "
                f"yaw={yaw_delta if yaw_delta is not None else 0.0}, "
                f"pitch={pitch_delta if pitch_delta is not None else 0.0}"
            )
        elif addr in (
            "/livelink/headpose/offset/reset",
            "livelink/headpose/offset/reset",
            "/headpose/offsets/reset",
            "headpose/offsets/reset",
            "/headpose/reset",
            "headpose/reset",
            "/livelink/headpose/offsets/reset",
            "livelink/headpose/offsets/reset",
            "/livelink/headpose/reset",
            "livelink/headpose/reset",
        ):
            if on_reset_headpose_offsets is not None:
                on_reset_headpose_offsets()
            print("[UDP] Head pose corrections reset requested")
        elif addr in ("/livelink/calibrate", "livelink/calibrate"):
            if on_calibrate is not None:
                on_calibrate("all")
            print("[UDP] Calibrate all requested")
        elif addr in ("/livelink/calibrate/clear", "livelink/calibrate/clear"):
            if on_calibrate is not None:
                on_calibrate("clear")
            print("[UDP] Calibration clear requested")
        elif addr in ("/livelink/headpose/calibrate", "livelink/headpose/calibrate"):
            if on_calibrate is not None:
                on_calibrate("headpose")
            print("[UDP] Head pose calibrate requested")
        elif addr in ("/livelink/blendshapes/calibrate", "livelink/blendshapes/calibrate"):
            if on_calibrate is not None:
                on_calibrate("blendshapes")
            print("[UDP] Blendshapes calibrate requested")
        elif addr in ("/livelink/restart", "livelink/restart"):
            action_queue.put("restart")
            print("[UDP] Restart requested")
        elif addr in ("/livelink/quit", "livelink/quit"):
            action_queue.put("quit")
            print("[UDP] Quit requested")
        else:
            print(f"[UDP] Unhandled command: {address} {args}")

    return handle_udp_command
