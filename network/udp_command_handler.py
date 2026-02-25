from typing import Any, Callable, List, Mapping, Optional

from network.live_link_sender import LiveLinkSender


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


def build_udp_command_handler(
    sender: LiveLinkSender,
    cfg: Mapping[str, Any],
    on_tracking: Optional[Callable[[Optional[bool]], None]] = None,
    on_get_state: Optional[Callable[[], None]] = None,
) -> Callable[[str, List[Any]], None]:
    def handle_udp_command(address: str, args: List[Any]):
        addr = address.strip().lower()
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
        elif addr in (
            "/livelink/tracking",
            "livelink/tracking",
            "/livelink/trackiing",  # Backward-compatible alias for previous typo.
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
        else:
            print(f"[UDP] Unhandled command: {address} {args}")

    return handle_udp_command
