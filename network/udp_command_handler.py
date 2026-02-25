from typing import Any, Callable, List, Mapping

from network.live_link_sender import LiveLinkSender


def build_udp_command_handler(
    sender: LiveLinkSender, cfg: Mapping[str, Any]
) -> Callable[[str, List[Any]], None]:
    def handle_udp_command(address: str, args: List[Any]):
        addr = address.lower()
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
        else:
            print(f"[UDP] Unhandled command: {address} {args}")

    return handle_udp_command
