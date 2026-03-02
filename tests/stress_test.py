#!/usr/bin/env python3
import argparse
import math
import os
import random
import socket
import sys
import threading
import time
import uuid


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from network.osc_udp_sender import encode_osc_message
except ImportError:
    print("ERROR: Could not import network.osc_udp_sender.")
    print("Please run this script from inside the project repository.")
    sys.exit(1)

try:
    from network.pylivelinkface import PyLiveLinkFace, FaceBlendShape
except ImportError:
    print("ERROR: Could not import network.pylivelinkface.")
    print("Please ensure dependencies are installed in your environment.")
    sys.exit(1)


TARGET_IP = "192.168.100.2"
OSC_TARGET_PORT = 9005
LIVE_LINK_TARGET_IP = TARGET_IP
DEFAULT_LIVELINK_PORTS = (11111, 11112, 11113, 11114)
LIVE_LINK_BASE_PORT = DEFAULT_LIVELINK_PORTS[0]
DEFAULT_PLAYERS = len(DEFAULT_LIVELINK_PORTS)
DEFAULT_TICK_MS = 33.0
DEFAULT_MOUSE_HZ = 0.75
DEFAULT_CAMERA_SWITCH_MIN_SEC = 2.0
DEFAULT_CAMERA_SWITCH_MAX_SEC = 6.0
DEFAULT_MODE = 4

MODE_MOVEMENT_ONLY = 1
MODE_MOUSE_ONLY = 2
MODE_LIVELINK_ONLY = 3
MODE_ALL = 4
MODE_LABELS = {
    MODE_MOVEMENT_ONLY: "only movement",
    MODE_MOUSE_ONLY: "only mouse movement",
    MODE_LIVELINK_ONLY: "only live link",
    MODE_ALL: "all together",
}
PLAYER_TOGGLE_KEYS = {
    "s": 1,
    "a": 2,
    "t": 3,
    "b": 4,
}

KEYBOARD_PATTERN_TICKS = (
    (1, 0, 8),   # Hold X
    (0, 0, 2),   # Release
    (0, 1, 8),   # Hold Y
    (0, 0, 2),   # Release
    (1, 1, 8),   # Hold both
    (0, 0, 4),   # Release
)


def keyboard_axes(player_idx, tick_idx):
    total_ticks = sum(segment[2] for segment in KEYBOARD_PATTERN_TICKS)
    player_offset_ticks = (player_idx - 1) * 6
    position = (tick_idx + player_offset_ticks) % total_ticks
    for x_axis, y_axis, duration_ticks in KEYBOARD_PATTERN_TICKS:
        if position < duration_ticks:
            return x_axis, y_axis
        position -= duration_ticks
    return 0, 0


def normalized_wave(phase):
    return 0.5 + (0.5 * math.sin(phase))


def build_camera_state(players, min_sec, max_sec):
    state = {}
    for player_idx in range(1, players + 1):
        rng = random.Random(1000 + (player_idx * 97))
        first_delay = rng.uniform(min_sec * 0.5, max_sec)
        state[player_idx] = {
            "value": player_idx,
            "next_switch_at": first_delay,
            "rng": rng,
        }
    return state


def camera_visible_value(player_idx, elapsed, players, mode, camera_state, min_sec, max_sec):
    if mode == "index":
        return player_idx

    player_state = camera_state[player_idx]
    while elapsed >= player_state["next_switch_at"]:
        player_state["value"] = (player_state["value"] % players) + 1
        player_state["next_switch_at"] += player_state["rng"].uniform(min_sec, max_sec)
    return player_state["value"]


def mode_flags(mode):
    return (
        mode in (MODE_MOVEMENT_ONLY, MODE_ALL),
        mode in (MODE_MOUSE_ONLY, MODE_ALL),
        mode in (MODE_LIVELINK_ONLY, MODE_ALL),
    )


def throttled_log(error_state, key, message, period_sec=1.0):
    now = time.perf_counter()
    last = error_state.get(key, 0.0)
    if now - last >= period_sec:
        print(message)
        error_state[key] = now


def send_osc_message(sock, ip, port, address, args, error_state):
    try:
        payload = encode_osc_message(address, args)
        sock.sendto(payload, (ip, port))
        return True
    except Exception as error:
        throttled_log(error_state, "osc_send", f"[OSC] send failed (example {address}): {error}", period_sec=2.0)
        return False


def build_livelink_clients(players, base_port, subject_prefix):
    clients = {}
    for player_idx in range(1, players + 1):
        subject_name = f"{subject_prefix}{player_idx:02d}"
        clients[player_idx] = {
            "face": PyLiveLinkFace(name=subject_name, uuid=str(uuid.uuid4())),
            "port": base_port + (player_idx - 1),
            "rng": random.Random(5000 + (player_idx * 131)),
        }
    return clients


def describe_player_toggles(runtime_state):
    with runtime_state["lock"]:
        parts = []
        for player_idx in sorted(runtime_state["player_enabled"].keys()):
            state = "ON" if runtime_state["player_enabled"][player_idx] else "OFF"
            parts.append(f"p{player_idx}={state}")
    return ", ".join(parts)


def update_random_livelink_values(face, rng):
    for blendshape in FaceBlendShape:
        if 0 <= blendshape.value <= 51:
            value = rng.uniform(0.0, 1.0)
        else:
            value = rng.uniform(-1.0, 1.0)
        face.set_blendshape(blendshape, value)


def send_movement_and_camera(
    sock,
    ip,
    port,
    player_idx,
    player_name,
    elapsed,
    tick_idx,
    args,
    movement_state,
    camera_state,
    movement_enabled,
    error_state,
    stats,
):
    x_axis, y_axis = keyboard_axes(player_idx, tick_idx)
    camera_value = camera_visible_value(
        player_idx=player_idx,
        elapsed=elapsed,
        players=args.players,
        mode=args.camera_mode,
        camera_state=camera_state,
        min_sec=args.camera_switch_min_sec,
        max_sec=args.camera_switch_max_sec,
    )

    axis_state = movement_state.setdefault(player_idx, {"x": None, "y": None, "camera": None})
    if movement_enabled:
        # Keep sending "1" while pressed. Send "0" only once on release.
        send_x = x_axis == 1 or axis_state["x"] != x_axis
        send_y = y_axis == 1 or axis_state["y"] != y_axis
        if send_x:
            if send_osc_message(
                sock,
                ip,
                port,
                f"/player_controller/{player_name}/move/xaxis/value",
                [x_axis],
                error_state,
            ):
                stats["movement_msgs"] += 1
            axis_state["x"] = x_axis
        if send_y:
            if send_osc_message(
                sock,
                ip,
                port,
                f"/player_controller/{player_name}/move/yaxis/value",
                [y_axis],
                error_state,
            ):
                stats["movement_msgs"] += 1
            axis_state["y"] = y_axis
        if axis_state["camera"] != camera_value:
            if send_osc_message(
                sock,
                ip,
                port,
                f"/player_controller/{player_name}/camera/visible/value",
                [camera_value],
                error_state,
            ):
                stats["camera_msgs"] += 1
            axis_state["camera"] = camera_value

    return x_axis, y_axis, camera_value


def send_mouse(
    sock,
    ip,
    port,
    player_idx,
    player_name,
    elapsed,
    players,
    mouse_hz,
    mouse_enabled,
    error_state,
    stats,
):
    phase_offset = (player_idx - 1) * (math.pi / max(players, 1))
    mouse_phase = (elapsed * mouse_hz * 2.0 * math.pi) + phase_offset
    mouse_x = round(normalized_wave(mouse_phase), 4)
    mouse_y = round(normalized_wave(mouse_phase + (math.pi / 3.0)), 4)

    if mouse_enabled:
        if send_osc_message(
            sock,
            ip,
            port,
            f"/player_controller/{player_name}/mouse/move/delta",
            [mouse_x, mouse_y],
            error_state,
        ):
            stats["mouse_msgs"] += 1

    return mouse_x, mouse_y


def send_livelink(sock, ip, player_idx, livelink_clients, livelink_enabled, error_state, stats):
    if not livelink_enabled:
        return

    client = livelink_clients[player_idx]
    try:
        update_random_livelink_values(client["face"], client["rng"])
        payload = client["face"].encode()
        sock.sendto(payload, (ip, client["port"]))
        stats["livelink_packets"] += 1
    except Exception as error:
        throttled_log(error_state, "livelink_send", f"[LiveLink] send failed (example player {player_idx}): {error}", period_sec=2.0)


def keyboard_listener(runtime_state):
    try:
        from pynput import keyboard
    except Exception as error:
        print(f"[Keys] Keyboard listener unavailable ({error}). Use --mode for fixed mode.")
        return

    print("Keyboard: [1]=only movement | [2]=only mouse | [3]=only live link | [4]=all | [ESC]=quit")
    print("Player toggles: [s]=player1 | [a]=player2 | [t]=player3 | [b]=player4")

    def on_press(key):
        try:
            key_char = key.char
        except AttributeError:
            if key == keyboard.Key.esc:
                with runtime_state["lock"]:
                    runtime_state["running"] = False
                print(">>> ESC pressed. Stopping...")
                return False
            return None

        key_char_lower = key_char.lower()
        if key_char_lower in ("1", "2", "3", "4"):
            next_mode = int(key_char_lower)
            with runtime_state["lock"]:
                runtime_state["mode"] = next_mode
            print(f">>> Mode {next_mode}: {MODE_LABELS[next_mode]}")
        elif key_char_lower in PLAYER_TOGGLE_KEYS:
            player_idx = PLAYER_TOGGLE_KEYS[key_char_lower]
            with runtime_state["lock"]:
                if player_idx in runtime_state["player_enabled"]:
                    runtime_state["player_enabled"][player_idx] = not runtime_state["player_enabled"][player_idx]
                    state_label = "ON" if runtime_state["player_enabled"][player_idx] else "OFF"
                    print(f">>> Player {player_idx} toggled {state_label}")
                else:
                    print(f">>> Player {player_idx} toggle ignored (configured players={len(runtime_state['player_enabled'])})")
            print(f">>> Players: {describe_player_toggles(runtime_state)}")
        elif key_char_lower == "q":
            with runtime_state["lock"]:
                runtime_state["running"] = False
            print(">>> Q pressed. Stopping...")
            return False
        return None

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def read_runtime_state(runtime_state):
    with runtime_state["lock"]:
        return runtime_state["running"], runtime_state["mode"], dict(runtime_state["player_enabled"])


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Unified stress test for OSC movement/mouse/camera + Live Link with keyboard mode toggles."
    )
    parser.add_argument("--ip", default=TARGET_IP, help=f"OSC target IP (default: {TARGET_IP})")
    parser.add_argument("--osc-port", type=int, default=OSC_TARGET_PORT, help=f"OSC target port (default: {OSC_TARGET_PORT})")
    parser.add_argument(
        "--livelink-ip",
        default=LIVE_LINK_TARGET_IP,
        help=f"Live Link target IP (default: {LIVE_LINK_TARGET_IP})",
    )
    parser.add_argument(
        "--livelink-base-port",
        type=int,
        default=LIVE_LINK_BASE_PORT,
        help=f"Live Link base port. Player i uses base + i - 1 (default: {LIVE_LINK_BASE_PORT})",
    )
    parser.add_argument("--players", type=int, default=DEFAULT_PLAYERS, help=f"Players to simulate (default: {DEFAULT_PLAYERS})")
    parser.add_argument(
        "--tick-ms",
        type=float,
        default=DEFAULT_TICK_MS,
        help=f"Send tick interval in milliseconds (default: {DEFAULT_TICK_MS})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional tick override. If set, it overrides --tick-ms.",
    )
    parser.add_argument("--mouse-hz", type=float, default=DEFAULT_MOUSE_HZ, help=f"Mouse wave speed in Hz (default: {DEFAULT_MOUSE_HZ})")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds. Use 0 for continuous run.")
    parser.add_argument(
        "--mode",
        type=int,
        choices=(MODE_MOVEMENT_ONLY, MODE_MOUSE_ONLY, MODE_LIVELINK_ONLY, MODE_ALL),
        default=DEFAULT_MODE,
        help="Startup mode: 1=movement, 2=mouse, 3=livelink, 4=all",
    )
    parser.add_argument(
        "--camera-mode",
        choices=("index", "cycle"),
        default="cycle",
        help="index=player index, cycle=irregular per-player camera switches (default: cycle)",
    )
    parser.add_argument(
        "--camera-switch-min-sec",
        type=float,
        default=DEFAULT_CAMERA_SWITCH_MIN_SEC,
        help=f"Minimum camera switch delay in cycle mode (default: {DEFAULT_CAMERA_SWITCH_MIN_SEC})",
    )
    parser.add_argument(
        "--camera-switch-max-sec",
        type=float,
        default=DEFAULT_CAMERA_SWITCH_MAX_SEC,
        help=f"Maximum camera switch delay in cycle mode (default: {DEFAULT_CAMERA_SWITCH_MAX_SEC})",
    )
    parser.add_argument("--camera-switch-sec", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--player-prefix", default="player", help="OSC player prefix (default: player)")
    parser.add_argument("--subject-prefix", default="PythonClient_", help="Live Link subject prefix (default: PythonClient_)")
    parser.add_argument("--status-sec", type=float, default=2.0, help="Status print interval in seconds (default: 2.0)")
    parser.add_argument("--no-keys", action="store_true", help="Disable keyboard mode toggles.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.players < 1:
        parser.error("--players must be >= 1")
    if args.osc_port < 1 or args.osc_port > 65535:
        parser.error("--osc-port must be between 1 and 65535")
    if args.livelink_base_port < 1 or args.livelink_base_port > 65535:
        parser.error("--livelink-base-port must be between 1 and 65535")
    last_livelink_port = args.livelink_base_port + args.players - 1
    if last_livelink_port > 65535:
        parser.error("livelink port range exceeds 65535 with current --players and --livelink-base-port")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be > 0")
    if args.tick_ms <= 0:
        parser.error("--tick-ms must be > 0")
    if args.camera_switch_sec is not None:
        args.camera_switch_min_sec = args.camera_switch_sec
        args.camera_switch_max_sec = args.camera_switch_sec
    if args.camera_switch_min_sec <= 0 or args.camera_switch_max_sec <= 0:
        parser.error("--camera-switch-min-sec and --camera-switch-max-sec must be > 0")
    if args.camera_switch_max_sec < args.camera_switch_min_sec:
        parser.error("--camera-switch-max-sec must be >= --camera-switch-min-sec")
    if args.status_sec <= 0:
        parser.error("--status-sec must be > 0")

    if args.fps is not None:
        interval = 1.0 / args.fps
    else:
        interval = args.tick_ms / 1000.0
    effective_fps = 1.0 / interval

    runtime_state = {
        "running": True,
        "mode": args.mode,
        "player_enabled": {player_idx: True for player_idx in range(1, args.players + 1)},
        "lock": threading.Lock(),
    }
    movement_state = {}
    camera_state = build_camera_state(args.players, args.camera_switch_min_sec, args.camera_switch_max_sec)
    error_state = {}
    stats = {
        "movement_msgs": 0,
        "camera_msgs": 0,
        "mouse_msgs": 0,
        "livelink_packets": 0,
    }
    last_stats = dict(stats)
    livelink_clients = build_livelink_clients(args.players, args.livelink_base_port, args.subject_prefix)

    osc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    livelink_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(
        f"Stress test started | players={args.players} | tick_ms={interval * 1000.0:.3f} | "
        f"fps={effective_fps:.2f} | startup_mode={args.mode}:{MODE_LABELS[args.mode]}"
    )
    print(
        f"OSC target={args.ip}:{args.osc_port} | LiveLink target={args.livelink_ip}:{args.livelink_base_port}-{last_livelink_port}"
    )
    if args.players == 4 and args.livelink_base_port == LIVE_LINK_BASE_PORT:
        print("LiveLink default player ports: p1=11111 p2=11112 p3=11113 p4=11114")
    print(
        f"camera_mode={args.camera_mode} | camera_switch={args.camera_switch_min_sec:.2f}-{args.camera_switch_max_sec:.2f}s"
    )
    print(f"Players enabled: {describe_player_toggles(runtime_state)}")
    if args.no_keys:
        print("Keyboard toggles disabled (--no-keys).")
    else:
        keyboard_thread = threading.Thread(target=keyboard_listener, args=(runtime_state,), daemon=True)
        keyboard_thread.start()
    print("Press Ctrl+C to stop.")

    start_time = time.perf_counter()
    next_tick = start_time
    frame_count = 0
    last_mode = None
    last_status_elapsed = 0.0

    try:
        while True:
            running, current_mode, player_enabled = read_runtime_state(runtime_state)
            if not running:
                break

            elapsed = time.perf_counter() - start_time
            if args.duration > 0 and elapsed >= args.duration:
                break

            if current_mode != last_mode:
                print(f">>> Active mode {current_mode}: {MODE_LABELS[current_mode]}")
                for player_state in movement_state.values():
                    player_state["x"] = None
                    player_state["y"] = None
                    player_state["camera"] = None
                last_mode = current_mode

            movement_enabled, mouse_enabled, livelink_enabled = mode_flags(current_mode)

            for player_idx in range(1, args.players + 1):
                if not player_enabled.get(player_idx, True):
                    player_state = movement_state.setdefault(player_idx, {"x": None, "y": None, "camera": None})
                    player_state["x"] = None
                    player_state["y"] = None
                    player_state["camera"] = None
                    continue

                player_name = f"{args.player_prefix}{player_idx}"

                send_movement_and_camera(
                    sock=osc_sock,
                    ip=args.ip,
                    port=args.osc_port,
                    player_idx=player_idx,
                    player_name=player_name,
                    elapsed=elapsed,
                    tick_idx=frame_count,
                    args=args,
                    movement_state=movement_state,
                    camera_state=camera_state,
                    movement_enabled=movement_enabled,
                    error_state=error_state,
                    stats=stats,
                )
                send_mouse(
                    sock=osc_sock,
                    ip=args.ip,
                    port=args.osc_port,
                    player_idx=player_idx,
                    player_name=player_name,
                    elapsed=elapsed,
                    players=args.players,
                    mouse_hz=args.mouse_hz,
                    mouse_enabled=mouse_enabled,
                    error_state=error_state,
                    stats=stats,
                )
                send_livelink(
                    sock=livelink_sock,
                    ip=args.livelink_ip,
                    player_idx=player_idx,
                    livelink_clients=livelink_clients,
                    livelink_enabled=livelink_enabled,
                    error_state=error_state,
                    stats=stats,
                )

            if elapsed - last_status_elapsed >= args.status_sec:
                enabled_count = sum(1 for enabled in player_enabled.values() if enabled)
                delta_movement = stats["movement_msgs"] - last_stats["movement_msgs"]
                delta_camera = stats["camera_msgs"] - last_stats["camera_msgs"]
                delta_mouse = stats["mouse_msgs"] - last_stats["mouse_msgs"]
                delta_livelink = stats["livelink_packets"] - last_stats["livelink_packets"]
                print(
                    f"status t={elapsed:6.2f}s mode={current_mode}:{MODE_LABELS[current_mode]} "
                    f"players={enabled_count}/{args.players} "
                    f"movement +{delta_movement} ({stats['movement_msgs']}) "
                    f"camera +{delta_camera} ({stats['camera_msgs']}) "
                    f"mouse +{delta_mouse} ({stats['mouse_msgs']}) "
                    f"livelink +{delta_livelink} ({stats['livelink_packets']})"
                )
                last_stats = dict(stats)
                last_status_elapsed = elapsed

            frame_count += 1
            next_tick += interval
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping...")
    finally:
        with runtime_state["lock"]:
            runtime_state["running"] = False
        osc_sock.close()
        livelink_sock.close()
        total_time = time.perf_counter() - start_time
        print(f"Stopped after {total_time:.2f}s. Frames sent: {frame_count}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
