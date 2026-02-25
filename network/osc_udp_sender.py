import socket
import struct
from typing import Any, List


def _pad_osc_string(value: str) -> bytes:
    data = value.encode("utf-8") + b"\0"
    padding = (4 - (len(data) % 4)) % 4
    return data + (b"\0" * padding)


def _encode_osc_arg(arg: Any):
    if isinstance(arg, bool):
        return ("T" if arg else "F"), b""
    if arg is None:
        return "N", b""
    if isinstance(arg, int):
        return "i", struct.pack(">i", int(arg))
    if isinstance(arg, float):
        return "f", struct.pack(">f", float(arg))
    return "s", _pad_osc_string(str(arg))


def encode_osc_message(address: str, args: List[Any]) -> bytes:
    if not address.startswith("/"):
        raise ValueError("OSC address must start with '/'.")

    type_tags = ","
    payload_parts = []
    for arg in args:
        tag, payload = _encode_osc_arg(arg)
        type_tags += tag
        payload_parts.append(payload)

    return b"".join([_pad_osc_string(address), _pad_osc_string(type_tags), *payload_parts])


class OscUdpSender:
    def __init__(self, ip: str, port: int):
        self._ip = ip
        self._port = int(port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_message(self, address: str, args: List[Any] | None = None):
        try:
            payload = encode_osc_message(address, args or [])
            self._socket.sendto(payload, (self._ip, self._port))
        except Exception as error:
            print(f"[OscUdpSender] Failed to send {address}: {error}")

    def close(self):
        try:
            self._socket.close()
        except OSError:
            pass
