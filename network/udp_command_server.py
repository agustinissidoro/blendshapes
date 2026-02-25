import socket
import struct
import threading
from typing import Callable, List, Tuple, Any


OSCMessage = Tuple[str, List[Any]]


def _read_padded_string(data: bytes, idx: int) -> Tuple[str, int]:
    end = data.find(b"\0", idx)
    if end == -1:
        return "", len(data)
    s = data[idx:end].decode("utf-8", errors="ignore")
    next_idx = (end + 4) & ~0x03
    return s, next_idx


def _parse_message(data: bytes) -> List[OSCMessage]:
    address, idx = _read_padded_string(data, 0)
    if not address:
        return []
    type_tags, idx = _read_padded_string(data, idx)
    if not type_tags:
        type_tags = ","
    args: List[Any] = []
    if type_tags.startswith(","):
        for tag in type_tags[1:]:
            if tag == "i":
                args.append(struct.unpack(">i", data[idx:idx + 4])[0])
                idx += 4
            elif tag == "f":
                args.append(struct.unpack(">f", data[idx:idx + 4])[0])
                idx += 4
            elif tag == "s":
                s, idx = _read_padded_string(data, idx)
                args.append(s)
            elif tag == "T":
                args.append(True)
            elif tag == "F":
                args.append(False)
            elif tag == "N":
                args.append(None)
            else:
                break
    return [(address, args)]


def parse_osc_packet(data: bytes) -> List[OSCMessage]:
    if data.startswith(b"#bundle"):
        # "#bundle\0" + 8-byte timetag, then elements
        idx = 16
        messages: List[OSCMessage] = []
        while idx + 4 <= len(data):
            size = struct.unpack(">i", data[idx:idx + 4])[0]
            idx += 4
            if size <= 0 or idx + size > len(data):
                break
            element = data[idx:idx + size]
            idx += size
            messages.extend(parse_osc_packet(element))
        return messages
    return _parse_message(data)


class UdpCommandServer(threading.Thread):
    def __init__(self, ip: str, port: int, handler: Callable[[str, List[Any]], None]):
        super().__init__(daemon=True, name="UdpCommandServer")
        self._ip = ip
        self._port = port
        self._handler = handler
        self._running = False
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self._ip, self._port))
        self._socket.settimeout(0.5)

    def run(self):
        self._running = True
        print(f"[UdpCommandServer] Listening on {self._ip}:{self._port}")
        while self._running:
            try:
                data, _addr = self._socket.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            for address, args in parse_osc_packet(data):
                try:
                    self._handler(address, args)
                except Exception as e:
                    print(f"[UdpCommandServer] Handler error: {e}")

    def stop(self):
        self._running = False
        try:
            self._socket.close()
        except OSError:
            pass
