"""Client for the persistent Blender rendering server.

The Blender server is an external service started via `make blender-server`.
This module provides a thin TCP client that sends rendering manifests and
receives rendered frame paths back.

Usage:
    client = BlenderServerClient.connect()
    images = client.render_clip(manifest_dict, image_size=224)
"""

from __future__ import annotations

import json
import logging
import socket
import struct
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876


class BlenderServerClient:
    """TCP client for the persistent Blender rendering server.

    The server is started externally via `make blender-server` and listens
    on a TCP port. This client sends rendering manifests and receives
    rendered frame paths.
    """

    def __init__(self, sock: socket.socket):
        self._sock = sock

    @classmethod
    def connect(
        cls, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
    ) -> BlenderServerClient:
        """Connect to a running Blender server.

        Raises ConnectionError if the server is not running.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((host, port))
        except (ConnectionRefusedError, OSError) as e:
            sock.close()
            raise ConnectionError(
                f"Cannot connect to Blender server at {host}:{port}. "
                f"Start it with `make blender-server`."
            ) from e
        return cls(sock)

    def render_clip(
        self, manifest: dict, image_size: int = 224,
    ) -> list[Image.Image]:
        """Send a rendering job to the Blender server.

        Args:
            manifest: Dict with keys: piece_set, material, board_theme,
                lighting, frames.
            image_size: Expected output image size.

        Returns:
            List of PIL Images (one per frame).
        """
        request = json.dumps(manifest).encode("utf-8")
        self._send_msg(request)

        response_data = self._recv_msg()
        response = json.loads(response_data.decode("utf-8"))

        if response.get("status") != "ok":
            error = response.get("error", "Unknown error")
            raise RuntimeError(f"Blender render error: {error}")

        images = []
        for frame_path in response.get("frames", []):
            img = Image.open(frame_path).convert("RGB")
            images.append(img)

        return images

    def _send_msg(self, data: bytes) -> None:
        """Send a length-prefixed message."""
        self._sock.sendall(struct.pack("!I", len(data)))
        self._sock.sendall(data)

    def _recv_msg(self) -> bytes:
        """Receive a length-prefixed message."""
        raw_len = self._recvall(4)
        if not raw_len:
            raise ConnectionError("Server closed connection")
        msg_len = struct.unpack("!I", raw_len)[0]
        return self._recvall(msg_len)

    def _recvall(self, n: int) -> bytes:
        """Receive exactly n bytes."""
        data = bytearray()
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Server closed connection")
            data.extend(chunk)
        return bytes(data)

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
