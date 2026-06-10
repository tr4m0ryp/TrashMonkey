"""In-process HTTP MJPEG fixture server + shared plumbing for the deploy tests.

``MJPEGServer`` serves ``multipart/x-mixed-replace`` JPEG frames the way an
ESP32-CAM does, on 127.0.0.1 with an OS-assigned port. ``max_frames_per_conn``
lets a test simulate stream death (the connection closes; a new connection
serves again), which is exactly the reconnect path CameraReader must survive.
``StubReader`` is the deterministic CameraReader stand-in for driving
``Runtime.step(now=...)`` without threads or sleeps.
"""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from PIL import Image

from yolo_waste_sorter.models.thresholds import ThresholdParams

CLASSES = ("plastic", "paper", "cardboard", "metal", "glass", "organic")


def make_params(**overrides: Any) -> ThresholdParams:
    """T9 starting values from configs/config.yaml; override per test."""
    kwargs: dict[str, Any] = {
        "tau_frame": 0.40,
        "min_votes": 3,
        "high_water": 0.60,
        "conf_floor": 0.25,
    }
    kwargs.update(overrides)
    return ThresholdParams(**kwargs)


def wait_until(condition: Callable[[], bool], timeout_s: float, poll_s: float = 0.02) -> bool:
    """Poll ``condition`` until true or ``timeout_s`` elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(poll_s)
    return condition()


class StubReader:
    """Deterministic CameraReader stand-in: the test sets the snapshot."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.stopped = False
        self._snapshot: tuple[Any, float] | None = None

    def set_frame(self, frame: Any, ts: float) -> None:
        self._snapshot = (frame, ts)

    def latest(self) -> tuple[Any, float] | None:
        return self._snapshot

    def is_stale(self, now: float | None = None) -> bool:
        return self._snapshot is None

    def stop(self, join_timeout_s: float = 5.0) -> None:
        self.stopped = True

BOUNDARY = b"mjpegfixtureboundary"


def solid_jpeg(rgb: tuple[int, int, int], size: tuple[int, int] = (96, 72)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, rgb).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # keep test output clean

    def do_GET(self) -> None:  # noqa: N802 -- http.server API
        server: MJPEGServer = self.server.owner  # type: ignore[attr-defined]
        server.connections += 1
        self.send_response(200)
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}"
        )
        self.end_headers()
        sent = 0
        # Close the TCP connection when the loop ends so the client sees EOF
        # immediately (a half-open keep-alive socket would stall FFmpeg until
        # its read timeout instead of triggering the reconnect path).
        self.close_connection = True
        while server.running:
            if server.max_frames_per_conn is not None and sent >= server.max_frames_per_conn:
                break  # simulate stream death; client must reconnect
            jpeg = server.frames[sent % len(server.frames)]
            try:
                self.wfile.write(b"--" + BOUNDARY + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return
            sent += 1
            time.sleep(server.frame_interval_s)


class MJPEGServer:
    """Tiny threaded MJPEG server; use as a context manager."""

    def __init__(
        self,
        frames: list[bytes],
        *,
        fps: float = 25.0,
        max_frames_per_conn: int | None = None,
    ) -> None:
        if not frames:
            raise ValueError("MJPEGServer needs at least one frame")
        self.frames = frames
        self.frame_interval_s = 1.0 / fps
        self.max_frames_per_conn = max_frames_per_conn
        self.connections = 0
        self.running = True
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.owner = self  # type: ignore[attr-defined]
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/stream"

    def __enter__(self) -> MJPEGServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def stop(self) -> None:
        self.running = False
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
