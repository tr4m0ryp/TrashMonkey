"""Grab-latest MJPEG camera readers.

One daemon thread per camera stream runs a continuous grab/retrieve loop
and keeps ONLY the newest frame under a lock -- ``CAP_PROP_BUFFERSIZE`` is
unreliable on HTTP MJPEG, so freshness is enforced in software.
Every frame is stamped with its monotonic receive time; consumers treat a
frame older than ``stale_after_s`` as stale. A dead stream is closed, the
thread sleeps ``reconnect_backoff_s``, then reopens (reconnects are counted).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

# FFmpeg's default 5 MB stream probe stalls for tens of seconds at low-cost
# IP-camera bitrates (tens of KB/frame) on EVERY open/reconnect; cap it before
# cv2 opens the capture. ``timeout`` (microseconds) bounds socket reads so a
# dead-but-open TCP stream fails grab() and enters the reconnect path instead
# of hanging forever. setdefault keeps any operator override authoritative.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "probesize;65536|analyzeduration;0|timeout;5000000"
)

import cv2  # noqa: E402  -- must come after the FFmpeg capture options guard
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

Frame = npt.NDArray[np.uint8]


class StreamError(Exception):
    """Camera stream configuration or lifecycle error."""


class CameraReader:
    """Background grab-latest reader for one MJPEG camera stream.

    ``latest()`` returns the newest ``(frame, monotonic_ts)`` pair or ``None``
    before the first frame arrives. ``stop()`` is idempotent and joins the
    daemon thread. ``capture_factory`` exists for tests that need to fake
    ``cv2.VideoCapture``; production uses the default.
    """

    def __init__(
        self,
        url: str,
        *,
        stale_after_s: float,
        reconnect_backoff_s: float,
        name: str | None = None,
        capture_factory: Any = cv2.VideoCapture,
    ) -> None:
        if not url:
            raise StreamError("camera url must be non-empty")
        if stale_after_s <= 0 or reconnect_backoff_s <= 0:
            raise StreamError(
                f"stale_after_s and reconnect_backoff_s must be > 0, "
                f"got {stale_after_s} / {reconnect_backoff_s}"
            )
        self.url = url
        self.name = name if name is not None else url
        self.stale_after_s = stale_after_s
        self.reconnect_backoff_s = reconnect_backoff_s
        self._capture_factory = capture_factory
        self._lock = threading.Lock()
        self._latest: tuple[Frame, float] | None = None
        self._reconnects = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> CameraReader:
        if self._thread is not None:
            raise StreamError(f"reader {self.name!r} already started")
        self._thread = threading.Thread(
            target=self._run, name=f"camera-reader-{self.name}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, join_timeout_s: float = 5.0) -> None:
        """Signal the loop to exit and join the thread (idempotent)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)
            self._thread = None

    # -- consumer surface --------------------------------------------------

    def latest(self) -> tuple[Frame, float] | None:
        """Newest ``(frame, monotonic_receive_ts)`` or None before first frame."""
        with self._lock:
            return self._latest

    def is_stale(self, now: float | None = None) -> bool:
        """True when no frame yet or the newest frame exceeds the staleness budget."""
        snapshot = self.latest()
        if snapshot is None:
            return True
        ts = snapshot[1]
        current = time.monotonic() if now is None else now
        return current - ts > self.stale_after_s

    @property
    def reconnects(self) -> int:
        """Times the stream died and the reader went through close/backoff/reopen."""
        with self._lock:
            return self._reconnects

    # -- reader loop -------------------------------------------------------

    def _run(self) -> None:
        first_attempt = True
        while not self._stop.is_set():
            if not first_attempt:
                with self._lock:
                    self._reconnects += 1
                if self._stop.wait(self.reconnect_backoff_s):
                    break
            first_attempt = False
            capture = self._capture_factory(self.url)
            try:
                if not capture.isOpened():
                    continue
                self._pump(capture)
            finally:
                capture.release()

    def _pump(self, capture: Any) -> None:
        """Grab/retrieve until the stream dies or stop is requested."""
        while not self._stop.is_set():
            if not capture.grab():
                return
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                return
            stamped = (np.asarray(frame, dtype=np.uint8), time.monotonic())
            with self._lock:
                self._latest = stamped


def start_readers(
    urls: tuple[str, ...], *, stale_after_s: float, reconnect_backoff_s: float
) -> list[CameraReader]:
    """Start one grab-latest reader per camera URL; names are cam0..camN."""
    if not urls:
        raise StreamError("deploy.cameras must list at least one stream URL")
    return [
        CameraReader(
            url,
            stale_after_s=stale_after_s,
            reconnect_backoff_s=reconnect_backoff_s,
            name=f"cam{i}",
        ).start()
        for i, url in enumerate(urls)
    ]
