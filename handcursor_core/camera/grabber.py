"""Threaded single-slot frame grabber.

Why a thread? OpenCV's ``VideoCapture`` buffers frames internally. If the main
loop reads slower than the camera delivers, ``read()`` returns progressively
*older* frames and you accumulate latency you can't see. The grabber thread
here continuously pulls and keeps only the newest frame, so the pipeline always
processes "now". This is the single biggest latency win for the MVP.

This is the grab loop from the full app's ``WebcamSource`` with the OpenCV
specifics lifted out: the frame source is any ``read_frame() -> (ok, frame)``
callable. A real webcam plugs in as::

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    grabber = LatestFrameGrabber(cap.read, transform=lambda f: cv2.flip(f, 1),
                                 on_stop=cap.release)

and the tests drive it with a synthetic counter source, so nothing here imports
``cv2``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from .base import CameraSource

FrameReader = Callable[[], tuple[bool, Any]]


class LatestFrameGrabber(CameraSource):
    def __init__(self, read_frame: FrameReader,
                 transform: Callable[[Any], Any] | None = None,
                 on_stop: Callable[[], None] | None = None,
                 clock: Callable[[], float] = time.perf_counter,
                 retry_sleep_s: float = 0.005) -> None:
        self._read_frame = read_frame
        self._transform = transform      # e.g. a horizontal mirror for "selfie" mapping
        self._on_stop = on_stop          # e.g. VideoCapture.release
        self._clock = clock
        self._retry_sleep_s = retry_sleep_s
        self._lock = threading.Lock()
        self._latest: Any = None
        self._latest_ts: float = 0.0
        self._grabbed = 0                # frames written into the slot
        self._consumed = 0               # value of _grabbed at the last read()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "LatestFrameGrabber":
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        return self

    def _grab_loop(self) -> None:
        while self._running:
            ok, frame = self._read_frame()
            if not ok:
                time.sleep(self._retry_sleep_s)
                continue
            if self._transform is not None:
                frame = self._transform(frame)
            with self._lock:
                # One slot: the previous frame is simply overwritten. Frames
                # the consumer was too slow to see are dropped, never queued.
                self._latest = frame
                self._latest_ts = self._clock()
                self._grabbed += 1

    def read(self) -> tuple[bool, Any, float]:
        with self._lock:
            if self._latest is None:
                return False, None, 0.0
            self._consumed = self._grabbed
            return True, self._latest, self._latest_ts

    @property
    def frames_grabbed(self) -> int:
        with self._lock:
            return self._grabbed

    @property
    def frames_skipped_since_last_read(self) -> int:
        """Frames overwritten since the last ``read()`` (0 = consumer keeping up)."""
        with self._lock:
            return max(self._grabbed - self._consumed - 1, 0)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._on_stop is not None:
            self._on_stop()
