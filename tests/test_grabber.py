"""Tests for the single-slot threaded grabber, driven by synthetic frame sources.

No camera and no OpenCV: the "camera" is a callable that hands out numbered
frames, so we can check exactly which frames the consumer saw.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from handcursor_core.camera.grabber import LatestFrameGrabber


class _CountingSource:
    """Emits frames 1, 2, 3, ... then reports failure once ``limit`` is hit."""

    def __init__(self, limit: int, delay_s: float = 0.0) -> None:
        self.limit = limit
        self.delay_s = delay_s
        self.n = 0
        self.released = False

    def read(self):
        if self.n >= self.limit:
            return False, None
        if self.delay_s:
            time.sleep(self.delay_s)
        self.n += 1
        return True, self.n

    def release(self) -> None:
        self.released = True


def _wait_until(pred, timeout_s: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.001)
    return pred()


def test_read_before_first_frame_is_not_ok():
    gate = threading.Event()

    def blocked():
        gate.wait(1.0)
        return False, None

    with LatestFrameGrabber(blocked, retry_sleep_s=0.001) as grabber:
        ok, frame, ts = grabber.read()
        assert (ok, frame, ts) == (False, None, 0.0)
        gate.set()


def test_slow_consumer_only_ever_sees_the_newest_frame():
    # The camera produces 50 frames while the consumer is "busy". A buffered
    # reader would hand back frame 1 next; the single slot hands back frame 50.
    src = _CountingSource(limit=50)
    with LatestFrameGrabber(src.read, retry_sleep_s=0.001) as grabber:
        assert _wait_until(lambda: grabber.frames_grabbed == 50)
        ok, frame, _ = grabber.read()
        assert ok and frame == 50
        # 49 frames were overwritten before the read, never queued.
        assert grabber.frames_grabbed == 50


def test_skipped_frame_counter_resets_on_read():
    src = _CountingSource(limit=10)
    with LatestFrameGrabber(src.read, retry_sleep_s=0.001) as grabber:
        assert _wait_until(lambda: grabber.frames_grabbed == 10)
        assert grabber.frames_skipped_since_last_read == 9
        grabber.read()
        assert grabber.frames_skipped_since_last_read == 0


def test_repeated_reads_return_same_frame_without_blocking():
    src = _CountingSource(limit=3)
    with LatestFrameGrabber(src.read, retry_sleep_s=0.001) as grabber:
        assert _wait_until(lambda: grabber.frames_grabbed == 3)
        first = grabber.read()
        second = grabber.read()
        assert first == second and first[1] == 3


def test_timestamps_come_from_the_clock_at_capture():
    ticks = iter(np.arange(100.0, 200.0, 1.0))
    src = _CountingSource(limit=5)
    with LatestFrameGrabber(src.read, clock=lambda: float(next(ticks)),
                            retry_sleep_s=0.001) as grabber:
        assert _wait_until(lambda: grabber.frames_grabbed == 5)
        ok, frame, ts = grabber.read()
        assert ok and frame == 5 and ts == 104.0   # 5th capture -> 5th tick


def test_transform_is_applied_before_publishing():
    # Stand-in for the selfie mirror (cv2.flip(frame, 1) in the full app).
    img = np.arange(6).reshape(2, 3)
    calls = {"n": 0}

    def once():
        calls["n"] += 1
        return (True, img) if calls["n"] == 1 else (False, None)

    with LatestFrameGrabber(once, transform=np.fliplr, retry_sleep_s=0.001) as grabber:
        assert _wait_until(lambda: grabber.frames_grabbed == 1)
        _, frame, _ = grabber.read()
        assert np.array_equal(frame, np.fliplr(img))


def test_failed_reads_are_retried_not_fatal():
    results = iter([(False, None), (False, None), (True, "frame")])

    def flaky():
        return next(results, (False, None))

    with LatestFrameGrabber(flaky, retry_sleep_s=0.001) as grabber:
        assert _wait_until(lambda: grabber.frames_grabbed == 1)
        assert grabber.read()[:2] == (True, "frame")


def test_stop_joins_thread_and_releases_source():
    src = _CountingSource(limit=10_000, delay_s=0.001)
    grabber = LatestFrameGrabber(src.read, on_stop=src.release).start()
    assert _wait_until(lambda: grabber.frames_grabbed > 0)
    grabber.stop()
    count = grabber.frames_grabbed
    time.sleep(0.02)
    assert grabber.frames_grabbed == count      # no more grabs after stop
    assert src.released
