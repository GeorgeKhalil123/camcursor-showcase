"""Camera source abstraction.

In the full app the concrete source is an OpenCV ``WebcamSource`` (plus a
``MultiCameraSource`` that composes one per device for triangulation). Here the
only implementation is :class:`~handcursor_core.camera.grabber.LatestFrameGrabber`,
which carries the same single-slot threading logic but takes any callable as
its frame source. Anything downstream only needs ``read()`` to return the
latest frame(s)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CameraSource(ABC):
    @abstractmethod
    def start(self) -> "CameraSource":
        ...

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Return ``(ok, frame_bgr, timestamp)``.

        ``timestamp`` is ``time.perf_counter()`` at capture, used downstream for
        frame-rate-independent smoothing.
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    def __enter__(self) -> "CameraSource":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
