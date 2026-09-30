"""The fusion interface every strategy implements.

A :class:`PoseFuser` turns a list of per-camera :class:`CameraView` into a list
of :class:`FusedObservation` (one per real hand). Swapping feature fusion for
triangulation is a one-line change at the call site — nothing downstream cares.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .types import CameraView, FusedObservation


class PoseFuser(ABC):
    @abstractmethod
    def fuse(self, views: Sequence[CameraView], timestamp: float) -> list[FusedObservation]:
        """Combine all cameras' detections into one observation per hand.

        ``views`` may contain a single camera (the MVP), in which case a correct
        implementation must degrade to exactly the single-camera result — the
        fusion layer is always present so the multi-camera upgrade has nothing
        new to wire in.
        """
        ...
