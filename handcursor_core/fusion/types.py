"""Shared types for the multi-camera fusion layer.

The fusion layer's job: take the *same hand* as seen by N cameras in one frame
and produce a single estimate the rest of the pipeline already understands. The
contract is deliberately one level above raw landmarks, because the two fusion
strategies fuse at different stages:

  * **Feature fusion** (calibration-free) runs the recognizer per camera and
    combines the derived :class:`GestureFeatures` (you cannot collapse two
    cameras' image-space landmarks into one pose without knowing where the
    cameras are).
  * **Triangulation** (needs calibration) fuses the *landmarks* into one metric
    3D :class:`HandPose`, then recognizes once.

Both emit a :class:`FusedObservation`, so the state machine never learns how
many cameras exist or which strategy produced the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tracking.types import GestureFeatures, HandPose


@dataclass
class CameraView:
    """One camera's detections for a single (roughly simultaneous) frame.

    ``camera_id`` is a *logical* id in ``0..N-1`` that identifies the camera
    within the rig — it is NOT the OpenCV device index. The single-webcam MVP
    uses ``camera_id=0``.
    """

    camera_id: int
    poses: list[HandPose]
    timestamp: float = 0.0


@dataclass
class FusedObservation:
    """One real hand after fusing every camera that saw it.

    ``features`` is what the state machine consumes — identical type to the
    single-camera path. ``pose`` carries the metric 3D hand only when a
    triangulating fuser produced it (``None`` for feature fusion); it is the
    hook for future spatial / depth mapping.
    """

    features: GestureFeatures
    pose: HandPose | None = None
    handedness: str = "Unknown"
    contributing_cameras: int = 0
    # Which camera supplied the image-space position (cursor anchor). Useful for
    # debugging position jumps when the primary camera loses the hand.
    position_camera: int = -1
