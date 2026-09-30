"""Core data structures shared across the pipeline.

These types are deliberately camera-agnostic. For the single-webcam MVP a
``HandPose`` carries MediaPipe's normalised 2D landmarks (plus its relative
``z``). When we move to multi-camera triangulation later, the same ``HandPose``
can be populated with metric 3D coordinates without changing any downstream
consumer (gestures, state machine, cursor mapping).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Landmark(IntEnum):
    """MediaPipe Hands 21-landmark topology (indices into ``HandPose.points``)."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# (mcp, pip, dip, tip) per finger; thumb uses (cmc, mcp, ip, tip).
FINGER_JOINTS: dict[str, tuple[int, int, int, int]] = {
    "thumb": (Landmark.THUMB_CMC, Landmark.THUMB_MCP, Landmark.THUMB_IP, Landmark.THUMB_TIP),
    "index": (Landmark.INDEX_MCP, Landmark.INDEX_PIP, Landmark.INDEX_DIP, Landmark.INDEX_TIP),
    "middle": (Landmark.MIDDLE_MCP, Landmark.MIDDLE_PIP, Landmark.MIDDLE_DIP, Landmark.MIDDLE_TIP),
    "ring": (Landmark.RING_MCP, Landmark.RING_PIP, Landmark.RING_DIP, Landmark.RING_TIP),
    "pinky": (Landmark.PINKY_MCP, Landmark.PINKY_PIP, Landmark.PINKY_DIP, Landmark.PINKY_TIP),
}

FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")


@dataclass
class HandPose:
    """A single detected hand at one instant.

    Attributes:
        points: ``(21, 3)`` array. For the MVP these are MediaPipe *normalised
            image* coordinates: x,y in ``[0, 1]`` relative to the frame, z a
            relative depth (smaller = closer to camera). In the multi-camera
            future these become metric 3D world coordinates.
        handedness: "Left" or "Right" (from the camera's point of view after
            the frame is mirrored for a natural selfie experience).
        score: detection/tracking confidence in ``[0, 1]``.
        timestamp: capture time in seconds (``time.perf_counter``), used by the
            One Euro filter for frame-rate-independent smoothing.
    """

    points: np.ndarray
    # Metric, hand-centred 3D landmarks from MediaPipe (meters). More
    # orientation-robust than the image-space z, so we derive shape features
    # (straightness/reach/pinch) from these when available. ``None`` falls back
    # to ``points``.
    world_points: np.ndarray | None = None
    handedness: str = "Unknown"
    score: float = 0.0
    timestamp: float = 0.0

    def point(self, lm: Landmark | int) -> np.ndarray:
        return self.points[int(lm)]

    def xy(self, lm: Landmark | int) -> np.ndarray:
        """Just the (x, y) of a landmark — what the 2D MVP cares about."""
        return self.points[int(lm)][:2]


@dataclass
class GestureFeatures:
    """Derived, scale/translation-invariant features used for classification.

    Keeping recognition separate from raw landmarks means the gesture rules and
    the state machine never need to know whether the data came from one camera
    or three.
    """

    fingers_extended: dict[str, bool] = field(default_factory=dict)
    pinch_distance: float = 1.0  # thumb-tip<->index-tip, normalised by hand size
    index_straightness: float = 1.0  # 0..1, for the HUD / tuning
    index_reach: float = 0.0         # tip-to-knuckle / hand size, for HUD / tuning
    # How well the palm faces the camera (normalised palm area). ~0 when the
    # hand is edge-on/oblique, where tracking is unreliable. Default 1.0 so
    # synthetic/test features are never gated.
    view_quality: float = 1.0
    index_tip: np.ndarray = field(default_factory=lambda: np.zeros(2))
    index_mcp: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # The point that actually drives the cursor (tip/mcp/blend, per config).
    pointer: np.ndarray = field(default_factory=lambda: np.zeros(2))
    palm_center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hand_size: float = 1.0
    timestamp: float = 0.0

    @property
    def num_extended(self) -> int:
        return sum(self.fingers_extended.values())

    @property
    def is_pointing(self) -> bool:
        # Index out, middle in. Ring/pinky are ignored — they twitch, and the
        # contrast that matters (vs. a future two-finger scroll) is index/middle.
        f = self.fingers_extended
        return f.get("index", False) and not f.get("middle", False)

    @property
    def is_fist(self) -> bool:
        # No fingers extended (thumb allowed to be ambiguous in a loose fist).
        f = self.fingers_extended
        return not any(f.get(name, False) for name in ("index", "middle", "ring", "pinky"))

    @property
    def is_open_palm(self) -> bool:
        # A spread hand: at least 4 of 5 fingers extended. Cleanly separable
        # from pointing (1), fist (0) and pinch (~1-2).
        return self.num_extended >= 4
