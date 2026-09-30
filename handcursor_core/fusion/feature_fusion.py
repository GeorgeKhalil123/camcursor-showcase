"""Calibration-free fusion: combine each camera's *derived features*.

This is the strategy you can run the day a second camera is plugged in — no
checkerboard, no extrinsics, no synchronisation beyond "same frame-ish". It
answers the user's question directly: every camera votes on the gesture, so a
pinch that's edge-on to one camera is resolved by another that sees it face-on.

What is fused vs. not, and why:

  * **Shape / gesture signals** (pinch distance, finger extension, view quality)
    are combined across ALL cameras. These are scale-invariant scalars, so
    blending them is meaningful and is exactly where multi-camera robustness
    comes from. Each camera's weight is its own ``view_quality`` (how face-on it
    sees the hand), so the camera with the clearest view dominates.
  * **Position** (the cursor anchor, tips, palm centre) is taken from ONE
    designated camera. These are image-space coordinates in *different* image
    planes, so averaging them across cameras is meaningless and would make the
    cursor jump. The primary camera (the one facing the screen) owns position;
    true 3D position is the job of the triangulation strategy.

With a single camera this reduces to ``recognizer.recognize(pose)`` exactly, so
wiring it into the MVP changes nothing until a second camera appears.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..config import GestureConfig
from ..gestures.recognizer import GestureRecognizer
from ..tracking.types import FINGER_NAMES, GestureFeatures
from .association import HandGroup, associate
from .base import PoseFuser
from .types import CameraView, FusedObservation


class FeatureFusion(PoseFuser):
    def __init__(self, gesture_config: GestureConfig, anchor: str = "mcp",
                 primary_camera: int = 0, min_view_weight: float = 0.05,
                 vote_threshold: float = 0.5) -> None:
        self.cfg = gesture_config
        self.anchor = anchor
        self.primary_camera = primary_camera
        self.min_view_weight = min_view_weight
        self.vote_threshold = vote_threshold
        # One recognizer per camera: each camera is its own continuous stream,
        # so per-finger extension hysteresis must be tracked independently.
        self._recognizers: dict[int, GestureRecognizer] = {}

    def fuse(self, views: Sequence[CameraView], timestamp: float) -> list[FusedObservation]:
        groups = associate(views, self.primary_camera)
        return [self._fuse_group(g, timestamp) for g in groups]

    # --- internals ----------------------------------------------------------

    def _fuse_group(self, group: HandGroup, timestamp: float) -> FusedObservation:
        # Single camera: return the recognizer's output verbatim. This makes the
        # MVP-equivalence invariant provable rather than a coincidence of the
        # vote/weight thresholds — with one camera, fusion is a no-op by
        # construction, so the working single-webcam path can never regress.
        if len(group) == 1:
            cam_id, pose = group[0]
            f = self._recognizer(cam_id).recognize(pose)
            f.timestamp = timestamp
            return FusedObservation(features=f, pose=None,
                                    handedness=pose.handedness,
                                    contributing_cameras=1, position_camera=cam_id)

        # Recognize each camera's view of this hand (per-camera hysteresis).
        per_cam: list[tuple[int, GestureFeatures]] = [
            (cam_id, self._recognizer(cam_id).recognize(pose))
            for cam_id, pose in group
        ]
        # Two weightings: a floored one for blending scalars (so even a poor view
        # nudges the average), and the RAW view quality for the finger vote (so a
        # near-edge-on camera, whose tracking is unreliable, can't tip a finger
        # on/off via the floor).
        weights = [max(f.view_quality, self.min_view_weight) for _, f in per_cam]
        total = sum(weights) or 1.0
        vote_weights = [f.view_quality for _, f in per_cam]
        vote_total = sum(vote_weights) or 1.0

        # Position: prefer the primary camera; else the highest-weight camera
        # that saw this hand (documented fallback — may jump if primary drops).
        pos_idx = next((i for i, (cam_id, _) in enumerate(per_cam)
                        if cam_id == self.primary_camera),
                       int(np.argmax(weights)))
        pos_cam_id, pos_f = per_cam[pos_idx]

        fused = GestureFeatures(
            fingers_extended=self._vote_fingers(per_cam, vote_weights, vote_total),
            pinch_distance=self._wavg([f.pinch_distance for _, f in per_cam], weights, total),
            index_straightness=self._wavg([f.index_straightness for _, f in per_cam], weights, total),
            index_reach=self._wavg([f.index_reach for _, f in per_cam], weights, total),
            # Best available view drives the gate: if ANY camera sees the hand
            # well, we're not edge-on. This is a core multi-camera win — the
            # view-quality freeze stops firing as soon as one camera has a look.
            view_quality=max(f.view_quality for _, f in per_cam),
            index_tip=pos_f.index_tip.copy(),
            index_mcp=pos_f.index_mcp.copy(),
            pointer=pos_f.pointer.copy(),
            palm_center=pos_f.palm_center.copy(),
            hand_size=pos_f.hand_size,
            timestamp=timestamp,
        )
        return FusedObservation(
            features=fused,
            pose=None,                       # feature fusion produces no 3D pose
            handedness=group[0][1].handedness,
            contributing_cameras=len(group),
            position_camera=pos_cam_id,
        )

    def _vote_fingers(self, per_cam: list[tuple[int, GestureFeatures]],
                      weights: list[float], total: float) -> dict[str, bool]:
        """Weighted vote per finger: extended if the cameras that agree it's
        extended hold more than ``vote_threshold`` of the total view weight."""
        out: dict[str, bool] = {}
        for name in FINGER_NAMES:
            agree = sum(w for (_, f), w in zip(per_cam, weights)
                        if f.fingers_extended.get(name, False))
            out[name] = agree > self.vote_threshold * total
        return out

    @staticmethod
    def _wavg(values: list[float], weights: list[float], total: float) -> float:
        return float(sum(v * w for v, w in zip(values, weights)) / total)

    def _recognizer(self, camera_id: int) -> GestureRecognizer:
        rec = self._recognizers.get(camera_id)
        if rec is None:
            rec = GestureRecognizer(self.cfg, anchor=self.anchor)
            self._recognizers[camera_id] = rec
        return rec
