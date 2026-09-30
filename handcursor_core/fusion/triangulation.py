"""Triangulation fusion: reconstruct ONE metric 3D hand from N cameras.

This is the endgame the whole architecture points at. Where feature fusion
blends scalars, triangulation fuses the *landmarks themselves*: each of the 21
points is reconstructed in real 3D from its position in every camera that saw
it, so the hand exists as true geometry — correct at any angle, and the
foundation for spatial monitor mapping, depth, and window-throwing.

The linear DLT solver below is real and tested. The one piece that genuinely
requires field data is **calibration** — the cameras' intrinsics ``K`` and
relative pose ``(R, t)``. Supply :class:`CameraCalibration` per camera and this
fuser produces metric 3D hands today; without it, it raises with a clear
message. Acquiring calibration (checkerboard, or bundle-adjusting the moving
hand itself) is the documented next milestone, and it lands here without
touching any other module.

Pipeline per hand:
    associate across cameras (handedness)
    -> for each of 21 landmarks: DLT-triangulate from all calibrated views
    -> assemble a metric 3D HandPose (world_points = the reconstruction)
    -> recognize ONCE on that pose (shape features now angle-proof)

Cursor position still comes from the primary camera's image landmarks until
spatial monitor mapping is built — the metric 3D lives on ``HandPose.points``'
companion and on the returned ``FusedObservation.pose`` for that future work.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..config import GestureConfig
from ..gestures.recognizer import GestureRecognizer
from ..tracking.types import HandPose
from .association import HandGroup, associate
from .base import PoseFuser
from .types import CameraView, FusedObservation


@dataclass
class CameraCalibration:
    """A camera's intrinsics and pose in the shared world frame.

    ``K`` is the 3x3 intrinsic matrix (focal lengths + principal point in
    pixels). ``R`` (3x3) and ``t`` (3,) place the camera in world space such
    that ``x_cam = R @ x_world + t``. ``image_size`` is ``(width, height)`` in
    pixels, used to lift MediaPipe's normalised ``[0,1]`` coords to pixels.
    """

    camera_id: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_size: tuple[int, int]

    @property
    def projection(self) -> np.ndarray:
        """The 3x4 projection matrix ``P = K [R | t]`` (world -> pixels)."""
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.K @ Rt


def triangulate_dlt(rays: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Direct Linear Transform: least-squares 3D point from >=2 camera rays.

    ``rays`` is a sequence of ``(P, xy)`` where ``P`` is a 3x4 projection matrix
    and ``xy`` is the observed pixel coordinate in that camera. Each view
    contributes two rows to ``A x = 0`` (from ``x * (P row 2) - (P row 0)`` and
    ``y * (P row 2) - (P row 1)``); the homogeneous solution is the right
    singular vector of ``A`` with the smallest singular value.
    """
    if len(rays) < 2:
        raise ValueError("triangulation needs at least two camera views")
    A = np.empty((2 * len(rays), 4), dtype=np.float64)
    for i, (P, xy) in enumerate(rays):
        x, y = float(xy[0]), float(xy[1])
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        # Homogeneous w ~ 0 => a point at infinity / rank-deficient views (e.g.
        # near-coincident cameras). Fail loudly rather than leak NaN downstream.
        raise ValueError("degenerate triangulation: point at infinity or rank-deficient views")
    return (X[:3] / X[3]).astype(np.float32)


class TriangulationFusion(PoseFuser):
    def __init__(self, gesture_config: GestureConfig, anchor: str = "mcp",
                 calibrations: dict[int, CameraCalibration] | None = None,
                 primary_camera: int = 0,
                 max_view_skew_s: float = 0.05) -> None:
        self.cfg = gesture_config
        self.anchor = anchor
        self.calibrations = calibrations or {}
        self.primary_camera = primary_camera
        # DLT assumes all views are of the same instant. If one camera lags the
        # others by more than this many seconds, drop it from the solve rather
        # than warp the reconstruction. 0 disables the gate.
        self.max_view_skew_s = max_view_skew_s
        # Single recognizer: it runs on the already-fused metric pose, so there
        # is one continuous stream, not one per camera.
        self._recognizer = GestureRecognizer(gesture_config, anchor=anchor)

    def fuse(self, views: Sequence[CameraView], timestamp: float) -> list[FusedObservation]:
        if not self.calibrations:
            raise RuntimeError(
                "TriangulationFusion needs camera calibrations (intrinsics + "
                "relative pose). Run calibration first, or use FeatureFusion "
                "for the calibration-free path."
            )
        groups = associate(views, self.primary_camera)
        out: list[FusedObservation] = []
        for group in groups:
            obs = self._fuse_group(group, timestamp)
            if obs is not None:
                out.append(obs)
        return out

    def _fuse_group(self, group: HandGroup, timestamp: float) -> FusedObservation | None:
        # Keep only views we can actually project with.
        calibrated = [(cam_id, pose) for cam_id, pose in group
                      if cam_id in self.calibrations]
        if len(calibrated) < 2:
            return None  # need >=2 calibrated views to reconstruct
        # Time-sync: DLT assumes one instant. A camera whose frame is much
        # older than the freshest one drags the reconstruction toward where the
        # hand WAS, so drop it. With a single positive max_view_skew_s, we
        # tolerate up to that many seconds behind the newest view in the group.
        if self.max_view_skew_s > 0:
            newest = max(pose.timestamp for _, pose in calibrated)
            calibrated = [(cid, pose) for cid, pose in calibrated
                          if newest - pose.timestamp <= self.max_view_skew_s]
            if len(calibrated) < 2:
                return None

        n_landmarks = calibrated[0][1].points.shape[0]
        world = np.empty((n_landmarks, 3), dtype=np.float32)
        for lm in range(n_landmarks):
            rays = []
            for cam_id, pose in calibrated:
                cal = self.calibrations[cam_id]
                w, h = cal.image_size
                # MediaPipe gives normalised [0,1]; lift to pixels for P.
                px = np.array([pose.points[lm, 0] * w, pose.points[lm, 1] * h])
                rays.append((cal.projection, px))
            world[lm] = triangulate_dlt(rays)

        # Cursor position stays in the primary camera's image space (until
        # spatial monitor mapping converts the metric 3D directly).
        pos_pose = next((p for cid, p in calibrated if cid == self.primary_camera),
                        calibrated[0][1])

        fused_pose = HandPose(
            points=pos_pose.points.copy(),   # image space -> cursor + view gate
            world_points=world,              # metric 3D -> angle-proof shape
            handedness=group[0][1].handedness,
            score=group[0][1].score,
            timestamp=timestamp,
        )
        features = self._recognizer.recognize(fused_pose)
        features.timestamp = timestamp
        return FusedObservation(
            features=features,
            pose=fused_pose,
            handedness=fused_pose.handedness,
            contributing_cameras=len(calibrated),
            position_camera=self.primary_camera,
        )
