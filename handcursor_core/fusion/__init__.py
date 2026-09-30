"""Multi-camera fusion layer.

Turns N cameras' per-frame detections into one fused observation per hand, so
the gesture recognizer, state machine, and cursor never learn how many cameras
there are. Two strategies share one interface:

  * :class:`FeatureFusion` — calibration-free; blends derived gesture features.
  * :class:`TriangulationFusion` — reconstructs a metric 3D hand (needs calibration).

Build one via :func:`make_fuser` from config and call ``fuse(views, ts)``.
"""

from __future__ import annotations

from ..config import FusionConfig, GestureConfig
from .base import PoseFuser
from .calibration_io import load_calibrations, save_calibrations
from .feature_fusion import FeatureFusion
from .triangulation import CameraCalibration, TriangulationFusion, triangulate_dlt
from .types import CameraView, FusedObservation

__all__ = [
    "PoseFuser",
    "FeatureFusion",
    "TriangulationFusion",
    "CameraCalibration",
    "triangulate_dlt",
    "CameraView",
    "FusedObservation",
    "make_fuser",
    "load_calibrations",
    "save_calibrations",
]


def make_fuser(fusion_cfg: FusionConfig, gesture_cfg: GestureConfig,
               anchor: str = "mcp", calibrations=None) -> PoseFuser:
    """Construct the fuser named by ``fusion_cfg.strategy``."""
    if fusion_cfg.strategy == "triangulation":
        return TriangulationFusion(gesture_cfg, anchor=anchor,
                                   calibrations=calibrations,
                                   primary_camera=fusion_cfg.primary_camera,
                                   max_view_skew_s=fusion_cfg.max_view_skew_s)
    if fusion_cfg.strategy == "feature":
        return FeatureFusion(gesture_cfg, anchor=anchor,
                             primary_camera=fusion_cfg.primary_camera,
                             min_view_weight=fusion_cfg.min_view_weight,
                             vote_threshold=fusion_cfg.vote_threshold)
    raise ValueError(f"Unknown fusion strategy: {fusion_cfg.strategy!r}")
