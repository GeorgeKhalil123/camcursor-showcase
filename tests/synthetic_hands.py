"""Synthetic hands and cameras shared by the fusion / triangulation tests.

Ported from the helper section of the full project's ``tests/test_fusion.py``.
"""

from __future__ import annotations

import numpy as np

from handcursor_core.fusion import CameraCalibration
from handcursor_core.tracking.types import FINGER_JOINTS, HandPose, Landmark


def _make_image_pts(rng: np.random.Generator | None = None) -> np.ndarray:
    """21x3 float32 image-space points in [0,1] with a realistic face-on palm.

    The wrist+MCP landmarks are spread so view_quality comes out well above the
    gate threshold, guaranteeing consistent feature extraction.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    pts = rng.random((21, 3)).astype(np.float32) * 0.3 + 0.35  # centre cluster
    # Fix the key landmarks that drive view_quality and position features.
    pts[Landmark.WRIST]      = [0.50, 0.80, 0.0]
    pts[Landmark.INDEX_MCP]  = [0.40, 0.55, 0.0]
    pts[Landmark.MIDDLE_MCP] = [0.50, 0.52, 0.0]
    pts[Landmark.RING_MCP]   = [0.60, 0.54, 0.0]
    pts[Landmark.PINKY_MCP]  = [0.68, 0.58, 0.0]
    pts[Landmark.INDEX_TIP]  = [0.38, 0.30, 0.0]
    pts[Landmark.THUMB_TIP]  = [0.32, 0.45, 0.0]
    return pts


def _make_world_pts_straight_index() -> np.ndarray:
    """21x3 float32 metric world points with index finger collinear (straight)
    and all other fingers curled, and thumb/index tips well separated.

    Joint layout (world, hand-centred, metres):
      - All joints initialised near origin.
      - Each finger's 4 joints are placed along a column so they are collinear
        when straight, or fold back when curled.
    """
    pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist at origin
    pts[Landmark.WRIST] = [0.0, 0.0, 0.0]

    xs = {
        "thumb":  -0.04,
        "index":  -0.01,
        "middle":  0.01,
        "ring":    0.03,
        "pinky":   0.05,
    }
    for name, (a, b, c, d) in FINGER_JOINTS.items():
        x = xs[name]
        if name == "index":
            # Straight: joints collinear along y axis
            pts[a] = [x, 0.04, 0.0]
            pts[b] = [x, 0.07, 0.0]
            pts[c] = [x, 0.09, 0.0]
            pts[d] = [x, 0.11, 0.0]
        else:
            # Curled: joints fold back toward palm
            pts[a] = [x, 0.04, 0.0]
            pts[b] = [x, 0.06, 0.0]
            pts[c] = [x, 0.05, 0.01]
            pts[d] = [x, 0.04, 0.01]

    # Thumb tip far from index tip for large (unpinched) distance
    pts[Landmark.THUMB_TIP]  = [-0.05, 0.06, 0.0]
    pts[Landmark.INDEX_TIP]  = [-0.01, 0.11, 0.0]  # from straight index above

    return pts


def _make_world_pts_pinched() -> np.ndarray:
    """World points with thumb and index tips very close (pinched)."""
    pts = _make_world_pts_straight_index()
    # Move thumb tip right next to index tip
    tip_pos = pts[Landmark.INDEX_TIP].copy()
    pts[Landmark.THUMB_TIP] = tip_pos + np.array([0.001, 0.001, 0.0], dtype=np.float32)
    return pts


def _neutral_hand(handedness: str = "Right", ts: float = 0.0,
                  score: float = 0.9) -> HandPose:
    """A deterministic neutral hand with a face-on palm and straight index finger."""
    return HandPose(
        points=_make_image_pts(),
        world_points=_make_world_pts_straight_index(),
        handedness=handedness,
        score=score,
        timestamp=ts,
    )


def _hand_with_extended_fingers(extended_names: set[str],
                                 handedness: str = "Right",
                                 ts: float = 0.0) -> HandPose:
    """Build a hand where exactly the named fingers are straight (collinear).

    Each finger's 4 joints (a,b,c,d) are placed along a separate column in world
    space so straightness = chord/path ≈ 1.0 when straight and ≈ 0.5 when curled.
    The FINGER_JOINTS loop sets all tips (including THUMB_TIP and INDEX_TIP) via
    the 'd' joint — we do NOT override them afterward so curled fingers stay curled.
    """
    pts_img = _make_image_pts()
    world = np.zeros((21, 3), dtype=np.float32)
    world[Landmark.WRIST] = [0.0, 0.0, 0.0]
    xs = {
        "thumb":  -0.04,
        "index":  -0.01,
        "middle":  0.01,
        "ring":    0.03,
        "pinky":   0.05,
    }
    for name, (a, b, c, d) in FINGER_JOINTS.items():
        x = xs[name]
        if name in extended_names:
            world[a] = [x, 0.04, 0.0]
            world[b] = [x, 0.07, 0.0]
            world[c] = [x, 0.09, 0.0]
            world[d] = [x, 0.11, 0.0]
        else:
            # Curled: tip folds back close to the base knuckle.
            # This keeps reach (tip-to-mcp / hand_size) low (~0.17) so the
            # reach signal does NOT fire the "extended" gate.
            world[a] = [x, 0.04, 0.0]
            world[b] = [x, 0.06, 0.0]
            world[c] = [x, 0.05, 0.01]
            world[d] = [x, 0.04, 0.01]
    # NOTE: deliberately NOT overriding THUMB_TIP or INDEX_TIP here.
    # The loop above sets them correctly via FINGER_JOINTS d-joints.

    return HandPose(
        points=pts_img,
        world_points=world,
        handedness=handedness,
        score=0.9,
        timestamp=ts,
    )


def _make_camera_calibration(camera_id: int, tx: float = 0.0) -> CameraCalibration:
    """A synthetic camera: standard K, identity R, translation along X by tx metres."""
    K = np.array([[800.0, 0.0, 640.0],
                  [0.0, 800.0, 360.0],
                  [0.0, 0.0,   1.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([tx, 0.0, 0.0], dtype=np.float64)
    return CameraCalibration(camera_id=camera_id, K=K, R=R, t=t,
                             image_size=(1280, 720))


def _project_3d_to_pixel(cal: CameraCalibration, pt3d: np.ndarray) -> np.ndarray:
    """Project a 3D world point through a CameraCalibration into pixel coords."""
    P = cal.projection
    x_h = np.append(pt3d.astype(np.float64), 1.0)
    px_h = P @ x_h
    return (px_h[:2] / px_h[2]).astype(np.float64)


def _make_hand_image_pts_from_3d(cal: CameraCalibration,
                                   hand3d: np.ndarray) -> np.ndarray:
    """Project a (21,3) world-space hand into normalised image coords for a camera.

    The triangulation fuser lifts ``pose.points[lm, :2]`` from [0,1] to pixel
    space by multiplying by ``image_size``. This helper is the inverse: take pixel
    coords from the projection and divide by image_size to get the normalised
    values the fuser expects.
    """
    W, H = cal.image_size
    pts = np.zeros((21, 3), dtype=np.float32)
    for i, p3 in enumerate(hand3d):
        px = _project_3d_to_pixel(cal, p3)
        pts[i, 0] = float(px[0] / W)
        pts[i, 1] = float(px[1] / H)
        pts[i, 2] = 0.0
    return pts


def _make_synthetic_3d_hand(seed: int = 99) -> np.ndarray:
    """A reproducible (21,3) hand in world space, centred at z=0.8m."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0.0, 0.03, (21, 3)).astype(np.float64)
    pts[:, 2] += 0.8   # depth 0.8 m so it projects into image plane
    return pts


def _make_face_on_image_pts() -> np.ndarray:
    """Image points with a clearly face-on palm (high view_quality)."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[Landmark.WRIST]      = [0.50, 0.80, 0.0]
    pts[Landmark.INDEX_MCP]  = [0.40, 0.55, 0.0]
    pts[Landmark.MIDDLE_MCP] = [0.50, 0.52, 0.0]
    pts[Landmark.RING_MCP]   = [0.60, 0.54, 0.0]
    pts[Landmark.PINKY_MCP]  = [0.68, 0.58, 0.0]
    pts[Landmark.INDEX_TIP]  = [0.38, 0.30, 0.0]
    pts[Landmark.THUMB_TIP]  = [0.32, 0.45, 0.0]
    # Fill remaining landmarks with reasonable positions
    rng = np.random.default_rng(7)
    for i in range(21):
        if np.all(pts[i] == 0):
            pts[i] = rng.random(3).astype(np.float32) * 0.2 + 0.4
    return pts


def _make_edge_on_image_pts() -> np.ndarray:
    """Image points with an edge-on palm (low view_quality, near collinear MCPs)."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[Landmark.WRIST]      = [0.50, 0.80, 0.0]
    pts[Landmark.INDEX_MCP]  = [0.50, 0.55, 0.0]
    pts[Landmark.MIDDLE_MCP] = [0.505, 0.52, 0.0]
    pts[Landmark.RING_MCP]   = [0.51, 0.54, 0.0]
    pts[Landmark.PINKY_MCP]  = [0.515, 0.58, 0.0]
    pts[Landmark.INDEX_TIP]  = [0.50, 0.30, 0.0]
    pts[Landmark.THUMB_TIP]  = [0.52, 0.45, 0.0]
    rng = np.random.default_rng(13)
    for i in range(21):
        if np.all(pts[i] == 0):
            pts[i] = rng.random(3).astype(np.float32) * 0.2 + 0.4
    return pts
