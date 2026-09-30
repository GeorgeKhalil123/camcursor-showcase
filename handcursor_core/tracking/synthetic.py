"""TOY STAND-IN for the MediaPipe hand tracker.

The full app wraps MediaPipe's Hand Landmarker (``tracking/hand_tracker.py``,
private) and emits one :class:`HandPose` per detected hand per frame. This
module emits the same type from hand-built skeletons instead of a camera, so
the recognizer, state machine, smoothing and fusion layers can be exercised
end to end with no hardware and no model file.

Three postures are modelled — enough to walk the state machine through every
interaction the README describes:

  * ``"open"``  — all five fingers straight (MOVING)
  * ``"pinch"`` — index bent toward the thumb, tips touching, middle/ring/pinky
                  straight (PINCH_CLICK / DRAGGING)
  * ``"fist"``  — every finger curled, thumb resting on the index (DISABLED)

Geometry is in metres, hand-centred (wrist at the origin, fingers along +y),
roughly adult-hand sized. ``world_points`` carries it directly; ``points`` is
the same skeleton projected into a normalised image frame at ``center``.
"""

from __future__ import annotations

import numpy as np

from .types import FINGER_JOINTS, HandPose, Landmark

# Knuckle x-positions across the palm (index..pinky), metres.
_MCP_X = {"index": -0.03, "middle": -0.01, "ring": 0.01, "pinky": 0.03}
_MCP_Y = 0.08


def _straight(x: float) -> list[list[float]]:
    return [[x, _MCP_Y, 0.0], [x, 0.11, 0.0], [x, 0.13, 0.0], [x, 0.15, 0.0]]


def _curled(x: float) -> list[list[float]]:
    # Folds back toward the knuckle: low straightness AND low reach.
    return [[x, _MCP_Y, 0.0], [x, 0.10, 0.0], [x, 0.09, 0.01], [x, _MCP_Y, 0.01]]


def skeleton(posture: str) -> np.ndarray:
    """(21, 3) hand-centred world landmarks for ``posture``."""
    pts = np.zeros((21, 3), dtype=np.float64)
    pts[Landmark.WRIST] = [0.0, 0.0, 0.0]

    thumb = [[-0.03, 0.02, 0.0], [-0.05, 0.04, 0.0], [-0.06, 0.06, 0.0], [-0.07, 0.08, 0.0]]
    fingers: dict[str, list[list[float]]] = {}
    if posture == "open":
        fingers = {name: _straight(x) for name, x in _MCP_X.items()}
    elif posture == "pinch":
        fingers = {name: _straight(x) for name, x in _MCP_X.items()}
        # Index reaches over to meet the thumb tip.
        fingers["index"] = [[-0.03, _MCP_Y, 0.0], [-0.035, 0.11, 0.0],
                            [-0.045, 0.125, 0.0], [-0.055, 0.125, 0.0]]
        thumb[3] = [-0.057, 0.122, 0.0]
    elif posture == "fist":
        fingers = {name: _curled(x) for name, x in _MCP_X.items()}
        thumb[3] = [-0.03, 0.085, 0.01]   # thumb tucked over the curled index
    else:
        raise ValueError(f"unknown posture: {posture!r}")

    for name, joints in [("thumb", thumb), *fingers.items()]:
        for lm, xyz in zip(FINGER_JOINTS[name], joints):
            pts[lm] = xyz
    return pts


def synthetic_pose(posture: str, center: tuple[float, float], timestamp: float,
                   scale: float = 2.0, jitter: float = 0.0,
                   rng: np.random.Generator | None = None,
                   handedness: str = "Right") -> HandPose:
    """A :class:`HandPose` shaped like MediaPipe output.

    ``center`` places the wrist in normalised image coords; ``scale`` converts
    metres to image units; ``jitter`` adds Gaussian noise (image units) to
    mimic landmark tremor so the smoothing filter has something to do.
    """
    world = skeleton(posture)
    img = np.zeros((21, 3), dtype=np.float64)
    img[:, 0] = center[0] + world[:, 0] * scale
    img[:, 1] = center[1] - world[:, 1] * scale      # image y grows downward
    img[:, 2] = world[:, 2]
    if jitter > 0.0:
        rng = rng or np.random.default_rng(0)
        img[:, :2] += rng.normal(0.0, jitter, (21, 2))
    return HandPose(points=img.astype(np.float32), world_points=world.astype(np.float32),
                    handedness=handedness, score=0.95, timestamp=timestamp)
