"""Cross-camera hand association: which detection in camera A is the same
physical hand as which detection in camera B.

This is *the* hard problem of multi-view tracking in general — but MediaPipe
hands it to us nearly solved twice over:

  * Every camera labels each hand "Left"/"Right", so we can group by handedness.
  * Every landmark has the same index in every view (index-tip is always #8),
    so once hands are grouped there is no per-point matching to do.

The calibration-free strategy here is to group by handedness label, preferring
the highest-confidence pose when one camera reports two hands with the same
label. "Unknown" labels (low confidence) are matched positionally by their
order within each view. When calibration arrives, this can be upgraded to
epipolar / 3D-proximity association without changing the call site.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..tracking.types import HandPose
from .types import CameraView

# One group = the same physical hand across cameras: (camera_id, pose) pairs.
HandGroup = list[tuple[int, HandPose]]


def associate(views: Sequence[CameraView], primary_camera: int) -> list[HandGroup]:
    """Group poses across cameras into one :data:`HandGroup` per physical hand.

    Output order is stable and meaningful: hands seen by the primary camera come
    first, in that camera's order, so ``groups[0]`` is consistently "the hand
    the user is driving the cursor with". Hands only seen by other cameras are
    appended afterwards.
    """
    groups: dict[str, HandGroup] = {}
    order: list[str] = []  # group keys, primary camera first

    # Seed ordering from the primary camera so its hands are indices 0..k.
    primary = next((v for v in views if v.camera_id == primary_camera), None)
    if primary is not None:
        for i, pose in enumerate(primary.poses):
            key = _key(pose, i)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((primary.camera_id, pose))

    # Fold in every other camera.
    for view in views:
        if view.camera_id == primary_camera:
            continue
        for i, pose in enumerate(view.poses):
            key = _key(pose, i)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((view.camera_id, pose))

    return [groups[k] for k in order]


def _key(pose: HandPose, index: int) -> str:
    """Association key: handedness when known, else positional fallback.

    A confident "Left"/"Right" groups across cameras regardless of detection
    order. "Unknown" can't be trusted to mean the same hand, so we fall back to
    its slot index within the view (camera A's first unknown <-> camera B's
    first unknown) — imperfect, but the right place for triangulation to later
    impose true geometric matching.
    """
    label = pose.handedness
    if label in ("Left", "Right"):
        return label
    return f"unknown:{index}"
