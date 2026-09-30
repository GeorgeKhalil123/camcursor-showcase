"""Landmark -> :class:`GestureFeatures`.

All features are normalised by hand size so they're invariant to how close the
hand is to the camera. That invariance is exactly what makes the same gesture
rules keep working when depth comes from triangulation later.
"""

from __future__ import annotations

import numpy as np

from ..config import GestureConfig
from ..geometry import distance
from ..tracking.types import FINGER_JOINTS, FINGER_NAMES, GestureFeatures, HandPose, Landmark


class GestureRecognizer:
    def __init__(self, config: GestureConfig, anchor: str = "mcp") -> None:
        self.cfg = config
        self.anchor = anchor  # which landmark drives the cursor: tip/mcp/blend
        self._ext_state: dict[str, bool] = {}  # per-finger extended state (hysteresis)

    def recognize(self, pose: HandPose) -> GestureFeatures:
        pts = pose.points[:, :2]  # IMAGE space — drives cursor position

        # SHAPE features come from the metric, hand-centred world landmarks when
        # available: distances/ratios there are orientation-robust, so a hand at
        # an angle is read far more reliably than from the image projection.
        geo = pose.world_points if pose.world_points is not None else pose.points
        geo_hand = distance(geo[Landmark.INDEX_MCP], geo[Landmark.PINKY_MCP])
        geo_hand = max(geo_hand, 1e-6)

        # Two extension signals (either one, with hysteresis, marks "extended"):
        #   straightness — chord/path ratio (is the finger straight?)
        #   reach        — tip-to-knuckle distance / hand size (is it reaching
        #                  out?). Catches fingers extended but bent (reaching to
        #                  the thumb in a pinch) that straightness alone misses.
        straightness = {name: self._straightness(geo, name) for name in FINGER_NAMES}
        reach = {name: self._reach(geo, name, geo_hand) for name in FINGER_NAMES}
        fingers_extended = {name: self._extended(name, straightness[name], reach[name])
                            for name in FINGER_NAMES}

        # Pinch from world landmarks: thumb-tip <-> index-tip distance. (An
        # earlier "thumb to index segment" variant was reverted — it read a
        # closed fist, where the thumb rests against the curled index, as the
        # strongest pinch of all. Fist vs pinch is separated downstream by the
        # index-reach floor, not by reaching to the proximal index joints.)
        pinch_distance = distance(geo[Landmark.THUMB_TIP], geo[Landmark.INDEX_TIP]) / geo_hand

        # Image-space points for cursor position + view-quality gating.
        hand_size = max(distance(pts[Landmark.INDEX_MCP], pts[Landmark.PINKY_MCP]), 1e-6)
        index_tip = pts[Landmark.INDEX_TIP]
        index_mcp = pts[Landmark.INDEX_MCP]
        view_quality = self._view_quality(pts)

        palm_center = pts[[Landmark.WRIST, Landmark.INDEX_MCP,
                           Landmark.MIDDLE_MCP, Landmark.RING_MCP,
                           Landmark.PINKY_MCP]].mean(axis=0)

        # The cursor anchor. "palm" (the palm centroid) is the steadiest — no
        # per-finger tremor — and the natural choice now that an open hand
        # drives movement. "mcp" (knuckle) is stable through pinch; "tip" is
        # precise but drifts; "blend" is the tip/knuckle midpoint.
        if self.anchor == "palm":
            pointer = palm_center
        elif self.anchor == "tip":
            pointer = index_tip
        elif self.anchor == "blend":
            pointer = (index_tip + index_mcp) * 0.5
        else:  # "mcp"
            pointer = index_mcp

        return GestureFeatures(
            fingers_extended=fingers_extended,
            pinch_distance=pinch_distance,
            index_straightness=straightness["index"],
            index_reach=reach["index"],
            view_quality=view_quality,
            index_tip=index_tip.copy(),
            index_mcp=index_mcp.copy(),
            pointer=pointer.copy(),
            palm_center=palm_center,
            hand_size=hand_size,
            timestamp=pose.timestamp,
        )

    def _straightness(self, pts: np.ndarray, finger: str) -> float:
        """Ratio of the finger's chord (base->tip) to its summed joint segments.

        A straight finger's joints are collinear so chord == path -> ratio ~1.0;
        a curled finger folds back so the path is much longer -> ratio drops.
        Rotation/scale/translation invariant, and far steadier than comparing
        landmark distances to the wrist. Operates on 3D points (x,y,z) so it is
        robust to fingers pointing along the camera axis (2D foreshortening).
        """
        a, b, c, d = FINGER_JOINTS[finger]  # base, pip, dip, tip
        chord = distance(pts[a], pts[d])
        path = distance(pts[a], pts[b]) + distance(pts[b], pts[c]) + distance(pts[c], pts[d])
        return chord / path if path > 1e-6 else 0.0

    def _reach(self, pts: np.ndarray, finger: str, hand_size: float) -> float:
        """Tip-to-knuckle distance normalised by hand size (3D).

        Large when the finger is extended in any direction — including bent
        toward the thumb (pinch) or angled at the camera — and small when it's
        curled back into the palm. Complements straightness, which only sees
        *straight* fingers.
        """
        base, _, _, tip = FINGER_JOINTS[finger]
        return distance(pts[base], pts[tip]) / hand_size

    def _view_quality(self, img_pts: np.ndarray) -> float:
        """How well the palm faces the camera, as the 2D area of the palm fan
        (wrist + 4 MCPs) normalised by palm LENGTH**2 (wrist->middle knuckle).

        Face-on, those five points spread into a real polygon -> large area.
        Edge-on / oblique, they project nearly collinear -> area ~0, which is
        exactly when monocular tracking becomes unreliable. Palm length is the
        normaliser (not the MCP span, which itself collapses edge-on and would
        make the ratio blow up).
        """
        idx = [Landmark.WRIST, Landmark.INDEX_MCP, Landmark.MIDDLE_MCP,
               Landmark.RING_MCP, Landmark.PINKY_MCP]
        poly = img_pts[idx]
        x, y = poly[:, 0], poly[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        palm_len = max(distance(img_pts[Landmark.WRIST], img_pts[Landmark.MIDDLE_MCP]), 1e-6)
        return float(area / (palm_len * palm_len))

    def _extended(self, finger: str, straightness: float, reach: float) -> bool:
        """Extended if straight OR reaching, with hysteresis on the result."""
        c = self.cfg
        if self._ext_state.get(finger, False):
            ext = straightness > c.finger_extend_off or reach > c.finger_reach_off
        else:
            ext = straightness > c.finger_extend_on or reach > c.finger_reach_on
        self._ext_state[finger] = ext
        return ext
