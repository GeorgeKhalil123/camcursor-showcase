"""Hardware-free tests for the multi-camera fusion layer: single-camera
passthrough, cross-camera association, and calibration-free feature fusion.

Ported from the full project's ``tests/test_fusion.py``. All tests use
synthetic HandPose objects — no camera, no MediaPipe.
"""

from __future__ import annotations

import numpy as np
from synthetic_hands import (
    _hand_with_extended_fingers,
    _make_edge_on_image_pts,
    _make_face_on_image_pts,
    _make_image_pts,
    _make_world_pts_straight_index,
    _neutral_hand,
)

from handcursor_core.config import GestureConfig
from handcursor_core.fusion import CameraView, FeatureFusion
from handcursor_core.gestures.recognizer import GestureRecognizer
from handcursor_core.tracking.types import FINGER_NAMES, HandPose

# ---------------------------------------------------------------------------
# 1. SINGLE-CAMERA PASSTHROUGH
# ---------------------------------------------------------------------------

def test_single_camera_passthrough_pointer():
    """FeatureFusion over one camera/pose produces pointer == recognizer result."""
    pose = _neutral_hand()
    cfg = GestureConfig()
    ref = GestureRecognizer(cfg, anchor="mcp").recognize(pose)

    fuser = FeatureFusion(cfg, anchor="mcp", primary_camera=0)
    view = CameraView(camera_id=0, poses=[pose], timestamp=0.0)
    obs_list = fuser.fuse([view], timestamp=0.0)

    assert len(obs_list) == 1
    fused = obs_list[0].features
    assert np.allclose(fused.pointer, ref.pointer, atol=1e-9), (
        f"pointer mismatch: fused={fused.pointer} ref={ref.pointer}"
    )


def test_single_camera_passthrough_pinch_distance():
    """FeatureFusion pinch_distance equals recognizer pinch_distance (within 1e-9)."""
    pose = _neutral_hand()
    cfg = GestureConfig()
    ref = GestureRecognizer(cfg, anchor="mcp").recognize(pose)

    fuser = FeatureFusion(cfg, anchor="mcp", primary_camera=0)
    view = CameraView(camera_id=0, poses=[pose], timestamp=0.0)
    fused = fuser.fuse([view], timestamp=0.0)[0].features

    assert abs(fused.pinch_distance - ref.pinch_distance) < 1e-9, (
        f"pinch_distance: fused={fused.pinch_distance:.6f} ref={ref.pinch_distance:.6f}"
    )


def test_single_camera_passthrough_fingers_extended():
    """FeatureFusion fingers_extended dict matches recognizer's exactly."""
    pose = _neutral_hand()
    cfg = GestureConfig()
    ref = GestureRecognizer(cfg, anchor="mcp").recognize(pose)

    fuser = FeatureFusion(cfg, anchor="mcp", primary_camera=0)
    view = CameraView(camera_id=0, poses=[pose], timestamp=0.0)
    fused = fuser.fuse([view], timestamp=0.0)[0].features

    for name in FINGER_NAMES:
        assert fused.fingers_extended[name] == ref.fingers_extended[name], (
            f"finger {name}: fused={fused.fingers_extended[name]} "
            f"ref={ref.fingers_extended[name]}"
        )


def test_single_camera_passthrough_view_quality():
    """FeatureFusion view_quality equals recognizer view_quality."""
    pose = _neutral_hand()
    cfg = GestureConfig()
    ref = GestureRecognizer(cfg, anchor="mcp").recognize(pose)

    fuser = FeatureFusion(cfg, anchor="mcp", primary_camera=0)
    view = CameraView(camera_id=0, poses=[pose], timestamp=0.0)
    fused = fuser.fuse([view], timestamp=0.0)[0].features

    assert abs(fused.view_quality - ref.view_quality) < 1e-9, (
        f"view_quality: fused={fused.view_quality:.6f} ref={ref.view_quality:.6f}"
    )


def test_single_camera_passthrough_palm_center():
    """FeatureFusion palm_center equals recognizer palm_center."""
    pose = _neutral_hand()
    cfg = GestureConfig()
    ref = GestureRecognizer(cfg, anchor="mcp").recognize(pose)

    fuser = FeatureFusion(cfg, anchor="mcp", primary_camera=0)
    view = CameraView(camera_id=0, poses=[pose], timestamp=0.0)
    fused = fuser.fuse([view], timestamp=0.0)[0].features

    assert np.allclose(fused.palm_center, ref.palm_center, atol=1e-9), (
        f"palm_center: fused={fused.palm_center} ref={ref.palm_center}"
    )


# ---------------------------------------------------------------------------
# 2. ASSOCIATION
# ---------------------------------------------------------------------------

def test_association_left_right_group_together():
    """Left and Right hands each form their own group across two cameras."""
    from handcursor_core.fusion.association import associate

    pose_left_cam0  = HandPose(points=_make_image_pts(), handedness="Left")
    pose_right_cam0 = HandPose(points=_make_image_pts(), handedness="Right")
    pose_left_cam1  = HandPose(points=_make_image_pts(), handedness="Left")
    pose_right_cam1 = HandPose(points=_make_image_pts(), handedness="Right")

    views = [
        CameraView(camera_id=0, poses=[pose_left_cam0, pose_right_cam0]),
        CameraView(camera_id=1, poses=[pose_left_cam1, pose_right_cam1]),
    ]
    groups = associate(views, primary_camera=0)

    # Exactly two groups: one per handedness label.
    assert len(groups) == 2, f"expected 2 groups, got {len(groups)}"

    # Each group has two entries (one per camera).
    for g in groups:
        assert len(g) == 2, f"expected 2 entries per group, got {len(g)}"

    # Groups contain consistent handedness within each group.
    for g in groups:
        handedness_values = {pose.handedness for _, pose in g}
        assert len(handedness_values) == 1, (
            f"mixed handedness in a group: {handedness_values}"
        )


def test_association_primary_camera_first():
    """Groups are ordered so the primary camera's hands come first."""
    from handcursor_core.fusion.association import associate

    # Camera 1 is primary; camera 0 is secondary.
    pose_primary_right = HandPose(points=_make_image_pts(), handedness="Right")
    pose_primary_left  = HandPose(points=_make_image_pts(), handedness="Left")
    pose_secondary     = HandPose(points=_make_image_pts(), handedness="Right")

    views = [
        CameraView(camera_id=0, poses=[pose_secondary]),   # secondary cam
        CameraView(camera_id=1, poses=[pose_primary_right, pose_primary_left]),  # primary
    ]
    groups = associate(views, primary_camera=1)

    # First group should contain the primary camera's first pose (Right hand).
    first_group_cam_ids = [cam_id for cam_id, _ in groups[0]]
    assert 1 in first_group_cam_ids, (
        "Primary camera's hand should be in the first group"
    )


def test_association_primary_hand_at_index_zero():
    """The very first group always belongs to the primary camera."""
    from handcursor_core.fusion.association import associate

    pose_a = HandPose(points=_make_image_pts(), handedness="Right")
    pose_b = HandPose(points=_make_image_pts(), handedness="Left")   # only in secondary

    views = [
        CameraView(camera_id=0, poses=[pose_a]),
        CameraView(camera_id=1, poses=[pose_b]),   # different handedness, secondary only
    ]
    groups = associate(views, primary_camera=0)

    # The first group must include camera 0's pose.
    first_cam_ids = {cam_id for cam_id, _ in groups[0]}
    assert 0 in first_cam_ids


def test_association_unknown_handedness_falls_back_to_slot():
    """Unknown-labelled poses group by slot index, not by label."""
    from handcursor_core.fusion.association import associate

    # Both cameras see two Unknown hands.
    poses_cam0 = [
        HandPose(points=_make_image_pts(), handedness="Unknown"),
        HandPose(points=_make_image_pts(), handedness="Unknown"),
    ]
    poses_cam1 = [
        HandPose(points=_make_image_pts(), handedness="Unknown"),
        HandPose(points=_make_image_pts(), handedness="Unknown"),
    ]

    views = [
        CameraView(camera_id=0, poses=poses_cam0),
        CameraView(camera_id=1, poses=poses_cam1),
    ]
    groups = associate(views, primary_camera=0)

    # Slot 0 pairs with slot 0, slot 1 pairs with slot 1 -> 2 groups of 2.
    assert len(groups) == 2, f"expected 2 groups, got {len(groups)}"
    for g in groups:
        assert len(g) == 2, f"expected 2 per slot group, got {len(g)}"

    # Each group should contain one entry from each camera.
    for g in groups:
        cam_ids = {cam_id for cam_id, _ in g}
        assert cam_ids == {0, 1}, f"each Unknown group should span both cameras, got {cam_ids}"


def test_association_only_secondary_camera_hands_appended():
    """Hands seen only by non-primary cameras are appended after primary's hands."""
    from handcursor_core.fusion.association import associate

    pose_primary = HandPose(points=_make_image_pts(), handedness="Right")
    pose_secondary_only = HandPose(points=_make_image_pts(), handedness="Left")

    views = [
        CameraView(camera_id=0, poses=[pose_primary]),            # primary
        CameraView(camera_id=1, poses=[pose_secondary_only]),     # secondary, Left only
    ]
    groups = associate(views, primary_camera=0)

    # Two groups: Right (from cam 0) and Left (from cam 1).
    assert len(groups) == 2

    # First group is the primary camera's Right hand.
    first_handedness = {pose.handedness for _, pose in groups[0]}
    assert first_handedness == {"Right"}

    # Second group is the secondary camera's Left hand.
    second_handedness = {pose.handedness for _, pose in groups[1]}
    assert second_handedness == {"Left"}


# ---------------------------------------------------------------------------
# 3. FEATURE FUSION ACROSS 2+ CAMERAS
# ---------------------------------------------------------------------------

def test_fusion_view_quality_is_max_across_cameras():
    """Fused view_quality equals the maximum over all cameras (best view wins)."""
    cfg = GestureConfig()

    # Camera 0 (primary): face-on, high view_quality.
    world = _make_world_pts_straight_index()
    pose0 = HandPose(points=_make_face_on_image_pts(), world_points=world,
                     handedness="Right", score=0.9)
    # Camera 1: edge-on, low view_quality.
    pose1 = HandPose(points=_make_edge_on_image_pts(), world_points=world,
                     handedness="Right", score=0.9)

    rec0 = GestureRecognizer(cfg, anchor="mcp").recognize(pose0)
    rec1 = GestureRecognizer(cfg, anchor="mcp").recognize(pose1)
    expected_max_vq = max(rec0.view_quality, rec1.view_quality)

    fuser = FeatureFusion(cfg, anchor="mcp", primary_camera=0)
    views = [
        CameraView(camera_id=0, poses=[pose0]),
        CameraView(camera_id=1, poses=[pose1]),
    ]
    fused = fuser.fuse(views, timestamp=0.0)[0].features

    assert abs(fused.view_quality - expected_max_vq) < 1e-9, (
        f"view_quality: fused={fused.view_quality:.6f} expected max={expected_max_vq:.6f}"
    )
    # The fused view_quality should be higher than the edge-on camera's quality.
    assert fused.view_quality > rec1.view_quality


def test_fusion_pinch_is_view_quality_weighted_average():
    """Fused pinch_distance equals view_quality-weighted average over cameras."""
    cfg = GestureConfig(min_view_quality=0.0)  # no gate so both cameras always contribute
    min_w = 0.05  # FeatureFusion default min_view_weight

    world = _make_world_pts_straight_index()
    pose0 = HandPose(points=_make_face_on_image_pts(), world_points=world,
                     handedness="Right", score=0.9)
    pose1 = HandPose(points=_make_edge_on_image_pts(), world_points=world,
                     handedness="Right", score=0.9)

    rec0 = GestureRecognizer(cfg, anchor="mcp").recognize(pose0)
    rec1 = GestureRecognizer(cfg, anchor="mcp").recognize(pose1)

    w0 = max(rec0.view_quality, min_w)
    w1 = max(rec1.view_quality, min_w)
    total = w0 + w1
    expected_pinch = (rec0.pinch_distance * w0 + rec1.pinch_distance * w1) / total

    fuser = FeatureFusion(cfg, anchor="mcp", primary_camera=0, min_view_weight=min_w)
    views = [
        CameraView(camera_id=0, poses=[pose0]),
        CameraView(camera_id=1, poses=[pose1]),
    ]
    fused = fuser.fuse(views, timestamp=0.0)[0].features

    assert abs(fused.pinch_distance - expected_pinch) < 1e-6, (
        f"pinch_distance: fused={fused.pinch_distance:.6f} expected={expected_pinch:.6f}"
    )


def test_fusion_high_quality_camera_dominates_finger_extension():
    """The high-view-quality camera's finger state wins the vote over low-quality."""
    cfg = GestureConfig()
    min_w = 0.05

    # Camera 0 (primary, HIGH view quality): index finger EXTENDED.
    # Camera 1 (secondary, LOW view quality): index finger CURLED.
    # High-quality camera should dominate -> fused says index IS extended.
    pose_high = HandPose(
        points=_make_face_on_image_pts(),
        world_points=_make_world_pts_straight_index(),  # straight index -> extended
        handedness="Right", score=0.9,
    )
    pose_low = HandPose(
        points=_make_edge_on_image_pts(),
        world_points=_hand_with_extended_fingers(set()).world_points,  # all curled
        handedness="Right", score=0.9,
    )

    # Verify the per-camera recognizers agree with our setup.
    rec_high = GestureRecognizer(cfg, anchor="mcp").recognize(pose_high)
    rec_low  = GestureRecognizer(cfg, anchor="mcp").recognize(pose_low)
    assert rec_high.fingers_extended["index"] is True, (
        "Setup error: high-quality camera should see index as extended"
    )
    assert rec_low.fingers_extended["index"] is False, (
        "Setup error: low-quality camera should see index as curled"
    )
    assert rec_high.view_quality > rec_low.view_quality, (
        "Setup error: face-on camera must have higher view_quality"
    )

    fuser = FeatureFusion(cfg, anchor="mcp", primary_camera=0, min_view_weight=min_w)
    views = [
        CameraView(camera_id=0, poses=[pose_high]),
        CameraView(camera_id=1, poses=[pose_low]),
    ]
    fused = fuser.fuse(views, timestamp=0.0)[0].features

    assert fused.fingers_extended["index"] is True, (
        "High-quality camera says index extended but fused says curled"
    )


def test_fusion_low_quality_camera_does_not_override_high():
    """Low-quality camera reporting extended does not beat a high-quality curled report."""
    cfg = GestureConfig()
    min_w = 0.05

    # Camera 0 (HIGH view quality): index finger CURLED.
    # Camera 1 (LOW view quality): index finger EXTENDED.
    # High-quality camera should dominate -> fused says index is CURLED.
    pose_high = HandPose(
        points=_make_face_on_image_pts(),
        world_points=_hand_with_extended_fingers(set()).world_points,  # all curled
        handedness="Right", score=0.9,
    )
    pose_low = HandPose(
        points=_make_edge_on_image_pts(),
        world_points=_make_world_pts_straight_index(),  # straight index -> extended
        handedness="Right", score=0.9,
    )

    rec_high = GestureRecognizer(cfg, anchor="mcp").recognize(pose_high)
    rec_low  = GestureRecognizer(cfg, anchor="mcp").recognize(pose_low)
    assert rec_high.fingers_extended["index"] is False, (
        "Setup error: high-quality camera should see index as curled"
    )
    assert rec_low.fingers_extended["index"] is True, (
        "Setup error: low-quality camera should see index as extended"
    )

    fuser = FeatureFusion(cfg, anchor="mcp", primary_camera=0, min_view_weight=min_w)
    views = [
        CameraView(camera_id=0, poses=[pose_high]),
        CameraView(camera_id=1, poses=[pose_low]),
    ]
    fused = fuser.fuse(views, timestamp=0.0)[0].features

    assert fused.fingers_extended["index"] is False, (
        "Low-quality camera says extended but high-quality says curled; fused should be curled"
    )


def test_fusion_position_from_primary_camera():
    """pointer, index_tip, and palm_center come from the primary camera, not an average."""
    cfg = GestureConfig()

    # Two cameras see the same world hand but different image-space positions.
    world = _make_world_pts_straight_index()

    # Primary camera image points (camera 0).
    img_primary = _make_face_on_image_pts()
    # Secondary camera with very different image-space positions.
    img_secondary = img_primary.copy()
    img_secondary[:, 0] = 1.0 - img_primary[:, 0]  # mirror x
    img_secondary[:, 1] = 1.0 - img_primary[:, 1]  # mirror y

    pose_primary   = HandPose(points=img_primary,   world_points=world, handedness="Right")
    pose_secondary = HandPose(points=img_secondary, world_points=world, handedness="Right")

    # Recognizers for reference values from the primary camera's perspective.
    ref_primary = GestureRecognizer(cfg, anchor="mcp").recognize(pose_primary)

    fuser = FeatureFusion(cfg, anchor="mcp", primary_camera=0)
    views = [
        CameraView(camera_id=0, poses=[pose_primary]),
        CameraView(camera_id=1, poses=[pose_secondary]),
    ]
    fused = fuser.fuse(views, timestamp=0.0)[0].features

    assert np.allclose(fused.pointer, ref_primary.pointer, atol=1e-9), (
        f"pointer should equal primary camera's: "
        f"fused={fused.pointer} primary={ref_primary.pointer}"
    )
    assert np.allclose(fused.index_tip, ref_primary.index_tip, atol=1e-9), (
        "index_tip should equal primary camera's"
    )
    assert np.allclose(fused.palm_center, ref_primary.palm_center, atol=1e-9), (
        "palm_center should equal primary camera's"
    )


def test_fusion_position_camera_id_recorded():
    """FusedObservation.position_camera records which camera supplied position."""
    cfg = GestureConfig()
    world = _make_world_pts_straight_index()
    pose0 = HandPose(points=_make_face_on_image_pts(), world_points=world, handedness="Right")
    pose1 = HandPose(points=_make_edge_on_image_pts(), world_points=world, handedness="Right")

    fuser = FeatureFusion(cfg, anchor="mcp", primary_camera=0)
    views = [
        CameraView(camera_id=0, poses=[pose0]),
        CameraView(camera_id=1, poses=[pose1]),
    ]
    obs = fuser.fuse(views, timestamp=0.0)[0]
    assert obs.position_camera == 0, (
        f"position_camera should be 0 (primary), got {obs.position_camera}"
    )


def test_fusion_contributing_cameras_count():
    """FusedObservation.contributing_cameras equals the number of cameras in the group."""
    cfg = GestureConfig()
    world = _make_world_pts_straight_index()
    pose0 = HandPose(points=_make_face_on_image_pts(), world_points=world, handedness="Right")
    pose1 = HandPose(points=_make_edge_on_image_pts(), world_points=world, handedness="Right")

    fuser = FeatureFusion(cfg, anchor="mcp", primary_camera=0)
    views = [
        CameraView(camera_id=0, poses=[pose0]),
        CameraView(camera_id=1, poses=[pose1]),
    ]
    obs = fuser.fuse(views, timestamp=0.0)[0]
    assert obs.contributing_cameras == 2, (
        f"expected contributing_cameras=2, got {obs.contributing_cameras}"
    )
