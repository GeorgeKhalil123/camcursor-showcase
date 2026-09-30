"""Hardware-free tests for DLT triangulation, TriangulationFusion, the fuser
factory, calibration I/O and multi-camera time-skew rejection.

Ported from the full project's ``tests/test_fusion.py``. Cameras and hands are
synthetic: a known 3D hand is projected through synthetic calibrations.
"""

from __future__ import annotations

import numpy as np
from synthetic_hands import (
    _make_camera_calibration,
    _make_face_on_image_pts,
    _make_hand_image_pts_from_3d,
    _make_synthetic_3d_hand,
    _make_world_pts_straight_index,
    _neutral_hand,
    _project_3d_to_pixel,
)

from handcursor_core.config import FusionConfig, GestureConfig
from handcursor_core.fusion import (
    CameraView,
    FeatureFusion,
    FusedObservation,
    TriangulationFusion,
    make_fuser,
    triangulate_dlt,
)
from handcursor_core.tracking.types import FINGER_NAMES, HandPose

# ---------------------------------------------------------------------------
# 4. TRIANGULATE_DLT
# ---------------------------------------------------------------------------

def test_triangulate_dlt_recovers_known_3d_point():
    """DLT recovers a known 3D point from two projected views to within 1e-4."""
    cal0 = _make_camera_calibration(camera_id=0, tx=0.0)
    cal1 = _make_camera_calibration(camera_id=1, tx=-0.15)  # 15 cm baseline

    # A known 3D world point (metres), chosen to project into both image planes.
    pt3d = np.array([0.05, 0.1, 0.8], dtype=np.float64)

    px0 = _project_3d_to_pixel(cal0, pt3d)
    px1 = _project_3d_to_pixel(cal1, pt3d)

    recovered = triangulate_dlt([(cal0.projection, px0), (cal1.projection, px1)])

    assert recovered.shape == (3,), f"expected shape (3,), got {recovered.shape}"
    error = np.linalg.norm(recovered.astype(np.float64) - pt3d)
    assert error < 1e-4, f"3D reconstruction error too large: {error:.2e}"


def test_triangulate_dlt_second_known_point():
    """DLT recovers a second arbitrary 3D point correctly."""
    cal0 = _make_camera_calibration(camera_id=0, tx=0.0)
    cal1 = _make_camera_calibration(camera_id=1, tx=0.2)

    pt3d = np.array([-0.1, -0.05, 1.2], dtype=np.float64)
    px0 = _project_3d_to_pixel(cal0, pt3d)
    px1 = _project_3d_to_pixel(cal1, pt3d)

    recovered = triangulate_dlt([(cal0.projection, px0), (cal1.projection, px1)])
    error = np.linalg.norm(recovered.astype(np.float64) - pt3d)
    assert error < 1e-4, f"3D reconstruction error too large: {error:.2e}"


def test_triangulate_dlt_raises_with_fewer_than_two_rays():
    """triangulate_dlt raises ValueError when called with < 2 camera rays."""
    cal = _make_camera_calibration(camera_id=0, tx=0.0)
    px = np.array([640.0, 360.0])

    try:
        triangulate_dlt([(cal.projection, px)])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "two" in str(e).lower() or "2" in str(e), (
            f"ValueError message should mention 2 views: {e}"
        )


def test_triangulate_dlt_raises_with_empty_rays():
    """triangulate_dlt raises ValueError when called with 0 rays."""
    try:
        triangulate_dlt([])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_triangulate_dlt_three_cameras_improves_accuracy():
    """With three cameras the DLT solution is at least as good as two."""
    cal0 = _make_camera_calibration(camera_id=0, tx=0.0)
    cal1 = _make_camera_calibration(camera_id=1, tx=-0.15)
    cal2 = _make_camera_calibration(camera_id=2, tx=0.15)

    pt3d = np.array([0.03, 0.07, 0.9], dtype=np.float64)
    px0 = _project_3d_to_pixel(cal0, pt3d)
    px1 = _project_3d_to_pixel(cal1, pt3d)
    px2 = _project_3d_to_pixel(cal2, pt3d)

    rec3 = triangulate_dlt([
        (cal0.projection, px0),
        (cal1.projection, px1),
        (cal2.projection, px2),
    ])
    error = np.linalg.norm(rec3.astype(np.float64) - pt3d)
    assert error < 1e-4, f"Three-camera DLT error too large: {error:.2e}"


# ---------------------------------------------------------------------------
# 5. TRIANGULATION FUSION
# ---------------------------------------------------------------------------

def test_triangulation_fusion_raises_without_calibrations():
    """TriangulationFusion.fuse() raises RuntimeError when no calibrations provided."""
    cfg = GestureConfig()
    fuser = TriangulationFusion(cfg, anchor="mcp", calibrations=None, primary_camera=0)

    world = _make_world_pts_straight_index()
    pose = HandPose(points=_make_face_on_image_pts(), world_points=world, handedness="Right")
    views = [CameraView(camera_id=0, poses=[pose])]

    try:
        fuser.fuse(views, timestamp=0.0)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "calibration" in str(e).lower(), (
            f"RuntimeError should mention calibration: {e}"
        )


def test_triangulation_fusion_raises_with_empty_calibrations():
    """TriangulationFusion.fuse() raises RuntimeError when calibrations dict is empty."""
    cfg = GestureConfig()
    fuser = TriangulationFusion(cfg, anchor="mcp", calibrations={}, primary_camera=0)

    pose = _neutral_hand()
    views = [CameraView(camera_id=0, poses=[pose])]

    try:
        fuser.fuse(views, timestamp=0.0)
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass


def test_triangulation_fusion_two_cameras_returns_fused_observation():
    """TriangulationFusion with two calibrated cameras returns FusedObservation.

    Image-space coords for each camera are synthesised by projecting a known 3D
    hand through each camera's projection matrix, so the two cameras see
    genuinely different pixels — avoiding the degenerate all-same-pixel case that
    causes X[3]=0 in the DLT solver.
    """
    cfg = GestureConfig()
    cal0 = _make_camera_calibration(camera_id=0, tx=0.0)
    cal1 = _make_camera_calibration(camera_id=1, tx=-0.15)

    hand3d = _make_synthetic_3d_hand(seed=99)
    img0 = _make_hand_image_pts_from_3d(cal0, hand3d)
    img1 = _make_hand_image_pts_from_3d(cal1, hand3d)

    pose0 = HandPose(points=img0, world_points=None,
                     handedness="Right", score=0.9, timestamp=0.0)
    pose1 = HandPose(points=img1, world_points=None,
                     handedness="Right", score=0.9, timestamp=0.0)

    fuser = TriangulationFusion(cfg, anchor="mcp",
                                 calibrations={0: cal0, 1: cal1},
                                 primary_camera=0)
    views = [
        CameraView(camera_id=0, poses=[pose0]),
        CameraView(camera_id=1, poses=[pose1]),
    ]
    results = fuser.fuse(views, timestamp=0.0)

    assert len(results) == 1, f"expected 1 FusedObservation, got {len(results)}"
    obs = results[0]
    assert isinstance(obs, FusedObservation)
    assert obs.pose is not None, "FusedObservation.pose should not be None for triangulation"
    assert obs.pose.world_points is not None, "FusedObservation.pose.world_points should not be None"
    assert obs.pose.world_points.shape == (21, 3), (
        f"world_points should be (21,3), got {obs.pose.world_points.shape}"
    )


def test_triangulation_fusion_features_not_none():
    """TriangulationFusion result has non-None features with valid fields."""
    cfg = GestureConfig()
    cal0 = _make_camera_calibration(camera_id=0, tx=0.0)
    cal1 = _make_camera_calibration(camera_id=1, tx=-0.15)

    hand3d = _make_synthetic_3d_hand(seed=42)
    img0 = _make_hand_image_pts_from_3d(cal0, hand3d)
    img1 = _make_hand_image_pts_from_3d(cal1, hand3d)

    pose0 = HandPose(points=img0, world_points=None,
                     handedness="Right", score=0.9, timestamp=0.0)
    pose1 = HandPose(points=img1, world_points=None,
                     handedness="Right", score=0.9, timestamp=0.0)

    fuser = TriangulationFusion(cfg, anchor="mcp",
                                 calibrations={0: cal0, 1: cal1},
                                 primary_camera=0)
    obs = fuser.fuse([
        CameraView(camera_id=0, poses=[pose0]),
        CameraView(camera_id=1, poses=[pose1]),
    ], timestamp=1.5)[0]

    assert obs.features is not None
    assert isinstance(obs.features.fingers_extended, dict)
    assert set(obs.features.fingers_extended.keys()) == set(FINGER_NAMES)
    assert obs.features.timestamp == 1.5


def test_triangulation_fusion_skips_group_with_only_one_calibrated_camera():
    """A hand seen by only one calibrated camera produces no output (needs >=2)."""
    cfg = GestureConfig()
    cal0 = _make_camera_calibration(camera_id=0, tx=0.0)
    cal_unused = _make_camera_calibration(camera_id=1, tx=-0.15)
    # Camera 1 is NOT in calibrations (not passed to TriangulationFusion).

    hand3d = _make_synthetic_3d_hand(seed=7)
    img0 = _make_hand_image_pts_from_3d(cal0, hand3d)
    img1 = _make_hand_image_pts_from_3d(cal_unused, hand3d)

    pose0 = HandPose(points=img0, world_points=None, handedness="Right", score=0.9)
    pose1 = HandPose(points=img1, world_points=None, handedness="Right", score=0.9)

    fuser = TriangulationFusion(cfg, anchor="mcp",
                                 calibrations={0: cal0},  # only cam 0 calibrated
                                 primary_camera=0)
    views = [
        CameraView(camera_id=0, poses=[pose0]),
        CameraView(camera_id=1, poses=[pose1]),   # cam 1 not in calibrations
    ]
    results = fuser.fuse(views, timestamp=0.0)
    # Only one calibrated camera -> can't triangulate -> no output.
    assert len(results) == 0, (
        f"Expected 0 results when only 1 calibrated camera, got {len(results)}"
    )


# ---------------------------------------------------------------------------
# 6. MAKE_FUSER
# ---------------------------------------------------------------------------

def test_make_fuser_feature_strategy():
    """make_fuser returns a FeatureFusion for strategy='feature'."""
    cfg_fusion = FusionConfig(strategy="feature")
    cfg_gesture = GestureConfig()
    fuser = make_fuser(cfg_fusion, cfg_gesture)
    assert isinstance(fuser, FeatureFusion), (
        f"Expected FeatureFusion, got {type(fuser).__name__}"
    )


def test_make_fuser_triangulation_strategy():
    """make_fuser returns a TriangulationFusion for strategy='triangulation'."""
    cfg_fusion = FusionConfig(strategy="triangulation")
    cfg_gesture = GestureConfig()
    fuser = make_fuser(cfg_fusion, cfg_gesture)
    assert isinstance(fuser, TriangulationFusion), (
        f"Expected TriangulationFusion, got {type(fuser).__name__}"
    )


def test_make_fuser_unknown_strategy_raises():
    """make_fuser raises ValueError for an unrecognised strategy name."""
    cfg_fusion = FusionConfig(strategy="neural_hologram")
    cfg_gesture = GestureConfig()
    try:
        make_fuser(cfg_fusion, cfg_gesture)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "neural_hologram" in str(e), (
            f"ValueError should name the bad strategy: {e}"
        )


def test_make_fuser_passes_primary_camera_to_feature_fuser():
    """make_fuser propagates primary_camera config into FeatureFusion."""
    cfg_fusion = FusionConfig(strategy="feature", primary_camera=2)
    fuser = make_fuser(cfg_fusion, GestureConfig())
    assert fuser.primary_camera == 2  # type: ignore[attr-defined]


def test_make_fuser_passes_primary_camera_to_triangulation_fuser():
    """make_fuser propagates primary_camera config into TriangulationFusion."""
    cfg_fusion = FusionConfig(strategy="triangulation", primary_camera=3)
    fuser = make_fuser(cfg_fusion, GestureConfig())
    assert fuser.primary_camera == 3  # type: ignore[attr-defined]


def test_make_fuser_passes_calibrations_to_triangulation_fuser():
    """make_fuser passes calibrations dict into TriangulationFusion."""
    cfg_fusion = FusionConfig(strategy="triangulation")
    cal = _make_camera_calibration(camera_id=0, tx=0.0)
    fuser = make_fuser(cfg_fusion, GestureConfig(), calibrations={0: cal})
    assert isinstance(fuser, TriangulationFusion)
    assert 0 in fuser.calibrations  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 7. CALIBRATION I/O
# ---------------------------------------------------------------------------

def test_calibration_io_roundtrip_preserves_values():
    """save_calibrations -> load_calibrations returns numerically equal K/R/t."""
    import tempfile

    from handcursor_core.fusion import load_calibrations, save_calibrations

    cal0 = _make_camera_calibration(camera_id=0, tx=0.0)
    cal1 = _make_camera_calibration(camera_id=1, tx=-0.15)
    # Bend cal1 into something less symmetric so we'd notice if axes were swapped.
    cal1.R = np.array([[0.999, -0.044, 0.0],
                       [0.044,  0.999, 0.0],
                       [0.0,    0.0,   1.0]], dtype=np.float64)
    cal1.t = np.array([-0.15, 0.02, 0.01], dtype=np.float64)

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        path = tmp.name
    save_calibrations({0: cal0, 1: cal1}, path)
    loaded = load_calibrations(path)

    assert set(loaded.keys()) == {0, 1}
    for cam_id, original in [(0, cal0), (1, cal1)]:
        got = loaded[cam_id]
        assert got.camera_id == cam_id
        assert got.image_size == original.image_size
        assert np.allclose(got.K, original.K, atol=1e-9), f"K mismatch for cam {cam_id}"
        assert np.allclose(got.R, original.R, atol=1e-9), f"R mismatch for cam {cam_id}"
        assert np.allclose(got.t, original.t, atol=1e-9), f"t mismatch for cam {cam_id}"


def test_calibration_io_load_missing_file_raises():
    """load_calibrations raises FileNotFoundError for a missing path."""
    from handcursor_core.fusion import load_calibrations

    try:
        load_calibrations("/tmp/__definitely_not_a_real_calibration_file__.yaml")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_calibration_io_loaded_pair_triangulates_correctly():
    """A round-tripped calibration pair recovers a known 3D point via DLT."""
    import tempfile

    from handcursor_core.fusion import load_calibrations, save_calibrations

    cal0 = _make_camera_calibration(camera_id=0, tx=0.0)
    cal1 = _make_camera_calibration(camera_id=1, tx=-0.15)

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        path = tmp.name
    save_calibrations({0: cal0, 1: cal1}, path)
    loaded = load_calibrations(path)

    pt3d = np.array([0.05, 0.1, 0.8], dtype=np.float64)
    px0 = _project_3d_to_pixel(loaded[0], pt3d)
    px1 = _project_3d_to_pixel(loaded[1], pt3d)
    recovered = triangulate_dlt([(loaded[0].projection, px0),
                                 (loaded[1].projection, px1)])
    error = np.linalg.norm(recovered.astype(np.float64) - pt3d)
    assert error < 1e-4, f"reconstruction after round-trip too large: {error:.2e}"


# ---------------------------------------------------------------------------
# 8. TRIANGULATION TIME-SKEW REJECTION
# ---------------------------------------------------------------------------

def test_triangulation_drops_stale_view_via_max_view_skew():
    """A view whose timestamp lags too far is excluded; with only 1 view left, no output."""
    cfg = GestureConfig()
    cal0 = _make_camera_calibration(camera_id=0, tx=0.0)
    cal1 = _make_camera_calibration(camera_id=1, tx=-0.15)

    hand3d = _make_synthetic_3d_hand(seed=11)
    img0 = _make_hand_image_pts_from_3d(cal0, hand3d)
    img1 = _make_hand_image_pts_from_3d(cal1, hand3d)

    # Camera 1 is 0.5 s behind camera 0; max_view_skew_s=0.05 rejects it.
    pose0 = HandPose(points=img0, world_points=None,
                     handedness="Right", score=0.9, timestamp=1.00)
    pose1 = HandPose(points=img1, world_points=None,
                     handedness="Right", score=0.9, timestamp=0.50)

    fuser = TriangulationFusion(cfg, anchor="mcp",
                                calibrations={0: cal0, 1: cal1},
                                primary_camera=0, max_view_skew_s=0.05)
    results = fuser.fuse([
        CameraView(camera_id=0, poses=[pose0], timestamp=1.00),
        CameraView(camera_id=1, poses=[pose1], timestamp=0.50),
    ], timestamp=1.00)
    # Only cam 0 is fresh enough; <2 calibrated views remain -> no output.
    assert len(results) == 0, (
        f"Expected stale view rejection to leave <2 cameras and emit nothing, "
        f"got {len(results)} result(s)"
    )


def test_triangulation_keeps_views_within_skew_tolerance():
    """Views within max_view_skew_s of each other are both kept; reconstruction proceeds."""
    cfg = GestureConfig()
    cal0 = _make_camera_calibration(camera_id=0, tx=0.0)
    cal1 = _make_camera_calibration(camera_id=1, tx=-0.15)

    hand3d = _make_synthetic_3d_hand(seed=12)
    img0 = _make_hand_image_pts_from_3d(cal0, hand3d)
    img1 = _make_hand_image_pts_from_3d(cal1, hand3d)

    # 10 ms gap, well within 50 ms tolerance.
    pose0 = HandPose(points=img0, world_points=None,
                     handedness="Right", score=0.9, timestamp=1.00)
    pose1 = HandPose(points=img1, world_points=None,
                     handedness="Right", score=0.9, timestamp=0.99)

    fuser = TriangulationFusion(cfg, anchor="mcp",
                                calibrations={0: cal0, 1: cal1},
                                primary_camera=0, max_view_skew_s=0.05)
    results = fuser.fuse([
        CameraView(camera_id=0, poses=[pose0], timestamp=1.00),
        CameraView(camera_id=1, poses=[pose1], timestamp=0.99),
    ], timestamp=1.00)
    assert len(results) == 1, "expected one reconstruction within skew tolerance"
    assert results[0].pose is not None and results[0].pose.world_points is not None


def test_triangulation_skew_zero_disables_gate():
    """max_view_skew_s=0 disables the gate even when timestamps differ wildly."""
    cfg = GestureConfig()
    cal0 = _make_camera_calibration(camera_id=0, tx=0.0)
    cal1 = _make_camera_calibration(camera_id=1, tx=-0.15)

    hand3d = _make_synthetic_3d_hand(seed=13)
    img0 = _make_hand_image_pts_from_3d(cal0, hand3d)
    img1 = _make_hand_image_pts_from_3d(cal1, hand3d)

    # Huge time delta — but the gate is off.
    pose0 = HandPose(points=img0, handedness="Right", score=0.9, timestamp=10.0)
    pose1 = HandPose(points=img1, handedness="Right", score=0.9, timestamp=0.0)

    fuser = TriangulationFusion(cfg, anchor="mcp",
                                calibrations={0: cal0, 1: cal1},
                                primary_camera=0, max_view_skew_s=0.0)
    results = fuser.fuse([
        CameraView(camera_id=0, poses=[pose0]),
        CameraView(camera_id=1, poses=[pose1]),
    ], timestamp=10.0)
    assert len(results) == 1, "max_view_skew_s=0 must let both views through"


def test_make_fuser_passes_max_view_skew_to_triangulation_fuser():
    """make_fuser propagates max_view_skew_s into TriangulationFusion."""
    cfg_fusion = FusionConfig(strategy="triangulation", max_view_skew_s=0.123)
    fuser = make_fuser(cfg_fusion, GestureConfig())
    assert fuser.max_view_skew_s == 0.123  # type: ignore[attr-defined]
