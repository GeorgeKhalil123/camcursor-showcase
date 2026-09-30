"""Hardware-free walkthrough of the core pipeline.

    python -m handcursor_core.demo

Part 1 replays a scripted, synthetic landmark stream (open hand -> pinch ->
drag -> fist) through the real recognizer, state machine, One Euro filter and
multi-monitor mapping, printing each state transition and the smoothed cursor.

Part 2 builds two synthetic calibrated cameras, projects a known 3D point and a
full 21-landmark hand into both, and recovers them with DLT triangulation.

The landmark source is a toy stand-in for MediaPipe and the cursor sink records
events instead of injecting them; see ``tracking/synthetic.py`` and
``cursor/sink.py``.
"""

from __future__ import annotations

import numpy as np

from .config import GestureConfig
from .cursor.sink import RecordingCursor
from .fusion import CameraCalibration, CameraView, TriangulationFusion, triangulate_dlt
from .gestures.recognizer import GestureRecognizer
from .gestures.state_machine import GestureStateMachine
from .monitors.layout import Monitor, MonitorLayout
from .tracking.synthetic import skeleton, synthetic_pose
from .tracking.types import HandPose

FPS = 30.0

# (posture, frames, wrist x at start, wrist x at end) — y stays fixed.
SCRIPT: list[tuple[str, int, float, float]] = [
    ("open", 20, 0.30, 0.45),    # move right with an open palm
    ("pinch", 20, 0.45, 0.45),   # pinch -> press; hold still -> one dwell re-click
    ("pinch", 20, 0.45, 0.70),   # keep pinching and move -> drag
    ("fist", 8, 0.70, 0.70),     # close into a fist -> release, then freeze
]


def _frames(jitter: float = 0.002, seed: int = 7):
    rng = np.random.default_rng(seed)
    i = 0
    for posture, n, x0, x1 in SCRIPT:
        for k in range(n):
            x = x0 + (x1 - x0) * (k / max(n - 1, 1))
            t = i / FPS
            yield t, posture, synthetic_pose(posture, (x, 0.75), t, jitter=jitter, rng=rng)
            i += 1


def run_gesture_demo() -> RecordingCursor:
    cfg = GestureConfig()
    recognizer = GestureRecognizer(cfg, anchor="palm")
    fsm = GestureStateMachine(cfg)
    # Two 1920x1080 displays side by side: one 3840-wide virtual desktop.
    layout = MonitorLayout([Monitor(0, 0, 1920, 1080, is_primary=True),
                            Monitor(1920, 0, 1920, 1080)])
    cursor = RecordingCursor(layout)

    print("== Gesture pipeline (synthetic landmarks, 30 fps) ==")
    print(f"{'t(s)':>6}  {'posture':<6}  {'state':<12} {'action':<8} "
          f"{'raw pointer':<16} cursor(px)")
    last_state = None
    for i, (t, posture, pose) in enumerate(_frames()):
        features = recognizer.recognize(pose)
        action = fsm.update(features, now=t)
        cursor.apply(action, now=t)
        changed = fsm.state != last_state or action.type.name in ("PRESS", "RELEASE", "RECLICK")
        if changed or i % 6 == 0:
            raw = f"({features.pointer[0]:.3f}, {features.pointer[1]:.3f})"
            mark = "*" if changed else " "
            print(f"{t:6.3f}{mark} {posture:<6}  {fsm.state.name:<12} "
                  f"{action.type.name:<8} {raw:<16} {cursor.position}")
        last_state = fsm.state
    print("button events:", ", ".join(f"{name}@{t:.3f}s" for t, name in cursor.events))
    print("(* = state change or button event)\n")
    return cursor


def _camera(camera_id: int, tx: float) -> CameraCalibration:
    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    return CameraCalibration(camera_id=camera_id, K=K, R=np.eye(3),
                             t=np.array([tx, 0.0, 0.0]), image_size=(1280, 720))


def _project(cal: CameraCalibration, X: np.ndarray) -> np.ndarray:
    x = cal.projection @ np.append(X, 1.0)
    return x[:2] / x[2]


def run_triangulation_demo() -> float:
    print("== DLT triangulation (two synthetic cameras, 15 cm baseline) ==")
    cams = {0: _camera(0, 0.0), 1: _camera(1, -0.15)}

    X = np.array([0.05, 0.10, 0.80])
    rays = [(c.projection, _project(c, X)) for c in cams.values()]
    for cid, (_, px) in zip(cams, rays):
        print(f"camera {cid} sees pixel ({px[0]:.2f}, {px[1]:.2f})")
    X_hat = triangulate_dlt(rays).astype(np.float64)
    err = float(np.linalg.norm(X_hat - X))
    print(f"true point  {X}\nrecovered   {np.round(X_hat, 6)}  (error {err:.2e} m)")

    # A whole hand: 21 landmarks placed 0.8 m in front of the rig.
    hand = skeleton("open") + np.array([0.0, -0.05, 0.8])
    views = []
    for cid, cal in cams.items():
        w, h = cal.image_size
        img = np.zeros((21, 3), dtype=np.float32)
        for lm, p in enumerate(hand):
            px = _project(cal, p)
            img[lm, :2] = [px[0] / w, px[1] / h]   # MediaPipe-style normalised coords
        views.append(CameraView(cid, [HandPose(points=img, handedness="Right")]))
    obs = TriangulationFusion(GestureConfig(), calibrations=cams).fuse(views, timestamp=0.0)[0]
    hand_err = np.linalg.norm(obs.pose.world_points.astype(np.float64) - hand, axis=1)
    print(f"21-landmark hand: mean error {hand_err.mean():.2e} m, "
          f"max {hand_err.max():.2e} m, open palm = {obs.features.is_open_palm}")
    return err


def main() -> None:
    run_gesture_demo()
    run_triangulation_demo()


if __name__ == "__main__":
    main()
