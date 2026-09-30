"""Persist :class:`CameraCalibration` dicts to / from YAML on disk.

In the full app a checkerboard calibration script (private) writes one file
(default ``calibration.yaml``); the app loads it when
``fusion.strategy = triangulation`` and feeds the dict into
:class:`TriangulationFusion`. Camera 0 is the world origin by convention
(``R = I``, ``t = 0``); every other camera's pose is expressed in that frame.

Format::

    cameras:
      - camera_id: 0
        image_size: [1280, 720]
        K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        R: [[1,0,0],[0,1,0],[0,0,1]]
        t: [0.0, 0.0, 0.0]
      - camera_id: 1
        ...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .triangulation import CameraCalibration


def save_calibrations(calibrations: dict[int, CameraCalibration],
                      path: str | Path) -> None:
    """Write ``calibrations`` to ``path`` as YAML. Overwrites if it exists."""
    cams = []
    for cam_id in sorted(calibrations):
        cal = calibrations[cam_id]
        cams.append({
            "camera_id": int(cam_id),
            "image_size": [int(cal.image_size[0]), int(cal.image_size[1])],
            "K": cal.K.astype(float).tolist(),
            "R": cal.R.astype(float).tolist(),
            "t": cal.t.astype(float).reshape(3).tolist(),
        })
    Path(path).write_text(yaml.safe_dump({"cameras": cams}, sort_keys=False),
                          encoding="utf-8")


def load_calibrations(path: str | Path) -> dict[int, CameraCalibration]:
    """Load calibrations from ``path``. Raises FileNotFoundError / ValueError on a bad file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = data.get("cameras")
    if not raw:
        raise ValueError(f"Calibration file {p} has no 'cameras' list")
    out: dict[int, CameraCalibration] = {}
    for entry in raw:
        cam_id = int(entry["camera_id"])
        K = np.asarray(entry["K"], dtype=np.float64)
        R = np.asarray(entry["R"], dtype=np.float64)
        t = np.asarray(entry["t"], dtype=np.float64).reshape(3)
        if K.shape != (3, 3) or R.shape != (3, 3):
            raise ValueError(f"camera {cam_id}: K and R must be 3x3")
        w, h = entry["image_size"]
        out[cam_id] = CameraCalibration(camera_id=cam_id, K=K, R=R, t=t,
                                        image_size=(int(w), int(h)))
    return out
