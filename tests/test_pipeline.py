"""End-to-end checks of the extracted pipeline on synthetic landmarks.

These exercise the toy tracker (``tracking/synthetic.py``) and recording cursor
(``cursor/sink.py``) together with the real recognizer, state machine, filter
and monitor mapping — i.e. what ``python -m handcursor_core.demo`` prints.
"""

from __future__ import annotations

import numpy as np
import pytest

from handcursor_core import demo
from handcursor_core.config import GestureConfig
from handcursor_core.cursor.sink import RecordingCursor
from handcursor_core.gestures.recognizer import GestureRecognizer
from handcursor_core.gestures.states import ActionType, CursorAction
from handcursor_core.monitors.layout import Monitor, MonitorLayout
from handcursor_core.tracking.synthetic import synthetic_pose


def _features(posture: str):
    pose = synthetic_pose(posture, (0.5, 0.7), 0.0)
    return GestureRecognizer(GestureConfig(), anchor="palm").recognize(pose)


def test_synthetic_open_hand_reads_as_open_palm():
    f = _features("open")
    assert f.is_open_palm and not f.is_fist
    assert f.pinch_distance > GestureConfig().pinch_release


def test_synthetic_pinch_passes_every_pinch_gate():
    cfg = GestureConfig()
    f = _features("pinch")
    assert f.pinch_distance < cfg.pinch_engage
    assert f.index_reach >= cfg.pinch_index_reach_min
    assert all(f.fingers_extended[n] for n in ("middle", "ring", "pinky"))


def test_synthetic_fist_is_not_mistaken_for_a_pinch():
    cfg = GestureConfig()
    f = _features("fist")
    assert f.is_fist
    # Thumb<->index is close inside a fist, but the index is curled.
    assert f.index_reach < cfg.pinch_index_reach_min


def test_unknown_posture_raises():
    with pytest.raises(ValueError):
        synthetic_pose("wave", (0.5, 0.5), 0.0)


def test_recording_cursor_maps_onto_the_virtual_desktop():
    layout = MonitorLayout([Monitor(0, 0, 1920, 1080, is_primary=True),
                            Monitor(1920, 0, 1920, 1080)])
    cursor = RecordingCursor(layout)
    cursor.apply(CursorAction(ActionType.MOVE, position=np.array([0.5, 0.5])), 0.0)
    assert cursor.position == (1920, 540)
    cursor.apply(CursorAction(ActionType.PRESS), 0.1)
    cursor.apply(CursorAction(ActionType.RELEASE), 0.2)
    assert [name for _, name in cursor.events] == ["press", "release"]
    assert cursor.button_down is False


def test_demo_script_presses_drags_across_monitors_and_releases_on_fist(capsys):
    cursor = demo.run_gesture_demo()
    names = [name for _, name in cursor.events]
    assert names[0] == "press" and names[-1] == "release"
    assert not cursor.button_down
    # The drag ends on the second display (x >= 1920).
    assert cursor.position is not None and cursor.position[0] >= 1920
    out = capsys.readouterr().out
    assert "DRAGGING" in out and "DISABLED" in out


def test_demo_triangulation_recovers_point():
    assert demo.run_triangulation_demo() < 1e-4
