"""TOY STAND-IN for the OS cursor controller.

The full app's ``CursorController`` (private) injects real mouse events via
pynput, supports a relative "clutch" mode, and snaps onto UI targets through the
macOS Accessibility API. None of that is here.

This sink keeps only the part that belongs to the core pipeline — One Euro
smoothing followed by the multi-monitor mapping — and *records* button events
instead of sending them, so the demo and tests can inspect exactly what would
have reached the OS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SmoothingConfig
from ..gestures.states import ActionType, CursorAction
from ..monitors.layout import MonitorLayout
from ..smoothing import Vector2Filter


@dataclass
class ActiveRegion:
    """The sub-rectangle of the camera frame mapped onto the whole desktop."""
    x_min: float = 0.10
    x_max: float = 0.90
    y_min: float = 0.10
    y_max: float = 0.90


@dataclass
class RecordingCursor:
    layout: MonitorLayout
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    region: ActiveRegion = field(default_factory=ActiveRegion)
    button_down: bool = False
    position: tuple[int, int] | None = None
    events: list[tuple[float, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        s = self.smoothing
        self._filter = Vector2Filter(s.min_cutoff, s.beta, s.d_cutoff)

    def apply(self, action: CursorAction, now: float) -> None:
        if action.type == ActionType.MOVE and action.position is not None:
            sx, sy = self._filter(action.position, now)
            r = self.region
            self.position = self.layout.active_region_to_screen(
                sx, sy, r.x_min, r.x_max, r.y_min, r.y_max)
        elif action.type == ActionType.PRESS:
            self.button_down = True
            self.events.append((now, "press"))
        elif action.type == ActionType.RELEASE:
            self.button_down = False
            self.events.append((now, "release"))
        elif action.type == ActionType.RECLICK:
            self.events.append((now, "reclick"))
