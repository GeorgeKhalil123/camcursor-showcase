"""Monitor enumeration and coordinate mapping.

MVP mapping (planar): the camera's *active region* maps linearly onto the whole
virtual desktop bounding box, so the cursor traverses every monitor naturally.

The key extension seam for the future is :meth:`active_region_to_screen`. Today
it's a 2D linear remap. With 2-3 cameras you'll instead resolve a metric 3D
fingertip position and ask "which monitor's plane does the pointing ray hit?".
Because every caller goes through this one method, that upgrade is local — the
state machine, gestures, and smoothing don't change.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..geometry import clamp, remap


@dataclass
class Monitor:
    x: int
    y: int
    width: int
    height: int
    is_primary: bool = False


class MonitorLayout:
    def __init__(self, monitors: list[Monitor]) -> None:
        if not monitors:
            raise ValueError("No monitors detected.")
        self.monitors = monitors
        # Virtual-desktop bounding box across all displays (global pixel coords).
        self.min_x = min(m.x for m in monitors)
        self.min_y = min(m.y for m in monitors)
        self.max_x = max(m.x + m.width for m in monitors)
        self.max_y = max(m.y + m.height for m in monitors)

    @property
    def virtual_width(self) -> int:
        return self.max_x - self.min_x

    @property
    def virtual_height(self) -> int:
        return self.max_y - self.min_y

    @classmethod
    def detect(cls) -> "MonitorLayout":
        """Enumerate physical displays.

        macOS: use Quartz directly. ``screeninfo`` reports y-coords for
        displays positioned above the primary as positive offsets below
        instead of negative offsets above, which breaks the virtual-desktop
        bounding box (top of an external above the MacBook gets clamped at
        the MacBook's top instead of the external's actual top).
        Other platforms: fall back to ``screeninfo``.
        """
        import sys
        if sys.platform == "darwin":
            try:
                import Quartz
            except ImportError:
                pass
            else:
                _, active, _ = Quartz.CGGetActiveDisplayList(16, None, None)
                main_id = Quartz.CGMainDisplayID()
                mons = []
                for did in active:
                    b = Quartz.CGDisplayBounds(did)
                    mons.append(Monitor(
                        int(b.origin.x), int(b.origin.y),
                        int(b.size.width), int(b.size.height),
                        is_primary=(did == main_id),
                    ))
                return cls(mons)

        try:
            from screeninfo import get_monitors
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("screeninfo not installed: pip install screeninfo") from exc
        mons = [
            Monitor(m.x, m.y, m.width, m.height, bool(getattr(m, "is_primary", False)))
            for m in get_monitors()
        ]
        return cls(mons)

    def active_region_to_screen(
        self, nx: float, ny: float,
        rx_min: float, rx_max: float, ry_min: float, ry_max: float,
    ) -> tuple[int, int]:
        """Map a normalised fingertip position to a global pixel coordinate.

        ``(nx, ny)`` are in [0,1] frame coordinates. We first clamp to the
        configured active region, then remap that region onto the full virtual
        desktop. Returns clamped global pixel coords ready for pynput.
        """
        nx = clamp(nx, rx_min, rx_max)
        ny = clamp(ny, ry_min, ry_max)
        screen_x = remap(nx, rx_min, rx_max, self.min_x, self.max_x)
        screen_y = remap(ny, ry_min, ry_max, self.min_y, self.max_y)
        screen_x = clamp(screen_x, self.min_x, self.max_x - 1)
        screen_y = clamp(screen_y, self.min_y, self.max_y - 1)
        return int(round(screen_x)), int(round(screen_y))
