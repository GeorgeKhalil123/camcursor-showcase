"""Typed configuration for the extracted core.

The field names and semantics match the full app's config, but the DEFAULT
VALUES here are generic, round placeholders chosen so the tests and the
synthetic demo behave sensibly. They are not the thresholds tuned against real
MediaPipe output (those live in the private repo's ``config.yaml``).

Only the sections the extracted modules consume are included: gestures, the
two-palm system toggle, pointer smoothing, and multi-camera fusion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GestureConfig:
    # Finger "extended" test uses a straightness ratio (chord / summed segment
    # lengths): ~1.0 when straight, drops when curled. Hysteresis (two
    # thresholds) stops chatter: a finger must exceed _on to become extended and
    # fall below _off to retract.
    finger_extend_on: float = 0.80
    finger_extend_off: float = 0.60
    # Second extension signal: tip-to-knuckle distance / hand size. Catches
    # extended-but-bent or camera-angled fingers that straightness misses.
    finger_reach_on: float = 0.90
    finger_reach_off: float = 0.70

    # Below this palm "view quality" (how face-on the hand is) the view is too
    # oblique to trust, so the cursor freezes instead of misfiring. 0 disables
    # the gate entirely.
    min_view_quality: float = 0.05

    # Pinch presses immediately: a quick release is a normal click, holding +
    # moving is a drag. Holding the pinch STILL (within dwell_radius) for
    # dwell_time_s re-clicks once — a reliability fallback for a missed tap.
    # dwell_radius is also the movement threshold that splits "still" from
    # "dragging".
    dwell_enabled: bool = True
    dwell_time_s: float = 0.5
    dwell_radius: float = 0.05

    # Pinch thresholds are the thumb-tip<->index-tip distance normalised by
    # hand size. Two thresholds give hysteresis so the click doesn't chatter.
    pinch_engage: float = 0.40
    pinch_release: float = 0.60
    min_pinch_frames: int = 2        # debounce: frames below engage before firing
    # A pinch only counts if the index is REACHING toward the thumb (tip-to-
    # knuckle reach / hand size). Separates a deliberate pinch from a closed
    # fist, which can also show a small thumb<->index distance.
    pinch_index_reach_min: float = 0.55
    click_cooldown_s: float = 0.3    # min time between two click-downs
    freeze_on_pinch: bool = False    # freeze cursor while pinching to avoid drift
    # Temporal stabilization: a fist must persist this many consecutive frames
    # before disabling, so a 1-frame misread doesn't freeze the cursor.
    min_fist_frames: int = 3


@dataclass
class SmoothingConfig:
    """One Euro filter knobs (see :mod:`handcursor_core.smoothing`)."""
    min_cutoff: float = 1.0   # lower = smoother but laggier at low speed
    beta: float = 0.05        # higher = less lag at high speed
    d_cutoff: float = 1.0


@dataclass
class FusionConfig:
    """Multi-camera fusion. With one camera every strategy is a passthrough."""
    # "feature" = calibration-free (blend each camera's gesture features).
    # "triangulation" = reconstruct one metric 3D hand (needs calibration).
    strategy: str = "feature"
    primary_camera: int = 0       # logical camera id that supplies cursor position
    min_view_weight: float = 0.05  # floor on a camera's fusion weight (so it still nudges)
    vote_threshold: float = 0.5    # fraction of view-weight that must agree a finger is out
    # Triangulation only: reject a camera view from the DLT solve if its
    # timestamp is more than this many seconds behind the newest view in the
    # same hand-group. DLT assumes one instant. 0 disables the gate.
    max_view_skew_s: float = 0.05


@dataclass
class SystemConfig:
    """The two-open-palms master switch for the whole gesture system."""
    enabled_on_start: bool = True
    toggle_min_frames: int = 5       # consecutive two-palm frames to fire a toggle
    toggle_cooldown_s: float = 1.0   # secondary guard against rapid re-toggling
