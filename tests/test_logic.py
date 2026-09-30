"""Hardware-free tests for the pure logic (no camera, no MediaPipe).

Ported from the full project's ``tests/test_logic.py``. These cover the parts
most likely to harbour subtle bugs: the pinch hysteresis/cooldown state
machine, the smoothing filter, and coordinate mapping across a multi-monitor
virtual desktop. Tests of the private cursor controller (relative clutch mode,
magnetic snap-to-target) and the OpenCV multi-camera source are not included.
"""

from __future__ import annotations

import numpy as np

from handcursor_core.config import GestureConfig, SystemConfig
from handcursor_core.gestures.recognizer import GestureRecognizer
from handcursor_core.gestures.state_machine import GestureStateMachine
from handcursor_core.gestures.states import ActionType, GestureState
from handcursor_core.gestures.system_control import SystemToggle
from handcursor_core.monitors.layout import Monitor, MonitorLayout
from handcursor_core.smoothing import OneEuroFilter
from handcursor_core.tracking.types import GestureFeatures, HandPose, Landmark


def _features(pinch: float, fist: bool = False, point: bool = False,
              ts: float = 0.0, view: float = 1.0, pos=(0.5, 0.5),
              index_reach: float | None = None,
              support_up: bool | None = None) -> GestureFeatures:
    # Default posture is an OPEN HAND — the move trigger. ``fist`` (all curled)
    # and ``point`` (index only, the reserved bare-point posture) override it.
    # ``index_reach`` gates pinch: defaults to reaching (1.0) unless a fist, where
    # the index is curled (0.3, below pinch_index_reach_min) — override to model
    # the in-between "OK" sign.
    # ``support_up`` overrides middle/ring/pinky extension only (independent of
    # the thumb/index posture above). True = those three are extended, False =
    # curled, None = follow the base posture. Used to model a real "OK" sign
    # (thumb+index touching, other three up) vs. a true fist (all curled).
    if fist:
        ext = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
    elif point:
        ext = {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False}
    else:
        ext = {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}
    if support_up is not None:
        for n in ("middle", "ring", "pinky"):
            ext[n] = support_up
    if index_reach is None:
        index_reach = 0.3 if fist else 1.0
    p = np.array(pos, dtype=float)
    return GestureFeatures(fingers_extended=ext, pinch_distance=pinch, index_reach=index_reach,
                           index_tip=p, index_mcp=p, pointer=p, view_quality=view, timestamp=ts)


def _palm(ts: float = 0.0) -> GestureFeatures:
    ext = {n: True for n in ("thumb", "index", "middle", "ring", "pinky")}
    return GestureFeatures(fingers_extended=ext, pinch_distance=1.0,
                           index_tip=np.array([0.5, 0.5]), timestamp=ts)


def test_open_hand_emits_move():
    sm = GestureStateMachine(GestureConfig())
    action = sm.update(_features(pinch=0.9), now=0.0)   # default posture = open hand
    assert sm.state == GestureState.MOVING
    assert action.type == ActionType.MOVE


def test_bare_point_holds_and_does_not_move():
    # A bare index point is reserved (future precision mode): it must NOT move
    # the cursor, just hold position.
    sm = GestureStateMachine(GestureConfig())
    action = sm.update(_features(pinch=0.9, point=True), now=0.0)
    assert sm.state == GestureState.IDLE
    assert action.type == ActionType.NONE


def test_fist_disables_and_freezes():
    cfg = GestureConfig(min_fist_frames=2)
    sm = GestureStateMachine(cfg)
    sm.update(_features(pinch=0.9, fist=True), now=0.0)
    action = sm.update(_features(pinch=0.9, fist=True), now=0.1)  # 2nd frame confirms
    assert sm.state == GestureState.DISABLED
    assert action.type == ActionType.NONE  # no MOVE -> cursor frozen


def test_pinch_with_reaching_index_beats_fist():
    # The "OK" sign: thumb+index touching, index reaching (reach above the floor),
    # AND the other three fingers extended (which is what an OK sign actually
    # looks like). That's a pinch, and it takes precedence over disabling.
    cfg = GestureConfig(min_pinch_frames=1, pinch_index_reach_min=0.6)
    sm = GestureStateMachine(cfg)
    a = sm.update(_features(pinch=0.2, fist=True, index_reach=0.7,
                            support_up=True), now=0.0)
    assert a.type == ActionType.PRESS
    assert sm.state == GestureState.PINCH_CLICK


def test_close_thumb_with_curled_supporting_fingers_does_not_pinch():
    # The user's heuristic: a real pinch keeps middle/ring/pinky extended (since
    # move requires an open hand). If all three are curled, it's a fist — and
    # even a close thumb<->index distance with a reaching index must NOT fire a
    # pinch. This is what stops triangulation-noise spikes from being misread as
    # clicks during a fist.
    cfg = GestureConfig(min_pinch_frames=1, min_fist_frames=1,
                        pinch_index_reach_min=0.6)
    sm = GestureStateMachine(cfg)
    a = sm.update(_features(pinch=0.2, fist=True, index_reach=0.7,
                            support_up=False), now=0.0)
    assert a.type == ActionType.NONE
    assert sm.state == GestureState.DISABLED
    assert sm._button_down is False


def test_fully_curled_fist_does_not_pinch():
    # A true fist (index reach below the floor) never pinches, even if the
    # thumb<->index distance reads small. It disables instead.
    cfg = GestureConfig(min_pinch_frames=1, min_fist_frames=1, pinch_index_reach_min=0.6)
    sm = GestureStateMachine(cfg)
    a = sm.update(_features(pinch=0.2, fist=True, index_reach=0.4), now=0.0)
    assert a.type == ActionType.NONE
    assert sm.state == GestureState.DISABLED
    assert sm._button_down is False


def test_held_pinch_releases_when_supporting_fingers_close_into_fist():
    # User's symptom: pinch fires on the way to a fist, then the fist forms and
    # thumb+index stay close inside it — sticky-pinch held the button down even
    # though the posture is now a fist. Closing supporting fingers for
    # min_fist_frames consecutive frames must release the pinch.
    cfg = GestureConfig(min_pinch_frames=1, min_fist_frames=2,
                        pinch_index_reach_min=0.6, pinch_release=0.60)
    sm = GestureStateMachine(cfg)
    # Frame 1: OK-sign-like pinch fires (supporting up).
    a = sm.update(_features(pinch=0.2, support_up=True), now=0.0)
    assert a.type == ActionType.PRESS
    assert sm._button_down is True
    # Frame 2: user closes into a fist — thumb+index stay close (still below
    # release), but supporting fingers are now all curled. One frame isn't
    # enough (min_fist_frames=2).
    a = sm.update(_features(pinch=0.2, support_up=False), now=0.05)
    assert a.type != ActionType.RELEASE
    assert sm._button_down is True
    # Frame 3: still in the fist posture — counter trips, release fires.
    a = sm.update(_features(pinch=0.2, support_up=False), now=0.10)
    assert a.type == ActionType.RELEASE
    assert sm._button_down is False


def test_held_pinch_survives_one_frame_supporting_misread():
    # Anti-flicker: a single frame where supporting fingers read as curled
    # must NOT release a held pinch — otherwise MediaPipe noise breaks drags.
    cfg = GestureConfig(min_pinch_frames=1, min_fist_frames=3,
                        pinch_index_reach_min=0.6, pinch_release=0.60)
    sm = GestureStateMachine(cfg)
    sm.update(_features(pinch=0.2, support_up=True), now=0.0)
    a = sm.update(_features(pinch=0.2, support_up=False), now=0.05)  # 1-frame blip
    assert a.type != ActionType.RELEASE
    a = sm.update(_features(pinch=0.2, support_up=True), now=0.10)   # recovers
    assert sm._button_down is True
    assert sm._no_support_frames == 0  # counter reset on recovery


def test_transient_fist_does_not_disable():
    # #3: a brief (sub-threshold) fist must NOT flip to DISABLED.
    cfg = GestureConfig(min_fist_frames=3)
    sm = GestureStateMachine(cfg)
    sm.update(_features(pinch=0.9, fist=True), now=0.0)
    sm.update(_features(pinch=0.9, fist=True), now=0.1)
    assert sm.state != GestureState.DISABLED          # only 2 frames < threshold
    sm.update(_features(pinch=0.9, fist=True), now=0.2)
    assert sm.state == GestureState.DISABLED            # 3rd consecutive confirms


def test_quick_pinch_presses_then_releases():
    # Pinch presses immediately; a quick release = down-up = a normal OS click.
    cfg = GestureConfig(min_pinch_frames=1, dwell_radius=0.04)
    sm = GestureStateMachine(cfg)
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.0).type == ActionType.PRESS
    assert sm.update(_features(pinch=0.9, pos=(0.5, 0.5)), 0.05).type == ActionType.RELEASE
    assert sm.state == GestureState.MOVING


def test_pinch_move_drags():
    cfg = GestureConfig(min_pinch_frames=1, dwell_radius=0.04, dwell_time_s=0.4)
    sm = GestureStateMachine(cfg)
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.0).type == ActionType.PRESS
    assert sm.update(_features(pinch=0.2, pos=(0.8, 0.5)), 0.1).type == ActionType.MOVE
    assert sm.state == GestureState.DRAGGING
    assert sm.update(_features(pinch=0.9, pos=(0.8, 0.5)), 0.2).type == ActionType.RELEASE
    assert sm.state == GestureState.MOVING


def test_hold_still_reclicks_once_then_resumes_drag():
    cfg = GestureConfig(min_pinch_frames=1, dwell_time_s=0.3, dwell_radius=0.04,
                        click_cooldown_s=0.0)
    sm = GestureStateMachine(cfg)
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.0).type == ActionType.PRESS
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.1).type == ActionType.MOVE    # holding
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.35).type == ActionType.RECLICK
    # Keep holding still -> NO machine-gun: only one re-click per still episode.
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.7).type == ActionType.MOVE
    # Move -> drag (this also re-arms the re-click).
    assert sm.update(_features(pinch=0.2, pos=(0.8, 0.5)), 0.75).type == ActionType.MOVE
    assert sm.state == GestureState.DRAGGING
    # Hold still again at the new spot -> re-click fires once more.
    assert sm.update(_features(pinch=0.2, pos=(0.8, 0.5)), 1.1).type == ActionType.RECLICK


def test_toggle_dwell_flips_reclick_at_runtime():
    # The hold-still re-click can be turned off mid-session via toggle_dwell().
    cfg = GestureConfig(min_pinch_frames=1, dwell_time_s=0.1, dwell_radius=0.04,
                        click_cooldown_s=0.0)
    sm = GestureStateMachine(cfg)
    assert sm.dwell_enabled is True
    # With dwell on, holding still re-clicks.
    sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.0)            # PRESS
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.2).type == ActionType.RECLICK
    # Toggle off -> a fresh hold-still no longer re-clicks (just holds/moves).
    assert sm.toggle_dwell() is False
    sm2 = GestureStateMachine(cfg)
    sm2.dwell_enabled = False
    sm2.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.0)           # PRESS
    assert sm2.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.5).type == ActionType.MOVE


def test_dwell_disabled_pinch_has_no_reclick():
    cfg = GestureConfig(min_pinch_frames=1, dwell_enabled=False, dwell_radius=0.04)
    sm = GestureStateMachine(cfg)
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.0).type == ActionType.PRESS
    # Holding still never re-clicks when disabled (just keeps the button held).
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 2.0).type == ActionType.MOVE
    # Moving still drags.
    assert sm.update(_features(pinch=0.2, pos=(0.8, 0.5)), 2.1).type == ActionType.MOVE
    assert sm.state == GestureState.DRAGGING


def test_sticky_pinch_survives_transient_fist():
    # A one-frame fist misread mid-drag must not release/disable.
    cfg = GestureConfig(min_pinch_frames=1, min_fist_frames=3, dwell_radius=0.04)
    sm = GestureStateMachine(cfg)
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.0).type == ActionType.PRESS
    sm.update(_features(pinch=0.2, pos=(0.8, 0.5)), 0.1)                        # drag
    a = sm.update(_features(pinch=0.2, pos=(0.82, 0.5), fist=True), 0.2)        # ambiguous frame
    assert sm.state == GestureState.DRAGGING
    assert a.type == ActionType.MOVE
    assert sm.update(_features(pinch=0.9, pos=(0.82, 0.5)), 0.3).type == ActionType.RELEASE


def test_bad_view_angle_freezes_without_releasing():
    cfg = GestureConfig(min_view_quality=0.08, min_pinch_frames=1, dwell_radius=0.04)
    sm = GestureStateMachine(cfg)
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5), view=0.5), 0.0).type == ActionType.PRESS
    assert sm._button_down is True
    # View goes oblique while held: freeze, but DON'T release the button.
    a = sm.update(_features(pinch=0.2, pos=(0.5, 0.5), view=0.01), 0.1)
    assert a.type == ActionType.NONE
    assert sm._button_down is True
    # Releasing the pinch (good view) lifts the button.
    assert sm.update(_features(pinch=0.9, pos=(0.5, 0.5), view=0.5), 0.2).type == ActionType.RELEASE


def test_view_quality_high_for_faceon_low_for_edgeon():
    r = GestureRecognizer(GestureConfig())
    # Face-on palm: wrist + MCPs spread in 2D (real area).
    faceon = np.zeros((21, 3), dtype=np.float32)
    faceon[Landmark.WRIST] = [0.50, 0.80, 0]
    faceon[Landmark.INDEX_MCP] = [0.40, 0.55, 0]
    faceon[Landmark.MIDDLE_MCP] = [0.50, 0.52, 0]
    faceon[Landmark.RING_MCP] = [0.60, 0.54, 0]
    faceon[Landmark.PINKY_MCP] = [0.68, 0.58, 0]
    q_face = r._view_quality(faceon[:, :2])
    # Edge-on palm: the same points nearly collinear (tiny area).
    edge = np.zeros((21, 3), dtype=np.float32)
    edge[Landmark.WRIST] = [0.50, 0.80, 0]
    edge[Landmark.INDEX_MCP] = [0.50, 0.55, 0]
    edge[Landmark.MIDDLE_MCP] = [0.505, 0.52, 0]
    edge[Landmark.RING_MCP] = [0.51, 0.54, 0]
    edge[Landmark.PINKY_MCP] = [0.515, 0.58, 0]
    q_edge = r._view_quality(edge[:, :2])
    assert q_face > q_edge
    assert q_edge < 0.08    # would be gated
    assert q_face > 0.08    # would pass


def test_edge_on_pinch_can_still_click():
    # A confident pinch must register even when the view is too oblique to move
    # the cursor (you rotate the hand sideways to pinch). The click lands on the
    # frozen cursor position.
    cfg = GestureConfig(min_view_quality=0.08, min_pinch_frames=1, dwell_radius=0.04)
    sm = GestureStateMachine(cfg)
    a = sm.update(_features(pinch=0.2, pos=(0.5, 0.5), view=0.01), 0.0)
    assert a.type == ActionType.PRESS
    assert sm._button_down is True
    # Releasing on the same poor view still lifts the button (a normal click).
    assert sm.update(_features(pinch=0.9, pos=(0.5, 0.5), view=0.01), 0.1).type == ActionType.RELEASE


def test_edge_on_non_pinch_still_freezes():
    # Without a pinch, a bad view freezes (no move, no state change).
    cfg = GestureConfig(min_view_quality=0.08)
    sm = GestureStateMachine(cfg)
    a = sm.update(_features(pinch=0.9, pos=(0.5, 0.5), view=0.01), 0.0)
    assert a.type == ActionType.NONE


def test_no_hand_goes_idle():
    sm = GestureStateMachine(GestureConfig())
    action = sm.update(None, now=0.0)
    assert sm.state == GestureState.IDLE
    assert action.type == ActionType.NONE


def test_pinch_debounce_delays_press():
    # min_pinch_frames=2: a single sub-engage frame shouldn't press.
    cfg = GestureConfig(min_pinch_frames=2, dwell_radius=0.04)
    sm = GestureStateMachine(cfg)
    assert sm.update(_features(pinch=0.2), now=0.0).type == ActionType.MOVE   # 1 frame: still moving
    assert sm.update(_features(pinch=0.2), now=0.1).type == ActionType.PRESS  # 2 frames -> press


def test_click_cooldown_blocks_rapid_reclick():
    cfg = GestureConfig(min_pinch_frames=1, dwell_time_s=0.1, click_cooldown_s=1.0,
                        dwell_radius=0.04)
    sm = GestureStateMachine(cfg)
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.0).type == ActionType.PRESS
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.15).type == ActionType.RECLICK
    # Another still period within the cooldown -> no second re-click.
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.3).type == ActionType.MOVE


def test_pinch_dead_zone_keeps_button_held():
    # A frame in the hysteresis dead zone (engage<d<release) must not release.
    cfg = GestureConfig(min_pinch_frames=1, pinch_engage=0.40, pinch_release=0.60,
                        dwell_radius=0.05)
    sm = GestureStateMachine(cfg)
    assert sm.update(_features(pinch=0.2, pos=(0.5, 0.5)), 0.0).type == ActionType.PRESS
    a = sm.update(_features(pinch=0.50, pos=(0.5, 0.5)), 0.1)       # dead zone -> still held
    assert a.type == ActionType.MOVE
    assert sm._button_down is True


def test_one_euro_reduces_jitter_but_tracks():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    # Noisy signal around 0.5; filtered output should have smaller variance.
    rng = np.random.default_rng(0)
    raw, filtered = [], []
    for i in range(200):
        v = 0.5 + rng.normal(0, 0.05)
        raw.append(v)
        filtered.append(f(v, timestamp=i / 60.0))
    assert np.std(filtered[50:]) < np.std(raw[50:])
    assert abs(np.mean(filtered[50:]) - 0.5) < 0.02  # no large bias


def test_two_palms_toggle_after_debounce():
    cfg = SystemConfig(enabled_on_start=True, toggle_min_frames=3, toggle_cooldown_s=0.0)
    tog = SystemToggle(cfg)
    assert not tog.update([_palm(), _palm()], now=0.0)   # frame 1
    assert not tog.update([_palm(), _palm()], now=0.1)   # frame 2
    assert tog.enabled                                    # not yet
    assert tog.update([_palm(), _palm()], now=0.2)        # frame 3 -> toggle
    assert not tog.enabled


def test_single_palm_never_toggles():
    tog = SystemToggle(SystemConfig(toggle_min_frames=2, toggle_cooldown_s=0.0))
    for i in range(6):
        tog.update([_palm()], now=i * 0.1)                # only one palm
    assert tog.enabled


def test_open_palm_property():
    assert _palm().is_open_palm
    assert _features(pinch=0.9).is_open_palm               # default posture is open hand
    assert not _features(pinch=0.9, point=True).is_open_palm   # a bare point is not a palm


def test_toggle_requires_releasing_hands_between_toggles():
    tog = SystemToggle(SystemConfig(toggle_min_frames=1, toggle_cooldown_s=0.0))
    assert tog.update([_palm(), _palm()], now=0.0)        # -> OFF
    assert not tog.enabled
    # Keep holding both palms: must NOT toggle again (not re-armed).
    assert not tog.update([_palm(), _palm()], now=0.1)
    assert not tog.enabled
    tog.update([], now=0.2)                                # drop hands -> re-arm
    assert tog.update([_palm(), _palm()], now=0.3)         # -> ON
    assert tog.enabled


def test_toggle_cooldown_blocks_immediate_retoggle():
    tog = SystemToggle(SystemConfig(toggle_min_frames=1, toggle_cooldown_s=1.0))
    assert tog.update([_palm(), _palm()], now=0.0)         # toggle OFF
    tog.update([], now=0.1)                                # re-arm
    # Within cooldown: re-arm happened but time guard blocks the toggle.
    assert not tog.update([_palm(), _palm()], now=0.5)
    assert not tog.enabled


def _pose_with(straight: set) -> HandPose:
    """Build a hand pose where the named fingers are straight (collinear joints)
    and the rest are curled (folded back). One finger per x-column."""
    from handcursor_core.tracking.types import FINGER_JOINTS
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[Landmark.WRIST] = [0.5, 0.9, 0.0]
    xs = {"thumb": 0.30, "index": 0.45, "middle": 0.50, "ring": 0.55, "pinky": 0.60}
    for name, (a, b, c, d) in FINGER_JOINTS.items():
        x = xs[name]
        if name in straight:                          # collinear -> ratio ~1.0
            pts[a], pts[b], pts[c], pts[d] = [x, .6, 0], [x, .5, 0], [x, .4, 0], [x, .3, 0]
        else:                                          # folds back -> ratio low
            pts[a], pts[b], pts[c], pts[d] = [x, .6, 0], [x, .5, 0], [x, .55, 0], [x, .6, 0]
    return HandPose(points=pts)


def test_straightness_separates_point_from_fist():
    point = GestureRecognizer(GestureConfig()).recognize(_pose_with({"index"}))
    assert point.fingers_extended["index"] and not point.fingers_extended["middle"]
    assert point.is_pointing and not point.is_fist
    assert point.index_straightness > 0.9

    fist = GestureRecognizer(GestureConfig()).recognize(_pose_with(set()))
    assert fist.is_fist and not fist.is_pointing
    assert fist.index_straightness < 0.7


def test_extension_hysteresis_holds_between_thresholds():
    # straightness in the [off, on] band keeps the prior state (no chatter).
    # reach forced to 0 here so only straightness drives the decision.
    r = GestureRecognizer(GestureConfig(finger_extend_on=0.82, finger_extend_off=0.68,
                                        finger_reach_on=9.9, finger_reach_off=9.9))
    assert r._extended("index", 0.90, 0.0) is True    # above on -> extended
    assert r._extended("index", 0.75, 0.0) is True     # in band -> stays extended
    assert r._extended("index", 0.60, 0.0) is False    # below off -> retracts
    assert r._extended("index", 0.75, 0.0) is False     # in band -> stays retracted


def test_reach_extends_a_bent_finger():
    # A bent finger (low straightness) that still reaches far from its knuckle
    # counts as extended — this is what keeps a pinch from reading as a fist.
    r = GestureRecognizer(GestureConfig(finger_extend_on=0.82, finger_reach_on=0.90))
    assert r._extended("index", straightness=0.40, reach=1.10) is True
    # Curled AND tucked (low on both) -> not extended.
    r2 = GestureRecognizer(GestureConfig(finger_extend_on=0.82, finger_reach_on=0.90))
    assert r2._extended("index", straightness=0.40, reach=0.45) is False


def test_recognizer_anchor_picks_knuckle_vs_tip():
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[Landmark.INDEX_TIP] = [0.5, 0.20, 0.0]
    pts[Landmark.INDEX_MCP] = [0.5, 0.60, 0.0]
    pts[Landmark.PINKY_MCP] = [0.7, 0.60, 0.0]      # gives non-zero hand_size
    pose = HandPose(points=pts)
    assert GestureRecognizer(GestureConfig(), anchor="tip").recognize(pose).pointer[1] == 0.20
    assert GestureRecognizer(GestureConfig(), anchor="mcp").recognize(pose).pointer[1] == 0.60
    blended = GestureRecognizer(GestureConfig(), anchor="blend").recognize(pose).pointer[1]
    assert abs(blended - 0.40) < 1e-6


def test_recognizer_palm_anchor_uses_palm_centroid():
    # "palm" anchor drives the cursor from the palm centroid (wrist + 4 MCPs),
    # the steadiest point and the default now that an open hand moves the cursor.
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[Landmark.WRIST] = [0.50, 0.80, 0.0]
    pts[Landmark.INDEX_MCP] = [0.40, 0.55, 0.0]
    pts[Landmark.MIDDLE_MCP] = [0.50, 0.52, 0.0]
    pts[Landmark.RING_MCP] = [0.60, 0.54, 0.0]
    pts[Landmark.PINKY_MCP] = [0.68, 0.58, 0.0]
    f = GestureRecognizer(GestureConfig(), anchor="palm").recognize(HandPose(points=pts))
    assert np.allclose(f.pointer, f.palm_center)


def test_monitor_mapping_spans_virtual_desktop():
    # Two 1920x1080 monitors side by side -> 3840x1080 virtual desktop.
    layout = MonitorLayout([
        Monitor(0, 0, 1920, 1080, is_primary=True),
        Monitor(1920, 0, 1920, 1080),
    ])
    assert layout.virtual_width == 3840
    # Center of active region -> center of virtual desktop.
    x, y = layout.active_region_to_screen(0.5, 0.5, 0.10, 0.90, 0.10, 0.90)
    assert abs(x - 1920) <= 1 and abs(y - 540) <= 1
    # Region edges -> desktop corners.
    assert layout.active_region_to_screen(0.10, 0.10, 0.10, 0.90, 0.10, 0.90) == (0, 0)
    x2, y2 = layout.active_region_to_screen(0.90, 0.90, 0.10, 0.90, 0.10, 0.90)
    assert x2 == 3839 and y2 == 1079


def test_monitor_mapping_clamps_outside_region():
    layout = MonitorLayout([Monitor(0, 0, 1920, 1080, is_primary=True)])
    # Way outside the active region clamps to a desktop corner, never beyond.
    x, y = layout.active_region_to_screen(-1.0, 2.0, 0.10, 0.90, 0.10, 0.90)
    assert x == 0 and y == 1079
