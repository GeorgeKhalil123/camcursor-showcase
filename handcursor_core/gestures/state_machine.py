"""The gesture state machine.

Pinch uses hysteresis + a frame counter + a cooldown so a single deliberate
pinch produces exactly one button-down. Two robustness features guard the
fragile fist/pinch boundary (both are pose/depth independent):

  * Sticky pinch (#2): once the button is held, a *transient* fist can't drop us
    to DISABLED — we leave the pinch only when the thumb-index distance clearly
    releases. A 1-2 frame misread can't break a drag.
  * Fist stabilization (#3): is_fist must persist for ``min_fist_frames``
    consecutive frames before we transition to DISABLED, so single-frame flicker
    doesn't freeze the cursor.

Click / drag model (pinch presses immediately on the rising edge):

    pinch engages              -> PRESS (button down)
    ... released quickly       -> RELEASE  => OS sees down-up = a normal CLICK
    ... held + moved           -> DRAGGING (MOVE) until release
    ... held still > dwell      -> RECLICK (release+press): registers a click and
                                   stays grabbed, so moving after resumes the drag
                                   (reliability fallback when a quick tap is missed)

Other transitions:
    no hand                    -> IDLE  (release any held button)
    fist for N frames          -> DISABLED (freeze)
    open hand, not pinched     -> MOVING (MOVE, anchored on the palm centre)

Move lives on the OPEN-HAND posture, not a finger point: an open palm is the
single most reliable pose to track (face-on, maximal view quality), so the most
frequent action sits where misreads are least likely, and the fragile pinch/fist
detection is only exercised briefly for clicks. A bare index point is left
unmapped (reserved for a future precision mode) and simply holds position.

Sticky pinch: once the button is held, a transient fist can't drop us to
DISABLED — we leave only when the thumb-index distance clears the release
threshold. Fist stabilization: is_fist must persist min_fist_frames before
DISABLED, so single-frame flicker doesn't freeze the cursor.
"""

from __future__ import annotations

import numpy as np

from ..config import GestureConfig
from .states import ActionType, CursorAction, GestureState


class GestureStateMachine:
    def __init__(self, config: GestureConfig) -> None:
        self.cfg = config
        # Runtime-toggleable (seeded from config) so the user can turn the
        # hold-still re-click off mid-session when it misfires.
        self.dwell_enabled = bool(config.dwell_enabled)
        self.state = GestureState.IDLE
        self._pinch_frames = 0          # consecutive frames below engage threshold
        self._fist_frames = 0           # consecutive frames classified as a fist
        self._button_down = False
        # -inf so the first click is never swallowed by the cooldown window.
        self._last_click_ts = float("-inf")
        self._anchor: np.ndarray | None = None  # cursor freeze point during pinch
        # Hold tracking while a pinch is held down: a quick release = click, a
        # held still pinch re-clicks (reliability fallback), movement = drag.
        self._hold_anchor: np.ndarray | None = None
        self._hold_start = 0.0
        self._hold_clicked = False      # one re-click per still episode
        # Consecutive frames during a held pinch where ALL of middle/ring/pinky
        # are curled. Once it crosses min_fist_frames the pinch releases — that
        # is the "closed into a full fist" exit from a held pinch, parallel to
        # the engage-side supporting-fingers check.
        self._no_support_frames = 0

    def update(self, features, now: float) -> CursorAction:
        """``features`` is a GestureFeatures or None (no hand this frame)."""
        if features is None:
            return self._to_idle()

        # Bad-angle gate: when the palm is too oblique to track reliably we hold
        # cursor MOVEMENT (positions are unreliable) — but, unlike before, we
        # still let a deliberate pinch through. A pinch is a discrete act and the
        # click lands on wherever the cursor was already parked, so an edge-on
        # palm (exactly the pose you rotate into to pinch) shouldn't swallow it.
        view_ok = features.view_quality >= self.cfg.min_view_quality

        pos = features.pointer

        # Pinch hysteresis: count frames below engage; reset only once clearly
        # released (above the higher release threshold). The gap = no chatter.
        if features.pinch_distance < self.cfg.pinch_engage:
            self._pinch_frames += 1
        elif features.pinch_distance > self.cfg.pinch_release:
            self._pinch_frames = 0
        # A pinch needs the index REACHING toward the thumb, not curled into the
        # palm: the reach floor is what separates a deliberate pinch / "OK" sign
        # (index reaching) from a fist (index fully curled), which can otherwise
        # show a small thumb<->index distance too.
        # PLUS a posture check: move requires an open hand, so when the user is
        # mid-motion and pinches, middle/ring/pinky tend to stay extended. A true
        # fist curls all five, so requiring at least one of those three to be up
        # vetoes the fist-as-pinch misread (especially under triangulation, where
        # the noisier 3D landmarks let reach occasionally cross the floor during
        # a fist).
        supporting_extended = any(features.fingers_extended.get(n, False)
                                  for n in ("middle", "ring", "pinky"))
        pinch_engaged = (self._pinch_frames >= self.cfg.min_pinch_frames
                         and features.index_reach >= self.cfg.pinch_index_reach_min
                         and supporting_extended)

        # --- Fix #2: sticky pinch ---
        # While the button is held we stay in the pinch/drag and ignore is_fist
        # entirely. Release on either (a) the thumb-index distance clearing the
        # release threshold, or (b) the supporting fingers staying curled for
        # min_fist_frames consecutive frames — that's the user explicitly
        # closing into a fist (thumb+index can stay close inside a fist, so
        # distance alone is insufficient). The frame counter preserves the
        # original anti-flicker protection: a 1-frame misread can't break a drag.
        if self._button_down:
            if features.pinch_distance > self.cfg.pinch_release:
                # Release: a quick down->up with no movement is an OS click; a
                # down->move->up is the end of a drag.
                return self._release(GestureState.MOVING)
            if not supporting_extended:
                self._no_support_frames += 1
                if self._no_support_frames >= self.cfg.min_fist_frames:
                    return self._release(GestureState.IDLE)
            else:
                self._no_support_frames = 0
            self._fist_frames = 0

            # Poor view while held: freeze position but keep the grab — don't
            # drag on unreliable landmarks, don't drop the button either.
            if not view_ok:
                return CursorAction(ActionType.NONE)

            moved = (self._hold_anchor is None
                     or float(np.linalg.norm(pos - self._hold_anchor)) > self.cfg.dwell_radius)
            if moved:
                # Moving while held => drag. Re-arm the hold timer + re-click.
                self._hold_anchor = pos.copy()
                self._hold_start = now
                self._hold_clicked = False
                self.state = GestureState.DRAGGING
                if self.cfg.freeze_on_pinch and self._anchor is not None:
                    return CursorAction(ActionType.MOVE, position=self._anchor)
                return CursorAction(ActionType.MOVE, position=pos)

            # Held still in one spot: re-click ONCE (reliability fallback).
            # RECLICK lifts then re-presses, so a click registers and the pinch
            # stays grabbed. You must move (-> drag) before it can re-click again,
            # so holding still doesn't machine-gun clicks.
            if (self.dwell_enabled and not self._hold_clicked
                    and (now - self._hold_start) >= self.cfg.dwell_time_s
                    and (now - self._last_click_ts) >= self.cfg.click_cooldown_s):
                self._hold_clicked = True
                self._last_click_ts = now
                self.state = GestureState.PINCH_CLICK
                return CursorAction(ActionType.RECLICK)
            self.state = GestureState.DRAGGING
            return CursorAction(ActionType.MOVE, position=pos)

        # Poor view and not holding: still let a confident pinch START a click
        # (it lands on the frozen cursor), but freeze everything else — fist /
        # open-hand reads are unreliable edge-on.
        if not view_ok:
            if pinch_engaged:
                return self._handle_pinch(pos, now)
            return CursorAction(ActionType.NONE)

        # Pinch takes precedence over the fist. A clear thumb+index touch with a
        # reaching index is a click (this is what lets an "OK" sign work) even on
        # a frame the hand would otherwise read as a fist; the reach floor in
        # pinch_engaged is what stops a truly curled fist from counting here.
        if pinch_engaged:
            return self._handle_pinch(pos, now)

        # --- Fix #3: temporal stabilization of the fist ---
        # Require the fist to persist before disabling; a brief misread just
        # reads as IDLE (also frozen) until confirmed.
        if features.is_fist:
            self._fist_frames += 1
        else:
            self._fist_frames = 0
        if self._fist_frames >= self.cfg.min_fist_frames:
            return self._to_disabled()

        # Not pinching, not a confirmed fist.
        self._reset_hold()

        if features.is_open_palm:
            self.state = GestureState.MOVING
            return CursorAction(ActionType.MOVE, position=pos)

        # Hand present but ambiguous posture (e.g. a bare index point): hold.
        self.state = GestureState.IDLE
        return CursorAction(ActionType.NONE)

    def _handle_pinch(self, pos: np.ndarray, now: float) -> CursorAction:
        """Rising edge of a pinch: press immediately (button down).

        A quick release then reads as a normal OS click; holding leads to a drag
        (on movement) or a re-click (held still) — all handled by the sticky
        branch on subsequent frames.
        """
        self._button_down = True
        self._anchor = pos.copy()
        self._hold_anchor = pos.copy()
        self._hold_start = now
        self._hold_clicked = False
        self.state = GestureState.PINCH_CLICK
        return CursorAction(ActionType.PRESS)

    def toggle_dwell(self) -> bool:
        """Flip the hold-still re-click on/off at runtime; returns the new state."""
        self.dwell_enabled = not self.dwell_enabled
        return self.dwell_enabled

    def _reset_hold(self) -> None:
        self._hold_anchor = None
        self._hold_clicked = False

    def _release(self, next_state: GestureState) -> CursorAction:
        self._button_down = False
        self._anchor = None
        self._fist_frames = 0
        self._no_support_frames = 0
        self._reset_hold()
        self.state = next_state
        return CursorAction(ActionType.RELEASE)

    def _to_idle(self) -> CursorAction:
        self._pinch_frames = 0
        self._fist_frames = 0
        self._no_support_frames = 0
        self._reset_hold()
        if self._button_down:
            return self._release(GestureState.IDLE)
        self.state = GestureState.IDLE
        return CursorAction(ActionType.NONE)

    def _to_disabled(self) -> CursorAction:
        self.state = GestureState.DISABLED
        self._pinch_frames = 0
        self._reset_hold()
        # _button_down is always False here (handled by the sticky-pinch branch).
        return CursorAction(ActionType.NONE)
