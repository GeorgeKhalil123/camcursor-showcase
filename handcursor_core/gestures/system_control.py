"""System-level master switch driven by the two-open-palms gesture.

This is deliberately separate from the per-hand cursor state machine: it needs
*both* hands and it gates everything else, so it's a higher-level concern. The
app consults it each frame; the cursor FSM only runs while ``enabled`` is True.

Robustness mirrors the rest of the system:
  * frame debounce  — both palms must hold for ``toggle_min_frames`` frames;
  * re-arm latch    — you must drop your hands between toggles, so one gesture
                      flips the switch exactly once;
  * cooldown        — a secondary time guard against rapid re-toggling.
"""

from __future__ import annotations


class SystemToggle:
    def __init__(self, config) -> None:
        self.cfg = config
        self.enabled = config.enabled_on_start
        self._frames = 0
        self._armed = True               # ready to accept a new toggle
        self._last_toggle = float("-inf")

    def update(self, all_features: list, now: float) -> bool:
        """Feed every detected hand's features. Returns True iff a toggle fired."""
        two_palms = sum(1 for f in all_features if f is not None and f.is_open_palm) >= 2

        if not two_palms:
            self._frames = 0
            self._armed = True           # hands released -> ready again
            return False

        self._frames += 1
        if (self._armed
                and self._frames >= self.cfg.toggle_min_frames
                and now - self._last_toggle >= self.cfg.toggle_cooldown_s):
            self.enabled = not self.enabled
            self._last_toggle = now
            self._armed = False          # must release hands before next toggle
            return True
        return False
