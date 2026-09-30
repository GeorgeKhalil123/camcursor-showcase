"""Gesture states and the actions the state machine emits.

The state machine is intentionally *pure*: it takes features in and returns
``CursorAction`` objects out. It never calls the OS. That keeps it trivially
testable and means the same FSM drives a real cursor, a dry-run logger, or a
future network/remote-display sink."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class GestureState(Enum):
    IDLE = auto()        # hand present but ambiguous (incl. a bare point) -> hold
    DISABLED = auto()    # fist -> cursor frozen, inputs ignored
    MOVING = auto()      # open hand -> cursor tracks the palm centre
    PINCH_CLICK = auto() # pinch just engaged -> button pressed
    DRAGGING = auto()    # pinch held while moving -> button still down
    # (POINTING — a bare index point — is reserved for a future precision mode.)


class ActionType(Enum):
    NONE = auto()
    MOVE = auto()        # move cursor to ``position`` (normalised active-region coords)
    PRESS = auto()       # button down
    RELEASE = auto()     # button up
    CLICK = auto()       # full press+release at the current cursor
    RECLICK = auto()     # release+press while held: registers a click, stays grabbed


@dataclass
class CursorAction:
    type: ActionType = ActionType.NONE
    # Position in normalised active-region space (0..1); the cursor controller
    # maps it to the virtual desktop. None when the action carries no movement.
    position: np.ndarray | None = None
