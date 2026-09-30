"""handcursor_core — the hardware-free core of CamCursor.

Gesture recognition and the gesture state machine, One Euro pointer smoothing,
multi-monitor mapping, the single-slot frame grabber, and the multi-camera
fusion layer (association, DLT triangulation, calibration I/O). Camera capture,
MediaPipe tracking and OS cursor injection live in the private app; toy
stand-ins are provided so everything here runs without hardware.
"""

__version__ = "0.1.0"
