# Architecture

CamCursor is a real-time computer-vision pipeline that turns a webcam feed into
desktop cursor control. This document explains how it's structured, the reasoning
behind the key decisions, and how the single-camera MVP was designed to grow into
multi-camera 3D tracking without a rewrite.

> This is the design-level companion to the [README](README.md). The full app lives
> in a private repository; the hardware-free core described in sections 2-7
> (grabber, recognizer, state machine, smoothing, monitor mapping, fusion) is
> included and runnable in [`handcursor_core/`](handcursor_core/). `config.yaml`
> below refers to the full app's tuned config, which is not included.

---

## 1. The pipeline

Everything is a one-directional dataflow. Each stage consumes one well-defined type
and produces the next, so any stage can be replaced in isolation.

```
┌────────────┐   BGR frame + ts   ┌────────────┐   HandPose    ┌────────────┐
│  camera/   │ ─────────────────▶ │  tracking/ │ ────────────▶ │ gestures/  │
│ threaded   │                    │ MediaPipe  │               │ recognizer │
│  capture   │                    │ landmarks  │               │  + FSM     │
└────────────┘                    └────────────┘               └─────┬──────┘
                                                                     │ CursorAction
                                                                     ▼
┌────────────┐   global (x,y)     ┌────────────┐  smoothed     ┌────────────┐
│  cursor/   │ ◀───────────────── │ monitors/  │ ◀──────────── │  cursor/   │
│  pynput    │                    │  mapping   │  normalised   │  One Euro  │
│ (OS events)│                    │            │  pos          │  filter    │
└────────────┘                    └────────────┘               └────────────┘
```

| Stage | Responsibility | Output contract |
|---|---|---|
| `camera/` | Grab frames without buffer lag | `(BGR frame, timestamp)` |
| `tracking/` | Detect the hand | `HandPose` — 21 landmarks (x, y, z), handedness, score |
| `gestures/` | Recognise intent | `GestureFeatures` → `CursorAction` |
| `cursor/` | Smooth + actuate | normalised cursor position; OS button events |
| `monitors/` | Place on screen | global pixel `(x, y)` across all displays |

Because the contracts are stable, the MVP's 2D path and the future 3D path share
every stage except the two that genuinely change (tracking source and screen mapping).

---

## 2. Capture: killing latency at the source

OpenCV's `VideoCapture` maintains an internal frame buffer. If the pipeline reads
slower than the camera produces frames — and MediaPipe inference is bursty — the
buffer fills and you end up acting on stale frames. Latency creeps up invisibly and
never recovers.

The fix is a dedicated grabber thread that continuously reads and keeps **only the
newest frame**. The pipeline always pulls "now," so end-to-end latency stays bounded
regardless of how inference time fluctuates.

---

## 3. Tracking → features: scale invariance

MediaPipe returns 21 landmarks per hand in normalised image coordinates. Raw
landmarks are the wrong abstraction for gesture logic: the same pinch looks different
near vs. far from the camera.

The recognizer converts landmarks into **scale-invariant features** — everything is
normalised by hand size (e.g. thumb–index distance divided by a stable palm span).
This means:

- Gestures behave identically at any distance from the camera.
- The thresholds in `config.yaml` are meaningful and portable, not per-setup magic
  numbers.
- The same feature definitions keep working when "size" becomes real metric depth
  from triangulation.

Representative features: `pinch_distance`, `index_reach`, `fingers_extended` (per
finger), `is_fist`, `is_open_palm`, `view_quality`, and the `pointer` position.

---

## 4. The gesture state machine

This is the most intricate logic in the project, so it's kept **pure**: it takes
`GestureFeatures` in and returns a `CursorAction` out, and never touches the OS. That
purity is what makes it unit-testable without any hardware.

### States

```
IDLE ──open palm──▶ MOVING ──pinch──▶ PINCH_CLICK ──move──▶ DRAGGING
  ▲                    │                    │                   │
  │                    └───fist (N frames)──┴──▶ DISABLED       │
  └──────────────────────── release / no hand ──────────────────┘
```

### Click and drag from one mechanism

The pinch presses the mouse button on the **rising edge**. From there:

- **Quick release** → the OS sees a down-up with no movement → a normal click.
- **Hold + move** → `DRAGGING`.
- **Hold still past a dwell time** → a re-click reliability fallback (releases and
  re-presses so a missed quick-tap still registers, without machine-gunning clicks).

One code path, three natural behaviours — no separate "is this a click or a drag?"
classifier.

### Robustness against misreads

Hand tracking is noisy at the fist/pinch boundary. Three mechanisms guard it:

1. **Hysteresis.** Separate `pinch_engage` and `pinch_release` thresholds create a
   dead-band, so a value hovering at the boundary can't rapidly toggle.
2. **Frame debounce.** A pinch must persist for `min_pinch_frames` before it fires;
   a fist must persist for `min_fist_frames` before it freezes the cursor. Single-frame
   flicker is absorbed.
3. **Sticky pinch.** Once the button is held, a *transient* fist can't drop the
   interaction to `DISABLED`. The drag ends only when the thumb–index distance clearly
   releases, or when the supporting fingers stay curled for `min_fist_frames` (a
   deliberate close-into-fist). A 1–2 frame glitch can never break a drag.

### Why "move" lives on the open palm

An open, face-on palm is the highest-quality pose MediaPipe can track. Putting the
*most frequent* action (moving) on the *most reliable* pose means the fragile
pinch/fist detection is only exercised briefly, for discrete clicks — where a rare
misread is cheap. A bare index point is intentionally left unmapped, reserved for a
future precision-pointing mode.

---

## 5. Smoothing: the One Euro filter

Pointer smoothing is a genuine trade-off. Too little and the cursor jitters while you
try to hold on a target; too much and it lags when you move fast. A fixed filter can't
win both.

The **One Euro filter** (Casiez, Roussel & Vogel, CHI 2012) adapts its cutoff
frequency to the signal's speed:

- **Slow motion** (holding on a target) → low cutoff → heavy smoothing → no jitter.
- **Fast motion** (crossing monitors) → high cutoff → light smoothing → no lag.

It also derives its filter coefficients from **real elapsed time** between samples,
so it's frame-rate independent — essential when inference time (and therefore
effective FPS) varies frame to frame.

```python
tau   = 1.0 / (2.0 * math.pi * cutoff)   # cutoff → time constant
alpha = 1.0 / (1.0 + tau / dt)           # time constant + real dt → EMA weight
```

Two intuitive knobs, both in `config.yaml`: `min_cutoff` (baseline smoothing) and
`beta` (how aggressively smoothing drops as speed rises).

---

## 6. Screen mapping: the single seam for 3D

`monitors/` enumerates the display layout and maps a normalised position from the
camera's active region onto the **entire virtual desktop**, so the cursor spans all
monitors. In the MVP this is a linear remap of an active sub-region of the frame.

Crucially, `active_region_to_screen` is the **one** place the "where on screen"
decision is made. That's the seam the 3D upgrade replaces: instead of a 2D remap, it
becomes "which monitor plane does the pointing ray intersect?" — and nothing upstream
changes.

---

## 7. Multi-camera fusion (built, calibration-gated)

The endgame is a true metric 3D hand reconstructed from several cameras. The fusion
layer for this is already implemented and unit-tested; it sits behind the same
`fuse(views, timestamp)` contract the app already calls.

### Fusion strategies

| Strategy | Needs calibration? | What it does |
|---|---|---|
| `FeatureFusion` | No | Blends per-camera scalar features; robust, angle-tolerant. Active default. |
| `TriangulationFusion` | Yes | Reconstructs the 21 landmarks in metric 3D via DLT. |

With a **single** camera, both strategies reduce to a provable passthrough — so the
fusion layer ships today without changing MVP behaviour.

### Association

Before triangulating, the same physical hand must be matched across cameras. The
current strategy associates by **handedness** (and position for the primary camera);
the documented upgrade is epipolar / 3D-proximity association once cameras are
calibrated.

### DLT triangulation

Each landmark is reconstructed independently. For a 3D point **X** seen at pixel
`(x, y)` through a 3×4 projection matrix **P = K·[R | t]**, each camera contributes
two linear constraints:

```
x · (P row 2) − (P row 0) = 0
y · (P row 2) − (P row 1) = 0
```

Stacking these from every camera that saw the landmark gives a homogeneous system
**A·X = 0**. The least-squares solution is the right singular vector of **A** with the
smallest singular value (via SVD), then de-homogenised. Degenerate configurations
(near-coincident cameras → a point at infinity) are detected and rejected rather than
leaking `NaN` downstream.

### Time synchronisation

DLT assumes all views capture the *same instant*. A camera whose frame lags the
freshest view in its group by more than `max_view_skew_s` is dropped from the solve —
otherwise it would drag the reconstruction toward where the hand *was*.

### Calibration

The one piece that genuinely needs field data is calibration: each camera's
intrinsics **K** and its pose **(R, t)** in a shared world frame. A checkerboard
routine (`cv2.calibrateCamera` + `stereoCalibrate`) produces these into a
`calibration.yaml`, which the calibration-I/O module loads to activate triangulation.

### Remaining milestones

- **Ray-to-monitor-plane mapping** in `monitors/` — the last seam, using the fused 3D
  pose to intersect the pointing ray with each physical display plane.
- **Triangulation quality** — Hartley normalisation, plus per-view score weighting and
  reprojection-error outlier rejection so one occluded fingertip can't corrupt a
  landmark.
- **Association confidence** — flag positionally-matched hands so downstream logic can
  distrust them until epipolar association is available.

Once those land, the queued gestures (scroll, right-click, window grab/throw, zoom,
presentation mode) are just new states in the state machine and new action types — no
pipeline surgery.

---

## 8. Configuration and testing

- **`config.yaml`** — every tunable (pinch thresholds, smoothing knobs, active region,
  fusion strategy, calibration path) is documented inline. A typed loader validates it.
- **Tests** — hardware-free unit tests (85 in the full project's working tree as of 2026-10-07) cover the state
  machine's transitions, the One Euro filter, coordinate mapping, and the DLT math. No
  camera, no mouse, deterministic — the parts most likely to regress are exactly the
  parts under test. 76 of them are ported into this repo's `tests/`.

---

## Design principles, in one line each

- **One direction, stable contracts** — stages are swappable in isolation.
- **Pure decision logic** — the hard parts never touch the OS, so they're testable.
- **Adapt to the signal** — smoothing and thresholds respond to speed and hand size,
  not fixed assumptions.
- **Reliability over cleverness** — the most frequent action sits on the most reliable
  pose; misreads are absorbed, not propagated.
- **Build the future behind a passthrough** — 3D math ships risk-free today and
  activates without a rewrite.
