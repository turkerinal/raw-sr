# Project Specification — RAW Multi-Frame Super-Resolution (No AI)

## 1. Project Goal

A desktop application that takes a burst of RAW photographs of a **static scene**, captured with small-to-moderate handheld camera movement (micro-jitter), and produces:

- A denoised image (SNR improvement via stacking), and/or
- A genuinely higher-resolution image (real spatial information recovered from subpixel offsets between frames)

**Core principle (must never be violated):** Do not fabricate unmeasured detail. Extract only the maximum spatial information that is physically supported by the actual RAW measurements.

**Explicitly forbidden:** No AI, no neural networks, no deep learning, no generative methods, anywhere in the pipeline. Only classical computer vision / signal processing techniques (feature matching, homography estimation, phase correlation, IRLS/weighted least squares fusion, etc.)

**Target use case for V1 ("Mode A"):** All frames in a burst show the *same* framing — the camera moves slightly between shots but every frame still contains the full scene of interest. This is different from wide-area tile scanning (Mode B, planned for V2) and different from photographing moving objects (explicitly out of scope, will never be supported — see Non-Goals).

## 2. Non-Goals (explicitly out of scope, do not attempt to solve these)

- Moving/dynamic subjects (vehicles, people, animals). The pipeline assumes a fully static scene. If frames show a moving object, results are undefined and the UI must warn the user about this rather than the pipeline trying to handle it.
- True 3D scene reconstruction / structure-from-motion. We do not want a 3D model of the scene, ever.
- Mesh warp / dense optical flow / occlusion-aware parallax correction — not in V1 or V2. Parallax is *avoided/rejected*, never *solved*, in this version of the project.
- Wide-area mosaic/tile scanning — this is V2 scope, not part of the version being built first.

## 3. Full Pipeline (V1, target state after phases 0.1.0 → 0.4.0)

```
1. RAW loading (multiple frames) + calibration
   (black level, white balance, dead pixel correction)
   NOTE: demosaic timing is phased — see Section 6 (Roadmap)
      ↓
2. Quality pre-filtering
   Laplacian sharpness score → auto-flag blurry frames,
   user can review/override the auto-flagged list
      ↓
3. Coarse fast preview
   Downscaled resolution, ORB + RANSAC global homography estimate
      ↓
   Motion sufficiency feedback (motion_analyzer)
   "Average inter-frame shift: 0.4px — suitable for subpixel SR"
   or "shift too small / too large — see notes"
      ↓
4. User: Target ROI selection (rectangle, OPTIONAL)
   User: Search Area selection (rectangle, INDEPENDENT shape from
   Target ROI, OPTIONAL)

   - If both left empty: system defaults target = search area =
     the full intersection area visible across all frames
   - If Target ROI is set but Search Area is empty: system proposes
     a default enlarged Search Area (editable by user)
      ↓
5. Feedback: "This region is visible in N/total frames"
      ↓
6. Homography estimation from Search Area features +
   subpixel phase correlation (on green channel)
      ↓
7. Homography applied to Target ROI → reprojection residual computed
      ↓
8. User: free-form SR ratio input (NOT fixed to 1.5x/2x — any value)
      ↓
9. Residual acceptance threshold is SCALED by the requested SR ratio
   (higher requested SR ratio → stricter residual tolerance required)
   Threshold computed ADAPTIVELY per frame set:
      threshold = median(residuals) + k * MAD(residuals)
   k has a sensible default; user can adjust via a
   strict <-> tolerant slider
      ↓
10. Frames/regions exceeding the threshold are automatically excluded
    (this parallax-avoidance step runs whether or not the user
    manually picked an ROI)

    If usable coverage drops below a low-usable-data threshold
    (e.g. <20% of target area or frame count): show a warning,
    but DO NOT abort — proceed with whatever data remains.
      ↓
11. System reports: "Requested SR: Xx — measurement-supported
    achievable SR: ~Yx" (this estimate is informed by both subpixel
    phase diversity AND the residual/SR-ratio relationship from step 9)
      ↓
12. HR grid construction + SNR-weighted fusion
    (drizzle-like weighted deposit; weighted shift-and-add / IRLS)
      ↓
13. Demosaic (exactly once — see Section 6 for which pipeline
    version does this on raw Bayer HR grid vs. on individual frames)
      ↓
14. Back-projection residual validation
      ↓
15. Alignment success assessment:
    - Overall score + level + human-readable verdict
    - Per-frame ZNCC / residual / shift list
    - One low-resolution per-pixel alignment-quality heatmap
      ↓
16. Output: 16-bit LINEAR TIFF (master file, never modified —
    this is the "ground truth" measured data)
      ↓
17. On-screen display applies a VIEW TRANSFORM (gamma/exposure)
    on top of the linear master, non-destructively — the linear
    master itself is never altered by this
      ↓
18. User: brightness/contrast adjustment (non-destructive, only
    affects the view transform layer, not the master file)
      ↓
19. Export: (a) linear master TIFF, or (b) baked/adjusted-view
    TIFF or PNG
```

### Alignment robustness (edge ghosting)

Parallel ghost lines on straight edges near the frame border mean frames are
not overlaying exactly. The dominant cause was **nearest-neighbour deposition
quantisation**: a nearest-neighbour HR deposit snaps each frame to the grid
(up to one HR pixel at 2x), so sub-pixel frames land on top of one another.
The deposit is now a classical **true-drizzle scatter**: each measured LR
sample is projected to its sub-pixel HR position and splatted with a compact
bilinear kernel (no low-pass, no fabricated detail). The reference frame is the
**sharpest frame near the burst centroid** (not always frame 0), so
displacements and model error stay small. Registration can optionally be
edge-weighted and refined at full scale with an affine-regularised fallback
(`edge_weighted` / `prefer_affine`, available but off by default). The approach
— sub-pixel, no fabricated detail — is fixed by Section 1.

## 4. Module Architecture

```
raw_sr/
├── core/
│   ├── raw_io.py             # RAW reading, sensor calibration (rawpy/LibRaw)
│   ├── quality_filter.py     # Laplacian sharpness scoring, blur rejection
│   ├── motion_analyzer.py    # Inter-frame shift estimation, sufficiency feedback
│   ├── roi_manager.py        # Target ROI + Search Area management
│   │                          # (defaults to full intersection if unset)
│   ├── registration.py       # Subpixel phase correlation; ECC homography,
│   │                          # optional edge-weighted / affine-regularised
│   │                          # refinement
│   ├── sr_ratio_analyzer.py  # Requested vs. physically-supported SR ratio
│   ├── validation.py         # Back-projection residual + ADAPTIVE threshold
│   │                          # (depends on sr_ratio_analyzer's target ratio)
│   ├── fusion.py              # SNR-weighted fusion; true-drizzle scatter
│   │                          # deposit onto the HR grid
│   ├── confidence.py          # Alignment-success assessment (score/level/
│   │                          # verdict, per-frame ZNCC/residual/shift list,
│   │                          # per-pixel alignment-quality heatmap)
│   ├── view_transform.py      # Linear -> display transform, non-destructive
│   │                          # brightness/contrast adjustment
│   └── export.py               # Master (linear) / baked export
├── ui/
│   ├── main_window.py
│   ├── file_panel.py          # RAW file selection — compact names-only list
│   │                          # (no thumbnails); per-frame kept/dropped
│   │                          # status after processing; click opens frame
│   │                          # in the ROI panel
│   ├── preview_panel.py       # Coarse preview + ROI/Search Area drawing
│   │                          # (8 resize handles, drag-to-move, Ctrl/⌘+wheel
│   │                          # zoom-to-cursor, pan); static warning hidden
│   │                          # while a job runs
│   ├── roi_selector.py        # Rectangle tool (Target + Search Area) with
│   │                          # numeric fields; a new target auto-expands the
│   │                          # search area by a motion-derived margin
│   ├── result_panel.py        # Side-by-side before/after (each independently
│   │                          # zoomable), alignment-success display,
│   │                          # brightness/contrast controls; layout:
│   │                          # Alignment → Comparison → Metrics → View
│   │                          # transform → Quality map
│   └── worker.py               # QThread background pipeline execution
├── metadata/
│   └── report.py               # JSON metrics report (frame count used,
│                               # SNR dB, effective resolution gain,
│                               # requested vs achieved SR ratio, RMS
│                               # registration error)
└── main.py
```

## 5. Key Cross-Module Dependency (must not be forgotten)

`validation.py` depends on `sr_ratio_analyzer.py`: the acceptable residual
threshold is not an absolute pixel value — it scales with how much SR the
user is requesting. A residual tolerable for 1x (denoise-only) is too loose
for 4x SR. This dependency must be wired explicitly; do not implement these
two modules as if they were independent.

## 6. Technology Stack

- UI: PySide6 (Qt6)
- Image processing: OpenCV, NumPy, rawpy/LibRaw
- Acceleration if needed: numba; later possibly pybind11 + C++ for
  critical paths (not needed for early phases)
- Threading: QThread + signal/slot (pipeline must never block the UI thread)
- Platform: Linux-first, but architecture should remain portable
  (avoid Linux-only hacks where a portable alternative exists)
- Input: RAW files ONLY. JPEG/PNG input is explicitly rejected with a
  clear error message ("this application requires RAW sensor data;
  JPEG/PNG are already demosaiced and cannot support this pipeline's
  linear master / calibration requirements").

## 7. Roadmap (SemVer) — Build in this order, one version at a time

### 0.1.0 — Core skeleton (early demosaic)
- RAW loading + calibration
- Demosaic performed EARLY (right after calibration), standard RGB
  pipeline from here on for this version
- Quality pre-filtering (blur rejection)
- Simple global registration (no ROI concept yet — target = full
  frame intersection automatically)
- Simple fusion (weighted shift-and-add; no IRLS yet)
- Export raw 16-bit Linear TIFF (no view transform/adjustment yet)
- Minimal UI: select files -> process -> save
- Acceptance: pipeline runs end-to-end on a real RAW test set without
  crashing, output opens correctly in an image viewer

### 0.2.0 — User control and transparency
- Target ROI + Search Area (rectangles, independent, both optional)
- motion_analyzer feedback
- Adaptive residual threshold (median + k*MAD), user-adjustable k
- Residual threshold scaled by requested SR ratio
- Free-form SR ratio input + requested-vs-achieved feedback
- Alignment-success display (score + level + verdict, per-frame
  ZNCC/residual/shift list, per-pixel alignment-quality heatmap)
- Non-destructive view transform (brightness/contrast)
- Export: master (linear) + baked options
- Low-coverage warning (proceed anyway, don't abort)
- Acceptance: on a flat test surface (e.g. printed document) with
  15+ frames, measurable SNR improvement of at least 3-5 dB;
  alignment-quality heatmap visibly reflects excluded/low-confidence regions

### 0.3.0 — Bayer-domain migration (late demosaic)
- registration.py and fusion.py migrated to operate on raw Bayer data
  (green channel primarily) instead of post-demosaic RGB
- Demosaic moved to occur exactly once, after fusion, on the HR grid
- Per-channel SNR weighting (compensating for sparser R/B sampling)
- Direct A/B comparison against 0.2.0 output on the same test set —
  must show measurable sharpness/resolution improvement to justify
  the added complexity; if it does not, document why and reconsider

### 0.4.0 — Hardening
- Test against multiple RAW formats (CR2, NEF, ARW, DNG minimum)
- Performance profiling and optimization of critical paths
- Improved error handling and user-facing error messages
- Finalized JSON metadata/report schema

### 1.0.0 — Stable release
- All acceptance criteria from 0.2.0/0.3.0 hold consistently across
  the tested RAW format set
- RAW-only, late-demosaic pipeline is the default and only path
  (early-demosaic code from 0.1.0 may be removed or kept behind a
  debug flag, developer's choice)
- Basic documentation + Linux packaging

## 8. Explicitly Deferred to V2 (do not build in V1, but keep the
architecture open to it)

- Mosaic / tile-based wide-area scanning ("Mode B" — camera physically
  swept across an area larger than a single frame's field of view)
- Tile-to-tile registration (bundle-adjustment-style global alignment
  across tiles)
- Seam/stitch confidence and blending
- Each tile, once V2 is built, should still use V1's Target ROI +
  Search Area + residual-based parallax-avoidance logic internally —
  V2 adds a layer on top of V1, it does not replace it

## 9. Explicitly Deferred Beyond V2 (undefined scope, not scheduled)

- Mesh warp / dense optical flow / structure-from-motion for true
  parallax/depth handling (V1 and V2 both only *avoid/reject*
  parallax via user-controlled Search Area + residual exclusion —
  neither actually solves it)
- Multiple simultaneous ROIs
- Real-time capture guidance (live camera feed feedback)
- GPU acceleration

## 10. UI Requirements Summary

- File selection with RAW-only validation and clear rejection message
  for non-RAW input
- Frames shown as a compact, names-only list (no thumbnails) with
  auto-flagged blurry frames (user can override); when processing
  finishes, each frame shows its kept/dropped status under its name in
  the same cell, and clicking a frame opens it in the ROI panel
- Coarse preview with rectangle tools for Target ROI and Search Area
  (independently shaped/positioned), each with 8 resize handles and
  drag-to-move (numeric x/y/w/h fields still work); Ctrl/⌘+wheel zooms
  toward the cursor (plain wheel still scrolls), plus pan and Fit/reset;
  a newly drawn Target ROI auto-expands the Search Area by a
  motion-derived margin; "Reset to full frame" sets the largest
  fully-covered crop computed from inter-frame motion (the intersection
  of all frames given their shifts), falling back to the full frame when
  motion is unknown
- Static-scene warning shown while idle: "This tool is designed for
  static scenes photographed with a moving camera. It does not produce
  reliable results for moving subjects (vehicles, people, animals)." It
  is hidden while a job runs and replaced by the progress bar; on success
  a green "completed" line shows briefly, then the slot hides.
- Live feedback during ROI selection: "N/total frames usable"
- Free-form SR ratio input with live requested-vs-achievable feedback
- Progress bar + cancel button during processing
- Result screen (top to bottom: Alignment, Comparison, Metrics, View
  transform, Quality map): before/after shown **side by side**, each pane
  **independently zoomable**, a single alignment-success display (overall
  score + level + human-readable verdict, a per-frame ZNCC/residual/shift
  list, and one low-resolution per-pixel alignment-quality heatmap), and
  non-destructive brightness/contrast controls. The `before` pane is the
  honest reference frame rendered through the same pipeline as the frame
  preview (no extra sharpening). The confidence-layer tabs are not used.
- Result panes zoom/pan: wheel zoom toward the cursor (1x-12x), drag to
  pan, double-click to reset, Fit / 1:1 toolbar buttons
- Export dialog: master (linear) vs. baked (adjusted view) choice

## 11. Acceptance Criteria for 1.0.0

- On a flat test surface with 15+ RAW frames: measurable SNR
  improvement of at least 3-5 dB
- Frequency-domain analysis shows genuine (non-aliased) high-frequency
  content gain in the HR output vs. any single input frame
- Processing time for a medium-sized ROI (e.g. 500x500px source
  region) completes in a reasonable time (seconds to ~1 minute,
  hardware-dependent)
- If the user selects a region with significant depth variation
  (parallax), the system does not crash — it either produces a
  low-confidence result with clear visual indication, or warns the
  user clearly

## 12. Development Process Notes (for the coding agent)

- This specification is final and already approved by the project
  owner. Do not re-run open-ended brainstorming on the overall
  architecture — go straight to planning implementation for the
  requested version.
- Work one version (0.1.0, then 0.2.0, etc.) at a time. Do not start
  implementing a later version's features early unless explicitly
  asked.
- After finishing a version, wait for explicit confirmation from the
  project owner (manual visual/metric validation on real RAW data)
  before proceeding to the next version.
- Record significant architectural decisions and any deviations from
  this spec in the session log as they happen.
