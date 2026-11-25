# raw_sr desktop (Electron + React + TypeScript)

A modern desktop UI over the existing `raw_sr` multi-frame super-resolution
engine. The Python core (`raw_sr/core`) is unchanged; this app adds an
Electron/React front end and a small FastAPI backend that exposes the engine.

## Architecture

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the frozen interface contract
(HTTP API, IPC bridge, design system, feature props).

```
Electron main ──spawn──► python -m desktop.backend.server  (FastAPI)
      │                                   ▲
      └── BrowserWindow ──HTTP────────────┘
              React + TypeScript renderer
```

## Result panel: alignment success, comparison, and zoom/pan

- The old multi-layer confidence map (coverage / SNR / registration) has
  been replaced by a single **alignment success** read-out: an overall score
  with level and human-readable verdict, a per-frame ZNCC / residual / shift
  list, and one low-resolution per-pixel alignment-quality heatmap. The
  confidence-layer tabs stay removed.
- The before/after images are shown **side by side**, each pane
  **independently zoomable**: wheel zoom toward the cursor (1x–12x), drag to
  pan, double-click to reset, and Fit / 1:1 toolbar buttons. The `before`
  pane is the honest reference frame rendered through the same pipeline as
  the frame preview — no extra sharpening.
- Layout order (top → bottom): **Alignment → Comparison → Metrics → View
  transform → Quality map**.
- **What the bottom heatmap is (and is not).** It is the *alignment-quality*
  map (`alignment.map` in the result JSON): per-pixel agreement between the
  warped frames. It is **not** a coverage map and **not** an “SNR map”.
  Drizzle's coverage array (`res.coverage`) — how much weight each HR pixel
  received — is a different quantity; the backend keeps it for the
  low-coverage warning and the desktop calls it “coverage” wherever it is
  mentioned. The legacy PySide UI (`raw_sr/ui/worker.py:142`) publishes that
  same coverage array under the name `snr_map`; that label is wrong. It is
  reported, not edited here, because `raw_sr/` belongs to another owner.
- The backend result JSON exposes this as the `alignment` field (previously
  `confidence`). `POST /api/jobs/{id}/render` returns both `fused` and
  `before`, each rendered with the same view transform.

## Honest metrics in the result panel

- The headline tile is **Detail vs Single**: the high-frequency band above one
  frame's Nyquist, measured against that same frame bicubically upscaled (the
  `scripts/sr_acceptance.py` convention). This is where genuine multi-frame
  resolution appears. A one-line note under the grid says what it means and
  that no held-out gain is claimed: the stricter leave-one-out test reads
  ≈ 0 dB on a real handheld burst (`docs/IBP_AND_HELDOUT.md` §7).
- The core's `hf_gain_db` (versus the **denoised LR stack**) is still reported,
  but only in the **Advanced** section as "Detail vs denoised stack
  (acutance)" — an already-averaged baseline, so it is not a resolution proof.
  The previously quoted +3.6 dB held-out figure was a border artefact of the
  old whole-frame metric and is never shown as evidence.
- Advanced also carries the run's fusion diagnostics, compactly: **Frame noise
  (fusion weight)** as a kept-frame count and σ range, and **Weight spread**
  (max/min of the normalised weights, 1.00 = uniform). A wide spread means a
  mixed-noise burst, which is exactly what the sharpness-per-noise weighting is
  for.

## Memory guard

The backend refuses an HR output larger than the memory budget *before* the
burst is decoded: `POST /api/process` answers **413** with an actionable
message ("Not enough memory … reduce the SR ratio or the ROI"), and a job-time
`MemoryError` from the core is mapped to a readable error with
`error_kind: "memory"` instead of a traceback. A run between 75 % and 100 % of
the budget still completes; the core's `RuntimeWarning` is captured and shown
as a non-fatal "Memory heads-up" note above the comparison. The optional
`memory_budget_bytes` request field raises the ceiling (default: 60 % of
available memory, clamped 2–6 GiB).

## ROI panel

- Zoom/pan over the preview: **Ctrl/⌘+wheel zooms toward the cursor** (plain
  wheel still scrolls), drag to pan, plus Fit and reset.
- Both the **Target ROI** and the **Search Area** have 8 resize handles and
  drag-to-move; the numeric x/y/w/h fields still work.
- Drawing a **new Target ROI auto-expands the Search Area** to a
  motion-derived margin around it. Moving/resizing an existing target leaves
  a hand-adjusted search area untouched.
- **Reset to full frame** sets the largest fully-covered crop computed from
  inter-frame motion (the intersection of all frames given their shifts) and
  places the search area around it; it falls back to the full frame when
  motion is unknown.

## Files panel

A compact **names-only** list (no thumbnails). When processing finishes, each
frame shows its **kept/dropped status under its name in the same cell**;
clicking a frame opens it in the ROI panel.

## Progress slot

The static-scene warning slot is hidden while a job runs and replaced by the
progress bar; on success a green "completed" line shows briefly, then the
slot hides.

## SR core algorithm

The classical-only SR core was refined for clarity and sharpness:

- Genuine iterative back-projection (IBP).
- **True-drizzle scatter deposition**: each measured LR sample is projected to
  its sub-pixel HR position and splatted with a compact bilinear kernel
  (`drizzle_deposit="scatter"`, `drizzle_splat="bilinear"`, pixfrac 0.5, 3×
  sub-samples); post-fusion blur default 0.
- Sharper DCB demosaic default.
- Sharpest-central reference-frame selection (not always frame 0).
- Optional edge-weighted, full-scale-refined ECC with an affine-regularised
  motion model (available; `registration_robust` is off by default because it
  did not improve the measured residual on real bursts).

### Removing edge ghosting

Parallel ghost lines on straight edges near the frame border came from frames
not overlaying exactly. The dominant cause was **nearest-neighbour deposition
quantisation** (up to one HR pixel per frame at 2×). The fix is a classical
**true-drizzle scatter**: every measured LR sample is projected to its
sub-pixel HR position and splatted with a compact bilinear kernel, so no
low-pass filters the recovered band and no detail is fabricated. Registration
uses the **sharpest-central reference frame**.

Tunables: `ibp_iterations`, `ibp_lambda`, `ibp_reg_sigma`, `ibp_max_step`,
`drizzle_deposit` / `drizzle_splat` / `drizzle_pixfrac` / `drizzle_subsamples`,
`drizzle_post_sigma`, `min_align_score`, and the `estimate_homography` settings
(`ecc_scale`, `edge_weighted`, `prefer_affine`). `SRResult` also carries
`kept_indices`, `align_scores`, and `align_residuals`.

Reported effect on the real 9-frame DNG burst (512 px ROI, 2×, shipped 1 IBP
sweep): **detail above a single frame's Nyquist ≈ +10.5 dB** versus that frame
bicubically upscaled (the honest resolution figure the result panel shows as
*Detail vs Single*), mean alignment residual ≈ 0.02 px, and **held-out
prediction gain ≈ −0.04 dB** — i.e. ≈ 0 dB: SR does not beat a denoised stack of
the same frames at predicting an unused frame, because a 9-frame handheld stack
already reaches the band a tenth frame carries (`docs/IBP_AND_HELDOUT.md` §7).

The older `hf_gain` ≈ +7.6 dB quoted here was measured against the **denoised
LR stack** — an already-averaged baseline, so it describes acutance relative to
that baseline, not recovered resolution. It is now labelled
"Detail vs denoised stack (acutance)" in the result panel's *Advanced* section,
together with the frame-noise and fusion-weight diagnostics. On the synthetic
ghost case, edge-ghost energy fell ≈ 0.90 → 0.20 and the recovered
above-Nyquist band rose ≈ 0.3% → 12.6%.

## Prerequisites

- Node.js 20+ (verified on 24)
- Python 3.11+ with the repo's runtime deps and the backend deps:

```bash
# from the repository root
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/pip install -r desktop/backend/requirements.txt
```

## Develop

```bash
cd desktop
npm install
npm run dev        # launches Electron + Vite with hot reload
```

The Electron main process starts the backend automatically using the repo
`venv` (override with `RAW_SR_PYTHON=/path/to/python`).

## Build

```bash
npm run typecheck  # tsc for main + renderer
npm run build      # electron-vite production build into out/
npm run package    # electron-builder installers (optional)
```

## Verify the backend headlessly

```bash
venv/bin/python -m pytest desktop/backend/tests -q
node desktop/scripts/e2e-smoke.mjs     # boots the backend, drives the API
```
