"""
Genuine multi-frame super-resolution core (v0.3.0 -> 1.3.0).

Classical computer vision / signal processing only — no AI, no neural nets.

The previous v0.2.0 "pipeline" produced a HALF-RESOLUTION image (its demosaic
emitted one sample per 2x2 Bayer block) and then did same-resolution
shift-and-add, so it could not upscale. This module recovers *real* spatial
information from the subpixel offsets between frames of a handheld burst and
deposits it onto a higher-resolution grid (drizzle-like), optionally refined
by iterative back-projection (IBP).

Core guarantee (see SPEC.md §1): never fabricate unmeasured detail — every HR
pixel is a weighted combination of actual RAW measurements placed at their
physically estimated subpixel positions.

Ghosting / border-alignment design notes (v1.2.0)
-------------------------------------------------
The failure targeted: after SR, zooming into straight edges near the frame
border shows parallel ghost lines — frames not overlaying exactly. A first
revision switched the deposition from a nearest *gather* to a linear *gather*;
that removed the ghosts on synthetic bursts but **destroyed clarity on real
data** (measured −4 dB HF gain, sharpness ratio 2.55, versus +5.1 dB / 5.4 for
the compact nearest gather). The reason is fundamental: any interpolated
*gather* is a low-pass. Its reconstruction kernel has its first zero exactly at
the single-frame Nyquist, so it cannot keep the band above it that multi-frame
SR exists to recover (linear/cubic/lanczos/area gathers all measured ≈ −3 dB).

This revision therefore uses a **true drizzle scatter** (Fruchter & Hook):

* **Scatter, not gather.** :func:`drizzle` (default ``deposit="scatter"``)
  computes each LR pixel's HR position through the drizzle matrix and
  *deposits* its own measured value there with a compact bilinear splat
  (``splat="bilinear"``, ratio-aware ``pixfrac``/``subsamples`` — see
  :func:`default_deposit_params`), accumulating a
  weight map. The value is never blended with its LR neighbours, and the splat
  kernel's first zero sits at the HR Nyquist, so the sub-pixel placement is
  exact **and** the recovered band is preserved. Measured on the real burst:
  +7.55 dB HF gain, sharpness ratio 7.20, residual 0.021 px (targets +5.0 /
  5.5 / 0.03) at the historical 2x tuning. The deposit footprint stays 0.5 LR
  px at every ratio, so in HR pixels it widens with ratio; shrinking it in HR
  terms (``min(0.5, 1/ratio)``) measured as **neutral in prediction** (a few
  tenths of a dB, inside the burst's frame-to-frame scatter) while multiplying
  Laplacian sharpness several-fold — see :func:`default_deposit_params` for the
  numbers. Only the sub-sample lattice scales with the footprint in HR pixels.
  The legacy gather remains selectable for A/B. (Wider splats — ``area`` at
  ``pixfrac=0.7``/``subsamples=4`` or a Gaussian — give a slightly smoother
  step response but lose real high-frequency gain, so bilinear is the shipped
  default.)
* **HR grid geometry.** The HR grid is **pixel-centre aligned**:
  :func:`to_drizzle_matrix` includes the ``(ratio - 1)/2`` HR-pixel offset, so
  HR pixel ``i`` has its centre at the reference LR coordinate
  ``(i - (ratio - 1)/2) / ratio``. That is exactly the ``cv2.resize``
  convention, so the drizzle output, the bicubic baseline, the "before" panel,
  the held-out prediction (:func:`predict_lr_from_hr`), the IBP forward model
  and a centre-aligned GT upscale all share one grid. The pre-2026-11-20 grid
  was corner-aligned and sat 0.5/1.0/1.5 HR px up-left at 2x/3x/4x, which
  misaligned the before/after comparison and left a 1-4 px black rim at the
  bottom/right at 3x.
* **Reference choice.** The reference is not frame 0: :func:`_select_reference`
  picks the sharpest frame near the burst centroid so every displacement stays
  small. This is the single largest registration win (residual 0.057 -> 0.021
  px on the real burst) and is kept. The chosen index is reported as
  ``SRResult.reference_index``, and **every** consumer that relates ``hr`` to a
  specific frame must start there (see :func:`predict_lr_from_hr`): the
  held-out proof used to start from ``frames[0]``, which silently discarded the
  inter-frame offset and reported ~0 dB gain at 2x/3x/4x instead of
  +3.5 dB (ROI 1024) / +7 dB (ROI 512).
* **Registration.** :func:`estimate_homography` still offers edge weighting and
  an affine fallback (``edge_weighted``/``prefer_affine``). Measured on the real
  burst this **hurts** once the scatter deposit is in place (mean residual
  0.034 px vs 0.021 px, and lower sharpness), so the orchestrator default is
  ``registration_robust=False`` (plain full homography) and the automatic
  reference selection is what carries the alignment. The robust path remains
  available for perspective-heavy bursts.
* **Fixed transforms.** :func:`iterative_back_projection` can re-estimate each
  frame's residual translation against the current HR estimate (EBP) and update
  its transform as it alternates with back-projection. Measured, this is neutral
  once the causes above are fixed, so it is available but off by default.
* **IBP strength is bounded by measurement, not by sharpness.** The forward
  model integrates the HR estimate over the LR pixel support (Gaussian,
  ``sigma = 0.289 * ratio`` HR px) and the back-projection applies the adjoint
  of *that assumed* forward model (the correction is blurred by the same PSF;
  it is the true adjoint of the Gaussian+sample operator, not of the drizzle
  deposit itself). Beyond that, extra sweeps only add acutance: on the real
  burst a 1- vs 3-sweep sweep measured the same HF band (+0.1 dB) but 0.8-1.3
  dB *less* held-out PSNR, so the default is one sweep. The sweep count and the
  rest of the configuration are reported in ``metrics['ibp_*']`` (see
  ``docs/IBP_AND_HELDOUT.md``), so runs made with a different setting stay
  comparable.

Lens distortion (investigated, not enabled). Real lenses add radial distortion
that a single homography cannot fit, so the residual grows toward the corners.
On a synthetic radial-distortion burst an *oracle* undistortion (using the known
division-model coefficient) lowers the border/center residual ratio from 2.43 to
2.09 and the MSE from 7.7e-3 to 3.7e-3, so the headroom is real. However, on a
short, small-motion handheld burst the radial coefficient and the homography are
nearly degenerate: the frames simply do not move enough for the corners to pin
the parameter, so estimating it jointly tends to absorb noise into geometry and
*fabricate* a radial warp (violating SPEC §1). No estimation is therefore
enabled; callers with a calibrated lens profile should undistort before calling
this module.

Performance (v1.3.0)
--------------------
Same outputs, faster wall clock, measured on the real 9-frame burst:

* **Scatter engine.** :func:`_scatter_accumulate` now runs bilinear/area splats
  through :func:`_scatter_fast`, a banded row-chunk kernel that keeps each
  band's accumulation table L3-resident while :func:`np.bincount` deposits
  into it, and computes the projected geometry once per band for all
  ``subsamples^2`` footprint offsets at once. Numerically equal to the
  previous per-tap loop (worst measured |diff| 5.6e-17 over translation/
  rotation+perspective/negative-offset/fractional-ratio matrices, impulse
  splats, both splats) at ~3.5x the speed; the generic loop remains for the
  cubic/gaussian splats.
* **OpenCL (OpenCV T-API, brand-agnostic).** ``SuperResolution(use_opencl=...)``
  routes the upsample-direction ``warpPerspective`` calls (IBP footprints and
  residual back-projections, drizzle gather) through UMat when the output fits
  a host-transfer guard; every device call falls back to the identical CPU call
  on any failure. Auto (``None``) enables it only when a device exists *and* a
  one-time calibration measures it faster — measured on a Mesa-CL userspace
  stack every candidate op ran 0.14-0.72x of the CPU, and measured on an
  NVIDIA RTX 3050 Laptop GPU (Optimus) with the *correct* device selected the
  routed warp still ran 0.09x of the CPU at 1024² and 0.63-0.82x at 4096²,
  so auto honestly keeps the CPU on both machines. The decision is reported in
  ``metrics['opencl_active']`` (executing device: ``metrics['opencl_device']``).
* **Registration.** The model-selection scores inside
  :func:`estimate_homography` are memoized per candidate matrix (they were
  computed 2-3 times each).
* **Registration work that was provably redundant (v1.5.0).** Four pieces of
  duplicate work per frame are gone, all bit-identical in output
  (``tests/test_superres_registration_perf.py`` compares transforms byte for
  byte, and a real-burst fingerprint of the shifts/homographies is pinned
  there): the ref->mov phase correlation is measured once and handed to
  :func:`estimate_homography` as its seed; the score and the residual share one
  :func:`_render_in_ref_frame` instead of warping twice; the reference frame's
  gradient mask is built once per run; and the Hanning window is cached per
  shape. The registration pool no longer pickles 3 full float32 frames per
  task either — tasks carry slot indices into one shared-memory block
  (:class:`_SharedRegFrames`), which is what the 'spawn' start method made
  necessary (fork-inherited globals are impossible: rawpy deadlocks a forked
  child). Measured on the real burst, registration-only wall: ROI 1024 serial
  7.72 -> 7.61 s, 4-worker pool 7.30 -> 7.01 s.
* **Registration solves cut from six per frame to four (v1.6.0).** The
  homography is no longer fitted three times. At ROI >= 1024 it is warm-started
  from the refined affine solution (the affine models are ECC fits that
  converge in ~25 iterations; the homography fit is the one that runs to its
  iteration cap, so it dominated the phase's cost). The half-scale homography
  pass and the independent full-scale solve started from the
  phase-correlation seed are gone; the seed-based solve remains as the
  fallback for a warm result that is missing or unconvincing. Measured on the real burst: <=0.004 px maximum
  corner displacement at ROI 1024 for both ``ecc_scale`` 0.25 (acceptance) and
  0.5, alignment score and residual unchanged to 1e-4, and registration-only
  wall (4-worker pool) 7.17 -> 3.4-4.4 s. At ROI 512 the original route is kept
  verbatim (the seed-based solve still wins on 3 of 8 frames there), so
  small-ROI registration stays bit-identical.
* **Parallel registration (v1.4.0).** An earlier multi-process registration
  experiment at ROI 1024 gained only 1.21x wall and was rejected — but its
  workers were oversubscribed (each still multi-threaded). Measured properly
  (single-threaded spawned workers, real frames), one frame's ECC fit is a
  ~4.3 s memory-latency-bound serial stretch regardless of thread count, so
  :class:`SuperResolution` registers frames in a pool sized
  ``min(4, frames, cpus-1)`` where the frame is big enough: the ROI-2048
  registration loop goes 41.1 s -> 34.1 s (1.21x wall; the fit saturates
  memory bandwidth, so the win plateaus at ~3 workers, and pool sizes beyond
  the plateau only burn CPU). Results are collected by frame index and the
  fit is bit-identical across thread counts (measured), so the pooled
  transforms equal the serial ones exactly; small frames stay on the serial
  path because pool spawn costs more than their whole fit (see
  ``_REG_POOL_MIN_REF_PIXELS``).
* **Rejected: parallel drizzle scatter.** A frame-parallel deposit (each frame
  scattered concurrently into disjoint shared-memory tables, folded in frame
  order — bit-identical by construction) was built and measured at product
  size: 40.7 s serial vs 41.9/46.2/51.1/54.5 s wall with 2/3/4/6 workers.
  The banded :func:`_scatter_fast` kernel already saturates the memory pipe on
  one core (~536 MB float64 tables per frame), so overlap only adds
  contention; the deposit stays single-threaded (see :func:`_scatter_fast`).
* **GPU scatter deposition (pyopencl, optional).** The deposit is the one op
  class where a real GPU has an architectural edge: it is pure
  memory-bandwidth streaming, and the CPU engine is bound by *shared* system
  RAM while the GPU streams its own dedicated GDDR6.
  :class:`_GpuScatterDeposit` mirrors the bilinear/area splat one OpenCL
  work-item per LR pixel (double-precision geometry, float32 atomic
  accumulation into HR tables shared by all frames; portable CAS atomics,
  row-chunked upload under a 64 MB input guard, and a 1.5 GB VRAM table guard
  that keeps huge ROIs on the CPU). The GPU tables equal the CPU float64
  tables within ~1e-6 relative (f32 accumulation), and — atomics being
  unordered — NOT bit-to-bit between GPU runs; the CPU path remains
  bit-identical and is the fallback for every failure mode (no module, no
  device, no fp64, build error, allocation failure). Auto
  (``use_opencl=None``) engages it only when :func:`gpu_scatter_beneficial`
  measures it >=5% faster (independent of the warp-routing calibration, which
  lost on every stack measured so far); the outcome is reported in
  ``metrics['opencl_scatter']`` = ``'cpu'|'gpu'|'unavailable'``.
* **CuPy/CUDA GPU backend (v1.7.0, ``raw_sr.core.gpu``).** Unlike the per-op
  OpenCL T-API routing above, ``SuperResolution(use_gpu=None|True|False)`` and
  ``enhance_image(..., use_gpu=...)`` keep buffers resident in VRAM across an
  entire stage (upload once, download once): drizzle scatter (35-42x faster),
  IBP + VST bilateral prox (2.6-8.2x faster), and RL-TV + starlet enhancement
  (1.5-11.5x faster), matching CPU within >=109 dB PSNR (>=132 dB end-to-end).
  Auto (``None``) calibrates once per stage and falls back to CPU on VRAM
  limits or ``OutOfMemoryError``. Reported in ``metrics['gpu_stages']``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import warnings

import numpy as np
import cv2

# BT.601 luma matrix for cv2.transform (one fused pass; see _to_gray).
_LUMA_BT601 = np.array([[0.299, 0.587, 0.114]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

@dataclass
class RawFrame:
    """A loaded, calibrated RAW frame in linear float32 RGB (values 0..1)."""
    name: str
    rgb: np.ndarray          # (H, W, 3) float32, linear, 0..1
    green: np.ndarray        # (H, W) float32, green channel (registration proxy)
    meta: dict = field(default_factory=dict)
    mosaic: Optional[np.ndarray] = None   # (H, W) float32 calibrated Bayer CFA
    pattern: Optional[np.ndarray] = None  # (2, 2) int8 Bayer pattern (0=R,1=G,2=B,3=G)


@dataclass
class SRResult:
    """Result of a super-resolution run."""
    hr: np.ndarray           # (H*r, W*r, 3) float32 linear, 0..1
    ratio: float
    shifts: List[Tuple[float, float]]
    weights: np.ndarray
    coverage: np.ndarray     # (H*r, W*r) float32, effective contributing weight
    used_frames: int
    total_frames: int
    metrics: dict = field(default_factory=dict)
    # Per-kept-frame ref->mov transform: either a (dx, dy) translation or a
    # 3x3 homography, in LR pixel coordinates, aligned to the reference frame.
    transforms: List[np.ndarray] = field(default_factory=list)
    # Alignment quality, aligned to the kept frames (see ARCHITECTURE.md).
    kept_indices: List[int] = field(default_factory=list)
    align_scores: List[float] = field(default_factory=list)
    align_residuals: List[float] = field(default_factory=list)
    # Per-kept-frame flag: True marks the reference frame's self-comparison
    # (zncc=1.0, residual=0.0 by construction). It carries no alignment
    # information and is excluded from the core's mean/max aggregations (D2).
    align_self: List[bool] = field(default_factory=list)
    # Index (into the frame list passed to ``run``) of the reference frame
    # whose LR coordinates the HR grid is aligned to: ``hr`` pixel
    # (r*x + (r-1)/2, r*y + (r-1)/2) is the reference frame's LR pixel (x, y)
    # (pixel-centre aligned, the cv2.resize convention). Anything that has to relate
    # ``hr`` to a *specific* frame — notably held-out prediction, which
    # predicts a frame that was not fused — must start from THIS frame, not
    # ``frames[0]``. ``None`` only for results built by hand.
    reference_index: Optional[int] = None
    # Optional classical (non-neural) enhanced image computed on top of ``hr``
    # (the measured linear master ``hr`` itself is never modified).
    enhanced: Optional[np.ndarray] = None
    # Optional per-frame local alignment outputs (when SuperResolution(local_align=True)).
    local_shifts: Optional[Tuple[List[np.ndarray], List[np.ndarray]]] = None
    local_weights: Optional[List[np.ndarray]] = None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_raw_linear(path: str, roi: Optional[Tuple[int, int, int, int]] = None,
                    demosaic_algorithm=None) -> RawFrame:
    """Load a RAW file to full-resolution linear float32 RGB.

    Demosaic is performed by LibRaw via rawpy (a classical, non-AI algorithm).
    Output is linear (gamma 1,1), camera white balance applied, no auto bright.

    Args:
        path: path to a RAW file (DNG/CR2/NEF/ARW...).
        roi: optional (x, y, w, h) crop applied after load.
        demosaic_algorithm: optional ``rawpy.DemosaicAlgorithm`` value. The
            default (``None``) selects :data:`rawpy.DemosaicAlgorithm.DCB`, a
            sharp classical directional demosaic. Pass any rawpy algorithm
            (e.g. ``rawpy.DemosaicAlgorithm.AHD`` or ``AMAZE``) to override.

    Returns:
        RawFrame with full-resolution linear RGB.
    """
    import rawpy

    if demosaic_algorithm is None:
        demosaic_algorithm = rawpy.DemosaicAlgorithm.DCB

    with rawpy.imread(path) as raw:
        rgb16 = raw.postprocess(
            output_bps=16,
            gamma=(1, 1),
            no_auto_bright=True,
            use_camera_wb=True,
            demosaic_algorithm=demosaic_algorithm,
        )
        meta = {
            "black_level": list(raw.black_level_per_channel)
            if raw.black_level_per_channel is not None else None,
            "white_level": int(raw.white_level) if raw.white_level is not None else None,
            "camera_whitebalance": list(raw.camera_whitebalance)
            if raw.camera_whitebalance is not None else None,
            "raw_shape": tuple(raw.raw_image.shape),
        }
        # Extract calibrated Bayer mosaic aligned with rgb16 (if dimensions match)
        vis = getattr(raw, "raw_image_visible", None)
        raw_pat = getattr(raw, "raw_pattern", None)
        pat = np.asarray(raw_pat, dtype=np.int8).copy() if raw_pat is not None else None
        raw_mos = None
        if vis is not None and pat is not None and pat.shape == (2, 2) and vis.shape == rgb16.shape[:2]:
            if roi is not None:
                rx, ry, rw, rh = roi
                raw_mos = np.ascontiguousarray(vis[ry:ry + rh, rx:rx + rw], dtype=np.float32)
                if ry % 2 or rx % 2:
                    pat = np.roll(pat, shift=(-(ry % 2), -(rx % 2)), axis=(0, 1))
            else:
                raw_mos = np.ascontiguousarray(vis, dtype=np.float32)
            black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
            white = float(raw.white_level)
            wb = np.asarray(raw.camera_whitebalance[:3], dtype=np.float32)
            for r_i in range(2):
                for c_i in range(2):
                    ci = int(pat[r_i, c_i])
                    bl = float(black[ci]) if ci < black.size else 0.0
                    sp = max(white - bl, 1.0)
                    raw_mos[r_i::2, c_i::2] = np.clip((raw_mos[r_i::2, c_i::2] - bl) / sp, 0.0, 1.0)
            if wb.size >= 3 and wb[1] > 0:
                g_wb = float(wb[1])
                gains_wb = (float(wb[0]) / g_wb, 1.0, float(wb[2]) / g_wb)
                for r_i in range(2):
                    for c_i in range(2):
                        ci = int(pat[r_i, c_i])
                        ch_i = 1 if ci in (1, 3) else (0 if ci == 0 else 2)
                        raw_mos[r_i::2, c_i::2] *= gains_wb[ch_i]

    rgb = rgb16.astype(np.float32) / 65535.0
    if roi is not None:
        x, y, w, h = roi
        # Detach the crop from the full-frame base: a plain slice is a VIEW,
        # which would pin the whole decoded frame (~190 MB on a 16 MP sensor)
        # for as long as the RawFrame lives. The copy is contiguous and
        # bit-identical to the slice.
        rgb = np.ascontiguousarray(rgb[y:y + h, x:x + w])

    # Level-calibrate raw_mos per channel to match LibRaw's linear RGB scale
    if raw_mos is not None and pat is not None and raw_mos.shape == rgb.shape[:2]:
        for c_i in range(3):
            # Gather all sites of channel c_i
            num, den = 0.0, 0.0
            for r_p in range(2):
                for c_p in range(2):
                    ci = int(pat[r_p, c_p])
                    ch_i = 1 if ci in (1, 3) else (0 if ci == 0 else 2)
                    if ch_i == c_i:
                        a = raw_mos[r_p::2, c_p::2].astype(np.float64)
                        b = rgb[r_p::2, c_p::2, c_i].astype(np.float64)
                        num += float(np.sum(a * b))
                        den += float(np.sum(a * a))
            gain = (num / den) if den > 1e-12 else 1.0
            if abs(gain - 1.0) > 1e-5:
                for r_p in range(2):
                    for c_p in range(2):
                        ci = int(pat[r_p, c_p])
                        ch_i = 1 if ci in (1, 3) else (0 if ci == 0 else 2)
                        if ch_i == c_i:
                            raw_mos[r_p::2, c_p::2] *= np.float32(gain)

    green = rgb[:, :, 1].copy()
    name = path.rsplit("/", 1)[-1]
    return RawFrame(name=name, rgb=rgb, green=green, meta=meta,
                    mosaic=raw_mos, pattern=pat)


# --------------------------------------------------------------------------- #
# Registration (subpixel)
# --------------------------------------------------------------------------- #

_HANNING_WINDOWS: dict = {}


def _hanning_window(shape: Tuple[int, int]) -> np.ndarray:
    """Phase-correlation window for ``shape``, cached per shape.

    ``cv2.createHanningWindow`` builds a full-frame float32 array (~4 MB at
    ROI 1024, ~16 MB at 2048) and phase correlation is called twice per frame
    (anchor shift + registration seed) plus once per pair inside
    :func:`_select_reference`; recomputing it every time was pure allocation.
    The array is read-only for :func:`estimate_translation` (which copies the
    window before handing it to OpenCV, so a caller can never corrupt the
    cache), and the cache is keyed by the exact shape, so every call returns
    the same bytes it always did.
    """
    key = (int(shape[0]), int(shape[1]))
    win = _HANNING_WINDOWS.get(key)
    if win is None:
        h, w = key
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        win.setflags(write=False)
        if len(_HANNING_WINDOWS) > 8:      # tiny, shape-keyed: bound it anyway
            _HANNING_WINDOWS.clear()
        _HANNING_WINDOWS[key] = win
    return win


# Minimum ``cv2.phaseCorrelate`` response (peak confidence, 0..1) accepted as a
# usable subpixel translation. The response is the normalized cross-power peak;
# a correct sub-pixel lock on real burst frames measured 0.55-1.00, while a
# failed correlation (frame 8 of the real 9-frame burst, which is dominated by
# content above the LR Nyquist) measured 0.0017. 0.1 sits ~60x above the
# observed failure and ~5x below the weakest observed good lock, so the two
# populations cannot overlap; below it the phase estimate is discarded and a
# template-match estimate is used instead (see _translation_by_template).
_PHASE_CORR_MIN_RESPONSE = 0.1

# Below this template-match peak the fallback itself has no trustworthy lock
# (no shared structure); it then returns (0, 0) rather than an arbitrary shift.
_TEMPLATE_MATCH_MIN_PEAK = 0.05


def _translation_by_template(ref: np.ndarray, mov: np.ndarray,
                             frac: float = 0.5,
                             ) -> Tuple[float, float, float]:
    """Integer+subpixel translation by template matching (phase-corr fallback).

    A central ``frac``-sized patch of ``ref`` is located in ``mov`` with
    ``cv2.matchTemplate`` (``TM_CCOEFF_NORMED``); the integer peak is refined to
    subpixel by a per-axis parabola fit on the correlation surface. Returns
    ``(dx, dy, peak)`` in the same sign convention as :func:`estimate_translation`
    (``mov(x) ~= ref(x + dx)``). Used only when phase correlation has failed, so
    the cost (a full search) is paid at most once per bad frame.
    """
    ref = np.ascontiguousarray(ref, dtype=np.float32)
    mov = np.ascontiguousarray(mov, dtype=np.float32)
    h, w = ref.shape[:2]
    th = max(8, min(h - 1, int(round(h * frac))))
    tw = max(8, min(w - 1, int(round(w * frac))))
    if th < 8 or tw < 8:
        # Too small to template-match; give up honestly.
        return 0.0, 0.0, 0.0
    ty0, tx0 = (h - th) // 2, (w - tw) // 2
    tmpl = np.ascontiguousarray(ref[ty0:ty0 + th, tx0:tx0 + tw])
    # A near-constant template/search image makes TM_CCOEFF_NORMED degenerate
    # (it returns a constant surface, whose peak is meaningless). Report no
    # structure instead of a spurious offset.
    if float(tmpl.std()) < 1e-6 or float(mov.std()) < 1e-6:
        return 0.0, 0.0, 0.0
    try:
        res = cv2.matchTemplate(np.ascontiguousarray(mov), tmpl,
                                cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return 0.0, 0.0, 0.0
    _mn, peak, _ml, maxloc = cv2.minMaxLoc(res)
    if not np.isfinite(peak):
        return 0.0, 0.0, 0.0
    px, py = int(maxloc[0]), int(maxloc[1])

    def _para(vm1: float, v0: float, vp1: float) -> float:
        den = float(vm1) - 2.0 * float(v0) + float(vp1)
        if abs(den) < 1e-9:
            return 0.0
        return 0.5 * (float(vm1) - float(vp1)) / den

    sx = sy = 0.0
    if 0 < px < res.shape[1] - 1:
        sx = _para(res[py, px - 1], res[py, px], res[py, px + 1])
    if 0 < py < res.shape[0] - 1:
        sy = _para(res[py - 1, px], res[py, px], res[py + 1, px])
    # Sign: a pattern of ref at position t lands in mov at t + (mov's shift),
    # so the located offset is the negative of the returned translation.
    dx = -((px + sx) - tx0)
    dy = -((py + sy) - ty0)
    return float(dx), float(dy), float(peak)


def estimate_translation(ref_green: np.ndarray, mov_green: np.ndarray,
                         window: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Subpixel translation of `mov` relative to `ref` via phase correlation.

    Returns (dx, dy) such that ``mov(x, y) ~= ref(x + dx, y + dy)``. To align
    ``mov`` onto ``ref`` it must be resampled as ``ref(x) = mov(x - dx)``.
    (Asserted by the synthetic test.)

    The phase-correlation response is gated: a low response means the global
    lock failed (real example: frame 8 of the 9-frame burst, response 0.0017,
    returned a bogus 22.98 px shift). Below
    :data:`_PHASE_CORR_MIN_RESPONSE` the estimate is discarded and
    :func:`_translation_by_template` supplies a template-match translation
    instead; if that also finds no structure the estimate is (0, 0). Above the
    threshold the phase result is returned unchanged, so every previously good
    measurement is byte-identical.

    Note: ``cv2.phaseCorrelate`` modifies its inputs in place, so explicit
    copies are mandatory here — otherwise a caller's frame buffers are silently
    corrupted across a burst (a real bug that once crippled this pipeline).
    """
    ref = np.array(ref_green, dtype=np.float32, copy=True)
    mov = np.array(mov_green, dtype=np.float32, copy=True)
    if window is None:
        window = _hanning_window(ref.shape)
    else:
        window = np.array(window, dtype=np.float32, copy=True)
    (dx, dy), resp = cv2.phaseCorrelate(ref, mov, window)
    if float(resp) >= _PHASE_CORR_MIN_RESPONSE:
        return -float(dx), -float(dy)
    tdx, tdy, peak = _translation_by_template(ref_green, mov_green)
    if peak >= _TEMPLATE_MATCH_MIN_PEAK:
        return tdx, tdy
    return 0.0, 0.0


def estimate_affine_ecc(ref: np.ndarray, mov: np.ndarray,
                        motion: int = cv2.MOTION_EUCLIDEAN,
                        warp: Optional[np.ndarray] = None,
                        iters: int = 30, eps: float = 1e-5) -> np.ndarray:
    """Refine a global 2x3 warp (ref -> mov) with ECC on grayscale images.

    Falls back to identity if ECC does not converge.
    """
    r = np.array(ref, dtype=np.float32, copy=True)
    m = np.array(mov, dtype=np.float32, copy=True)
    if r.ndim == 3:
        r = r.mean(axis=2)
    if m.ndim == 3:
        m = m.mean(axis=2)
    if warp is None:
        warp = np.eye(2, 3, dtype=np.float32)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
        _, warp = cv2.findTransformECC(r, m, warp.copy(), motion, crit, None, 5)
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32)
    return warp.astype(np.float32)


def _warp_from_ecc(ecc_warp: np.ndarray) -> np.ndarray:
    """Convert an ECC ref->mov 2x3 warp to a mov->ref 3x3 homography."""
    full = np.eye(3, dtype=np.float64)
    full[:2, :] = ecc_warp.astype(np.float64)
    try:
        inv = np.linalg.inv(full)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)
    return inv


def translation_matrix(dx: float, dy: float) -> np.ndarray:
    """A ref->mov 3x3 translation homography (``p_mov = p_ref + (dx, dy)``).

    Note the sign: :func:`estimate_translation` returns the *resampling*
    offset ``d`` with ``mov(x) ~= ref(x + d)``, so the ref->mov warp that
    aligns that pair is ``translation_matrix(-d[0], -d[1])``. Passing ``d``
    unchanged deposits the frame displaced by ``2 d`` — the exact bug that
    made ``registration="translation"`` ghost (B8 #1). The functions that
    consume a phase-correlation estimate for alignment must negate it
    (:func:`_register_frame`, :func:`_select_reference`,
    :func:`estimate_homography`'s seed); the ones that build a matrix for a
    *known* displacement need not (:func:`drizzle` callers, tests).
    """
    return np.array([[1.0, 0.0, float(dx)],
                     [0.0, 1.0, float(dy)],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _normalize_homography(W: np.ndarray) -> np.ndarray:
    arr = np.asarray(W, dtype=np.float64)
    if arr.shape == (3, 3):
        W_n = arr
    elif arr.shape == (2, 3):
        W_n = np.eye(3, dtype=np.float64)
        W_n[:2, :] = arr
    elif arr.shape == (2,):
        W_n = translation_matrix(arr[0], arr[1])
    else:
        raise ValueError(f"cannot interpret alignment of shape {arr.shape}")
    if abs(W_n[2, 2]) < 1e-12:
        return np.eye(3, dtype=np.float64)
    return W_n / W_n[2, 2]


def gradient_mask(img: np.ndarray, min_frac: float = 0.12,
                  dilate: int = 1) -> Optional[np.ndarray]:
    """A gradient-magnitude edge mask (uint8, 0/255) used to weight ECC/ZNCC.

    Straight edges near the frame border are exactly where a global homography
    cannot fit radial distortion and where a bare global ZNCC sees too few
    gradient pixels to notice the error. Restricting the registration score to
    the gradient-bearing pixels (and, for ECC, masking the flat/noisy regions)
    makes subpixel alignment track structure rather than flat area.

    The threshold is relative to the frame's own maximum gradient so it adapts
    to exposure/contrast. Returns ``None`` when the frame carries too little
    structure to build a mask (caller then uses the full frame).
    """
    g = np.asarray(img, dtype=np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    peak = float(mag.max())
    if peak <= 1e-12:
        return None
    mask = (mag >= float(min_frac) * peak)
    if float(mask.mean()) < 0.02:
        return None
    # Conditioning guard: an ECC fit weighted on gradients that run in only one
    # direction constrains only one axis of motion and can wander. A scene
    # dominated by a single straight edge (e.g. the synthetic ghost burst) is
    # exactly such a case, so require a reasonable balance of horizontal and
    # vertical gradient energy before trusting the mask; otherwise use the full
    # frame (which carries the other-axis information via the flat-area noise).
    ex = float(np.abs(gx[mask]).sum())
    ey = float(np.abs(gy[mask]).sum())
    if min(ex, ey) < 0.15 * max(ex, ey):
        return None
    mask = mask.astype(np.uint8)
    if dilate:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=int(dilate))
    return (mask * 255).astype(np.uint8)


def _render_in_ref_frame(ref: np.ndarray, mov: np.ndarray, W: np.ndarray):
    """Warp ``mov`` into the reference frame plus its validity map.

    ``W`` is a ref->mov transform, so ``inv(W)`` renders ``mov`` in reference
    coordinates. Both :func:`_alignment_score` and
    :func:`_alignment_residual_px` need exactly this pair of warps (the moved
    image and the coverage mask of a ones-image); each used to compute them
    itself, i.e. two identical ``warpPerspective`` calls over a full LR frame
    per frame pair were redundant (A3 #5). Computing them once and passing the
    result to both is bit-identical: same matrices, same flags, same inputs.

    ``mov`` is only read (OpenCV does not modify the source), so a float32
    input is used through a view instead of being copied again.

    Returns ``(aligned, valid)``, or ``(None, None)`` when ``W`` is singular.
    """
    try:
        Mi = np.linalg.inv(_normalize_homography(W)).astype(np.float32)
    except np.linalg.LinAlgError:
        return None, None
    h, w = ref.shape[:2]
    src = np.asarray(mov, dtype=np.float32)
    aligned = cv2.warpPerspective(src, Mi, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
    ones = np.ones((h, w), dtype=np.float32)
    valid = cv2.warpPerspective(ones, Mi, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    return aligned, valid


def _alignment_score(ref: np.ndarray, mov: np.ndarray, W: np.ndarray,
                     mask: Optional[np.ndarray] = None,
                     rendered: Optional[Tuple[np.ndarray, np.ndarray]] = None
                     ) -> float:
    """Zero-normalized cross-correlation of mov warped onto ref by W.

    W is a ref->mov transform, so ``mov(W(p)) ~= ref(p)``; to render the moved
    image in the reference frame we apply ``inv(W)``. Only pixels covered by
    the moved image participate; when ``mask`` is given (uint8, nonzero =
    usable, in *reference* coordinates) only masked pixels count. Returns -1.0
    when the overlap is too small.

    ``rendered`` optionally supplies the ``(aligned, valid)`` pair from
    :func:`_render_in_ref_frame` (already computed for the residual
    measurement), which skips the duplicate warps.
    """
    if rendered is None:
        aligned, valid = _render_in_ref_frame(ref, mov, W)
        if aligned is None:
            return -1.0
    else:
        aligned, valid = rendered
        if aligned is None:
            return -1.0
    h, w = ref.shape[:2]
    sel = valid > 0.99
    if mask is not None:
        sel = sel & (mask > 0)
    if int(sel.sum()) < max(100, int(0.05 * sel.size)):
        return -1.0
    a = ref[sel].astype(np.float64)
    b = aligned[sel].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum())) + 1e-12
    return float((a * b).sum() / denom)


def _alignment_residual_px(ref: np.ndarray, mov: np.ndarray,
                           W: np.ndarray,
                           rendered: Optional[Tuple[np.ndarray, np.ndarray]] = None
                           ) -> float:
    """Residual misalignment (LR px) left after warping ``mov`` onto ``ref``.

    ``W`` is the ref->mov transform. We render ``mov`` in the reference frame
    and run one more subpixel phase correlation between the reference and that
    render, restricted to the valid overlap. The magnitude of the leftover
    shift is the residual: ~0 px means the frames are aligned, large values
    mean the model could not explain the actual motion. Returns a large value
    when the overlap is too small to measure.

    ``rendered`` optionally supplies the ``(aligned, valid)`` pair from
    :func:`_render_in_ref_frame` (the score is measured on the same pair).

    The measurement covers the whole valid overlap, including structureless
    regions where the phase correlation has no peak (the reference burst's sky:
    response ~0.05). The value therefore depends on the ROI's content mix as
    well as on the fit — the same transform reads 0.03 px on a 1024 crop and
    0.06-0.17 px inside its 512 sub-windows, and a 512 crop that is ~96 % sky
    reads 0.59 px for a fit only ~1.3x worse in ``mean|diff|``. Treat it as a
    gross-misalignment detector, not a sub-pixel quality score; the
    measurements are in docs/REGISTRATION_RESIDUAL.md.
    """
    if rendered is None:
        aligned, valid = _render_in_ref_frame(ref, mov, W)
        if aligned is None:
            return float("nan")
    else:
        aligned, valid = rendered
        if aligned is None:
            return float("nan")
    h, w = ref.shape[:2]
    mask = (valid > 0.99).astype(np.float32)
    if float(mask.sum()) < max(100.0, 0.1 * mask.size):
        # Too little overlap to measure: report the frame diagonal as "far off"
        # rather than an unbounded value.
        return float(np.hypot(h, w))
    # Zero the invalid border on both images so it cannot bias the correlation.
    a = (np.asarray(ref, dtype=np.float32) - float(ref[mask > 0].mean())) * mask
    b = (np.asarray(aligned, dtype=np.float32)
         - float(aligned[mask > 0].mean())) * mask
    try:
        dx, dy = estimate_translation(a, b)
    except cv2.error:
        return float(np.hypot(h, w))
    return float(np.hypot(dx, dy))


def _corner_perspective_px(W: np.ndarray, shape: Tuple[int, int]) -> float:
    """How far the perspective terms move the image corners, in LR pixels.

    Compares ``W`` against its own affine part at the four image corners. A
    near-frontal handheld frame has perspective terms that are only weakly
    determined; measured in pixels their effect is tiny, yet left unregularised
    they extrapolate wildly at the frame border — the ghosting the user sees.
    The corner excursion is the natural scale on which to decide whether the
    perspective terms are meaningful.
    """
    W = _normalize_homography(W)
    A = W.copy()
    A[2, 0] = 0.0
    A[2, 1] = 0.0
    h, w = shape
    pts = np.array([[0.0, 0.0, 1.0], [w - 1.0, 0.0, 1.0],
                    [0.0, h - 1.0, 1.0], [w - 1.0, h - 1.0, 1.0]]).T

    def _apply(M):
        p = M @ pts
        return (p[:2] / p[2]).T

    return float(np.max(np.linalg.norm(_apply(A) - _apply(W), axis=1)))


def _ecc_warp(ref: np.ndarray, mov: np.ndarray, init, motion: int,
              scale: float, mask: Optional[np.ndarray],
              iters: int, eps: float) -> Optional[np.ndarray]:
    """One ECC solve (translation/affine/homography) at a working scale.

    ``init`` is the starting warp in LR coordinates (a (dx, dy) translation, a
    2x3 affine matrix, or a 3x3 homography). ``mask`` is a uint8 input mask in
    LR mov-image coordinates: the ECC score ignores zero pixels, which both
    focuses the fit on edges and discards invalid/flat borders. Returns a 3x3
    warp (affine rows padded) with ``W[2,2]==1``, or None on failure.
    """
    r = np.array(ref, dtype=np.float32, copy=True)
    m = np.array(mov, dtype=np.float32, copy=True)
    if r.ndim == 3:
        r = r.mean(axis=2)
    if m.ndim == 3:
        m = m.mean(axis=2)

    working_mask = mask
    if scale and abs(scale - 1.0) > 1e-9:
        r = cv2.resize(r, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if mask is not None:
            working_mask = cv2.resize(mask, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_NEAREST)
        arr = np.asarray(init, dtype=np.float64)
        if arr.shape == (3, 3):
            I = _normalize_homography(arr)
        elif arr.ndim == 1:
            I = translation_matrix(arr[0], arr[1])
        elif arr.shape == (2, 3):
            I = np.vstack([arr, [0.0, 0.0, 1.0]])
        else:
            I = _normalize_homography(arr)
        # init expressed in the scaled coordinate system: u = scale * q, so a
        # warp W in LR maps to S @ W @ S^-1 in scaled coordinates.
        S = np.diag([scale, scale, 1.0])
        Sin = np.diag([1.0 / scale, 1.0 / scale, 1.0])
        I_scaled = S @ I @ Sin
        if motion == cv2.MOTION_HOMOGRAPHY:
            Wm = I_scaled.astype(np.float32)
        else:
            Wm = I_scaled[:2, :].astype(np.float32)
    else:
        if motion == cv2.MOTION_HOMOGRAPHY:
            Wm = _normalize_homography(init).astype(np.float32)
        else:
            I = np.asarray(init, dtype=np.float64)
            if I.shape == (3, 3):
                Wm = I[:2, :].astype(np.float32).copy()
            elif I.ndim == 1:
                Wm = translation_matrix(I[0], I[1])[:2, :].astype(np.float32)
            else:
                Wm = I[:2, :].astype(np.float32).copy()

    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    try:
        _, W = cv2.findTransformECC(r, m, Wm, motion, crit, working_mask, 5)
    except cv2.error:
        return None
    W = np.asarray(W, dtype=np.float64)
    if W.shape == (2, 3):
        W = np.vstack([W, [0.0, 0.0, 1.0]])
    W = _normalize_homography(W)
    if scale and abs(scale - 1.0) > 1e-9:
        # Conjugate the scaled-coordinate warp back to LR pixels.
        Sin = np.diag([1.0 / scale, 1.0 / scale, 1.0])
        S = np.diag([scale, scale, 1.0])
        W = _normalize_homography(Sin @ W @ S)
    return W


def estimate_homography(ref_green: np.ndarray, mov_green: np.ndarray,
                        ecc_scale: float = 0.5, iters: int = 100,
                        eps: float = 1e-6, edge_weighted: bool = True,
                        perspective_tol: float = 0.5,
                        persp_score_margin: float = 0.002,
                        prefer_affine: bool = True,
                        seed: Optional[Tuple[float, float]] = None,
                        ref_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Global ref->mov 3x3 warp (ECC, init from phase correlation).

    Real handheld bursts are not pure translation: rotation/perspective makes
    translation-only registration leave residual misalignment that caps the
    recoverable resolution. This estimator:

    1. seeds with the phase-correlation translation (negated: the estimate is
       a mov->ref resampling offset, ECC works in ref->mov coordinates);
    2. runs ECC at a low scale (``ecc_scale``) and **always refines at full
       scale 1.0** from that result for the affine fit. The homography is then
       refined at full scale from the *refined affine* solution: it is the
       better-conditioned starting point and it saves the half-scale homography
       pass. On frames below ``_REG_AFFINE_WARMSTART_MIN_PX`` (ROI 512) the
       original route is kept — half-scale homography, full-scale refinement
       and an independent full-scale solve from the seed — because there the
       seed-based solve still wins on 3 of 8 real frames. Whichever route runs,
       a full-scale homography started from the seed remains as the fallback
       when the warm result is missing or scores more than 0.05 below the best
       affine candidate;
    3. weights the fit (``inputMask``) by gradient magnitude, so the score
       tracks straight edges instead of flat area (flat regions carry no
       alignment information and can be dominated by noise);
    4. selects the model by data — see :func:`_corner_perspective_px`:
       **the homography is kept only if its perspective terms move the image
       corners by more than ``perspective_tol`` LR px *and* they improve the
       edge-weighted ZNCC by more than ``persp_score_margin``; otherwise the
       affine fit is returned.** A weakly determined perspective term
       extrapolates badly at the border and produces exactly the parallel ghost
       lines the user reported, while an affine model (rotation + shear + scale
       + translation) is well conditioned for a near-frontal burst.
    5. never returns a warp worse (by edge-weighted ZNCC) than the translation
       seed or the discarded model.

    ``prefer_affine=False`` disables the affine fallback (step 4) and
    ``edge_weighted=False`` disables the gradient mask (step 3); both together
    reproduce the legacy full-homography behaviour for A/B measurement.

    ``ecc_scale=0.5`` remains the default: the low-scale pass locks the global
    motion robustly and cheaply, and the mandatory full-scale refinement then
    delivers sub-pixel precision. Returns ``W`` with ``W[2,2]==1`` such that
    ``mov(W(p)) ~= ref(p)``.

    ``seed`` (optional, a ``(dx, dy)`` phase-correlation estimate for THIS
    ref/mov pair) skips the internal :func:`estimate_translation` call. The
    caller in :func:`_register_frame` measures that translation anyway, so
    passing it removes one full-frame phase correlation per frame without
    changing the fit — the estimator starts from exactly the same matrix.
    ``ref_mask`` (optional) is :func:`gradient_mask` of ``ref``, which is the
    same array for every frame of a burst; supplying it saves recomputing a
    full-frame Sobel/magnitude/dilate per frame.
    """
    # A float32, C-contiguous input is used through a view: nothing below
    # mutates these images (ECC and phase correlation copy their inputs), so
    # the defensive full-frame copies this function used to make were pure
    # bandwidth on the registration hot path.
    ref = np.ascontiguousarray(ref_green, dtype=np.float32)
    mov = np.ascontiguousarray(mov_green, dtype=np.float32)
    if seed is None:
        dx, dy = estimate_translation(ref, mov)
    else:
        sd = np.asarray(seed, dtype=np.float64).ravel()
        dx, dy = float(sd[0]), float(sd[1])
    # ``estimate_translation`` returns the resampling offset (mov(x) ~=
    # ref(x + dx)) while ECC's warp is ref->mov, so the seed must be its
    # NEGATIVE (B8, fixed 2026-11-20). Measured on 96x96 ground truth: the
    # unnegated seed left ECC in the mirror/no-lock solution (shift
    # (1.3,-0.7): ZNCC -0.02, residual 2.9 px; shift (3.1,2.2): 7.4 px), and
    # on content above the LR Nyquist it failed at every size tried (scores
    # -0.05..0.01 at 96, 192 and 256 px). With the negated seed the same pairs
    # score 0.80-0.94 and land at 0.03-0.07 px. Real 9-frame burst: ROI 1024
    # agrees to 1e-6 px, ROI 512 mean residual 0.593 vs 0.598 px. Because the
    # fit changes, the ghost gate's D2 population statistic lost its compact
    # blob on two moving-subject fixtures; a registration-residual branch (D4,
    # see _GHOST_GATE_RESIDUAL_MAX) now carries those bursts and the N=2
    # flooded case is pinned xfail(strict) in the B2 tests as a follow-up.
    dx, dy = -dx, -dy
    W_t = translation_matrix(dx, dy)

    mask = gradient_mask(mov) if edge_weighted else None
    if ref_mask is None:
        ref_mask = gradient_mask(ref) if edge_weighted else None
    if mask is not None and ref_mask is not None:
        # Weight edges present in the *reference* too (the mask lives in mov
        # coordinates after the seed translation; the seed is sub-pixel, so a
        # direct union is a good approximation for a small motion).
        T = translation_matrix(dx, dy)
        shifted = cv2.warpPerspective(ref_mask, np.linalg.inv(T).astype(np.float32),
                                      (mov.shape[1], mov.shape[0]),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.bitwise_or(mask, shifted)

    def score(W):
        # Memoized: the selection steps below re-score the same candidate
        # matrices (max over the A/H candidate lists, the best-tracking loop,
        # and the final preference check) — 2-3 redundant warp+ZNCC passes per
        # candidate. Same object/bytes => deterministic, bit-identical result.
        key = np.ascontiguousarray(W).tobytes()
        cached = _score_cache.get(key)
        if cached is None:
            cached = _alignment_score(ref, mov, W, mask)
            _score_cache[key] = cached
        return cached

    _score_cache: dict = {}

    best, best_score = W_t, score(W_t)
    low = float(ecc_scale) if ecc_scale else 1.0

    # Affine fit: a low-scale pass locks the global motion, then a full-scale
    # pass refines it; a full-scale pass straight from the translation seed is
    # also kept, because on a texture-rich frame the low-scale pass can wander
    # and would otherwise poison the refinement.
    A_cands = []
    A_low = _ecc_warp(ref, mov, (dx, dy), cv2.MOTION_AFFINE, low, mask,
                      iters, eps)
    if A_low is not None:
        A_ref = _ecc_warp(ref, mov, A_low, cv2.MOTION_AFFINE, 1.0, mask,
                          iters, eps)
        A_cands.append(A_ref if A_ref is not None else A_low)
    A_full = _ecc_warp(ref, mov, (dx, dy), cv2.MOTION_AFFINE, 1.0, mask,
                       iters, eps)
    if A_full is not None:
        A_cands.append(A_full)
    A = max(A_cands, key=score) if A_cands else None

    # Homography: two coarse-to-fine routes, chosen by frame size.
    #
    # Below ``_REG_AFFINE_WARMSTART_MIN_PX`` (ROI 512) the original route is kept
    # verbatim — half-scale homography, full-scale refinement, and an
    # independent full-scale solve from the phase-correlation seed — because
    # there the seed-based solve still wins the selection on 3 of 8 real frames
    # (up to 0.51 px corner distance), so the legacy route is what reproduces
    # the existing output bit for bit.
    #
    # At or above it, the homography is warm-started from the *refined affine*
    # solution instead of from a half-scale homography solve. That drops one
    # solve (the coarse homography) and reaches the same optimum: measured on
    # the real burst at ROI 1024, <=0.004 px maximum corner displacement from
    # the legacy route (ecc_scale 0.25 and 0.5), score and residual unchanged
    # to 1e-4. The seed-based full-scale solve remains as the fallback for a
    # warm result that is missing or unconvincing, so a scene the affine model
    # cannot initialise still gets an independent fit.
    warm_homography = min(ref.shape[:2]) >= _REG_AFFINE_WARMSTART_MIN_PX
    H_cands = []
    if warm_homography:
        if A is not None:
            s = score(A)
            if s > best_score:
                best, best_score = A, s
        H_warm = _ecc_warp(ref, mov, A if A is not None else (dx, dy),
                           cv2.MOTION_HOMOGRAPHY, 1.0, mask, iters, eps)
        if H_warm is not None:
            H_cands.append(H_warm)
        if H_warm is None or score(H_warm) < best_score - 0.05:
            H_full = _ecc_warp(ref, mov, (dx, dy), cv2.MOTION_HOMOGRAPHY,
                               1.0, mask, iters, eps)
            if H_full is not None:
                H_cands.append(H_full)
    else:
        H_low = _ecc_warp(ref, mov, (dx, dy), cv2.MOTION_HOMOGRAPHY, low,
                          mask, iters, eps)
        if H_low is not None and score(H_low) > best_score - 0.05:
            H_ref = _ecc_warp(ref, mov, H_low, cv2.MOTION_HOMOGRAPHY, 1.0,
                              mask, iters, eps)
            H_cands.append(H_ref if H_ref is not None else H_low)
        H_full = _ecc_warp(ref, mov, (dx, dy), cv2.MOTION_HOMOGRAPHY, 1.0,
                           mask, iters, eps)
        if H_full is not None:
            H_cands.append(H_full)
    H = max(H_cands, key=score) if H_cands else None

    for cand in (A, H):
        if cand is None:
            continue
        s = score(cand)
        if s > best_score:
            best, best_score = cand, s

    # Prefer the affine model unless perspective is both meaningful (moves the
    # corners) and measurably better on edges. This is the robustness rule that
    # stops a weakly determined homography from extrapolating at the border.
    if prefer_affine and H is not None and A is not None:
        dev = _corner_perspective_px(H, ref.shape)
        preferred = H if (dev > perspective_tol
                          and (score(H) - score(A)) > persp_score_margin) else A
    elif H is not None:
        preferred = H
    elif A is not None:
        preferred = A
    else:
        preferred = W_t
    # Never return a model worse than the best already measured.
    if score(preferred) < best_score:
        preferred = best
    return _normalize_homography(preferred)


# --------------------------------------------------------------------------- #
# Parallel registration (large frames)
# --------------------------------------------------------------------------- #
#
# Per-frame registration is embarrassingly parallel and keyed by frame index,
# so a process pool changes nothing about *which* transform belongs to *which*
# frame; results are collected BY INDEX and therefore bit-identical to the
# serial loop. Measured on the real burst: one frame's ECC fit costs ~4.3 s
# wall *regardless of the cv2 thread count* (1, 4 and all-core runs are equal
# and bit-identical) — the fit is one long serial, memory-latency-bound
# stretch inside each frame. Frames are independent, so the pool removes those
# stretches from the loop wall; but because the stretches saturate memory
# bandwidth when overlapped, the gain plateaus early: at ROI 2048 the
# registration loop takes 41.1 s serial vs 34.1/33.2/36.3 s with 4/3/8 workers
# (1.2x wall, CPU cost ~2x) — a real but modest win, flat from ~3 workers on
# (the machine's shared L3/DRAM, not its cores, set the ceiling). At small
# ROIs the per-frame work (~0.15 s) is comparable to pool spawn (~0.5 s), so
# the auto mode keeps small frames on the serial path
# (threshold: ``_REG_POOL_MIN_REF_PIXELS``, calibrated by measurement).
#
# Determinism facts this rests on (measured on real ROI-2048 frames):
# :func:`estimate_homography` and the translation/score/residual helpers return
# bit-identical matrices run-to-run and across ``cv2`` thread counts, so
# single-threaded workers produce exactly the matrices the serial path does.
#
# The pool uses the 'spawn' start method on purpose: rawpy's LibRaw decoding
# deadlocks a forked child, and 'spawn' children import modules fresh (the SR
# core imports nothing heavy that is not already imported here). Workers pin
# ``cv2.setNumThreads(1)`` so N workers never oversubscribe the machine, and
# mirror the parent's OpenCL routing state so their numerics match the serial
# path exactly.

# Minimum reference-frame pixel count for the auto pool: below it, per-frame
# registration is cheaper than spawning the pool, so stay serial. Measured on
# the real burst (serial vs 4-worker pool wall, k=2..8 all flat within +-2 s):
# ROI 512 (0.26M px) 1.54 vs 1.51 s — a wash; ROI 1024 (1.05M px) 8.2 vs
# 7.1 s — 1.14x; ROI 2048 (4.19M px) 41.1 vs 34.1 s — 1.21x. The gate sits
# between the 512 wash and the 1024 win. See the v1.4.0 performance notes in
# the module docstring.
_REG_POOL_MIN_REF_PIXELS = 600_000

# Upper bound on auto-pool workers. The fit is memory-bound, so wall plateaus
# from ~3 workers on (measured 33-38 s for k=2..8 at ROI 2048) while consumed
# CPU grows ~linearly with k; 4 workers take the whole plateau at a third of
# the CPU of 8 and leave cores free for the parent and the desktop.
_REG_POOL_MAX_WORKERS = 4

# Short-side frame size from which the homography is warm-started from the
# refined affine solution instead of running a half-scale homography pass of its
# own (see estimate_homography). Measured on the real burst at ROI 1024 (both
# ecc_scale 0.25 and 0.5): <=0.004 px maximum corner displacement from the
# legacy route, score/residual within 1e-4, one fewer ECC solve per frame. At
# ROI 512 the legacy route is kept verbatim: there the seed-based full-scale
# solve still wins on 3 of 8 frames (up to 0.51 px), so small frames stay
# bit-identical to the pre-change code. The threshold is the measured boundary.
_REG_AFFINE_WARMSTART_MIN_PX = 1024


# Worker-side state for the shared registration frame block (see
# _SharedRegFrames): set by _spawn_worker_init in each pool worker, left None in
# the parent so a bare array payload still works when the worker is driven
# directly.
_SH_REG = None
_SH_REG_SPECS: Optional[List] = None


def _spawn_worker_init(shm_name: Optional[str] = None,
                       specs: Optional[Sequence] = None) -> None:
    """Pool-worker bootstrap: one OpenCV thread per worker.

    The parent may run OpenCV multi-threaded; workers must not, or N workers
    oversubscribe N cores (that oversubscription — not the pool itself — is
    what made an earlier registration-pool experiment lose).

    When ``shm_name`` is given the worker also attaches the registration
    phase's shared frame block (see :class:`_SharedRegFrames`), so pool tasks
    carry slot indices rather than 3 full float32 frames each.
    """
    cv2.setNumThreads(1)
    global _SH_REG, _SH_REG_SPECS
    if shm_name is not None:
        from multiprocessing import shared_memory as _shm
        _SH_REG = _shm.SharedMemory(name=shm_name)
        _SH_REG_SPECS = list(specs or ())


class _SharedRegFrames:
    """One shared-memory block holding the arrays the registration pool needs.

    A registration task used to carry ``anchor_green``, ``ref_green`` and its
    own frame in its payload. With the 'spawn' start method (mandatory, see
    above) every payload is pickled, so that was ~12 MB of IPC per task at
    ROI 1024 and ~48 MB at ROI 2048 — ~430 MB for a 9-frame burst (A3 #5).
    The arrays are immutable for the whole registration phase, so they are
    copied **once** into a shared block and each worker attaches by name.

    Fork-inherited globals would be cheaper still, but 'spawn' is required
    (rawpy's LibRaw deadlocks a forked child), so the name is passed through
    the pool initializer instead. Attach is read-only and by slot index, and
    the block is unlinked by the parent as soon as the pool is joined.
    """

    def __init__(self, arrays: Sequence[np.ndarray]):
        from multiprocessing import shared_memory as _shm
        self._mod = _shm
        self._index: dict = {}
        unique: List[np.ndarray] = []
        for a in arrays:
            key = id(a)
            if key not in self._index:
                self._index[key] = len(unique)
                unique.append(np.ascontiguousarray(a, dtype=np.float32))
        total = int(sum(int(a.nbytes) for a in unique))
        self._shm = _shm.SharedMemory(create=True, size=max(total, 1))
        off = 0
        specs = []
        for a in unique:
            self._shm.buf[off:off + int(a.nbytes)] = memoryview(
                a.reshape(-1)).cast("B")
            specs.append((off, int(a.shape[0]), int(a.shape[1])))
            off += int(a.nbytes)
        self.specs = specs
        self.name = self._shm.name

    def slot(self, arr: np.ndarray) -> int:
        """Slot index of ``arr`` (by identity; copies are not deduplicated)."""
        return self._index[id(arr)]

    def close(self) -> None:
        try:
            self._shm.close()
        finally:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


def _reg_array(slot: int) -> np.ndarray:
    """Read-only view of one slot of the attached shared frame block."""
    off, h, w = _SH_REG_SPECS[int(slot)]
    return np.ndarray((int(h), int(w)), dtype=np.float32,
                      buffer=_SH_REG.buf, offset=int(off))


def _registration_worker(payload: Tuple) -> Tuple[int, Tuple]:
    """Pool entry point: :func:`_register_frame` with the worker's OpenCL
    routing state mirrored from the parent (so the worker's cv2 behaves
    exactly like the parent's during :meth:`SuperResolution._run_inner`).

    With a shared frame block attached the three images are slot indices; a
    bare ``(idx, anchor, ref, mov, ...)`` payload (arrays) is still accepted so
    the worker can be driven directly, e.g. from a test or a script.
    """
    (idx, anchor, ref, mov, is_ref, registration, ecc_scale,
     robust, ocl_active) = payload
    try:
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(bool(ocl_active))
    except cv2.error:
        pass
    if _SH_REG_SPECS is not None:
        anchor, ref, mov = _reg_array(anchor), _reg_array(ref), _reg_array(mov)
    return idx, _register_frame(anchor, ref, mov, is_ref, registration,
                                ecc_scale, robust)


def _map_in_processes(fn, payloads: Sequence, workers: int,
                      initargs: Sequence = ()) -> List:
    """``fn`` over ``payloads`` in ``workers`` spawned processes.

    Each payload's result carries its own index; the output list is assembled
    by index, so it equals ``[fn(p) for p in payloads]`` exactly, independent
    of worker scheduling. Raises on any pool failure — the caller decides the
    serial fallback, so a machine that cannot host child processes only loses
    the speedup, never the run.

    ``initargs`` are handed to :func:`_spawn_worker_init` once per worker (used
    to attach the shared registration frame block).
    """
    import multiprocessing as _mp
    from concurrent.futures import ProcessPoolExecutor

    ctx = _mp.get_context("spawn")
    out: List = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=ctx,
                             initializer=_spawn_worker_init,
                             initargs=tuple(initargs)) as ex:
        for idx, res in ex.map(fn, payloads, chunksize=1):
            out[int(idx)] = res
    if any(r is None for r in out):  # should not happen; guard anyway
        raise RuntimeError("registration pool returned missing results")
    return out


def _register_frame(anchor_green: np.ndarray,
                    ref_green: np.ndarray,
                    mov_green: np.ndarray,
                    is_ref: bool,
                    registration: str,
                    ecc_scale: float,
                    edge_weighted: bool,
                    ref_mask: Optional[np.ndarray] = None) -> Tuple:
    """All per-frame registration measurements for one frame.

    Returns ``(dx, dy, W, score, residual)``: the anchor-relative shift (phase
    correlation to frame 0, for reporting), the ref->mov transform, its
    edge-weighted ZNCC alignment score and its residual misalignment in px.
    Shared verbatim by the serial loop and the pool worker, so both paths run
    the same code (and, with the bit-identical ECC measured above, produce the
    same bytes).

    Two pieces of work the previous version repeated are removed here, both
    without changing a single output bit (A3 #5):

    * the ref->mov phase correlation is measured **once** and handed to
      :func:`estimate_homography` as its seed, instead of that function
      measuring the identical translation again;
    * the score and the residual are two statistics of the *same* rendered
      pair (moved image + coverage mask), so the render happens once instead
      of twice. Only the anchor-relative translation is a different pair and
      still needs its own phase correlation.

    ``ref_mask`` is the reference frame's gradient mask, recomputed per frame
    before; it is identical for every frame of a burst.
    """
    dx, dy = estimate_translation(anchor_green, mov_green)
    if is_ref:
        return dx, dy, np.eye(3, dtype=np.float64), 1.0, 0.0
    if registration == "translation":
        rdx, rdy = estimate_translation(ref_green, mov_green)
        # estimate_translation's offset is a mov->ref resampling offset; the
        # alignment matrix is ref->mov, so it carries the opposite sign (B8 #1).
        W = translation_matrix(-rdx, -rdy)
    else:
        rdx, rdy = estimate_translation(ref_green, mov_green)
        W = estimate_homography(ref_green, mov_green,
                                ecc_scale=ecc_scale,
                                edge_weighted=edge_weighted,
                                prefer_affine=edge_weighted,
                                seed=(rdx, rdy),
                                ref_mask=ref_mask)
    rendered = _render_in_ref_frame(ref_green, mov_green, W)
    score = _alignment_score(ref_green, mov_green, W, rendered=rendered)
    residual = _alignment_residual_px(ref_green, mov_green, W,
                                      rendered=rendered)
    return dx, dy, W, score, residual


# --------------------------------------------------------------------------- #
# Fusion / drizzle
# --------------------------------------------------------------------------- #

def _quality_weight(green: np.ndarray) -> float:
    """Sharpness proxy weight (variance of Laplacian) for a frame."""
    g = green.astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var()) + 1e-6


# Bound on how far a frame's fusion weight may sit from the burst median
# (both directions). A factor of 4 in weight is 2x in noise sigma; beyond that
# the frame is better dropped than averaged, and an unclipped weight would let
# one pathological frame dominate the deposit.
FUSION_WEIGHT_CLIP = 4.0

# High-pass scale for the per-frame read-noise estimate (pixels, LR data).
_FUSION_HP_SIGMA = 1.5

# Below this sigma a frame is treated as noise-free (synthetic data): only its
# sharpness is weighed, instead of dividing by a vanishing estimate.
_FUSION_NOISE_FLOOR = 1e-6


def _noise_variance(green: np.ndarray) -> float:
    """Robust per-frame read-noise variance from a high-pass residual.

    ``green`` is calibrated linear data (0..1), so the estimate is in the same
    units as the signal and is comparable across frames of one burst. The
    high-pass (``g - blur(g)``) removes scene structure; the MAD of what is
    left rejects the share of real edges, which a plain standard deviation
    would count as noise. Returns ``0.0`` for a perfectly flat frame.
    """
    g = np.asarray(green, dtype=np.float32)
    if g.ndim == 3:
        g = g[..., 1]
    hp = g - cv2.GaussianBlur(g, (0, 0), _FUSION_HP_SIGMA)
    med = float(np.median(hp))
    sigma = 1.4826 * float(np.median(np.abs(hp - med)))
    return sigma * sigma


def _clip_noise_correction(noise_vars: np.ndarray) -> np.ndarray:
    """Floor per-frame noise variances at the burst's quiet quartile.

    The weight divides by a noise variance estimated from each frame's own
    high-frequency tail. That estimator can only overstate noise: on a frame
    whose scene detail reaches the pixel scale (a sharp lens, or a synthetic
    pattern), scene energy contaminates the estimate upward, and a soft frame —
    whose high frequencies are gone — can look deceptively quiet.
    Rewarding such a frame contradicts the sharpness term, so estimates below
    the 25th percentile of the burst are raised to it (frames that cannot be
    shown quieter than the burst's clean quartile are treated as equal). The
    upper side is left uncapped: a genuinely noisier frame must keep its full
    penalty, and the final weight clip bounds how far that can go.
    """
    if not noise_vars.size:
        return noise_vars
    floor = float(np.percentile(noise_vars, 25))
    if floor <= 0.0:
        return noise_vars
    return np.maximum(noise_vars, floor)


def _hf_sharpness(green: np.ndarray) -> float:
    """Scene acutance of one frame: Laplacian variance of a lightly denoised copy.

    Light Gaussian denoising (sigma 1) suppresses pixel-scale noise so this
    measures scene edges rather than the frame's noise; it is the numerator of
    :func:`_fusion_weight`.
    """
    g = np.asarray(green, dtype=np.float32)
    if g.ndim == 3:
        g = g[..., 1]
    return float(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0),
                               cv2.CV_32F).var()) + 1e-6


def _fusion_weight(green: np.ndarray,
                   noise_var: Optional[float] = None) -> float:
    """Sharpness per unit noise variance: denoised Laplacian / noise variance.

    Fusion should average measurements by their information content. The
    Laplacian-variance sharpness proxy used by the blur gate also rises with
    sensor noise, so using it directly as a fusion weight over-weights the
    *noisier* frame of a mixed-ISO / mixed-exposure burst. Two changes make it
    a weighting quantity without losing what it was good at:

    * the sharpness is measured on a lightly denoised copy
      (``GaussianBlur`` sigma 1), so the numerator reflects scene acutance
      rather than the frame's noise; and
    * it is divided by the robust read-noise variance
      (:func:`_noise_variance` from a high-pass MAD), so a frame's weight is
      its information per unit noise.

    A noisy frame loses its noise-driven "sharpness", a defocused frame keeps
    its lower signal share, and a clean sharp frame keeps the full weight. The
    smooth numerator also keeps the estimate stable on low-texture bursts,
    where subtracting the noise share from the raw high-pass variance was
    measured to vary ~9x among equally noisy frames.

    ``noise_var`` overrides the per-frame estimate; :meth:`SuperResolution.run`
    passes the burst-floored value (:func:`_clip_noise_correction`) so one
    frame cannot be rewarded for a scene-limited noise estimate, and clips the
    final weights to ``FUSION_WEIGHT_CLIP`` x the median. Measured on the real
    9-frame burst, the resulting spread is 1.17 (old Laplacian weights: 1.09)
    and the held-out gain changes by <0.01 dB; on a burst whose odd frames
    carry 10x the read noise it recovers ~0.6/0.8 dB at 2x/4x.
    """
    lap = _hf_sharpness(green)
    var = _noise_variance(green) if noise_var is None else float(noise_var)
    if var <= _FUSION_NOISE_FLOOR * _FUSION_NOISE_FLOOR:
        # Noise-free synthetic data: no noise to discount, keep sharpness.
        return lap
    return lap / var


def _select_reference(frames: Sequence[RawFrame],
                      weights: np.ndarray,
                      candidates: Sequence[int],
                      consistency_min: float = 0.2,
                      stats: Optional[dict] = None) -> int:
    """Pick the sharpest frame near the burst centroid.

    Why not frame 0: the reference is the coordinate frame every other frame is
    warped into, so a reference far from the burst centroid forces large
    displacements on the frames at the other extreme, and a motion model fitted
    at the centre extrapolates worst at the border. A central reference keeps
    every displacement small. Why not simply the sharpest frame: on a burst the
    sharpest frame can sit at one end, and the sharpness proxy is fooled by
    noise. So we take the sharpest frame *among those close to the motion
    centroid*, using the sharpness (Laplacian) weight — the *gate* quantity,
    which is what "sharpest" means — with a consistency gate (ZNCC to the
    central frame) that rejects junk frames whose Laplacian variance is pure
    noise. (Fusion itself uses the inverse-noise weight,
    :func:`_fusion_weight`, from the same principle in the other direction.)

    Positions are measured relative to the temporally middle candidate via
    phase correlation (cheap, sub-pixel). The centroid of those positions is
    the burst centre; proximity is a Gaussian on the distance to it, narrowed
    to the median spread so off-centre frames are genuinely penalised.
    """
    cand = [int(i) for i in candidates]
    if len(cand) <= 1:
        return cand[0]

    mid = cand[len(cand) // 2]
    pos = np.zeros((len(cand), 2), dtype=np.float64)
    consistency = np.zeros(len(cand), dtype=np.float64)
    for j, i in enumerate(cand):
        if i == mid:
            consistency[j] = 1.0
            continue
        try:
            dx, dy = estimate_translation(frames[mid].green, frames[i].green)
        except cv2.error:
            dx, dy = 0.0, 0.0
        pos[j] = (dx, dy)
        # estimate_translation returns the resampling offset; the ref->mov
        # matrix the score renders with is its negative (B8 #2: the unnegated
        # form scored a >=2 px shifted frame ~0.01-0.2 and silently disqualified
        # it as a reference candidate).
        s = _alignment_score(frames[mid].green, frames[i].green,
                             translation_matrix(-dx, -dy))
        consistency[j] = s if np.isfinite(s) else -1.0

    centroid = pos.mean(axis=0)
    d = np.linalg.norm(pos - centroid, axis=1)
    scale = max(float(np.median(d)) * 0.75, 0.25)
    prox = np.exp(-(d * d) / (2.0 * scale * scale))

    w = np.asarray([weights[i] for i in cand], dtype=np.float64)
    med = float(np.median(w[w > 0])) if np.any(w > 0) else 1.0
    if med > 0:
        w = np.clip(w / med, 0.0, 4.0)
    w = w / (w.sum() + 1e-12)

    ok = consistency >= consistency_min
    rejected = [int(cand[j]) for j in np.flatnonzero(~ok)]
    if not np.any(ok):
        ok = np.ones(len(cand), dtype=bool)
    score = w * prox
    score[~ok] = -1.0
    if stats is not None:
        stats["rejected"] = rejected
        stats["scores"] = [round(float(s), 4) for s in consistency]
        stats["candidates"] = [int(c) for c in cand]
    return cand[int(np.argmax(score))]


# Std (in HR px) of the sensor's per-LR-pixel integration window per unit of
# ratio: a uniform window of width ``ratio`` HR px has sigma = ratio/sqrt(12).
# This is the forward-model PSF used by :func:`iterative_back_projection` and by
# the held-out prediction (:func:`predict_lr_from_hr`), so "an HR image that
# could have produced this LR measurement" means the same thing everywhere.
SENSOR_PSF_SIGMA_PER_RATIO = 0.289


def to_drizzle_matrix(align, ratio: float) -> np.ndarray:
    """Build the 3x3 src(frame)->dst(HR) warp for ``warpPerspective``.

    ``align`` is a ref->mov transform: either a ``(dx, dy)`` translation in
    ref->mov coordinates (the **negative** of :func:`estimate_translation`'s
    resampling offset — passing that offset unchanged deposits the frame
    displaced by ``2 d``) or a 3x3 homography (as returned by
    :func:`estimate_homography`). The returned matrix ``M`` maps source
    frame coordinates to the HR grid aligned to the reference frame, so
    ``dst(x) = src(M^-1 x)`` samples the frame at its true subpixel position.

    The HR grid is **pixel-centre aligned**: HR pixel ``i`` has its centre at
    the reference LR coordinate ``(i - (r - 1)/2) / r``, which is exactly the
    geometry of ``cv2.resize`` (and therefore of the bicubic baseline, the
    "before" panel and a centre-aligned GT upscale). Without the offset the
    output sits ``(r - 1)/2`` HR px up-left of a centre-aligned upscale
    (0.5 / 1.0 / 1.5 px at 2x / 3x / 4x; review item #4).

    Derivation: with ``S = diag(ratio, ratio, 1)`` scaling LR to HR and
    ``T = translation((r - 1)/2, (r - 1)/2)`` the half-pixel centre offset,
    the alignment warp is ``M = T @ S @ inv(W)``.
    """
    W = _normalize_homography(align)
    ratio = float(ratio)
    S = np.diag([ratio, ratio, 1.0])
    offset = 0.5 * (ratio - 1.0)
    T = np.array([[1.0, 0.0, offset],
                  [0.0, 1.0, offset],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    M = T @ S @ np.linalg.inv(W)
    return _normalize_homography(M).astype(np.float32)


def predict_lr_from_hr(hr: np.ndarray, ref_to_frame,
                       ratio: float, lr_shape: Tuple[int, int],
                       forward_sigma: Optional[float] = None) -> np.ndarray:
    """Predict one frame's LR measurement from an SR image (held-out proof).

    ``hr`` lives on the grid of a *reference* frame: HR pixel
    ``(r*x + (r-1)/2, r*y + (r-1)/2)`` is reference LR pixel ``(x, y)`` (that
    is what :func:`drizzle` produces; the reference is
    ``SRResult.reference_index``, which is NOT necessarily ``frames[0]``).
    ``ref_to_frame`` is a ref->frame transform in LR pixels, the same
    convention as :func:`estimate_homography` and :func:`drizzle`'s
    ``aligns``.

    The prediction is the sensor's own forward model — blur the HR estimate by
    the per-LR-pixel integration window, then sample it on the target frame's
    LR grid — i.e. the inverse of :func:`to_drizzle_matrix`:

        ``pred(p_frame) = (hr * G_sigma)((r - 1)/2 + r * W(p_frame))``.

    ``forward_sigma`` defaults to :data:`SENSOR_PSF_SIGMA_PER_RATIO` * ratio;
    it is applied with replicated borders so the border does not read as an
    error. Passing ``0`` returns the un-band-limited sample, which aliases at
    ratio > 1: it measured 0.1 / 0.3 / 0.7 dB lower held-out gain at 2x / 3x /
    4x on the real burst. It is kept available only so a caller can report
    that variant alongside the headline number.
    """
    if forward_sigma is None:
        forward_sigma = SENSOR_PSF_SIGMA_PER_RATIO * float(ratio)
    src = hr
    if forward_sigma and forward_sigma > 0:
        k = int(np.ceil(3.0 * float(forward_sigma))) | 1
        src = cv2.GaussianBlur(hr, (k, k), float(forward_sigma),
                               borderType=cv2.BORDER_REPLICATE)
    M = np.linalg.inv(to_drizzle_matrix(ref_to_frame, ratio)
                      .astype(np.float64)).astype(np.float32)
    return cv2.warpPerspective(src, M, (int(lr_shape[1]), int(lr_shape[0])),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def lr_sampling_valid_mask(hr_shape: Tuple[int, int], ref_to_frame,
                           ratio: float, lr_shape: Tuple[int, int],
                           forward_sigma: Optional[float] = None) -> np.ndarray:
    """Boolean mask of the LR samples :func:`predict_lr_from_hr` can support.

    A predicted LR sample reads the HR estimate at ``r * W^-1(p)`` and, when
    the sensor PSF is applied, it also reads the HR neighbourhood within
    ``3 * sigma`` of that point. Near the HR border those reads fall outside
    the reconstructed area; ``predict_lr_from_hr`` replicates the border there,
    so the value is defined but *unconstrained* — it can be right or wrong for
    reasons that have nothing to do with resolution recovery.

    The mask is True where every sample the prediction uses lies strictly
    inside ``hr_shape``, so a metric can compare SR and baseline on pixels
    both sides actually determine. It is the exact geometric support of the
    same forward model :func:`predict_lr_from_hr` applies (same matrix, same
    PSF), so the two cannot drift apart.

    Returns a ``bool`` array of ``lr_shape``.
    """
    if forward_sigma is None:
        forward_sigma = SENSOR_PSF_SIGMA_PER_RATIO * float(ratio)
    h, w = int(lr_shape[0]), int(lr_shape[1])
    hh, wh = int(hr_shape[0]), int(hr_shape[1])
    # Held-out LR pixel centre -> HR sample point: p_hr = M @ p_lr, the same
    # deposit/sample map predict_lr_from_hr inverts (including the HR grid's
    # (r-1)/2 pixel-centre offset).
    A = to_drizzle_matrix(ref_to_frame, ratio).astype(np.float64)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)], axis=0)
    q = A @ pts
    z = q[2]
    ok = np.abs(z) > 1e-12
    z = np.where(ok, z, 1.0)
    qx = (q[0] / z).reshape(h, w)
    qy = (q[1] / z).reshape(h, w)
    # Bilinear interpolation needs the four enclosing HR pixels; the Gaussian
    # support adds 3*sigma (ceil, because the kernel radius is an int).
    pad = 1.0 + (np.ceil(3.0 * float(forward_sigma))
                 if forward_sigma and forward_sigma > 0 else 0.0)
    inside = (ok.reshape(h, w)
              & (qx >= pad) & (qx <= wh - 1 - pad)
              & (qy >= pad) & (qy <= hh - 1 - pad))
    return inside


# Upper bound on the byte size of a warp *output* that may be routed through
# OpenCL (UMat). Larger outputs lose more to host<->device transfer than the
# kernel saves (measured: an 8192^2 f64 UMat round-trip alone costs ~0.6 s),
# so they always stay on the CPU.
_OCL_MAX_OUT_BYTES = 64 * 1024 * 1024

# Cached result of the one-time per-process OpenCL calibration.
_opencl_beneficial_cache: Optional[bool] = None


def _opencl_warp(src: np.ndarray, M: np.ndarray, dsize: Tuple[int, int],
                 flags: int, active: bool,
                 border_mode: int = cv2.BORDER_CONSTANT) -> np.ndarray:
    """``cv2.warpPerspective`` with optional OpenCL (T-API) execution.

    ``active=True`` runs the kernel on the OpenCL device through a :class:`cv2.UMat`
    wrapper when the *output* is small enough to beat a host round-trip
    (``_OCL_MAX_OUT_BYTES``); any failure — kernel build, allocation, device
    loss — falls back to the identical CPU call, so a machine without a usable
    GPU is never wrong, only (unmeasurably) slower. Brand-agnostic: whatever
    device OpenCV's T-API finds (AMD/NVIDIA/Intel, any vendor runtime).

    Deliberately used only for the *upsample-direction* warps: measured per-op
    (CPU vs T-API, ROI 512/1024), ``warpPerspective`` upscale is the only op
    class in this pipeline where the device competes at all — ``GaussianBlur``,
    ``resize`` and the downsample-direction warps are all slower through
    T-API than on the CPU threads, so routing them would lose time.
    """
    if active:
        out_bytes = dsize[0] * dsize[1] * np.dtype(np.float32).itemsize \
            * (src.shape[2] if src.ndim == 3 else 1)
        if out_bytes <= _OCL_MAX_OUT_BYTES:
            try:
                um = cv2.warpPerspective(cv2.UMat(src), M, dsize, flags=flags,
                                         borderMode=border_mode)
                return um.get()
            except cv2.error:
                pass  # device/kernel failure -> CPU fallback below
    return cv2.warpPerspective(src, M, dsize, flags=flags,
                               borderMode=border_mode)


def opencl_beneficial() -> bool:
    """Whether OpenCL actually accelerates this pipeline on this machine.

    ``use_opencl=None`` (auto) asks this question once per process and caches
    the answer. "A device exists" alone is not enough to be honest about
    performance: measured on a Mesa-OpenCL userspace stack every candidate op
    ran at 0.14-0.72x of the CPU (and the f64 warp kernel failed to build
    entirely, silently falling back), so blindly routing to the GPU can *lose*
    time. The same holds for a correct discrete NVIDIA device: measured on an
    RTX 3050 Laptop GPU (Optimus laptop, OpenCL 3.0, the device OpenCV selects
    by default on that machine) the routed f32 upsample warp ran 0.09x of the
    CPU at 1024^2 output and 0.63-0.82x at 4096^2 — even with host<->device
    transfers fully amortised the device kernel costs ~0.70 ms/warp against
    this CPU's ~0.25 ms multithreaded warp. Auto therefore requires both a
    device and a win: a representative
    upsample ``warpPerspective`` (the one op class routed to the device) must
    time at least ~15% faster through T-API than on the CPU. Explicit
    ``use_opencl=True`` bypasses the calibration and routes anyway (that is
    what the desktop benchmark compares), ``False`` never routes.
    """
    global _opencl_beneficial_cache
    if _opencl_beneficial_cache is None:
        verdict = False
        try:
            if cv2.ocl.haveOpenCL():
                import time as _time
                rng = np.random.default_rng(1234)
                # float32: the dtype of every warp _opencl_warp actually
                # routes, so the calibration measures the routed op exactly.
                lr = rng.random((512, 512), dtype=np.float32)
                mat = np.array([[2.0, 0.0, 3.0], [0.0, 2.0, -2.0],
                                [1e-5, -8e-6, 1.0]], dtype=np.float64)
                dsize = (1024, 1024)
                kw = dict(flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
                um = cv2.UMat(lr)
                cv2.warpPerspective(um, mat, dsize, **kw).get()  # warm/build

                def _bench(fn):
                    best = float("inf")
                    for _ in range(3):
                        t0 = _time.perf_counter()
                        fn()
                        best = min(best, _time.perf_counter() - t0)
                    return best

                t_cpu = _bench(lambda: cv2.warpPerspective(lr, mat, dsize, **kw))
                t_ocl = _bench(lambda: cv2.warpPerspective(um, mat, dsize, **kw).get())
                verdict = t_ocl < 0.85 * t_cpu
        except Exception:
            verdict = False
        _opencl_beneficial_cache = verdict
    return _opencl_beneficial_cache


def opencl_device_name() -> Optional[str]:
    """Name of the OpenCL device OpenCV's T-API selects, or None.

    Reads the already-resolved default context (``OPENCV_OPENCL_DEVICE`` is
    consumed by OpenCV once per process, at first UMat use; the desktop
    backend pins a deterministic NVIDIA choice there — see
    ``desktop/backend/server.py``). Returns None when no device exists or the
    OpenCV binding predates the ``cv2.ocl.Device`` introspection class.
    """
    try:
        dev_cls = getattr(cv2.ocl, 'Device', None)
        if dev_cls is None or not cv2.ocl.haveOpenCL():
            return None
        dev = dev_cls.getDefault()
        name = getattr(dev, 'name', None)
        if callable(name):  # OpenCV 5 exposes it as a method
            name = name()
        return name if isinstance(name, str) and name else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# GPU scatter deposition (optional pyopencl backend)
# --------------------------------------------------------------------------- #

# Mirror of :func:`_scatter_fast`'s bilinear/area splat as one OpenCL work-item
# per LR pixel. Every per-splat computation replicates the CPU path op for op:
# the sub-sample offsets (same (i+0.5)/S-0.5 formula), the double-precision
# homography projection and divide, floor/fract tap decomposition, and tap
# weight ((wu*wv)*subw)*frame_weight — the frame weight is folded into the
# deposit instead of multiplied onto a finished f64 table, which differs only
# by f64 round-off (~1e-16 relative, far below the f32 table resolution below).
# The ONE deliberate difference: cross-splat summation happens with float32
# global atomics, whose order is nondeterministic run to run. The GPU result
# therefore equals the CPU float64 result within f32 accumulation round-off
# (~1e-6 relative — the documented GPU tolerance) but is NOT bit-reproducible
# on the GPU itself; the CPU path remains bit-identical.
_GPU_SCATTER_KERNEL = r"""
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifdef cl_khr_global_int32_base_atomics
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

// Float32 atomic add through the int compare-and-swap primitive: portable
// across OpenCL vendors (NVIDIA's CUDA 13 headers dropped the legacy float
// atomic_add overload; the CAS loop needs only the base-atomics extension).
static inline void atom_add_f32(__global float* p, float v)
{
    __global int* ip = (__global int*)p;
    while (1) {
        int old = *ip;
        int want = as_int(as_float(old) + v);
        if (atomic_cmpxchg(ip, old, want) == old) break;
    }
}

__kernel void scatter_splat(
    __global const float* __restrict__ src,   // chunk of (ny, w, 3) LR rows
    const int y0,                             // first row of the chunk
    const int w,                              // LR row width
    const double m00, const double m01, const double m02,
    const double m10, const double m11, const double m12,
    const double m20, const double m21,
    const int oh, const int ow,               // HR grid size
    const int s_n, const double pixfrac, const double subw,
    const double fweight, const int area,     // frame weight; 0 bilinear, 1 area
    __global const float* __restrict__ rweight, // per-LR-pixel robustness w
    const int use_rw,                         // 0 = plain (weight 1)
    __global float* acc,                      // HR accumulator, stride 4/cell
    __global float* cov)
{
    const int i = get_global_id(0);
    const int ly = i / w;
    const int x = i - ly * w;
    const double X = (double)x;
    const double Y = (double)(y0 + ly);
    const float v0 = src[3 * i + 0];
    const float v1 = src[3 * i + 1];
    const float v2 = src[3 * i + 2];
    double rw = 1.0;
    if (use_rw) {
        rw = (double)rweight[(y0 + ly) * w + x];
    }
    for (int sy = 0; sy < s_n; sy++) {
        const double Yp = Y + (((double)sy + 0.5) / s_n - 0.5) * pixfrac;
        for (int sx = 0; sx < s_n; sx++) {
            const double Xp = X + (((double)sx + 0.5) / s_n - 0.5) * pixfrac;
            const double den = m20 * Xp + m21 * Yp + 1.0;
            double U = (m00 * Xp + m01 * Yp + m02) / den;
            double V = (m10 * Xp + m11 * Yp + m12) / den;
            if (area) { U += 0.5; V += 0.5; }
            const int bu = (int)floor(U);
            const int bv = (int)floor(V);
            const double tu = U - (double)bu;
            const double tv = V - (double)bv;
            if (area) {
                if (bu >= 0 && bu < ow && bv >= 0 && bv < oh) {
                    const int j = 4 * (bv * ow + bu);
                    atom_add_f32(acc + j + 0, (float)(((double)v0 * subw) * fweight * rw));
                    atom_add_f32(acc + j + 1, (float)(((double)v1 * subw) * fweight * rw));
                    atom_add_f32(acc + j + 2, (float)(((double)v2 * subw) * fweight * rw));
                    atom_add_f32(cov + (bv * ow + bu), (float)(subw * fweight * rw));
                }
            } else {
                for (int dj = 0; dj < 2; dj++) {
                    const int vv = bv + dj;
                    if (vv < 0 || vv >= oh) continue;
                    const double wv = dj ? tv : 1.0 - tv;
                    for (int di = 0; di < 2; di++) {
                        const int uu = bu + di;
                        if (uu < 0 || uu >= ow) continue;
                        const double wu = di ? tu : 1.0 - tu;
                        const double ww = ((wu * wv) * subw) * fweight * rw;
                        const int j = 4 * (vv * ow + uu);
                        atom_add_f32(acc + j + 0, (float)((double)v0 * ww));
                        atom_add_f32(acc + j + 1, (float)((double)v1 * ww));
                        atom_add_f32(acc + j + 2, (float)((double)v2 * ww));
                        atom_add_f32(cov + (vv * ow + uu), (float)ww);
                    }
                }
            }
        }
    }
}
#endif
#endif
"""

# Upper bound on the device-side HR tables (acc float4 + cov float per cell)
# for one drizzle() call. Same philosophy as _OCL_MAX_OUT_BYTES, sized for
# VRAM: ROI 2048 (HR 4096^2) needs ~335 MB; anything past this stays on the
# CPU. 1.5 GB leaves headroom on a 4 GB laptop GPU next to the desktop's own
# allocations.
_OCL_SCATTER_MAX_TABLE_BYTES = 1536 * 1024 * 1024

# LR rows are uploaded/deposited in chunks whose host slice fits this budget
# (the 64 MB guard philosophy of _OCL_MAX_OUT_BYTES applied to the input side).
_OCL_SCATTER_CHUNK_BYTES = 64 * 1024 * 1024

# Lazy pyopencl environment: None = not probed yet, False = probed and
# unavailable (no module / no platform / no GPU device / kernel build error),
# dict = ready. Probing NEVER raises: on a machine without OpenCL the module
# behaves exactly as before the backend existed.
_gpu_scatter_env_cache: Optional[dict] = None
# Cached POSITIVE result of the GPU-scatter-vs-CPU calibration. Only True is
# cached (a measured win is stable); a False is re-measured on the next call so
# one noisy sample can no longer pin a whole session to the CPU.
_gpu_scatter_beneficial_cache: Optional[bool] = None
# Runs per arm in the calibration; the arm statistic is their MEDIAN, so a single
# transient stall cannot flip the verdict.
_GPU_SCATTER_BENCH_REPS = 3


def _gpu_scatter_env() -> Optional[dict]:
    """Build (once) the pyopencl context, program and device for scatter.

    Brand-agnostic: the GPU device with the most global memory across every
    ICD platform (NVIDIA/AMD/Intel, any vendor runtime) is chosen. Returns
    None when pyopencl is not installed or no usable GPU/device exists —
    every caller then stays on the identical CPU path.
    """
    global _gpu_scatter_env_cache
    if _gpu_scatter_env_cache is None:
        env: object = False
        try:
            import pyopencl as cl  # guarded: absent module => CPU-only path
            devices = []
            for plat in cl.get_platforms():
                try:
                    devices.extend(
                        plat.get_devices(device_type=cl.device_type.GPU))
                except Exception:
                    continue
            if devices:
                dev = max(devices, key=lambda d: d.global_mem_size)
                ctx = cl.Context(devices=[dev])
                # -w: NVIDIA's build emits a cosmetic warning note for this
                # source; keep it from bubbling a CompilerWarning to users.
                prog = cl.Program(ctx, _GPU_SCATTER_KERNEL).build(options=["-w"])
                env = {"cl": cl, "ctx": ctx, "kern": prog.scatter_splat,
                       "name": str(dev.name).strip(),
                       "mem_mb": int(dev.global_mem_size) // 2**20}
        except Exception:
            env = False
        _gpu_scatter_env_cache = env
    return _gpu_scatter_env_cache or None


class _GpuScatterDeposit:
    """Device-side HR accumulator tables for one drizzle() call.

    ``deposit()`` mirrors one :func:`_scatter_accumulate` bilinear/area splat
    (same normalization of ``M``, same sub-sample/tap math — see
    :data:`_GPU_SCATTER_KERNEL`), adding float32 atomically into persistent
    device tables shared by every frame of the run; the per-frame weight is
    folded into each deposit. ``fetch()`` returns the tables as float32 —
    the GPU counterpart of the CPU path's float64 tables, equal within the
    f32 atomic-accumulation tolerance (~1e-6 relative) and, because atomics
    are unordered, NOT bit-reproducible between GPU runs. The caller
    (:func:`drizzle`) normalizes and finishes the fold exactly as on the CPU.
    """

    def __init__(self, env: dict, oh: int, ow: int):
        cl = env["cl"]
        self._cl = cl
        self._ctx = env["ctx"]
        self._kern = env["kern"]
        self.oh, self.ow = oh, ow
        self.queue = cl.CommandQueue(self._ctx)
        self.acc = cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE,
                             oh * ow * 16)
        self.cov = cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE, oh * ow * 4)
        zero = np.float32(0.0)
        cl.enqueue_fill_buffer(self.queue, self.acc, zero, 0, oh * ow * 16)
        cl.enqueue_fill_buffer(self.queue, self.cov, zero, 0, oh * ow * 4)
        self._src = None
        self._src_bytes = 0
        self._rw = None
        self._rw_bytes = 0

    def deposit(self, src: np.ndarray, M: np.ndarray, pixfrac: float,
                subsamples: int, splat: str, fweight: float,
                rweight: Optional[np.ndarray] = None) -> None:
        h, w = src.shape[:2]
        M = np.asarray(M, dtype=np.float64)
        if not np.all(np.isfinite(M)) or abs(M[2, 2]) < 1e-12:
            M = np.eye(3, dtype=np.float64)
        else:
            M = M / M[2, 2]
        s_n = max(int(subsamples), 1)
        p = float(pixfrac) if (pixfrac is not None and pixfrac > 0) else 1.0
        subw = 1.0 / float(s_n * s_n)
        area = 1 if splat == "area" else 0
        rows = max(1, min(h, _OCL_SCATTER_CHUNK_BYTES // (w * 12)))
        need = rows * w * 12
        if self._src is None or self._src_bytes < need:
            self._src = self._cl.Buffer(self._ctx,
                                        self._cl.mem_flags.READ_ONLY, need)
            self._src_bytes = need
        # Per-LR-pixel robustness weights (robust fusion). The kernel indexes
        # them by global LR row, so the FULL frame map is uploaded once per
        # deposit (not per chunk). use_rw=0 keeps the plain path byte-identical.
        rw_buf = None
        use_rw = 0
        if rweight is not None:
            rw_np = np.ascontiguousarray(rweight, dtype=np.float32).reshape(h, w)
            rw_need = h * w * 4
            if self._rw is None or self._rw_bytes < rw_need:
                self._rw = self._cl.Buffer(self._ctx,
                                           self._cl.mem_flags.READ_ONLY,
                                           rw_need)
                self._rw_bytes = rw_need
            self._cl.enqueue_copy(self.queue, self._rw, rw_np.reshape(-1))
            rw_buf = self._rw
            use_rw = 1
        args_tail = (np.int32(self.oh), np.int32(self.ow), np.int32(s_n),
                     np.float64(p), np.float64(subw), np.float64(fweight),
                     np.int32(area), rw_buf, np.int32(use_rw),
                     self.acc, self.cov)
        for y0 in range(0, h, rows):
            y1 = min(y0 + rows, h)
            chunk = np.ascontiguousarray(src[y0:y1], dtype=np.float32)
            self._cl.enqueue_copy(self.queue, self._src, chunk.reshape(-1))
            self._kern(self.queue, (w * (y1 - y0),), None, self._src,
                       np.int32(y0), np.int32(w),
                       np.float64(M[0, 0]), np.float64(M[0, 1]),
                       np.float64(M[0, 2]), np.float64(M[1, 0]),
                       np.float64(M[1, 1]), np.float64(M[1, 2]),
                       np.float64(M[2, 0]), np.float64(M[2, 1]),
                       *args_tail)
        self.queue.finish()

    def fetch(self) -> Tuple[np.ndarray, np.ndarray]:
        acc = np.empty((self.oh, self.ow, 4), dtype=np.float32)
        cov = np.empty((self.oh, self.ow), dtype=np.float32)
        self._cl.enqueue_copy(self.queue, acc, self.acc)
        self._cl.enqueue_copy(self.queue, cov, self.cov)
        self.queue.finish()
        return acc[:, :, :3], cov

    def close(self) -> None:
        self._src = None
        self._rw = None
        self.acc = None
        self.cov = None


def _gpu_scatter_begin(oh: int, ow: int,
                       splat: str) -> Tuple[Optional["_GpuScatterDeposit"],
                                            str]:
    """Try to open GPU tables for a drizzle() scatter run.

    Returns ``(engine, why)``: ``(engine, 'gpu')`` when ready; ``(None,
    'unavailable')`` when no device exists; ``(None, 'cpu')`` when the device
    exists but this run declines it (splat not GPU-supported, or the HR tables
    would exceed the VRAM budget guard). Never raises.
    """
    env = _gpu_scatter_env()
    if env is None:
        return None, "unavailable"
    if splat not in ("bilinear", "area"):
        return None, "cpu"
    if oh * ow * 20 > _OCL_SCATTER_MAX_TABLE_BYTES:
        return None, "cpu"
    try:
        return _GpuScatterDeposit(env, oh, ow), "gpu"
    except Exception:
        return None, "cpu"


def _gpu_scatter_bench() -> Tuple[float, float]:
    """Median wall seconds of one GPU and one CPU scatter run (calibration).

    Builds the exact shipped calibration workload (LR 256^2 -> HR 512^2,
    pixfrac 0.5, subsamples 3, bilinear, 2 frames) and times
    :data:`_GPU_SCATTER_BENCH_REPS` runs per arm after one GPU warm-up. The
    reported time is the **median** of the runs, so a single transient stall in
    either arm cannot flip the verdict. Raises if no GPU engine can be opened.
    """
    import time as _time
    rng = np.random.default_rng(4321)
    lr = rng.random((256, 256, 3), dtype=np.float32)
    mat = np.array([[2.0, 0.0, 3.0], [0.0, 2.0, -2.0],
                    [1e-5, -8e-6, 1.0]], dtype=np.float64)
    oh = ow = 512
    n_frames = 2
    wts = 0.4

    def _cpu_run():
        acc = np.zeros((oh, ow, 3), dtype=np.float64)
        cov = np.zeros((oh, ow), dtype=np.float64)
        for _ in range(n_frames):
            a, c = _scatter_fast(lr, mat, oh, ow, 0.5, 3)
            acc += a * wts
            cov += c * wts
        return float(acc[0, 0, 0] + cov[0, 0])

    def _gpu_run():
        eng, why = _gpu_scatter_begin(oh, ow, "bilinear")
        if eng is None:
            raise RuntimeError(why)
        try:
            for _ in range(n_frames):
                eng.deposit(lr, mat, 0.5, 3, "bilinear", wts)
            a, c = eng.fetch()
            return float(a[0, 0, 0] + c[0, 0])
        finally:
            eng.close()

    def _median(fn):
        times = []
        for _ in range(_GPU_SCATTER_BENCH_REPS):
            t0 = _time.perf_counter()
            fn()
            times.append(_time.perf_counter() - t0)
        return float(np.median(times))

    _gpu_run()  # warm-up: first enqueue allocs, builds nothing new
    t_gpu = _median(_gpu_run)
    t_cpu = _median(_cpu_run)
    return t_gpu, t_cpu


def gpu_scatter_beneficial() -> bool:
    """Whether GPU scatter deposition actually beats the CPU engine here.

    ``use_opencl=None`` (auto) asks this alongside the warp calibration in
    :func:`opencl_beneficial`. The warp decision and this one are independent:
    routing *warps* lost on every stack measured so far (T-API warp kernels
    cost ~0.70 ms against this CPU's ~0.25 ms multithreaded warp), but the
    scatter deposit is a different op class — pure memory-bandwidth streaming,
    which the ~536 MB float64 banded tables already saturate on one CPU core
    while the GPU has dedicated GDDR6.

    The calibration (:func:`_gpu_scatter_bench`) runs the exact shipped kernel
    and engine at a representative chunk, timing the **median of 3 runs per
    arm** (a one-shot wall-clock comparison is load-sensitive: measured False
    for 13 consecutive fresh processes under transient load and True 12/12 when
    idle). Auto engages GPU scatter only when the GPU median is at least 5%
    faster; any missing piece (module, platform, device, kernel build) returns
    False.

    A **True** verdict is cached for the process lifetime (a measured win is
    stable). A **False** verdict is deliberately NOT cached: the next call
    re-measures (~150 ms), so one noisy sample under transient load can no
    longer pin an entire session to the CPU — the root cause of the
    "GPU works for me, not for you" report.
    """
    global _gpu_scatter_beneficial_cache
    if _gpu_scatter_beneficial_cache:  # only ever set to True
        return True
    verdict = False
    try:
        if _gpu_scatter_env() is not None:
            t_gpu, t_cpu = _gpu_scatter_bench()
            verdict = t_gpu < 0.95 * t_cpu
    except Exception:
        verdict = False
    if verdict:
        _gpu_scatter_beneficial_cache = True
    return verdict


def _scatter_fast(src: np.ndarray, M: np.ndarray, oh: int, ow: int,
                  pixfrac: float, subsamples: int, splat: str = "bilinear",
                  chunk: int = 16,
                  rw: Optional[np.ndarray] = None,
                  fweight: float = 1.0,
                  out_acc: Optional[np.ndarray] = None,
                  out_wt: Optional[np.ndarray] = None,
                  local_du: Optional[np.ndarray] = None,
                  local_dv: Optional[np.ndarray] = None,
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Banded row-chunk engine for the bilinear/area splats (the hot path).

    Same deposition as the generic loop in :func:`_scatter_accumulate`
    (numerically equal to ~1e-16: validated over identity/translation/
    rotation+perspective/negative-offset/fractional-ratio matrices, both
    supported splats, and the impulse cases), but ~3.5x faster because it

    * processes the source in thin row bands (``chunk`` rows), so each
      band's scatter table (3 channel planes + weight, one float64 per HR
      cell of the band) stays resident in L3 while :func:`np.bincount`
      accumulates into it — bincount into a small table sustains ~440M
      deposits/s versus ~80M/s into a full-frame table;
    * computes the projected ``(U, V)`` geometry once per band for *all*
      ``subsamples x subsamples`` footprint offsets at once (shape
      ``(S, S, w*chunk)``) instead of re-deriving the warp per offset;
    * skips per-tap bounds masking for interior bands, where every tap
      provably lands inside the band (band bounds come from the band's own
      projected ``V`` range), leaving masking only for the few border bands.

    Not a new algorithm: the deposited values and weights are identical.

    ``rw`` (optional, ``(h, w)`` float): a per-LR-pixel robustness weight. When
    given, every deposit from source pixel ``p`` is multiplied by ``rw[p]`` in
    *both* the accumulated value and the coverage weight, so the deposited
    average stays ``sum(rw * a * w) / sum(rw * w)`` — the robust-fusion
    re-scatter (see :func:`drizzle`). ``rw=None`` reproduces the plain deposit
    exactly.

    ``fweight`` is the frame's fusion weight, folded into the deposit so no
    separate ``acc * weight`` full-frame temp exists. ``out_acc``/``out_wt``
    (both or neither) make the engine add each band straight into caller-owned
    run-level tables — the drizzle path passes float32 arrays, so a whole run
    holds one pair of tables instead of one per frame. Without them the engine
    returns fresh float64 tables, exactly as before.

    (A frame-parallel variant of this loop was built and measured at product
    size — frames scattered concurrently into disjoint shared-memory tables,
    folded in frame order. It is bit-identical but never faster here: 40.7 s
    serial vs 41.9/46.2/51.1 s with 2/3/4 workers, because the ~536 MB
    float64 per-frame tables already saturate the memory pipe on one core.
    The deposit therefore stays single-threaded.)
    """
    h, w = src.shape[:2]
    m00, m01, m02 = M[0]
    m10, m11, m12 = M[1]
    m20, m21 = M[2, :2]

    s_n = max(int(subsamples), 1)
    p = float(pixfrac) if (pixfrac is not None and pixfrac > 0) else 1.0
    offs = ((np.arange(s_n) + 0.5) / s_n - 0.5) * p
    subw = float(fweight) / float(s_n * s_n)

    if out_acc is None:
        acc = np.zeros((oh, ow, 3), dtype=np.float64)
        wt = np.zeros((oh, ow), dtype=np.float64)
    else:
        # Streamed deposit: write the band tables straight into the caller's
        # run-level accumulator (float32 in the drizzle path). No per-frame
        # full-frame table is materialised and no frame-weight temp exists,
        # which is where the old path spent ~1.0 GB at 4x ROI 1024.
        acc, wt = out_acc, out_wt
    xs = np.arange(w, dtype=np.float64)
    one_off = (splat == "area")
    if rw is not None:
        rw = np.ascontiguousarray(rw, dtype=np.float64).reshape(h, w)
    if local_du is not None:
        local_du = np.ascontiguousarray(local_du, dtype=np.float64).reshape(h, w)
    if local_dv is not None:
        local_dv = np.ascontiguousarray(local_dv, dtype=np.float64).reshape(h, w)

    for y0 in range(0, h, int(chunk)):
        y1 = min(y0 + int(chunk), h)
        ny = y1 - y0
        X = np.tile(xs, ny)
        Y = np.repeat(np.arange(y0, y1, dtype=np.float64), w)
        vals_c = [np.ascontiguousarray(src[y0:y1, :, c], dtype=np.float64).ravel()
                  for c in range(3)]
        rwb = rw[y0:y1].ravel() if rw is not None else None

        # Projected positions for every (oy, ox) subsample offset: (S, S, n).
        Xp = X[None, None, :] + offs[None, :, None]
        Yp = Y[None, None, :] + offs[:, None, None]
        den = m20 * Xp + m21 * Yp + 1.0
        U = (m00 * Xp + m01 * Yp + m02) / den
        V = (m10 * Xp + m11 * Yp + m12) / den
        if local_du is not None:
            U = U + local_du[y0:y1].ravel()[None, None, :]
        if local_dv is not None:
            V = V + local_dv[y0:y1].ravel()[None, None, :]
        if one_off:  # area splat: whole sub-sample to the cell containing it
            U = U + 0.5
            V = V + 0.5
        u_min, u_max = float(U.min()), float(U.max())
        v_min, v_max = float(V.min()), float(V.max())
        b0 = max(0, int(np.floor(v_min)))
        b1 = min(oh, int(np.floor(v_max)) + (1 if one_off else 2))
        nb = b1 - b0
        if nb <= 0:
            continue
        # Interior band: every tap of every point lands inside [b0, b1) x
        # [0, ow), so the bounds mask below is always true and is skipped.
        interior = (v_min >= 0.0 and v_max < oh - 1.0 and u_min >= 0.0
                    and u_max < ow - 1.0)
        tb0 = b0 * ow
        tbl_v = [np.zeros(nb * ow, dtype=np.float64) for _ in range(3)]
        tbl_w = np.zeros(nb * ow, dtype=np.float64)

        for s_i in range(s_n):     # oy index
            for q_i in range(s_n):  # ox index
                u_p = U[s_i, q_i]
                v_p = V[s_i, q_i]
                base_u = np.floor(u_p).astype(np.int64)
                base_v = np.floor(v_p).astype(np.int64)
                if one_off:
                    taps_u: Tuple = ((0, None),)
                    taps_v: Tuple = ((0, None),)
                else:
                    tu = u_p - np.floor(u_p)
                    tv = v_p - np.floor(v_p)
                    taps_u = ((0, 1.0 - tu), (1, tu))
                    taps_v = ((0, 1.0 - tv), (1, tv))
                for dj, wv in taps_v:
                    vv = base_v + dj
                    row = (base_v + dj) * ow
                    for di, wu in taps_u:
                        uu = base_u + di
                        if wu is None:
                            wgt = np.full(uu.size, subw)
                        else:
                            wgt = (wu * wv) * subw
                        if rwb is not None:
                            wgt = wgt * rwb
                        if interior:
                            idx = row + uu - tb0
                            tbl_w += np.bincount(idx, weights=wgt,
                                                 minlength=nb * ow)
                            for c in range(3):
                                tbl_v[c] += np.bincount(
                                    idx, weights=vals_c[c] * wgt,
                                    minlength=nb * ow)
                        else:
                            msk = ((uu >= 0) & (uu < ow)
                                   & (vv >= b0) & (vv < b1))
                            if not msk.any():
                                continue
                            idx = (vv[msk] - b0) * ow + uu[msk]
                            wg = wgt[msk]
                            tbl_w += np.bincount(idx, weights=wg,
                                                 minlength=nb * ow)
                            for c in range(3):
                                tbl_v[c] += np.bincount(
                                    idx, weights=vals_c[c][msk] * wg,
                                    minlength=nb * ow)
        sl = slice(b0, b1)
        for c in range(3):
            acc[sl, :, c] += tbl_v[c].reshape(nb, ow)
        wt[sl] += tbl_w.reshape(nb, ow)
    return acc, wt


def _scatter_accumulate(src: np.ndarray, M: np.ndarray, oh: int, ow: int,
                        pixfrac: float, subsamples: int, splat: str = "bilinear",
                        splat_sigma: float = 0.5,
                        rw: Optional[np.ndarray] = None,
                        fweight: float = 1.0,
                        out_acc: Optional[np.ndarray] = None,
                        out_wt: Optional[np.ndarray] = None,
                        local_du: Optional[np.ndarray] = None,
                        local_dv: Optional[np.ndarray] = None,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """True-drizzle scatter of one LR frame's measured samples onto the HR grid.

    Unlike a gather (which, for each output pixel, *interpolates* neighbouring
    LR samples), this deposits (scatters) every LR pixel's own measured value
    at its true HR position through ``M``, using a compact bilinear splat and
    accumulating a weight map. Because the value itself is never blended with
    its LR neighbours, the deposit does not low-pass the recovered band the way
    a bilinear *gather* does (a gather's reconstruction kernel has its first
    zero exactly at the single-frame Nyquist, so it cannot keep the extra band;
    the splat kernel's first zero is at the HR Nyquist).

    ``pixfrac`` (Fruchter & Hook, "drizzle") shrinks the input pixel's footprint
    in LR units before deposition, and ``subsamples`` sub-samples that footprint
    so a footprint smaller than an HR cell still leaves no holes. A value is
    never invented: each deposited number is an actual measured LR sample, only
    spread over its own sub-HR-pixel footprint.

    ``splat`` selects the deposit kernel. ``"bilinear"`` (default) spreads each
    sub-sample over its four nearest HR cells: it is a compact first-order
    reconstruction that keeps the recovered band (measured bandpass MSE 0.62x
    naive, beating the denoised-stack baseline). ``"area"`` assigns each
    sub-sample wholly to the HR cell containing it (classical area overlap);
    ``"cubic"`` is a cubic B-spline; ``"gaussian"`` uses a separable Gaussian of
    ``splat_sigma`` HR px. All are linear splats of measured samples, so no
    detail is fabricated. (Internally measured: area/cubic/gaussian give a
    smoother step response but lose real high-frequency gain — area 1.2% worse
    than the stack baseline on the bandpass case — so bilinear is the default.)

    Returns ``(acc (oh, ow, 3), wt (oh, ow))`` in float64.
    """
    h, w = src.shape[:2]
    M = np.asarray(M, dtype=np.float64)
    if not np.all(np.isfinite(M)) or abs(M[2, 2]) < 1e-12:
        M = np.eye(3, dtype=np.float64)
    else:
        M = M / M[2, 2]

    if splat in ("bilinear", "area"):
        # Shipped default kernels: banded engine, numerically equal to the
        # loop below (worst measured |diff| 5.6e-17) and ~3.5x faster.
        return _scatter_fast(src, M, oh, ow, pixfrac, subsamples, splat=splat,
                             rw=rw, fweight=fweight,
                             out_acc=out_acc, out_wt=out_wt,
                             local_du=local_du, local_dv=local_dv)

    ys, xs = np.mgrid[0:h, 0:w]
    X = xs.ravel().astype(np.float64)
    Y = ys.ravel().astype(np.float64)
    vals = src.reshape(-1, 3).astype(np.float64)

    m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
    m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
    m20, m21 = M[2, 0], M[2, 1]

    subsamples = max(int(subsamples), 1)
    p = float(pixfrac) if (pixfrac is not None and pixfrac > 0) else 1.0
    offs = ((np.arange(subsamples) + 0.5) / subsamples - 0.5) * p
    subw = float(fweight) / float(subsamples * subsamples)

    acc = np.zeros(oh * ow * 3, dtype=np.float64)
    wt = np.zeros(oh * ow, dtype=np.float64)
    ar = np.arange(3)
    rw_flat = None
    if rw is not None:
        rw_flat = np.ascontiguousarray(rw, dtype=np.float64).reshape(-1)

    for oy in offs:
        Yp = Y + oy
        for ox in offs:
            Xp = X + ox
            den = m20 * Xp + m21 * Yp + 1.0
            U = (m00 * Xp + m01 * Yp + m02) / den
            V = (m10 * Xp + m11 * Yp + m12) / den
            if splat == "area":
                ui = np.floor(U + 0.5).astype(np.int64)
                vi = np.floor(V + 0.5).astype(np.int64)
                m = (ui >= 0) & (ui < ow) & (vi >= 0) & (vi < oh)
                if not m.any():
                    continue
                idx = vi[m] * ow + ui[m]
                wcell = np.full(int(m.sum()), subw)
                if rw_flat is not None:
                    wcell = wcell * rw_flat[m]
                wt += np.bincount(idx, weights=wcell,
                                  minlength=oh * ow)
                idx3 = (idx[:, None] * 3 + ar[None, :]).ravel()
                acc += np.bincount(
                    idx3, weights=(vals[m] * wcell[:, None]).ravel(),
                    minlength=oh * ow * 3)
            else:
                base_u = np.floor(U).astype(np.int64)
                base_v = np.floor(V).astype(np.int64)
                tu = U - np.floor(U)
                tv = V - np.floor(V)
                if splat == "cubic":
                    # Cubic B-spline weights over taps -1,0,1,2.
                    wu_taps = [(-1, (1.0 - tu) ** 3 / 6.0),
                               (0, (3.0 * tu ** 3 - 6.0 * tu ** 2 + 4.0) / 6.0),
                               (1, (-3.0 * tu ** 3 + 3.0 * tu ** 2 + 3.0 * tu + 1.0) / 6.0),
                               (2, tu ** 3 / 6.0)]
                    wv_taps = [(-1, (1.0 - tv) ** 3 / 6.0),
                               (0, (3.0 * tv ** 3 - 6.0 * tv ** 2 + 4.0) / 6.0),
                               (1, (-3.0 * tv ** 3 + 3.0 * tv ** 2 + 3.0 * tv + 1.0) / 6.0),
                               (2, tv ** 3 / 6.0)]
                elif splat == "gaussian":
                    sg = max(float(splat_sigma), 1e-3)
                    rr = int(np.ceil(3.0 * sg))
                    kk = np.arange(-rr + 1, rr + 1)
                    wu_taps = [(int(k), np.exp(-((k - tu) ** 2) / (2.0 * sg * sg)))
                               for k in kk]
                    wv_taps = [(int(k), np.exp(-((k - tv) ** 2) / (2.0 * sg * sg)))
                               for k in kk]
                else:  # bilinear
                    wu_taps = [(0, 1.0 - tu), (1, tu)]
                    wv_taps = [(0, 1.0 - tv), (1, tv)]
                for di, wu in wu_taps:
                    uu = base_u + di
                    ok_u = (uu >= 0) & (uu < ow)
                    if not ok_u.any():
                        continue
                    for dj, wv in wv_taps:
                        vv = base_v + dj
                        m = ok_u & (vv >= 0) & (vv < oh)
                        if not m.any():
                            continue
                        wgt = (wu[m] * wv[m]) * subw
                        if rw_flat is not None:
                            wgt = wgt * rw_flat[m]
                        idx = vv[m] * ow + uu[m]
                        wt += np.bincount(idx, weights=wgt, minlength=oh * ow)
                        idx3 = (idx[:, None] * 3 + ar[None, :]).ravel()
                        acc += np.bincount(
                            idx3, weights=(vals[m] * wgt[:, None]).ravel(),
                            minlength=oh * ow * 3)
    if out_acc is not None:
        out_acc += acc.reshape(oh, ow, 3)
        out_wt += wt.reshape(oh, ow)
        return out_acc, out_wt
    return acc.reshape(oh, ow, 3), wt.reshape(oh, ow)


def _scatter_frames(frames, aligns, ratio, weights, oh, ow, pixfrac, subsamples,
                    splat, splat_sigma, use_gpu_scatter, rweights,
                    scatter_state, local_shifts=None, use_cupy: bool = False):
    """One weighted scatter pass over every frame (CPU or GPU engine).

    ``rweights`` is an optional list of per-frame ``(h, w)`` robustness weight
    maps (or None for a plain pass); each deposit is multiplied by it in both
    value and weight (see :func:`_scatter_fast`). ``local_shifts`` is an
    optional ``(dx_mov_list, dy_mov_list)`` tuple of per-frame residual
    displacement fields in LR pixels. Returns ``(acc, cov, scatter)``.
    """
    scatter = scatter_state
    acc = cov = None
    if use_cupy:
        from .gpu import scatter_frames_gpu
        res_gpu = scatter_frames_gpu(
            frames, aligns, ratio, weights, oh, ow, pixfrac, subsamples,
            splat=splat, rweights=rweights, local_shifts=local_shifts,
        )
        if res_gpu is not None:
            acc, cov = res_gpu
            return acc, cov, "cupy"
    eng = None
    if use_gpu_scatter and local_shifts is None:
        eng, scatter = _gpu_scatter_begin(oh, ow, splat)
    if eng is not None:
        try:
            for j, (fr, al, wt) in enumerate(zip(frames, aligns, weights)):
                M = to_drizzle_matrix(al, ratio)
                rw = rweights[j] if rweights is not None else None
                eng.deposit(fr.rgb, M, pixfrac, subsamples, splat, float(wt),
                            rweight=rw)
            acc, cov = eng.fetch()  # float32 tables (documented tolerance)
        except Exception:
            acc = cov = None  # device failure -> full CPU re-deposit
        finally:
            eng.close()
    if acc is None:  # CPU engine (default, and every GPU fallback)
        if scatter != "unavailable":
            scatter = "cpu"
        # Run-level tables are float32 and shared by every frame: the deposit
        # streams each band straight into them, so no per-frame full-frame
        # float64 table or `a * weight` temp is ever materialised (the GPU
        # path has always accumulated in float32; the CPU path now matches it
        # to ~1e-7 relative, far inside the documented f32 tolerance).
        acc = np.zeros((oh, ow, 3), dtype=np.float32)
        cov = np.zeros((oh, ow), dtype=np.float32)
        r_f = float(ratio)
        for j, (fr, al, wt) in enumerate(zip(frames, aligns, weights)):
            M = to_drizzle_matrix(al, ratio)
            rw = rweights[j] if rweights is not None else None
            du = (local_shifts[0][j] * r_f) if local_shifts is not None else None
            dv = (local_shifts[1][j] * r_f) if local_shifts is not None else None
            _scatter_accumulate(fr.rgb, M, oh, ow, pixfrac, subsamples,
                                splat=splat, splat_sigma=splat_sigma,
                                rw=rw, fweight=float(wt),
                                out_acc=acc, out_wt=cov,
                                local_du=du, local_dv=dv)
    return acc, cov, scatter


def _warp_frames_to_hr(frames, aligns, ratio):
    """Gather every frame onto the HR grid (luminance) plus its validity mask.

    Returns ``(S (N,oh,ow) float32, V (N,oh,ow) bool, oh, ow)``. This is the
    common grid on which frames are compared to each other for robust fusion.
    """
    h, w = frames[0].rgb.shape[:2]
    oh, ow = int(round(h * ratio)), int(round(w * ratio))
    ones = np.ones((h, w), dtype=np.float32)
    # Memory note (B2): the output arrays are allocated up front and filled per
    # frame. At 4x the HR grid is 4096^2, so collecting per-frame arrays and
    # np.stack-ing them held a second full copy (~1.8 GiB at N=9) at peak.
    n = len(frames)
    S = np.empty((n, oh, ow), dtype=np.float32)
    V = np.empty((n, oh, ow), dtype=bool)
    for j, (fr, al) in enumerate(zip(frames, aligns)):
        M = to_drizzle_matrix(al, ratio).astype(np.float32)
        wp = cv2.warpPerspective(np.asarray(fr.rgb, dtype=np.float32), M, (ow, oh),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
        mk = cv2.warpPerspective(ones, M, (ow, oh), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
        _to_gray(wp, out=S[j])
        V[j] = mk > 0.99
        del wp, mk
    return S, V, oh, ow


def _masked_temporal_median(S: np.ndarray, V: np.ndarray,
                            chunk: int = 256) -> np.ndarray:
    """Per-pixel temporal median of ``S`` ignoring invalid frames (row chunked).

    Same numbers as ``np.nanmedian(np.where(V, S, np.nan), axis=0)`` without
    materializing a stack-wide masked copy (600 MB at 4x / N=9).
    """
    return _nanmedian_axis0(
        S, mask=V,
        target_bytes=max(1, int(chunk)) * S.shape[2] * S.shape[0] * 4)


def _robust_residual_weights(S, V, frames, aligns, ratio, k, min_weight,
                             ref_hr=None, region=None):
    """Per-frame Tukey-biweight weights from *inter-frame* disagreement.

    Frames are compared to each other on the shared HR grid ``S`` (see
    :func:`_warp_frames_to_hr`), not to a forward model: a static edge lands at
    the same HR position in every correctly registered frame, whereas a moving
    subject / parallax layer lands somewhere different in each. So the robust
    reference is the across-frame **median** of ``S`` (or ``ref_hr`` when
    iterating) and each frame is weighted by a Tukey biweight of its deviation
    from that reference, scaled per HR pixel by 1.4826 * MAD across frames.

    Caveat (measured): with only a handful of frames the per-pixel MAD is a
    noisy, *small* scale, so on a near-static noisy burst it also rejects
    ordinary sensor/interpolation deviations — and, above the single-frame
    Nyquist, inter-frame aliasing is indistinguishable from ghosting. Robust
    fusion therefore costs HF/sharpness on a well-registered static burst (see
    the ``robust_fusion`` note on :class:`SuperResolution`); it is opt-in. It
    pays off where the disagreement is structural (parallax / a moving subject),
    which is orders of magnitude larger than the noise scale. Each HR weight map
    is sampled at the frame's LR pixels' projected HR positions, so the scatter
    deposit (and its GPU kernel) still takes a plain per-LR-pixel weight.
    """
    h, w = frames[0].rgb.shape[:2]
    # Memory note (B2): at ROI 1024 / 4x the HR grid is 4096^2, so every extra
    # full-size float64 temporary is 134 MB and a stack-wide one is >1 GB; the
    # robust path must not materialize them. The reference and the median-scale
    # field are built from ONE float32 masked copy, each frame's weight map is
    # built in float32 (it is only ever *sampled* at LR positions, so the final
    # per-LR weight map — the only array drizzle sees — is unchanged to within
    # 1e-7), and every full-size per-frame buffer is released before the next.
    if ref_hr is None:
        ref = _masked_temporal_median(S, V)
    else:
        ref = _to_gray(ref_hr).astype(np.float32)
    # Per-pixel median |frame - ref| (the Tukey scale), in row chunks: the
    # stack-wide NaN-masked copy + nanmedian this replaced cost ~1.2 GiB at
    # 4x / N=9, which is most of the ON path's excess over the 5 GiB budget.
    aad = np.empty(S.shape[1:], dtype=np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r0 in range(0, S.shape[1], 256):
            r1 = min(r0 + 256, S.shape[1])
            ch = S[:, r0:r1] - ref[r0:r1][None]
            np.abs(ch, out=ch)
            ch[~V[:, r0:r1]] = np.nan
            aad[r0:r1] = np.nanmedian(ch, axis=0)
            del ch
    scale = 1.4826 * np.nan_to_num(aad, nan=0.0) + 1e-6
    c = np.maximum(float(k) * scale, 1e-6).astype(np.float32)
    del aad, scale
    out = []
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)])
    for j, al in enumerate(aligns):
        u = S[j] - ref
        np.abs(u, out=u)
        u /= c
        Wh = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2,
                      0.0).astype(np.float32, copy=False)
        del u
        if region is not None:
            # Outside the gate's region there is no evidence of disagreement,
            # so the frame keeps full weight (see _ghost_gate_region).
            Wh = np.where(region, Wh, 1.0).astype(np.float32, copy=False)
        Wh[~V[j]] = 1.0
        M = to_drizzle_matrix(al, ratio).astype(np.float64)
        pr = M @ pts
        U = pr[0] / pr[2]
        Vv = pr[1] / pr[2]
        ui = np.clip(np.round(U), 0, Wh.shape[1] - 1).astype(np.int64)
        vi = np.clip(np.round(Vv), 0, Wh.shape[0] - 1).astype(np.int64)
        rw = Wh[vi, ui].reshape(h, w)
        del Wh, ui, vi, pr, U, Vv
        if min_weight > 0:
            np.maximum(rw, float(min_weight), out=rw)
        out.append(rw.astype(np.float64).reshape(h, w))
    return out


def _interframe_ghost_energy(frames, aligns, ratio, stack=None, median=None):
    """Truth-free multi-frame ghost energy (see the robust-fusion motivation).

    Wraps every frame onto the shared HR grid and measures
    ``mean |mean_k - median_k|`` over pixels covered by all frames, plus a
    gradient-normalized version and the mean inter-frame std. A single global
    homography cannot align a parallax layer or a moving subject, so those show
    up here as a positive, structure-linked disagreement; a well-registered
    static burst keeps it small. Reported in ``SRResult.metrics``.

    ``stack`` (internal) is the ``(S, V)`` pair the auto ghost gate already
    warped for the *same* frames and transforms (:func:`_warp_frames_to_hr`).
    Passing it reuses that gather instead of warping the burst a third time;
    the numbers are identical because both call sites produce the same arrays
    from the same inputs. ``None`` warps here (explicit ``robust_fusion``
    overrides have no gate stack). ``median`` (internal) is the temporal median
    the gate already computed on that stack; it equals the median computed
    here on every pixel the metric reduces over (``cov``).

    NOTE (measured, B1): with exactly two frames the across-frame median *is*
    the mean, so ``ghost_energy`` is structurally 0.0 and only
    ``interframe_std`` carries information.
    """
    if len(frames) < 2:
        return {"ghost_energy": 0.0, "ghost_energy_norm": 0.0,
                "interframe_std": 0.0}
    if stack is not None:
        S, V = stack
    else:
        S, V, _oh, _ow = _warp_frames_to_hr(frames, aligns, ratio)
    cov = V.all(axis=0)
    if int(cov.sum()) < 100:
        return {"ghost_energy": float("nan"), "ghost_energy_norm": float("nan"),
                "interframe_std": float("nan")}
    # B2 memory: no (N, #coverage) copy and no stack-wide temporary. Every
    # reduction is a frame-sized streaming pass; the temporal median is chunked.
    mean_img = S[0].astype(np.float32, copy=True)
    for j in range(1, S.shape[0]):
        mean_img += S[j]
    mean_img /= float(S.shape[0])
    med_img = (median if median is not None
               else _ghost_gate_median(S))
    acc = (S[0] - mean_img) ** 2
    for j in range(1, S.shape[0]):
        d = S[j] - mean_img
        np.multiply(d, d, out=d)
        acc += d
        del d
    np.divide(acc, float(S.shape[0]), out=acc)
    np.sqrt(acc, out=acc)
    std = float(acc[cov].mean())
    del acc
    diff = np.abs(mean_img - med_img)
    gx = cv2.Sobel(med_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(med_img, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)[cov]
    return {"ghost_energy": float(diff[cov].mean()),
            "ghost_energy_norm": float(diff[cov].mean()
                                       / (grad.mean() + 1e-9)),
            "interframe_std": std}


# --------------------------------------------------------------------------- #
# Automatic ghost/subject detection (robust_fusion=None)
# --------------------------------------------------------------------------- #
# Thresholds for :func:`_ghost_gate_dispersion`, measured on a real 9-frame
# handheld burst (NVIDIA test_data, ROI 512/1024) against synthetic bursts with
# a moving subject and with a parallax edge:
#
#   case                       dispersion tail (99.5p)   largest-component share
#   real static burst                  18.1 / 25.2          0.35 / 0.12
#   moving 14-px bar (swept)              304.5                  1.00
#   parallax edge (jittered)              275.5                  1.00
#
# 60 and 0.6 sit in the wide gap between the two populations (>2x margin above
# the worst static tail, >4x below the smallest synthetic tail). The tail is the
# 99.5th percentile of the per-HR-pixel peak |frame - across-frame median|,
# normalized by the robust noise scale 1.4826*MAD (floored at 1e-3 of the
# frame contrast so a perfectly static synthetic cannot collapse it). The
# compactness requirement is what separates a real moving subject/parallax layer
# (a coherent, connected region) from ordinary above-Nyquist aliasing, whose
# disagreement is scattered across the whole texture (largest connected
# component is a small share of the flagged pixels on the real static burst).
_GHOST_GATE_TAIL_X = 60.0
_GHOST_GATE_COMPACT = 0.6
_GHOST_GATE_PIXEL_U = 8.0

# Population detector D2 (see _ghost_gate_population). Measured on the same
# bursts plus moving-subject/parallax synthetics (B1, 2026-11-18) and re-derived
# on a 6-ROI real grid (B2, 2026-11-19):
#
#   case                              share   blob/w h   bbox fill   verdict
#   synthetic static (N=2/3/9)        0.000     -          -          off
#   moving subject, small (N=2/3/9)   0.048-0.050  1.00     0.92-0.95  ON
#   moving subject, ~30% area         0.30-0.47  1.00     0.94-0.97  ON
#   moving subject, 57% area (N=2)    0.575     0.99     0.65-0.81  ON
#   parallax layer (1.9x, N=2/3/9)    0.011-0.162  0.78-1.00  0.55-0.86 D3
#   real burst ROI 1024 (6 ROIs)      0.029-0.313  0.34-0.70  0.20-0.52  off
#   real burst ROI 512  (8 ROIs)      0.007-0.163  0.23-1.00  0.40-0.68  off
#
# B2 fix: the *flooded* shortcut (``share >= 0.25`` -> moving) is gone. It let a
# real static crop through at share 0.31 with blob 0.34/0.20, which cost 1.8 dB
# HF and ~5 GiB of RAM. The blob test (compactness + bbox fill) is now required
# in every branch. The compactness cut is what separates a moving *object* from
# a real crop's texture-following residual: worst static shares 0.14-0.31 with
# compactness 0.33-0.73 (scattered specks), while a moving subject's mask is
# 0.99-1.00 one blob at every N. bbox fill is the second gate (0.40-0.68 static,
# >= 0.65 moving) and excludes the elongated aliasing specks.
_GHOST_GATE_TILE = 32
_GHOST_GATE_POP_MULT = 3.0
_GHOST_GATE_POP_SHARE_MIN = 0.02
_GHOST_GATE_POP_COMPACT = 0.85
_GHOST_GATE_POP_EXTENT = 0.55

# Detector D3 (see _ghost_gate_spread): gain-normalized inter-frame spread
# divided by the scene's 5-95 percentile contrast, *corroborated* by spatial
# structure. Measured (B2, per-frame median gain normalization applied):
#
#   case                                     d3      compact  verdict
#   real burst, 6 ROIs x 1024/512            0.005-0.048  -      off
#   synthetic static, noise 0.0035-0.020     0.009-0.043  -      off
#   synthetic static, contrast 0.06 / 0.02   0.155 / 0.344  -     off
#   static + one frame +5% exposure          0.009      -       off
#   moving subject small / ~30% / ~57%       0.010-0.126 0.99-1.00 ON (D2/D1)
#   parallax layer 1.9x, N=9 / N=3           0.069 / 0.047  0.78-1.00 ON
#
# D3 was an inverse-SNR proxy: it fired on 24 of 72 static variants (audit),
# including a low-contrast crop and an exposure step. The fixes are (a) normalize
# each frame by its median level first, so an exposure step is not "motion", and
# (b) require the disagreement to also be spatially compact (a parallax layer is
# one coherent region; noise and wind-blown foliage are diffuse). The threshold
# rose 0.025 -> 0.060 because the static population reaches 0.048 once the
# real-burst grid is honest.
_GHOST_GATE_FLOOD_RATIO = 0.040
_GHOST_GATE_FLOOD_STRONG = 0.100
_GHOST_GATE_FLOOD_COMPACT = 0.35

# Detector D5 (see _ghost_gate_pair_at): the N=2 *pair* disagreement. With two
# frames there is no median to vote on -- np.median of two samples IS their mean
# -- so every statistic D1-D3 builds from "deviation from the across-frame
# median" measures HALF the real frame-to-frame disagreement and both frames
# deviate equally (they get the SAME robust weight, so the robust pass cannot
# change the average at all: measured, forced ON = plain to 1e-4 on the flooded
# N=2 burst). D5 measures the pair difference directly, gain-normalized, against
# the same scene-contrast scale D3 uses (so pair_ratio = 2 x D3's ratio):
#
#   case (N=2, one frame pair)             pair_ratio   D2 share   verdict
#   real burst, 12 ROIs x 512/1024         0.003-0.082  0.009-0.223  off
#   synthetic static +-5% exposure         0.010        0.000        off
#   synthetic static, noise 0.012-0.020    0.030-0.050  0.000        off
#   synthetic static, contrast 0.02-0.10   0.085-0.351  0.000        off
#   flooded moving subject (57% area)      0.137        0.366        ON
#
# So both cuts sit in a wide gap: the worst static *ratio* (0.082, a border-heavy
# (0,0) 512 crop) is 1.2x below the 0.10 cut while the flooded case is 1.4x
# above it; and the share cut (0.10) is 3.7x below the flooded 0.366 and 1.2x
# above the worst static share that also had a high ratio (0.223 there, but its
# ratio was 0.006). The share requirement is what a no-redundancy regime needs:
# a low-contrast/noisy static burst reaches pair_ratio 0.35 yet has no tile that
# stands out (share 0.000), so a pair alone is only trusted when a LARGE part of
# the frame disagrees as one population.
_GHOST_GATE_PAIR_RATIO = 0.10
_GHOST_GATE_PAIR_SHARE = 0.10

# The gate can only reason where the frames actually overlap. A crop near the
# image corner on a burst with 15 px of drift has under half its HR pixels
# covered by every frame; measured 0.409 at ROI (0,0) 512 where D2 then
# "detected" a 6%-of-frame blob that was pure border junk.
_GHOST_GATE_MIN_COVERAGE = 0.6

# D4: registration residual. When no single global warp explains the frames,
# the leftover misalignment is itself evidence that the burst is not static --
# and it is the one statistic here that *registration* measured directly rather
# than a property of the warped stack. Added with B8's corrected ECC seed: on a
# burst with a large moving subject that seed converges to a globally fitted
# warp whose error is spatially DIFFUSE (4.14 px over the whole frame), which
# leaves D2's population mask empty (share 0.003) although the subject covers
# 33% of the area -- the gate called that burst static while forced-on robust
# fusion lowered the trail error by 12% (0.02888 -> 0.02529).
#
# The statistic is the MEDIAN residual, not the mean (review round 2, N3): the
# mean lets ONE badly locked frame among eight (20 px -> mean 2.5) force robust
# fusion ON for an otherwise static burst, which cannot be distinguished from
# motion. The median requires that at least half of the registered frames
# disagree with a single global warp. Measured medians (LR px), same runs:
#   real 9-frame burst ROI 1024 0.027, ROI 512 0.593   -> OFF (74x / 3.4x margin)
#   synthetic static bursts (sigma 2e-3 noise)      0.023-0.045 -> OFF
#   aliased-burst mislocked frames (worst seen)     1.034       -> OFF (1.9x)
#   moving-subject burst, 33% area                  3.318       -> ON
#   moving-subject burst, N=3                       9.526       -> ON
# So 2.0 px sits ~2x above the worst median that must stay OFF and 1.66x below
# the weakest that must fire. With 8 frames one outlier cannot move the median.
_GHOST_GATE_RESIDUAL_MAX = 2.0


def _ghost_gate_residual_moving(residuals: Sequence[float]) -> dict:
    """D4 on a list of per-frame registration residuals (LR px).

    The registration residual measures exactly what the stack statistics
    cannot see when the fit error is spatially diffuse. The MEDIAN is used so a
    single failed lock cannot flip the decision (see the threshold note above);
    ``frac_above`` is reported alongside so a caller can see how unanimous the
    evidence is.
    """
    r = np.asarray([float(v) for v in residuals], dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"moving": False, "median": 0.0, "mean": 0.0,
                "frac_above": 0.0, "n": 0}
    med = float(np.median(r))
    return {"moving": bool(med > _GHOST_GATE_RESIDUAL_MAX),
            "median": med, "mean": float(r.mean()),
            "frac_above": float(np.mean(r > _GHOST_GATE_RESIDUAL_MAX)),
            "n": int(r.size)}


def _ghost_gate_gain_normalize(S: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Remove per-frame exposure differences before comparing frames.

    The gate looks for *structural* disagreement between frames. An exposure
    step (a real thing on a burst: aperture/shutter drift, a flickering light)
    is a pure gain on one frame and would otherwise read as a large inter-frame
    difference — the audit measured D3 firing on a static burst with one frame
    at +5% exposure. Dividing each frame by its own median level inside the
    common coverage makes an exposure change a no-op while leaving any
    geometric/layer disagreement untouched (a gain is a scalar per frame; a
    moving subject is not).

    Returns a new ``(N, oh, ow)`` float32 array (the caller's stack is shared
    with the ghost-energy metric and must not be modified in place).
    """
    cov = V.all(axis=0)
    if int(cov.sum()) < 100:
        return S.astype(np.float32, copy=True)
    n = S.shape[0]
    levels = np.array([float(np.median(S[j][cov])) for j in range(n)],
                      dtype=np.float64)
    gains = float(np.median(levels)) / np.maximum(levels, 1e-6)
    return S * gains[:, None, None].astype(np.float32)


def _ghost_gate_median(Sg: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Per-pixel temporal median, computed in row chunks.

    A plain ``np.median(Sg, axis=0)`` copies the whole ``(N, oh, ow)`` stack
    internally; at 4x that is another 600 MB at N=9. Chunking keeps the
    temporary at a few tens of MB and reads the same numbers.
    """
    return _nanmedian_axis0(
        Sg, mask=None,
        target_bytes=max(1, int(chunk)) * Sg.shape[2] * Sg.shape[0] * 4)


def _ghost_gate_dispersion_at(Sg: np.ndarray, V: np.ndarray, ref: np.ndarray,
                              cov: np.ndarray) -> Tuple[bool, float, float]:
    """D1 on an already gain-normalized stack (see _ghost_gate_dispersion)."""
    # mem3: both reductions are row-chunked; the full-stack |S-ref| buffer
    # (~604 MB at 4x / N=9) is never materialised. The samples are the same:
    # the MAD over common-coverage pixels, then the per-pixel scaled peak.
    if int(cov.sum()) == 0:
        return False, 0.0, 0.0
    mad = _masked_absdev_median_value(
        Sg, np.broadcast_to(cov, Sg.shape), ref)
    ref_cov = ref[cov]
    contrast = float(np.percentile(np.abs(ref_cov), 95)
                     - np.percentile(ref_cov, 5)) + 1e-9
    scale = max(1.4826 * mad, 1e-3 * contrast)
    u_peak = _masked_scaled_absdev_max_axis0(Sg, V, ref, scale)
    tail = float(np.percentile(u_peak[cov], 99.5))
    mask = ((u_peak > _GHOST_GATE_PIXEL_U) & cov).astype(np.uint8)
    n_high = int(mask.sum())
    if n_high == 0:
        return False, tail, 0.0
    n_cc, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    largest = int(stats[1:, 4].max()) if n_cc > 1 else 0
    compact = float(largest) / float(n_high)
    moving = (tail >= _GHOST_GATE_TAIL_X) and (compact >= _GHOST_GATE_COMPACT)
    return moving, tail, compact


def _ghost_gate_population_at(Sg: np.ndarray, V: np.ndarray, ref: np.ndarray,
                              cov: np.ndarray,
                              tile: int = _GHOST_GATE_TILE,
                              mult: float = _GHOST_GATE_POP_MULT
                              ) -> Optional[dict]:
    """D2 on an already gain-normalized stack (see _ghost_gate_population)."""
    if int(cov.sum()) < 100:
        return None
    # mem3: chunked, so the stack-wide |S-ref| temp is gone (same values).
    dev = _masked_absdev_max_axis0(Sg, V, ref)
    field = cv2.boxFilter(dev, -1, (tile, tile),
                          borderType=cv2.BORDER_REFLECT)
    del dev
    vals = field[cov]
    floor = float(np.percentile(vals, 10))
    del vals
    mask = ((field > mult * max(floor, 1e-12)) & cov).astype(np.uint8)
    share = float(mask.mean())
    n_flag = int(mask.sum())
    compact = extent = 0.0
    if n_flag:
        n_cc, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        if n_cc > 1:
            i = 1 + int(np.argmax(stats[1:, 4]))
            area = float(stats[i, 4])
            compact = area / float(n_flag)
            extent = area / max(float(stats[i, 2]) * float(stats[i, 3]), 1.0)
    # One blob, in every regime. The old ``share >= 0.25`` shortcut fired on a
    # static real crop whose mask was 0.31 of the frame in scattered specks
    # (compact 0.34, bbox fill 0.20): a large *population* is not evidence of a
    # moving *object*.
    blob = compact >= _GHOST_GATE_POP_COMPACT and extent >= _GHOST_GATE_POP_EXTENT
    moving = bool(share >= _GHOST_GATE_POP_SHARE_MIN and blob)
    return {"moving": moving, "share": share, "compact": compact,
            "extent": extent, "floor": floor, "coverage": float(cov.mean())}


def _ghost_gate_spread_at(Sg: np.ndarray, V: np.ndarray, ref: np.ndarray,
                          cov: np.ndarray, pop: Optional[dict]) -> Optional[dict]:
    """D3 on an already gain-normalized stack (see _ghost_gate_spread)."""
    if int(cov.sum()) < 100:
        return None
    # Streaming temporal std: one frame-sized temporary instead of the
    # (N, #coverage) copy a vectorized std would make (600 MB at 4x, N=9).
    acc = (Sg[0] - ref) ** 2
    for j in range(1, Sg.shape[0]):
        d = Sg[j] - ref
        np.multiply(d, d, out=d)
        acc += d
        del d
    np.divide(acc, float(Sg.shape[0]), out=acc)
    np.sqrt(acc, out=acc)
    std = float(acc[cov].mean())
    del acc
    ref_cov = ref[cov]
    contrast = float(np.percentile(np.abs(ref_cov), 95)
                     - np.percentile(ref_cov, 5)) + 1e-9
    ratio = std / contrast
    if pop is None:
        return {"moving": False, "ratio": ratio, "std": std,
                "contrast": contrast, "compact": 0.0, "share": 0.0}
    broad = (pop["share"] >= _GHOST_GATE_POP_SHARE_MIN
             and pop["compact"] >= _GHOST_GATE_FLOOD_COMPACT
             and pop["extent"] >= _GHOST_GATE_POP_EXTENT)
    populated = pop["share"] >= _GHOST_GATE_POP_SHARE_MIN
    moving = bool((ratio >= _GHOST_GATE_FLOOD_RATIO and broad)
                  or (ratio >= _GHOST_GATE_FLOOD_STRONG and populated))
    return {"moving": moving, "ratio": ratio, "std": std,
            "contrast": contrast, "compact": float(pop["compact"]),
            "share": float(pop["share"]), "broad": bool(broad)}


def _ghost_gate_pair_at(Sg: np.ndarray, V: np.ndarray, ref: np.ndarray,
                        cov: np.ndarray,
                        pop: Optional[dict]) -> Optional[dict]:
    """D5 on an already gain-normalized stack (see _ghost_gate_pair)."""
    if int(cov.sum()) < 100 or Sg.shape[0] != 2:
        return None
    # mean |pair difference|: exactly 2x D3's deviation-from-mean statistic
    # (which is mean(|d|/2)), i.e. the full frame-to-frame disagreement that
    # the N=2 median hides. The mean, not the rms: a flooded layer has a
    # heavy-tailed difference (agreement regions near 0), and the rms would let
    # the same physical case read 1.9x higher while a static crop's aliasing
    # tail reads proportionally higher too -- the mean shares D3's scale, so
    # D3's thresholds keep their meaning.
    d = Sg[0] - Sg[1]
    np.abs(d, out=d)
    pair_mean = float(d[cov].mean())
    del d
    ref_cov = ref[cov]
    contrast = float(np.percentile(np.abs(ref_cov), 95)
                     - np.percentile(np.abs(ref_cov), 5)) + 1e-9
    ratio = pair_mean / contrast
    share = float(pop["share"]) if pop is not None else 0.0
    moving = bool(ratio >= _GHOST_GATE_PAIR_RATIO
                  and share >= _GHOST_GATE_PAIR_SHARE)
    return {"moving": moving, "ratio": ratio, "mean_abs": pair_mean,
            "contrast": contrast, "share": share}


def _ghost_gate_evidence(S: np.ndarray, V: np.ndarray,
                         cache: Optional[dict] = None):
    """All detectors on ONE gain-normalized stack + ONE median.

    B2 memory: each detector used to normalize the stack and take the temporal
    median itself, and two of the three still used nan-masked copies. At 4x
    /N=9 that put the gate's transient peak near 3 GiB on top of the warped
    stack, i.e. the gate alone exceeded the project's 5 GiB budget. Normalizing
    and median-ing once, and doing every reduction in place or in row chunks,
    keeps the gate's extra memory at ~1.3 GiB.

    ``cache`` (internal) receives ``"median"``, the raw-stack temporal median
    D1 already computed, so the ghost-energy metric can reuse it instead of
    median-ing the same stack a second time.

    Returns ``None`` when the stack is degenerate, else
    ``(d1, pop, spread, pair)``; ``pair`` (D5) is ``None`` unless the burst has
    exactly two frames, since only then does it need to say anything D3 cannot.
    """
    cov = V.all(axis=0)
    if int(cov.sum()) < 100:
        return None
    # D1 on the raw stack: its scale is the MAD of |frame - median|, which a
    # per-frame gain leaves order-1 unchanged, and normalizing *moves* the peak
    # relative to that scale on a level-jumping edge (measured: a jittering
    # half-frame step fell from tail 275/compact 1.00 to 112/0.50).
    # The plain (mask=None) median is identical to the masked one on ``cov``
    # (all frames valid there), and every D1 reduction that matters runs on
    # ``cov`` — so D1 is unchanged and the same array serves the ghost metric.
    raw_med = _ghost_gate_median(S)
    d1 = _ghost_gate_dispersion_at(S, V, raw_med, cov)
    Sg = _ghost_gate_gain_normalize(S, V)
    ref = _masked_temporal_median(Sg, V)
    pop = _ghost_gate_population_at(Sg, V, ref, cov)
    spread = _ghost_gate_spread_at(Sg, V, ref, cov, pop)
    pair = _ghost_gate_pair_at(Sg, V, ref, cov, pop)
    if cache is not None:
        # The same array _interframe_ghost_energy would compute on its own.
        cache["median"] = raw_med
    return d1, pop, spread, pair


def _nanmedian_axis0(S: np.ndarray, mask: Optional[np.ndarray] = None,
                     target_bytes: int = 64 * 2 ** 20) -> np.ndarray:
    """Exact ``np.nanmedian(S, axis=0)`` without its full-stack temporaries.

    ``np.nanmedian`` on a ``(N, 4096, 4096)`` float32 stack allocates several
    full copies internally (measured peak ~3.5 GB at 4x ROI 1024, the single
    largest allocation in a run). The median is per output pixel, so computing
    it in row bands gives the identical result while bounding the temporary to
    ``target_bytes``. ``mask`` marks the valid samples (invalid -> NaN), the
    same thing ``np.where(V, S, np.nan)`` did at every existing call site.
    """
    n, hh, ww = S.shape
    out = np.empty((hh, ww), dtype=np.float32)
    rows = max(1, int(target_bytes // max(1, n * ww * 4)))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r0 in range(0, hh, rows):
            r1 = min(r0 + rows, hh)
            blk = np.array(S[:, r0:r1, :], dtype=np.float32, copy=True)
            if mask is not None:
                blk[~mask[:, r0:r1, :]] = np.nan
                # Exact select-k median, vectorised over the row band. Sorting
                # the small temporal axis puts NaNs last (np.sort), so the
                # valid count per pixel picks the two middle samples; the same
                # values np.nanmedian returns, without its masked-array
                # argsort (which dominated the 4x gate: 11 s of the 30 s run).
                blk.sort(axis=0)
                cnt = np.sum(~np.isnan(blk), axis=0)
                lo = np.maximum((cnt - 1) // 2, 0)
                hi = np.maximum(cnt // 2, 0)
                a = np.take_along_axis(blk, lo[None], axis=0)[0]
                b = np.take_along_axis(blk, hi[None], axis=0)[0]
                # All-NaN pixels stay NaN (a+b with NaN), matching nanmedian.
                out[r0:r1] = 0.5 * (a + b)
                continue
            # No mask: no NaNs, so skip the counts/gather passes and read the
            # middle rank(s) straight out of a partial sort.
            k1 = (n - 1) // 2
            k2 = n // 2
            if k1 == k2:
                blk.partition(k1, axis=0)
                out[r0:r1] = blk[k1]
            else:
                blk.partition((k1, k2), axis=0)
                out[r0:r1] = 0.5 * (blk[k1] + blk[k2])
    return out


def _masked_absdev_max_axis0(S: np.ndarray, V: np.ndarray,
                             ref: np.ndarray,
                             target_bytes: int = 64 * 2 ** 20) -> np.ndarray:
    """Exact ``where(V, |S - ref|, 0).max(axis=0)`` in row bands."""
    n, hh, ww = S.shape
    out = np.empty((hh, ww), dtype=np.float32)
    rows = max(1, int(target_bytes // max(1, n * ww * 4)))
    for r0 in range(0, hh, rows):
        r1 = min(r0 + rows, hh)
        blk = S[:, r0:r1, :] - ref[None, r0:r1, :]
        np.abs(blk, out=blk)
        blk[~V[:, r0:r1, :]] = 0.0
        out[r0:r1] = blk.max(axis=0)
    return out


def _masked_absdev_median_value(S: np.ndarray, V: np.ndarray,
                                ref: np.ndarray) -> float:
    """Exact ``median(|S - ref| over valid samples)`` with one value buffer."""
    n_valid = int(V.sum())
    if n_valid == 0:
        return 0.0
    finite = np.empty(n_valid, dtype=np.float32)
    pos = 0
    for j in range(S.shape[0]):
        blk = np.abs(S[j] - ref)
        vals = blk[V[j]]
        finite[pos:pos + vals.size] = vals
        pos += vals.size
    k = n_valid // 2
    finite.partition(k)
    if n_valid % 2:
        return float(finite[k])
    return float(0.5 * (float(finite[:k].max()) + float(finite[k])))


def _masked_scaled_absdev_max_axis0(S: np.ndarray, V: np.ndarray,
                                    ref: np.ndarray, scale: float,
                                    target_bytes: int = 64 * 2 ** 20
                                    ) -> np.ndarray:
    """Exact ``where(V, |S - ref| / scale, 0).max(axis=0)`` in row bands."""
    n, hh, ww = S.shape
    out = np.empty((hh, ww), dtype=np.float32)
    rows = max(1, int(target_bytes // max(1, n * ww * 4)))
    inv = np.float32(scale)
    for r0 in range(0, hh, rows):
        r1 = min(r0 + rows, hh)
        blk = S[:, r0:r1, :] - ref[None, r0:r1, :]
        np.abs(blk, out=blk)
        blk /= inv
        blk[~V[:, r0:r1, :]] = 0.0
        out[r0:r1] = blk.max(axis=0)
    return out


def _ghost_gate_dispersion(S: np.ndarray, V: np.ndarray
                           ) -> Tuple[bool, float, float]:
    """Detect a localized moving subject / parallax layer on the warped stack.

    ``S`` is the ``(N, oh, ow)`` luminance stack of correctly registered frames
    (see :func:`_warp_frames_to_hr`), ``V`` its validity mask. A global
    homography aligns the static scene, so ``|frame - median|`` is small noise
    everywhere except where a layer moves independently: there it is a large,
    *compact* (connected) region. Ordinary aliasing is spread over the texture,
    so its large residuals form many small, disconnected specks.

    Returns ``(moving, tail, compact)``: ``tail`` is the 99.5th percentile of the
    per-pixel peak normalized deviation; ``compact`` is the share of the
    flagged pixels (deviation > 8 robust scales) that lie in the single largest
    8-connected component. See the thresholds above.
    """
    cov = V.all(axis=0)
    if int(cov.sum()) < 100:
        return False, 0.0, 0.0
    if float(cov.mean()) < _GHOST_GATE_MIN_COVERAGE:
        # Too little common overlap for |frame - median| to mean anything: near
        # an image corner most "disagreement" is one frame's border, not motion.
        return False, 0.0, 0.0
    return _ghost_gate_dispersion_at(S, V, _masked_temporal_median(S, V), cov)


def _ghost_gate_population(S: np.ndarray, V: np.ndarray,
                           tile: int = _GHOST_GATE_TILE,
                           mult: float = _GHOST_GATE_POP_MULT
                           ) -> Optional[dict]:
    """Detect a moving layer by the *population* of disagreeing HR pixels.

    Why this exists (measured): the compact-tail gate D1
    (:func:`_ghost_gate_dispersion`) reasons on a per-pixel peak deviation
    normalized by one global MAD. A subject covering a large share of the frame
    — or a high-contrast one that drags the homography — makes that MAD large,
    so the *tail* shrinks even though most of the frame disagrees. D1 therefore
    misses a dominant/subject-following burst and, with only two frames, it
    misses a small subject too (the "median" of two frames is the mean, which
    halves every deviation).

    So: tile-smooth the peak deviation (``tile`` HR px box mean), take the
    variance floor as the 10th percentile of that field, and look at the *mask*
    of tiles above ``mult`` x floor. A moving subject/parallax layer is one
    coherent blob — large share of the frame, one connected component, filling
    most of its bounding box. A static burst's aliasing/registration residual
    is either below the floor (synthetic static burst: share 0.000) or forms
    scattered, edge-following specks (real handheld burst: largest component is
    a minority of the mask and fills only 0.20-0.68 of its bbox).

    Returns ``None`` when there is no usable coverage, else a dict with the
    measured ``share``/``compact``/``extent`` and the ``moving`` verdict.
    See the threshold notes above.
    """
    cov = V.all(axis=0)
    if int(cov.sum()) < 100:
        return None
    if float(cov.mean()) < _GHOST_GATE_MIN_COVERAGE:
        return {"moving": False, "share": 0.0, "compact": 0.0, "extent": 0.0,
                "floor": 0.0, "coverage": float(cov.mean())}
    Sg = _ghost_gate_gain_normalize(S, V)
    ref = _masked_temporal_median(Sg, V)
    return _ghost_gate_population_at(Sg, V, ref, cov, tile=tile, mult=mult)


def _ghost_gate_spread(S: np.ndarray, V: np.ndarray) -> Optional[dict]:
    """Detect a *flooded* / broad moving layer: spread + spatial structure.

    D3's raw form (mean inter-frame std / scene contrast) is an inverse-SNR
    proxy: it fires on a low-contrast crop, on a noisier exposure and on
    wind-blown foliage as readily as on a moving layer (the audit measured 24 of
    72 static variants firing). Two guards make it evidence about *motion*:

    * **gain normalization** (:func:`_ghost_gate_gain_normalize`): a per-frame
      exposure step is removed before the spread is measured;
    * **spatial corroboration**: the same tile field D2 uses must be one
      coherent region (``compact >= _GHOST_GATE_FLOOD_COMPACT``). Wind-blown
      foliage and aliasing are diffuse; a parallax layer or a moving subject is
      not.

    Two branches, both corroborated by the D2 mask (so nothing that has no
    populated, coherent region can fire):

    * ``ratio >= FLOOD_RATIO`` **and** the D2 mask is a broad coherent region
      (``share >= POP_SHARE_MIN``, ``compact >= FLOOD_COMPACT``,
      ``extent >= POP_EXTENT``) — the parallax case that D1's sharp tail and
      D2's strict blob test both miss (measured parallax: d3 0.047/0.069,
      share 0.020/0.050, compact 0.39/0.55, extent 0.82/0.84);
    * ``ratio >= FLOOD_STRONG`` and any populated mask — the flooded regime at
      N=2, where the median is a mean and the mask cannot be compact (measured
      flooded N=2: d3 0.178, share 0.108, compact 0.12). No static case
      reaches 0.10 with a populated mask: the low-contrast statics do reach
      0.10-0.34 but their mask share is exactly 0.000.
    """
    cov = V.all(axis=0)
    if int(cov.sum()) < 100:
        return None
    if float(cov.mean()) < _GHOST_GATE_MIN_COVERAGE:
        return {"moving": False, "ratio": 0.0, "std": 0.0, "contrast": 0.0,
                "compact": 0.0}
    Sg = _ghost_gate_gain_normalize(S, V)
    ref = _masked_temporal_median(Sg, V)
    pop = _ghost_gate_population_at(Sg, V, ref, cov)
    return _ghost_gate_spread_at(Sg, V, ref, cov, pop)


def _ghost_gate_pair(S: np.ndarray, V: np.ndarray) -> Optional[dict]:
    """Detect strong frame-to-frame disagreement in a *two-frame* burst (D5).

    Two frames are the degenerate case for every median-based detector: the
    across-frame median is the mean, so each frame's deviation from it is half
    the true pair difference and *both* frames sit at that same distance. D1-D3
    therefore see half the signal, and the robust pass that follows them is
    inert by construction (each frame gets the same weight, so the weighted
    average is the plain average — measured: forced ON differs from plain by
    under 1e-4 on the flooded N=2 burst).

    D5 measures the disagreement that is actually there: the rms of the
    gain-normalized pair difference, divided by the scene's 5-95 percentile
    contrast — the same scale D3 uses, so its ratio is exactly twice D3's — and
    requires the D2 tile mask to show that a large part of the frame disagrees
    as one population (``share >= _GHOST_GATE_PAIR_SHARE``). See the threshold
    note for the measured table and the margins.

    Returns ``None`` for any burst that does not have exactly two frames.
    """
    cov = V.all(axis=0)
    if int(cov.sum()) < 100 or S.shape[0] != 2:
        return None
    coverage = float(cov.mean())
    if coverage < _GHOST_GATE_MIN_COVERAGE:
        return {"moving": False, "ratio": 0.0, "mean_abs": 0.0,
                "contrast": 0.0, "share": 0.0, "coverage": coverage}
    Sg = _ghost_gate_gain_normalize(S, V)
    ref = _masked_temporal_median(Sg, V)
    pop = _ghost_gate_population_at(Sg, V, ref, cov)
    out = _ghost_gate_pair_at(Sg, V, ref, cov, pop)
    if out is not None:
        out["coverage"] = coverage
    return out


def _ghost_gate_region(S: np.ndarray, V: np.ndarray,
                       dilate_tiles: int = 2, soft: bool = False
                       ) -> Optional[np.ndarray]:
    """HR mask of the region the gate actually found disagreement in.

    Robust fusion is justified *where the gate found evidence*, not everywhere:
    down-weighting the whole frame is what costs 1.2-1.8 dB of HF on a burst
    whose disagreement is one small region (measured: a static crop containing a
    near-depth facade corner — 7% of the frame flagged, hf_gain_db 12.94 -> 11.71
    when the whole frame was re-weighted, -> within 0.15 dB with this mask).

    The mask is the union of the D1 pixel mask, the D2 tile mask and the D3
    corroboration, dilated by ``dilate_tiles`` tiles (a subject's edge and the
    registration residual around it are wider than the tile that first trips the
    test). Returns ``None`` when nothing is flagged, which means "no local
    evidence" — the caller then does not run robust fusion at all.

    ``soft=True`` returns the 1-tile ramp as a float map in ``[0, 1]`` instead
    of thresholding it to a binary mask. The reference-only fallback at N=2
    needs the ramp: a binary edge would leave a visible sharpness step where
    the fusion goes from two frames to one.
    """
    cov = V.all(axis=0)
    if int(cov.sum()) < 100:
        return None
    Sg = _ghost_gate_gain_normalize(S, V)
    ref = _masked_temporal_median(Sg, V)
    dev = _masked_absdev_max_axis0(Sg, V, ref)
    finite = dev[cov]
    mad = float(np.median(finite)) if finite.size else 0.0
    del finite
    contrast = float(np.percentile(np.abs(ref[cov]), 95)
                     - np.percentile(ref[cov], 5)) + 1e-9
    scale = max(1.4826 * mad, 1e-3 * contrast)
    mask = ((dev / scale > _GHOST_GATE_PIXEL_U) & cov)
    tile = _GHOST_GATE_TILE
    field = cv2.boxFilter(dev, -1, (tile, tile),
                          borderType=cv2.BORDER_REFLECT)
    vals = field[cov]
    if vals.size:
        floor = float(np.percentile(vals, 10))
        mask |= ((field > _GHOST_GATE_POP_MULT * max(floor, 1e-12)) & cov)
    if not mask.any():
        return None
    m8 = mask.astype(np.uint8)
    k = 2 * int(dilate_tiles) * tile + 1
    m8 = cv2.dilate(m8, np.ones((k, k), np.uint8),
                    borderType=cv2.BORDER_REFLECT)
    # soften: a 1-tile ramp so the weight map has no hard edge
    m8 = cv2.boxFilter(m8.astype(np.float32), -1, (tile, tile),
                       borderType=cv2.BORDER_REFLECT)
    m8 = np.clip(m8 * 3.0, 0.0, 1.0)
    if soft:
        return m8.astype(np.float32)
    return (m8 > 0.0).astype(np.float32)


def _auto_robust_fusion(frames: Sequence[RawFrame], aligns: Sequence, ratio: float,
                        cache: Optional[dict] = None,
                        residuals: Optional[Sequence[float]] = None
                        ) -> Tuple[bool, str, Tuple[np.ndarray, np.ndarray]]:
    """Auto (``robust_fusion=None``) decision + reason + the warped stack it used.

    Returns ``(use_robust, reason, (S, V))``. ``S``/``V`` are returned so the
    caller can hand them to :func:`drizzle` and to the ghost-energy metric,
    so the burst is gathered to the HR grid once for the whole run.

    Four independent detectors run on that one stack (see the threshold notes
    above each): D1 the compact peak-deviation tail, D2 the population/blob
    statistic, D3 the contrast-normalized inter-frame spread (the flooded
    regime, where a subject covers most of the frame and D1/D2's
    self-referential noise floor is contaminated by the subject itself) and D5
    the raw pair disagreement, which is the only honest statistic for a
    two-frame burst (see :func:`_ghost_gate_pair`).
    """
    S, V, _oh, _ow = _warp_frames_to_hr(frames, aligns, ratio)
    caveat = (" \u2014 2-frame burst: no median to vote on, so the flagged "
              "region keeps only the reference frame" if len(frames) == 2
              else "")
    cov = float(V.all(axis=0).mean())
    if cov < _GHOST_GATE_MIN_COVERAGE:
        # Honest refusal: with less than this much common coverage the
        # disagreement statistics are dominated by one frame's border, and the
        # robust re-scatter would have almost nothing to weight anyway.
        return (False, f"static burst \u2014 plain fusion (only {cov:.0%} of the "
                f"frame is covered by every frame)", (S, V))
    evidence = _ghost_gate_evidence(S, V, cache=cache)
    if evidence is None:
        return False, "static burst \u2014 plain fusion", (S, V)
    (moving, tail, compact), pop, spread, pair = evidence
    if moving:
        reason = (f"moving subject detected (dispersion tail {tail:.1f}x, "
                  f"compact {compact:.2f}){caveat}")
        return True, reason, (S, V)
    if pop is not None and pop["moving"]:
        reason = (f"moving subject detected (population share "
                  f"{pop['share']:.3f} of frame, blob {pop['compact']:.2f}/"
                  f"{pop['extent']:.2f}){caveat}")
        return True, reason, (S, V)
    if spread is not None and spread["moving"]:
        reason = (f"moving subject detected (inter-frame spread "
                  f"{spread['ratio']:.3f} of scene contrast, coherent "
                  f"{spread['compact']:.2f}){caveat}")
        return True, reason, (S, V)
    # D4 backstop: the frames cannot be explained by one global warp. The stack
    # statistics above are self-referential when the error is diffuse (every
    # pixel deviates, so no compact population stands out); the registration
    # residual from step 2 measures exactly that case. The MEDIAN keeps one bad
    # lock from forcing the decision (N3).
    #
    # It needs at least TWO non-self residuals to mean anything: with one (a
    # two-frame burst) the "median" is that single number, and a bad ECC lock
    # and a moving layer are then indistinguishable — measured on the real
    # burst at ROI (0,0) 512, where the 2-frame fit misses by 85.7 px and D4
    # would call the static crop "moving" (and replace it with single-frame
    # content). N=2 motion is D5's job below, on stack statistics that do not
    # depend on the fit succeeding.
    if residuals is not None and len(residuals) >= 2:
        d4 = _ghost_gate_residual_moving(residuals)
        if d4["moving"]:
            return True, (f"moving subject detected (registration residual "
                          f"median {d4['median']:.2f} px, "
                          f"{d4['frac_above']:.0%} of frames above "
                          f"{_GHOST_GATE_RESIDUAL_MAX:.1f} px: no single global "
                          f"warp explains the frames){caveat}"), (S, V)
    if len(frames) < 3:
        # D5, last resort for a two-frame burst: everything above reasons
        # through the across-frame median, which for two frames IS the mean, so
        # the disagreement is understated by half and both frames look equally
        # "wrong". Measure the pair difference itself (see _ghost_gate_pair).
        # The robust pass at N=2 then keeps only the reference frame inside the
        # flagged region (drizzle's reference-only fallback): with two samples
        # and no third observation to break the tie, that is the only fusion
        # that cannot ghost.
        if pair is not None and pair["moving"]:
            return True, (f"moving subject detected (2-frame pair disagreement "
                          f"{pair['ratio']:.3f} of scene contrast over "
                          f"{pair['share']:.0%} of the frame: with no median "
                          f"to vote on, the flagged region keeps only the "
                          f"reference frame)"), (S, V)
        # Two or three frames: the across-frame median the gate reasons about
        # is a mean (N=2) or barely robust (N=3), so a *detected* subject is
        # reported but the detector is weaker. Say so instead of implying the
        # gate had the same evidence as on a 9-frame burst.
        return False, (f"static burst \u2014 plain fusion ({len(frames)}-frame "
                       f"burst: limited ghost evidence)"), (S, V)
    return False, "static burst \u2014 plain fusion", (S, V)


def default_deposit_params(ratio: float) -> Tuple[float, int]:
    """Ratio-aware drizzle deposit defaults: ``(pixfrac, subsamples)``.

    Two separate knobs, measured separately on the real 9-frame burst with the
    **fixed** held-out metric (``predict_lr_from_hr`` with the run's own
    reference frame and the sensor PSF; the pre-``fix/heldout`` metric warped
    from ``frames[0]`` and its numbers are noise — anything quoted from before
    that fix, including an earlier version of this docstring, is void).

    **``pixfrac`` stays 0.5 LR px at every ratio.** ``pixfrac`` is in LR
    pixels, so 0.5 makes the deposited footprint ``0.5 * ratio`` HR pixels
    wide, and on one deposited LR impulse the deposit PSF does widen with
    ratio: rms 0.667 / 0.817 / 0.943 HR px at 2x / 3x / 4x, central-cell share
    0.56 / 0.44 / 0.31. A3 #1 proposed shrinking it to hold a constant 1-HR-px
    footprint (``min(0.5, 1/ratio)``). That fixes the PSF number and even wins
    on a synthetic burst — and it is **not** harmful on real data either, which
    is the finding that matters here: with the fixed metric the alternatives
    are neutral, not better. Held-out gain, real burst, ROI 512, four held-out
    frames, same code otherwise:

    =====  ==========  =====  =============  ============  ==========
    ratio  pixfrac     subs   held-out mean  HF (dB)       sharpness
    =====  ==========  =====  =============  ============  ==========
    3x     0.5 (ship)  3      +7.224         17.1          16.5
    3x     1/3         3      +6.999         19.5          43.4
    4x     0.5 (ship)  3      +6.873         21.1          455
    4x     1/4         3      +6.130         43.2          3631
    =====  ==========  =====  =============  ============  ==========

    (an independent audit, 4x with held-out frames {0,4}: 0.5/2 +7.030,
    0.5/3 +6.911, 1/3 +6.979, 0.25 +6.851 — i.e. spread within ~0.2 dB, all
    inside the frame-to-frame scatter this burst shows.)

    So the honest statement is: shrinking the footprint is **within a few
    tenths of a dB of neutral** in prediction while multiplying Laplacian
    sharpness by 2.6x-8x (16.5 -> 43.4 at 3x, 455 -> 3631 at 4x) — the
    acutance-without-resolution pattern this project already distrusts, and the
    reason 0.5 is kept rather than "measured to be better". Widening is clearly
    worse for the band SR exists for (4x 512, pre-fix probe: pixfrac 1.0 kept
    held-out but dropped HF gain to 8.5 dB). The PSF widening with ratio is
    therefore a known, deliberately accepted property of the shipped default.

    **``subsamples`` scales with the footprint in HR pixels.** The lattice only
    exists to leave no holes: its pitch in HR px is
    ``pixfrac * ratio / subsamples``, so with ``pixfrac = 0.5`` a fixed count
    of 3 over-samples 2x (pitch 0.33 HR px) and under-samples 4x (0.67 HR px).
    At 2x the deposit really does change slightly — with fractional shifts it
    is 0.0365 max / 0.0034 mean absolute difference (0.67 % of the value), the
    impulse PSF moves 0.04 HR px rms (0.707 vs 0.667) — and the measured
    effect is neutral-to-slightly-positive: real burst ROI 512, held-out
    +7.614 (subs 2) vs +7.615 (subs 3), HF gain 4.32 vs 4.08 dB, sharpness
    gain 2.8 vs 2.6. What it *is* is cheaper: the scatter cost scales with
    ``subsamples^2``, i.e. 2.25x of the ``np.bincount`` work (the module's
    documented bottleneck) for that 0.67 % change. The derived rule
    ``clip(ceil(2 * pixfrac * ratio), 2, 3)`` gives **2 / 3 / 3** at
    2x / 3x / 4x, so 3x/4x (which do need the count for their pitch) keep the
    previously shipped 3 and only 2x changes. A fourth sub-sample at 4x was
    not re-measured with the fixed metric (the pre-fix number this file used to
    quote is void); its impulse PSF is identical to two sub-samples' (the
    lattice is already dense at 0.5 HR px pitch) and it costs another 1.78x of
    scatter time, so the clamp is 3 and step 4 stays one explicit argument
    away.

    An explicit ``drizzle_pixfrac``/``drizzle_subsamples`` override is honoured
    as given; when only ``pixfrac`` is given the sub-sample count is still
    derived from it, so a wide explicit footprint never leaves holes.
    """
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        return 0.5, 2
    pixfrac = 0.5
    return pixfrac, deposit_subsamples(pixfrac, ratio)


def deposit_subsamples(pixfrac: float, ratio: float) -> int:
    """Cheapest sub-sample count whose lattice pitch stays <= 0.5 HR px.

    ``pixfrac`` is in LR px, so the footprint is ``pixfrac * ratio`` HR px
    wide; ``subsamples^2`` offsets fill it. Clamped to [2, 3]: 1 cannot
    represent a fractional footprint offset at all, and a fourth sub-sample
    measured as noise-level change for another 1.78x of scatter time.
    """
    try:
        pf = float(pixfrac)
        r = float(ratio)
    except (TypeError, ValueError):
        return 2
    if not np.isfinite(pf) or not np.isfinite(r) or pf <= 0 or r <= 0:
        return 2
    return int(np.clip(np.ceil(2.0 * pf * r), 2, 3))


def _reference_only_region(hr: np.ndarray, frames: Sequence[RawFrame],
                           aligns: Sequence, ratio: float, mask: np.ndarray,
                           reference_pos: int) -> np.ndarray:
    """Replace ``mask`` with the reference frame alone (the N=2 fallback).

    With two frames the robust pass cannot separate the layers: the across-frame
    median is the mean, so both frames deviate equally and receive the *same*
    Tukey weight — the weighted average is the plain average (measured: forced
    robust differs from plain by <1e-4 on the flooded N=2 synthetic burst).
    Down-weighting is therefore not a fusion policy at N=2; the only action that
    cannot ghost is to take the content from one observation.

    The reference frame is that observation: it is the frame the HR grid is
    aligned to (``reference_pos`` is the position of
    :attr:`SRResult.reference_index` within ``frames``), the anchor every
    consumer that relates ``hr`` to a frame uses, and the frame a held-out
    prediction is defined against. So the flagged region shows the subject once,
    where the reference has it, instead of a superposition of two positions.

    The fallback is a plain bicubic resample of that one frame onto the HR grid
    (no scatter: a single frame's scatter lattice at ``pixfrac=0.5`` leaves 3 of
    4 HR pixels uncovered at 2x — the deposit footprint is narrower than the LR
    sampling pitch — so it would punch a checkerboard of holes). It is therefore
    single-frame resolution inside the region, which is honest: the two frames
    contradict each other there, so there is no valid sub-pixel information to
    recover. ``alpha`` is a soft HR map in ``[0, 1]`` (1 = reference only); the
    gate's ramp already fades it, and a small Gaussian removes any remaining
    step so the transition to the fused result has no seam. Pixels the reference
    does not cover (border) keep the fused value.
    """
    h, w = frames[0].rgb.shape[:2]
    oh, ow = hr.shape[:2]
    if mask.shape != (oh, ow):
        mask = cv2.resize(np.asarray(mask, np.float32), (ow, oh),
                          interpolation=cv2.INTER_NEAREST)
    a = np.asarray(mask, np.float32)
    sigma = max(1.0, float(ratio))
    a = cv2.GaussianBlur(a, (0, 0), sigma)
    np.clip(a, 0.0, 1.0, out=a)
    M = to_drizzle_matrix(aligns[reference_pos], ratio)
    ref_rgb = cv2.warpPerspective(np.asarray(frames[reference_pos].rgb,
                                            dtype=np.float32),
                                 M, (ow, oh), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    ones = np.ones((h, w), dtype=np.float32)
    covered = cv2.warpPerspective(ones, M, (ow, oh),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT) > 0.999
    a = a * covered
    a = a[:, :, None].astype(np.float32)
    return (hr * (1.0 - a) + ref_rgb * a).astype(np.float32)


def drizzle(frames: Sequence[RawFrame],
            aligns: Sequence,
            ratio: float,
            weights: Optional[Sequence[float]] = None,
            interp: int = cv2.INTER_LINEAR,
            out_shape: Optional[Tuple[int, int]] = None,
            post_sigma: float = 0.0,
            footprint_sigma: float = 0.0,
            deposit: str = "scatter",
            pixfrac: Optional[float] = None,
            subsamples: Optional[int] = None,
            splat: str = "bilinear",
            splat_sigma: float = 0.5,
            low_cov_thresh: float = 0.0,
            low_cov_sigma: float = 0.8,
            use_opencl: bool = False,
            use_gpu_scatter: bool = False,
            robust_fusion: bool = False,
            robust_k: float = 4.685,
            robust_iters: int = 1,
            robust_min_weight: float = 0.0,
            robust_stack: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            robust_localize: bool = True,
            reference_pos: Optional[int] = None,
            engine_out: Optional[dict] = None,
            local_shifts: Optional[Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] = None,
            local_weights: Optional[Sequence[np.ndarray]] = None,
            use_cupy: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Deposit aligned frames onto an HR grid (drizzle).

    ``deposit="scatter"`` (default) is a **true drizzle** (Fruchter & Hook):
    each LR pixel's measured value is scattered onto the HR grid at its true
    sub-pixel position with a compact bilinear splat (``pixfrac`` footprint,
    ``subsamples`` sub-samples) and a weight map is accumulated. This both
    keeps the sub-pixel placement (so frames do not ghost on top of each other
    along straight edges) **and** preserves the recovered band above the
    single-frame Nyquist. A gather *interpolation* cannot do the latter: its
    reconstruction kernel's first zero sits exactly at the single-frame Nyquist
    (measured: every interpolated gather — linear/cubic/lanczos/area — costs
    several dB of high-frequency gain versus the measured scatter deposit).

    ``deposit="gather"`` reproduces the legacy behaviour and is kept for A/B
    measurement. ``interp``/``footprint_sigma`` then apply: each frame is
    pre-blurred by ``footprint_sigma`` (LR px) and resampled with ``interp``.
    In scatter mode ``interp``/``footprint_sigma`` are ignored: the footprint
    *is* the splat kernel (``pixfrac``).

    ``post_sigma`` applies a small anti-alias blur (in HR pixels) to the
    normalized result, then re-zeroes uncovered pixels, in both modes.

    ``pixfrac``/``subsamples`` default to ``None`` = **ratio-aware** (see
    :func:`default_deposit_params`): the footprint stays 0.5 LR px at every
    ratio (so it widens in HR px as the ratio grows — measured to be a
    deliberate, prediction-neutral choice) with the cheapest sub-sample lattice
    whose pitch keeps it hole-free (2 / 3 / 3 at 2x / 3x / 4x). Explicit values
    are used as given.

    ``use_opencl=True`` routes the resampling warps through OpenCV's T-API
    (OpenCL) when the output fits the transfer guard; the scatter engine
    itself is a CPU :func:`np.bincount` kernel and stays there either way.

    ``use_gpu_scatter=True`` additionally routes the *scatter deposition*
    itself onto a pyopencl GPU device (:class:`_GpuScatterDeposit`: float32
    atomic accumulation of the same bilinear/area splat, tables shared by all
    frames). Any missing or failing GPU piece falls back to the identical CPU
    deposition, per call. When ``engine_out`` (a dict) is provided, it records
    which engine ran (``engine_out['scatter'] = 'cpu'|'gpu'|'unavailable'``)
    and the scatter-phase wall seconds.

    ``robust_fusion=True`` re-scatters every frame with a per-LR-pixel
    Tukey-biweight weight measured from inter-frame disagreement (see
    :func:`_robust_residual_weights`), so a moving subject / parallax layer is
    down-weighted where it disagrees with the across-frame median. It costs
    high-frequency gain on a well-registered static burst, so it is off by
    default and opt-in for genuinely ghosted bursts. :class:`SuperResolution`
    resolves its tri-state ``robust_fusion=None`` (auto) before calling here.
    ``robust_stack`` (internal) is a precomputed ``(S, V)`` warped stack from
    :func:`_warp_frames_to_hr`, supplied by the auto gate so the burst is not
    warped twice; ``None`` computes it here.

    ``reference_pos`` (internal) is the position within ``frames`` of the
    reference frame (:attr:`SRResult.reference_index`). With exactly two frames
    the robust pass cannot average anything away -- the across-frame median is
    the mean, so both frames get the same weight -- so it instead replaces the
    gate's region with the reference frame alone (see
    :func:`_reference_only_region`). ``None`` keeps the plain Tukey weighting
    (direct callers that do not relate the HR grid to a frame); a whole-frame
    fallback is used when ``robust_localize`` is False.

    Returns (hr_rgb float32, coverage weight map float32).
    """
    h, w = frames[0].rgb.shape[:2]
    if out_shape is None:
        oh, ow = int(round(h * ratio)), int(round(w * ratio))
    else:
        oh, ow = out_shape
    if pixfrac is None or subsamples is None:
        pf_auto, sub_auto = default_deposit_params(ratio)
        if pixfrac is None:
            pixfrac = pf_auto
        elif subsamples is None:
            # Derive the sub-sample count for an explicit footprint.
            sub_auto = deposit_subsamples(pixfrac, ratio)
        subsamples = sub_auto if subsamples is None else subsamples

    if weights is None:
        weights = [1.0] * len(frames)

    import time as _time
    t_sc0 = _time.perf_counter()
    scatter = "cpu"
    acc = cov = None
    hr_fused = None

    if deposit == "scatter":
        acc, cov, scatter = _scatter_frames(
            frames, aligns, ratio, weights, oh, ow, pixfrac, subsamples,
            splat, splat_sigma, use_gpu_scatter, local_weights, scatter,
            local_shifts=local_shifts, use_cupy=use_cupy)
        if robust_fusion and int(robust_iters) > 0 and len(frames) >= 2:
            # Robust fusion: re-scatter with per-pixel Tukey-biweight weights
            # measured from *inter-frame* disagreement on the shared HR grid.
            # Every frame is gathered to that grid once (_warp_frames_to_hr)
            # and compared to the across-frame median there, so a static edge
            # keeps full weight in all frames while a moving subject / parallax
            # layer is down-weighted in the frames it disagrees with the rest.
            # A minority subject therefore cannot drag the reference: the
            # median ignores it. Later passes (robust_iters>1) refine against
            # the robust HR estimate instead of the median.
            if robust_stack is not None:
                S, V = robust_stack
            else:
                S, V, _oh_w, _ow_w = _warp_frames_to_hr(frames, aligns, ratio)
            hr_cur = (acc / np.maximum(cov, 1e-8)[:, :, None]).astype(np.float32)
            hr_cur[cov < 1e-6] = 0.0
            if reference_pos is not None and len(frames) == 2:
                # Two frames carry no redundancy: the across-frame median IS the
                # mean, so both frames get the same Tukey weight and re-weighting
                # leaves the average unchanged (<1e-4 measured). Do what is
                # actually robust with one spare sample instead -- keep the
                # reference frame alone where the pair disagrees (see
                # _reference_only_region). The fallback alpha comes from the
                # inter-frame disagreement itself, dilated by a single tile: the
                # weight path's 2-tile dilation exists to cover a subject's
                # *edge* for a down-weight, but replacing that much of the frame
                # with single-frame content costs HF far outside the contradiction
                # (measured: 7.72 -> 1.58 dB on a 34-px subject).
                del acc
                if robust_localize:
                    alpha = _ghost_gate_region(S, V, dilate_tiles=1, soft=True)
                else:
                    alpha = np.ones((oh, ow), np.float32)
                if alpha is not None:
                    hr_cur = _reference_only_region(
                        hr_cur, frames, aligns, ratio, alpha, reference_pos)
                hr_fused = hr_cur
            else:
                # B2: re-weight only where the gate found disagreement (see
                # _ghost_gate_region). Down-weighting a whole frame that is 93%
                # static is what made an ON decision cost 1.2-1.8 dB of HF.
                region = _ghost_gate_region(S, V) if robust_localize else None
                # B2 memory: the plain accumulation tables are dead once hr_cur
                # exists, and the robust pass allocates a fresh pair. Release
                # them first — at ROI 1024 / 4x these are 403 MB + 134 MB, which
                # is the difference between fitting the project's 5 GiB cap and
                # not.
                del acc, cov
                for _it in range(int(robust_iters)):
                    ref = None if _it == 0 else hr_cur
                    rwts = _robust_residual_weights(
                        S, V, frames, aligns, ratio, robust_k,
                        robust_min_weight, ref_hr=ref, region=region)
                    if local_weights is not None:
                        rwts = [r * lw for r, lw in zip(rwts, local_weights)]
                    acc, cov, scatter = _scatter_frames(
                        frames, aligns, ratio, weights, oh, ow, pixfrac,
                        subsamples, splat, splat_sigma, use_gpu_scatter, rwts,
                        scatter, local_shifts=local_shifts, use_cupy=use_cupy)
                    hr_cur = (acc / np.maximum(cov, 1e-8)[:, :, None]).astype(
                        np.float32)
                    hr_cur[cov < 1e-6] = 0.0
                    if _it + 1 < int(robust_iters):
                        del acc, cov
    else:
        acc = np.zeros((oh, ow, 3), dtype=np.float64)
        cov = np.zeros((oh, ow), dtype=np.float64)
        ones = np.ones((h, w), dtype=np.float32)
        for fr, al, wt in zip(frames, aligns, weights):
            wt = float(wt)
            M = to_drizzle_matrix(al, ratio)
            src = np.asarray(fr.rgb, dtype=np.float32)
            if footprint_sigma and footprint_sigma > 0:
                k = int(np.ceil(3.0 * footprint_sigma)) | 1
                src = cv2.GaussianBlur(src, (k, k), footprint_sigma,
                                       borderType=cv2.BORDER_REPLICATE)
            warped = _opencl_warp(src, M, (ow, oh), interp, use_opencl)
            mask = _opencl_warp(ones, M, (ow, oh), cv2.INTER_LINEAR,
                                use_opencl)
            acc += warped * wt
            cov += mask * wt

    if engine_out is not None:
        engine_out["scatter"] = scatter
        engine_out["seconds"] = _time.perf_counter() - t_sc0

    if hr_fused is not None:
        # The N=2 reference-only fallback already produced the fused image; the
        # coverage map is still the plain pass's (what the deposit measured).
        hr = hr_fused
    else:
        cov_safe = np.maximum(cov, 1e-8)
        hr = (acc / cov_safe[:, :, None]).astype(np.float32)
        if (deposit == "scatter" and low_cov_thresh > 0.0
                and low_cov_sigma > 0.0):
            # Normalised-convolution regularisation for sparsely covered HR
            # cells (cov < low_cov_thresh * E[cov], where E[cov] = sum(w)/r^2).
            # At 2x with 9 frames E[cov]=2.25 and almost no interior pixel is
            # below 0.5*E[cov]; at 3x/4x ~4.5-10% of interior cells have low
            # effective sample weight (down to ~0.004 at 4x) where raw acc/cov
            # amplifies noise. Blending only those cells toward the local
            # weighted mean blur(acc)/blur(cov) leaves well-covered cells
            # bit-identical and preserves CPU/GPU scatter parity.
            exp_cov = float(np.sum(weights)) / max(float(ratio) ** 2, 1e-8)
            thresh = float(low_cov_thresh) * exp_cov
            low_mask = (cov > 1e-6) & (cov < thresh)
            if thresh > 0.0 and np.any(low_mask):
                k = int(np.ceil(3.0 * low_cov_sigma)) | 1
                acc_b = cv2.GaussianBlur(
                    acc.astype(np.float32, copy=False), (k, k), low_cov_sigma,
                    borderType=cv2.BORDER_REPLICATE)
                cov_b = cv2.GaussianBlur(
                    cov.astype(np.float32, copy=False), (k, k), low_cov_sigma,
                    borderType=cv2.BORDER_REPLICATE)
                acc_b /= np.maximum(cov_b, 1e-8)[:, :, None]
                del cov_b
                t = np.clip(cov[low_mask] / thresh, 0.0, 1.0).astype(np.float32)
                hr[low_mask] = (t[:, None] * hr[low_mask]
                                + (1.0 - t[:, None]) * acc_b[low_mask])
                del acc_b
    # Zero out pixels no frame covered.
    hr[cov < 1e-6] = 0.0

    if post_sigma and post_sigma > 0:
        k = int(np.ceil(3.0 * post_sigma)) | 1
        hr = cv2.GaussianBlur(hr, (k, k), post_sigma,
                              borderType=cv2.BORDER_REPLICATE)
        hr[cov < 1e-6] = 0.0
    return hr, cov.astype(np.float32)


def _vst_burst_noise_sigma(frames: Sequence[RawFrame],
                           weights: Sequence[float],
                           residual_mads: Optional[Sequence[float]] = None,
                           eps: float = 1e-4) -> float:
    """Estimate fused-burst noise std in Anscombe/sqrt space (fused frames only).

    In linear RAW, photon shot noise scales as ``sqrt(I)``. Applying the
    variance-stabilising transform ``y = sqrt(max(I, 0) + eps)`` equalises the
    noise floor between bright flat regions (e.g. sky) and dark micro-contrast
    (e.g. roof tiles).

    When IBP forward-model residuals ``residual_mads`` (per-frame MAD of
    ``sqrt(obs) - sqrt(sim)``) are available, the temporal noise estimate
    ``median(mad) / (0.6745 * sqrt(N_eff - 1))`` cancels static scene texture
    so noise-free high-frequency scenes are not mistaken for sensor noise.
    Taking the minimum of the spatial and temporal estimates is conservative
    against both static texture and registration outliers.
    """
    if len(frames) < 2:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    wsum = float(np.sum(w))
    if wsum > 0.0:
        w = w / wsum
        n_eff = 1.0 / max(float(np.sum(w * w)), 1e-9)
    else:
        n_eff = float(len(frames))
    if n_eff <= 1.05:
        return 0.0
    nvs = []
    stds = []
    for fr in frames:
        g_vst = np.sqrt(
            np.maximum(np.asarray(fr.green, dtype=np.float32), 0.0)
            + float(eps))
        nvs.append(_noise_variance(g_vst))
        stds.append(float(np.std(g_vst)))
    frame_sigma = float(np.sqrt(max(float(np.median(nvs)), 0.0)))
    frame_std = float(np.median(stds))
    # Validity guard: a spatial high-pass estimator is only measuring sensor
    # noise when the high-pass std is well below the image's own total std.
    # On a pure high-frequency / aliased texture (e.g. a bandpass grating),
    # high-pass energy IS the signal (frame_sigma / frame_std >= 0.5), so the
    # proximal denoiser must stand down rather than blur real structure.
    if frame_std <= 1e-8 or frame_sigma >= 0.5 * frame_std:
        return 0.0
    spatial = float(frame_sigma / np.sqrt(n_eff))
    if residual_mads:
        med_mad = float(np.median(residual_mads))
        temporal = float((med_mad / 0.67448975) / np.sqrt(max(n_eff - 1.0, 1.0)))
        return min(spatial, temporal)
    return spatial


def _vst_bilateral_prox(hr: np.ndarray, ratio: float, sigma_vst: float,
                        color_scale: float = 2.5, space_lr: float = 0.75,
                        eps: float = 1e-4, max_d: int = 11,
                        in_place: bool = False) -> np.ndarray:
    """Variance-stabilised edge-preserving proximal step for MAP-IBP.

    Applies a bilateral filter in ``sqrt(max(I, 0) + eps)`` space with range
    width ``color_scale * max(ratio / 2, 1) ** 0.75 * sigma_vst`` (calibrated to
    the burst's own Poisson noise floor and scaled by
    ``max(ratio / 2, 1) ** 0.75`` because HR expected sample density
    ``E[cov] = sum(w) / ratio^2`` scales noise std with ``ratio``) and spatial
    width ``space_lr * ratio`` HR pixels
    (matching the LR-pixel noise correlation scale). Because every bilateral
    weight is non-negative and normalised to 1, the step is monotone and cannot
    introduce ringing or overshoot.
    """
    ratio_scale = max(float(ratio) / 2.0, 1.0) ** 0.75
    sigma_c = float(color_scale) * ratio_scale * float(sigma_vst)
    sigma_s = float(space_lr) * float(ratio)
    if sigma_c <= 1e-7 or sigma_s <= 1e-6:
        return hr if (in_place and hr.dtype == np.float32) else hr.astype(np.float32, copy=True)
    d = min(max(3, int(np.ceil(2.0 * sigma_s)) * 2 + 1), int(max_d))
    if in_place and hr.dtype == np.float32:
        y = hr
        np.maximum(y, 0.0, out=y)
    else:
        y = np.maximum(np.asarray(hr, dtype=np.float32), 0.0)
    y += float(eps)
    np.sqrt(y, out=y)
    yf = cv2.bilateralFilter(y, d, sigma_c, sigma_s,
                             borderType=cv2.BORDER_REPLICATE)
    del y
    np.multiply(yf, yf, out=yf)
    yf -= float(eps)
    np.clip(yf, 0.0, 1.0, out=yf)
    return yf


def iterative_back_projection(hr: np.ndarray,
                              frames: Sequence[RawFrame],
                              aligns: Sequence,
                              ratio: float,
                              weights: Optional[Sequence[float]] = None,
                              iterations: int = 1,
                              lam: float = 0.5,
                              reg_sigma: float = 0.5,
                              max_step: Optional[float] = 0.1,
                              forward_sigma: Optional[float] = None,
                              coverage: Optional[np.ndarray] = None,
                              joint_refine: bool = False,
                              joint_refine_damping: float = 0.6,
                              joint_refine_max_step: float = 0.5,
                              forward_sigma_shrink: float = 1.0,
                              adjoint: bool = True,
                              use_opencl: bool = False,
                              prox_noise_scale: float = 2.5,
                              prox_space_lr: float = 0.75,
                              use_gpu: Optional[bool] = None,
                              engine_out: Optional[dict] = None
                              ) -> np.ndarray:
    """Classical iterative back-projection (IBP) refinement of the HR estimate.

    IBP (Irani & Peleg, 1991) enforces data consistency: an HR image is correct
    if, when it is pushed back through the *imaging model* (warp + downsample),
    it reproduces each measured LR frame. Each iteration forms, per frame,

        ``r_k = obs_k - D U_k hr``            (LR residual)
        ``hr <- hr + lam * sum_k w_k U_k^T r_k``  (back-projection update)

    where ``U_k`` is the frame's deposition operator (the same ``M = S inv(W)``
    homography used by :func:`drizzle`, LR source -> HR grid), ``D U_k`` is the
    forward projection (blur the HR estimate to the sensor's pixel support,
    then sample it at the LR grid of frame ``k``), and ``U_k^T`` places the LR
    residual back on the HR grid. The forward blur matters: sampling the HR
    grid directly at the LR grid would alias away the very subpixel detail IBP
    is meant to recover and make the iteration unstable. The residual is
    weighted by the frame weight and its deposition coverage, then normalized,
    so each HR pixel receives a weighted average of real measurement errors.

    Regularization keeps noise from being amplified: the correction is smoothed
    with a small Gaussian (``reg_sigma``, HR px) and clipped to ``max_step``
    before it is added, and the result is clamped to [0, 1]. This is a
    data-consistency method — every correction derives from measured RAW
    samples, no detail is invented.

    Args:
        hr: initial HR image from :func:`drizzle` (H*r, W*r, 3).
        frames: the kept LR frames (same order as ``aligns``).
        aligns: ref->mov transforms for each frame (translation or homography).
        ratio: HR/LR scale.
        weights: per-frame (normalized) fusion weights.
        iterations: number of back-projection sweeps (0 disables IBP).
        lam: back-projection step size.
        reg_sigma: Gaussian sigma (HR px) applied to the correction; <=0 off.
        max_step: clip the per-iteration correction to +/- this value; None off.
        forward_sigma: Gaussian sigma (HR px) of the sensor-support blur in the
            forward model. ``None`` uses ``0.29 * ratio`` (the std of a uniform
            LR-pixel integration window).
        coverage: drizzle coverage map; uncovered pixels are kept at 0.
        joint_refine: error-back-projection (EBP). When True, after each sweep
            the residual translation between every frame and the *current* HR
            estimate is re-estimated by phase correlation at LR and that frame's
            deposition transform is nudged by it (damped, clipped), so residual
            registration error is not baked into the fixed transforms. Off by
            default: once sub-pixel deposition and edge-weighted/affine
            registration are in place it measured as neutral (border/center
            residual and edge ZNCC unchanged, edge ghost energy slightly worse),
            because the residual translation it chases is already below the
            deposition footprint; it is kept available for experiments.
        joint_refine_damping: fraction of the measured residual applied per
            sweep when ``joint_refine`` is on.
        joint_refine_max_step: clip on the per-frame residual (LR px).
        forward_sigma_shrink: per-sweep multiplier on ``forward_sigma`` (the
            forward-model PSF shrinks across sweeps, resolving finer structure
            later). 1.0 disables it.
        adjoint: when True (default) the back-projection applies the adjoint of
            the **assumed** forward model (Gaussian sensor-support PSF + bilinear
            sampling): the accumulated residual correction is blurred by the
            same PSF the forward pass used, so the update adds mainly frequency
            the sensors measured. It is the exact adjoint of that Gaussian+sample
            operator, *not* of the drizzle deposit k; with the ``reg_sigma``
            smoothing, the ``max_step`` clip and the ``norm`` preconditioning
            the step is a preconditioned, clipped descent direction on
            ``||obs - D U hr||^2`` rather than an exact steepest-descent step.
            ``False`` reproduces the legacy plain splat (no PSF in the
            back-projection), which measured 0.2-0.4 dB *lower* held-out PSNR
            at 2x/3x/4x while reading slightly sharper on screen (HF band
            +0.1-0.3 dB): acutance that is not reproducible on an unseen frame.
        use_opencl: route the upsample-direction warps (per-frame footprints
            and residual back-projections) through OpenCV's T-API/OpenCL when
            their output fits the transfer guard; the forward (downsample)
            warps and blurs stay on the CPU, where they measured faster than
            the device. Any device failure falls back to the CPU per call.
        prox_noise_scale: multiplier on the burst's variance-stabilised
            (Anscombe/sqrt) noise std for the MAP-IBP edge-preserving bilateral
            proximal step (default 2.5, chosen on synthetic GT burst HR PSNR).
            ``<= 0`` or ``reg_sigma <= 0`` disables the proximal step.
        prox_space_lr: spatial sigma of the bilateral proximal step in LR
            pixels (default 0.75, i.e. ``0.75 * ratio`` HR px).

    Returns:
        Refined HR image (float32), same shape as ``hr``.
    """
    if iterations <= 0 or len(frames) == 0:
        if engine_out is not None:
            engine_out["ibp"] = "cpu"
        return hr.astype(np.float32, copy=True)

    if use_gpu is not False and not joint_refine:
        from .gpu import resolve_use_gpu, iterative_back_projection_gpu
        if resolve_use_gpu(use_gpu, "ibp", int(hr.shape[0]) * int(hr.shape[1])):
            gpu_hr = iterative_back_projection_gpu(
                hr, frames, aligns, ratio,
                weights=weights, iterations=iterations, lam=lam,
                reg_sigma=reg_sigma, max_step=max_step,
                forward_sigma=forward_sigma, coverage=coverage,
                forward_sigma_shrink=forward_sigma_shrink,
                adjoint=adjoint,
                prox_noise_scale=prox_noise_scale,
                prox_space_lr=prox_space_lr,
            )
            if gpu_hr is not None:
                if engine_out is not None:
                    engine_out["ibp"] = "cupy"
                return gpu_hr
    if engine_out is not None:
        engine_out["ibp"] = "cpu"

    hr = hr.astype(np.float32).copy()
    oh, ow = hr.shape[:2]
    h, w = frames[0].rgb.shape[:2]
    ones = np.ones((h, w), dtype=np.float32)
    if weights is None:
        weights = [1.0] * len(frames)
    if forward_sigma is None:
        forward_sigma = SENSOR_PSF_SIGMA_PER_RATIO * float(ratio)

    # Deposition operators and their footprints (computed once). ``M`` maps the
    # frame's LR coordinates to the HR grid; its inverse is the forward map that
    # samples the HR estimate back onto that frame's LR grid.
    matrices = [to_drizzle_matrix(al, ratio).astype(np.float64) for al in aligns]
    matrices_inv = [np.linalg.inv(M).astype(np.float32) for M in matrices]
    # Per-frame footprints used to weight the back-projection. Default path
    # recomputes the ones-image warp inside each sweep instead of keeping an
    # (N, oh, ow) list (~604 MB at 4x ROI 1024, 9 frames); the warp is
    # deterministic, so the numbers are unchanged and the extra per-frame warp
    # is cheap next to the residual warp. ``joint_refine`` updates the
    # transforms between sweeps, and the historical (fixed) footprints are
    # kept there so its output does not change.
    store_footprints = bool(joint_refine)
    footprints = ([_opencl_warp(ones, M.astype(np.float32), (ow, oh),
                                cv2.INTER_LINEAR, use_opencl)
                   for M in matrices] if store_footprints else None)
    norm = np.zeros((oh, ow), dtype=np.float32) if store_footprints else None
    if footprints is not None:
        for m, wt in zip(footprints, weights):
            norm += m * float(wt)
    norm_safe = np.maximum(norm, 1e-6) if norm is not None else None
    fs_k = 0
    if forward_sigma and forward_sigma > 0:
        fs_k = int(np.ceil(3.0 * forward_sigma)) | 1

    last_vst_mads: List[float] = []
    for it in range(int(iterations)):
        # Forward model: integrate the HR estimate over an LR pixel, then
        # sample it on each frame's LR grid. The sensor-support PSF may shrink
        # across sweeps (EBP): early sweeps are conservative, later sweeps
        # resolve finer structure.
        sigma_it = float(forward_sigma) * (float(forward_sigma_shrink) ** it)
        fs_it = 0
        if sigma_it and sigma_it > 0:
            fs_it = int(np.ceil(3.0 * sigma_it)) | 1
        hr_fwd = hr
        if fs_it:
            hr_fwd = cv2.GaussianBlur(hr, (fs_it, fs_it), sigma_it,
                                      borderType=cv2.BORDER_REPLICATE)
        corr = np.zeros_like(hr)
        norm_pending = norm is None
        if norm_pending:
            norm = np.zeros((oh, ow), dtype=np.float32)
        collect_vst = (it == int(iterations) - 1
                       and bool(reg_sigma and reg_sigma > 0
                                and prox_noise_scale and prox_noise_scale > 0
                                and prox_space_lr and prox_space_lr > 0))
        if collect_vst:
            last_vst_mads = []
        for k, (fr, M, M_inv, wt) in enumerate(zip(frames, matrices,
                                                   matrices_inv, weights)):
            if footprints is not None:
                m = footprints[k]
            else:
                m = _opencl_warp(ones, M.astype(np.float32), (ow, oh),
                                 cv2.INTER_LINEAR, use_opencl)
            # Forward projection: HR estimate -> this frame's LR grid.
            sim = cv2.warpPerspective(hr_fwd, M_inv, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
            if collect_vst:
                # Temporal noise estimate in VST space on valid interior
                # pixels (where sim > 0 avoids BORDER_CONSTANT rims).
                g_obs = np.asarray(fr.rgb[:, :, 1], dtype=np.float32)
                g_sim = sim[:, :, 1]
                valid = g_sim > 1e-6
                if np.any(valid):
                    dv = (np.sqrt(np.maximum(g_obs[valid], 0.0) + 1e-4)
                          - np.sqrt(np.maximum(g_sim[valid], 0.0) + 1e-4))
                    last_vst_mads.append(float(np.median(np.abs(dv))))
            residual = np.asarray(fr.rgb, dtype=np.float32) - sim
            # Back-projection: LR residual -> HR grid (adjoint placement).
            up = _opencl_warp(residual, M.astype(np.float32), (ow, oh),
                              cv2.INTER_LINEAR, use_opencl)
            if norm_pending:
                norm += m * float(wt)
            if footprints is not None:
                wgt = m * float(wt)
            else:
                m *= float(wt)   # fresh per-sweep array: safe in place
                wgt = m
            np.multiply(up, wgt[:, :, None], out=up)
            corr += up
        if norm_safe is None:
            norm_safe = np.maximum(norm, 1e-6)
        corr /= norm_safe[:, :, None]

        # Adjoint of the assumed forward model (Gaussian sensor PSF + bilinear
        # sampling): the forward pass blurred the HR estimate by the
        # sensor-support PSF before sampling, so the adjoint splats the
        # residual and then blurs it by the same PSF (the blur is linear and
        # identical for every frame, so it commutes out of the weighted sum).
        # It is exact for that operator, not for the drizzle deposit; with the
        # reg_sigma blur and max_step clip below the step is a preconditioned,
        # clipped descent direction. Without it the update injects frequency
        # the sensors never measured — sharp on screen, not reproducible on a
        # held-out frame.
        if adjoint and fs_it:
            corr = cv2.GaussianBlur(corr, (fs_it, fs_it), sigma_it,
                                    borderType=cv2.BORDER_REPLICATE)

        if reg_sigma and reg_sigma > 0:
            k = int(np.ceil(3.0 * reg_sigma)) | 1
            corr = cv2.GaussianBlur(corr, (k, k), reg_sigma,
                                    borderType=cv2.BORDER_REPLICATE)
        if max_step is not None and max_step > 0:
            corr = np.clip(corr, -float(max_step), float(max_step))

        hr += float(lam) * corr
        np.clip(hr, 0.0, 1.0, out=hr)
        if coverage is not None:
            hr[coverage < 1e-6] = 0.0
        del hr_fwd, corr, up, m

        # Optional EBP: re-estimate each frame's residual translation against
        # the current HR estimate and update its deposition transform.
        if joint_refine:
            hr_chk = hr
            if fs_it:
                hr_chk = cv2.GaussianBlur(hr, (fs_it, fs_it), sigma_it,
                                          borderType=cv2.BORDER_REPLICATE)
            for j in range(len(frames)):
                sim = cv2.warpPerspective(hr_chk, matrices_inv[j], (w, h),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT)
                try:
                    ddx, ddy = estimate_translation(sim[:, :, 1],
                                                    frames[j].green)
                except cv2.error:
                    continue
                if not (np.isfinite(ddx) and np.isfinite(ddy)):
                    continue
                ddx = float(np.clip(ddx, -joint_refine_max_step,
                                    joint_refine_max_step))
                ddy = float(np.clip(ddy, -joint_refine_max_step,
                                    joint_refine_max_step))
                # obs(q) ~= sim(q + d)  =>  M_new = M @ T(+damping * d)
                T = translation_matrix(joint_refine_damping * ddx,
                                       joint_refine_damping * ddy)
                matrices[j] = (matrices[j] @ T).astype(np.float64)
                matrices_inv[j] = np.linalg.inv(matrices[j]).astype(np.float32)
                footprints[j] = _opencl_warp(
                    ones, matrices[j].astype(np.float32), (ow, oh),
                    cv2.INTER_LINEAR, use_opencl)
            norm = np.zeros((oh, ow), dtype=np.float32)
            for m, wt in zip(footprints, weights):
                norm += m * float(wt)
            norm_safe = np.maximum(norm, 1e-6)

    # Release loop buffers before the proximal step so peak RSS stays bounded.
    del norm, norm_safe, footprints
    # MAP-IBP proximal step: variance-stabilised edge-preserving denoising
    # calibrated to the fused burst's own Poisson noise floor. Disabled when
    # reg_sigma <= 0 (unregularised data-consistency mode) or prox_noise_scale
    # <= 0.
    if (reg_sigma and reg_sigma > 0
            and prox_noise_scale and prox_noise_scale > 0
            and prox_space_lr and prox_space_lr > 0):
        sigma_vst = _vst_burst_noise_sigma(frames, weights,
                                           residual_mads=last_vst_mads)
        hr = _vst_bilateral_prox(hr, ratio, sigma_vst,
                                 color_scale=float(prox_noise_scale),
                                 space_lr=float(prox_space_lr),
                                 in_place=True)
        if coverage is not None:
            hr[coverage < 1e-6] = 0.0
    return hr


# --------------------------------------------------------------------------- #
# Metrics (evidence)
# --------------------------------------------------------------------------- #

def _to_gray(img: np.ndarray, out: Optional[np.ndarray] = None
             ) -> np.ndarray:
    """BT.601 luma. ``out`` writes into an existing float32 plane (B2: the
    warped stack is filled per frame instead of collecting full-size arrays)."""
    if img.ndim == 3:
        if img.dtype == np.float32:
            # One fused SIMD pass (cv2.transform) instead of three temporaries;
            # float32 rounding only (~1e-7), the same luma definition.
            return cv2.transform(img, _LUMA_BT601, dst=out)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if out is None:
            return 0.299 * r + 0.587 * g + 0.114 * b
        np.multiply(r, 0.299, out=out)
        out += 0.587 * g
        out += 0.114 * b
        return out
    if out is not None:
        out[...] = img
        return out
    return img


def high_frequency_gain_db(sr: np.ndarray, baseline: np.ndarray,
                           cutoff: float = 0.25) -> float:
    """Energy ratio (dB) in the band above the single-frame Nyquist.

    For a ratio-r output, the input Nyquist sits at 0.5/r cycles/output-pixel;
    for r=2 that is 0.25. Genuine multi-frame SR puts real signal energy there;
    a bicubic-upsampled single image has essentially none.

    Inputs should be view-transformed (see :func:`to_view_float`) when working
    with dark, noisy linear data: in linear space the band is dominated by
    sensor noise and the ratio can read negative even for a correct SR.
    """
    def band_energy(img: np.ndarray) -> float:
        g = _to_gray(img).astype(np.float32)
        g = g - g.mean()
        win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
        # cv2.dft (float32, SIMD) measures the same power spectrum as
        # np.fft.fft2 but 6x faster and at half the memory (complex64 vs
        # complex128); both are unscaled, so the band sum is comparable. The
        # old float64 path cost 8 s of the 4x run.
        F = cv2.dft(g * win, flags=cv2.DFT_COMPLEX_OUTPUT)
        re = F[:, :, 0]
        im = F[:, :, 1]
        # |F|^2 in place: keeps one float32 plane, no fftshift copy.
        P = re * re
        P += im * im
        del F, re, im
        h, w = P.shape
        fy2 = np.fft.fftfreq(h)[:, None] ** 2
        fx2 = np.fft.fftfreq(w)[None, :] ** 2
        mask = (fy2 + fx2) >= cutoff * cutoff
        return float(P[mask].sum())

    e_sr = band_energy(sr)
    e_b = band_energy(baseline)
    return 10.0 * np.log10((e_sr + 1e-12) / (e_b + 1e-12))


def laplacian_sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian of the luminance — a sharpness proxy."""
    g = _to_gray(img).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def to_view_float(img: np.ndarray, gamma: float = 2.2,
                  exposure: float = 4.0) -> np.ndarray:
    """Apply the display transform in float (same curve as the PNG preview).

    Dark linear RAW data has to be stretched before frequency metrics mean
    anything, otherwise the >Nyquist band is pure sensor noise.
    """
    v = np.clip(np.asarray(img, dtype=np.float32) * float(exposure), 0.0, 1.0)
    # cv2.pow is the same elementwise power in one SIMD pass (float32, ~4x
    # faster than np.power at product size; max difference 1 ulp).
    return cv2.pow(v, 1.0 / max(float(gamma), 1e-6))


def denoised_lr_stack(frames: Sequence[RawFrame],
                      aligns: Sequence,
                      weights: Optional[Sequence[float]] = None,
                      use_opencl: bool = False,
                      coverage_normalised: bool = False,
                      return_coverage: bool = False):
    """Align and average frames at native resolution (denoised LR baseline).

    ``aligns`` may be translations or 3x3 homographies (same convention as
    :func:`drizzle`).

    ``use_opencl`` is accepted for API symmetry but ignored: these warps are
    same-resolution (not upsample-direction) and measured slower through
    T-API than on the CPU, so routing them would only lose time.

    ``coverage_normalised`` divides each output pixel by the weight of the
    frames that *actually covered it*, i.e. it is a weighted mean per pixel
    (the drizzle normalisation). The default (``False``) divides by the total
    weight of all frames, which is only correct where every frame overlaps:
    with real registration shifts (up to ~14 LR px on the 9-frame burst) a
    shifted frame leaves a rim its neighbours do not fill, the accumulator is
    ~0 there, and the result is a dark band along all four borders. That band
    is not a property of the measurement and must never be scored.

    ``return_coverage`` also returns the coverage map, ``sum_i w_i * c_i(x)``
    in [0, ``sum_i w_i``], which is 1-scaled where every frame contributed.
    Callers that need a common valid region pass it through the same warp as
    the stack (see ``scripts/sr_acceptance.py``).
    """
    h, w = frames[0].rgb.shape[:2]
    acc = np.zeros((h, w, 3), dtype=np.float32)
    cov = np.zeros((h, w), dtype=np.float32)
    tot = 0.0
    if weights is None:
        weights = [1.0] * len(frames)
    ones = np.ones((h, w), dtype=np.float32)
    for fr, al, wt in zip(frames, aligns, weights):
        M = to_drizzle_matrix(al, 1.0)
        acc += cv2.warpPerspective(np.asarray(fr.rgb, dtype=np.float32), M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT) * wt
        cov += cv2.warpPerspective(ones, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT) * float(wt)
        tot += float(wt)
    if coverage_normalised:
        stack = acc / np.maximum(cov, 1e-6)[..., None]
    else:
        stack = acc / max(tot, 1e-8)
    if return_coverage:
        return stack, cov
    return stack


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# HR-size memory guard
# --------------------------------------------------------------------------- #
# Conservative estimate of a run's peak bytes per HR pixel, measured on the
# real 9-frame burst at 4x ROI 1024 (4096^2 = 16.8 MP) after the streaming
# deposit / IBP changes: the run-level float32 tables (acc 3ch + cov = 16 B/px)
# plus the per-frame gather stack the auto gate holds for N frames
# (float32 luminance 4 B/px + bool mask 1 B/px) dominate. The guard only has to
# be in the right order of magnitude: it exists to refuse a run that would
# exhaust the machine, not to be an allocator.
#
# The constant is calibrated at 9 frames; longer bursts grow the per-frame
# stacks linearly. Measured peak RSS at 2x/4x ROI 1024 (val16, 2026-11-20):
#   4x: N=1 134, N=3 142, N=9 144, N=18 232 B/HR-px
#   2x: N=1 223, N=3 246, N=9 284, N=18 394 B/HR-px
# i.e. ~10-12 B/HR-px for every frame beyond the calibrated 9 (the floor stays
# 160 so shorter bursts keep the old, conservative estimate).
_HR_GUARD_BYTES_PER_PIXEL = 160
_HR_GUARD_CALIBRATED_FRAMES = 9
_HR_GUARD_BYTES_PER_PIXEL_PER_FRAME = 12
# Warn when the estimate crosses this share of the budget, raise above it.
_HR_GUARD_WARN_FRACTION = 0.75
_HR_GUARD_MIN_BUDGET_BYTES = 2 * 2 ** 30
_HR_GUARD_MAX_BUDGET_BYTES = 6 * 2 ** 30


# --------------------------------------------------------------------------- #
# Frame acceptance (the alignment-quality gate, ``min_align_score``)
# --------------------------------------------------------------------------- #
#
# The gate exists to stop a frame whose registration FAILED from being fused as
# ghost/blur. Two measured properties of its statistic (edge-weighted ZNCC of
# the rendered frame against the reference) set the policy:
#
# 1. The null distribution grows as the frame shrinks. Independent white-noise
#    frames registered against a real reference, 40 probes per size, MAXIMUM
#    score: 0.013 at 512 px, 0.024 at 256, 0.032 at 192, 0.047 at 160,
#    0.058 at 128, 0.094 at 96. Meanwhile correctly registered content that
#    lives ABOVE the LR Nyquist measures as low as -0.02 (band 0.26-0.50) or
#    -0.10 (0.35-0.49), with medians of only 0.11-0.15, at EVERY size. So below
#    ~192 px the two populations overlap and no absolute threshold can separate
#    them (review round 2, N2).
# 2. The achievable score is content-dependent: 0.78-0.91 for smooth scenes,
#    0.11-0.15 for purely aliased ones. A fixed 0.1 therefore sits inside the
#    good population of an aliased burst.
#
# Policy: the effective threshold is ``min(min_align_score,
# share * median(non-reference scores))``, a RELATIVE floor so a burst that
# is uniformly low-scoring (aliased content) is not decimated, while junk
# (score ~0) is still dropped because the burst median of a normal burst is
# ~0.99. Below ``_ALIGN_GATE_MIN_ROI_PX`` the gate is ADVISORY: it drops
# nothing, and reports what it would have dropped, because there the junk and
# aliased populations overlap. Everything is reported in the metrics.
_ALIGN_GATE_MEDIAN_SHARE = 0.25
_ALIGN_GATE_MIN_ROI_PX = 192


def _hr_guard_bytes_per_pixel(n_frames: Optional[int] = None) -> int:
    """Peak bytes per HR pixel for a burst of ``n_frames`` frames."""
    if n_frames is None:
        return int(_HR_GUARD_BYTES_PER_PIXEL)
    extra = max(0, int(n_frames) - _HR_GUARD_CALIBRATED_FRAMES)
    return int(_HR_GUARD_BYTES_PER_PIXEL
               + extra * _HR_GUARD_BYTES_PER_PIXEL_PER_FRAME)


def _hr_guard_budget_bytes() -> int:
    """Memory budget for one SR run: 60% of ``MemAvailable``, clamped.

    Reading ``/proc/meminfo`` keeps the guard honest on a small machine without
    hard-coding this box's size; the clamp keeps it from refusing / allowing
    absurd runs on a heavily loaded or very large host.
    """
    avail = 0
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        avail = 0
    if avail <= 0:
        return _HR_GUARD_MIN_BUDGET_BYTES
    budget = int(0.6 * avail)
    return max(_HR_GUARD_MIN_BUDGET_BYTES, min(_HR_GUARD_MAX_BUDGET_BYTES, budget))


def estimate_hr_peak_bytes(hr_pixels: int,
                           bytes_per_pixel: Optional[int] = None,
                           n_frames: Optional[int] = None) -> int:
    """Conservative peak-memory estimate for an HR output of this size.

    ``bytes_per_pixel`` overrides the calibration outright. Otherwise the
    per-pixel cost grows with ``n_frames`` (the auto gate holds a per-frame
    stack), see :func:`_hr_guard_bytes_per_pixel`.
    """
    bpp = (int(bytes_per_pixel) if bytes_per_pixel is not None
           else _hr_guard_bytes_per_pixel(n_frames))
    return int(hr_pixels) * bpp


# fusion='auto' host-RAM guard for MAP: HR pixels x frames (3072^2 x 9, measured safe < 3 GB).
_AUTO_MAP_MAX_HR_PX_FRAMES = 3072 * 3072 * 9

class SuperResolution:
    """Multi-frame SR orchestrator for a burst of RAW frames."""

    def __init__(self, ratio: float = 2.0, blur_threshold_ratio: float = 0.6,
                 registration: str = "homography",
                 registration_robust: bool = False,
                 ecc_scale: float = 0.5, drizzle_interp: int = cv2.INTER_LINEAR,
                 drizzle_post_sigma: float = 0.0,
                 drizzle_footprint_sigma: float = 0.5,
                 drizzle_deposit: str = "scatter",
                 drizzle_pixfrac: Optional[float] = None,
                 drizzle_subsamples: Optional[int] = None,
                 drizzle_splat: str = "bilinear",
                 drizzle_splat_sigma: float = 0.5,
                 drizzle_low_cov_thresh: float = 0.0,
                 drizzle_low_cov_sigma: float = 0.8,
                 ibp_iterations: int = 1, ibp_lambda: float = 0.5,
                 ibp_reg_sigma: float = 0.5,
                 ibp_max_step: Optional[float] = 0.1,
                 ibp_forward_sigma: Optional[float] = None,
                 ibp_joint_refine: bool = False,
                 ibp_forward_sigma_shrink: float = 1.0,
                 ibp_adjoint: bool = True,
                 ibp_prox_noise_scale: float = 2.5,
                 ibp_prox_space_lr: float = 0.75,
                 robust_fusion: Optional[bool] = None,
                 robust_k: float = 4.685,
                 robust_iters: int = 1,
                 robust_min_weight: float = 0.0,
                 robust_localize: bool = True,
                 min_align_score: float = 0.1,
                 use_opencl: Optional[bool] = None,
                 use_gpu_scatter: Optional[bool] = None,
                 registration_workers: Optional[int] = None,
                 max_hr_pixels: Optional[int] = None,
                 memory_budget_bytes: Optional[int] = None,
                 fusion: str = "auto",
                 max_exposure_ratio: Optional[float] = 1.5,
                 cfa_k_detail: float = 0.18,
                 cfa_chroma_scale: float = 1.25,
                 cfa_adaptive: bool = False,
                 local_align: bool = False,
                 local_tile_size: int = 48,
                 local_max_shift: float = 2.0,
                 local_select: Optional[float] = None,
                 local_robust: bool = True,
                 use_gpu: Optional[bool] = None,
                 reg_refine: int = 0,
                 map_mode: str = "wp13",
                 map_kwargs: Optional[Dict[str, Any]] = None):
        """Args:
            ratio: HR/LR scale factor.
            blur_threshold_ratio: drop frames whose Laplacian-variance sharpness
                falls below this fraction of the sharpest frame.
            registration: "homography" (default) or "translation".
            registration_robust: when True, the homography fit is edge-weighted
                (gradient inputMask) and may fall back to an affine model when
                the perspective terms are weakly determined. Default **False**:
                measured on the real 9-frame burst, the unweighted full
                homography leaves a smaller mean alignment residual (0.021 px
                vs 0.034 px) and a slightly sharper result, while the automatic
                reference selection (kept regardless) delivers the large
                registration win. The robust path stays available for
                A/B comparison and for genuinely perspective-heavy bursts.
            ecc_scale: working scale for the first ECC pass; the fit is then
                always refined at full scale 1.0 (0.5 locks reliably and stays
                ~4x cheaper than starting at full scale).
            drizzle_deposit / drizzle_splat / drizzle_pixfrac /
            drizzle_subsamples: define the **true-drizzle scatter** deposit (see
            :func:`drizzle`). Each LR pixel's measured value is scattered onto
            the HR grid at its estimated sub-pixel position with a compact
            bilinear splat, so the deposit is both sharp and sub-pixel
            accurate. This is the default; an interpolated *gather* deposit
            low-passes the recovered band (its reconstruction kernel is zero at
            the single-frame Nyquist) and must not be used.
            ``drizzle_pixfrac=None`` / ``drizzle_subsamples=None`` (default)
            mean **ratio-aware**: the footprint stays 0.5 LR px at every ratio
            (so it widens in HR px with the ratio; shrinking it in HR terms
            measured as prediction-neutral — the numbers are in
            :func:`default_deposit_params`) and the sub-sample count is derived
            so its lattice pitch stays <= 0.5 HR px (2 / 3 / 3 at
            2x / 3x / 4x). At 2x dropping the third sub-sample changes the
            deposit by 0.67 % of its value and is measured neutral-to-positive
            in prediction (held-out +7.614 vs +7.615 dB, HF +0.24 dB), while
            the ``np.bincount`` scatter cost falls 2.25x; a fixed count of 3
            over-samples 2x. The resolved values are
            reported in ``SRResult.metrics['drizzle_pixfrac'|'drizzle_subsamples']``.
            Explicit values are honoured as given (2x keeps pixfrac 0.5).
            drizzle_interp / drizzle_footprint_sigma: only apply to the legacy
                ``drizzle_deposit="gather"`` path (kept for A/B measurement);
                ignored by the scatter default. ``drizzle_post_sigma`` applies
                to both.
            ibp_iterations / ibp_lambda / ibp_reg_sigma / ibp_max_step:
                iterative back-projection refinement (see
                :func:`iterative_back_projection`). ``ibp_iterations=0``
                disables it. Default **1** (formerly 3): with the adjoint
                back-projection the held-out PSNR gain is monotone in IBP
                strength, and 3 sweeps measured 1.0-1.4 dB *lower* held-out
                gain at 2x/3x/4x (ROI 512 sweep in state.md) for the same
                HF-band gain — the extra sweeps only added acutance.
            ibp_joint_refine: when True, IBP re-estimates each frame's residual
                translation against the current HR estimate after every sweep
                and updates its transform (error-back-projection / EBP). Off by
                default: it measured as neutral once sub-pixel deposition and
                robust registration are in place.
            ibp_forward_sigma_shrink: per-sweep multiplier on the forward-model
                PSF (1.0 disables); tested and left at 1.0 because shrinking
                did not reduce border ghosting.
            ibp_adjoint: when True (default) the IBP correction is blurred by
                the same sensor-support PSF the forward model applied, i.e. the
                back-projection is the adjoint of the assumed forward model
                (preconditioned and clipped, so a descent direction rather than
                an exact steepest-descent step). False restores the legacy
                plain splat: it measures slightly sharper on screen but
                0.2-0.4 dB worse held-out PSNR (see
                :func:`iterative_back_projection`).
            The resolved configuration (``ibp_iterations``, ``ibp_lambda``,
            ``ibp_reg_sigma``, ``ibp_max_step``, ``ibp_forward_sigma_hr``,
            ``ibp_joint_refine``, ``ibp_adjoint`` and a display-ready
            ``ibp_summary``) is reported in ``SRResult.metrics``, so results
            produced with different settings can be told apart without
            re-deriving them; see ``docs/IBP_AND_HELDOUT.md``.
            min_align_score: alignment quality gate in [0, 1]; frames whose
                ZNCC alignment score falls below the *resolved threshold* are
                dropped (the reference frame is always kept). The resolved
                threshold is the smaller of this value and
                ``0.25 x median(non-reference scores)``, so a burst that is
                uniformly low-scoring is not decimated: measured, smooth scenes
                score 0.78-0.91 and real burst frames 0.997-0.998 (0.92-0.98
                with 10x the read noise), while content living entirely above
                the LR Nyquist scores 0.11-0.15 (down to -0.02) even when
                perfectly registered. Independent white-noise frames score at
                most 0.013 / 0.024 / 0.032 / 0.058 / 0.094 at 512 / 256 / 192 /
                128 / 96 px (40 probes per size), i.e. the null floor rises as
                the frame shrinks; below ``_ALIGN_GATE_MIN_ROI_PX`` (192 px)
                the two populations overlap and the gate drops nothing and is
                reported as ``advisory``. 0.1 keeps all 9 real frames at ROI 512
                and 1024 and drops noise frames; it is deliberately not 0.3
                (that would drop correctly registered aliased data).
                The resolved threshold, its mode and every drop are reported in
                ``metrics['align_gate_threshold']``,
                ``['align_gate_mode']``, ``['align_gate_dropped']`` and
                ``['align_gate_reason']``; frames kept only because the gate was
                advisory are listed in ``['align_gate_advisory']``. 0.0
                disables the gate (legacy behaviour).
            use_opencl: OpenCL (OpenCV T-API) routing for the upsample warps.
                ``None`` (default, auto) enables it when an OpenCL device
                exists *and* a one-time calibration measures the routed op
                class actually faster on it than on the CPU (see
                :func:`opencl_beneficial` — on some GPU stacks the device is
                slower for these kernels, and auto then keeps the CPU path).
                ``True`` forces the T-API routing whenever a device exists
                (what the desktop GPU benchmark measures), ``False`` disables
                OpenCL for the run. Either way every device call falls back to
                the identical CPU call on failure, so results never depend on
                the device being healthy. The warp-routing decision alone is
                reported as ``SRResult.metrics['opencl_warp']``; the combined
                "a GPU engine ran" verdict (warp OR scatter) is
                ``SRResult.metrics['opencl_active']`` / ``self.opencl_active``
                (D3).
            use_gpu_scatter: tri-state override for the **scatter deposition**
                GPU backend (pyopencl atomics, see :class:`_GpuScatterDeposit`)
                independent of the warp routing above. ``None`` (default)
                follows ``use_opencl``: auto (both None) engages GPU scatter
                only when a device exists *and* :func:`gpu_scatter_beneficial`
                measures it >=5% faster than the banded CPU engine; True
                forces it (per-call CPU fallback on any device error); False
                keeps the deposit on the CPU. Set it explicitly (with
                ``use_opencl=False``) to A/B-measure the scatter engine alone.
                Reported in ``SRResult.metrics['opencl_scatter']``.
            registration_workers: process-pool size for the per-frame
                registration loop (see the "Parallel registration" section).
                ``None`` (default, auto) pools only when the reference frame is
                large enough for the work to beat pool spawn — at least
                ``_REG_POOL_MIN_REF_PIXELS`` pixels and at least two frames to
                register — and then uses at most ``_REG_POOL_MAX_WORKERS``
                workers, one per frame, and one less than the machine's CPUs.
                ``1`` forces the serial path
                (what small ROIs get anyway; exposed so tests can compare the
                paths bit-for-bit); any ``N > 1`` forces a pool of min(N,
                frames, cpus-1). Results are collected by frame index and the
                fit is bit-identical across thread counts (measured), so the
                pooled transforms equal the serial transforms exactly; if the
                environment cannot host child processes the run transparently
                falls back to serial.
            max_hr_pixels / memory_budget_bytes: HR-size memory guard. Peak
                memory scales with the HR output (``(h*ratio) x (w*ratio)``),
                not the LR ROI, so a ratio/ROI combination can ask for
                gigabytes before any kernel runs. ``max_hr_pixels=None``
                (default) derives the limit from ``memory_budget_bytes=None``
                (default: 60% of ``MemAvailable``, clamped to 2-6 GiB) with a
                conservative 160 bytes/HR-pixel estimate. Above the limit
                :meth:`run` raises ``MemoryError`` with the output size, the
                estimate and the knob to change; above 75% of the budget it
                emits a ``RuntimeWarning`` and continues. The estimate helper is
                :func:`estimate_hr_peak_bytes`.
            robust_fusion / robust_k / robust_iters / robust_min_weight:
                robust scatter fusion (see :func:`drizzle` and
                :func:`_robust_residual_weights`). Each frame's deposit is
                down-weighted where it disagrees with the across-frame median on
                the shared HR grid, so genuine parallax / moving-subject ghosts
                are rejected instead of smeared. **Tri-state, default None
                (auto):** before deposition the already-warped HR stack is
                tested by three detectors — a compact region of large,
                coherent inter-frame disagreement
                (:func:`_ghost_gate_dispersion`), the *population* of
                disagreeing tiles as one blob (:func:`_ghost_gate_population`,
                the case where the subject drags the homography and dilutes the
                tail statistic) and the contrast-normalized inter-frame spread
                (:func:`_ghost_gate_spread`, the case where the subject covers
                most of the frame so every |-median| statistic loses its
                reference). A moving subject / parallax layer trips the gate and
                robust fusion runs;
                a plain static burst does not, and the result is then exactly
                the plain path (measured on the real 9-frame burst, robust
                fusion costs ~1.2-1.4 sharpness units and 1.4-2.6 dB HF gain
                there because above the single-frame Nyquist inter-frame
                aliasing looks like ghosting). ``True``/``False`` override the
                gate explicitly. The decision and its reason are reported in
                ``metrics['robust_active']`` / ``metrics['robust_reason']``.
                The gate's warped stack is reused by the ghost-energy metric,
                so the burst is gathered to the HR grid once per run; with only
                two frames the across-frame median is the mean, so D1-D3 see
                half the disagreement -- the pair detector
                (:func:`_ghost_gate_pair`) measures it directly, and when it
                fires (or when robust fusion is forced on) the flagged region
                keeps only the reference frame instead of an average of two
                contradictory observations (see
                :func:`_reference_only_region`). ``ghost_energy`` stays
                structurally 0 at N=2 and the reason says so.
                ``robust_k`` is the Tukey biweight constant (larger = more
                permissive), ``robust_iters`` the number of re-weight passes (1
                refined by the median is enough; >1 uses the robust estimate),
                and ``robust_min_weight`` a floor on every weight.
        """
        self.ratio = float(ratio)
        self.blur_threshold_ratio = blur_threshold_ratio
        self.registration = registration
        self.registration_robust = bool(registration_robust)
        self.ecc_scale = float(ecc_scale)
        self.drizzle_interp = drizzle_interp
        self.drizzle_post_sigma = float(drizzle_post_sigma)
        self.drizzle_low_cov_thresh = float(drizzle_low_cov_thresh)
        self.drizzle_low_cov_sigma = float(drizzle_low_cov_sigma)
        self.drizzle_footprint_sigma = float(drizzle_footprint_sigma)
        self.drizzle_deposit = str(drizzle_deposit)
        # Ratio-aware defaults (see default_deposit_params); explicit overrides
        # are kept exactly as given.
        _pf_auto, _sub_auto = default_deposit_params(self.ratio)
        if drizzle_pixfrac is None:
            self.drizzle_pixfrac = float(_pf_auto)
        else:
            self.drizzle_pixfrac = float(drizzle_pixfrac)
        if drizzle_subsamples is None:
            if drizzle_pixfrac is not None:
                _sub_auto = deposit_subsamples(self.drizzle_pixfrac, self.ratio)
            self.drizzle_subsamples = int(_sub_auto)
        else:
            self.drizzle_subsamples = int(drizzle_subsamples)
        if self.drizzle_pixfrac <= 0:
            raise ValueError("drizzle_pixfrac must be > 0")
        if self.drizzle_subsamples < 1:
            raise ValueError("drizzle_subsamples must be >= 1")
        self.drizzle_splat = str(drizzle_splat)
        self.drizzle_splat_sigma = float(drizzle_splat_sigma)
        self.ibp_iterations = int(ibp_iterations)
        self.ibp_lambda = float(ibp_lambda)
        self.ibp_reg_sigma = float(ibp_reg_sigma)
        self.ibp_max_step = ibp_max_step
        self.ibp_forward_sigma = ibp_forward_sigma
        self.ibp_joint_refine = bool(ibp_joint_refine)
        self.ibp_forward_sigma_shrink = float(ibp_forward_sigma_shrink)
        self.ibp_adjoint = bool(ibp_adjoint)
        self.ibp_prox_noise_scale = float(ibp_prox_noise_scale)
        self.ibp_prox_space_lr = float(ibp_prox_space_lr)
        self.robust_fusion = (None if robust_fusion is None
                              else bool(robust_fusion))
        self.robust_k = float(robust_k)
        self.robust_iters = int(robust_iters)
        self.robust_min_weight = float(robust_min_weight)
        # B2: re-weight only where the gate found disagreement (default), so an
        # ON decision does not spend the static part of the frame's HF. False
        # reproduces the pre-B2 whole-frame behaviour for A/B measurement.
        self.robust_localize = bool(robust_localize)
        self.min_align_score = float(min_align_score)
        self.use_opencl = None if use_opencl is None else bool(use_opencl)
        self.use_gpu_scatter = (None if use_gpu_scatter is None
                                else bool(use_gpu_scatter))
        if registration_workers is not None:
            registration_workers = int(registration_workers)
            if registration_workers < 1:
                raise ValueError("registration_workers must be >= 1")
        self.registration_workers = registration_workers
        if max_hr_pixels is not None:
            max_hr_pixels = int(max_hr_pixels)
            if max_hr_pixels < 1:
                raise ValueError("max_hr_pixels must be >= 1")
        self.max_hr_pixels = max_hr_pixels
        if memory_budget_bytes is not None:
            memory_budget_bytes = int(memory_budget_bytes)
            if memory_budget_bytes < 1:
                raise ValueError("memory_budget_bytes must be >= 1")
        self.memory_budget_bytes = memory_budget_bytes
        self.local_align = bool(local_align)
        self.local_tile_size = int(local_tile_size)
        self.local_max_shift = float(local_max_shift)
        self.local_select = (None if local_select is None else float(local_select))
        self.local_robust = bool(local_robust)
        if fusion not in ("auto", "demosaic_drizzle", "cfa_drizzle", "cfa_kr", "map"):
            raise ValueError(
                "fusion must be 'auto', 'demosaic_drizzle', 'cfa_drizzle', 'cfa_kr', "
                f"or 'map', got {fusion!r}"
            )
        # 'auto' (default, WP12 decision): forward-model MAP at 2x when a CUDA
        # GPU is usable, otherwise demosaic_drizzle. Resolved per run() in
        # _resolve_fusion; self.fusion holds the fusion actually used.
        self.fusion_requested = str(fusion)
        self.fusion = "demosaic_drizzle" if fusion == "auto" else str(fusion)
        self.fusion_reason = "explicit" if fusion != "auto" else "not run yet"
        self.max_exposure_ratio = (
            None if (max_exposure_ratio is None or float(max_exposure_ratio) <= 0)
            else float(max_exposure_ratio)
        )
        self.cfa_k_detail = float(cfa_k_detail)
        self.cfa_chroma_scale = float(cfa_chroma_scale)
        self.cfa_adaptive = bool(cfa_adaptive)
        self.use_gpu = None if use_gpu is None else bool(use_gpu)
        reg_refine_i = int(reg_refine)
        if reg_refine_i < 0:
            raise ValueError(f"reg_refine must be >= 0, got {reg_refine!r}")
        self.reg_refine = reg_refine_i
        if map_mode not in ("legacy", "wp13"):
            raise ValueError(f"map_mode must be 'legacy' or 'wp13', got {map_mode!r}")
        self.map_mode = str(map_mode)
        self.map_kwargs = dict(map_kwargs) if map_kwargs else {}
        # Set by run(): whether OpenCL actually executed the routed ops.
        self.opencl_active = False

    def _registration_workers(self, n_tasks: int,
                              ref_shape: Tuple[int, ...]) -> int:
        """Worker-process count for the registration loop; 1 means serial.

        Auto (``registration_workers=None``) pools only where the per-frame
        work clearly beats pool spawn (see the parallel-registration notes):
        a reference frame of at least ``_REG_POOL_MIN_REF_PIXELS``, at least
        two frames to register, and at least 4 CPUs (one for the parent plus
        headroom). Explicit N forces min(N, frames, cpus-1); 1 forces serial.
        """
        ncpu = max(int(os.cpu_count() or 1), 1)
        want = self.registration_workers
        if want is None:
            if (int(ref_shape[0]) * int(ref_shape[1])
                    < _REG_POOL_MIN_REF_PIXELS or int(n_tasks) < 2
                    or ncpu < 4):
                return 1
            want = _REG_POOL_MAX_WORKERS
        return max(1, min(int(want), int(n_tasks), ncpu - 1))

    def _check_hr_size(self, frames: Sequence[RawFrame]) -> Tuple[int, int]:
        """Refuse (or warn about) an HR output the machine cannot hold.

        Peak memory scales with the HR pixel count, not with the LR ROI, so a
        ratio/ROI combination can ask for gigabytes before any kernel runs.
        ``max_hr_pixels`` (or ``memory_budget_bytes``) bounds it explicitly; by
        default the budget is 60% of ``MemAvailable``. Above the budget this
        raises :class:`MemoryError` with the numbers and the knobs to change;
        between 75% and 100% it emits a ``RuntimeWarning`` and continues.
        """
        h, w = frames[0].rgb.shape[:2]
        oh = int(round(h * self.ratio))
        ow = int(round(w * self.ratio))
        pixels = oh * ow
        n_frames = len(frames)
        bpp = _hr_guard_bytes_per_pixel(n_frames)
        if self.max_hr_pixels is not None:
            limit = int(self.max_hr_pixels)
            budget = None
        else:
            budget = (self.memory_budget_bytes
                      if self.memory_budget_bytes is not None
                      else _hr_guard_budget_bytes())
            limit = max(1, int(budget) // bpp)
        est = estimate_hr_peak_bytes(pixels, n_frames=n_frames)
        detail = (f"HR output {oh}x{ow} = {pixels:,} px at ratio "
                  f"{self.ratio:g} (LR {h}x{w}, {n_frames} frames); estimated "
                  f"peak {est / 2**30:.1f} GiB, limit {limit:,} px")
        if pixels > limit:
            raise MemoryError(
                detail + " — reduce the ratio or the ROI, or raise "
                "SuperResolution(max_hr_pixels=.../memory_budget_bytes=...).")
        if (budget is not None
                and est > _HR_GUARD_WARN_FRACTION * int(budget)):
            warnings.warn(detail + " — close to the memory limit.", RuntimeWarning,
                          stacklevel=2)
        elif budget is None and pixels > _HR_GUARD_WARN_FRACTION * limit:
            warnings.warn(detail + " — close to the memory limit.", RuntimeWarning,
                          stacklevel=2)
        return oh, ow

    def _resolve_fusion(self, frames: Sequence[RawFrame]) -> Tuple[str, str]:
        """Resolve ``fusion='auto'`` into the fusion used for this run.

        MAP (WP8/WP11) won the colour-fair comparison (WP12: synthetic GT
        +7..+8 dB, simulated chart +3.5 dB, colour-fair held-out ahead on 4 of
        5 real bursts) and is the 2x default when a CUDA GPU is usable; on CPU
        it is slower, so auto keeps demosaic_drizzle there. 3x/4x stay on
        demosaic_drizzle (experimental ratios).
        """
        if self.fusion_requested != "auto":
            return self.fusion_requested, "explicit"
        if abs(float(self.ratio) - 2.0) > 1e-6:
            return "demosaic_drizzle", "auto: map is the default only at 2x"
        if not all(getattr(f, "mosaic", None) is not None
                   and getattr(f, "pattern", None) is not None for f in frames):
            return "demosaic_drizzle", "auto: frames carry no Bayer mosaic"
        # Host-RAM guard: MAP is not tiled yet and its peak RSS grows faster
        # than the default path's (measured 2x: 1536^2 ROI x 9 frames 2.8 GB,
        # 1024^2 x 20 frames 2.4 GB, 2048^2 x 9 frames > 6 GB -> OOM kill, which
        # cannot be caught). Auto keeps MAP to <= 3072^2 HR px x 9 frames.
        h0, w0 = frames[0].rgb.shape[:2]
        hr_px = int(round(h0 * self.ratio)) * int(round(w0 * self.ratio))
        if hr_px * len(frames) > _AUTO_MAP_MAX_HR_PX_FRAMES:
            return "demosaic_drizzle", "auto: ROI x frames too large for map (memory guard)"
        if self.use_gpu is False:
            return "demosaic_drizzle", "auto: GPU disabled (use_gpu=False)"
        try:
            from .gpu import gpu_available
            if not gpu_available():
                return "demosaic_drizzle", "auto: no CUDA GPU"
        except Exception:  # pragma: no cover - import/driver failure
            return "demosaic_drizzle", "auto: GPU backend unavailable"
        return "map", "auto: 2x with CUDA GPU"

    def run(self, frames: Sequence[RawFrame],
            reference_index: Optional[int] = None) -> SRResult:
        """Register, drizzle and refine the burst; returns an :class:`SRResult`.

        Resolves the OpenCL routing decision (see ``use_opencl`` in the
        constructor), applies it for the duration of the run — restoring the
        process-wide ``cv2.ocl`` state afterwards — and records the outcome in
        ``metrics['opencl_warp']`` (the warp decision) and
        ``metrics['opencl_active']`` / ``self.opencl_active`` (True when either a
        warp or the scatter deposit actually ran on the GPU; D3).
        """
        if not frames:
            raise ValueError("No RAW frames provided")
        if self.ratio <= 0:
            raise ValueError("ratio must be > 0")
        self.fusion, self.fusion_reason = self._resolve_fusion(frames)

        # Public-API validation (fail here, in plain words, instead of a raw
        # IndexError / cv2.error from deep inside registration or fusion).
        if reference_index is not None:
            if (isinstance(reference_index, bool)
                    or not isinstance(reference_index, (int, np.integer))):
                raise ValueError(
                    "reference_index must be an integer or None, got "
                    f"{type(reference_index).__name__}")
            reference_index = int(reference_index)
            if not 0 <= reference_index < len(frames):
                raise ValueError(
                    f"reference_index {reference_index} is out of range for "
                    f"{len(frames)} frames (valid: 0..{len(frames) - 1})")
        ref_hw: Optional[Tuple[int, int]] = None
        for i, fr in enumerate(frames):
            rgb = getattr(fr, "rgb", None)
            green = getattr(fr, "green", None)
            if rgb is None or green is None:
                raise ValueError(f"frame {i} has no rgb/green array")
            rgb = np.asarray(rgb)
            green = np.asarray(green)
            if rgb.ndim != 3 or rgb.shape[2] != 3 or green.ndim != 2:
                raise ValueError(
                    f"frame {i}: expected rgb (H, W, 3) and green (H, W), got "
                    f"rgb {rgb.shape} / green {green.shape}")
            hw = (int(rgb.shape[0]), int(rgb.shape[1]))
            if (green.shape[0], green.shape[1]) != hw:
                raise ValueError(
                    f"frame {i}: green {green.shape} does not match rgb "
                    f"{rgb.shape}")
            if hw[0] < 1 or hw[1] < 1:
                raise ValueError(f"frame {i} is empty ({hw[1]}x{hw[0]})")
            if ref_hw is None:
                ref_hw = hw
            elif hw != ref_hw:
                raise ValueError(
                    f"frame {i} is {hw[1]}x{hw[0]} but frame 0 is "
                    f"{ref_hw[1]}x{ref_hw[0]}; every frame must share one ROI "
                    "and shape")
            if not np.isfinite(rgb).all():
                raise ValueError(f"frame {i} contains NaN or Inf in rgb")
            if not np.isfinite(green).all():
                raise ValueError(f"frame {i} contains NaN or Inf in green")

        # Refuse an HR output that cannot fit in memory before any kernel runs.
        self._check_hr_size(frames)

        # OpenCL routing decision (auto = device exists AND calibration says
        # the routed op class wins there; see opencl_beneficial()).
        have = bool(cv2.ocl.haveOpenCL())
        if self.use_opencl is None:
            active = have and opencl_beneficial()
        elif self.use_opencl:
            active = have
        else:
            active = False
        self.opencl_active = bool(active)

        # GPU scatter-deposition decision. Same tri-state as the warp routing
        # (auto = device exists AND gpu_scatter_beneficial() measured it
        # faster; True forces with CPU fallback; False pure CPU) but decided
        # INDEPENDENTLY — warp routing and scatter have opposite measured
        # winners, so neither inherits the other. ``use_gpu_scatter`` overrides
        # it alone (A/B measurement knob). The decision only *requests* the
        # engine; drizzle() still declines it per run (VRAM guard) and falls
        # back to the identical CPU deposit on any device error.
        want_scatter = self.use_gpu_scatter
        if want_scatter is None:
            want_scatter = self.use_opencl
        if want_scatter is None:
            scatter_gpu = gpu_scatter_beneficial()
        else:
            scatter_gpu = bool(want_scatter)

        prev_ocl = None
        if have:
            try:
                prev_ocl = cv2.ocl.useOpenCL()
                cv2.ocl.setUseOpenCL(bool(active))
            except cv2.error:
                prev_ocl = None
        try:
            return self._run_inner(frames, reference_index, bool(active),
                                   bool(scatter_gpu))
        finally:
            if prev_ocl is not None:
                try:
                    cv2.ocl.setUseOpenCL(prev_ocl)
                except cv2.error:
                    pass

    def _run_inner(self, frames: Sequence[RawFrame],
                   reference_index: Optional[int],
                   ocl_active: bool,
                   scatter_gpu: bool = False) -> SRResult:
        # 0. Exposure outlier gating (before reference selection) + per-frame
        #    exposure normalisation. Rejects frames whose median luma deviates
        #    > max_exposure_ratio (e.g. 1.5x, the HDR+ N009 case) from the
        #    burst median so reference selection never picks an exposure outlier.
        exp_dropped: List[dict] = []
        # Mean green is invariant to defocus blur (unlike median on sparse scenes)
        exp_meds = np.array([float(np.mean(f.green)) for f in frames],
                            dtype=np.float64)
        exp_ok = np.ones(len(frames), dtype=bool)
        med_burst = float(np.median(exp_meds)) if exp_meds.size else 0.0
        if (self.max_exposure_ratio is not None and len(frames) >= 3
                and med_burst > 1e-7):
            for i, m in enumerate(exp_meds):
                r = m / med_burst if m > 1e-12 else 0.0
                dev = max(r, 1.0 / max(r, 1e-12)) if r > 0 else float("inf")
                if dev > self.max_exposure_ratio:
                    if reference_index is not None and i == reference_index:
                        continue
                    exp_ok[i] = False
                    exp_dropped.append({
                        "index": int(i),
                        "name": frames[i].name,
                        "exposure_ratio": round(float(r), 4),
                    })
            if not np.any(exp_ok):
                exp_ok[:] = True
                exp_dropped.clear()

        # 1. Quality weights (sharpness proxy) on exposure-valid frames.
        raw_weights = np.array([_quality_weight(f.green) for f in frames], dtype=np.float64)
        best = float(raw_weights[exp_ok].max()) if np.any(exp_ok) else float(raw_weights.max())
        keep = exp_ok & (raw_weights >= self.blur_threshold_ratio * best)
        if not np.any(keep):
            keep = exp_ok.copy()
        # Reference selection: an explicit index always wins; otherwise pick
        # the sharpest frame near the burst centroid (see _select_reference).
        if reference_index is None:
            ref_stats: dict = {}
            reference_index = _select_reference(
                frames, raw_weights, [i for i, k in enumerate(keep) if k],
                stats=ref_stats)
        else:
            ref_stats = {}
        keep[reference_index] = True

        # Normalise each kept frame's exposure to match the reference frame
        # (shallow-copying RawFrame so caller's arrays are never mutated).
        med_ref = float(exp_meds[reference_index])
        exp_scales: List[float] = [1.0] * len(frames)
        norm_frames: List[RawFrame] = list(frames)
        if med_ref > 1e-7:
            for i in range(len(frames)):
                if not keep[i] or i == reference_index:
                    continue
                m = float(exp_meds[i])
                if m > 1e-7:
                    sc = med_ref / m
                    exp_scales[i] = float(sc)
                    if abs(sc - 1.0) > 0.02:
                        fr = frames[i]
                        sc32 = np.float32(sc)
                        norm_frames[i] = RawFrame(
                            name=fr.name,
                            rgb=np.clip(fr.rgb * sc32, 0.0, 1.0),
                            green=np.clip(fr.green * sc32, 0.0, 1.0),
                            meta=fr.meta,
                            mosaic=(None if fr.mosaic is None
                                    else np.clip(fr.mosaic * sc32, 0.0, 1.0)),
                            pattern=fr.pattern,
                        )
        frames = norm_frames

        ref_frame = frames[reference_index]
        ref_green = ref_frame.green

        # 2. Subpixel registration of sharpness-kept frames to the reference.
        #    A global homography/affine absorbs the rotation/perspective that
        #    translation-only registration misses on a handheld burst. The
        #    reported shifts (metrics + held-out prediction) stay as the
        #    phase-correlation translation relative to frame 0 so magnitude
        #    reporting is stable regardless of which frame is the reference.
        anchor_green = frames[0].green
        candidates = [i for i, k in enumerate(keep) if k]

        # 2a. Per-frame registration measurements. Serial or pooled: both run
        #     the same :func:`_register_frame`, and the pool collects results
        #     BY INDEX, so the outputs are identical either way (the fit is
        #     bit-identical across thread counts, see the parallel-registration
        #     notes).
        regs: Optional[List[Tuple]] = None
        workers = self._registration_workers(len(candidates), ref_green.shape)
        # The reference frame's gradient mask is the same for every frame; it
        # was recomputed (full-frame Sobel + dilate) once per frame before.
        ref_mask = (gradient_mask(ref_green)
                    if self.registration_robust else None)
        if workers >= 2:
            store: Optional[_SharedRegFrames] = None
            try:
                # One shared block instead of 3 full frames pickled per task
                # (see _SharedRegFrames). Slot indices travel in the payload.
                store = _SharedRegFrames(
                    [anchor_green, ref_green]
                    + [frames[i].green for i in candidates])
                payloads = [
                    (j, store.slot(anchor_green), store.slot(ref_green),
                     store.slot(frames[i].green), i == reference_index,
                     self.registration, self.ecc_scale,
                     self.registration_robust, ocl_active)
                    for j, i in enumerate(candidates)
                ]
                rows = _map_in_processes(_registration_worker, payloads,
                                         workers,
                                         initargs=(store.name, store.specs))
                regs = list(rows)
            except Exception:
                regs = None  # pool unavailable -> serial fallback below
            finally:
                if store is not None:
                    store.close()
        if regs is None:
            regs = [
                _register_frame(anchor_green, ref_green, frames[i].green,
                                i == reference_index, self.registration,
                                self.ecc_scale, self.registration_robust,
                                ref_mask=ref_mask)
                for i in candidates
            ]

        # 2b. Resolve the frame-acceptance threshold for THIS burst (see the
        #     policy note at _ALIGN_GATE_MIN_ROI_PX): the absolute default can
        #     be relaxed towards the burst's own score distribution, and below
        #     a measured frame size it cannot separate junk from aliased
        #     content at all, so it becomes advisory and drops nothing.
        _ns_scores = np.asarray(
            [float(r[3]) for i, r in zip(candidates, regs)
             if i != reference_index and np.isfinite(r[3])], dtype=np.float64)
        _scores_median = (float(np.median(_ns_scores))
                          if _ns_scores.size else 1.0)
        _roi_min_px = int(min(ref_green.shape[:2]))
        if self.min_align_score <= 0.0:
            gate_threshold = 0.0
            gate_mode = "off"
        elif _roi_min_px < _ALIGN_GATE_MIN_ROI_PX:
            gate_threshold = float(self.min_align_score)
            gate_mode = "advisory"
        else:
            gate_threshold = min(float(self.min_align_score),
                                 _ALIGN_GATE_MEDIAN_SHARE * _scores_median)
            gate_mode = ("absolute" if gate_threshold
                         >= float(self.min_align_score)
                         else "relative")

        kept_indices: List[int] = []
        transforms: List[np.ndarray] = []
        shifts: List[Tuple[float, float]] = []
        align_scores: List[float] = []
        align_residuals: List[float] = []
        align_self: List[bool] = []
        gate_dropped: List[dict] = []
        gate_advisory: List[dict] = []
        for i, (dx, dy, W, score, residual) in zip(candidates, regs):
            # 3. Alignment quality gate (reference always kept). A frame whose
            #    ZNCC cannot find the reference scene is fused only as ghost /
            #    blur, so it is dropped and the drop is reported, never silent.
            if (gate_mode != "off" and i != reference_index
                    and score < gate_threshold):
                entry = {"index": int(i), "score": float(score)}
                if gate_mode == "advisory":
                    gate_advisory.append(entry)
                else:
                    gate_dropped.append(entry)
                    continue
            kept_indices.append(i)
            transforms.append(W)
            shifts.append((dx, dy))
            align_scores.append(float(np.clip(score, 0.0, 1.0)))
            align_residuals.append(
                float(residual) if np.isfinite(residual)
                else float(np.hypot(*ref_green.shape)))
            # The reference compares to itself (score 1.0, residual 0.0); flag
            # it so nothing ever averages that non-measurement (D2).
            align_self.append(i == reference_index)

        kept = [frames[i] for i in kept_indices]

        # 4. Normalized fusion weights: denoised-Laplacian sharpness per unit
        #    noise variance (see _fusion_weight). The noise floor stops a soft
        #    frame being rewarded for a scene-limited estimate, and the weight
        #    clip keeps one pathological frame from dominating the deposit.
        #    The raw Laplacian variance above stays the *gate* quantity (blur
        #    reject / reference choice); on the fusion side its noise-driven
        #    share is divided out.
        kept_noise = np.array([_noise_variance(f.green) for f in kept],
                              dtype=np.float64)
        kept_sigma = np.sqrt(kept_noise)
        kept_noise_c = _clip_noise_correction(kept_noise)
        kept_w = np.array([
            _fusion_weight(f.green, noise_var=nc)
            for f, nc in zip(kept, kept_noise_c)], dtype=np.float64)
        w_med = float(np.median(kept_w)) if kept_w.size else 0.0
        if w_med > 0.0:
            kept_w = np.clip(kept_w, w_med / FUSION_WEIGHT_CLIP,
                             w_med * FUSION_WEIGHT_CLIP)
        wsum = kept_w.sum()
        weights = (kept_w / wsum) if wsum > 0 else np.ones(len(kept)) / len(kept)

        # 5. Drizzle onto the HR grid with the sub-pixel deposition footprint.
        #    Resolve the robust_fusion tri-state: True/False are explicit
        #    overrides; None runs the auto gate on the warped stack (and reuses
        #    that stack for the weights, so the burst is warped once).
        robust_use = self.robust_fusion
        robust_reason = "forced on" if robust_use is True else "forced off"
        robust_stack: Optional[Tuple[np.ndarray, np.ndarray]] = None
        gate_cache: dict = {}
        if self.drizzle_deposit != "scatter":
            # The robust path only exists in the true-drizzle scatter deposit;
            # never report it active for a gather run.
            robust_use = False
            if self.robust_fusion is None:
                robust_reason = "not applicable (gather deposit)"
        elif robust_use is None:
            # D4 gets the registration residual (excluding the reference's
            # trivial self-measurement, D2), measured in step 2 above.
            _res_ns = [float(r) for r, s_ in zip(align_residuals, align_self)
                       if not s_]
            robust_use, robust_reason, robust_stack = _auto_robust_fusion(
                kept, transforms, self.ratio, cache=gate_cache,
                residuals=_res_ns)
        _engine_out: dict = {}
        _cfa_metrics: dict = {}
        _local_metrics: dict = {"local_align_active": False}
        _local_shifts = None
        _local_weights = None
        if self.local_align and len(kept) >= 2:
            from .local_align import estimate_burst_local_alignment
            _ref_pos_loc = kept_indices.index(reference_index)
            _loc_res = estimate_burst_local_alignment(
                kept, transforms, _ref_pos_loc,
                tile_size=self.local_tile_size,
                max_shift=self.local_max_shift,
                local_robust=self.local_robust,
                local_select=self.local_select,
            )
            _local_shifts = (_loc_res.dx_mov, _loc_res.dy_mov)
            _local_weights = _loc_res.weights_mov
            _local_metrics = dict(_loc_res.metrics)
            # When local_robust is active, it already handles local motion /
            # ghost suppression per tile, so skip the redundant HR-stack
            # global robust re-scatter unless explicitly forced on.
            if self.robust_fusion is None and self.local_robust:
                robust_use = False
                robust_reason = "handled by local_align"
        if self.fusion in ("cfa_drizzle", "cfa_kr"):
            if not all(f.mosaic is not None and f.pattern is not None for f in kept):
                raise ValueError(
                    f"fusion={self.fusion!r} requires every frame to carry calibrated "
                    "Bayer mosaic and pattern (loaded via load_raw_linear or "
                    "provided on RawFrame)"
                )
            from .cfa_drizzle import cfa_drizzle as _cfa_drizzle, cfa_kernel_regression
            from .map_sr import equalize_green_imbalance, estimate_colour_matrix

            ref_pos = kept_indices.index(reference_index)
            mosaics_eq, gr_gb_ratio = equalize_green_imbalance(
                [f.mosaic for f in kept], kept[ref_pos].pattern, ref_index=ref_pos
            )
            _cfa_C, _cfa_C_info = estimate_colour_matrix(
                kept[ref_pos].rgb, mosaics_eq[ref_pos], kept[ref_pos].pattern
            )
            _cfa_C_inv = np.linalg.inv(_cfa_C).astype(np.float32)

            import time as _time
            _t0_cfa = _time.perf_counter()
            if self.fusion == "cfa_drizzle":
                cfa_res = _cfa_drizzle(
                    mosaics_eq,
                    transforms,
                    self.ratio,
                    kept[ref_pos].pattern,
                    weights=weights.tolist(),
                    pixfrac=self.drizzle_pixfrac,
                    subsamples=self.drizzle_subsamples,
                    local_shifts=_local_shifts,
                    local_weights=_local_weights,
                )
                _engine_out["scatter"] = "cpu_cfa_drizzle"
            else:
                oh = int(round(ref_green.shape[0] * self.ratio))
                ow = int(round(ref_green.shape[1] * self.ratio))
                fallback_hr = cv2.resize(
                    kept[ref_pos].rgb, (ow, oh), interpolation=cv2.INTER_CUBIC
                )
                fallback_hr_cam = np.matmul(
                    fallback_hr.astype(np.float32), _cfa_C.T.astype(np.float32)
                )
                # Compute per-frame LR robustness maps if ghost gate fired
                robust_maps = None
                if robust_use and len(kept) >= 2:
                    robust_maps = []
                    ref_g = kept[ref_pos].green
                    sig_ref = max(float(np.sqrt(_noise_variance(ref_g))), 1e-4)
                    local_std = np.sqrt(np.maximum(
                        cv2.GaussianBlur(ref_g * ref_g, (0, 0), 1.5)
                        - cv2.GaussianBlur(ref_g, (0, 0), 1.5) ** 2,
                        0.0,
                    ))
                    denom = np.maximum(3.0 * sig_ref + 0.5 * local_std, 1e-4)
                    for idx_k, (fk, Wk) in enumerate(zip(kept, transforms)):
                        if idx_k == ref_pos:
                            robust_maps.append(np.ones_like(ref_g, dtype=np.float32))
                            continue
                        M_warp = np.asarray(Wk, dtype=np.float64)
                        if M_warp.shape == (2, 3):
                            M_warp = np.vstack([M_warp, [0.0, 0.0, 1.0]])
                        ref_in_k = cv2.warpPerspective(
                            ref_g, M_warp, (ref_g.shape[1], ref_g.shape[0]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                        )
                        diff = np.abs(fk.green - ref_in_k)
                        diff_s = cv2.GaussianBlur(diff, (0, 0), 1.0)
                        u = diff_s / (denom * np.float32(self.robust_k * 0.5))
                        w_rob = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0).astype(np.float32)
                        if self.robust_min_weight > 0.0:
                            w_rob = np.maximum(w_rob, np.float32(self.robust_min_weight))
                        robust_maps.append(w_rob)

                cfa_res = cfa_kernel_regression(
                    mosaics_eq,
                    transforms,
                    self.ratio,
                    kept[ref_pos].pattern,
                    ref_green=ref_green,
                    weights=weights.tolist(),
                    fallback_hr=fallback_hr_cam,
                    robust_maps=robust_maps,
                    k_detail=self.cfa_k_detail,
                    chroma_scale=self.cfa_chroma_scale,
                    adaptive=self.cfa_adaptive,
                )
                _engine_out["scatter"] = "cpu_cfa_kr"
            _engine_out["seconds"] = float(_time.perf_counter() - _t0_cfa)
            hr = np.clip(np.matmul(cfa_res.hr, _cfa_C_inv.T), 0.0, 1.0).astype(np.float32)
            coverage = cfa_res.coverage
            _cfa_metrics = dict(cfa_res.metrics)
            _cfa_metrics["colour_cond"] = _cfa_C_info.get("cond", 1.0)
            _cfa_metrics["colour_fallback"] = bool(_cfa_C_info.get("fallback", False))
            _cfa_metrics["gr_gb_ratio"] = round(float(gr_gb_ratio), 5)
        else:
            from .gpu import resolve_use_gpu as _resolve_gpu
            _hr_px = int(round(ref_green.shape[0] * self.ratio)) * int(round(ref_green.shape[1] * self.ratio))
            _cupy_drizzle = (
                self.drizzle_deposit == "scatter"
                and self.drizzle_splat == "bilinear"
                and _resolve_gpu(self.use_gpu, "drizzle", _hr_px)
            )
            hr, coverage = drizzle(kept, transforms, self.ratio, weights.tolist(),
                                   interp=self.drizzle_interp,
                                   post_sigma=self.drizzle_post_sigma,
                                   footprint_sigma=self.drizzle_footprint_sigma,
                                   deposit=self.drizzle_deposit,
                                   pixfrac=self.drizzle_pixfrac,
                                   subsamples=self.drizzle_subsamples,
                                   splat=self.drizzle_splat,
                                   splat_sigma=self.drizzle_splat_sigma,
                                   low_cov_thresh=self.drizzle_low_cov_thresh,
                                   low_cov_sigma=self.drizzle_low_cov_sigma,
                                   use_opencl=ocl_active,
                                   use_gpu_scatter=scatter_gpu,
                                   robust_fusion=bool(robust_use),
                                   robust_k=self.robust_k,
                                   robust_iters=self.robust_iters,
                                   robust_min_weight=self.robust_min_weight,
                                   robust_stack=robust_stack,
                                   robust_localize=self.robust_localize,
                                   reference_pos=kept_indices.index(reference_index),
                                   engine_out=_engine_out,
                                   local_shifts=_local_shifts,
                                   local_weights=_local_weights,
                                   use_cupy=_cupy_drizzle)
            if self.fusion == "map":
                try:
                    if not all(f.mosaic is not None and f.pattern is not None for f in kept):
                        raise ValueError(
                            "fusion='map' requires every frame to carry calibrated "
                            "Bayer mosaic and pattern (loaded via load_raw_linear or "
                            "provided on RawFrame)"
                        )
                    from .map_sr import estimate_colour_matrix, map_super_resolution
                    ref_pos = kept_indices.index(reference_index)
                    _map_C, _map_C_info = estimate_colour_matrix(
                        kept[ref_pos].rgb, kept[ref_pos].mosaic, kept[ref_pos].pattern
                    )
                    _mkw = dict(self.map_kwargs)
                    _mkw.setdefault("ref_rgb", kept[ref_pos].rgb)
                    map_res = map_super_resolution(
                        hr,
                        [f.mosaic for f in kept],
                        transforms,
                        self.ratio,
                        kept[ref_pos].pattern,
                        weights=weights.tolist(),
                        colour_matrix=_map_C,
                        ref_index=ref_pos,
                        use_gpu=self.use_gpu,
                        mode=self.map_mode,
                        **_mkw,
                    )
                    hr = map_res.hr
                    _cfa_metrics = dict(map_res.metrics)
                    _cfa_metrics["colour_cond"] = _map_C_info.get("cond", 1.0)
                    _cfa_metrics["colour_fallback"] = bool(_map_C_info.get("fallback", False))
                    _engine_out["scatter"] = f"map_{map_res.metrics.get('engine', 'cpu')}"
                except Exception as _map_exc:
                    if self.fusion_requested != "auto":
                        raise
                    # auto: keep the demosaic_drizzle HR (MAP's own init) and
                    # continue on the default path (IBP below runs for it).
                    self.fusion = "demosaic_drizzle"
                    self.fusion_reason = (
                        f"auto: map failed ({type(_map_exc).__name__}), fell back")
                    _cfa_metrics = {}

        _reg_refine_metrics: dict = {"reg_refine_iters": 0}
        if self.reg_refine > 0 and len(kept) >= 2:
            import time as _time
            from .reg_refine import refine_burst_registration

            _t0_rr = _time.perf_counter()
            _ref_pos_rr = kept_indices.index(reference_index)
            _rr_iters_done = 0
            _rr_max_delta = 0.0
            _rr_mean_delta = 0.0
            _rr_gains = [1.0] * len(kept)
            _rr_offsets = [0.0] * len(kept)
            _rr_base_kept = list(kept)

            for _rr_it in range(self.reg_refine):
                _rr = refine_burst_registration(
                    hr, _rr_base_kept, transforms, _ref_pos_rr, self.ratio,
                    forward_sigma=self.ibp_forward_sigma,
                )
                _rr_iters_done = _rr_it + 1
                _rr_max_delta = _rr.max_corner_delta_lr_px
                _rr_mean_delta = _rr.mean_corner_delta_lr_px
                transforms = _rr.transforms
                _rr_gains = _rr.gains
                _rr_offsets = _rr.offsets

                # Apply refined per-frame photometric gain/offset to base frames
                _rr_kept = list(_rr_base_kept)
                for _ki, (_g, _o) in enumerate(zip(_rr_gains, _rr_offsets)):
                    if _ki == _ref_pos_rr:
                        continue
                    if abs(_g - 1.0) > 1e-3 or abs(_o) > 1e-4:
                        _fr = _rr_base_kept[_ki]
                        _inv_g = np.float32(1.0 / _g)
                        _o32 = np.float32(_o)
                        _rr_kept[_ki] = RawFrame(
                            name=_fr.name,
                            rgb=np.clip((_fr.rgb - _o32) * _inv_g, 0.0, 1.0),
                            green=np.clip((_fr.green - _o32) * _inv_g, 0.0, 1.0),
                            meta=_fr.meta,
                            mosaic=(None if _fr.mosaic is None
                                    else np.clip((_fr.mosaic - _o32) * _inv_g, 0.0, 1.0)),
                            pattern=_fr.pattern,
                        )
                kept = _rr_kept

                # Re-fuse with refined transforms and exposure
                if self.fusion in ("cfa_drizzle", "cfa_kr"):
                    mosaics_eq_rr, gr_gb_rr = equalize_green_imbalance(
                        [f.mosaic for f in kept], kept[_ref_pos_rr].pattern, ref_index=_ref_pos_rr
                    )
                    _cfa_C_rr, _cfa_C_rr_info = estimate_colour_matrix(
                        kept[_ref_pos_rr].rgb, mosaics_eq_rr[_ref_pos_rr], kept[_ref_pos_rr].pattern
                    )
                    _cfa_C_rr_inv = np.linalg.inv(_cfa_C_rr).astype(np.float32)
                    if self.fusion == "cfa_drizzle":
                        cfa_res = _cfa_drizzle(
                            mosaics_eq_rr,
                            transforms,
                            self.ratio,
                            kept[_ref_pos_rr].pattern,
                            weights=weights.tolist(),
                            pixfrac=self.drizzle_pixfrac,
                            subsamples=self.drizzle_subsamples,
                            local_shifts=_local_shifts,
                            local_weights=_local_weights,
                        )
                    else:
                        cfa_res = cfa_kernel_regression(
                            mosaics_eq_rr,
                            transforms,
                            self.ratio,
                            kept[_ref_pos_rr].pattern,
                            ref_green=ref_green,
                            weights=weights.tolist(),
                            fallback_hr=fallback_hr_cam,
                            robust_maps=robust_maps,
                            k_detail=self.cfa_k_detail,
                            chroma_scale=self.cfa_chroma_scale,
                            adaptive=self.cfa_adaptive,
                        )
                    hr = np.clip(np.matmul(cfa_res.hr, _cfa_C_rr_inv.T), 0.0, 1.0).astype(np.float32)
                    coverage = cfa_res.coverage
                    _cfa_metrics = dict(cfa_res.metrics)
                    _cfa_metrics["colour_cond"] = _cfa_C_rr_info.get("cond", 1.0)
                    _cfa_metrics["colour_fallback"] = bool(_cfa_C_rr_info.get("fallback", False))
                    _cfa_metrics["gr_gb_ratio"] = round(float(gr_gb_rr), 5)
                else:
                    hr, coverage = drizzle(
                        kept, transforms, self.ratio, weights.tolist(),
                        interp=self.drizzle_interp,
                        post_sigma=self.drizzle_post_sigma,
                        footprint_sigma=self.drizzle_footprint_sigma,
                        deposit=self.drizzle_deposit,
                        pixfrac=self.drizzle_pixfrac,
                        subsamples=self.drizzle_subsamples,
                        splat=self.drizzle_splat,
                        splat_sigma=self.drizzle_splat_sigma,
                        low_cov_thresh=self.drizzle_low_cov_thresh,
                        low_cov_sigma=self.drizzle_low_cov_sigma,
                        use_opencl=ocl_active,
                        use_gpu_scatter=scatter_gpu,
                        robust_fusion=bool(robust_use),
                        robust_k=self.robust_k,
                        robust_iters=self.robust_iters,
                        robust_min_weight=self.robust_min_weight,
                        robust_stack=None,
                        robust_localize=self.robust_localize,
                        reference_pos=_ref_pos_rr,
                        engine_out=_engine_out,
                        local_shifts=_local_shifts,
                        local_weights=_local_weights,
                        use_cupy=_cupy_drizzle,
                    )
                    if self.fusion == "map":
                        from .map_sr import estimate_colour_matrix, map_super_resolution
                        _map_C_rr, _map_C_rr_info = estimate_colour_matrix(
                            kept[_ref_pos_rr].rgb, kept[_ref_pos_rr].mosaic, kept[_ref_pos_rr].pattern
                        )
                        _mkw_rr = dict(self.map_kwargs)
                        _mkw_rr.setdefault("ref_rgb", kept[_ref_pos_rr].rgb)
                        map_res = map_super_resolution(
                            hr,
                            [f.mosaic for f in kept],
                            transforms,
                            self.ratio,
                            kept[_ref_pos_rr].pattern,
                            weights=weights.tolist(),
                            colour_matrix=_map_C_rr,
                            ref_index=_ref_pos_rr,
                            use_gpu=self.use_gpu,
                            mode=self.map_mode,
                            **_mkw_rr,
                        )
                        hr = map_res.hr
                        _cfa_metrics = dict(map_res.metrics)
                        _cfa_metrics["colour_cond"] = _map_C_rr_info.get("cond", 1.0)
                        _cfa_metrics["colour_fallback"] = bool(_map_C_rr_info.get("fallback", False))
                        _engine_out["scatter"] = f"map_{map_res.metrics.get('engine', 'cpu')}"
                if _rr_max_delta < 0.005:
                    break

            _reg_refine_metrics = {
                "reg_refine_iters": int(_rr_iters_done),
                "reg_refine_max_delta_lr_px": round(float(_rr_max_delta), 5),
                "reg_refine_mean_delta_lr_px": round(float(_rr_mean_delta), 5),
                "reg_refine_seconds": round(float(_time.perf_counter() - _t0_rr), 3),
                "reg_refine_gains": [round(float(g), 5) for g in _rr_gains],
                "reg_refine_offsets": [round(float(o), 6) for o in _rr_offsets],
            }

        # B2 memory: the gate's warped stack (up to ~750 MB at 4x / N=9) is
        # only needed for the ghost-energy metric. Measure it now, while the
        # stack is alive, and release it before IBP allocates its full-size
        # footprints and correction buffers.
        _ge = _interframe_ghost_energy(kept, transforms, self.ratio,
                                       stack=robust_stack,
                                       median=gate_cache.get("median"))
        robust_stack = None

        # 6. Iterative back-projection to restore the high frequency the
        #    deposition footprint attenuated, using the real measurements only.
        _ibp_engine_out: dict = {}
        if self.fusion == "demosaic_drizzle":
            hr = iterative_back_projection(
                hr, kept, transforms, self.ratio, weights.tolist(),
                iterations=self.ibp_iterations, lam=self.ibp_lambda,
                reg_sigma=self.ibp_reg_sigma, max_step=self.ibp_max_step,
                forward_sigma=self.ibp_forward_sigma,
                coverage=coverage, joint_refine=self.ibp_joint_refine,
                forward_sigma_shrink=self.ibp_forward_sigma_shrink,
                adjoint=self.ibp_adjoint,
                use_opencl=ocl_active,
                prox_noise_scale=self.ibp_prox_noise_scale,
                prox_space_lr=self.ibp_prox_space_lr,
                use_gpu=self.use_gpu,
                engine_out=_ibp_engine_out)

        # 7. Evidence metrics vs a bicubic-upsampled denoised stack baseline.
        #    The HF band is measured on the display-transformed view: in raw
        #    linear space on a dark burst the band is sensor noise, not detail.
        lr_stack = denoised_lr_stack(kept, transforms, weights.tolist())
        baseline = cv2.resize(lr_stack, (hr.shape[1], hr.shape[0]),
                              interpolation=cv2.INTER_CUBIC)
        # Which engine deposited the scatter pass — reused below so the device
        # report can consider EITHER GPU engine (warp routing or scatter).
        scatter_engine = (_engine_out.get("scatter", "cpu")
                          if self.drizzle_deposit == "scatter" else "cpu")
        gpu_device_used = bool(ocl_active) or scatter_engine == "gpu"
        # D3: opencl_active means "a GPU engine actually ran", i.e. EITHER the
        # T-API warp routing OR the scatter deposition used the device.
        # opencl_warp keeps the warp-routing decision alone, so nothing is lost.
        opencl_warp = bool(ocl_active)
        opencl_active = opencl_warp or (scatter_engine == "gpu")
        # D2: exclude the reference frame's self-comparison from the residual
        # and score aggregates (it is a trivial 1.0/0.0, not a measurement).
        _self_mask = np.asarray(align_self, dtype=bool)
        _ar_all = np.asarray(align_residuals, dtype=np.float64)
        _ar = (_ar_all[~_self_mask] if _self_mask.size
               else _ar_all)
        _sc = np.asarray(align_scores, dtype=np.float64)
        _sc_ns = _sc[~_self_mask] if _self_mask.size else _sc
        metrics = {
            "hf_gain_db": high_frequency_gain_db(to_view_float(hr),
                                                 to_view_float(baseline)),
            "hf_gain_db_linear": high_frequency_gain_db(hr, baseline),
            "sharpness_ratio": (laplacian_sharpness(hr)
                                / max(laplacian_sharpness(baseline), 1e-9)),
            "mean_shift_px": float(np.mean([np.hypot(dx, dy)
                                            for dx, dy in shifts])),
            "max_shift_px": float(np.max([np.hypot(dx, dy) for dx, dy in shifts])),
            # Per-frame registration residual summary (px), excluding the
            # reference self-comparison, and the truth-free inter-frame ghost
            # energy of the kept frames.
            "align_residual_mean_px": float(_ar.mean()) if _ar.size else 0.0,
            "align_residual_max_px": float(_ar.max()) if _ar.size else 0.0,
            "ghost_energy": _ge["ghost_energy"],
            "ghost_energy_norm": _ge["ghost_energy_norm"],
            "interframe_std": _ge["interframe_std"],
            "registration": self.registration,
            # IBP configuration actually used. ``ibp_iterations`` changed
            # default 3 -> 1 (fix/heldout2), so every archived result must be
            # read with this key: a UI/desktop run that pins nothing uses the
            # core default and reports it here, which is what makes two runs
            # comparable without re-deriving the configuration from the image.
            "ibp_iterations": self.ibp_iterations,
            "ibp_lambda": float(self.ibp_lambda),
            "ibp_reg_sigma": float(self.ibp_reg_sigma),
            "ibp_max_step": (None if self.ibp_max_step is None
                             else float(self.ibp_max_step)),
            "ibp_forward_sigma_hr": float(
                SENSOR_PSF_SIGMA_PER_RATIO * self.ratio
                if self.ibp_forward_sigma is None
                else self.ibp_forward_sigma),
            "ibp_joint_refine": self.ibp_joint_refine,
            "ibp_adjoint": self.ibp_adjoint,
            "ibp_prox_noise_scale": float(self.ibp_prox_noise_scale),
            "ibp_prox_space_lr": float(self.ibp_prox_space_lr),
            # Display-ready one-line description of the refinement actually
            # applied ("" when IBP is off). Exists so a report/UI can show the
            # configuration next to hf_gain_db without knowing the defaults.
            "ibp_summary": (
                ("%d sweep%s x lambda %.2f%s, forward sigma %.2f HR px"
                 % (self.ibp_iterations,
                    "" if self.ibp_iterations == 1 else "s",
                    self.ibp_lambda,
                    ", adjoint" if self.ibp_adjoint else ", plain splat",
                    SENSOR_PSF_SIGMA_PER_RATIO * self.ratio
                    if self.ibp_forward_sigma is None
                    else self.ibp_forward_sigma))
                if self.ibp_iterations > 0 else ""),
            # Resolved ratio-aware deposit parameters (see
            # default_deposit_params): the footprint stays 0.5 LR px at every
            # ratio and the sub-sample lattice scales with it in HR pixels
            # (2 / 3 / 3 at 2x / 3x / 4x).
            "drizzle_pixfrac": float(self.drizzle_pixfrac),
            "drizzle_subsamples": int(self.drizzle_subsamples),
            "reference_index": int(reference_index),
            # Fusion weighting (see _fusion_weight): the noise estimate each
            # kept frame was weighted by, and how far the normalized weights
            # spread (1.0 = uniform). Large spread means a mixed-noise burst.
            "frame_noise_sigma": [float(s) for s in kept_sigma],
            "fusion_weight_spread": (
                float(np.max(weights) / np.min(weights))
                if len(weights) and float(np.min(weights)) > 0.0 else 1.0),
            "mean_align_score": (float(_sc_ns.mean()) if _sc_ns.size else 0.0),
            # Frame acceptance, reported so a drop is never silent: the gate
            # threshold actually used (and how it was resolved), every frame it
            # dropped with the ZNCC that failed it, and a one-line reason. B8:
            # the old default 0.0 fused white-noise frames without a word.
            # ``align_gate_advisory`` lists frames a strict gate WOULD have
            # dropped on a frame too small for the statistic to separate junk
            # from aliased content (N2) -- they were kept.
            "min_align_score": float(self.min_align_score),
            "align_gate_threshold": float(gate_threshold),
            "align_gate_mode": gate_mode,
            "align_gate_dropped": gate_dropped,
            "align_gate_advisory": gate_advisory,
            "align_gate_reason": (
                "gate off"
                if gate_mode == "off" else
                ("advisory: %d px frame is below %d px, where junk and "
                 "above-Nyquist content score alike; kept all %d frames "
                 "(would have dropped %d)"
                 % (_roi_min_px, _ALIGN_GATE_MIN_ROI_PX, len(kept),
                    len(gate_advisory))
                 if gate_mode == "advisory" else
                 ("no frame dropped: every registered frame scored >= %.2f"
                  " (%s threshold; burst median %.2f)"
                  % (gate_threshold, gate_mode, _scores_median)
                  if not gate_dropped else
                  "%d frame%s dropped: alignment ZNCC below %.2f (%s "
                  "threshold; burst median %.2f) -- registration failed or "
                  "the frame is noise, so fusing it adds ghost/blur only; "
                  "fused %d of %d"
                  % (len(gate_dropped),
                     "" if len(gate_dropped) == 1 else "s",
                     gate_threshold, gate_mode, _scores_median,
                     len(kept), len(frames))))),
            # Reference selection (see _select_reference): candidates rejected
            # by the consistency gate, and the burst-centroid proximity that
            # chose the reference. B8 #2: with the shift sign fixed this gate
            # actually rejects junk instead of rejecting every frame that is
            # displaced by more than ~2 px.
            "reference_consistency_rejected": ref_stats.get("rejected", []),
            "reference_consistency_scores": ref_stats.get("scores", []),
            # Robust-fusion decision (tri-state auto resolved above).
            "robust_active": bool(robust_use),
            "robust_reason": robust_reason,
            # opencl_active: a GPU engine ran (warp OR scatter). opencl_warp:
            # the T-API warp routing decision alone. Both are documented.
            "opencl_active": opencl_active,
            "opencl_warp": opencl_warp,
            # Which device actually executed a GPU engine (None = neither the
            # warps nor the scatter deposition ran on the GPU). Either engine
            # engages the GPU independently, so the name is reported whenever
            # the scatter engine used it too.
            "opencl_device": opencl_device_name() if gpu_device_used else None,
            # Which engine deposited the scatter pass, and its wall seconds
            # ('cpu' also covers 'gather' mode and every GPU fallback).
            "opencl_scatter": scatter_engine,
            "scatter_seconds": float(_engine_out.get("seconds", 0.0)),
            "gpu_drizzle": scatter_engine == "cupy",
            "gpu_ibp": _ibp_engine_out.get("ibp") == "cupy",
            "gpu_stages": [
                s for s, on in (
                    ("drizzle", scatter_engine == "cupy"),
                    ("ibp", _ibp_engine_out.get("ibp") == "cupy"),
                ) if on
            ],
            "fusion": self.fusion,
            "fusion_requested": self.fusion_requested,
            "fusion_reason": self.fusion_reason,
            "exposure_gate_dropped": exp_dropped,
            "exposure_scales": [round(float(exp_scales[i]), 4) for i in kept_indices],
            **_local_metrics,
            **_reg_refine_metrics,
            **({f"cfa_kr_{k}": v for k, v in _cfa_metrics.items()
                if k not in ("pattern",)} if _cfa_metrics else {}),
        }
        self.opencl_active = bool(opencl_active)

        return SRResult(
            hr=hr,
            ratio=self.ratio,
            shifts=shifts,
            weights=weights,
            coverage=coverage,
            used_frames=len(kept),
            total_frames=len(frames),
            metrics=metrics,
            transforms=transforms,
            kept_indices=kept_indices,
            align_scores=align_scores,
            align_residuals=align_residuals,
            align_self=align_self,
            reference_index=int(reference_index),
            local_shifts=_local_shifts,
            local_weights=_local_weights,
        )


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #

def to_uint16_linear(img: np.ndarray) -> np.ndarray:
    """Convert a linear float 0..1 image to 16-bit without gamma.

    Rounds to nearest rather than truncating, so the quantisation carries no
    systematic -0.5 LSB bias.
    """
    return np.rint(np.clip(img * 65535.0, 0, 65535)).astype(np.uint16)


def to_uint8_view(img: np.ndarray, gamma: float = 2.2, exposure: float = 1.0) -> np.ndarray:
    """Non-destructive display transform for on-screen / PNG preview."""
    v = np.clip(img * exposure, 0.0, 1.0)
    v = np.power(v, 1.0 / max(gamma, 1e-6))
    return (v * 255.0 + 0.5).astype(np.uint8)
