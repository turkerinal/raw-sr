"""CFA-domain multi-frame drizzle — **EXPERIMENTAL, not used in production**.

Status (2026-11-20, after the independent audit ``internal audit report``):
the module is geometrically correct on real DNGs (phase, ROI alignment,
per-site calibration all verified) but it does **not** beat the shipped
demosaic-then-drizzle pipeline on ground truth.  With the audit's demosaic bug
fixed, the measured paired difference is **-1.2…+0.3 dB** at 2x (seed- and
target-dependent) and about **-0.6/-0.9 dB** at 3x/4x for the default footprint,
and the extra high-frequency energy it produces is partly **false colour**
(chroma zipper along edges; chroma-consistency PSNR 3-6 dB worse than the
pipeline).  It is kept as a research prototype behind no default path: nothing
in ``superres.py``, the UI, the desktop app or the acceptance defaults calls
it.  Do not quote it as a resolution improvement; see
``docs/CFA_DRIZZLE.md`` and ``internal audit report`` for the corrected
numbers and the two harness defects that originally produced the wrong claim.

Why this module exists
----------------------
The shipped :mod:`raw_sr.core.superres` pipeline demosaics every RAW frame
(LibRaw/DCB) and then drizzles the resulting **RGB** images.  Two thirds of
every deposited RGB pixel is therefore an *interpolated* value, not a
measurement, and the demosaic has already mixed the four Bayer sites of each
2x2 block.  A CFA-domain deposit instead puts each measured site at its true
sub-pixel position in **its own channel plane** and normalises per channel, so

* only real measurements are deposited (no invented samples),
* the green channel keeps its (comparatively dense) true sampling lattice,
* R/B chroma is reconstructed from real R/B sites only, and
* sub-pixel placement is the same physical placement the mosaic had.

This is the "drizzle on CFA" / true multi-frame demosaic direction.  It is a
standalone module: :mod:`superres` is deliberately not modified.

What it is *not*
----------------
It does not fix the held-out PSNR metric.  On the real 9-DNG burst that metric
is noise-limited (see ``internal audit report``): a perfect prediction of a
single noisy held-out frame can gain at most ``10*log10(1 + 1/N)`` dB over an
N-frame denoised stack (~0.5 dB here), and the measured residual is already at
that floor.  CFA deposition is therefore validated on **synthetic bursts with
known HR ground truth** (see ``scripts/exp_cfa_sr.py``) and on HF/sharpness
metrics for real data; held-out gain stays ≈0 dB either way.

API
---
``cfa_drizzle(mosaics, aligns, ratio, pattern, ...) -> CFADrizzleResult``

* ``mosaics``: sequence of ``(H, W)`` float arrays, black-subtracted and
  white-normalised to 0..1 (see :func:`load_cfa_frame`).
* ``aligns``: the same ref->frame transforms :func:`superres.drizzle` takes
  (a ``(dx, dy)`` translation or a 3x3 homography, in LR pixels).
* ``pattern``: the 2x2 Bayer colour-index array rawpy reports
  (``raw.raw_pattern``: 0=R, 1=G, 2=B, 3=G), or a string such as ``"BGGR"``.
* ``wb``: optional per-channel white-balance gains to apply to the measured
  samples before deposition (default 1,1,1 = already balanced).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .superres import to_drizzle_matrix

_COLOR_NAMES = {0: "R", 1: "G", 2: "B", 3: "G"}


# --------------------------------------------------------------------------- #
# Pattern / data containers
# --------------------------------------------------------------------------- #
def pattern_array(pattern) -> np.ndarray:
    """Return a validated 2x2 colour-index pattern (0=R, 1=G, 2=B, 3=G)."""
    if isinstance(pattern, str):
        s = pattern.upper()
        if len(s) != 4 or set(s) != {"R", "G", "B"} or s.count("G") != 2:
            raise ValueError(f"bad Bayer pattern string {pattern!r}")
        code = {"R": 0, "G": 1, "B": 2}
        arr = np.array([[code[s[0]], code[s[1]]],
                        [code[s[2]], code[s[3]]]], dtype=np.int8)
    else:
        arr = np.asarray(pattern, dtype=np.int8)
    if arr.shape != (2, 2):
        raise ValueError(f"Bayer pattern must be 2x2, got {arr.shape}")
    if arr.min() < 0 or arr.max() > 3:
        raise ValueError("Bayer colour indices must be in 0..3")
    g = np.isin(arr, (1, 3))
    if int(g.sum()) != 2:
        raise ValueError("a Bayer pattern must contain exactly two green sites")
    # The two greens of a Bayer CFA sit on a diagonal of the 2x2 block.
    if not ((g[0, 0] and g[1, 1]) or (g[0, 1] and g[1, 0])):
        raise ValueError("Bayer greens must occupy a 2x2 diagonal")
    return arr


def channel_map(pattern, shape: Tuple[int, int]) -> np.ndarray:
    """Map every mosaic site to channel 0 (R), 1 (G) or 2 (B).

    Both green phases (rawpy indices 1 and 3) map to channel 1.
    """
    pat = pattern_array(pattern)
    h, w = int(shape[0]), int(shape[1])
    ch = np.empty((h, w), dtype=np.int8)
    for r in range(2):
        for c in range(2):
            ci = int(pat[r, c])
            ch[r::2, c::2] = 1 if ci in (1, 3) else (0 if ci == 0 else 2)
    return ch


@dataclass
class CFAFrame:
    """One calibrated Bayer mosaic (values 0..1, black-subtracted)."""
    name: str
    mosaic: np.ndarray          # (H, W) float32
    pattern: np.ndarray         # 2x2 colour indices
    meta: dict = field(default_factory=dict)


@dataclass
class CFADrizzleResult:
    """HR RGB estimate plus the per-channel support it was built from."""
    hr: np.ndarray              # (H*r, W*r, 3) float32 linear RGB
    coverage: np.ndarray        # (H*r, W*r) float32, fraction of channels filled
    channel_coverage: Optional[np.ndarray]  # (3, H*r, W*r) or None
    metrics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def calibrate_cfa_levels(frame: "CFAFrame", reference_rgb: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel gains that put a mosaic on ``reference_rgb``'s level scale.

    LibRaw's colour scaling does not exactly reproduce
    ``camera_whitebalance / green``: measured on the project DNG, the ratio of
    the demosaiced level to the mosaic level at the *same sites* is R 0.965,
    G 0.973, B 1.057 (i.e. up to 5.7 % per channel).  Comparing a CFA
    reconstruction with a LibRaw RGB pipeline without removing that offset
    would charge it a constant per-channel bias, so any such comparison must
    calibrate first.

    The gain for each channel is the least-squares scale of the reference's
    value onto the mosaic's value at that channel's sites
    (``<ref*mosaic> / <mosaic^2>``), which is insensitive to a smooth scene
    gradient and to which pixel happens to be brightest.

    Args:
        frame: a loaded :class:`CFAFrame` (``load_cfa_frame``).
        reference_rgb: the same ROI loaded by :func:`superres.load_raw_linear`,
            i.e. the LibRaw post-processed RGB image.

    Returns:
        ``(gains, stats)``: a length-3 float array (R, G, B) and a dict with the
        measured ratios for reporting.  Multiply the mosaic's sites by
        ``gains[channel]`` (see ``apply_level_gains``) to compare the two.
    """
    ref = np.asarray(reference_rgb, dtype=np.float64)
    m = np.asarray(frame.mosaic, dtype=np.float64)
    if ref.shape[:2] != m.shape[:2]:
        raise ValueError(f"reference {ref.shape[:2]} != mosaic {m.shape[:2]}")
    ch = channel_map(frame.pattern, m.shape)
    gains = np.ones(3, dtype=np.float64)
    stats = {}
    for c in range(3):
        mask = ch == c
        if not np.any(mask):
            continue
        a = m[mask]
        b = ref[:, :, c][mask]
        den = float(np.dot(a, a))
        g = float(np.dot(a, b)) / den if den > 0 else 1.0
        gains[c] = g
        stats[_COLOR_NAMES[c]] = {
            "gain": g,
            "mean_ratio": (float(b.mean() / a.mean()) if a.mean() else 1.0),
        }
    return gains, stats


def apply_level_gains(frame: "CFAFrame", gains: Sequence[float]) -> "CFAFrame":
    """Return a copy of ``frame`` with per-channel level gains applied."""
    g = np.asarray(gains, dtype=np.float64)
    if g.size != 3:
        raise ValueError("gains must have three entries (R, G, B)")
    out = frame.mosaic.copy()
    ch = channel_map(frame.pattern, out.shape)
    for c in range(3):
        out[ch == c] = (out[ch == c].astype(np.float64) * g[c]).astype(
            np.float32)
    meta = dict(frame.meta)
    meta["level_gains"] = g.tolist()
    return CFAFrame(name=frame.name, mosaic=out, pattern=frame.pattern,
                    meta=meta)


def load_cfa_frame(path: str, roi: Optional[Tuple[int, int, int, int]] = None,
                   apply_wb: bool = True) -> CFAFrame:
    """Load a RAW file as a calibrated Bayer mosaic.

    ``roi`` is ``(x, y, w, h)`` in the *post-processed* image coordinate
    system (the same one :func:`superres.load_raw_linear` uses).  On the DNGs
    in this project rawpy reports ``crop_left_margin == crop_top_margin == 0``
    and ``raw_image_visible`` is already aligned with the post-processed crop,
    so the ROI is used as given; an odd origin is rounded down to keep the CFA
    phase and recorded in ``meta["roi_phase_adjust"]``.

    ``apply_wb`` multiplies each channel by its camera white-balance gain,
    matching ``load_raw_linear(..., use_camera_wb=True)``.  The returned mosaic
    is therefore directly comparable with that RGB loader's colour space.
    """
    import rawpy

    with rawpy.imread(path) as raw:
        pat = np.asarray(raw.raw_pattern, dtype=np.int8).copy()
        black = np.asarray(raw.black_level_per_channel, dtype=np.float64)
        white = float(raw.white_level)
        wb = np.asarray(raw.camera_whitebalance[:3], dtype=np.float64)
        vis = raw.raw_image_visible
        meta = {
            "black_level": black.tolist(),
            "white_level": white,
            "camera_whitebalance": wb.tolist(),
            "raw_shape": tuple(vis.shape),
            "roi_phase_adjust": (0, 0),
        }
        x, y, w, h = (0, 0, vis.shape[1], vis.shape[0]) if roi is None else roi
        x0, y0 = int(x), int(y)
        adj = (0, 0)
        if x0 % 2 or y0 % 2:
            adj = (x0 % 2, y0 % 2)
            x0 -= x0 % 2
            y0 -= y0 % 2
        meta["roi_phase_adjust"] = adj
        m = vis[y0:y0 + int(h), x0:x0 + int(w)].astype(np.float64)

    # Per-site black subtraction (black levels may differ per channel).
    bl = np.empty_like(m)
    for r in range(2):
        for c in range(2):
            bl[r::2, c::2] = black[int(pat[r, c])]
    span = np.empty_like(m)
    for r in range(2):
        for c in range(2):
            span[r::2, c::2] = white - black[int(pat[r, c])]
    m = np.clip((m - bl) / np.maximum(span, 1.0), 0.0, 1.0)

    if apply_wb and wb.size >= 3 and wb[1] > 0:
        g = float(wb[1])
        gains = np.array([wb[0] / g, 1.0, wb[2] / g], dtype=np.float64)
        ch = channel_map(pat, m.shape)
        for c in range(3):
            m[ch == c] *= gains[c]
        meta["applied_wb_gains"] = gains.tolist()

    name = path.rsplit("/", 1)[-1]
    return CFAFrame(name=name, mosaic=m.astype(np.float32), pattern=pat,
                    meta=meta)


# --------------------------------------------------------------------------- #
# Deposition
# --------------------------------------------------------------------------- #
def _deposit_plane(vals: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                   M: np.ndarray, out_shape: Tuple[int, int],
                   pixfrac: float, subsamples: int,
                   fweight: float = 1.0,
                   local_du: Optional[np.ndarray] = None,
                   local_dv: Optional[np.ndarray] = None,
                   rweight: Optional[np.ndarray] = None,
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Drizzle-splat one channel's measured samples onto the HR grid.

    Mirrors :func:`superres._scatter_accumulate` for a single plane: every
    measured site is placed at ``M @ (x, y)`` and spread with a compact
    bilinear splat over its ``pixfrac`` LR-pixel footprint (``subsamples``
    sub-samples per axis, each weighted ``fweight / subsamples**2``).
    """
    oh, ow = int(out_shape[0]), int(out_shape[1])
    M = np.asarray(M, dtype=np.float64)
    if not np.all(np.isfinite(M)) or abs(M[2, 2]) < 1e-12:
        M = np.eye(3, dtype=np.float64)
    else:
        M = M / M[2, 2]

    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    vals = np.asarray(vals, dtype=np.float64).ravel()
    if vals.size != xs.size or vals.size != ys.size:
        raise ValueError("vals, xs and ys must have the same length")
    if vals.size == 0:
        return (np.zeros((oh, ow), np.float64),
                np.zeros((oh, ow), np.float64))

    n_sub = max(int(subsamples), 1)
    p = float(pixfrac) if (pixfrac is not None and pixfrac > 0) else 1.0
    offs = ((np.arange(n_sub) + 0.5) / n_sub - 0.5) * p
    subw = float(fweight) / float(n_sub * n_sub)

    acc = np.zeros(oh * ow, dtype=np.float64)
    wt = np.zeros(oh * ow, dtype=np.float64)
    m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
    m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
    m20, m21 = M[2, 0], M[2, 1]

    for oy in offs:
        Yp = ys + float(oy)
        for ox in offs:
            Xp = xs + float(ox)
            den = m20 * Xp + m21 * Yp + 1.0
            U = (m00 * Xp + m01 * Yp + m02) / den
            V = (m10 * Xp + m11 * Yp + m12) / den
            if local_du is not None:
                U = U + local_du
            if local_dv is not None:
                V = V + local_dv
            bu = np.floor(U)
            bv = np.floor(V)
            tu = U - bu
            tv = V - bv
            bu = bu.astype(np.int64)
            bv = bv.astype(np.int64)
            for du, wu in ((0, 1.0 - tu), (1, tu)):
                uu = bu + du
                ok_u = (uu >= 0) & (uu < ow)
                if not ok_u.any():
                    continue
                for dv, wv in ((0, 1.0 - tv), (1, tv)):
                    vv = bv + dv
                    m = ok_u & (vv >= 0) & (vv < oh)
                    if not m.any():
                        continue
                    w = (wu[m] * wv[m]) * subw
                    if rweight is not None:
                        w = w * rweight[m]
                    idx = vv[m] * ow + uu[m]
                    wt += np.bincount(idx, weights=w, minlength=oh * ow)
                    acc += np.bincount(idx, weights=vals[m] * w,
                                       minlength=oh * ow)
    return acc.reshape(oh, ow), wt.reshape(oh, ow)


# --------------------------------------------------------------------------- #
# Hole filling
# --------------------------------------------------------------------------- #
def _fill_multiscale(est: np.ndarray, wt: np.ndarray,
                     max_sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Fill uncovered pixels by multiscale normalised convolution.

    Only holes are written; measured values are never blurred.  At each scale
    the normalised convolution ``G_s*(est*valid) / G_s*valid`` is evaluated and
    used wherever it has support.  This reconstructs a smooth chroma/detail
    field across the gaps between a channel's sampling lattice, which is what
    a per-channel CFA deposit needs at 3x/4x where one frame leaves holes.
    """
    valid = wt > 1e-8
    out = np.where(valid, est, 0.0).astype(np.float32)
    if valid.all() or not valid.any():
        if not valid.any():
            out[:] = float(est[valid].mean()) if valid.any() else 0.0
        return out, valid

    sigmas = []
    s = 0.7
    while s < max_sigma:
        sigmas.append(s)
        s *= 1.7
    sigmas.append(max_sigma)
    filled = out.copy()
    known = valid.copy()
    for s in sigmas:
        num = cv2.GaussianBlur(np.where(known, filled, 0.0),
                               (0, 0), float(s),
                               borderType=cv2.BORDER_REPLICATE)
        den = cv2.GaussianBlur(known.astype(np.float32), (0, 0), float(s),
                               borderType=cv2.BORDER_REPLICATE)
        cand = num / np.maximum(den, 1e-8)
        use = (~known) & (den > 1e-4)
        filled[use] = cand[use]
        known |= use
        if known.all():
            break
    if not known.all():
        # A pixel no scale reached (empty channel / tiny mosaic): use the mean.
        filled[~known] = float(filled[known].mean()) if known.any() else 0.0
        known |= (~known)
    return filled, known


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def cfa_drizzle(mosaics: Sequence[np.ndarray],
                aligns: Sequence,
                ratio: float,
                pattern,
                weights: Optional[Sequence[float]] = None,
                wb: Optional[Sequence[float]] = None,
                pixfrac: float = 0.5,
                subsamples: Optional[int] = None,
                coverage_threshold: float = 1e-8,
                return_channel_coverage: bool = False,
                local_shifts: Optional[Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]] = None,
                local_weights: Optional[Sequence[np.ndarray]] = None,
                ) -> CFADrizzleResult:
    """Deposit CFA samples per channel and reconstruct an HR RGB image.

    Args:
        mosaics: per-frame ``(H, W)`` calibrated Bayer mosaics (0..1).
        aligns: per-frame ref->frame transforms, LR pixels, same convention
            as :func:`superres.drizzle`.
        ratio: HR/LR scale factor.
        pattern: 2x2 Bayer colour indices or a pattern string.
        weights: per-frame fusion weights (default uniform).  They are
            normalised by the accumulated coverage, so only ratios matter.
        wb: optional per-channel white-balance gains applied to the samples.
        pixfrac: deposit footprint in LR pixels.  Default **0.5**, the same
            compact splat the shipped RGB drizzle uses.  A wider footprint
            (1.0) averages more samples per HR pixel and measures ~0.7-0.9 dB
            higher *total* PSNR on the noisy synthetic burst, but it is much
            softer on real data (2x HF +7.8 dB vs the stack against +13.9 dB
            at 0.5, and 1.1x vs 4.4x Laplacian sharpness): total PSNR is
            dominated by low frequencies and cannot see that loss.  At 0.5 the
            CFA deposit beats the demosaiced RGB pipeline by +2.6/+2.0/+1.7 dB
            GT PSNR at 2x/3x/4x *and* keeps at least the pipeline's HF gain,
            so it is the honest default.  The footprint is exposed because the
            noise/sharpness trade-off is scene-dependent.
        subsamples: sub-samples per footprint axis.  ``None`` derives
            ``ceil(2 * pixfrac * ratio)`` (the same rule the RGB drizzle uses),
            capped at 4 so the cost stays bounded; at the default 0.5 that is
            2/3/4 at 2x/3x/4x and gives a hole-free sub-sample lattice.
        coverage_threshold: coverage below which a pixel is reported as
            unsupported in the metrics.  Hole filling itself uses an exact
            zero-coverage test, so a measurement is never blurred; this knob
            only controls the reported ``measured_support_fraction``.
        return_channel_coverage: also return the per-channel coverage maps
            (3, H*r, W*r); costs 3x the coverage memory.

    Returns:
        :class:`CFADrizzleResult` with the HR RGB estimate, a coverage map
        (fraction of channels with support, 0..1), optional per-channel
        coverage and ``metrics`` describing the resolved configuration and
        fill fractions.

    Notes:
        * All three channels are deposited independently; no demosaic is
          applied at any point.  Green uses both green phases.
        * Holes (if any) are filled by :func:`_fill_multiscale`, which only
          writes where no measurement landed; measured values are preserved.
        * The returned HR grid is the reference frame's LR grid scaled by
          ``ratio`` on the same **pixel-centre aligned** geometry as
          :func:`superres.drizzle` (HR pixel ``i`` at LR coordinate
          ``(i - (ratio-1)/2)/ratio``).
    """
    pat = pattern_array(pattern)
    n = len(mosaics)
    if n == 0:
        raise ValueError("cfa_drizzle needs at least one frame")
    if len(aligns) != n:
        raise ValueError(f"{n} mosaics but {len(aligns)} transforms")
    if ratio <= 0:
        raise ValueError("ratio must be > 0")
    h, w = np.asarray(mosaics[0]).shape[:2]
    for i, m in enumerate(mosaics):
        if np.asarray(m).shape[:2] != (h, w):
            raise ValueError(
                f"mosaic {i} shape {np.asarray(m).shape[:2]} != {(h, w)}")
    oh, ow = int(round(h * ratio)), int(round(w * ratio))

    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(f"{n} mosaics but {len(weights)} weights")
    if wb is None:
        wb = (1.0, 1.0, 1.0)
    wb = np.asarray(wb, dtype=np.float64)
    if wb.size != 3:
        raise ValueError("wb must have three gains")

    if subsamples is None:
        subsamples = int(max(2, min(4, np.ceil(2.0 * float(pixfrac) * ratio))))
    subsamples = int(max(1, subsamples))

    ch = channel_map(pat, (h, w))
    mats = [to_drizzle_matrix(a, ratio).astype(np.float64) for a in aligns]

    hr = np.zeros((oh, ow, 3), dtype=np.float32)
    cov_out = np.zeros((oh, ow), dtype=np.float32)
    channel_cov = (np.zeros((3, oh, ow), dtype=np.float32)
                   if return_channel_coverage else None)
    fill_frac = {}
    support_frac = {}

    # One channel at a time keeps the float64 accumulators to one plane.
    for c in range(3):
        mask = ch == c
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            fill_frac[c] = 1.0
            support_frac[c] = 0.0
            continue
        acc = np.zeros((oh, ow), dtype=np.float64)
        wt = np.zeros((oh, ow), dtype=np.float64)
        for k in range(n):
            vals = np.asarray(mosaics[k], dtype=np.float64)[mask] * wb[c]
            ldu = (np.asarray(local_shifts[0][k], dtype=np.float64)[mask] * float(ratio)
                   if local_shifts is not None else None)
            ldv = (np.asarray(local_shifts[1][k], dtype=np.float64)[mask] * float(ratio)
                   if local_shifts is not None else None)
            rw_k = (np.asarray(local_weights[k], dtype=np.float64)[mask]
                    if local_weights is not None else None)
            a, v = _deposit_plane(vals, xs, ys, mats[k], (oh, ow),
                                  float(pixfrac), subsamples,
                                  fweight=float(weights[k]),
                                  local_du=ldu, local_dv=ldv, rweight=rw_k)
            acc += a
            wt += v
        est = acc / np.maximum(wt, 1e-12)
        known = wt > float(coverage_threshold)
        support_frac[c] = float(known.mean())
        # Fill only where the channel has no measurement; max_sigma scales
        # with the lattice gap (green: 2 LR px, R/B: 2 LR px in each axis).
        filled, _ = _fill_multiscale(est, wt, max_sigma=max(4.0, 2.0 * ratio))
        fill_frac[c] = float((~known).mean())
        hr[:, :, c] = filled.astype(np.float32)
        cov_out += known.astype(np.float32) / 3.0
        if channel_cov is not None:
            channel_cov[c] = known.astype(np.float32)

    np.clip(hr, 0.0, 1.0, out=hr)
    metrics = {
        "method": "cfa_drizzle",
        "ratio": float(ratio),
        "frames": int(n),
        "pattern": [[int(v) for v in row] for row in pat],
        "pixfrac_lr": float(pixfrac),
        "subsamples": int(subsamples),
        "wb": wb.tolist(),
        "coverage_fraction": float((cov_out > 0).mean()),
        "fill_fraction": {_COLOR_NAMES[c]: fill_frac.get(c, 1.0)
                          for c in range(3)},
        "measured_support_fraction": {
            _COLOR_NAMES[c]: support_frac.get(c, 0.0) for c in range(3)},
    }
    return CFADrizzleResult(hr=hr, coverage=cov_out,
                            channel_coverage=channel_cov,
                            metrics=metrics)


# --------------------------------------------------------------------------- #
# Wronski-style Anisotropic Structure-Tensor Kernel Regression (CFA-KR)
# --------------------------------------------------------------------------- #
def _immerkaer_noise_sigma(gray: np.ndarray) -> float:
    """Fast Immerkaer (1996) noise standard deviation estimator."""
    g = np.asarray(gray, dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    k = np.array([[1.0, -2.0, 1.0],
                  [-2.0, 4.0, -2.0],
                  [1.0, -2.0, 1.0]], dtype=np.float32)
    r = cv2.filter2D(g, cv2.CV_32F, k, borderType=cv2.BORDER_REPLICATE)
    return float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(r)) / 6.0)


def compute_structure_tensor_precision(
    ref_green: np.ndarray,
    noise_sigma: Optional[float] = None,
    k_detail: float = 0.28,
    k_denoise_min: float = 0.55,
    k_denoise_max: float = 0.85,
    k_stretch: float = 2.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute half-res anisotropic Gaussian precision matrices (a, b, c).

    Follows Wronski et al. (2019) "Handheld Multi-Frame Super-Resolution":
    - Computes the local structure tensor on a half-resolution green plane.
    - Along edges (high anisotropy), the kernel is compact across the edge
      (``sigma_perp ~ k_detail`` LR px) and elongated along the edge
      (``sigma_para ~ k_detail * k_stretch`` LR px).
    - In flat regions, the kernel is isotropic and noise-adaptive
      (``sigma ~ k_denoise`` LR px), suppressing 4x4 grid artifacts and noise.

    Returns:
        ``(a, b, c, info)`` where ``[[a, b], [b, c]]`` is the symmetric
        positive-definite precision matrix ``Omega^{-1}`` in LR pixel^-2
        coordinates on the half-res grid.
    """
    g = np.asarray(ref_green, dtype=np.float32)
    h, w = g.shape[:2]
    if noise_sigma is None:
        noise_sigma = _immerkaer_noise_sigma(g)
    noise_sigma = float(max(noise_sigma, 0.0))

    # Half-res green for clean, fast structure tensor
    if h >= 32 and w >= 32:
        g_half = cv2.resize(g, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    else:
        g_half = g

    gx = cv2.Sobel(g_half, cv2.CV_32F, 1, 0, ksize=3) * np.float32(0.125)
    gy = cv2.Sobel(g_half, cv2.CV_32F, 0, 1, ksize=3) * np.float32(0.125)

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)

    trace = jxx + jyy
    diff = jxx - jyy
    disc = np.sqrt(np.maximum(diff * diff + 4.0 * jxy * jxy, 0.0))
    lam1 = 0.5 * (trace + disc)
    lam2 = 0.5 * np.maximum(trace - disc, 0.0)

    # Dominant eigenvector angle (gradient / across-edge direction)
    theta = 0.5 * np.arctan2(2.0 * jxy, diff)
    ct = np.cos(theta).astype(np.float32)
    st = np.sin(theta).astype(np.float32)

    # Noise-adaptive flatness D in [0, 1] and anisotropy A in [1, 2]
    noise_floor = float(max(1e-7, (noise_sigma * 2.5) ** 2))
    d_tr = float(max(0.012, 6.0 * noise_sigma))
    s1 = np.sqrt(np.maximum(lam1, 0.0))
    flatness = np.clip(1.0 - s1 / d_tr, 0.0, 1.0).astype(np.float32)

    aniso = 1.0 + np.sqrt(
        np.maximum(lam1 - lam2, 0.0) / (lam1 + lam2 + noise_floor)
    ).astype(np.float32)

    k_denoise = float(np.clip(k_denoise_min + 15.0 * noise_sigma,
                              k_denoise_min, k_denoise_max))
    stretch = 1.0 + (aniso - 1.0) * float(k_stretch - 1.0)

    sigma_perp = (1.0 - flatness) * float(k_detail) + flatness * k_denoise
    sigma_para = (1.0 - flatness) * (float(k_detail) * stretch) + flatness * k_denoise

    inv_p2 = 1.0 / np.maximum(sigma_perp * sigma_perp, 1e-4)
    inv_t2 = 1.0 / np.maximum(sigma_para * sigma_para, 1e-4)

    ct2 = ct * ct
    st2 = st * st
    a = (ct2 * inv_p2 + st2 * inv_t2).astype(np.float32)
    b = (ct * st * (inv_p2 - inv_t2)).astype(np.float32)
    c = (st2 * inv_p2 + ct2 * inv_t2).astype(np.float32)

    info = {
        "noise_sigma": noise_sigma,
        "k_detail": float(k_detail),
        "k_denoise": k_denoise,
        "k_stretch": float(k_stretch),
        "mean_sigma_perp": float(sigma_perp.mean()),
        "mean_sigma_para": float(sigma_para.mean()),
        "mean_anisotropy": float(aniso.mean()),
    }
    return a, b, c, info


def balance_gr_gb(
    mosaic: np.ndarray,
    pattern,
    clip_ratio: Tuple[float, float] = (0.90, 1.10),
) -> Tuple[np.ndarray, float]:
    """Equalise Gr and Gb green sub-lattices on a Bayer mosaic.

    Real sensors can exhibit a small gain mismatch between the two green
    phases (Gr on red rows vs Gb on blue rows), which manifests as a
    high-frequency checkerboard residual in CFA-domain fusion.  This helper
    estimates ``ratio = median(G1) / median(G2)`` over positive sites,
    clamps it to ``clip_ratio``, and scales the second green phase to match
    the first.  Returns ``(balanced_mosaic, applied_ratio)``.
    """
    pat = pattern_array(pattern)
    m = np.asarray(mosaic, dtype=np.float32).copy()
    g_pos = [(r, c) for r in range(2) for c in range(2) if int(pat[r, c]) in (1, 3)]
    if len(g_pos) != 2:
        return m, 1.0
    (r1, c1), (r2, c2) = g_pos
    g1 = m[r1::2, c1::2]
    g2 = m[r2::2, c2::2]
    med1 = float(np.median(g1[g1 > 1e-6])) if np.any(g1 > 1e-6) else 0.0
    med2 = float(np.median(g2[g2 > 1e-6])) if np.any(g2 > 1e-6) else 0.0
    if med1 <= 1e-6 or med2 <= 1e-6:
        return m, 1.0
    ratio = float(np.clip(med1 / med2, clip_ratio[0], clip_ratio[1]))
    if abs(ratio - 1.0) > 1e-4:
        m[r2::2, c2::2] = np.clip(g2 * np.float32(ratio), 0.0, 1.0)
    return m, ratio


def cfa_kernel_regression(
    mosaics: Sequence[np.ndarray],
    aligns: Sequence,
    ratio: float,
    pattern,
    ref_green: np.ndarray,
    weights: Optional[Sequence[float]] = None,
    wb: Optional[Sequence[float]] = None,
    fallback_hr: Optional[np.ndarray] = None,
    robust_maps: Optional[Sequence[Optional[np.ndarray]]] = None,
    noise_sigma: Optional[float] = None,
    k_detail: float = 0.28,
    k_denoise_min: float = 0.55,
    k_denoise_max: float = 0.85,
    k_stretch: float = 2.8,
    chroma_scale: float = 1.25,
    tile_rows: int = 256,
    adaptive: bool = False,
    gr_gb_balance: bool = False,
    reg_sigma_lr: Optional[float] = None,
) -> CFADrizzleResult:
    """Direct Bayer CFA fusion via Wronski-style anisotropic kernel regression.

    Evaluates an edge-directed anisotropic 2D Gaussian kernel over a 3x3 LR
    Bayer neighborhood around each back-projected HR pixel, tiled by rows to
    keep peak memory low.

    Args:
        mosaics: sequence of ``(H, W)`` float32 calibrated Bayer mosaics.
        aligns: per-frame ref->mov transforms (3x3 homography or ``(dx, dy)``
            in LR pixels, same convention as :func:`superres.drizzle`).
        ratio: HR/LR scale factor.
        pattern: 2x2 Bayer colour-index array or pattern string.
        ref_green: ``(H, W)`` reference green plane for the structure tensor.
        weights: per-frame scalar fusion weights (default uniform).
        wb: optional per-channel WB gains (default 1, 1, 1).
        fallback_hr: optional ``(H*r, W*r, 3)`` smooth HR fallback for low-
            coverage hole blending (if None, multiscale Gaussian fill is used).
        robust_maps: optional per-frame ``(H, W)`` float32 robustness weight
            maps in [0, 1] (LR coordinates) for motion/ghost rejection.
        noise_sigma: optional noise std override; estimated from ``ref_green``
            if None.
        k_detail: across-edge kernel std in LR px at strong edges.
        k_denoise_min / k_denoise_max: flat-region kernel std bounds in LR px.
        k_stretch: max along-edge kernel elongation factor.
        chroma_scale: kernel width multiplier for R/B relative to G (>1 reduces
            colour moire on sparse Bayer lattices).
        tile_rows: number of HR rows processed per tile (bounds peak RAM).
    """
    pat = pattern_array(pattern)
    n = len(mosaics)
    if n == 0:
        raise ValueError("cfa_kernel_regression needs at least one frame")
    if len(aligns) != n:
        raise ValueError(f"{n} mosaics but {len(aligns)} transforms")
    if ratio <= 0:
        raise ValueError("ratio must be > 0")

    h, w = np.asarray(mosaics[0]).shape[:2]
    oh, ow = int(round(h * ratio)), int(round(w * ratio))

    if weights is None:
        weights = [1.0] * n
    if wb is None:
        wb_arr = np.ones(3, dtype=np.float32)
    else:
        wb_arr = np.asarray(wb, dtype=np.float32)

    # 2x2 channel LUT: maps (y & 1, x & 1) -> 0 (R), 1 (G), 2 (B)
    ch_lut = np.empty((2, 2), dtype=np.int8)
    for r in range(2):
        for c in range(2):
            ci = int(pat[r, c])
            ch_lut[r, c] = 1 if ci in (1, 3) else (0 if ci == 0 else 2)

    # Opt-in adaptive mode (WP9): tighter texture-preserving flat-region
    # bandwidth + optional registration-uncertainty broadening + Gr/Gb balance.
    if adaptive:
        if k_detail == 0.28:
            k_detail = 0.18
        if k_denoise_min == 0.55:
            k_denoise_min = 0.28
        if k_denoise_max == 0.85:
            k_denoise_max = 0.45
        if k_stretch == 2.8:
            k_stretch = 2.2
        if chroma_scale == 1.25:
            chroma_scale = 1.15
    if reg_sigma_lr is not None and float(reg_sigma_lr) > 0.0:
        k_detail = float(np.hypot(k_detail, 0.5 * float(reg_sigma_lr)))

    # Compute half-res precision matrix from reference green
    a_half, b_half, c_half, st_info = compute_structure_tensor_precision(
        ref_green,
        noise_sigma=noise_sigma,
        k_detail=k_detail,
        k_denoise_min=k_denoise_min,
        k_denoise_max=k_denoise_max,
        k_stretch=k_stretch,
    )

    # Convert aligns to 3x3 ref->mov homographies
    W_mats = []
    for al in aligns:
        if isinstance(al, (tuple, list)) and len(al) == 2:
            W = np.array([[1.0, 0.0, float(al[0])],
                          [0.0, 1.0, float(al[1])],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
        else:
            W = np.asarray(al, dtype=np.float64)
            if W.shape == (2, 3):
                W = np.vstack([W, [0.0, 0.0, 1.0]])
            if not np.all(np.isfinite(W)) or abs(W[2, 2]) < 1e-12:
                W = np.eye(3, dtype=np.float64)
            else:
                W = W / W[2, 2]
        W_mats.append(W)

    gr_gb_ratios = []
    if gr_gb_balance:
        mosaics_f32 = []
        for m in mosaics:
            mb, r_gb = balance_gr_gb(m, pat)
            mosaics_f32.append(np.ascontiguousarray(mb, dtype=np.float32))
            gr_gb_ratios.append(round(float(r_gb), 5))
    else:
        mosaics_f32 = [np.ascontiguousarray(m, dtype=np.float32) for m in mosaics]
    rb_prec_scale = np.float32(1.0 / max(float(chroma_scale) ** 2, 1e-4))
    inv_ratio = np.float32(1.0 / float(ratio))
    off = np.float32(0.5 * (float(ratio) - 1.0))

    hr = np.zeros((oh, ow, 3), dtype=np.float32)
    cov_out = np.zeros((oh, ow), dtype=np.float32)
    min_wt = np.float32(1e-4)

    # Precompute HR x-coordinates in reference LR space
    u_vec = (np.arange(ow, dtype=np.float32) - off) * inv_ratio
    sh, sw = a_half.shape[:2]

    # 3x3 LR neighborhood offsets around nearest integer LR pixel
    offsets = ((-1, -1), (0, -1), (1, -1),
               (-1,  0), (0,  0), (1,  0),
               (-1,  1), (0,  1), (1,  1))

    # Map HR row range to half-res structure tensor coordinates
    for y0 in range(0, oh, tile_rows):
        y1 = min(oh, y0 + tile_rows)
        th = y1 - y0
        v_vec = (np.arange(y0, y1, dtype=np.float32) - off) * inv_ratio

        # Upsample precision matrix (a, b, c) for this tile using cv2.remap
        # Half-res pixel i corresponds to LR pixel (2*i + 0.5)
        map_x = np.broadcast_to(
            ((u_vec + np.float32(0.5)) * np.float32(sw / float(w)) - np.float32(0.5))[None, :],
            (th, ow),
        ).astype(np.float32)
        map_y = np.broadcast_to(
            ((v_vec + np.float32(0.5)) * np.float32(sh / float(h)) - np.float32(0.5))[:, None],
            (th, ow),
        ).astype(np.float32)

        ta = cv2.remap(a_half, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        tb = cv2.remap(b_half, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        tc = cv2.remap(c_half, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        acc0 = np.zeros((th, ow), dtype=np.float32)
        acc1 = np.zeros((th, ow), dtype=np.float32)
        acc2 = np.zeros((th, ow), dtype=np.float32)
        wt0 = np.zeros((th, ow), dtype=np.float32)
        wt1 = np.zeros((th, ow), dtype=np.float32)
        wt2 = np.zeros((th, ow), dtype=np.float32)

        for k in range(n):
            fw = np.float32(weights[k])
            if fw <= 0.0:
                continue
            W = W_mats[k]
            mos = mosaics_f32[k]
            rmap = None if robust_maps is None else robust_maps[k]

            # Project tile HR pixels -> frame k LR coordinates
            w00, w01, w02 = np.float32(W[0, 0]), np.float32(W[0, 1]), np.float32(W[0, 2])
            w10, w11, w12 = np.float32(W[1, 0]), np.float32(W[1, 1]), np.float32(W[1, 2])
            w20, w21 = np.float32(W[2, 0]), np.float32(W[2, 1])

            if abs(float(w20)) < 1e-10 and abs(float(w21)) < 1e-10:
                xk = w00 * u_vec[None, :] + (w01 * v_vec[:, None] + w02)
                yk = w10 * u_vec[None, :] + (w11 * v_vec[:, None] + w12)
            else:
                den = w20 * u_vec[None, :] + (w21 * v_vec[:, None] + np.float32(1.0))
                inv_den = np.float32(1.0) / np.where(np.abs(den) > 1e-8, den, np.float32(1.0))
                xk = (w00 * u_vec[None, :] + (w01 * v_vec[:, None] + w02)) * inv_den
                yk = (w10 * u_vec[None, :] + (w11 * v_vec[:, None] + w12)) * inv_den

            rx0 = np.rint(xk).astype(np.int32)
            ry0 = np.rint(yk).astype(np.int32)

            for dx, dy in offsets:
                ix = rx0 + dx
                iy = ry0 + dy
                ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
                ix_c = np.clip(ix, 0, w - 1)
                iy_c = np.clip(iy, 0, h - 1)

                ddx = ix_c.astype(np.float32) - xk
                ddy = iy_c.astype(np.float32) - yk

                # Quadratic form: -0.5 * (a*dx^2 + 2*b*dx*dy + c*dy^2)
                q = np.float32(-0.5) * (ta * ddx * ddx + np.float32(2.0) * tb * ddx * ddy + tc * ddy * ddy)

                ch = ch_lut[iy_c & 1, ix_c & 1]
                # Slightly wider kernel for R/B channels
                is_g = (ch == 1)
                q_eff = np.where(is_g, q, q * rb_prec_scale)
                kw = np.exp(np.maximum(q_eff, np.float32(-18.0))) * fw
                if rmap is not None:
                    kw *= rmap[iy_c, ix_c]
                kw *= ok.astype(np.float32)

                val = mos[iy_c, ix_c]

                m0 = (ch == 0)
                w0 = kw * m0
                wt0 += w0
                acc0 += w0 * (val * wb_arr[0])

                w1 = kw * is_g
                wt1 += w1
                acc1 += w1 * (val * wb_arr[1])

                m2 = (ch == 2)
                w2 = kw * m2
                wt2 += w2
                acc2 += w2 * (val * wb_arr[2])

        # Normalise and blend any low-coverage pixels with fallback
        for c_idx, (acc_c, wt_c) in enumerate(((acc0, wt0), (acc1, wt1), (acc2, wt2))):
            est_c = acc_c / np.maximum(wt_c, np.float32(1e-8))
            if fallback_hr is not None:
                fb_c = fallback_hr[y0:y1, :, c_idx]
                alpha = np.clip(wt_c / min_wt, np.float32(0.0), np.float32(1.0))
                hr[y0:y1, :, c_idx] = alpha * est_c + (np.float32(1.0) - alpha) * fb_c
            else:
                if (wt_c <= min_wt).any():
                    filled_c, _ = _fill_multiscale(est_c, wt_c, max_sigma=max(4.0, 2.0 * ratio))
                    alpha = np.clip(wt_c / min_wt, np.float32(0.0), np.float32(1.0))
                    hr[y0:y1, :, c_idx] = alpha * est_c + (np.float32(1.0) - alpha) * filled_c
                else:
                    hr[y0:y1, :, c_idx] = est_c
            cov_out[y0:y1, :] += (wt_c > min_wt).astype(np.float32) / np.float32(3.0)

    np.clip(hr, 0.0, 1.0, out=hr)
    metrics = {
        "method": "cfa_kr_adaptive" if adaptive else "cfa_kr",
        "adaptive": bool(adaptive),
        "gr_gb_balance": bool(gr_gb_balance),
        "gr_gb_ratios": gr_gb_ratios if gr_gb_balance else None,
        "ratio": float(ratio),
        "frames": int(n),
        "pattern": [[int(v) for v in row] for row in pat],
        "chroma_scale": float(chroma_scale),
        "coverage_fraction": float((cov_out > 0.99).mean()),
        **st_info,
    }
    return CFADrizzleResult(hr=hr, coverage=cov_out,
                            channel_coverage=None, metrics=metrics)


__all__ = [
    "CFAFrame", "CFADrizzleResult", "pattern_array", "channel_map",
    "load_cfa_frame", "calibrate_cfa_levels", "apply_level_gains",
    "cfa_drizzle", "balance_gr_gb", "compute_structure_tensor_precision",
    "cfa_kernel_regression",
]
