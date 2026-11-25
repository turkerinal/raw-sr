"""Held-out metric in **measurement space** (Bayer CFA) for multi-frame SR.

Why a second held-out metric
----------------------------
The honest held-out metric in ``scripts/sr_acceptance.py`` (B4) predicts a frame
that was not fused and compares it with a denoised LR stack of the fused
frames.  Both sides live in **DCB-demosaiced RGB space**, because that is what
``load_raw_linear`` returns.  That space has a fixed cost for anything that does
not reproduce DCB's own interpolation and noise smoothing: on the real burst a
demosaic-free CFA-domain reconstruction scores ~-3 dB there while being *more*
accurate against ground truth, and ~17 % of the metric's floor is DCB-vs-AHD/VNG
ambiguity alone (``internal audit report`` §2.2-2.4).  The metric therefore
cannot rank deposition strategies.

This module scores the same held-out question where it is defined by the
sensor: at the **held-out frame's own Bayer sites**.

* **Prediction** ``(a)``: an HR RGB estimate is pushed through the forward model
  (sensor-support PSF, then sampling at ``(r-1)/2 + ratio * W^-1 p``) and read at the
  channel of each Bayer site: this is the re-mosaic of the estimate.
* **Baseline** ``(b)``: the fused frames' own Bayer samples are aligned into the
  held-out frame's LR grid and averaged **per channel**, coverage-normalised —
  a CFA-domain denoised LR stack, with no demosaic anywhere.
* Both sides are scored on the same sites: the interior of the held-out frame
  (same 24 LR px margin idea as B4), sites the baseline filled from a
  sufficient share of the fused weight, and sites whose prediction only reads
  HR pixels that are inside the estimate (the same geometric support rule
  ``superres.lr_sampling_valid_mask`` applies, but per site and per channel).

The held-out frame is excluded from the SR estimate by the caller *and* from the
baseline here (``mosaics`` must be the frames that were fused), so the
comparison cannot leak.

The gain is in dB of PSNR over that baseline, pooled over all measured sites
(so green counts twice, as it does on the sensor) and also reported per
channel.  It is a *measurement-space* number: never compare it with the
DCB-space ``heldout_gain_db`` from B4, and never mix the two in one table.

What it does not remove
-----------------------
The baseline is a weighted mean of sub-pixel-shifted measurements, so it is
slightly softer than one sensor PSF; the prediction is band-limited by that
PSF.  Band-limiting the baseline too changes the gain by ~0.001 dB (B4 §5), so
the default leaves measured data unblurred; ``baseline_psf=True`` applies the
identical forward model to the baseline for an A/B check.

At high ratio the held-out frame's own sampling is sparse, so the metric still
measures "can the estimate predict the samples that were actually taken", not
"is the HR image closest to the scene".  Ground-truth synthetic PSNR remains the
accuracy reference; this metric is the truth-free proof and the method ranking
instrument.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .cfa_drizzle import _deposit_plane, channel_map, pattern_array
from .superres import SENSOR_PSF_SIGMA_PER_RATIO

# Interior margin in LR pixels, measured from the held-out frame's border.  The
# value matches B4's ``HELDOUT_INTERIOR_MARGIN_LR`` so a CFA-space run and a
# DCB-space run describe the same region; it is 1.7x the worst measured
# inter-frame shift on the real burst (18.6 LR px at ROI 512) and >3x the
# sensor-PSF support in LR px.
CFA_HELDOUT_MARGIN_LR = 24

# Share of the *channel's attainable* fused weight a baseline site must receive.
# A Bayer lattice has gaps: only 1/4 of the sites carry R (or B) and 1/2 carry
# green, so a perfectly aligned site can only ever collect that share of the
# fused weight.  The requirement is therefore scaled by the channel's site
# density in the 2x2 block (``density_c``): 0.5 means "this site is a genuine
# multi-frame average that collected at least half of the weight its channel
# could possibly receive", not "every frame reached it" (which no R/B site can
# satisfy).  Applied identically to every method.
CFA_HELDOUT_MIN_COVERAGE_FRACTION = 0.5


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _as_h(W) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    if W.shape == (2, 2):
        H = np.eye(3, dtype=np.float64)
        H[:2, :2] = W
        return H
    if W.shape == (3, 3):
        return W / W[2, 2] if abs(W[2, 2]) > 1e-12 else W
    if W.size == 2:                      # (dx, dy) translation
        H = np.eye(3, dtype=np.float64)
        H[0, 2], H[1, 2] = float(W[0]), float(W[1])
        return H
    raise ValueError(f"unsupported transform of shape {W.shape}")


def frame_to_held_matrix(ref_to_frame, ref_to_held) -> np.ndarray:
    """Matrix (cv2 ``src->dst`` convention) mapping frame k onto the held grid.

    ``ref_to_frame`` / ``ref_to_held`` are ref->frame transforms in LR pixels,
    the convention of :func:`superres.estimate_homography`.  For a held pixel
    ``x_h`` we need frame ``k`` at
    ``x_k = W_k W_h^-1 x_h``, and ``cv2.warpPerspective(src, M)`` samples
    ``src`` at ``M^-1 x``, so ``M = W_h W_k^-1``.
    """
    Wh = _as_h(ref_to_held)
    Wk = _as_h(ref_to_frame)
    M = Wh @ np.linalg.inv(Wk)
    return (M / M[2, 2]).astype(np.float64)


def site_support(ref_to_held, ratio, lr_shape, forward_sigma: Optional[float],
                 hr_shape: Tuple[int, int]) -> np.ndarray:
    """Sites whose forward-model read stays inside the HR estimate.

    Same rule as :func:`superres.lr_sampling_valid_mask` (bilinear kernel plus
    the 3-sigma PSF support), evaluated per LR pixel; the caller AND-s it with
    the site's channel mask.
    """
    h, w = int(lr_shape[0]), int(lr_shape[1])
    hh, wh = int(hr_shape[0]), int(hr_shape[1])
    off = 0.5 * (float(ratio) - 1.0)
    A = np.diag([float(ratio), float(ratio), 1.0]) @ np.linalg.inv(
        _as_h(ref_to_held))
    A[0, 2] += off
    A[1, 2] += off
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    q = A @ np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)], axis=0)
    z = q[2]
    ok = np.abs(z) > 1e-12
    z = np.where(ok, z, 1.0)
    qx = (q[0] / z).reshape(h, w)
    qy = (q[1] / z).reshape(h, w)
    pad = 1.0 + (np.ceil(3.0 * float(forward_sigma))
                 if forward_sigma and forward_sigma > 0 else 0.0)
    return (ok.reshape(h, w)
            & (qx >= pad) & (qx <= wh - 1 - pad)
            & (qy >= pad) & (qy <= hh - 1 - pad))


# --------------------------------------------------------------------------- #
# (a) HR estimate -> predicted Bayer samples
# --------------------------------------------------------------------------- #
def predict_cfa_from_hr(hr: np.ndarray, ref_to_held, ratio: float,
                        pattern, forward_sigma: Optional[float] = None,
                        lr_shape: Optional[Tuple[int, int]] = None,
                        colour_matrix: Optional[np.ndarray] = None,
                        lens_sigma_lr: Optional[object] = None,
                        ) -> Tuple[List[np.ndarray], np.ndarray]:
    """Re-mosaic an HR RGB estimate at a frame's Bayer sites.

    The estimate is band-limited by the sensor-support PSF (``forward_sigma``
    HR px, default :data:`superres.SENSOR_PSF_SIGMA_PER_RATIO` * ratio — the
    same forward model the B4 metric and IBP use), optionally combined in
    quadrature with a per-burst optical lens PSF ``lens_sigma_lr`` (scalar or
    3-sequence of R, G, B sigmas in LR px), then sampled at
    ``(r-1)/2 + ratio * W^-1 p`` for every site ``p`` of the held-out frame.

    ``lr_shape`` is the held-out frame's ``(h, w)``.  ``None`` derives it as
    ``hr.shape[:2] / ratio`` and requires an exact integer ratio; pass it
    explicitly when ``hr`` is ``round(h*ratio)`` (which the core produces for
    a non-integer ``h*ratio``).

    Returns ``(planes, support)``: a list of three float32 ``(h, w)`` planes
    holding the predicted value at every site of that channel (``nan``
    elsewhere) and the geometric-support mask from :func:`site_support`.
    """
    if forward_sigma is None:
        forward_sigma = SENSOR_PSF_SIGMA_PER_RATIO * float(ratio)
    hr = np.asarray(hr, dtype=np.float32)
    if hr.ndim != 3 or hr.shape[2] != 3:
        raise ValueError("hr must be (H, W, 3)")
    if lr_shape is None:
        r = int(round(ratio))
        if abs(float(ratio) - r) > 1e-9 or hr.shape[0] % r or hr.shape[1] % r:
            raise ValueError("pass lr_shape for a non-integer HR/LR ratio")
        h, w = hr.shape[0] // r, hr.shape[1] // r
    else:
        h, w = int(lr_shape[0]), int(lr_shape[1])
    pat = pattern_array(pattern)
    ch = channel_map(pat, (h, w))

    src = hr
    support_sigma = float(forward_sigma or 0.0)
    if lens_sigma_lr is not None:
        if isinstance(lens_sigma_lr, (int, float, np.floating, np.integer)):
            lens_rgb = (float(lens_sigma_lr),) * 3
        else:
            lens_rgb = tuple(float(v) for v in lens_sigma_lr)
            if len(lens_rgb) != 3:
                raise ValueError(f"lens_sigma_lr must be scalar or length-3, got {len(lens_rgb)}")
        base_sig = float(forward_sigma or 0.0)
        sig_hr_rgb = [float(np.hypot(base_sig, max(s, 0.0) * float(ratio))) for s in lens_rgb]
        support_sigma = max(sig_hr_rgb)
        blurred_ch = []
        for c, sc in enumerate(sig_hr_rgb):
            if sc > 0.0:
                kc = 2 * int(np.ceil(3.0 * sc)) + 1
                blurred_ch.append(cv2.GaussianBlur(
                    src[:, :, c], (kc, kc), sc, borderType=cv2.BORDER_REPLICATE
                ))
            else:
                blurred_ch.append(src[:, :, c])
        src = np.stack(blurred_ch, axis=2)
        if colour_matrix is not None:
            C = np.asarray(colour_matrix, dtype=np.float32)
            if C.shape != (3, 3):
                raise ValueError(f"colour_matrix must be (3, 3), got {C.shape}")
            src = np.matmul(src, C.T)
    else:
        if colour_matrix is not None:
            C = np.asarray(colour_matrix, dtype=np.float32)
            if C.shape != (3, 3):
                raise ValueError(f"colour_matrix must be (3, 3), got {C.shape}")
            src = np.matmul(src, C.T)
        if forward_sigma and forward_sigma > 0:
            k = int(np.ceil(3.0 * float(forward_sigma))) | 1
            src = cv2.GaussianBlur(src, (k, k), float(forward_sigma),
                                   borderType=cv2.BORDER_REPLICATE)

    off = 0.5 * (float(ratio) - 1.0)
    A = (np.diag([float(ratio), float(ratio), 1.0])
         @ np.linalg.inv(_as_h(ref_to_held))).astype(np.float64)
    # Pixel-centre aligned HR grid (same convention as to_drizzle_matrix).
    A[0, 2] += off
    A[1, 2] += off
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    q = A @ np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)], axis=0)
    z = np.where(np.abs(q[2]) > 1e-12, q[2], 1.0)
    mx = (q[0] / z).reshape(h, w).astype(np.float32)
    my = (q[1] / z).reshape(h, w).astype(np.float32)

    planes: List[np.ndarray] = []
    for c in range(3):
        # remap once per channel on the full grid (cheap at LR size) and keep
        # only this channel's sites.
        vals = cv2.remap(src[:, :, c], mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        plane = np.full((h, w), np.nan, dtype=np.float32)
        m = ch == c
        plane[m] = vals[m]
        planes.append(plane)
    support = site_support(ref_to_held, ratio, (h, w), support_sigma,
                           hr.shape[:2])
    return planes, support


# --------------------------------------------------------------------------- #
# (b) CFA-domain denoised LR stack on the held-out grid
# --------------------------------------------------------------------------- #
def cfa_lr_stack_at_held(mosaics: Sequence[np.ndarray],
                         transforms: Sequence,
                         ref_to_held,
                         pattern,
                         weights: Optional[Sequence[float]] = None,
                         pixfrac: float = 1.0,
                         subsamples: int = 2) -> Tuple[List[np.ndarray],
                                                       List[np.ndarray]]:
    """Coverage-normalised per-channel CFA stack on the held-out LR grid.

    Every fused frame's same-channel samples are deposited at their true
    position in the held-out frame's LR grid (matrix from
    :func:`frame_to_held_matrix`) with the drizzle splat, accumulating value
    and weight **per channel**.  No demosaic is applied and no cross-channel
    value is ever used, so the stack is a pure measurement average.

    ``pixfrac``/``subsamples`` default to a 1 LR px footprint with 2 sub-samples
    per axis: a Bayer lattice has gaps between same-channel sites and a purely
    bilinear gather would leave holes wherever a frame's parity differs, while
    this footprint fills them and stays as compact as the bilinear warp the
    DCB-space baseline uses.

    Returns ``(stack, coverage)``, each a list of three ``(h, w)`` float32
    planes; ``stack[c] = acc[c] / max(cov[c], eps)``.
    """
    n = len(mosaics)
    if n == 0:
        raise ValueError("cfa_lr_stack_at_held needs at least one frame")
    if len(transforms) != n:
        raise ValueError(f"{n} mosaics but {len(transforms)} transforms")
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(f"{n} mosaics but {len(weights)} weights")
    h, w = np.asarray(mosaics[0]).shape[:2]
    pat = pattern_array(pattern)
    ch = channel_map(pat, (h, w))

    acc = [np.zeros((h, w), np.float64) for _ in range(3)]
    cov = [np.zeros((h, w), np.float64) for _ in range(3)]
    for k in range(n):
        M = frame_to_held_matrix(transforms[k], ref_to_held)
        for c in range(3):
            m = ch == c
            ys, xs = np.nonzero(m)
            vals = np.asarray(mosaics[k], dtype=np.float64)[m]
            a, v = _deposit_plane(vals, xs, ys, M, (h, w), float(pixfrac),
                                  int(subsamples), fweight=float(weights[k]))
            acc[c] += a
            cov[c] += v
    stack = [(acc[c] / np.maximum(cov[c], 1e-12)).astype(np.float32)
             for c in range(3)]
    return stack, cov


# --------------------------------------------------------------------------- #
# Mask + comparison
# --------------------------------------------------------------------------- #
def cfa_heldout_masks(pattern, shape: Tuple[int, int], support: np.ndarray,
                      coverage: Sequence[np.ndarray], total_weight: float,
                      margin_lr: int = CFA_HELDOUT_MARGIN_LR,
                      min_coverage_fraction: float =
                      CFA_HELDOUT_MIN_COVERAGE_FRACTION
                      ) -> List[np.ndarray]:
    """Per-channel site masks: interior AND supported AND baseline-covered.

    * interior: ``margin_lr`` LR px from every border of the held-out frame;
    * supported: the forward-model read is inside the HR estimate
      (:func:`site_support`);
    * covered: the baseline received at least
      ``min_coverage_fraction * total_weight * density_c`` at that site, where
      ``density_c`` is the share of the 2x2 block that carries channel ``c``
      (1/2 green, 1/4 red/blue).  Scaling by density is what makes the rule
      meaningful for a CFA: an R site can only ever collect a quarter of the
      fused weight, because only a quarter of every frame's sites are red.
      This keeps the comparison on sites the baseline actually averaged over
      several frames instead of one.
    """
    h, w = int(shape[0]), int(shape[1])
    ch = channel_map(pattern, (h, w))
    inner = np.zeros((h, w), dtype=bool)
    m = int(margin_lr)
    if m > 0:
        if h > 2 * m and w > 2 * m:
            inner[m:h - m, m:w - m] = True
    else:
        inner[:] = True
    # Per-channel site density in the 2x2 block: 1/2 for green, 1/4 for R/B.
    density = np.array([np.mean(ch == c) for c in range(3)], dtype=np.float64)
    return [inner & support & (ch == c)
            & (np.asarray(coverage[c])
               >= float(total_weight) * float(min_coverage_fraction)
               * float(density[c]))
            for c in range(3)]


def _psnr_from_pairs(pred: np.ndarray, truth: np.ndarray,
                     masks: Sequence[np.ndarray]) -> Tuple[float, dict]:
    """Pooled measurement-space PSNR plus per-channel PSNRs and site counts."""
    counts = [int(np.count_nonzero(m)) for m in masks]
    if int(np.sum(counts)) == 0:
        raise ValueError("CFA held-out metric: no valid sites")
    sse = 0.0
    n_valid = 0
    per_channel = {}
    for c in range(3):
        m = masks[c]
        if not counts[c]:
            per_channel[c] = None
            continue
        d = pred[c][m].astype(np.float64) - truth[m].astype(np.float64)
        d = d[np.isfinite(d)]
        if d.size == 0:
            per_channel[c] = None
            continue
        s = float(np.square(d).sum())
        sse += s
        n_valid += int(d.size)
        per_channel[c] = 10.0 * np.log10(d.size / max(s, 1e-30))
    if n_valid == 0:
        raise ValueError("CFA held-out metric: no finite sites")
    return 10.0 * np.log10(n_valid / max(sse, 1e-30)), {
        "sites_per_channel": counts,
        "finite_sites": n_valid,
        "psnr_db_per_channel": per_channel,
    }


def cfa_heldout_compare(pred_planes: Sequence[np.ndarray],
                        base_planes: Sequence[np.ndarray],
                        truth_mosaic: np.ndarray,
                        masks: Sequence[np.ndarray],
                        label: str = "") -> Dict:
    """Score a prediction and the baseline on identical Bayer sites.

    ``pred_planes`` / ``base_planes`` are per-channel ``(h, w)`` planes holding
    the value under test at that channel's sites (``nan`` elsewhere);
    ``truth_mosaic`` the held-out frame's measured mosaic; ``masks`` the
    per-channel site masks from :func:`cfa_heldout_masks`.

    Returns the pooled PSNRs, the gain (prediction - baseline), the per-channel
    numbers and the site accounting.  Both sides see exactly the same pixels,
    so the gain isolates the reconstruction.
    """
    psnr_p, extra_p = _psnr_from_pairs(pred_planes, truth_mosaic, masks)
    psnr_b, extra_b = _psnr_from_pairs(base_planes, truth_mosaic, masks)
    return {
        "method": label,
        "heldout_cfa_psnr_db": psnr_p,
        "heldout_cfa_baseline_psnr_db": psnr_b,
        "heldout_cfa_gain_db": psnr_p - psnr_b,
        "heldout_cfa_sites": int(np.sum(extra_p["sites_per_channel"])),
        "heldout_cfa_sites_per_channel": extra_p["sites_per_channel"],
        "heldout_cfa_psnr_db_per_channel": extra_p["psnr_db_per_channel"],
        "heldout_cfa_baseline_psnr_db_per_channel":
            extra_b["psnr_db_per_channel"],
    }


# --------------------------------------------------------------------------- #
# One-call evaluation
# --------------------------------------------------------------------------- #
def evaluate_cfa_heldout(hr_by_method: Dict[str, np.ndarray],
                         held_mosaic: np.ndarray,
                         mosaics_fused: Sequence[np.ndarray],
                         transforms: Sequence,
                         weights: Sequence[float],
                         ref_to_held,
                         ratio: float,
                         pattern,
                         margin_lr: int = CFA_HELDOUT_MARGIN_LR,
                         forward_sigma: Optional[float] = None,
                         min_coverage_fraction: float =
                         CFA_HELDOUT_MIN_COVERAGE_FRACTION,
                         baseline_psf: bool = False,
                         colour_matrix: Optional[np.ndarray] = None,
                         lens_sigma_lr: Optional[object] = None) -> Dict:
    """Run the full CFA-space held-out comparison for several HR estimates.

    Args:
        hr_by_method: name -> HR RGB estimate ``(ratio*h, ratio*w, 3)``.  Every
            estimate is scored against the same baseline and mask.
        held_mosaic: the held-out frame's calibrated Bayer mosaic.  It must not
            appear in ``mosaics_fused`` (no leakage).
        mosaics_fused: mosaics of the frames that produced the estimates, in
            the order of ``transforms`` / ``weights``.
        transforms: ref->frame transforms for those frames, LR pixels.
        weights: per-frame fusion weights (any positive scale).
        ref_to_held: ref->held transform, LR pixels (a fresh estimate; the
            metric must not grade itself against the run's registration).
        ratio: HR/LR scale of the estimates.
        pattern: 2x2 Bayer pattern or its string form.
        forward_sigma: PSF (HR px) of the prediction's forward model.  ``None``
            = :data:`superres.SENSOR_PSF_SIGMA_PER_RATIO` * ratio.  ``0`` gives
            the un-band-limited variant.
        baseline_psf: apply the identical forward model to the *baseline* too
            (diagnostic; ~0.001 dB on the real burst).
        margin_lr / min_coverage_fraction: see :func:`cfa_heldout_masks`.

    Returns:
        ``{"per_method": {name: compare_dict}, "baseline": {...}, "mask":
        {...}, "config": {...}}``; ``heldout_cfa_gain_db`` per method is the
        number to quote.
    """
    if forward_sigma is None:
        forward_sigma = SENSOR_PSF_SIGMA_PER_RATIO * float(ratio)
    hm = np.asarray(held_mosaic, dtype=np.float32)
    h, w = hm.shape[:2]
    pat = pattern_array(pattern)

    base, cov = cfa_lr_stack_at_held(mosaics_fused, transforms, ref_to_held,
                                     pat, weights)
    if baseline_psf and forward_sigma and forward_sigma > 0:
        # Same forward model as the prediction, applied to the LR stack: the
        # stack is already at LR resolution, so this is only a diagnostic of
        # how much of the gain is the baseline's slightly lower bandwidth.
        sig_lr = float(forward_sigma) / float(ratio)
        k = max(3, int(np.ceil(3.0 * sig_lr)) | 1)
        base = [cv2.GaussianBlur(b, (k, k), sig_lr,
                                 borderType=cv2.BORDER_REPLICATE)
                for b in base]

    out: Dict = {"per_method": {}, "config": {
        "ratio": float(ratio),
        "margin_lr": int(margin_lr),
        "min_coverage_fraction": float(min_coverage_fraction),
        "forward_sigma_hr": float(forward_sigma or 0.0),
        "baseline_psf": bool(baseline_psf),
        "frames_fused": int(len(mosaics_fused)),
        "colour_fair": bool(colour_matrix is not None),
        "lens_sigma_lr": (
            [float(v) for v in lens_sigma_lr]
            if isinstance(lens_sigma_lr, (tuple, list))
            else (float(lens_sigma_lr) if lens_sigma_lr is not None else None)
        ),
    }}
    if not hr_by_method:
        raise ValueError("no HR estimates to score")
    # The mask only needs one prediction's support; every method shares the
    # same geometry, so it is computed once from the first estimate.
    masks = None
    for name, hr in hr_by_method.items():
        planes, support = predict_cfa_from_hr(
            hr, ref_to_held, ratio, pat, forward_sigma=forward_sigma,
            lr_shape=(h, w), colour_matrix=colour_matrix,
            lens_sigma_lr=lens_sigma_lr)
        if masks is None:
            masks = cfa_heldout_masks(
                pat, (h, w), support, cov, float(np.sum(weights)),
                margin_lr=margin_lr,
                min_coverage_fraction=min_coverage_fraction)
            out["mask"] = {
                "sites": int(sum(int(np.count_nonzero(mm)) for mm in masks)),
                "sites_per_channel": [int(np.count_nonzero(mm))
                                      for mm in masks],
                "fraction_of_frame": float(
                    sum(int(np.count_nonzero(mm)) for mm in masks)
                    / float(h * w)),
                "baseline_coverage_fraction": [
                    float((np.asarray(cov[c]) >= float(np.sum(weights))
                           * float(min_coverage_fraction)
                           * float(np.mean(channel_map(pat, (h, w)) == c))).mean())
                    for c in range(3)],
            }
        out["per_method"][name] = cfa_heldout_compare(
            planes, base, hm, masks, label=name)
    out["baseline"] = {
        "psnr_db": out["per_method"][next(iter(out["per_method"]))][
            "heldout_cfa_baseline_psnr_db"],
        "mask": out["mask"],
        "sites": out["mask"]["sites"],
    }
    return out


__all__ = [
    "CFA_HELDOUT_MARGIN_LR", "CFA_HELDOUT_MIN_COVERAGE_FRACTION",
    "frame_to_held_matrix", "site_support", "predict_cfa_from_hr",
    "cfa_lr_stack_at_held", "cfa_heldout_masks", "cfa_heldout_compare",
    "evaluate_cfa_heldout",
]
