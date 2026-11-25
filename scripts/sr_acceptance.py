#!/usr/bin/env python3
"""Real-data acceptance for the multi-frame SR core.

Loads the 9-frame DNG burst, runs genuine multi-frame SR at a requested ratio,
writes:
  - output/sr_master_<ratio>x.tiff   (16-bit LINEAR master, the measured data)
  - output/sr_<ratio>x_before.png     (naive single-frame bicubic upscale)
  - output/sr_<ratio>x_after.png      (SR result)
  - output/sr_acceptance_<ratio>x.json (metrics; --out-suffix adds a tag)

Metrics are honest and truth-free:
  * hf_gain_db   : HF energy above the single-frame Nyquist, measured on the
                   display-transformed view (linear dark RAW puts pure sensor
                   noise in that band). Positive => extra real high frequency
                   versus the bicubic-upsampled denoised stack.
  * hf_gain_db_vs_single: same band versus a bicubic-upsampled *single* frame.
  * sharpness_gain: Laplacian-variance ratio (SR / naive single frame).
  * held-out PSNR: predict a frame NOT used in fusion, compare against its real
                   samples. SR must beat the denoised LR stack of the SAME
                   frames (equal denoising power). Both are warped into the
                   held-out frame starting from the run's own reference frame
                   (``SRResult.reference_index`` — automatic reference selection
                   is not frame 0), and the SR prediction goes through the
                   sensor's forward model (its per-LR-pixel integration PSF),
                   because an HR estimate is compared against a *measured* LR
                   sample. The baseline is a per-pixel weighted mean
                   (coverage-normalised) and both sides are scored on a common
                   valid mask (prediction support, a 24 LR px interior margin,
                   full baseline coverage); the whole-frame number is kept only
                   as a ``*_legacy`` field. This is the strong,
                   ground-truth-free SR proof.

Usage:
  venv/bin/python scripts/sr_acceptance.py [--ratio 2] [--heldout 8]
  venv/bin/python scripts/sr_acceptance.py --heldout-sweep   # every frame
  venv/bin/python scripts/sr_acceptance.py --out-suffix _B1heldout --out-dir output
  venv/bin/python scripts/sr_acceptance.py --heldout-psf 0.4   # sigma sensitivity

  The held-out prediction's forward-model PSF defaults to the sensor's per-LR-
  pixel integration window (sigma = 0.289*ratio HR px, 1/sqrt(12)); it is not a
  tuned parameter. --heldout-psf only exists to reproduce the sensitivity table
  in docs/IBP_AND_HELDOUT.md.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The CFA-domain prototype and its measurement-space metric are EXPERIMENTAL
# and imported lazily, only under --cfa-heldout, so a default acceptance run
# never touches them (docs/CFA_DRIZZLE.md).  The literal defaults below mirror
# ``cfa_heldout.CFA_HELDOUT_MARGIN_LR`` / ``..._MIN_COVERAGE_FRACTION``; the
# function asserts they still agree.
_CFA_HELDOUT_MARGIN_LR = 24
_CFA_HELDOUT_MIN_COVERAGE_FRACTION = 0.5
from raw_sr.core.superres import (  # noqa: E402
    SENSOR_PSF_SIGMA_PER_RATIO,
    load_raw_linear,
    SuperResolution,
    estimate_homography,
    denoised_lr_stack,
    predict_lr_from_hr,
    lr_sampling_valid_mask,
    high_frequency_gain_db,
    laplacian_sharpness,
    to_uint16_linear,
    to_uint8_view,
    to_view_float,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acceptance_verdict  # noqa: E402  (explicit PASS/FAIL/N-A verdict)


def find_burst():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [
        os.environ.get("RAW_SR_TEST_DATA"),
        os.path.join(here, "test_data"),
    ]
    for c in cands:
        if not c:
            continue
        files = sorted(glob.glob(os.path.join(c, "*.dng")))
        if len(files) >= 2:
            return c, files
    raise SystemExit(
        "no DNG burst found (set RAW_SR_TEST_DATA or add test_data/*.dng)")


def heldout_baseline_lr(kept_frames, res):
    """Coverage-normalised denoised LR stack of exactly the frames fused.

    ``kept_frames`` is the list handed to ``SuperResolution.run``; ``res`` its
    result. ``res.transforms`` / ``res.weights`` are per *kept* frame, indexed
    by ``res.kept_indices`` (which index ``kept_frames``), so the frames must be
    selected through ``kept_indices`` before zipping. Passing the full list
    (the pre-2026-11 behaviour) silently pairs frames with another frame's
    transform whenever the run drops one (blur threshold or alignment gate),
    which corrupts the baseline without any error.

    The stack is normalised by the weight that covered each pixel, not by the
    total weight (B2's metric divided by the total and produced a dark rim of
    ~40-80 % of the baseline's error wherever a shifted frame ended; see
    docs/IBP_AND_HELDOUT.md §5). The coverage map is returned so the caller
    can restrict both sides of the comparison to fully covered pixels.

    Returns ``(base_lr, coverage, fused_frames)``.
    """
    fused = [kept_frames[i] for i in res.kept_indices]
    if len(fused) != len(res.transforms) or len(fused) != len(res.weights):
        raise ValueError(
            "held-out baseline: %d fused frames but %d transforms / %d weights"
            % (len(fused), len(res.transforms), len(res.weights)))
    base_lr, cov = denoised_lr_stack(fused, res.transforms,
                                     res.weights.tolist(),
                                     coverage_normalised=True,
                                     return_coverage=True)
    return base_lr, cov, fused


# Interior margin of the common valid region, in LR pixels. Justification: the
# largest measured inter-frame displacement on the 9-frame burst at ROI 1024 is
# 13.9 LR px, so 24 px is 1.7x the worst shift — every fused frame fully covers
# the retained area, no baseline pixel is built from fewer frames than the ROI
# centre — and it is >20x the bilinear interpolation kernel and >3x the sensor
# PSF support in LR px. It removes 2*(24/1024) - (24/1024)^2 = 4.6 % of the
# area. ``0`` disables the margin (the documentation's sensitivity knob).
HELDOUT_INTERIOR_MARGIN_LR = 24


def heldout_valid_mask(pred_sr, hr_shape, ref_to_frame, ratio, lr_shape,
                       forward_sigma=None,
                       margin_lr=HELDOUT_INTERIOR_MARGIN_LR):
    """Pixels where the SR prediction AND the baseline are both determined.

    Three conditions, all necessary for a fair per-pixel comparison:

    * the SR prediction's own geometric support — ``lr_sampling_valid_mask``,
      the exact forward model applied by :func:`predict_lr_from_hr`;
    * a fixed interior margin (``margin_lr``; see the constant for the
      justification) that keeps the comparison away from every warped border,
      independent of how far any particular frame moved;
    * finite, non-negative inputs (defensive; a NaN must never average in).
    """
    h, w = int(lr_shape[0]), int(lr_shape[1])
    mask = lr_sampling_valid_mask(hr_shape, ref_to_frame, ratio, lr_shape,
                                  forward_sigma=forward_sigma)
    m = int(margin_lr)
    if m > 0:
        inner = np.zeros((h, w), dtype=bool)
        if h > 2 * m and w > 2 * m:
            inner[m:h - m, m:w - m] = True
        mask &= inner
    mask &= np.isfinite(pred_sr).all(axis=2)
    return mask


def psnr_over_mask(image, truth, mask):
    """PSNR (peak 1.0) of ``image`` against ``truth`` on ``mask`` only."""
    if not np.any(mask):
        raise ValueError("held-out metric: the common valid mask is empty")
    diff = (image - truth) ** 2
    mse = float(diff[mask].mean())
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def heldout_compare(pred_sr, base_lr_held, truth, mask, base_legacy=None):
    """The held-out comparison, computed on a common valid mask.

    Both images are scored on exactly the same pixels, so the gain is not
    inflated by a rim that only one of them reconstructs.

    ``base_legacy`` is the pre-2026-11 baseline (global-weight normalisation);
    when given, the ``*_legacy`` keys reproduce the archived number exactly so
    an old JSON can be compared field by field. They are NOT a measurement of
    resolution recovery and must never be quoted as the headline.
    """
    psnr_sr = psnr_over_mask(pred_sr, truth, mask)
    psnr_base = psnr_over_mask(base_lr_held, truth, mask)
    mse_sr = float(((pred_sr - truth) ** 2)[mask].mean())
    mse_base = float(((base_lr_held - truth) ** 2)[mask].mean())
    if base_legacy is None:
        base_legacy = base_lr_held
    psnr_sr_legacy = float(10 * np.log10(
        1.0 / max(float(np.mean((pred_sr - truth) ** 2)), 1e-12)))
    psnr_base_legacy = float(10 * np.log10(
        1.0 / max(float(np.mean((base_legacy - truth) ** 2)), 1e-12)))
    return {
        "heldout_psnr_db_sr": psnr_sr,
        "heldout_psnr_db_denoised_stack": psnr_base,
        "heldout_gain_db": psnr_sr - psnr_base,
        "heldout_mse_sr": mse_sr,
        "heldout_mse_denoised_stack": mse_base,
        "heldout_valid_pixels": int(mask.sum()),
        "heldout_valid_fraction": float(mask.mean()),
        "heldout_psnr_db_sr_legacy": psnr_sr_legacy,
        "heldout_psnr_db_denoised_stack_legacy": psnr_base_legacy,
        "heldout_gain_db_legacy": psnr_sr_legacy - psnr_base_legacy,
    }


def evaluate_heldout(frames, ratio, registration, ecc_scale, hidx,
                     use_gpu_scatter=None, forward_sigma=None,
                     blur_threshold_ratio=0.0, ibp_iterations=None,
                     min_align_score=None, fusion="demosaic_drizzle"):
    """Truth-free SR proof: reconstruct without `hidx`, predict it.

    Returns a dict with SR and denoised-stack PSNR (both in the held-out
    frame's coordinate system) and their gain in dB.

    The SR image lives on the grid of the *reference frame that the run
    actually chose* (``SRResult.reference_index``; automatic reference
    selection picks the sharpest frame near the burst centroid, so this is
    normally NOT ``frames[0]``). Predicting the held-out frame therefore has
    to start from that frame; using ``frames[0]`` instead throws away the
    registration offset between the two and was the cause of the ~0 dB
    held-out gain reported at 2x/3x/4x on the real burst.

    ``forward_sigma`` is the sensor-support PSF of the prediction's forward
    model (see :func:`predict_lr_from_hr`); ``None`` uses the core's default
    and ``0`` reports the un-band-limited variant.

    The reported PSNR/gain are computed on a **common valid mask** (see
    :func:`heldout_valid_mask`): the SR prediction's own geometric support, a
    fixed 24 LR px interior margin, and pixels the coverage-normalised
    baseline actually filled from every fused frame. Both sides are scored on
    exactly those pixels, so neither can win on a border the other cannot
    reconstruct. The pre-2026-11 whole-frame values are returned as
    ``*_legacy`` fields only; they contain the baseline's unfilled rim and are
    not a measurement of resolution recovery (docs/IBP_AND_HELDOUT.md §5).

    Two properties of this measurement are stated here because they bound what
    the number means:

    * ``W_held`` is a **fresh 6-dof homography fitted on the held-out frame
      itself** (ref->heldout, LR px), not one taken from the SR run. It is
      deliberately not the run's own registration (the metric must not grade
      itself), and it is applied identically to the SR prediction and to the
      baseline, so it cancels in the *gain*. The absolute PSNR is therefore
      fitted with the held-out frame's geometry and is optimistic in absolute
      terms; the gain is the quantity to quote.
    * ``blur_threshold_ratio`` defaults to **0.0**, i.e. the held-out SR run
      keeps every frame, whereas the shipped ``SuperResolution`` default is
      0.6 (blurred frames are dropped). The held-out gain therefore describes
      "all frames fused" and is not bit-comparable with an acceptance run of
      the production configuration when the burst has a blurred frame. Pass a
      value here to measure the shipped gating instead; the baseline follows
      the same kept set either way (:func:`heldout_baseline_lr`).
    """
    kept = [f for i, f in enumerate(frames) if i != hidx]
    hw = frames[hidx]
    # The held-out run must use the SAME reconstruction configuration the caller
    # is measuring; otherwise ``--ibp-iterations`` would move the headline SR
    # image while every held-out number silently stayed at the core default.
    extra = {} if ibp_iterations is None else {"ibp_iterations": ibp_iterations}
    # Frame acceptance is a separate knob from the metric. On bursts whose
    # content is dominated by frequencies above the LR Nyquist, correct
    # registration scores a low ZNCC (aliasing decorrelates neighbouring
    # subpixel phases), so a metric study of that band may want the gate off;
    # ``None`` keeps the core default (0.1).
    if min_align_score is not None:
        extra["min_align_score"] = float(min_align_score)
    sr_ho = SuperResolution(ratio=ratio,
                            blur_threshold_ratio=blur_threshold_ratio,
                            registration=registration, ecc_scale=ecc_scale,
                            use_gpu_scatter=use_gpu_scatter,
                            fusion=fusion, **extra)
    res_ho = sr_ho.run(kept)
    ref_idx = res_ho.reference_index
    # ``reference_index`` indexes the list given to ``run`` (= ``kept`` here).
    ref_green = kept[ref_idx].green
    # ref -> held-out transform, LR pixels (fresh estimate: it must not come
    # from the SR run's own registration, or the metric would grade itself).
    W_held = estimate_homography(ref_green, hw.green, ecc_scale=ecc_scale)
    pred_sr = predict_lr_from_hr(res_ho.hr, W_held, ratio, hw.rgb.shape[:2],
                                 forward_sigma=forward_sigma)
    # Fair baseline: denoised LR stack of the SAME fused frames (equal denoising
    # power, same homography alignment), warped into the held-out frame so the
    # comparison happens in one coordinate system. The delta then isolates
    # resolution recovery, not alignment. The stack is coverage-normalised and
    # its coverage map travels through the same warp, so pixels built from
    # fewer frames (or from none) are visible to the mask below.
    base_lr, base_cov, fused = heldout_baseline_lr(kept, res_ho)
    dst = (hw.rgb.shape[1], hw.rgb.shape[0])
    base_lr_held = cv2.warpPerspective(base_lr, W_held.astype(np.float32),
                                       dst, flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
    base_cov_held = cv2.warpPerspective(base_cov, W_held.astype(np.float32),
                                        dst, flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)
    mask = heldout_valid_mask(pred_sr, res_ho.hr.shape[:2], W_held, ratio,
                              hw.rgb.shape[:2], forward_sigma=forward_sigma)
    total_w = float(np.sum(res_ho.weights))
    mask &= base_cov_held >= total_w * 0.999
    # Pre-2026-11 baseline, reconstructed exactly from the two arrays already in
    # hand: global = acc / sum(w) = (acc / cov) * cov / sum(w). No extra warps.
    base_legacy_lr = base_lr * (base_cov / max(total_w, 1e-8))[..., None]
    base_legacy_held = cv2.warpPerspective(
        base_legacy_lr, W_held.astype(np.float32), dst,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = heldout_compare(pred_sr, base_lr_held, hw.rgb, mask,
                          base_legacy=base_legacy_held)
    return {
        "heldout_index": hidx,
        "heldout_reference_index": int(ref_idx),
        "heldout_frames_fused": len(fused),
        "heldout_blur_threshold_ratio": float(blur_threshold_ratio),
        "heldout_ibp_iterations": (None if ibp_iterations is None
                                   else int(ibp_iterations)),
        "heldout_interior_margin_lr": int(HELDOUT_INTERIOR_MARGIN_LR),
        **out,
    }


def evaluate_heldout_cfa(frames, mosaics, pattern, ratio, registration,
                         ecc_scale, hidx, use_gpu_scatter=None, pixfrac=0.5,
                         margin_lr=_CFA_HELDOUT_MARGIN_LR,
                         min_coverage_fraction=(
                             _CFA_HELDOUT_MIN_COVERAGE_FRACTION),
                         ibp_iterations=None, min_align_score=None):
    """Held-out comparison **in measurement (Bayer) space**.

    The DCB-space metric above compares two demosaiced RGB images, so it
    charges a reconstruction for not reproducing DCB's own interpolation and
    cannot rank deposition strategies (see docs/CFA_DRIZZLE.md).  This variant
    predicts the held-out frame's own Bayer samples and compares against a
    CFA-domain denoised LR stack of the same fused frames:

    * prediction: each HR estimate is pushed through the sensor forward model
      (same PSF constant as :func:`predict_lr_from_hr`) and read at the
      channel of every Bayer site;
    * baseline: the fused frames' measured samples, aligned into the held-out
      frame's grid and averaged per channel, coverage-normalised;
    * both sides scored on the same sites (interior margin, prediction support,
      baseline coverage scaled by the channel's lattice density).

    The fused frames are the run's ``kept_indices`` of ``frames`` minus
    ``hidx``, and ``mosaics`` must be the mosaics of ``frames`` in the same
    order, so the held-out frame is excluded from both sides (no leakage).

    Returns the CFA-space PSNR/gain of the shipped pipeline and of the
    CFA-domain drizzle, plus the shared mask accounting.
    """
    from raw_sr.core.cfa_drizzle import cfa_drizzle
    from raw_sr.core.cfa_heldout import (
        CFA_HELDOUT_MARGIN_LR, CFA_HELDOUT_MIN_COVERAGE_FRACTION,
        evaluate_cfa_heldout,
    )
    assert int(margin_lr) == int(CFA_HELDOUT_MARGIN_LR)
    assert float(min_coverage_fraction) == float(
        CFA_HELDOUT_MIN_COVERAGE_FRACTION)
    kept = [f for i, f in enumerate(frames) if i != hidx]
    kept_m = [m for i, m in enumerate(mosaics) if i != hidx]
    hw = frames[hidx]
    extra = {} if ibp_iterations is None else {"ibp_iterations": ibp_iterations}
    # Frame acceptance is a separate knob from the metric. On bursts whose
    # content is dominated by frequencies above the LR Nyquist, correct
    # registration scores a low ZNCC (aliasing decorrelates neighbouring
    # subpixel phases), so a metric study of that band may want the gate off;
    # ``None`` keeps the core default (0.1).
    if min_align_score is not None:
        extra["min_align_score"] = float(min_align_score)
    sr_ho = SuperResolution(ratio=ratio, blur_threshold_ratio=0.0,
                            registration=registration, ecc_scale=ecc_scale,
                            use_gpu_scatter=use_gpu_scatter, **extra)
    res_ho = sr_ho.run(kept)
    kept_m = [kept_m[i] for i in res_ho.kept_indices]
    cfa = cfa_drizzle(kept_m, res_ho.transforms, ratio, pattern,
                      res_ho.weights.tolist(), pixfrac=pixfrac)
    ref_green = kept[res_ho.reference_index].green
    W_held = estimate_homography(ref_green, hw.green, ecc_scale=ecc_scale)
    m = evaluate_cfa_heldout(
        {"pipeline": res_ho.hr, "cfa": cfa.hr},
        mosaics[hidx], kept_m, res_ho.transforms,
        res_ho.weights.tolist(), W_held, ratio, pattern,
        margin_lr=int(margin_lr),
        min_coverage_fraction=float(min_coverage_fraction))
    out = {
        "heldout_index": hidx,
        "heldout_reference_index": int(res_ho.reference_index),
        "heldout_frames_fused": len(res_ho.kept_indices),
        "heldout_cfa_psnr_db": {k: v["heldout_cfa_psnr_db"]
                                for k, v in m["per_method"].items()},
        "heldout_cfa_gain_db": {k: v["heldout_cfa_gain_db"]
                                for k, v in m["per_method"].items()},
        "heldout_cfa_baseline_psnr_db": m["baseline"]["psnr_db"],
        "heldout_cfa_mask": m["mask"],
        "heldout_cfa_pixfrac": float(pixfrac),
    }
    return out


def build_parser():
    """CLI contract (also used by tests/test_heldout_metric.py)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--roi-x", type=int, default=1650)
    ap.add_argument("--roi-y", type=int, default=1240)
    ap.add_argument("--roi-size", type=int, default=1024)
    ap.add_argument("--registration", choices=["homography", "translation"],
                    default="homography")
    ap.add_argument("--ecc-scale", type=float, default=0.25)
    ap.add_argument("--heldout", type=int, default=-1,
                    help="index of frame to hold out; -1 = skip")
    ap.add_argument("--gpu-scatter", choices=["auto", "on", "off"],
                    default="auto",
                    help="scatter deposit engine override (default auto)")
    ap.add_argument("--heldout-sweep", action="store_true",
                    help="evaluate every frame as held-out (no cherry-picking)")
    ap.add_argument("--heldout-no-psf", action="store_true",
                    help="held-out prediction without the sensor-support PSF "
                         "in the forward model (diagnostic; the default model "
                         "is the physically correct one)")
    ap.add_argument("--heldout-psf", type=float, default=None,
                    help="forward-model PSF sigma per unit ratio for the "
                         "held-out prediction (default: the core's 0.289 = "
                         "1/sqrt(12), the uniform per-LR-pixel integration "
                         "window). For sensitivity checks only — see "
                         "docs/IBP_AND_HELDOUT.md")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: <repo>/output)")
    ap.add_argument("--out-suffix", default="",
                    help="suffix appended to every output name, e.g. _B1heldout")
    ap.add_argument("--ibp-iterations", type=int, default=None,
                    help="override ibp_iterations (default: core default)")
    ap.add_argument("--fusion", choices=["demosaic_drizzle", "cfa_kr"],
                    default="demosaic_drizzle",
                    help="fusion mode: demosaic_drizzle (default) or cfa_kr")
    ap.add_argument("--cfa-heldout", action="store_true",
                    help="also report the held-out metric in measurement "
                         "(Bayer CFA) space for the shipped pipeline and the "
                         "CFA-domain drizzle (docs/CFA_DRIZZLE.md). Uses the "
                         "same held-out frames as --heldout/--heldout-sweep.")
    ap.add_argument("--cfa-heldout-pixfrac", type=float, default=0.5,
                    help="deposit footprint (LR px) of the CFA-domain "
                         "reconstruction scored by --cfa-heldout "
                         "(default 0.5, the cfa_drizzle default)")
    ap.add_argument("--cfa-heldout-margin", type=int,
                    default=_CFA_HELDOUT_MARGIN_LR,
                    help="interior margin in LR px for the CFA-space mask")
    ap.add_argument("--cfa-heldout-min-coverage", type=float,
                    default=_CFA_HELDOUT_MIN_COVERAGE_FRACTION,
                    help="share of the channel's attainable fused weight a "
                         "baseline site must receive")
    ap.add_argument("--verdict", dest="verdict", action="store_true",
                    default=True,
                    help="compute the explicit PASS/FAIL/N-A verdict and write "
                         "it into the JSON (default on)")
    ap.add_argument("--no-verdict", dest="verdict", action="store_false",
                    help="skip the verdict section")
    ap.add_argument("--self-checks", dest="self_checks", action="store_true",
                    default=True,
                    help="run the synthetic SPEC checks (parallax, "
                         "low-coverage) that need no real data (default on)")
    ap.add_argument("--no-self-checks", dest="self_checks",
                    action="store_false",
                    help="skip the synthetic SPEC checks (saves a few seconds)")
    ap.add_argument("--hf-all-frames", dest="hf_all_frames", type=int,
                    default=0, metavar="N",
                    help="measure HF gain against the first N input frames "
                         "(0 = all; --verdict needs all frames for the SPEC "
                         "'vs any single frame')")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    gpu_scatter = {"auto": None, "on": True, "off": False}[args.gpu_scatter]
    sfx = args.out_suffix
    # Resolved prediction PSF, in HR px. 0.289 (= 1/sqrt(12)) is the sensor's
    # uniform per-LR-pixel integration window; --heldout-psf exists only for
    # sensitivity reporting (docs/IBP_AND_HELDOUT.md), never to choose it.
    if args.heldout_no_psf:
        fwd_sigma_per_ratio = 0.0
    elif args.heldout_psf is None:
        fwd_sigma_per_ratio = SENSOR_PSF_SIGMA_PER_RATIO
    else:
        fwd_sigma_per_ratio = float(args.heldout_psf)
    fwd_sigma = fwd_sigma_per_ratio * float(args.ratio)

    data_dir, files = find_burst()
    x, y, s = args.roi_x, args.roi_y, args.roi_size
    roi = (x, y, s, s)
    print(f"burst: {len(files)} frames in {data_dir}")
    print(f"ROI: x={x} y={y} size={s}  ratio={args.ratio}")

    t0 = time.time()
    frames = []
    for f in files:
        t = time.time()
        fr = load_raw_linear(f, roi=roi)
        frames.append(fr)
        print(f"  loaded {os.path.basename(f)} {fr.rgb.shape} in {time.time()-t:.2f}s "
              f"mean={fr.rgb.mean():.4f}")
    print(f"loaded in {time.time()-t0:.1f}s")

    out_dir = (args.out_dir if args.out_dir else
               os.path.join(os.path.dirname(os.path.dirname(
                   os.path.abspath(__file__))), "output"))
    os.makedirs(out_dir, exist_ok=True)

    kwargs = {} if args.ibp_iterations is None else {
        "ibp_iterations": args.ibp_iterations}
    sr = SuperResolution(ratio=args.ratio,
                         registration=args.registration, ecc_scale=args.ecc_scale,
                         use_gpu_scatter=gpu_scatter, fusion=args.fusion, **kwargs)
    t = time.time()
    res = sr.run(frames)
    dt = time.time() - t
    print(f"SR done in {dt:.1f}s -> {res.hr.shape}  used {res.used_frames}/{res.total_frames}")

    # The HR grid is the reference frame's grid; comparing it with frame 0's
    # upscale adds that frame's registration shift and understates (or
    # overstates) the detail-vs-single number. Use the reference frame.
    ref_i = res.reference_index if res.reference_index is not None else 0
    ref_i = int(ref_i) if 0 <= int(ref_i) < len(frames) else 0
    naive = cv2.resize(frames[ref_i].rgb, (res.hr.shape[1], res.hr.shape[0]),
                       interpolation=cv2.INTER_CUBIC)
    metrics = dict(res.metrics)
    metrics["hf_gain_db_vs_single"] = high_frequency_gain_db(
        to_view_float(res.hr), to_view_float(naive))
    metrics["hf_gain_db_vs_single_linear"] = high_frequency_gain_db(res.hr, naive)
    metrics["sharpness_sr"] = laplacian_sharpness(res.hr)
    metrics["sharpness_naive"] = laplacian_sharpness(naive)
    metrics["sharpness_gain"] = metrics["sharpness_sr"] / max(metrics["sharpness_naive"], 1e-9)
    metrics["ratio"] = args.ratio
    metrics["roi"] = roi
    metrics["frames_total"] = res.total_frames
    metrics["frames_used"] = res.used_frames
    metrics["seconds"] = dt
    metrics["shifts"] = [[round(a, 3), round(b, 3)] for a, b in res.shifts]

    # Held-out prediction (truth-free SR proof).
    if args.heldout_sweep:
        sweep = [evaluate_heldout(frames, args.ratio, args.registration,
                                  args.ecc_scale, h, gpu_scatter, fwd_sigma,
                                  ibp_iterations=args.ibp_iterations,
                                  fusion=args.fusion)
                 for h in range(len(frames))]
        gains = [d["heldout_gain_db"] for d in sweep]
        metrics["heldout_sweep"] = sweep
        metrics["heldout_gain_min_db"] = float(min(gains))
        metrics["heldout_gain_mean_db"] = float(np.mean(gains))
        # The pre-2026-11 whole-frame number, kept only so an archived JSON can
        # be compared field by field. It is inflated by the baseline's unfilled
        # rim; never quote it (docs/IBP_AND_HELDOUT.md §5).
        legacy = [d["heldout_gain_db_legacy"] for d in sweep]
        metrics["heldout_gain_mean_db_legacy"] = float(np.mean(legacy))
        metrics["heldout_gain_min_db_legacy"] = float(min(legacy))
        metrics["heldout_valid_fraction"] = float(np.mean(
            [d["heldout_valid_fraction"] for d in sweep]))
        metrics["heldout_interior_margin_lr"] = HELDOUT_INTERIOR_MARGIN_LR
        metrics["heldout_metric"] = "masked-coverage-normalised"
        metrics["heldout_forward_psf"] = "none" if args.heldout_no_psf else "sensor"
        metrics["heldout_forward_sigma_per_ratio"] = fwd_sigma_per_ratio
        # Same sweep without the sensor-support PSF in the prediction's forward
        # model. Reported alongside the headline number so the gain is never
        # read as a model-free quantity.
        sweep_np = [evaluate_heldout(frames, args.ratio, args.registration,
                                     args.ecc_scale, h, gpu_scatter, 0.0,
                                     ibp_iterations=args.ibp_iterations,
                                     fusion=args.fusion)
                    for h in range(len(frames))]
        gains_np = [d["heldout_gain_db"] for d in sweep_np]
        metrics["heldout_gain_min_db_no_psf"] = float(min(gains_np))
        metrics["heldout_gain_mean_db_no_psf"] = float(np.mean(gains_np))
        metrics["heldout_psnr_db_sr_no_psf_mean"] = float(
            np.mean([d["heldout_psnr_db_sr"] for d in sweep_np]))
        for d, dnp in zip(sweep, sweep_np):
            print(f"held-out {d['heldout_index']}: SR={d['heldout_psnr_db_sr']:.2f}dB "
                  f"stack={d['heldout_psnr_db_denoised_stack']:.2f}dB "
                  f"gain={d['heldout_gain_db']:+.2f}dB "
                  f"(no-psf gain={dnp['heldout_gain_db']:+.2f}dB)")
        print(f"held-out gain: min={min(gains):+.2f}dB mean={np.mean(gains):+.2f}dB "
              f"| no-psf min={min(gains_np):+.2f}dB mean={np.mean(gains_np):+.2f}dB")
    elif 0 <= args.heldout < len(frames):
        d = evaluate_heldout(frames, args.ratio, args.registration,
                             args.ecc_scale, args.heldout, gpu_scatter, fwd_sigma,
                             ibp_iterations=args.ibp_iterations,
                             fusion=args.fusion)
        metrics.update(d)
        metrics["heldout_forward_psf"] = "none" if args.heldout_no_psf else "sensor"
        metrics["heldout_forward_sigma_per_ratio"] = fwd_sigma_per_ratio
        print(f"held-out frame {d['heldout_index']} PSNR: "
              f"SR={d['heldout_psnr_db_sr']:.2f}dB "
              f"stack={d['heldout_psnr_db_denoised_stack']:.2f}dB "
              f"gain={d['heldout_gain_db']:+.2f}dB")

    # CFA-space held-out metric (measurement space). Default off: it needs the
    # raw mosaics loaded a second time and a CFA-domain reconstruction per
    # held-out frame.
    if args.cfa_heldout:
        from raw_sr.core.cfa_drizzle import (
            apply_level_gains, calibrate_cfa_levels, load_cfa_frame,
            pattern_array,
        )
        cmosaic = [load_cfa_frame(f, roi=roi) for f in files]
        # Match LibRaw's per-channel level (up to ~5 % per channel) so the CFA
        # comparison is not charged a constant colour offset.
        level_stats = []
        for i, cf in enumerate(cmosaic):
            gains, stats = calibrate_cfa_levels(cf, frames[i].rgb)
            cmosaic[i] = apply_level_gains(cf, gains)
            level_stats.append({"gains": gains.tolist(), "ratios": stats})
        mos = [c.mosaic for c in cmosaic]
        pat = pattern_array(cmosaic[0].pattern)
        metrics["heldout_cfa_level_calibration"] = level_stats[0]
        metrics["heldout_cfa_pattern"] = [[int(v) for v in row] for row in pat]
        if args.heldout_sweep:
            crows = [
                evaluate_heldout_cfa(frames, mos, pat, args.ratio,
                                     args.registration, args.ecc_scale, h,
                                     gpu_scatter,
                                     pixfrac=args.cfa_heldout_pixfrac,
                                     margin_lr=args.cfa_heldout_margin,
                                     min_coverage_fraction=args.cfa_heldout_min_coverage,
                                     ibp_iterations=args.ibp_iterations)
                for h in range(len(frames))]
            metrics["heldout_cfa_sweep"] = crows
            for method in ("pipeline", "cfa"):
                g = [r["heldout_cfa_gain_db"][method] for r in crows]
                metrics[f"heldout_cfa_{method}_gain_mean_db"] = float(
                    np.mean(g))
                metrics[f"heldout_cfa_{method}_gain_min_db"] = float(
                    np.min(g))
            metrics["heldout_cfa_sites"] = int(
                np.mean([r["heldout_cfa_mask"]["sites"] for r in crows]))
            metrics["heldout_cfa_margin_lr"] = int(args.cfa_heldout_margin)
            metrics["heldout_cfa_min_coverage_fraction"] = float(
                args.cfa_heldout_min_coverage)
            metrics["heldout_cfa_metric"] = "measurement-space-cfa"
            for r in crows:
                print("held-out CFA %d: pipeline=%+.3f dB cfa=%+.3f dB "
                      "(baseline %.2f dB, %d sites)"
                      % (r["heldout_index"],
                         r["heldout_cfa_gain_db"]["pipeline"],
                         r["heldout_cfa_gain_db"]["cfa"],
                         r["heldout_cfa_baseline_psnr_db"],
                         r["heldout_cfa_mask"]["sites"]))
            print("held-out CFA gain: pipeline mean %+.3f dB, cfa mean %+.3f "
                  "dB" % (metrics["heldout_cfa_pipeline_gain_mean_db"],
                          metrics["heldout_cfa_cfa_gain_mean_db"]))
        elif 0 <= args.heldout < len(frames):
            r = evaluate_heldout_cfa(
                frames, mos, pat, args.ratio, args.registration,
                args.ecc_scale, args.heldout, gpu_scatter,
                pixfrac=args.cfa_heldout_pixfrac,
                margin_lr=args.cfa_heldout_margin,
                min_coverage_fraction=args.cfa_heldout_min_coverage,
                ibp_iterations=args.ibp_iterations)
            metrics["heldout_cfa"] = r
            metrics["heldout_cfa_metric"] = "measurement-space-cfa"
            print("held-out CFA frame %d: pipeline=%+.3f dB cfa=%+.3f dB "
                  "(baseline %.2f dB)"
                  % (r["heldout_index"],
                     r["heldout_cfa_gain_db"]["pipeline"],
                     r["heldout_cfa_gain_db"]["cfa"],
                     r["heldout_cfa_baseline_psnr_db"]))

    # ------------------------------------------------------------------ #
    # Explicit PASS / FAIL / N-A verdict (scripts/acceptance_verdict.py).
    # The metrics above are honest but undecided; this turns them into the
    # SPEC's criteria so a FAIL can never hide behind a plausible number.
    # ------------------------------------------------------------------ #
    if args.verdict:
        # SPEC §11 "vs. any single input frame": measure every frame, not only
        # the (sharpest) reference whose bicubic upscale is `naive`.
        n_hf = int(args.hf_all_frames) if args.hf_all_frames > 0 else len(frames)
        print(f"HF vs single frame: measuring {n_hf}/{len(frames)} frames...")
        metrics["hf_gain_db_vs_single_frames"] = (
            acceptance_verdict.single_frame_hf_gains(
                res.hr, frames[:n_hf], high_frequency_gain_db,
                to_view_float, max_frames=n_hf))
        metrics["hf_gain_db_vs_single_frames_scope"] = (
            "all %d input frames, bicubic upscale each" % n_hf
            if n_hf == len(frames) else
            "first %d of %d input frames" % (n_hf, len(frames)))
        # Flat-surface SNR substitute (SPEC §7 L219 wants a 15+ frame flat
        # target; this burst has 9 frames of ordinary scene, so the quietest
        # region stands in and the row stays N/A -- see the verdict detail).
        flat = acceptance_verdict.flat_region_snr(
            res.hr, naive, frames[ref_i].green, args.ratio)
        flat["frames_used"] = int(res.used_frames)
        metrics["flat_snr"] = flat
        self_checks = (acceptance_verdict.run_self_checks(ratio=args.ratio)
                       if args.self_checks else {})
        metrics["self_checks"] = self_checks
        verdict = acceptance_verdict.evaluate(
            metrics, {"flat_snr": flat}, label=f"{args.ratio:g}x")
        metrics["acceptance"] = verdict
        print()
        print(acceptance_verdict.format_table(verdict))
        failures = [c for c in verdict["criteria"]
                    if c["status"] == acceptance_verdict.STATUS_FAIL]
        metrics["acceptance_failed_ids"] = [c["id"] for c in failures]

    # Exports.
    master_path = os.path.join(out_dir, f"sr_master_{args.ratio:g}x{sfx}.tiff")
    import tifffile
    tifffile.imwrite(master_path, to_uint16_linear(res.hr))
    # Normalise display exposure for both before/after so the comparison is fair.
    expo = 4.0
    cv2.imwrite(os.path.join(out_dir, f"sr_{args.ratio:g}x_before{sfx}.png"),
                cv2.cvtColor(to_uint8_view(naive, gamma=2.2, exposure=expo), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, f"sr_{args.ratio:g}x_after{sfx}.png"),
                cv2.cvtColor(to_uint8_view(res.hr, gamma=2.2, exposure=expo), cv2.COLOR_RGB2BGR))
    with open(os.path.join(out_dir, f"sr_acceptance_{args.ratio:g}x{sfx}.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("wrote:", master_path)
    return metrics


if __name__ == "__main__":
    main()
