#!/usr/bin/env python3
"""CFA-domain drizzle versus the shipped demosaic+drizzle pipeline.

Two experiments, both driven by the same comparison code:

1. ``--synthetic`` — a burst generated from a known HR scene (detail-rich 1/f
   texture, coloured hard edges, a frequency-sweep grating), a lens PSF, per-LR
   pixel integration, a Bayer CFA, photon+read noise and 10-bit quantisation.
   Reports **ground-truth** PSNR / SSIM at HR, plus a high-pass PSNR that
   isolates the detail band, for

     * ``stack``    — denoised demosaiced LR stack, bicubic to HR (baseline)
     * ``demosaic`` — the shipped pipeline (demosaic -> RGB drizzle -> IBP)
     * ``cfa``      — :func:`raw_sr.core.cfa_drizzle.cfa_drizzle`
     * ``oracle``   — RGB drizzle of the same frames *before* the CFA.  NOTE:
                      those RGB frames are the noiseless scene samples (only
                      the mosaics receive noise), so this is a
                      **noise-free upper bound**, not a like-for-like method.
                      With a correct demosaic the real pipeline reaches it
                      within 0.07 dB (see internal audit report §2.1),
                      i.e. there is no "demosaic headroom" to win here.

2. ``--real`` — the 9-DNG burst. Reports the honest held-out gain (B4 metric)
   and the HF/sharpness numbers for the CFA output versus the shipped
   pipeline. The held-out number is expected to stay ≈0 dB: it is
   noise-limited (see docs/IBP_AND_HELDOUT.md and internal audit report), so
   it cannot rank deposition strategies. It is reported because it was the
   original acceptance criterion, not because it is sensitive here.

Usage:
  venv/bin/python scripts/exp_cfa_sr.py --synthetic
  venv/bin/python scripts/exp_cfa_sr.py --synthetic --ratios 2 3 4 --seeds 3 4 5
  venv/bin/python scripts/exp_cfa_sr.py --real --ratio 2 --roi-size 512
  venv/bin/python scripts/exp_cfa_sr.py --real --ratio 2 --real-heldout 4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from raw_sr.core.cfa_drizzle import (  # noqa: E402
    apply_level_gains, calibrate_cfa_levels, channel_map, cfa_drizzle,
    load_cfa_frame, pattern_array,
)
from raw_sr.core.cfa_heldout import (  # noqa: E402
    CFA_HELDOUT_MARGIN_LR, CFA_HELDOUT_MIN_COVERAGE_FRACTION,
    evaluate_cfa_heldout,
)
from raw_sr.core.superres import (  # noqa: E402
    RawFrame, SuperResolution, denoised_lr_stack, drizzle,
    estimate_homography, high_frequency_gain_db, iterative_back_projection,
    laplacian_sharpness, load_raw_linear, predict_lr_from_hr, to_view_float,
)
from sr_acceptance import (  # noqa: E402
    heldout_baseline_lr, heldout_compare, heldout_valid_mask,
)

BGGR = np.array([[2, 3], [1, 0]], dtype=np.int8)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _luma(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return (0.299 * img[..., 0] + 0.587 * img[..., 1]
            + 0.114 * img[..., 2]).astype(np.float32)


def psnr(a, b, mask=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    mse = float(d[mask].mean()) if mask is not None else float(d.mean())
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def ssim(a, b, mask=None):
    """Gaussian-window SSIM on luma (no skimage dependency)."""
    ga, gb = _luma(a), _luma(b)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma = cv2.GaussianBlur(ga, (0, 0), 1.5)
    mb = cv2.GaussianBlur(gb, (0, 0), 1.5)
    saa = cv2.GaussianBlur(ga * ga, (0, 0), 1.5) - ma ** 2
    sbb = cv2.GaussianBlur(gb * gb, (0, 0), 1.5) - mb ** 2
    sab = cv2.GaussianBlur(ga * gb, (0, 0), 1.5) - ma * mb
    s = ((2 * ma * mb + c1) * (2 * sab + c2)) / (
        (ma ** 2 + mb ** 2 + c1) * (saa + sbb + c2))
    return float(s[mask].mean()) if mask is not None else float(s.mean())


def hp_psnr(a, b, mask=None, sigma=1.2):
    """PSNR of the detail residual (image minus a Gaussian low-pass).

    Total PSNR is dominated by low frequencies and barely moves between
    reconstructions; this number isolates the band the SR is actually for.
    """
    def hp(x):
        y = x.astype(np.float32)
        return y - cv2.GaussianBlur(y, (0, 0), sigma,
                                    borderType=cv2.BORDER_REPLICATE)
    return psnr(hp(a), hp(b), mask)


def all_metrics(img, target, mask, sigma_hp=1.2):
    return {
        "psnr": psnr(img, target, mask),
        "ssim": ssim(img, target, mask),
        "hp_psnr": hp_psnr(img, target, mask, sigma=sigma_hp),
        "sharpness": laplacian_sharpness(img),
        "per_channel_psnr": [psnr(img[..., c], target[..., c], mask)
                             for c in range(3)],
    }


# --------------------------------------------------------------------------- #
# Synthetic scene + burst
# --------------------------------------------------------------------------- #
def make_scene(size=768, seed=3, n_objects=36):
    """Detail-rich HR scene: 1/f^0.85 texture, coloured hard edges, grating."""
    rng = np.random.default_rng(seed)
    H = W = size
    F = np.fft.fftshift(np.fft.fft2(rng.standard_normal((H, W))))
    fy = np.fft.fftshift(np.fft.fftfreq(H))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(W))[None, :]
    fr = np.sqrt(fx ** 2 + fy ** 2)
    F *= 1.0 / (fr + 0.015) ** 0.85
    base = np.real(np.fft.ifft2(np.fft.ifftshift(F))).astype(np.float32)
    base = (base - base.mean()) / (base.std() + 1e-9)
    luma = np.clip(0.42 + 0.10 * base, 0.02, 0.92)

    scene = np.stack([luma * 0.96, luma, luma * 1.04], axis=2).astype(np.float32)
    # Coloured, hard-edged objects: demosaic has to interpolate real chroma
    # edges here (this is where a CFA-first deposit stops inventing values).
    for _ in range(n_objects):
        kind = rng.integers(0, 4)
        cx, cy = rng.integers(30, W - 30), rng.integers(30, H - 30)
        a, b = int(rng.integers(8, 70)), int(rng.integers(8, 70))
        col = rng.uniform(0.05, 0.9, 3).astype(np.float32)
        if kind == 0:
            cv2.rectangle(scene, (cx, cy), (cx + a, cy + b), col.tolist(), -1)
        elif kind == 1:
            cv2.ellipse(scene, (cx, cy), (a // 2, b // 2),
                        float(rng.uniform(0, 180)), 0, 360, col.tolist(), -1)
        elif kind == 2:
            pts = np.stack([cx + rng.integers(-a, a, 3),
                            cy + rng.integers(-b, b, 3)], 1).astype(np.int32)
            cv2.fillPoly(scene, [pts], col.tolist())
        else:
            cv2.line(scene, (cx, cy), (cx + a, cy + b), col.tolist(),
                     int(rng.integers(1, 4)))

    # A frequency-sweep grating (0.15 -> 0.45 cyc/HR px) and a checkerboard
    # patch: the content that can only be recovered from >Nyquist aliasing.
    yy, xx = np.mgrid[0:110, 0:110]
    f = 0.15 + 0.30 * xx / 109.0
    amp = 0.09 * np.exp(-((xx - 55) ** 2 + (yy - 55) ** 2) / (2 * 55.0 ** 2))
    patch = amp * np.sin(2 * np.pi * (f * xx + 0.19 * yy))
    scene[H // 2 - 55:H // 2 + 55, W // 2 - 55:W // 2 + 55] += patch[..., None]
    chk = (np.indices((48, 48)).sum(0) % 2) * 0.25 - 0.12
    scene[H // 4:H // 4 + 48, 3 * W // 4:3 * W // 4 + 48] += chk[..., None]
    return np.clip(scene, 0.0, 1.0).astype(np.float32)


def _mosaic_from_rgb(rgb, pattern=BGGR):
    h, w = rgb.shape[:2]
    ch = channel_map(pattern, (h, w))
    return np.take_along_axis(rgb, ch[..., None], axis=2)[..., 0]


def _demosaic(mosaic, pattern=BGGR):
    if not np.array_equal(pattern_array(pattern) if not isinstance(
            pattern, str) else pattern, BGGR):
        raise ValueError("_demosaic helper is BGGR-specific")
    u16 = np.clip(mosaic * 65535.0, 0, 65535).astype(np.uint16)
    # Code choice is empirical: OpenCV's Bayer names are offset by one pixel
    # relative to a mosaic whose (0,0) is the first letter, so a BGGR mosaic
    # (B G / G R, ``[[2, 3], [1, 0]]``) must be demosaiced as "BayerRG" here.
    # Verified on a per-channel correlation test (scene R/G/B -> output R/G/B)
    # and pinned in tests/test_cfa_drizzle.py.  Using the "BGGR" code (or a BGR
    # variant plus a swap) puts B in the R slot and made the demosaic path look
    # ~10 dB worse than it is.
    return cv2.cvtColor(u16, cv2.COLOR_BayerRG2RGB_EA).astype(
        np.float32) / 65535.0


def _shift_hr(img, dx, dy):
    h, w = img.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    ph = np.exp(-2j * np.pi * (fx * dx + fy * dy))
    return np.stack([np.real(np.fft.ifft2(np.fft.fft2(img[..., c]) * ph))
                     for c in range(img.shape[2])], axis=2).astype(np.float32)


def synthetic_burst(ratio, lr=384, n=9, seed=3, sigma_lens=0.55,
                    full_well=1023.0, read_noise=1.5, shift_max=4.0):
    rng = np.random.default_rng(seed)
    scene = make_scene(int(lr * ratio), seed=seed)
    blurred = cv2.GaussianBlur(scene, (0, 0), sigma_lens * ratio,
                               borderType=cv2.BORDER_REFLECT)
    box = cv2.blur(blurred, (int(ratio), int(ratio)),
                   borderType=cv2.BORDER_REFLECT)
    offs = np.stack([rng.uniform(-shift_max, shift_max, n),
                     rng.uniform(-shift_max, shift_max, n)], axis=1)
    offs[0] = (0.0, 0.0)
    xs, ys = np.meshgrid(np.arange(lr, dtype=np.float32),
                         np.arange(lr, dtype=np.float32))
    mosaics, rgb_frames = [], []
    for k in range(n):
        off = 0.5 * (float(ratio) - 1.0)
        mx = (ratio * (xs + np.float32(offs[k, 0])) + off).astype(np.float32)
        my = (ratio * (ys + np.float32(offs[k, 1])) + off).astype(np.float32)
        rgb = cv2.remap(box, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
        rgb_frames.append(rgb)
        m = _mosaic_from_rgb(rgb)
        m = (rng.poisson(np.clip(m, 0, 1) * full_well)
             + rng.normal(0.0, read_noise, m.shape)) / full_well
        m = np.round(np.clip(m, 0, 1) * full_well) / full_well
        mosaics.append(m.astype(np.float32))
    # Frame k samples the scene at LR coordinate (x + offs[k]), i.e.
    # frame_k(x) = target_LR(x + offs[k]).  ``estimate_homography`` and
    # ``drizzle`` use the cv2 src->dst convention dst(x) = src(M^-1 x), so the
    # ref->frame transform is T(-offs[k]): W p = p - d puts dst(p) = frame(p+d).
    # (The opposite sign deposits every frame with a 2x shift error.)
    aligns = [np.array([[1, 0, -(offs[k, 0] - offs[0, 0])],
                        [0, 1, -(offs[k, 1] - offs[0, 1])],
                        [0, 0, 1]], dtype=np.float64) for k in range(n)]
    # Ground truth: the HR signal the frames actually measure is the
    # sensor-integrated scene (box over one LR pixel), not the un-integrated
    # lens image: comparing against the latter charges every LR-domain method
    # for the pixel-aperture blur it can never remove.
    target = _shift_hr(box, ratio * offs[0, 0], ratio * offs[0, 1])
    target = target[:int(lr * ratio), :int(lr * ratio)]
    margin = int(np.ceil(shift_max)) * int(ratio) + 6
    mask = np.zeros(target.shape[:2], dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    return {"mosaics": mosaics, "rgb": rgb_frames, "target": target,
            "mask": mask, "aligns": aligns, "offsets": offs,
            "scene": scene}


def validate_metric_synthetic(ratio, lr=384, n=9, seed=3, pixfrac=1.0,
                             holdout=4):
    """Does the CFA-space metric rank methods like ground-truth HR PSNR?

    Builds the four HR estimates from the frames that exclude ``holdout``,
    scores each against the known HR target (GT PSNR) and against the held-out
    frame's own Bayer sites (CFA-space gain).  Returns both dictionaries; the
    caller checks that the orderings agree.
    """
    b = synthetic_burst(ratio, lr=lr, n=n, seed=seed)
    mos, tgt, mask, rgb = b["mosaics"], b["target"], b["mask"], b["rgb"]
    kept_idx = [i for i in range(n) if i != holdout]
    kept_m = [mos[i] for i in kept_idx]
    dems = [_demosaic(m) for m in mos]
    frames = [RawFrame(str(k), d, d[:, :, 1].copy())
              for k, d in enumerate(dems)]
    kept = [frames[i] for i in kept_idx]
    sr = SuperResolution(ratio=ratio, blur_threshold_ratio=0.0)
    res = sr.run(kept, reference_index=0)
    w = res.weights.tolist()

    stack = denoised_lr_stack(kept, res.transforms, w,
                              coverage_normalised=True)
    stack_hr = cv2.resize(stack, (tgt.shape[1], tgt.shape[0]),
                          interpolation=cv2.INTER_CUBIC)
    orf = [RawFrame(str(k), rgb[i], rgb[i][:, :, 1].copy())
           for k, i in enumerate(kept_idx)]
    hr_or, oc = drizzle(orf, res.transforms, ratio, w)
    hr_or = iterative_back_projection(hr_or, orf, res.transforms, ratio, w,
                                      1, 0.5, 0.5, 0.1, coverage=oc)
    cfa = cfa_drizzle(kept_m, res.transforms, ratio, BGGR, w,
                      pixfrac=pixfrac)
    est = {"stack": stack_hr, "pipeline": res.hr, "cfa": cfa.hr,
           "oracle_noiseless": hr_or}
    gt = {k: psnr(v, tgt, mask) for k, v in est.items()}
    W = estimate_homography(kept[0].green, frames[holdout].green,
                            ecc_scale=0.25)
    m = evaluate_cfa_heldout(est, mos[holdout], kept_m, res.transforms, w, W,
                             ratio, BGGR)
    gain = {k: v["heldout_cfa_gain_db"] for k, v in m["per_method"].items()}
    return {"gt_psnr": gt, "cfa_gain_db": gain, "mask": m["mask"],
            "cfa_psnr_db": {k: v["heldout_cfa_psnr_db"]
                            for k, v in m["per_method"].items()}}


def run_synthetic(ratio, lr=384, n=9, seed=3, pixfrac=0.5):
    burst = synthetic_burst(ratio, lr=lr, n=n, seed=seed)
    mosaics, rgbs = burst["mosaics"], burst["rgb"]
    target, mask, aligns = burst["target"], burst["mask"], burst["aligns"]
    weights = [1.0 / n] * n
    dems = [_demosaic(m) for m in mosaics]
    frames = [RawFrame(str(k), d, d[:, :, 1].copy())
              for k, d in enumerate(dems)]

    stack = denoised_lr_stack(frames, aligns, weights,
                              coverage_normalised=True)
    stack_hr = cv2.resize(stack, (target.shape[1], target.shape[0]),
                          interpolation=cv2.INTER_CUBIC)

    hr_dem, cov = drizzle(frames, aligns, ratio, weights)
    hr_dem = iterative_back_projection(
        hr_dem, frames, aligns, ratio, weights, iterations=1, lam=0.5,
        reg_sigma=0.5, max_step=0.1, coverage=cov)

    t0 = time.perf_counter()
    res_cfa = cfa_drizzle(mosaics, aligns, ratio, BGGR, weights,
                          pixfrac=pixfrac)
    cfa_seconds = time.perf_counter() - t0

    oframes = [RawFrame(str(k), r, r[:, :, 1].copy())
               for k, r in enumerate(rgbs)]
    hr_or, ocov = drizzle(oframes, aligns, ratio, weights)
    hr_or = iterative_back_projection(
        hr_or, oframes, aligns, ratio, weights, iterations=1, lam=0.5,
        reg_sigma=0.5, max_step=0.1, coverage=ocov)

    out = {"ratio": ratio, "seed": seed, "lr": lr, "frames": n,
           "pixfrac": pixfrac, "cfa_seconds": cfa_seconds, "methods": {}}
    for name, img in (("stack", stack_hr), ("demosaic", hr_dem),
                      ("cfa", res_cfa.hr), ("oracle_noiseless", hr_or)):
        out["methods"][name] = all_metrics(img, target, mask)
    out["cfa_metrics"] = res_cfa.metrics
    return out


# --------------------------------------------------------------------------- #
# Real burst
# --------------------------------------------------------------------------- #
def find_burst():
    here = ROOT
    cands = [os.environ.get("RAW_SR_TEST_DATA"),
             os.path.join(here, "test_data")]
    for c in cands:
        if not c:
            continue
        files = sorted(glob.glob(os.path.join(c, "*.dng")))
        if len(files) >= 2:
            return c, files
    raise SystemExit("no DNG burst found (set RAW_SR_TEST_DATA)")


def _cfa_hr(mosaics, res, ratio, pixfrac):
    """CFA drizzle on a run's kept frames with the run's own transforms."""
    kept = [mosaics[i] for i in res.kept_indices]
    return cfa_drizzle(kept, res.transforms, ratio, BGGR,
                       res.weights.tolist(), pixfrac=pixfrac)


def evaluate_heldout_cfa(frames, mosaics, ratio, hidx, pixfrac,
                         registration="homography", ecc_scale=0.25,
                         margin_lr=CFA_HELDOUT_MARGIN_LR):
    """Held-out comparison for the shipped pipeline and the CFA output.

    Two metrics on the same held-out frame and the same excluded-from-fusion
    reconstruction:

    * ``heldout_gain_db`` (B4, DCB RGB space) for each method, for continuity
      with the previously reported numbers;
    * ``heldout_cfa_gain_db`` (measurement space): predict the held-out frame's
      own Bayer sites, baseline = CFA-domain denoised LR stack of the fused
      frames.  This is the metric that is not locked into DCB's representation.
    """
    kept = [f for i, f in enumerate(frames) if i != hidx]
    kept_m = [m for i, m in enumerate(mosaics) if i != hidx]
    hw = frames[hidx]
    sr = SuperResolution(ratio=ratio, blur_threshold_ratio=0.0,
                         registration=registration, ecc_scale=ecc_scale)
    res = sr.run(kept)
    cfa = _cfa_hr(kept_m, res, ratio, pixfrac)
    ref = kept[res.reference_index]
    W_held = estimate_homography(ref.green, hw.green, ecc_scale=ecc_scale)

    # --- DCB-space (B4) gains, for continuity -----------------------------
    base, base_cov, _fused = heldout_baseline_lr(kept, res)
    dst = (hw.rgb.shape[1], hw.rgb.shape[0])
    base_h = cv2.warpPerspective(base, W_held.astype(np.float32), dst,
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    cov_h = cv2.warpPerspective(base_cov, W_held.astype(np.float32), dst,
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    dcb = {}
    mask = None
    for name, hr in (("pipeline", res.hr), ("cfa", cfa.hr)):
        pred = predict_lr_from_hr(hr, W_held, ratio, hw.rgb.shape[:2])
        if mask is None:
            mask = heldout_valid_mask(pred, hr.shape[:2], W_held, ratio,
                                      hw.rgb.shape[:2])
            mask &= cov_h >= float(np.sum(res.weights)) * 0.999
        dcb[name] = heldout_compare(pred, base_h, hw.rgb, mask)

    # --- measurement (CFA) space ------------------------------------------
    cfa_metric = evaluate_cfa_heldout(
        {"pipeline": res.hr, "cfa": cfa.hr}, mosaics[hidx], kept_m,
        res.transforms, res.weights.tolist(), W_held, ratio, BGGR,
        margin_lr=int(margin_lr))
    out = {
        "heldout_index": hidx,
        "heldout_reference_index": int(res.reference_index),
        "heldout_frames_fused": len(res.kept_indices),
        "heldout_dcb_gain_db": {k: v["heldout_gain_db"]
                                for k, v in dcb.items()},
        "heldout_cfa_gain_db": {k: v["heldout_cfa_gain_db"]
                                for k, v in cfa_metric["per_method"].items()},
        "heldout_cfa_psnr_db": {k: v["heldout_cfa_psnr_db"]
                                for k, v in cfa_metric["per_method"].items()},
        "heldout_cfa_baseline_psnr_db": cfa_metric["baseline"]["psnr_db"],
        "heldout_cfa_mask": cfa_metric["mask"],
        "heldout_cfa_sites_per_channel":
            cfa_metric["mask"]["sites_per_channel"],
        "cfa_metrics": cfa.metrics,
    }
    return out


def run_real(ratio, roi_size, pixfrac, heldout_frames, out_dir):
    data_dir, files = find_burst()
    x, y = 1650, 1240
    roi = (x, y, roi_size, roi_size)
    print(f"burst: {len(files)} frames in {data_dir}; ROI {roi}; ratio {ratio}")
    frames = [load_raw_linear(f, roi=roi) for f in files]
    cfa_frames = [load_cfa_frame(f, roi=roi) for f in files]
    # Put the mosaic on LibRaw's level scale (per-channel LS gains; up to 5 %
    # per channel) so a CFA comparison is not reading a constant colour bias.
    level_stats = []
    for i, cf in enumerate(cfa_frames):
        gains, stats = calibrate_cfa_levels(cf, frames[i].rgb)
        cfa_frames[i] = apply_level_gains(cf, gains)
        level_stats.append({"gains": gains.tolist(), "ratios": stats})
    mosaics = [c.mosaic for c in cfa_frames]
    pattern = pattern_array(cfa_frames[0].pattern)
    print("pattern", pattern.tolist())

    sr = SuperResolution(ratio=ratio, blur_threshold_ratio=0.6)
    t0 = time.perf_counter()
    res = sr.run(frames)
    sr_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    cfa = _cfa_hr(mosaics, res, ratio, pixfrac)
    cfa_seconds = time.perf_counter() - t0

    lr_stack = denoised_lr_stack([frames[i] for i in res.kept_indices],
                                 res.transforms, res.weights.tolist(),
                                 coverage_normalised=True)
    base_hr = cv2.resize(lr_stack, (res.hr.shape[1], res.hr.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
    # Single-frame bicubic is the HF metric's reference, as in sr_acceptance.
    naive = cv2.resize(frames[0].rgb, (res.hr.shape[1], res.hr.shape[0]),
                       interpolation=cv2.INTER_CUBIC)
    out = {"ratio": ratio, "roi": roi, "pixfrac": pixfrac,
           "level_calibration": level_stats[0],
           "frames_used": int(res.used_frames), "sr_seconds": sr_seconds,
           "cfa_seconds": cfa_seconds, "methods": {}}
    for name, img in (("stack", base_hr), ("demosaic", res.hr),
                      ("cfa", cfa.hr)):
        out["methods"][name] = {
            "hf_gain_db_vs_stack": high_frequency_gain_db(
                to_view_float(img), to_view_float(base_hr)),
            "hf_gain_db_vs_single": high_frequency_gain_db(
                to_view_float(img), to_view_float(naive)),
            "sharpness": laplacian_sharpness(img),
            "sharpness_ratio_vs_single": (laplacian_sharpness(img)
                                          / max(laplacian_sharpness(naive),
                                                1e-9)),
        }
    for name, vals in out["methods"].items():
        print(f"  {name:10s} hf/stack {vals['hf_gain_db_vs_stack']:6.2f} dB  "
              f"hf/single {vals['hf_gain_db_vs_single']:6.2f} dB  "
              f"sharpness {vals['sharpness_ratio_vs_single']:6.2f}x")
    out["cfa_metrics"] = cfa.metrics
    out["demosaic_metrics"] = {k: res.metrics[k] for k in
                               ("hf_gain_db", "sharpness_ratio",
                                "mean_shift_px", "max_shift_px",
                                "align_residual_mean_px")}

    if heldout_frames:
        idxs = ([int(h) for h in heldout_frames] if isinstance(heldout_frames,
                                                               list)
                else list(range(len(frames)))[:int(heldout_frames)])
        rows = []
        for h in idxs:
            t0 = time.perf_counter()
            d = evaluate_heldout_cfa(frames, mosaics, ratio, h, pixfrac)
            d["seconds"] = time.perf_counter() - t0
            rows.append(d)
            dcb = d["heldout_dcb_gain_db"]
            cg = d["heldout_cfa_gain_db"]
            print(f"  held-out {h}: DCB-space pipeline {dcb['pipeline']:+.3f} "
                  f"cfa {dcb['cfa']:+.3f} dB | CFA-space pipeline "
                  f"{cg['pipeline']:+.3f} cfa {cg['cfa']:+.3f} dB "
                  f"({d['seconds']:.0f}s)")
        out["heldout_sweep"] = rows
        for space, key in (("cfa", "heldout_cfa_gain_db"),
                           ("dcb", "heldout_dcb_gain_db")):
            for method in ("pipeline", "cfa"):
                g = [r[key][method] for r in rows]
                out[f"heldout_{space}_{method}_gain_mean_db"] = float(
                    np.mean(g))
                out[f"heldout_{space}_{method}_gain_min_db"] = float(
                    np.min(g))
        print("  held-out gains (mean / min, dB):")
        for space in ("cfa", "dcb"):
            for method in ("pipeline", "cfa"):
                g = [r[f"heldout_{space}_gain_db"][method] for r in rows]
                print(f"    {space}-space {method:9s} "
                      f"{np.mean(g):+.3f} / {np.min(g):+.3f}")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"exp_cfa_{ratio:g}x_roi{roi_size}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print("wrote", path)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--ratios", type=float, nargs="+", default=[2, 3, 4])
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[3])
    ap.add_argument("--lr", type=int, default=384,
                    help="synthetic LR size (HR = ratio*lr)")
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--pixfrac", type=float, default=0.5)
    ap.add_argument("--roi-size", type=int, default=512)
    ap.add_argument("--real-heldout", default="0",
                    help="'0' disables; an int = first N frames; or a "
                         "comma-separated list")
    ap.add_argument("--synthetic-metric", action="store_true",
                    help="also validate the CFA-space held-out metric against "
                         "GT HR PSNR on the synthetic burst")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "output"))
    args = ap.parse_args(argv)
    if not (args.synthetic or args.real):
        ap.error("choose --synthetic and/or --real")

    results = {"synthetic": [], "real": []}
    if args.synthetic:
        for ratio in args.ratios:
            for seed in args.seeds:
                t0 = time.perf_counter()
                out = run_synthetic(ratio, lr=args.lr, n=args.frames,
                                    seed=seed, pixfrac=args.pixfrac)
                out["wall_seconds"] = time.perf_counter() - t0
                results["synthetic"].append(out)
                m = out["methods"]
                print(f"ratio {ratio:g} seed {seed}: "
                      f"stack {m['stack']['psnr']:.2f}/{m['stack']['hp_psnr']:.2f}  "
                      f"demosaic {m['demosaic']['psnr']:.2f}/{m['demosaic']['hp_psnr']:.2f}  "
                      f"cfa {m['cfa']['psnr']:.2f}/{m['cfa']['hp_psnr']:.2f}  "
                      f"oracle* {m['oracle_noiseless']['psnr']:.2f}/"
                      f"{m['oracle_noiseless']['hp_psnr']:.2f}  "
                      f"(PSNR/HP-PSNR dB, {out['wall_seconds']:.0f}s)")
        if results["synthetic"]:
            print("\nmean over runs (PSNR / HP-PSNR / SSIM):")
            for ratio in sorted({r["ratio"] for r in results["synthetic"]}):
                rows = [r for r in results["synthetic"] if r["ratio"] == ratio]
                line = [f"{ratio:g}x"]
                for method in ("stack", "demosaic", "cfa",
                               "oracle_noiseless"):
                    line.append(
                        f"{method} "
                        f"{np.mean([r['methods'][method]['psnr'] for r in rows]):.2f}/"
                        f"{np.mean([r['methods'][method]['hp_psnr'] for r in rows]):.2f}/"
                        f"{np.mean([r['methods'][method]['ssim'] for r in rows]):.3f}")
                print("  " + "  ".join(line))
        if args.synthetic_metric:
            from scipy.stats import spearmanr
            xs, ys = [], []
            print("\nCFA-space metric vs GT HR PSNR (frame held out):")
            for ratio in args.ratios:
                for seed in args.seeds:
                    v = validate_metric_synthetic(ratio, lr=args.lr,
                                                 n=args.frames, seed=seed)
                    gt, g = v["gt_psnr"], v["cfa_gain_db"]
                    order_ok = (sorted(gt, key=lambda k: gt[k])
                                == sorted(g, key=lambda k: g[k]))
                    print(f"  {ratio:g}x seed {seed}: GT "
                          + " ".join(f"{k}={gt[k]:.2f}" for k in
                                     ("stack", "pipeline", "cfa",
                                      "oracle_noiseless"))
                          + " | gain "
                          + " ".join(f"{k}={g[k]:+.2f}" for k in
                                     ("stack", "pipeline", "cfa",
                                      "oracle_noiseless"))
                          + f" | order {'OK' if order_ok else 'MISMATCH'}")
                    xs += [g[k] for k in gt]
                    ys += [gt[k] for k in gt]
                    results.setdefault("synthetic_metric", []).append(v)
            rho = float(spearmanr(xs, ys).statistic)
            print(f"  spearman(gain, GT PSNR) = {rho:.4f} over {len(xs)} "
                  f"method-runs")
            results["synthetic_metric_spearman"] = rho
    if args.real:
        ho = args.real_heldout
        if ho != "0":
            ho = ([int(v) for v in ho.split(",")] if "," in str(ho)
                  else int(ho))
        else:
            ho = 0
        results["real"].append(run_real(args.ratio, args.roi_size,
                                        args.pixfrac, ho, args.out_dir))

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "synthetic" if args.synthetic and not args.real else (
        "real" if args.real and not args.synthetic else "both")
    path = os.path.join(args.out_dir, f"exp_cfa_{tag}.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
