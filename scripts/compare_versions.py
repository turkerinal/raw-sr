#!/usr/bin/env python3
"""Reproducible before/after comparison of two SR core versions.

The same script is copied into both checkouts (master and the integration
branch) and run there, so each version reconstructs with *its own* core.  All
comparison arithmetic lives here, not in either version: the held-out
evaluator (coverage-normalised per-pixel baseline, 24 LR px interior margin,
sensor-PSF forward model) and the image metrics are copied verbatim from the
integration branch's ``scripts/sr_acceptance.py`` and are applied identically
to both versions' HR outputs.  Nothing in ``raw_sr/`` is modified; the only
core function touched is ``drizzle``/``iterative_back_projection``, wrapped for
timing and restored immediately.

Two modes
---------
run     Reconstruct one ratio, evaluate every held-out frame, and dump the
        HR images + a JSON record of every metric.
collect Read the per-version dumps and write the side-by-side crops, the
        amplified new-minus-master difference image and a comparison JSON.

Usage
-----
  python scripts/compare_versions.py run --label master --ratio 2 \
      --out-dir output/compare --dump-dir /tmp/rawsr/cmp
  python scripts/compare_versions.py collect --dump-dir /tmp/rawsr/cmp \
      --out-dir output/compare --ratios 2 3 4
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import raw_sr.core.superres as ss  # noqa: E402
from raw_sr.core.superres import (  # noqa: E402
    load_raw_linear,
    SuperResolution,
    estimate_homography,
    to_drizzle_matrix,
    high_frequency_gain_db,
    laplacian_sharpness,
    to_uint8_view,
    to_view_float,
)

# The sensor's per-LR-pixel integration window (same constant the integ branch
# uses; master predates it, so it is defined here for both).
SENSOR_PSF_SIGMA_PER_RATIO = 0.289
HELDOUT_INTERIOR_MARGIN_LR = 24


# --------------------------------------------------------------------------- #
# Shared evaluator (copied semantics from scripts/sr_acceptance.py on fix/bench9)
# --------------------------------------------------------------------------- #
def predict_lr_from_hr(hr, ref_to_frame, ratio, lr_shape, forward_sigma=None):
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


def lr_sampling_valid_mask(hr_shape, ref_to_frame, ratio, lr_shape,
                           forward_sigma=None):
    if forward_sigma is None:
        forward_sigma = SENSOR_PSF_SIGMA_PER_RATIO * float(ratio)
    h, w = int(lr_shape[0]), int(lr_shape[1])
    hh, wh = int(hr_shape[0]), int(hr_shape[1])
    A = np.linalg.inv(ss._normalize_homography(ref_to_frame)).astype(np.float64)
    A = np.diag([float(ratio), float(ratio), 1.0]) @ A
    # Pixel-centre aligned HR grid (matches ss.to_drizzle_matrix).
    off = 0.5 * (float(ratio) - 1.0)
    A[0, 2] += off
    A[1, 2] += off
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)], axis=0)
    q = A @ pts
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


def heldout_valid_mask(pred_sr, hr_shape, ref_to_frame, ratio, lr_shape,
                       forward_sigma=None, margin_lr=HELDOUT_INTERIOR_MARGIN_LR):
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


def coverage_lr_stack(frames, aligns, weights):
    """Coverage-normalised (per-pixel weighted mean) denoised LR stack."""
    h, w = frames[0].rgb.shape[:2]
    acc = np.zeros((h, w, 3), dtype=np.float32)
    cov = np.zeros((h, w), dtype=np.float32)
    ones = np.ones((h, w), dtype=np.float32)
    tot = 0.0
    for fr, al, wt in zip(frames, aligns, weights):
        M = to_drizzle_matrix(al, 1.0)
        acc += cv2.warpPerspective(fr.rgb.astype(np.float32), M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT) * float(wt)
        cov += cv2.warpPerspective(ones, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT) * float(wt)
        tot += float(wt)
    return acc / np.maximum(cov, 1e-6)[..., None], cov, tot


def psnr_over_mask(image, truth, mask):
    diff = (image - truth) ** 2
    mse = float(diff[mask].mean())
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def heldout_compare(pred_sr, base_lr_held, truth, mask, base_legacy_held):
    psnr_sr = psnr_over_mask(pred_sr, truth, mask)
    psnr_base = psnr_over_mask(base_lr_held, truth, mask)
    psnr_sr_legacy = float(10 * np.log10(
        1.0 / max(float(np.mean((pred_sr - truth) ** 2)), 1e-12)))
    psnr_base_legacy = float(10 * np.log10(
        1.0 / max(float(np.mean((base_legacy_held - truth) ** 2)), 1e-12)))
    return {
        "heldout_psnr_db_sr": psnr_sr,
        "heldout_psnr_db_denoised_stack": psnr_base,
        "heldout_gain_db": psnr_sr - psnr_base,
        "heldout_psnr_db_sr_legacy": psnr_sr_legacy,
        "heldout_psnr_db_denoised_stack_legacy": psnr_base_legacy,
        "heldout_gain_db_legacy": psnr_sr_legacy - psnr_base_legacy,
        "heldout_valid_pixels": int(mask.sum()),
        "heldout_valid_fraction": float(mask.mean()),
    }


def get_reference_index(res):
    if hasattr(res, "reference_index"):
        return int(res.reference_index)
    return int(res.metrics["reference_index"])


def evaluate_heldout(frames, ratio, hidx, registration, ecc_scale):
    """Exactly the integ evaluator's protocol, on this version's core."""
    kept = [f for i, f in enumerate(frames) if i != hidx]
    hw = frames[hidx]
    # blur_threshold_ratio=0.0 is the evaluator's documented default: the
    # held-out reconstruction fuses every frame, in both versions alike.
    sr_ho = SuperResolution(ratio=ratio, blur_threshold_ratio=0.0,
                            registration=registration, ecc_scale=ecc_scale)
    res_ho = sr_ho.run(kept)
    ref_idx = get_reference_index(res_ho)
    W_held = estimate_homography(kept[ref_idx].green, hw.green,
                                 ecc_scale=ecc_scale)
    pred_sr = predict_lr_from_hr(res_ho.hr, W_held, ratio, hw.rgb.shape[:2])
    fused = [kept[i] for i in res_ho.kept_indices]
    base_lr, base_cov, tot = coverage_lr_stack(fused, res_ho.transforms,
                                               res_ho.weights.tolist())
    dst = (hw.rgb.shape[1], hw.rgb.shape[0])
    base_lr_held = cv2.warpPerspective(base_lr, W_held.astype(np.float32), dst,
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
    base_cov_held = cv2.warpPerspective(base_cov, W_held.astype(np.float32),
                                        dst, flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)
    mask = heldout_valid_mask(pred_sr, res_ho.hr.shape[:2], W_held, ratio,
                              hw.rgb.shape[:2])
    mask &= base_cov_held >= tot * 0.999
    base_legacy_lr = base_lr * (base_cov / max(tot, 1e-8))[..., None]
    base_legacy_held = cv2.warpPerspective(
        base_legacy_lr, W_held.astype(np.float32), dst,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = heldout_compare(pred_sr, base_lr_held, hw.rgb, mask, base_legacy_held)
    out.update({
        "heldout_index": int(hidx),
        "heldout_reference_index": int(ref_idx),
        "heldout_frames_fused": len(fused),
        "heldout_interior_margin_lr": HELDOUT_INTERIOR_MARGIN_LR,
        "heldout_blur_threshold_ratio": 0.0,
    })
    return out


# --------------------------------------------------------------------------- #
# Resource monitoring
# --------------------------------------------------------------------------- #
def _tree_sample(pid):
    total = 0
    stack = [pid]
    while stack:
        p = stack.pop()
        try:
            st = open("/proc/%d/status" % p).read()
            for line in st.splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
            kids = open("/proc/%d/task/%d/children" % (p, p)).read().split()
            stack.extend(int(k) for k in kids)
        except (OSError, ValueError):
            pass
    return total


class PeakRSS:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak_kb = 0
        self._stop = threading.Event()
        self._th = None

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                self.peak_kb = max(self.peak_kb, _tree_sample(os.getpid()))
                time.sleep(self.interval)
        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.0)


# --------------------------------------------------------------------------- #
# run mode
# --------------------------------------------------------------------------- #
def find_burst():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [os.environ.get("RAW_SR_TEST_DATA"), os.path.join(here, "test_data")]
    for c in cands:
        if not c:
            continue
        import glob
        files = sorted(glob.glob(os.path.join(c, "*.dng")))
        if len(files) >= 2:
            return c, files
    raise SystemExit("no DNG burst found (set RAW_SR_TEST_DATA)")


def run_mode(args):
    data_dir, files = find_burst()
    x, y, s = args.roi_x, args.roi_y, args.roi_size
    roi = (x, y, s, s)
    print("burst %d frames in %s | ROI %s | ratio %s | label %s"
          % (len(files), data_dir, roi, args.ratio, args.label))
    t_load = time.time()
    frames = [load_raw_linear(f, roi=roi) for f in files]
    print("loaded in %.2fs" % (time.time() - t_load))

    sr = SuperResolution(ratio=args.ratio, registration=args.registration,
                         ecc_scale=args.ecc_scale)
    timings = {}
    orig_drizzle = ss.drizzle
    orig_ibp = ss.iterative_back_projection

    def drizzle_wrap(*a, **k):
        timings["t_drizzle"] = time.perf_counter()
        out = orig_drizzle(*a, **k)
        timings["t_drizzle_end"] = time.perf_counter()
        return out

    def ibp_wrap(*a, **k):
        timings["t_ibp"] = time.perf_counter()
        out = orig_ibp(*a, **k)
        timings["t_ibp_end"] = time.perf_counter()
        return out

    ss.drizzle = drizzle_wrap
    ss.iterative_back_projection = ibp_wrap
    with PeakRSS() as rss:
        t0 = time.perf_counter()
        res = sr.run(frames)
        t1 = time.perf_counter()
    ss.drizzle = orig_drizzle
    ss.iterative_back_projection = orig_ibp
    td = timings.get("t_drizzle", t1)
    tde = timings.get("t_drizzle_end", t1)
    tibp = timings.get("t_ibp", tde)
    tibpe = timings.get("t_ibp_end", t1)

    hr = res.hr
    naive = cv2.resize(frames[0].rgb, (hr.shape[1], hr.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    lr_stack = ss.denoised_lr_stack(
        [frames[i] for i in res.kept_indices], res.transforms,
        res.weights.tolist())
    stack_up = cv2.resize(lr_stack, (hr.shape[1], hr.shape[0]),
                          interpolation=cv2.INTER_CUBIC)

    metrics = {
        "label": args.label,
        "ratio": float(args.ratio),
        "roi": list(roi),
        "frames_total": int(res.total_frames),
        "frames_used": int(res.used_frames),
        "reference_index": get_reference_index(res),
        "kept_indices": [int(i) for i in res.kept_indices],
        "seconds_sr_wall": float(t1 - t0),
        "seconds_registration": float(td - t0),
        "seconds_drizzle": float(tde - td),
        "seconds_ibp": float(tibpe - tibp),
        "seconds_total_after_ibp": float(t1 - tibpe),
        "peak_rss_kb": int(rss.peak_kb),
        "hf_gain_db": high_frequency_gain_db(to_view_float(hr),
                                             to_view_float(stack_up)),
        "hf_gain_db_vs_single": high_frequency_gain_db(
            to_view_float(hr), to_view_float(naive)),
        "sharpness_gain": (laplacian_sharpness(hr)
                           / max(laplacian_sharpness(naive), 1e-9)),
    }
    # Version-emitted metrics (gate, residual, ghost) are copied verbatim;
    # they describe that version's run, not a re-derivation.
    for k in ("align_residual_mean_px", "align_residual_max_px",
              "ghost_energy", "ghost_energy_norm", "interframe_std",
              "robust_active", "robust_reason", "mean_align_score",
              "max_shift_px", "ibp_summary", "fusion_weight_spread"):
        if k in res.metrics:
            metrics[k] = res.metrics[k]
    metrics["shifts"] = [[float(a), float(b)] for a, b in res.shifts]

    if args.heldout_indices:
        idx = [int(v) for v in args.heldout_indices.split(",") if v != ""]
    else:
        idx = list(range(len(frames)))
    sweep = []
    for h in idx:
        if h >= len(frames):
            continue
        t = time.perf_counter()
        d = evaluate_heldout(frames, args.ratio, h, args.registration,
                             args.ecc_scale)
        d["seconds"] = time.perf_counter() - t
        sweep.append(d)
        print("held-out %d: gain %+.3f dB (legacy %+.3f) SR %.2f dB"
              % (h, d["heldout_gain_db"], d["heldout_gain_db_legacy"],
                 d["heldout_psnr_db_sr"]))
    metrics["heldout_sweep"] = sweep
    if sweep:
        gains = [d["heldout_gain_db"] for d in sweep]
        legacy = [d["heldout_gain_db_legacy"] for d in sweep]
        metrics["heldout_gain_mean_db"] = float(np.mean(gains))
        metrics["heldout_gain_min_db"] = float(np.min(gains))
        metrics["heldout_gain_mean_db_legacy"] = float(np.mean(legacy))
        metrics["heldout_gain_min_db_legacy"] = float(np.min(legacy))
        metrics["heldout_valid_fraction"] = float(np.mean(
            [d["heldout_valid_fraction"] for d in sweep]))
        metrics["heldout_metric"] = "masked-coverage-normalised"
        metrics["heldout_forward_psf"] = "sensor"
        metrics["heldout_forward_sigma_per_ratio"] = SENSOR_PSF_SIGMA_PER_RATIO
        print("held-out gain mean %+.3f / min %+.3f dB (legacy mean %+.3f)"
              % (metrics["heldout_gain_mean_db"],
                 metrics["heldout_gain_min_db"],
                 metrics["heldout_gain_mean_db_legacy"]))

    os.makedirs(args.dump_dir, exist_ok=True)
    stem = "%s_%gx" % (args.label, args.ratio)
    import tifffile
    tifffile.imwrite(os.path.join(args.dump_dir, stem + "_hr.tiff"),
                     np.clip(hr * 65535.0, 0, 65535).astype(np.uint16))
    tifffile.imwrite(os.path.join(args.dump_dir, stem + "_naive.tiff"),
                     np.clip(naive * 65535.0, 0, 65535).astype(np.uint16))
    with open(os.path.join(args.dump_dir, stem + ".json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    # Display preview (linear -> sRGB-ish, exposure 4) for visual crop choice.
    prev = to_uint8_view(hr, gamma=2.2, exposure=4.0)
    scale = 1024.0 / max(prev.shape[:2])
    if scale < 1:
        prev = cv2.resize(prev, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(args.dump_dir, stem + "_preview.png"),
                cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
    print("wrote", stem)


# --------------------------------------------------------------------------- #
# collect mode
# --------------------------------------------------------------------------- #
def display(img):
    return to_uint8_view(img, gamma=2.2, exposure=4.0)


def immerkaer_sigma(img):
    """Immerkaer (1996) fast noise-variance estimate on one channel."""
    g = img.astype(np.float64)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    r = cv2.filter2D(g, cv2.CV_64F, k, borderType=cv2.BORDER_REPLICATE)
    return float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(r)) / 6.0)


def crop_box(hr_shape, frac=(0.52, 0.46), size=256):
    h, w = hr_shape[:2]
    cx, cy = int(w * frac[0]), int(h * frac[1])
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    return x0, y0, size


def collect_mode(args):
    import tifffile
    os.makedirs(args.out_dir, exist_ok=True)
    _data_dir, files = find_burst()
    roi = (args.roi_x, args.roi_y, args.roi_size, args.roi_size)
    frame0 = load_raw_linear(files[0], roi=roi)
    frac = tuple(float(v) for v in args.crop_frac.split(","))
    rows = []
    for ratio in args.ratios:
        recs = {}
        for label in ("master", "new"):
            stem = "%s_%gx" % (label, ratio)
            with open(os.path.join(args.dump_dir, stem + ".json")) as fh:
                recs[label] = json.load(fh)
        m, n = recs["master"], recs["new"]
        hr_m = tifffile.imread(os.path.join(args.dump_dir,
                                            "master_%gx_hr.tiff" % ratio))
        hr_n = tifffile.imread(os.path.join(args.dump_dir,
                                            "new_%gx_hr.tiff" % ratio))
        naive = tifffile.imread(os.path.join(args.dump_dir,
                                             "master_%gx_naive.tiff" % ratio))
        hr_m = hr_m.astype(np.float32) / 65535.0
        hr_n = hr_n.astype(np.float32) / 65535.0
        naive = naive.astype(np.float32) / 65535.0
        # Metric baseline: bicubic-upsampled single frame, the same convention
        # scripts/sr_acceptance.py uses. The PNG panel deliberately uses the
        # nearest-neighbour upscale asked for by the comparison protocol.
        cubic = cv2.resize(frame0.rgb, (hr_m.shape[1], hr_m.shape[0]),
                           interpolation=cv2.INTER_CUBIC)
        for label, img in (("master", hr_m), ("new", hr_n)):
            recs[label]["hf_gain_db_vs_single"] = high_frequency_gain_db(
                to_view_float(img), to_view_float(cubic))
            recs[label]["sharpness_gain"] = (
                laplacian_sharpness(img)
                / max(laplacian_sharpness(cubic), 1e-9))

        x0, y0, size = crop_box(hr_m.shape, frac=frac, size=args.crop_size)
        # Nearest-neighbour upscale of the single frame: already at HR size.
        crop_naive = display(naive[y0:y0 + size, x0:x0 + size])
        crop_m = display(hr_m[y0:y0 + size, x0:x0 + size])
        crop_n = display(hr_n[y0:y0 + size, x0:x0 + size])
        sep = np.full((size, 4, 3), 255, np.uint8)
        panel = np.concatenate([crop_naive, sep, crop_m, sep, crop_n], axis=1)
        name = "compare_%gx_crop.png" % ratio
        cv2.imwrite(os.path.join(args.out_dir, name),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        # Crop-local noise proxy (Immerkaer) on the crop's display-space green
        # channel: same crop and same code for both versions. Ringing/halos are
        # judged visually (see docs/BENCHMARK.md) because a 2-D local-extremum
        # test cannot see a fringe that runs along a long edge.
        for label, img in (("master", hr_m), ("new", hr_n)):
            cl = img[y0:y0 + size, x0:x0 + size]
            cd = display(cl)
            recs[label]["crop_noise_sigma_display_green"] = immerkaer_sigma(
                cd[..., 1].astype(np.float32))

        # Full-frame comparison at 460 px per panel for the visual notes.
        def preview(img):
            v = display(img)
            s = 460.0 / max(v.shape[:2])
            return cv2.resize(v, None, fx=s, fy=s,
                              interpolation=cv2.INTER_AREA)
        pm, pn = preview(hr_m), preview(hr_n)
        panel_full = np.concatenate([pm, np.full((pm.shape[0], 4, 3), 255,
                                                np.uint8), pn], axis=1)
        cv2.imwrite(os.path.join(args.out_dir,
                                 "compare_%gx_full.png" % ratio),
                    cv2.cvtColor(panel_full, cv2.COLOR_RGB2BGR))

        # Amplified (x4) signed difference, new - master, in display space:
        # literally 128 + 4*(new-master) per channel, clipped. No per-crop
        # normalisation, so the grey level is an absolute scale (one display
        # level of difference = 32 grey levels of the image) and a structural
        # change near an edge reads as a coloured fringe while sensor-level
        # noise stays within a few grey levels.
        diff = (crop_n.astype(np.float32) - crop_m.astype(np.float32))
        vis = np.clip(128.0 + 4.0 * diff, 0, 255)
        cv2.imwrite(os.path.join(args.out_dir,
                                 "diff_%gx_amp4.png" % ratio),
                    cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR))
        # Companion view for reading a difference that is mostly below one
        # display level: same signed field, additionally stretched so its own
        # 99.5th percentile maps to full scale. Structure (edges, ringing,
        # texture) becomes visible; the scale is per-image, not absolute.
        d4 = diff * 4.0
        sc = max(float(np.percentile(np.abs(d4), 99.5)), 0.5)
        u = np.clip(d4 / sc, -1.0, 1.0)
        st = np.empty_like(u)
        st[..., 0] = 128.0 + u.mean(axis=2) * 127.0
        st[..., 1] = 128.0 - np.abs(u).mean(axis=2) * 90.0
        st[..., 2] = 128.0 - u.mean(axis=2) * 127.0
        cv2.imwrite(os.path.join(args.out_dir,
                                 "diff_%gx_amp4_stretch.png" % ratio),
                    cv2.cvtColor(np.clip(st, 0, 255).astype(np.uint8),
                                 cv2.COLOR_RGB2BGR))
        diff_scale_after_x4 = sc * 255.0
        diff_lin = hr_n - hr_m
        diff_stats = {
            "diff_rms_linear": float(np.sqrt(np.mean(diff_lin ** 2))),
            "diff_max_abs_linear": float(np.max(np.abs(diff_lin))),
            "diff_rms_display": float(np.sqrt(np.mean(diff ** 2))),
            "diff_p995_abs_linear": float(np.percentile(np.abs(diff_lin),
                                                        99.5)),
        }

        rows.append({
            "ratio": ratio,
            "crop_box_hr": [x0, y0, size],
            "master": {k: m.get(k) for k in (
                "frames_used", "reference_index", "kept_indices",
                "seconds_sr_wall", "seconds_registration", "seconds_drizzle",
                "seconds_ibp", "peak_rss_kb", "hf_gain_db",
                "hf_gain_db_vs_single", "sharpness_gain",
                "align_residual_mean_px", "align_residual_max_px",
                "ghost_energy", "robust_active", "robust_reason",
                "heldout_gain_mean_db", "heldout_gain_min_db",
                "heldout_gain_mean_db_legacy", "heldout_valid_fraction",
                "crop_noise_sigma_display_green")},
            "new": {k: n.get(k) for k in (
                "frames_used", "reference_index", "kept_indices",
                "seconds_sr_wall", "seconds_registration", "seconds_drizzle",
                "seconds_ibp", "peak_rss_kb", "hf_gain_db",
                "hf_gain_db_vs_single", "sharpness_gain",
                "align_residual_mean_px", "align_residual_max_px",
                "ghost_energy", "robust_active", "robust_reason",
                "heldout_gain_mean_db", "heldout_gain_min_db",
                "heldout_gain_mean_db_legacy", "heldout_valid_fraction",
                "crop_noise_sigma_display_green")},
            "diff": diff_stats,
            "diff_scale_after_x4_display": diff_scale_after_x4,
            "images": {"crop_panel": name,
                       "full_panel": "compare_%gx_full.png" % ratio,
                       "diff_amp4": "diff_%gx_amp4.png" % ratio,
                       "diff_amp4_stretch": (
                           "diff_%gx_amp4_stretch.png" % ratio)},
        })
        print("collected %gx" % ratio)
    out = {
        "protocol": {
            "burst": "9-DNG handheld burst",
            "roi": [args.roi_x, args.roi_y, args.roi_size],
            "heldout": "all 9 frames, coverage-normalised per-pixel baseline, "
                       "24 LR px interior margin, sensor PSF forward model",
            "crop": "256 HR px, same coordinates for all three panels; "
                    "single frame nearest-neighbour upscaled",
            "diff": "new - master in display space, amplified x4",
        },
        "rows": rows,
    }
    with open(os.path.join(args.out_dir, "comparison.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", os.path.join(args.out_dir, "comparison.json"))


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--ratio", type=float, required=True)
    r.add_argument("--roi-x", type=int, default=1650)
    r.add_argument("--roi-y", type=int, default=1240)
    r.add_argument("--roi-size", type=int, default=1024)
    r.add_argument("--registration", choices=["homography", "translation"],
                   default="homography")
    r.add_argument("--ecc-scale", type=float, default=0.25)
    r.add_argument("--heldout-indices", default="",
                   help="comma list; empty = every frame")
    r.add_argument("--dump-dir", required=True)
    r.add_argument("--out-dir", default=None)
    r.set_defaults(func=run_mode)
    c = sub.add_parser("collect")
    c.add_argument("--dump-dir", required=True)
    c.add_argument("--out-dir", required=True)
    c.add_argument("--ratios", type=float, nargs="+", default=[2, 3, 4])
    c.add_argument("--crop-frac", default="0.52,0.46",
                   help="crop centre as a fraction of the HR frame (x,y)")
    c.add_argument("--crop-size", type=int, default=256,
                   help="crop side in HR px")
    c.add_argument("--roi-x", type=int, default=1650)
    c.add_argument("--roi-y", type=int, default=1240)
    c.add_argument("--roi-size", type=int, default=1024)
    c.set_defaults(func=collect_mode)
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    a.func(a)
