#!/usr/bin/env python3
"""WP1 Evaluation: CFA Anisotropic Kernel Regression (cfa_kr) vs current SR.

Measures (all with ESTIMATED registration):
1. Synthetic GT (audit_c11_part2 harness) at 2x and 4x: GT PSNR, MTF50,
   flat noise, Siemens star chroma RMS.
2. Real-edge MTF50 at 4x on own9 and HDR+ 0919 (same edge ROI as audit_c11_part2).
3. Periodic 4x4 grid spectral peak at f = 1/4 cyc/HR px in a flat area at 4x.
4. Exposure gating on full 10-frame HDR+ 33TJ_20150810_140637_256 (9/10 frames,
   N009 rejected).
5. Honest 4-panel 4x crops under docs/wp1/:
   (single bicubic | current SR | new cfa_kr SR | difference amplified 8x)
   for own9 and HDR+ 0919 (< 1.5 MB each).
"""

from __future__ import annotations

import glob
import json
import os
import resource
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from raw_sr.core.superres import RawFrame, SuperResolution, load_raw_linear, to_uint8_view
from scripts.audit_c11_part1 import choose_exposure, make_labelled_grid, pick_high_detail_crop
from scripts.audit_c11_part2 import (
    BGGR,
    _demosaic_bggr,
    compute_slanted_edge_mtf50,
    immerkaer_sigma,
    psnr,
    synthesize_target_burst,
)


def measure_lr_grid_peak(img: np.ndarray, ratio: int = 4) -> float:
    """Measure spectral peak ratio at LR-pixel frequency (f = 1/ratio cyc/HR px)
    in the smoothest 256x256 interior patch of a 4x HR image."""
    gray = img[..., 1].astype(np.float64)
    H, W = gray.shape
    sz = 256
    best_var = 1e18
    best_patch = gray[H // 2 - sz // 2:H // 2 + sz // 2, W // 2 - sz // 2:W // 2 + sz // 2]
    for y in range(256, H - 256 - sz, 256):
        for x in range(256, W - 256 - sz, 256):
            p = gray[y:y + sz, x:x + sz]
            v = float(np.var(p))
            if v < best_var and float(p.mean()) > 0.005:
                best_var = v
                best_patch = p
    p = best_patch - best_patch.mean()
    win = np.outer(np.hanning(sz), np.hanning(sz))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(p * win))) ** 2
    c = sz // 2
    f_lr = sz // ratio  # 64 for sz=256, ratio=4
    # Peak at (+/- f_lr, 0) and (0, +/- f_lr) vs local annulus background
    peaks = [
        spec[c, c + f_lr],
        spec[c, c - f_lr],
        spec[c + f_lr, c],
        spec[c - f_lr, c],
    ]
    peak_val = float(np.max(peaks))
    # Local background around f_lr (excluding +/- 3 bins around the peak)
    yy, xx = np.ogrid[-c:sz - c, -c:sz - c]
    rr = np.sqrt(xx * xx + yy * yy)
    ring = (rr >= f_lr - 8) & (rr <= f_lr + 8)
    for py, px in ((0, f_lr), (0, -f_lr), (f_lr, 0), (-f_lr, 0)):
        ring &= ~((np.abs(yy - py) <= 3) & (np.abs(xx - px) <= 3))
    bg = float(np.median(spec[ring]))
    return round(peak_val / max(bg, 1e-18), 2)


def eval_synthetic() -> dict:
    out = {}
    for ratio in (2, 4):
        b = synthesize_target_burst(ratio=ratio, lr=256, n=9, seed=7)
        mos, mask = b["mosaics"], b["mask"]
        target_box, target_lens = b["target"], b["blurred"]
        dems = [_demosaic_bggr(m) for m in mos]
        frames = [
            RawFrame(str(k), d, d[..., 1].copy(), mosaic=m, pattern=BGGR)
            for k, (d, m) in enumerate(zip(dems, mos))
        ]
        hr_w = target_box.shape[1]
        cx, cy, sz = int(hr_w * 0.75), int(hr_w * 0.5), 64 * ratio

        row = {}
        for tag, fusion, ibp in (
            ("current_sr", "demosaic_drizzle", 1),
            ("cfa_kr_ibp0", "cfa_kr", 0),
            ("cfa_kr_ibp1", "cfa_kr", 1),
        ):
            sr = SuperResolution(
                ratio=float(ratio), blur_threshold_ratio=0.0,
                registration="homography", ecc_scale=0.25,
                fusion=fusion, ibp_iterations=ibp,
            )
            t0 = time.perf_counter()
            res = sr.run(frames, reference_index=0)
            dt = time.perf_counter() - t0
            patch = res.hr[cy - sz // 2:cy + sz // 2, cx - sz // 2:cx + sz // 2]
            crms = float(np.sqrt(
                np.mean((patch[..., 0] - patch[..., 1]) ** 2
                        + (patch[..., 2] - patch[..., 1]) ** 2)
            ))
            row[tag] = {
                "mtf50": round(compute_slanted_edge_mtf50(res.hr, float(ratio)), 4),
                "psnr_box_db": round(psnr(res.hr, target_box, mask), 2),
                "psnr_lens_db": round(psnr(res.hr, target_lens, mask), 2),
                "flat_noise_sigma": round(immerkaer_sigma(res.hr[30:90, 30:90, 1]), 5),
                "star_chroma_rms": round(crms, 5),
                "seconds": round(dt, 2),
            }
        out[f"{ratio}x"] = row
    return out


def eval_real_4x() -> dict:
    out = {}
    configs = (
        ("own9", "data/own9", (1650, 1240, 1024, 1024),
         os.path.join(ROOT, "docs", "wp1", "own9_4x_wp1_4panel.png")),
        ("hdrplus_0919", "data/hdrplus/0919_20150910_131720_137", None,
         os.path.join(ROOT, "docs", "wp1", "hdrplus_0919_4x_wp1_4panel.png")),
    )
    for key, bdir, roi, out_png in configs:
        files = sorted(glob.glob(os.path.join(bdir, "*.dng")))
        if roi is None:
            import rawpy
            with rawpy.imread(files[0]) as raw:
                H, W = raw.sizes.height, raw.sizes.width
            s = 1024
            roi = (((W - s) // 2) & ~1, ((H - s) // 2) & ~1, s, s)
        frames = [load_raw_linear(f, roi=roi) for f in files]

        sr_cur = SuperResolution(
            ratio=4.0, registration="homography", ecc_scale=0.25,
            fusion="demosaic_drizzle", memory_budget_bytes=4 * 1024**3,
        )
        t0 = time.perf_counter()
        res_cur = sr_cur.run(frames)
        t_cur = time.perf_counter() - t0

        sr_kr0 = SuperResolution(
            ratio=4.0, registration="homography", ecc_scale=0.25,
            fusion="cfa_kr", ibp_iterations=0, memory_budget_bytes=4 * 1024**3,
        )
        t0 = time.perf_counter()
        res_kr0 = sr_kr0.run(frames)
        t_kr0 = time.perf_counter() - t0

        sr_kr1 = SuperResolution(
            ratio=4.0, registration="homography", ecc_scale=0.25,
            fusion="cfa_kr", ibp_iterations=1, memory_budget_bytes=4 * 1024**3,
        )
        t0 = time.perf_counter()
        res_kr1 = sr_kr1.run(frames)
        t_kr1 = time.perf_counter() - t0

        ref_i = int(res_cur.reference_index)
        g_lr = frames[ref_i].green
        gx = np.abs(cv2.Sobel(g_lr, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(g_lr, cv2.CV_32F, 0, 1, ksize=3))
        score_map = gx - 1.5 * gy
        best_val, best_xy = -1.0, (512, 512)
        for y in range(128, 1024 - 128, 32):
            for x in range(128, 1024 - 128, 32):
                sc = float(score_map[y - 16:y + 16, x - 16:x + 16].mean())
                if sc > best_val:
                    best_val = sc
                    best_xy = (x, y)
        ex, ey = best_xy
        edge_roi_hr = (ex * 4 - 64, ey * 4 - 64, 128, 128)
        bic = cv2.resize(frames[ref_i].rgb, (res_cur.hr.shape[1], res_cur.hr.shape[0]),
                         interpolation=cv2.INTER_CUBIC)

        mtf_bic = compute_slanted_edge_mtf50(bic, 4.0, roi=edge_roi_hr)
        mtf_cur = compute_slanted_edge_mtf50(res_cur.hr, 4.0, roi=edge_roi_hr)
        mtf_kr0 = compute_slanted_edge_mtf50(res_kr0.hr, 4.0, roi=edge_roi_hr)
        mtf_kr1 = compute_slanted_edge_mtf50(res_kr1.hr, 4.0, roi=edge_roi_hr)

        grid_cur = measure_lr_grid_peak(res_cur.hr, 4)
        grid_kr0 = measure_lr_grid_peak(res_kr0.hr, 4)
        grid_kr1 = measure_lr_grid_peak(res_kr1.hr, 4)

        # Save 4-panel crop: single bicubic | current SR | new cfa_kr SR | 8x amplified diff
        exp = choose_exposure(res_cur.hr, target_p98=0.82)
        crop_sz = 384
        cx0, cy0 = pick_high_detail_crop(to_uint8_view(res_cur.hr, gamma=2.2, exposure=exp) / 255.0,
                                         crop_size=crop_sz)
        sl = (slice(cy0, cy0 + crop_sz), slice(cx0, cx0 + crop_sz))
        u_bic = to_uint8_view(bic[sl], gamma=2.2, exposure=exp)
        u_cur = to_uint8_view(res_cur.hr[sl], gamma=2.2, exposure=exp)
        u_kr = to_uint8_view(res_kr1.hr[sl], gamma=2.2, exposure=exp)
        diff_amp = np.clip(np.abs(u_kr.astype(np.float32) - u_cur.astype(np.float32)) * 8.0, 0, 255).astype(np.uint8)
        png_info = make_labelled_grid(
            [
                ("(a) Single ref 4x bicubic", u_bic),
                ("(b) Current raw_sr 4x", u_cur),
                ("(c) New cfa_kr 4x (IBP=1)", u_kr),
                ("(d) |cfa_kr - current| x8", diff_amp),
            ],
            out_png,
            f"{key} 4x WP1 comparison (exp={exp:.2f}x, crop={cx0},{cy0})",
        )

        out[key] = {
            "edge_roi_hr": list(edge_roi_hr),
            "mtf50": {
                "single_bicubic": round(mtf_bic, 4),
                "current_sr": round(mtf_cur, 4),
                "cfa_kr_ibp0": round(mtf_kr0, 4),
                "cfa_kr_ibp1": round(mtf_kr1, 4),
                "gain_kr1_vs_current_pct": round((mtf_kr1 - mtf_cur) / max(mtf_cur, 1e-9) * 100.0, 1),
            },
            "lr_grid_peak_ratio": {
                "current_sr": grid_cur,
                "cfa_kr_ibp0": grid_kr0,
                "cfa_kr_ibp1": grid_kr1,
            },
            "time_s": {
                "current_sr": round(t_cur, 2),
                "cfa_kr_ibp0": round(t_kr0, 2),
                "cfa_kr_ibp1": round(t_kr1, 2),
            },
            "png": png_info,
        }
    return out


def eval_33tj_full_exposure_gate() -> dict:
    bdir = "data/hdrplus/33TJ_20150810_140637_256"
    files = sorted(glob.glob(os.path.join(bdir, "*.dng")))
    import rawpy
    with rawpy.imread(files[0]) as raw:
        H, W = raw.sizes.height, raw.sizes.width
    s = 1024
    roi = (((W - s) // 2) & ~1, ((H - s) // 2) & ~1, s, s)
    frames = [load_raw_linear(f, roi=roi) for f in files]
    sr = SuperResolution(ratio=2.0, registration="homography", ecc_scale=0.25, fusion="cfa_kr")
    res = sr.run(frames)
    return {
        "total_frames": res.total_frames,
        "used_frames": res.used_frames,
        "reference_index": res.reference_index,
        "reference_name": frames[res.reference_index].name,
        "exposure_gate_dropped": res.metrics.get("exposure_gate_dropped", []),
    }


def main():
    print("1. Synthetic target...")
    syn = eval_synthetic()
    print(json.dumps(syn, indent=2))

    print("2. Real 4x (own9 + HDR+ 0919)...")
    real = eval_real_4x()
    print(json.dumps(real, indent=2))

    print("3. Full 10-frame 33TJ exposure gating...")
    exp_33tj = eval_33tj_full_exposure_gate()
    print(json.dumps(exp_33tj, indent=2))

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    out = {"synthetic": syn, "real_4x": real, "exp_33tj": exp_33tj, "eval_peak_rss_mb": round(rss_mb, 1)}
    with open("/tmp/rawsr/wp1_eval.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("Wrote /tmp/rawsr/wp1_eval.json")


if __name__ == "__main__":
    main()
