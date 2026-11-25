#!/usr/bin/env python3
"""C11 Audit — Part 2: Quantitative diagnosis of resolution loss.

Measures:
1. Synthetic ISO 12233 slanted-edge + Siemens star burst (2x and 4x):
   - MTF50 (cyc/LR px) and GT PSNR (dB) for:
     single_bicubic, current_sr (estimated shifts), current_sr_exact,
     drizzle_pf03 / pf05 / pf10, cfa_drizzle_exact, oracle_rgb_exact,
     plus RL-deconvolved single_bicubic vs RL-deconvolved current_sr.
   - Registration error (ECC vs exact shifts) vs required 1/(2*ratio) LR px.
   - Saves visual comparison PNG docs/audit-c11/synthetic_target_4x.png.
2. Real-burst diagnostics (own9 + HDR+ 0919, 0382, 33TJ_short, 6G7M_short):
   - Sub-pixel diversity: (dx mod 1, dy mod 1) 2x2 and 4x4 bin occupancy.
   - Registration residuals (mean/max LR px).
   - Linear noise level (single frame vs denoised stack).
   - Real-edge MTF50 on own9 and HDR+ 0919 (single_bicubic vs current_sr).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from raw_sr.core.cfa_drizzle import cfa_drizzle, channel_map
from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    denoised_lr_stack,
    drizzle,
    iterative_back_projection,
    load_raw_linear,
    to_uint8_view,
)

BGGR = np.array([[2, 3], [1, 0]], dtype=np.int8)


# --------------------------------------------------------------------------- #
# Synthetic resolution target (slanted edge + Siemens star)
# --------------------------------------------------------------------------- #
def make_resolution_target(hr_size: int = 1024, tilt_deg: float = 5.7) -> np.ndarray:
    """Render an HR test chart: left half = slanted edge, right half = Siemens star."""
    H = W = hr_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    # Left half: slanted edge at x = W/4, tilted by tilt_deg
    theta = np.deg2rad(tilt_deg)
    cx_e, cy_e = W * 0.25, H * 0.5
    # Signed distance to edge
    dist = (xx - cx_e) * np.cos(theta) + (yy - cy_e) * np.sin(theta)
    # Smooth 1-px analytical anti-aliasing at HR
    edge = 0.15 + 0.60 * np.clip(0.5 + dist, 0.0, 1.0)

    # Right half: 36-spoke Siemens star centred at (0.75*W, 0.5*H)
    cx_s, cy_s = W * 0.75, H * 0.5
    dx = xx - cx_s
    dy = yy - cy_s
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)
    n_spokes = 36
    star_val = 0.45 + 0.30 * np.tanh(8.0 * np.sin(n_spokes * ang))
    # Mask star inside radius 0.22*W
    r_max = W * 0.22
    star = np.where(r <= r_max, star_val, 0.45)

    gray = np.where(xx < W * 0.5, edge, star).astype(np.float32)
    return np.stack([gray, gray, gray], axis=2)


def _mosaic_from_rgb(rgb: np.ndarray, pattern=BGGR) -> np.ndarray:
    ch = channel_map(pattern, rgb.shape[:2])
    return np.take_along_axis(rgb, ch[..., None], axis=2)[..., 0]


def _demosaic_bggr(mosaic: np.ndarray) -> np.ndarray:
    u16 = np.clip(mosaic * 65535.0, 0, 65535).astype(np.uint16)
    return cv2.cvtColor(u16, cv2.COLOR_BayerRG2RGB_EA).astype(np.float32) / 65535.0


def synthesize_target_burst(
    ratio: int,
    lr: int = 256,
    n: int = 9,
    seed: int = 7,
    sigma_lens_lr: float = 0.35,
    full_well: float = 2048.0,
    read_noise: float = 1.5,
):
    rng = np.random.default_rng(seed)
    hr_size = lr * ratio
    scene = make_resolution_target(hr_size=hr_size, tilt_deg=5.7)
    # Lens optical PSF
    blurred = cv2.GaussianBlur(
        scene, (0, 0), sigma_lens_lr * ratio, borderType=cv2.BORDER_REFLECT
    )
    # Sensor pixel aperture (box integration over 1 LR pixel)
    box = cv2.blur(blurred, (ratio, ratio), borderType=cv2.BORDER_REFLECT)

    # Known subpixel shifts (frame 0 = (0,0), others uniform in [-3, 3])
    offs = np.stack([rng.uniform(-3.0, 3.0, n), rng.uniform(-3.0, 3.0, n)], axis=1)
    offs[0] = (0.0, 0.0)

    xs, ys = np.meshgrid(np.arange(lr, dtype=np.float32), np.arange(lr, dtype=np.float32))
    off = 0.5 * (float(ratio) - 1.0)
    mosaics, rgb_frames = [], []
    for k in range(n):
        mx = (ratio * (xs + np.float32(offs[k, 0])) + off).astype(np.float32)
        my = (ratio * (ys + np.float32(offs[k, 1])) + off).astype(np.float32)
        rgb = cv2.remap(box, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        rgb_frames.append(rgb)
        m = _mosaic_from_rgb(rgb)
        m = (
            rng.poisson(np.clip(m, 0, 1) * full_well)
            + rng.normal(0.0, read_noise, m.shape)
        ) / full_well
        m = np.clip(m, 0.0, 1.0).astype(np.float32)
        mosaics.append(m)

    aligns = [
        np.array([[1, 0, -offs[k, 0]], [0, 1, -offs[k, 1]], [0, 0, 1]], dtype=np.float64)
        for k in range(n)
    ]
    margin = 4 * ratio + 8
    mask = np.zeros((hr_size, hr_size), dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    return {
        "scene": scene,
        "blurred": blurred,
        "target": box,
        "mosaics": mosaics,
        "rgb": rgb_frames,
        "aligns": aligns,
        "offsets": offs,
        "mask": mask,
    }


def compute_slanted_edge_mtf50(
    img: np.ndarray,
    ratio: float,
    tilt_deg: float = 5.7,
    roi: tuple[int, int, int, int] | None = None,
) -> float:
    """ISO 12233-style slanted-edge MTF50 in cycles per LR pixel."""
    gray = img[..., 1].astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    H, W = gray.shape
    if roi is None:
        # Default: central band around x = W/4
        cx = int(W * 0.25)
        hw = int(W * 0.08)
        y0, y1 = int(H * 0.2), int(H * 0.8)
        x0, x1 = max(0, cx - hw), min(W, cx + hw)
    else:
        x0, y0, rw, rh = roi
        x1, y1 = x0 + rw, y0 + rh

    patch = gray[y0:y1, x0:x1]
    ph, pw = patch.shape
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float64)

    # Estimate subpixel edge centre per row via centroid of horizontal gradient
    gx = np.diff(patch, axis=1)
    # Use absolute gradient weights
    w_g = np.abs(gx) ** 2
    row_sum = w_g.sum(axis=1)
    valid = row_sum > np.percentile(row_sum, 20)
    xs_g = np.arange(pw - 1, dtype=np.float64) + 0.5
    centers = (w_g[valid] @ xs_g) / row_sum[valid]
    ys_v = np.arange(ph, dtype=np.float64)[valid]
    # Fit line x = a*y + b
    a, b = np.polyfit(ys_v, centers, 1)
    norm = np.sqrt(1.0 + a * a)
    # Signed distance perpendicular to edge (in HR px)
    dist = ((xx - (a * yy + b)) / norm).ravel()
    vals = patch.ravel()

    # Oversample ESF at 4x HR resolution (bin width = 0.25 HR px)
    oversamp = 4
    bin_w = 1.0 / oversamp
    d_max = min(16.0 * ratio, pw * 0.4)
    bins = np.arange(-d_max, d_max + bin_w, bin_w)
    idx = np.digitize(dist, bins) - 1
    n_bins = len(bins) - 1
    esf = np.zeros(n_bins, dtype=np.float64)
    cnt = np.zeros(n_bins, dtype=np.float64)
    ok = (idx >= 0) & (idx < n_bins)
    np.add.at(esf, idx[ok], vals[ok])
    np.add.at(cnt, idx[ok], 1.0)
    # Interpolate any empty bins
    good = cnt > 0
    if good.sum() < 16:
        return 0.0
    esf[good] /= cnt[good]
    esf = np.interp(np.arange(n_bins), np.where(good)[0], esf[good])

    # LSF = finite difference of ESF
    lsf = np.diff(esf)
    # Centre window on peak |LSF|
    pk = int(np.argmax(np.abs(lsf)))
    win_half = min(pk, len(lsf) - 1 - pk, int(8 * ratio * oversamp))
    if win_half < 8:
        return 0.0
    lsf_w = lsf[pk - win_half:pk + win_half + 1]
    lsf_w = lsf_w * np.hamming(len(lsf_w))

    # FFT -> MTF
    n_fft = max(1024, 1 << int(np.ceil(np.log2(len(lsf_w) * 4))))
    mtf = np.abs(np.fft.rfft(lsf_w, n=n_fft))
    if mtf[0] <= 1e-12:
        return 0.0
    mtf /= mtf[0]
    # Frequency in cycles per HR pixel -> convert to cycles per LR pixel (* ratio)
    freqs_hr = np.fft.rfftfreq(n_fft, d=bin_w)
    freqs_lr = freqs_hr * float(ratio)

    # Find first crossing below 0.5
    below = np.where(mtf < 0.5)[0]
    if len(below) == 0:
        return float(freqs_lr[-1])
    i1 = int(below[0])
    if i1 == 0:
        return 0.0
    i0 = i1 - 1
    # Linear interpolation
    t = (0.5 - mtf[i0]) / (mtf[i1] - mtf[i0] + 1e-12)
    return float(freqs_lr[i0] + t * (freqs_lr[i1] - freqs_lr[i0]))


def rl_deconvolve(img: np.ndarray, sigma: float, iterations: int = 8) -> np.ndarray:
    """Classical Richardson-Lucy deconvolution (non-neural) with Gaussian PSF."""
    k = int(np.ceil(3.0 * sigma)) * 2 + 1
    eps = 1e-6
    obs = np.clip(img.astype(np.float32), eps, None)
    est = obs.copy()
    for _ in range(iterations):
        conv = cv2.GaussianBlur(est, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
        rel = obs / np.maximum(conv, eps)
        corr = cv2.GaussianBlur(rel, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
        est = np.clip(est * corr, 0.0, 1.0)
    return est


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2)[mask].mean())
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def immerkaer_sigma(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    r = cv2.filter2D(g, cv2.CV_64F, k, borderType=cv2.BORDER_REPLICATE)
    return float(np.sqrt(np.pi / 2.0) * np.mean(np.abs(r)) / 6.0)


def run_synthetic_diagnosis():
    results = {}
    for ratio in (2, 4):
        b = synthesize_target_burst(ratio=ratio, lr=256, n=9, seed=7)
        mos, rgb, aligns, mask = b["mosaics"], b["rgb"], b["aligns"], b["mask"]
        target_box, target_lens = b["target"], b["blurred"]
        dems = [_demosaic_bggr(m) for m in mos]
        frames = [RawFrame(str(k), d, d[..., 1].copy()) for k, d in enumerate(dems)]
        w = [1.0 / len(frames)] * len(frames)
        hr_h, hr_w = target_box.shape[:2]

        # 1. Single bicubic
        bic = cv2.resize(frames[0].rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

        # 2. Current SR (estimated shifts)
        sr = SuperResolution(ratio=float(ratio), blur_threshold_ratio=0.0, registration="homography", ecc_scale=0.25)
        res_sr = sr.run(frames, reference_index=0)

        # Measure ECC registration error vs exact shifts (sign-matched)
        est_shifts = np.array(res_sr.shifts, dtype=np.float64)
        true_shifts = b["offsets"]
        err_pos = np.linalg.norm(est_shifts[1:] - true_shifts[1:], axis=1)
        err_neg = np.linalg.norm(est_shifts[1:] + true_shifts[1:], axis=1)
        err = err_pos if err_pos.mean() <= err_neg.mean() else err_neg
        reg_stats = {
            "mean_err_lr_px": round(float(err.mean()), 4),
            "max_err_lr_px": round(float(err.max()), 4),
            "rms_err_lr_px": round(float(np.sqrt(np.mean(err ** 2))), 4),
            "needed_precision_lr_px": round(1.0 / (2.0 * ratio), 4),
        }

        # 3. Drizzle only (exact shifts) at pixfrac = 0.3, 0.5, 1.0
        drz03, _ = drizzle(frames, aligns, float(ratio), w, pixfrac=0.3)
        drz05, cov05 = drizzle(frames, aligns, float(ratio), w, pixfrac=0.5)
        drz10, _ = drizzle(frames, aligns, float(ratio), w, pixfrac=1.0)

        # 4. Current SR with exact shifts (drizzle pf=0.5 + default 1-sweep MAP-IBP)
        sr_exact = iterative_back_projection(
            drz05, frames, aligns, float(ratio), w,
            iterations=1, lam=0.5, reg_sigma=0.5, max_step=0.1, coverage=cov05,
        )

        # 5. Direct Bayer CFA drizzle (exact shifts, pixfrac=0.5)
        cfa_exact = cfa_drizzle(mos, aligns, float(ratio), BGGR, w, pixfrac=0.5).hr

        # 6. Oracle pre-Bayer RGB drizzle (exact shifts, pixfrac=0.5)
        oframes = [RawFrame(str(k), r, r[..., 1].copy()) for k, r in enumerate(rgb)]
        ora_exact, _ = drizzle(oframes, aligns, float(ratio), w, pixfrac=0.5)

        # 7. RL deconvolution (non-neural enhancement) on single bicubic vs current SR
        rl_sigma = 0.45 * float(ratio)
        bic_rl = rl_deconvolve(bic, sigma=rl_sigma, iterations=8)
        sr_rl = rl_deconvolve(res_sr.hr, sigma=rl_sigma, iterations=8)

        methods = {
            "single_bicubic": bic,
            "current_sr_est_shifts": res_sr.hr,
            "current_sr_exact_shifts": sr_exact,
            "drizzle_pf10_exact": drz10,
            "drizzle_pf05_exact": drz05,
            "drizzle_pf03_exact": drz03,
            "cfa_drizzle_pf05_exact": cfa_exact,
            "oracle_rgb_pf05_exact": ora_exact,
            "single_bicubic_plus_rl8": bic_rl,
            "current_sr_plus_rl8": sr_rl,
        }
        table = {}
        for name, img in methods.items():
            table[name] = {
                "mtf50_cyc_per_lr_px": round(compute_slanted_edge_mtf50(img, float(ratio)), 4),
                "psnr_vs_sensor_gt_db": round(psnr(img, target_box, mask), 2),
                "psnr_vs_lens_gt_db": round(psnr(img, target_lens, mask), 2),
                "flat_noise_sigma": round(immerkaer_sigma(img[30:90, 30:90, 1]), 5),
            }
        results[f"{ratio}x"] = {"registration": reg_stats, "methods": table}

        # Save visual Siemens star crop at 4x
        if ratio == 4:
            # Crop around Siemens star centre (0.75*W, 0.5*H), 256x256
            cx, cy = int(hr_w * 0.75), int(hr_h * 0.5)
            sz = 256
            sl = (slice(cy - sz // 2, cy + sz // 2), slice(cx - sz // 2, cx + sz // 2))
            vis_list = [
                ("1. Single bicubic", bic[sl]),
                ("2. Current raw_sr 4x", res_sr.hr[sl]),
                ("3. CFA drizzle (exact)", cfa_exact[sl]),
                ("4. raw_sr + RL(8)", sr_rl[sl]),
            ]
            hdr = 24
            gap = 4
            W_out = 4 * sz + 3 * gap
            H_out = hdr + sz
            canvas = np.full((H_out, W_out, 3), 24, dtype=np.uint8)
            for idx, (lbl, patch) in enumerate(vis_list):
                x0 = idx * (sz + gap)
                cv2.putText(canvas, lbl, (x0 + 6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
                u8 = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
                canvas[hdr:hdr + sz, x0:x0 + sz] = u8
            out_png = os.path.join(ROOT, "docs", "audit-c11", "synthetic_star_4x.png")
            cv2.imwrite(out_png, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])

    return results


# --------------------------------------------------------------------------- #
# Real-burst diagnostics: subpixel diversity, residuals, noise, real MTF50
# --------------------------------------------------------------------------- #
def subpixel_diversity_stats(shifts: list[list[float]]) -> dict:
    arr = np.array(shifts, dtype=np.float64)
    fx = np.mod(arr[:, 0], 1.0)
    fy = np.mod(arr[:, 1], 1.0)
    bins2 = set(zip((fx * 2).astype(int) % 2, (fy * 2).astype(int) % 2))
    bins4 = set(zip((fx * 4).astype(int) % 4, (fy * 4).astype(int) % 4))
    # Mean nearest-neighbour distance on [0,1)^2 torus
    n = len(arr)
    nnd = []
    for i in range(n):
        best = 1e9
        for j in range(n):
            if i == j:
                continue
            dx = abs(fx[i] - fx[j])
            dx = min(dx, 1.0 - dx)
            dy = abs(fy[i] - fy[j])
            dy = min(dy, 1.0 - dy)
            best = min(best, np.hypot(dx, dy))
        nnd.append(best)
    return {
        "n_frames": int(n),
        "bins_2x2_occupied": f"{len(bins2)}/4",
        "bins_4x4_occupied": f"{len(bins4)}/16",
        "mean_torus_nnd": round(float(np.mean(nnd)), 3),
        "max_shift_px": round(float(np.max(np.hypot(arr[:, 0], arr[:, 1]))), 2),
    }


def real_burst_diagnostics():
    out = {}

    # Load HDR+ 4x JSONs from previous runs
    hdr_dir = "output/hdrplus"
    hdr_jsons = {
        "hdrplus_0919": "sr_acceptance_4x_0919_20150910_131720_137.json",
        "hdrplus_0382": "sr_acceptance_4x_0382_20150912_130914_161.json",
        "hdrplus_33TJ_short": "sr_acceptance_4x_33TJ_20150810_140637_256_short.json",
        "hdrplus_6G7M_short": "sr_acceptance_4x_6G7M_20150404_132758_328_short.json",
    }
    for key, fn in hdr_jsons.items():
        p = os.path.join(hdr_dir, fn)
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        out[key] = {
            "subpixel": subpixel_diversity_stats(d["shifts"]),
            "align_residual_mean_px": round(float(d.get("align_residual_mean_px", -1)), 4),
            "align_residual_max_px": round(float(d.get("align_residual_max_px", -1)), 4),
            "achievable_ratio": d.get("achievable_ratio"),
        }

    # Run own9 and 0919 for noise + real-edge MTF50 + own9 subpixel diversity
    for key, bdir, roi in (
        ("own9", "data/own9", (1650, 1240, 1024, 1024)),
        ("hdrplus_0919", "data/hdrplus/0919_20150910_131720_137", None),
    ):
        files = sorted(glob.glob(os.path.join(bdir, "*.dng")))
        if roi is None:
            import rawpy
            with rawpy.imread(files[0]) as raw:
                H, W = raw.sizes.height, raw.sizes.width
            s = 1024
            roi = (((W - s) // 2) & ~1, ((H - s) // 2) & ~1, s, s)
        frames = [load_raw_linear(f, roi=roi) for f in files]
        sr = SuperResolution(
            ratio=4.0, registration="homography", ecc_scale=0.25,
            memory_budget_bytes=4 * 1024**3,
        )
        res = sr.run(frames)
        ref_i = int(res.reference_index)

        # Noise level: single reference frame vs denoised LR stack
        lr_stack = denoised_lr_stack(
            [frames[i] for i in res.kept_indices], res.transforms, res.weights.tolist(),
            coverage_normalised=True,
        )
        sig_single = immerkaer_sigma(frames[ref_i].green[64:-64, 64:-64])
        sig_stack = immerkaer_sigma(lr_stack[64:-64, 64:-64, 1])
        mean_luma = float(frames[ref_i].green[64:-64, 64:-64].mean())

        # Real-edge MTF50: find a strong vertical/near-vertical edge patch in LR
        g_lr = frames[ref_i].green
        gx = np.abs(cv2.Sobel(g_lr, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(g_lr, cv2.CV_32F, 0, 1, ksize=3))
        # Prefer strong horizontal gradient with small vertical gradient (near-vertical edge)
        score_map = gx - 1.5 * gy
        best_val, best_xy = -1.0, (512, 512)
        for y in range(128, 1024 - 128, 32):
            for x in range(128, 1024 - 128, 32):
                sc = float(score_map[y - 16:y + 16, x - 16:x + 16].mean())
                if sc > best_val:
                    best_val = sc
                    best_xy = (x, y)
        ex, ey = best_xy
        # HR ROI around that edge (128x128 HR px = 32x32 LR px)
        edge_roi_hr = (ex * 4 - 64, ey * 4 - 64, 128, 128)
        bic = cv2.resize(frames[ref_i].rgb, (res.hr.shape[1], res.hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        mtf_bic = compute_slanted_edge_mtf50(bic, 4.0, roi=edge_roi_hr)
        mtf_sr = compute_slanted_edge_mtf50(res.hr, 4.0, roi=edge_roi_hr)

        entry = out.setdefault(key, {})
        if key == "own9":
            entry["subpixel"] = subpixel_diversity_stats([[float(a), float(b)] for a, b in res.shifts])
            entry["align_residual_mean_px"] = round(float(res.metrics.get("align_residual_mean_px", -1)), 4)
            entry["align_residual_max_px"] = round(float(res.metrics.get("align_residual_max_px", -1)), 4)
        entry["noise"] = {
            "linear_mean_green": round(mean_luma, 5),
            "single_frame_sigma": round(sig_single, 6),
            "stack_sigma": round(sig_stack, 6),
            "noise_reduction_factor": round(sig_single / max(sig_stack, 1e-12), 2),
            "single_snr_db": round(float(20 * np.log10(mean_luma / max(sig_single, 1e-12))), 1),
            "stack_snr_db": round(float(20 * np.log10(mean_luma / max(sig_stack, 1e-12))), 1),
        }
        entry["real_edge_mtf50"] = {
            "edge_roi_hr": list(edge_roi_hr),
            "single_bicubic_cyc_per_lr_px": round(mtf_bic, 4),
            "current_sr_cyc_per_lr_px": round(mtf_sr, 4),
            "gain_pct": round((mtf_sr - mtf_bic) / max(mtf_bic, 1e-9) * 100.0, 1),
        }

    return out


def main():
    syn = run_synthetic_diagnosis()
    print("=== Synthetic Resolution Target Diagnosis ===")
    print(json.dumps(syn, indent=2))

    real = real_burst_diagnostics()
    print("=== Real Burst Diagnostics ===")
    print(json.dumps(real, indent=2))

    with open("/tmp/rawsr/c11_part2.json", "w") as fh:
        json.dump({"synthetic": syn, "real": real}, fh, indent=2)
    print("Wrote /tmp/rawsr/c11_part2.json")


if __name__ == "__main__":
    main()
