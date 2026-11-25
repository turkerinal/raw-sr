#!/usr/bin/env python3
"""WP2 Evaluation: Synthetic GT (2x/4x), Real Bursts (own9, HDR+ 0919, 0382),
4-panel visual crops under docs/wp2/, and 4096x4096 runtime/memory benchmark.
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

from raw_sr.core.enhance import (
    DEFAULT_STRENGTH,
    effective_stack_noise,
    enhance_fair_pair,
    enhance_image,
)
from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    load_raw_linear,
)
from scripts.audit_c11_part2 import (
    _demosaic_bggr,
    compute_slanted_edge_mtf50,
    immerkaer_sigma,
    psnr,
    synthesize_target_burst,
)


def measure_edge_overshoot(img: np.ndarray, lo: float = 0.15, hi: float = 0.75) -> float:
    """Measure max overshoot/undershoot near the synthetic slanted edge as fraction of step."""
    gray = img[..., 1].astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    H, W = gray.shape
    cx = int(W * 0.25)
    hw = int(W * 0.08)
    patch = gray[int(H * 0.2):int(H * 0.8), max(0, cx - hw):min(W, cx + hw)]
    step = hi - lo
    over = max(float(patch.max()) - hi, lo - float(patch.min()), 0.0)
    return over / step


def auto_expose_uint8(img: np.ndarray, exposure: float, gamma: float = 2.2) -> np.ndarray:
    v = np.clip(img.astype(np.float32) * exposure, 0.0, 1.0)
    v = np.power(v, 1.0 / gamma)
    return np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)


def compute_safe_exposure(img: np.ndarray, target_p995: float = 0.92) -> float:
    """Choose exposure so that < 1% of pixels clip."""
    p995 = float(np.percentile(img, 99.5))
    if p995 <= 1e-6:
        return 1.0
    return float(target_p995 / p995)


def make_4panel(
    panels: list[tuple[str, np.ndarray]],
    exposure: float,
    out_path: str,
) -> dict:
    """Render 4-panel horizontal strip with identical display transform."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rendered = []
    clip_fracs = []
    for label, arr in panels:
        u8 = auto_expose_uint8(arr, exposure=exposure)
        clip_frac = float(np.mean(np.any(u8 >= 255, axis=2)))
        clip_fracs.append(clip_frac)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        # Add clean top banner
        bar = np.full((28, bgr.shape[1], 3), 20, dtype=np.uint8)
        cv2.putText(
            bar, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA
        )
        rendered.append(np.vstack([bar, bgr]))

    sep = np.full((rendered[0].shape[0], 4, 3), 40, dtype=np.uint8)
    strip = rendered[0]
    for p in rendered[1:]:
        strip = np.hstack([strip, sep, p])

    cv2.imwrite(out_path, strip, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    size_kb = os.path.getsize(out_path) / 1024.0
    return {
        "path": out_path,
        "size_kb": round(size_kb, 1),
        "exposure": round(exposure, 3),
        "max_clip_pct": round(max(clip_fracs) * 100.0, 3),
    }


def eval_synthetic(strength: float = DEFAULT_STRENGTH) -> dict:
    out = {}
    for ratio in (2, 4):
        b = synthesize_target_burst(ratio=ratio, lr=256, n=9, seed=7)
        mos, mask = b["mosaics"], b["mask"]
        target_box, target_lens = b["target"], b["blurred"]
        dems = [_demosaic_bggr(m) for m in mos]
        frames = [RawFrame(str(k), d, d[..., 1].copy()) for k, d in enumerate(dems)]
        hr_h, hr_w = target_box.shape[:2]

        bic = cv2.resize(frames[0].rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        sr = SuperResolution(
            ratio=float(ratio), blur_threshold_ratio=0.0, registration="homography", ecc_scale=0.25
        )
        res_sr = sr.run(frames, reference_index=0)
        sr_hr = res_sr.hr

        sr_noise = effective_stack_noise(
            res_sr.metrics.get("frame_noise_sigma"), effective_frames=res_sr.used_frames
        )
        enh_sr, enh_bic = enhance_fair_pair(
            sr_hr, bic, ratio=float(ratio), sr_noise_sigma=sr_noise, strength=strength
        )

        # Flat region on left of slanted edge
        flat_slice = (slice(int(hr_h * 0.2), int(hr_h * 0.8)), slice(int(hr_w * 0.05), int(hr_w * 0.14)))

        def metrics(img):
            return {
                "mtf50": round(compute_slanted_edge_mtf50(img, ratio=float(ratio)), 4),
                "psnr_sensor": round(psnr(img, target_box, mask), 3),
                "psnr_lens": round(psnr(img, target_lens, mask), 3),
                "overshoot_pct": round(measure_edge_overshoot(img) * 100.0, 2),
                "flat_sigma": round(float(img[flat_slice][..., 1].std()), 6),
            }

        m_bic = metrics(bic)
        m_bic_e = metrics(enh_bic.image)
        m_sr = metrics(sr_hr)
        m_sr_e = metrics(enh_sr.image)
        mtf_gain_pct = (m_sr_e["mtf50"] / max(m_sr["mtf50"], 1e-6) - 1.0) * 100.0

        out[f"{ratio}x"] = {
            "single_bicubic": m_bic,
            "single_enhanced": m_bic_e,
            "plain_sr": m_sr,
            "enhanced_sr": m_sr_e,
            "mtf50_gain_vs_plain_sr_pct": round(mtf_gain_pct, 1),
            "sr_optics_sigma_est": round(enh_sr.optics_sigma_lr, 3),
            "sr_rl_iters": enh_sr.rl_iterations_run,
            "sr_rl_stop": enh_sr.rl_stopped_reason,
            "single_rl_iters": enh_bic.rl_iterations_run,
            "single_rl_stop": enh_bic.rl_stopped_reason,
        }
    return out


def _find_best_edge_roi(gray: np.ndarray, win: int = 160) -> tuple[int, int, int, int]:
    """Find a high-contrast vertical-ish edge patch for slanted-edge MTF50."""
    H, W = gray.shape
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    # Prefer vertical edges (|gx| >> |gy|)
    score = np.maximum(gx - 0.5 * gy, 0.0)
    margin = win
    best = (0.0, W // 2 - win // 2, H // 2 - win // 2)
    step = max(32, win // 4)
    for y in range(margin, H - margin - win, step):
        for x in range(margin, W - margin - win, step):
            s = float(score[y:y + win, x:x + win].mean())
            if s > best[0]:
                best = (s, x, y)
    return (best[1], best[2], win, win)


def _find_flat_patch(gray: np.ndarray, win: int = 96) -> tuple[int, int, int, int]:
    """Find the quietest interior patch (lowest local gradient) for flat noise sigma."""
    H, W = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    margin = win
    best = (float("inf"), W // 2, H // 2)
    step = max(32, win // 2)
    for y in range(margin, H - margin - win, step):
        for x in range(margin, W - margin - win, step):
            patch = gray[y:y + win, x:x + win]
            if float(patch.mean()) < 0.02 or float(patch.max()) > 0.95:
                continue
            s = float(gmag[y:y + win, x:x + win].mean())
            if s < best[0]:
                best = (s, x, y)
    return (best[1], best[2], win, win)


def eval_real_burst(
    name: str,
    paths: list[str],
    roi: tuple[int, int, int, int] | None,
    crop_size: int = 384,
    strength: float = DEFAULT_STRENGTH,
) -> dict:
    frames = []
    for p in paths:
        if roi is None:
            # Load first to get dimensions for centred 1024 ROI
            full = load_raw_linear(p)
            H, W = full.rgb.shape[:2]
            sz = min(1024, H, W)
            roi = ((W - sz) // 2, (H - sz) // 2, sz, sz)
            frames.append(
                RawFrame(
                    p,
                    full.rgb[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]].copy(),
                    full.green[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]].copy(),
                    full.meta,
                )
            )
        else:
            frames.append(load_raw_linear(p, roi=roi))

    ratio = 4.0
    sr = SuperResolution(ratio=ratio, registration="homography", ecc_scale=0.25, max_hr_pixels=20_000_000)
    res = sr.run(frames)
    ref_frame = frames[res.reference_index]
    hr_h, hr_w = res.hr.shape[:2]
    bic = cv2.resize(ref_frame.rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

    sr_noise = effective_stack_noise(
        res.metrics.get("frame_noise_sigma"), effective_frames=res.used_frames
    )
    enh_sr, enh_bic = enhance_fair_pair(
        res.hr, bic, ratio=ratio, sr_noise_sigma=sr_noise, strength=strength
    )

    # Edge MTF50 & flat noise on the exact same patches for all 4 images
    edge_roi = _find_best_edge_roi(res.hr[..., 1], win=192)
    flat_roi = _find_flat_patch(res.hr[..., 1], win=128)
    fx, fy, fw, fh = flat_roi

    def _m(img):
        mtf = compute_slanted_edge_mtf50(img, ratio=ratio, roi=edge_roi)
        flat_sig = immerkaer_sigma(img[fy:fy + fh, fx:fx + fw, 1])
        return {"mtf50": round(mtf, 4), "flat_sigma": round(flat_sig, 6)}

    # Pick high-detail crop centred around edge_roi for visual 4-panel
    ex, ey, ew, eh = edge_roi
    cx = int(np.clip(ex + ew // 2, crop_size // 2, hr_w - crop_size // 2))
    cy = int(np.clip(ey + eh // 2, crop_size // 2, hr_h - crop_size // 2))
    y0, x0 = cy - crop_size // 2, cx - crop_size // 2

    crops = [
        ("Single Bicubic 4x", bic[y0:y0 + crop_size, x0:x0 + crop_size]),
        ("Single + Enhance", enh_bic.image[y0:y0 + crop_size, x0:x0 + crop_size]),
        ("Plain SR 4x", res.hr[y0:y0 + crop_size, x0:x0 + crop_size]),
        ("Enhanced SR 4x", enh_sr.image[y0:y0 + crop_size, x0:x0 + crop_size]),
    ]
    exp = compute_safe_exposure(crops[2][1])
    vis_info = make_4panel(crops, exposure=exp, out_path=os.path.join(ROOT, "docs", "wp2", f"{name}_4x_4panel.png"))

    return {
        "used_frames": res.used_frames,
        "total_frames": res.total_frames,
        "roi": list(roi),
        "single_bicubic": _m(bic),
        "single_enhanced": _m(enh_bic.image),
        "plain_sr": _m(res.hr),
        "enhanced_sr": _m(enh_sr.image),
        "sr_optics_sigma_est": round(enh_sr.optics_sigma_lr, 3),
        "sr_rl_iters": enh_sr.rl_iterations_run,
        "single_rl_iters": enh_bic.rl_iterations_run,
        "visual": vis_info,
    }


def eval_benchmark_4096(strength: float = DEFAULT_STRENGTH) -> dict:
    """Measure runtime and peak extra RSS on a 4096x4096 float32 RGB image."""
    rng = np.random.default_rng(0)
    img = rng.uniform(0.1, 0.8, (4096, 4096, 3)).astype(np.float32)
    rss_before_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    t0 = time.perf_counter()
    res = enhance_image(img, ratio=4.0, noise_sigma=0.001, strength=strength)
    dt = time.perf_counter() - t0
    rss_after_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "shape": list(img.shape),
        "seconds": round(dt, 2),
        "rss_before_mib": round(rss_before_mb, 1),
        "rss_after_mib": round(rss_after_mb, 1),
        "rss_delta_mib": round(max(rss_after_mb - rss_before_mb, 0.0), 1),
        "rl_iters": res.rl_iterations_run,
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    strength = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_STRENGTH
    report = {"strength": strength}

    if mode in ("all", "synth"):
        report["synthetic"] = eval_synthetic(strength=strength)

    if mode in ("all", "real"):
        data_root = "data"
        own9 = sorted(glob.glob(os.path.join(data_root, "own9", "*.DNG")) + glob.glob(os.path.join(data_root, "own9", "*.CR2")) + glob.glob(os.path.join(data_root, "own9", "*.NEF")) + glob.glob(os.path.join(data_root, "own9", "*.ARW")) + glob.glob(os.path.join(data_root, "own9", "*.dng")))
        hdr0919 = sorted(glob.glob(os.path.join(data_root, "hdrplus", "0919*", "*.dng")))
        hdr0382 = sorted(glob.glob(os.path.join(data_root, "hdrplus", "0382*", "*.dng")))
        report["real"] = {}
        if own9:
            report["real"]["own9"] = eval_real_burst("own9", own9, roi=(1650, 1240, 1024, 1024), strength=strength)
        if hdr0919:
            report["real"]["hdrplus_0919"] = eval_real_burst("hdrplus_0919", hdr0919, roi=None, strength=strength)
        if hdr0382:
            report["real"]["hdrplus_0382"] = eval_real_burst("hdrplus_0382", hdr0382, roi=None, strength=strength)

    if mode in ("all", "bench"):
        report["bench_4096"] = eval_benchmark_4096(strength=strength)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
