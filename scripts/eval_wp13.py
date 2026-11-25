#!/usr/bin/env python3
"""WP13 Shared Evaluation Protocol for Forward-Model MAP Super-Resolution.

Modes:
  synth : Protocol A — realistic synthetic GT at sigma_lens_lr in {0.35, 0.6, 0.9},
          n in {9, 20}, seed in {7, 23}.
  chart : Protocol B — simulated test chart at sigma_lens_lr in {0.35, 0.7}.
  real  : Protocol C — real bursts (own9, hdrplus_0919, hdrplus_0382, samsung14,
          plus internal pixel4a20): showcase panels, truth-free crop metrics,
          colour-fair CFA held-out, and DCB held-out.
  bench : Protocol D — runtime and peak RAM/VRAM at 2x 1024^2 x 9 (GPU & CPU).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import resource
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))

from audit_c11_part2 import (
    BGGR,
    _demosaic_bggr,
    compute_slanted_edge_mtf50,
    immerkaer_sigma,
    psnr,
    synthesize_target_burst,
)
from make_showcase import (
    choose_safe_exposure,
    find_flat_patch,
    measure_phase_shift,
    pick_high_detail_crop,
    pick_zoom_subcrop,
    render_showcase,
)
from raw_sr.core.cfa_drizzle import pattern_array
from raw_sr.core.cfa_heldout import evaluate_cfa_heldout
from raw_sr.core.enhance import (
    effective_stack_noise,
    enhance_fair_pair,
)
from raw_sr.core.map_sr import estimate_colour_matrix
from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    denoised_lr_stack,
    estimate_homography,
    load_raw_linear,
    lr_sampling_valid_mask,
    predict_lr_from_hr,
)


# --------------------------------------------------------------------------- #
# Shared metrics
# --------------------------------------------------------------------------- #

def compute_ssim(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Global mean SSIM on luminance/green (Wang et al. 2004, Gaussian window)."""
    ga = a[..., 1].astype(np.float64) if a.ndim == 3 else a.astype(np.float64)
    gb = b[..., 1].astype(np.float64) if b.ndim == 3 else b.astype(np.float64)
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    mu_a = cv2.GaussianBlur(ga, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    mu_b = cv2.GaussianBlur(gb, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sig_a2 = cv2.GaussianBlur(ga * ga, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT) - mu_a2
    sig_b2 = cv2.GaussianBlur(gb * gb, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT) - mu_b2
    sig_ab = cv2.GaussianBlur(ga * gb, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT) - mu_ab
    num = (2.0 * mu_ab + c1) * (2.0 * sig_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
    smap = num / np.maximum(den, 1e-12)
    if mask is not None:
        return float(np.mean(smap[mask]))
    return float(np.mean(smap))


def measure_gradient_energy(crop: np.ndarray) -> float:
    """Mean |dx| + |dy| of the green channel."""
    g = crop[..., 1].astype(np.float64)
    return float(np.mean(np.abs(np.diff(g, axis=0))) + np.mean(np.abs(np.diff(g, axis=1))))


def measure_nyquist_energy_frac(crop: np.ndarray) -> float:
    """Period-2 (Nyquist) energy fraction of the green FFT (row + col at N/2)."""
    g = crop[..., 1].astype(np.float64)
    g = g - g.mean()
    F = np.abs(np.fft.fft2(g))
    total = float(F.sum())
    if total <= 1e-12:
        return 0.0
    ny = g.shape[0] // 2
    nx = g.shape[1] // 2
    nyq = float(F[ny, :].sum() + F[:, nx].sum() - F[ny, nx])
    return nyq / total


def measure_crop_overshoot(crop: np.ndarray, ref_crop: Optional[np.ndarray] = None) -> float:
    """Max over strong edge patches of (peak - plateau) / step in green channel.

    Uses ``ref_crop`` (if provided) to locate identical strong edge patches across
    methods, then measures local overshoot on ``crop`` in each patch.
    """
    src = ref_crop[..., 1].astype(np.float64) if ref_crop is not None else crop[..., 1].astype(np.float64)
    g = crop[..., 1].astype(np.float64)
    gx = cv2.Sobel(src.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    h, w = g.shape
    pr = 8
    if h < 2 * pr + 4 or w < 2 * pr + 4:
        return 0.0
    p95 = float(np.percentile(gmag[pr:-pr, pr:-pr], 95))
    if p95 < 1e-4:
        return 0.0

    # Sample up to 16 strong edge centres on a coarse grid
    step = max(16, min(h, w) // 8)
    overshoots: List[float] = []
    for y0 in range(pr, h - pr, step):
        for x0 in range(pr, w - pr, step):
            y1 = min(y0 + step, h - pr)
            x1 = min(x0 + step, w - pr)
            sub = gmag[y0:y1, x0:x1]
            idx = int(np.argmax(sub))
            if float(sub.ravel()[idx]) < 0.7 * p95:
                continue
            ly, lx = divmod(idx, sub.shape[1])
            cy, cx = y0 + ly, x0 + lx
            patch = g[cy - pr:cy + pr + 1, cx - pr:cx + pr + 1]
            p10 = float(np.percentile(patch, 10))
            p90 = float(np.percentile(patch, 90))
            s = p90 - p10
            if s < 0.02:
                continue
            ov_hi = max(0.0, float(patch.max()) - p90) / s
            ov_lo = max(0.0, p10 - float(patch.min())) / s
            overshoots.append(max(ov_hi, ov_lo))
    if not overshoots:
        return 0.0
    return float(max(overshoots))


def measure_colour_drift_pct(crop: np.ndarray, ref_crop: np.ndarray) -> Tuple[float, List[float]]:
    """Max relative per-channel mean colour drift (%) vs reference crop."""
    m = crop.astype(np.float64).mean(axis=(0, 1))
    m_ref = ref_crop.astype(np.float64).mean(axis=(0, 1))
    rel = np.abs(m - m_ref) / np.maximum(m_ref, 1e-6) * 100.0
    return float(np.max(rel)), [round(float(v), 5) for v in m]


def _dcb_heldout_gain_db(
    hr: np.ndarray,
    held_frame: RawFrame,
    kept_frames: List[RawFrame],
    transforms: list,
    weights: List[float],
    W_held: np.ndarray,
    ratio: float,
    margin_lr: int = 24,
) -> float:
    lr_shape = held_frame.rgb.shape[:2]
    pred = predict_lr_from_hr(hr, W_held, ratio, lr_shape)
    stack = denoised_lr_stack(kept_frames, transforms, weights)
    base = cv2.warpPerspective(
        stack, W_held.astype(np.float32), (lr_shape[1], lr_shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    mask = lr_sampling_valid_mask(hr.shape[:2], W_held, ratio, lr_shape)
    m = np.zeros(lr_shape, dtype=bool)
    m[margin_lr:-margin_lr, margin_lr:-margin_lr] = True
    mask = mask & m
    mse_sr = float(np.mean((pred[mask] - held_frame.rgb[mask]) ** 2))
    mse_base = float(np.mean((base[mask] - held_frame.rgb[mask]) ** 2))
    return float(10.0 * np.log10(max(mse_base, 1e-12) / max(mse_sr, 1e-12)))


# --------------------------------------------------------------------------- #
# Method runner
# --------------------------------------------------------------------------- #

def _run_methods(
    frames: List[RawFrame],
    ratio: float = 2.0,
    ref_idx: Optional[int] = 0,
    ecc_scale: float = 0.25,
    include_wp13: bool = True,
    use_gpu: Optional[bool] = None,
    map_wp13_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict], Any]:
    """Run all 5 protocol methods on a burst."""
    hrs: Dict[str, np.ndarray] = {}
    meta: Dict[str, dict] = {}

    t0 = time.perf_counter()
    rd = SuperResolution(
        ratio=ratio,
        blur_threshold_ratio=0.0,
        registration="homography",
        ecc_scale=ecc_scale,
        fusion="demosaic_drizzle",
        use_gpu=use_gpu,
    ).run(frames, reference_index=ref_idx)
    t_def = time.perf_counter() - t0
    ref_used = int(rd.reference_index)
    hr_h, hr_w = rd.hr.shape[:2]

    bic = cv2.resize(frames[ref_used].rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    hrs["single_bicubic"] = bic
    meta["single_bicubic"] = {"sec": 0.0}

    hrs["old_default"] = rd.hr
    meta["old_default"] = {"sec": round(t_def, 3)}

    t0 = time.perf_counter()
    sr_noise = effective_stack_noise(
        rd.metrics.get("frame_noise_sigma"), effective_frames=rd.used_frames
    )
    enh_sr, _ = enhance_fair_pair(
        rd.hr, bic, ratio=ratio, sr_noise_sigma=sr_noise, strength=0.65
    )
    hrs["old_enh"] = enh_sr.image
    meta["old_enh"] = {
        "sec": round(t_def + (time.perf_counter() - t0), 3),
        "optics_sigma_lr": round(float(enh_sr.optics_sigma_lr), 4),
    }

    t0 = time.perf_counter()
    rm_cur = SuperResolution(
        ratio=ratio,
        blur_threshold_ratio=0.0,
        registration="homography",
        ecc_scale=ecc_scale,
        fusion="map",
        use_gpu=use_gpu,
    ).run(frames, reference_index=ref_used)
    hrs["map_current"] = rm_cur.hr
    meta["map_current"] = {
        "sec": round(time.perf_counter() - t0, 3),
        "engine": rm_cur.metrics.get("cfa_kr_engine"),
        "iters": rm_cur.metrics.get("cfa_kr_iterations"),
        "stop": rm_cur.metrics.get("cfa_kr_stop_reason"),
    }

    if include_wp13:
        t0 = time.perf_counter()
        kw = dict(map_wp13_kwargs or {})
        rm_new = SuperResolution(
            ratio=ratio,
            blur_threshold_ratio=0.0,
            registration="homography",
            ecc_scale=ecc_scale,
            fusion="map",
            map_mode="wp13",
            use_gpu=use_gpu,
            **kw,
        ).run(frames, reference_index=ref_used)
        hrs["map_wp13"] = rm_new.hr
        meta["map_wp13"] = {
            "sec": round(time.perf_counter() - t0, 3),
            "engine": rm_new.metrics.get("cfa_kr_engine"),
            "iters": rm_new.metrics.get("cfa_kr_iterations"),
            "stop": rm_new.metrics.get("cfa_kr_stop_reason"),
            "psf_rgb": rm_new.metrics.get("cfa_kr_psf_sigma_rgb"),
            "psf_source": rm_new.metrics.get("cfa_kr_psf_source"),
        }

    return hrs, meta, rd


# --------------------------------------------------------------------------- #
# Protocol A: Realistic synthetic GT
# --------------------------------------------------------------------------- #

def run_synth(
    out_dir: str,
    sigmas: Tuple[float, ...] = (0.35, 0.6, 0.9),
    ns: Tuple[int, ...] = (9, 20),
    seeds: Tuple[int, ...] = (7, 23),
    include_wp13: bool = True,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, Any] = {}
    for sig in sigmas:
        for n in ns:
            for seed in seeds:
                key = f"s{sig:.2f}_n{n}_seed{seed}"
                b = synthesize_target_burst(ratio=2, lr=256, n=n, seed=seed, sigma_lens_lr=sig)
                mos = b["mosaics"]
                dems = [_demosaic_bggr(m) for m in mos]
                frames = [
                    RawFrame(str(k), d, d[..., 1].copy(), mosaic=m, pattern=BGGR)
                    for k, (d, m) in enumerate(zip(dems, mos))
                ]
                hrs, meta, _ = _run_methods(frames, ratio=2.0, ref_idx=0, ecc_scale=0.25, include_wp13=include_wp13)
                row = {}
                for mname, img in hrs.items():
                    row[mname] = {
                        "psnr_scene": round(float(psnr(img, b["scene"], b["mask"])), 2),
                        "ssim_scene": round(compute_ssim(img, b["scene"], b["mask"]), 4),
                        "psnr_target": round(float(psnr(img, b["target"], b["mask"])), 2),
                        "psnr_blurred": round(float(psnr(img, b["blurred"], b["mask"])), 2),
                        "mtf50": round(float(compute_slanted_edge_mtf50(img, 2.0)), 4),
                        "flat_sigma": round(float(immerkaer_sigma(img[30:90, 30:90, 1])), 6),
                        **meta[mname],
                    }
                results[key] = row
                summary_str = " | ".join(
                    f"{k}:{v['psnr_scene']:.2f}({v['mtf50']:.3f})"
                    for k, v in row.items()
                )
                print(f"[SYNTH {key}] {summary_str}", flush=True)

    # Aggregate mean over (n, seed) per sigma
    agg: Dict[str, Any] = {}
    for sig in sigmas:
        skey = f"s{sig:.2f}"
        matching = [v for k, v in results.items() if k.startswith(skey)]
        methods = list(matching[0].keys())
        agg[skey] = {}
        for m in methods:
            agg[skey][m] = {
                "psnr_scene": round(float(np.mean([r[m]["psnr_scene"] for r in matching])), 2),
                "ssim_scene": round(float(np.mean([r[m]["ssim_scene"] for r in matching])), 4),
                "psnr_target": round(float(np.mean([r[m]["psnr_target"] for r in matching])), 2),
                "mtf50": round(float(np.mean([r[m]["mtf50"] for r in matching])), 4),
                "flat_sigma": round(float(np.mean([r[m]["flat_sigma"] for r in matching])), 6),
            }
        print(f"=== SYNTH MEAN {skey} ===", json.dumps({m: agg[skey][m]["psnr_scene"] for m in methods}), flush=True)

    out = {"per_run": results, "mean_by_sigma": agg}
    with open(os.path.join(out_dir, "synth_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


# --------------------------------------------------------------------------- #
# Protocol B: Simulated test chart
# --------------------------------------------------------------------------- #

def run_chart(
    out_dir: str,
    sigmas: Tuple[float, ...] = (0.35, 0.7),
    include_wp13: bool = True,
) -> dict:
    from scripts.testchart.make_chart import render_chart_raster
    from scripts.testchart.measure_chart import measure_triplet
    from scripts.testchart.validate_sim import (
        _build_perspective_H,
        _demosaic_bggr as _chart_dem,
        _mosaic_from_rgb,
        analytical_mtf_crossing,
    )

    os.makedirs(out_dir, exist_ok=True)
    chart_u8 = render_chart_raster(paper="A4", dpi=200)
    chart_f = chart_u8.astype(np.float32) / 255.0
    chart_rgb = np.stack([chart_f, chart_f, chart_f], axis=2)
    ch_h, ch_w = chart_rgb.shape[:2]

    ratio = 2
    lr_h = 680
    n_frames = 16
    all_out: Dict[str, Any] = {}

    for sig in sigmas:
        rng = np.random.default_rng(42 + int(round(sig * 100)))
        lr_w = int(round(lr_h * (297.0 / 210.0)))
        lr_h_e = lr_h - (lr_h % 2)
        lr_w_e = lr_w - (lr_w % 2)
        hr_h, hr_w = lr_h_e * ratio, lr_w_e * ratio

        H_base = _build_perspective_H(ch_w, ch_h, hr_w, hr_h, margin=0.07, tilt_px=8.0)
        scene_hr = cv2.warpPerspective(
            chart_rgb, H_base, (hr_w, hr_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        blurred_hr = cv2.GaussianBlur(scene_hr, (0, 0), sig * ratio, borderType=cv2.BORDER_REFLECT)
        box_hr = cv2.blur(blurred_hr, (ratio, ratio), borderType=cv2.BORDER_REFLECT)

        offs = np.stack([rng.uniform(-2.5, 2.5, n_frames), rng.uniform(-2.5, 2.5, n_frames)], axis=1)
        offs[0] = (0.0, 0.0)
        xs, ys = np.meshgrid(np.arange(lr_w_e, dtype=np.float32), np.arange(lr_h_e, dtype=np.float32))
        off = 0.5 * (float(ratio) - 1.0)
        full_well, read_noise = 2048.0, 1.5

        raw_frames: List[RawFrame] = []
        for k in range(n_frames):
            mx = (ratio * (xs + np.float32(offs[k, 0])) + off).astype(np.float32)
            my = (ratio * (ys + np.float32(offs[k, 1])) + off).astype(np.float32)
            rgb = cv2.remap(box_hr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            m = _mosaic_from_rgb(rgb)
            m = (rng.poisson(np.clip(m, 0.0, 1.0) * full_well) + rng.normal(0.0, read_noise, m.shape)) / full_well
            m = np.clip(m, 0.0, 1.0).astype(np.float32)
            dem = _chart_dem(m)
            raw_frames.append(RawFrame(f"sim_{k:02d}.dng", dem, dem[..., 1].copy(), mosaic=m, pattern=BGGR.copy()))

        single_img = cv2.resize(raw_frames[0].rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        gt_blurred = cv2.GaussianBlur(scene_hr, (0, 0), sig, borderType=cv2.BORDER_REFLECT)
        gt_stack = []
        for _ in range(3):
            m_gt = _mosaic_from_rgb(gt_blurred)
            m_gt = (rng.poisson(np.clip(m_gt, 0.0, 1.0) * full_well) + rng.normal(0.0, read_noise, m_gt.shape)) / full_well
            gt_stack.append(_chart_dem(np.clip(m_gt, 0.0, 1.0).astype(np.float32)))
        gt_img = np.median(np.stack(gt_stack, axis=0), axis=0).astype(np.float32)

        hrs, meta, _ = _run_methods(raw_frames, ratio=2.0, ref_idx=0, ecc_scale=0.25, include_wp13=include_wp13)
        skey = f"s{sig:.2f}"
        out_s: Dict[str, Any] = {
            "analytical_f50": round(analytical_mtf_crossing(sig, 0.50), 4),
            "analytical_f10": round(analytical_mtf_crossing(sig, 0.10), 4),
        }
        for name, hr_img in hrs.items():
            rep = measure_triplet(single_img=single_img, sr_img=hr_img.astype(np.float32), ratio=2.0, gt_img=gt_img)
            out_s[name] = {
                "star_f50": round(float(rep["chart_only"]["sr"]["star_sin"]["f50_cyc_lr_px"]), 4),
                "star_f10": round(float(rep["chart_only"]["sr"]["star_sin"]["f10_cyc_lr_px"]), 4),
                "edge_mtf50": round(float(rep["chart_only"]["sr"]["edge_high"]["mtf50_cyc_lr_px"]), 4),
                "zoom_gt_psnr": round(float(rep["zoom_gt"]["sr_vs_gt"]["overall"]["psnr_db"]), 2),
                "zoom_gt_ssim": round(float(rep["zoom_gt"]["sr_vs_gt"]["overall"]["ssim"]), 4),
                "sec": meta[name]["sec"],
            }
        all_out[skey] = out_s
        print(
            f"=== CHART {skey} ===",
            json.dumps({k: v["zoom_gt_psnr"] for k, v in out_s.items() if isinstance(v, dict)}),
            flush=True,
        )

    with open(os.path.join(out_dir, "chart_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2)
    return all_out


# --------------------------------------------------------------------------- #
# Protocol C: Real bursts
# --------------------------------------------------------------------------- #

BURSTS: Dict[str, Tuple[str, Tuple[int, int], Optional[Tuple[int, int]], bool]] = {
    # name: (dir, (roi_x, roi_y), fixed_crop_xy or None, publishable)
    "own9": (os.path.join(ROOT, "test_data"), (1650, 1240), (96, 1576), True),
    "hdrplus_0919": ("data/hdrplus/0919_20150910_131720_137", (1504, 1000), (1056, 216), True),
    "hdrplus_0382": ("data/hdrplus/0382_20150912_130914_161", (1504, 1000), (456, 776), True),
    "samsung14": ("data/samsung14", (2400, 520), None, True),
    "pixel4a20": ("data/inria_friant20", (2080, 800), None, False),
}


def run_real(
    burst_names: List[str],
    out_dir: str,
    docs_panel_dir: str,
    n_ho: int = 3,
    skip_heldout: bool = False,
    include_wp13: bool = True,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(docs_panel_dir, exist_ok=True)
    summary_all: Dict[str, Any] = {}
    cs = 320

    for name in burst_names:
        bdir, (rx, ry), fixed_cxy, publishable = BURSTS[name]
        files = sorted(glob.glob(os.path.join(bdir, "*.dng")) + glob.glob(os.path.join(bdir, "*.DNG")))
        roi = (rx, ry, 1024, 1024)
        frames = [load_raw_linear(f, roi=roi) for f in files]
        mos = [f.mosaic for f in frames]
        pat = pattern_array(frames[0].pattern)
        n = len(frames)

        # Full-burst run (let SR select its own reference frame, then share it)
        hrs, meta, rd = _run_methods(frames, ratio=2.0, ref_idx=None, ecc_scale=0.25, include_wp13=include_wp13)
        ref_i = int(rd.reference_index)

        if fixed_cxy is not None:
            x0, y0 = fixed_cxy
        else:
            x0, y0 = pick_high_detail_crop(hrs["old_default"], bic=hrs["single_bicubic"], crop_size=cs, margin=96)
        sl = (slice(y0, y0 + cs), slice(x0, x0 + cs))

        # Quietest flat patch on old_default
        fx, fy, fw, fh = find_flat_patch(hrs["old_default"][..., 1], win=96)
        flat_sl = (slice(fy, fy + fh), slice(fx, fx + fw))

        # Truth-free metrics on the showcase crop
        ref_crop_old = hrs["old_default"][sl]
        ref_crop_enh = hrs["old_enh"][sl]
        crop_metrics: Dict[str, Any] = {}
        for mname, img in hrs.items():
            crop = img[sl]
            drift_old, mean_rgb = measure_colour_drift_pct(crop, ref_crop_old)
            drift_enh, _ = measure_colour_drift_pct(crop, ref_crop_enh)
            crop_metrics[mname] = {
                "grad_energy": round(measure_gradient_energy(crop), 5),
                "flat_immerkaer": round(float(immerkaer_sigma(img[flat_sl][..., 1])), 6),
                "crop_immerkaer": round(float(immerkaer_sigma(crop[..., 1])), 6),
                "nyquist_frac": round(measure_nyquist_energy_frac(crop), 5),
                "overshoot": round(measure_crop_overshoot(crop, ref_crop=ref_crop_old), 4),
                "colour_drift_vs_old_pct": round(drift_old, 3),
                "colour_drift_vs_enh_pct": round(drift_enh, 3),
                "mean_rgb": mean_rgb,
                **meta[mname],
            }

        # Render 4-panel showcase: single | old+enh | current MAP | WP13 MAP
        panel_keys = ["single_bicubic", "old_enh", "map_current", "map_wp13" if include_wp13 else "old_default"]
        panel_labels = [
            "1. Single (bicubic)",
            "2. Old default + enhance",
            "3. Current MAP",
            "4. WP13 MAP" if include_wp13 else "4. Old default",
        ]
        crops = [hrs[k][sl] for k in panel_keys]
        expo, _ = choose_safe_exposure(crops)
        sub_size = cs // 2
        sub_xy = pick_zoom_subcrop(hrs["old_enh"][sl], sub_size=sub_size)
        dx, dy = measure_phase_shift(hrs["single_bicubic"][sl], hrs[panel_keys[-1]][sl])
        title = (
            f"{name} | 2x | {rd.used_frames}/{n} frames | ref #{ref_i} "
            f"| shift {math.hypot(dx, dy):.2f} HR px | exp {expo:.2f}x"
        )
        panel_path = (
            os.path.join(docs_panel_dir, f"{name}_2x.png")
            if publishable
            else os.path.join(out_dir, f"{name}_2x.png")
        )
        img_info = render_showcase(
            list(zip(panel_labels, crops)),
            sub_xy=sub_xy,
            sub_size=sub_size,
            exposure=expo,
            title=title,
            out_path=panel_path,
        )

        # Held-out evaluation
        dcb_mean: Dict[str, float] = {}
        cfa_fair_mean: Dict[str, float] = {}
        if not skip_heldout:
            methods_ho = [k for k in hrs.keys() if k != "single_bicubic"]
            hids = sorted(set(int(round(v)) for v in np.linspace(0, n - 1, min(n_ho, n))))
            dcb: Dict[str, List[float]] = {k: [] for k in methods_ho}
            cfa_fair: Dict[str, List[float]] = {k: [] for k in methods_ho}
            for hidx in hids:
                kept = [f for i, f in enumerate(frames) if i != hidx]
                kept_m = [m for i, m in enumerate(mos) if i != hidx]
                hrs_ho, _, rd_ho = _run_methods(kept, ratio=2.0, ref_idx=0, ecc_scale=0.25, include_wp13=include_wp13)
                ki = list(rd_ho.kept_indices)
                km = [kept_m[i] for i in ki]
                ref_pos = rd_ho.reference_index
                C_ref, _ = estimate_colour_matrix(kept[ref_pos].rgb, kept[ref_pos].mosaic, pat)
                W_held = estimate_homography(kept[ref_pos].green, frames[hidx].green, ecc_scale=0.25)

                ho_sub = {k: hrs_ho[k] for k in methods_ho}
                for k in methods_ho:
                    dcb[k].append(_dcb_heldout_gain_db(
                        ho_sub[k], frames[hidx], [kept[i] for i in ki], rd_ho.transforms, rd_ho.weights.tolist(), W_held, 2.0
                    ))
                res_fair = evaluate_cfa_heldout(
                    ho_sub, mos[hidx], km, rd_ho.transforms, rd_ho.weights.tolist(), W_held, 2.0, pat, colour_matrix=C_ref
                )
                for k in methods_ho:
                    cfa_fair[k].append(float(res_fair["per_method"][k]["heldout_cfa_gain_db"]))
            dcb_mean = {k: round(float(np.mean(v)), 3) for k, v in dcb.items()}
            cfa_fair_mean = {k: round(float(np.mean(v)), 3) for k, v in cfa_fair.items()}

        summary_all[name] = {
            "frames": n,
            "ref_index": ref_i,
            "crop_xy": [x0, y0],
            "flat_roi": [fx, fy, fw, fh],
            "panel": img_info,
            "crop_metrics": crop_metrics,
            "cfa_fair_mean_db": cfa_fair_mean,
            "dcb_mean_db": dcb_mean,
        }
        print(f"=== REAL {name} ===", flush=True)
        for mname, cm in crop_metrics.items():
            print(
                f"  {mname:15s} grad={cm['grad_energy']:.5f} flat_n={cm['flat_immerkaer']:.6f} "
                f"nyq={cm['nyquist_frac']:.5f} ov={cm['overshoot']:.3f} "
                f"drift={cm['colour_drift_vs_old_pct']:.2f}% sec={cm['sec']:.2f}",
                flush=True,
            )
        if cfa_fair_mean:
            print("  CFA-fair dB:", json.dumps(cfa_fair_mean), flush=True)
            print("  DCB      dB:", json.dumps(dcb_mean), flush=True)

    with open(os.path.join(out_dir, "real_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_all, f, indent=2)
    return summary_all


# --------------------------------------------------------------------------- #
# Protocol D: Runtime & Peak RAM/VRAM
# --------------------------------------------------------------------------- #

def run_bench(out_dir: str, include_wp13: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    bdir, (rx, ry), _, _ = BURSTS["own9"]
    files = sorted(glob.glob(os.path.join(bdir, "*.dng")) + glob.glob(os.path.join(bdir, "*.DNG")))
    frames = [load_raw_linear(f, roi=(rx, ry, 1024, 1024)) for f in files]

    out: Dict[str, Any] = {}
    for label, use_gpu in (("gpu_auto", None), ("cpu", False)):
        sub: Dict[str, Any] = {}
        modes = [("old_default", "demosaic_drizzle", {}), ("map_current", "map", {})]
        if include_wp13:
            modes.append(("map_wp13", "map", {"map_mode": "wp13"}))
        for mname, fusion, extra in modes:
            vram_before = None
            try:
                from raw_sr.core.gpu import _import_cupy, free_gpu_memory
                cp, _ = _import_cupy()
                if cp is not None:
                    free_gpu_memory()
                    cp.get_default_memory_pool().free_all_blocks()
                    vram_before = cp.cuda.runtime.memGetInfo()[0]
            except Exception:
                cp = None

            t0 = time.perf_counter()
            res = SuperResolution(
                ratio=2.0, blur_threshold_ratio=0.0, registration="homography",
                ecc_scale=0.25, fusion=fusion, use_gpu=use_gpu, **extra,
            ).run(frames, reference_index=0)
            elapsed = time.perf_counter() - t0
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            vram_peak_mb = None
            if cp is not None:
                try:
                    vram_peak_mb = round(cp.get_default_memory_pool().total_bytes() / (1024 * 1024), 1)
                except Exception:
                    pass
            sub[mname] = {
                "sec": round(elapsed, 2),
                "peak_rss_mb": round(rss_mb, 1),
                "vram_pool_mb": vram_peak_mb,
                "engine": res.metrics.get("cfa_kr_engine", res.metrics.get("scatter_engine")),
                "iters": res.metrics.get("cfa_kr_iterations"),
            }
            print(f"[BENCH {label} {mname}] {json.dumps(sub[mname])}", flush=True)
        out[label] = sub

    with open(os.path.join(out_dir, "bench_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WP13 Shared Evaluation Protocol")
    ap.add_argument("mode", choices=["synth", "chart", "real", "bench"])
    ap.add_argument("--bursts", nargs="+", default=list(BURSTS.keys()))
    ap.add_argument("--out-dir", default="/tmp/rawsr/wp13")
    ap.add_argument("--panel-dir", default=os.path.join(ROOT, "docs", "wp13"))
    ap.add_argument("--no-wp13", action="store_true", help="Run baselines only (skip map_wp13)")
    ap.add_argument("--skip-heldout", action="store_true", help="Skip held-out loop in real mode")
    ap.add_argument("--n-ho", type=int, default=3, help="Number of held-out frames")
    args = ap.parse_args()

    inc = not args.no_wp13
    if args.mode == "synth":
        run_synth(args.out_dir, include_wp13=inc)
    elif args.mode == "chart":
        run_chart(args.out_dir, include_wp13=inc)
    elif args.mode == "real":
        run_real(args.bursts, args.out_dir, args.panel_dir, n_ho=args.n_ho, skip_heldout=args.skip_heldout, include_wp13=inc)
    elif args.mode == "bench":
        run_bench(args.out_dir, include_wp13=inc)
