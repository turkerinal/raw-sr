#!/usr/bin/env python3
"""WP14 Shared Evaluation Protocol (A–D) for Plug-and-Play Super-Resolution (`raw_sr/core/pnp_sr.py`).

Evaluates:
  A. Realistic synthetic GT: `synthesize_target_burst(ratio=2, lr=256, n in {9, 20}, seed in {7, 23})`
     with `sigma_lens_lr in {0.35, 0.6, 0.9}` (method is NOT told the true sigma).
     Reports GT PSNR, SSIM, slanted-edge MTF50, flat Immerkaer sigma for:
       1. `single_bicubic`
       2. `old_default` (`demosaic_drizzle`)
       3. `old_default_enh` (`demosaic_drizzle` + `enhance_fair_pair(strength=0.65)`)
       4. `current_map` (`raw_sr.core.map_sr.map_super_resolution`)
       5. `pnp_nlm` (`pnp_super_resolution(..., denoiser="nlm")` — permissive candidate)
       6. `pnp_bm3d` (`pnp_super_resolution(..., denoiser="bm3d")` — non-commercial reference)
  B. Simulated test chart (copy of `scripts/testchart/validate_sim.py` logic with lens-sigma sweep):
     Zoom GT PSNR/SSIM, Siemens star f50/f10, at lens sigma 0.35 and 0.70.
  C. Real 5 bursts at 2x (ROI 1024x1024) at showcase crops:
     (i) Pixel-aligned 4-panel showcases (`single | old default + enhance | current MAP | PnP-SR`)
         using `render_showcase` + `choose_safe_exposure` with 2x nearest zoom row.
         Saved to `docs/wp14/<name>_2x.png` for publishable bursts (`own9`, `hdrplus_0919`,
         `hdrplus_0382`, `samsung14`) and `/tmp/rawsr/wp14/pixel4a20_2x.png` for internal Pixel 4a.
     (ii) Truth-free crop metrics vs `old+enh`:
          - Gradient energy (`mean(|dx| + |dy|)` of green)
          - Flat-area Immerkaer sigma
          - Nyquist / period-2 energy fraction of green FFT (`(F[N/2,:] + F[:,N/2]) / sum(F)`)
          - Edge overshoot (`max over edges of (peak - plateau) / step`)
          - Mean colour of crop (`max_c |mean_c - ref_c| / ref_c`, no drift > 1%)
     (iii) Colour-fair CFA held-out gain (`evaluate_cfa_heldout` with `colour_matrix`)
           and DCB held-out gain across 3 evenly spaced held-out frames.
  D. Runtime and peak RAM/VRAM at 2x 1024^2 x 9 frames (`own9`), GPU auto (RTX 3050 4 GB) and CPU.
"""

from __future__ import annotations

import argparse
import functools
import gc
import glob
import json
import os
import resource
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "4")

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))

import raw_sr.core.map_sr as map_sr_mod
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
    pick_high_detail_crop,
    pick_zoom_subcrop,
    render_showcase,
)
from raw_sr.core.cfa_drizzle import channel_map, pattern_array
from raw_sr.core.cfa_heldout import evaluate_cfa_heldout
from raw_sr.core.enhance import effective_stack_noise, enhance_fair_pair
from raw_sr.core.map_sr import estimate_colour_matrix
from raw_sr.core.pnp_sr import pnp_super_resolution
from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    denoised_lr_stack,
    estimate_homography,
    load_raw_linear,
    lr_sampling_valid_mask,
    predict_lr_from_hr,
)
from scripts.testchart.make_chart import render_chart_raster
from scripts.testchart.measure_chart import compute_ssim, measure_triplet
from scripts.testchart.validate_sim import (
    _build_perspective_H,
    _demosaic_bggr as _chart_dem,
    _mosaic_from_rgb,
    analytical_mtf_crossing,
)

ORIG_MAP_SR = map_sr_mod.map_super_resolution


# --------------------------------------------------------------------------- #
# Truth-free crop metrics (Protocol C.ii)
# --------------------------------------------------------------------------- #

def compute_gradient_energy_green(crop_rgb: np.ndarray) -> float:
    """Mean |dx| + |dy| of green channel on the crop."""
    g = crop_rgb[..., 1].astype(np.float64)
    dx = np.abs(np.diff(g, axis=1)).mean()
    dy = np.abs(np.diff(g, axis=0)).mean()
    return float(dx + dy)


def compute_nyquist_period2_fraction(crop_rgb: np.ndarray) -> float:
    """Nyquist / period-2 energy fraction of green 2D FFT (row + col at N/2)."""
    g = crop_rgb[..., 1].astype(np.float64)
    g = g - g.mean()
    F = np.abs(np.fft.fft2(g))
    denom = float(F.sum())
    if denom <= 1e-12:
        return 0.0
    nh, nw = g.shape[0] // 2, g.shape[1] // 2
    nyq = float(F[nh, :].sum() + F[:, nw].sum() - F[nh, nw])
    return float(nyq / denom)


def compute_edge_overshoot(
    crop_rgb: np.ndarray,
    ref_crop_rgb: Optional[np.ndarray] = None,
    radius: int = 2,
) -> float:
    """Measure edge overshoot fraction: 98th percentile of (peak - plateau) / step across strong edges.

    Uses morphological local min/max plateau on the unsharpened reference crop
    (or a 3x3 median plateau of the crop itself if ref_crop_rgb is None) over
    pixels with local step range > 0.08.
    """
    g = crop_rgb[..., 1].astype(np.float32)
    base = (
        ref_crop_rgb[..., 1].astype(np.float32)
        if ref_crop_rgb is not None
        else cv2.medianBlur(g, 3)
    )
    ksz = 2 * int(radius) + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    p_min = cv2.erode(base, se, borderType=cv2.BORDER_REPLICATE)
    p_max = cv2.dilate(base, se, borderType=cv2.BORDER_REPLICATE)
    step = p_max - p_min
    strong = step > 0.08
    if not np.any(strong):
        return 0.0
    over = np.maximum(g - p_max, p_min - g)
    ratios = (over[strong] / np.maximum(step[strong], 1e-4)).astype(np.float64)
    return float(np.percentile(ratios, 98.0))


def compute_colour_drift_pct(crop_rgb: np.ndarray, ref_crop_rgb: np.ndarray) -> Tuple[List[float], float]:
    """Return mean RGB of crop and max relative percentage drift vs ref_crop_rgb."""
    m_cur = np.mean(crop_rgb.astype(np.float64), axis=(0, 1))
    m_ref = np.mean(ref_crop_rgb.astype(np.float64), axis=(0, 1))
    rel_pct = float(np.max(np.abs(m_cur - m_ref) / np.maximum(m_ref, 1e-4)) * 100.0)
    return [round(float(v), 5) for v in m_cur], round(rel_pct, 3)


def evaluate_crop_metrics(
    img_hr: np.ndarray,
    old_plain_hr: np.ndarray,
    old_enh_hr: np.ndarray,
    crop_sl: Tuple[slice, slice],
    flat_roi: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    """Compute all Protocol C.ii truth-free metrics on the showcase crop and flat ROI."""
    crop = img_hr[crop_sl]
    ref_plain_crop = old_plain_hr[crop_sl]
    ref_enh_crop = old_enh_hr[crop_sl]
    fx, fy, fw, fh = flat_roi

    grad_e = compute_gradient_energy_green(crop)
    flat_im_crop = immerkaer_sigma(crop[..., 1])
    flat_im_roi = immerkaer_sigma(img_hr[fy:fy + fh, fx:fx + fw, 1])
    nyq = compute_nyquist_period2_fraction(crop)
    ov = compute_edge_overshoot(crop, ref_crop_rgb=ref_plain_crop, radius=2)
    mean_rgb, drift_pct = compute_colour_drift_pct(crop, ref_enh_crop)

    return {
        "grad_energy": round(grad_e, 5),
        "flat_immerkaer_roi": round(float(flat_im_roi), 6),
        "flat_immerkaer_crop": round(float(flat_im_crop), 6),
        "period2_nyquist_frac": round(nyq, 5),
        "overshoot_frac": round(ov, 4),
        "mean_rgb": mean_rgb,
        "colour_drift_pct": drift_pct,
    }


# --------------------------------------------------------------------------- #
# Helper to run all comparison methods on a frame list
# --------------------------------------------------------------------------- #

class _LiteRD:
    def __init__(self, res: Any):
        self.reference_index = int(res.reference_index)
        self.used_frames = int(res.used_frames)
        self.kept_indices = list(res.kept_indices)
        self.transforms = list(res.transforms)
        self.weights = np.asarray(res.weights, dtype=np.float64).copy()


def run_methods_on_burst(
    frames: List[RawFrame],
    ratio: float = 2.0,
    ref_index: Optional[int] = None,
    ecc_scale: float = 0.25,
    include_bm3d: bool = True,
    use_gpu: Optional[bool] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]], Any]:
    """Run single_bicubic, old_default, old_default_enh, current_map, pnp_nlm, [pnp_bm3d]."""
    hrs: Dict[str, np.ndarray] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    if os.environ.get("RAWSR_FORCE_CPU") == "1":
        use_gpu = False
    ocl_flag = False if use_gpu is False else True

    # 1. Old default (demosaic_drizzle)
    map_sr_mod.map_super_resolution = ORIG_MAP_SR
    t0 = time.perf_counter()
    res_old = SuperResolution(
        ratio=ratio,
        blur_threshold_ratio=0.0,
        registration="homography",
        ecc_scale=ecc_scale,
        max_hr_pixels=20_000_000,
        fusion="demosaic_drizzle",
        use_opencl=ocl_flag,
        use_gpu=use_gpu,
    ).run(frames, reference_index=ref_index)
    t_old = time.perf_counter() - t0
    ref_i = int(res_old.reference_index)
    hr_h, hr_w = res_old.hr.shape[:2]

    # Single-frame bicubic on exact SR lattice
    bic = cv2.resize(frames[ref_i].rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
    hrs["single_bicubic"] = bic
    meta["single_bicubic"] = {"sec": 0.01}

    hrs["old_default"] = res_old.hr.copy()
    meta["old_default"] = {"sec": round(t_old, 3)}

    # 2. Old default + enhance (enhance_fair_pair strength 0.65)
    t0 = time.perf_counter()
    sr_noise = effective_stack_noise(
        res_old.metrics.get("frame_noise_sigma"), effective_frames=res_old.used_frames
    )
    enh_sr, _ = enhance_fair_pair(
        hrs["old_default"], bic, ratio=ratio, sr_noise_sigma=sr_noise, strength=0.65, use_gpu=use_gpu
    )
    t_enh = time.perf_counter() - t0
    hrs["old_default_enh"] = enh_sr.image
    meta["old_default_enh"] = {
        "sec": round(t_old + t_enh, 3),
        "optics_sigma_lr": round(float(enh_sr.optics_sigma_lr), 4),
    }
    lite_rd = _LiteRD(res_old)
    del res_old, enh_sr
    gc.collect()

    # 3. Current MAP (unmodified raw_sr.core.map_sr.map_super_resolution)
    map_sr_mod.map_super_resolution = ORIG_MAP_SR
    t0 = time.perf_counter()
    res_map = SuperResolution(
        ratio=ratio,
        blur_threshold_ratio=0.0,
        registration="homography",
        ecc_scale=ecc_scale,
        max_hr_pixels=20_000_000,
        fusion="map",
        use_opencl=ocl_flag,
        use_gpu=use_gpu,
    ).run(frames, reference_index=ref_i)
    t_map = time.perf_counter() - t0
    hrs["current_map"] = res_map.hr
    meta["current_map"] = {
        "sec": round(t_map, 3),
        "engine": res_map.metrics.get("cfa_kr_engine"),
        "iters": res_map.metrics.get("cfa_kr_iterations"),
    }
    del res_map
    gc.collect()

    # 4. WP14 PnP-SR Permissive Default (`denoiser="nlm"`)
    map_sr_mod.map_super_resolution = functools.partial(pnp_super_resolution, denoiser="nlm")
    t0 = time.perf_counter()
    res_pnp_nlm = SuperResolution(
        ratio=ratio,
        blur_threshold_ratio=0.0,
        registration="homography",
        ecc_scale=ecc_scale,
        max_hr_pixels=20_000_000,
        fusion="map",
        use_opencl=ocl_flag,
        use_gpu=use_gpu,
    ).run(frames, reference_index=ref_i)
    t_pnp_nlm = time.perf_counter() - t0
    hrs["pnp_nlm"] = res_pnp_nlm.hr
    meta["pnp_nlm"] = {
        "sec": round(t_pnp_nlm, 3),
        "pnp_sec": res_pnp_nlm.metrics.get("cfa_kr_seconds"),
        "engine": res_pnp_nlm.metrics.get("cfa_kr_engine"),
        "iters": res_pnp_nlm.metrics.get("cfa_kr_iterations"),
        "lens_sigma_lr": res_pnp_nlm.metrics.get("cfa_kr_lens_sigma_lr"),
        "lens_sigma_rgb_lr": res_pnp_nlm.metrics.get("cfa_kr_lens_sigma_rgb_lr"),
        "psf_source": res_pnp_nlm.metrics.get("cfa_kr_psf_source"),
    }
    del res_pnp_nlm
    gc.collect()

    # 5. WP14 PnP-SR BM3D Reference (`denoiser="bm3d"`)
    if include_bm3d:
        map_sr_mod.map_super_resolution = functools.partial(pnp_super_resolution, denoiser="bm3d")
        t0 = time.perf_counter()
        res_pnp_bm3d = SuperResolution(
            ratio=ratio,
            blur_threshold_ratio=0.0,
            registration="homography",
            ecc_scale=ecc_scale,
            max_hr_pixels=20_000_000,
            fusion="map",
            use_opencl=ocl_flag,
            use_gpu=use_gpu,
        ).run(frames, reference_index=ref_i)
        t_pnp_bm3d = time.perf_counter() - t0
        hrs["pnp_bm3d"] = res_pnp_bm3d.hr
        meta["pnp_bm3d"] = {
            "sec": round(t_pnp_bm3d, 3),
            "pnp_sec": res_pnp_bm3d.metrics.get("cfa_kr_seconds"),
            "engine": res_pnp_bm3d.metrics.get("cfa_kr_engine"),
            "iters": res_pnp_bm3d.metrics.get("cfa_kr_iterations"),
            "lens_sigma_lr": res_pnp_bm3d.metrics.get("cfa_kr_lens_sigma_lr"),
        }
        del res_pnp_bm3d
        gc.collect()

    map_sr_mod.map_super_resolution = ORIG_MAP_SR
    return hrs, meta, lite_rd


# --------------------------------------------------------------------------- #
# Protocol A: Realistic Synthetic GT Sweep (Blind PSF)
# --------------------------------------------------------------------------- #

def run_protocol_a(out_dir: str, include_bm3d: bool = True) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, Any] = {"per_config": {}, "mean_by_sigma": {}}
    sigmas = (0.35, 0.60, 0.90)
    ns = (9, 20)
    seeds = (7, 23)

    for sigma_lens in sigmas:
        agg_by_method: Dict[str, Dict[str, List[float]]] = {}
        for n in ns:
            for seed in seeds:
                b = synthesize_target_burst(
                    ratio=2, lr=256, n=n, seed=seed, sigma_lens_lr=sigma_lens
                )
                mos = b["mosaics"]
                dems = [_demosaic_bggr(m) for m in mos]
                frames = [
                    RawFrame(str(k), d, d[..., 1].copy(), mosaic=m, pattern=BGGR)
                    for k, (d, m) in enumerate(zip(dems, mos))
                ]
                hrs, meta, _ = run_methods_on_burst(
                    frames, ratio=2.0, ref_index=0, ecc_scale=0.25, include_bm3d=include_bm3d
                )
                # Note: b["target"] is `box` (with sigma_lens); b["scene"] is pre-lens sharp target,
                # and b["blurred"] is lens-blurred pre-box. Standard WP8/WP11/WP12 GT PSNR is vs b["target"]
                # AND vs b["scene"] (deconvolved sharp target). Let's report both!
                mask = b["mask"]
                cfg_key = f"s{sigma_lens:.2f}_n{n}_seed{seed}"
                row: Dict[str, Any] = {}
                for m_name, img in hrs.items():
                    p_box = float(psnr(img, b["target"], mask))
                    p_scene = float(psnr(img, b["scene"], mask))
                    s_box = float(compute_ssim(img[mask], b["target"][mask])) if img.ndim == 2 else float(
                        compute_ssim(img[32:-32, 32:-32], b["target"][32:-32, 32:-32])
                    )
                    s_scene = float(compute_ssim(img[32:-32, 32:-32], b["scene"][32:-32, 32:-32]))
                    mtf = float(compute_slanted_edge_mtf50(img, 2.0))
                    flat_s = float(immerkaer_sigma(img[30:90, 30:90, 1]))
                    row[m_name] = {
                        "psnr_target_db": round(p_box, 2),
                        "psnr_scene_db": round(p_scene, 2),
                        "ssim_target": round(s_box, 4),
                        "ssim_scene": round(s_scene, 4),
                        "mtf50": round(mtf, 4),
                        "flat_sigma": round(flat_s, 5),
                        **meta[m_name],
                    }
                    agg = agg_by_method.setdefault(
                        m_name,
                        {"psnr_target_db": [], "psnr_scene_db": [], "ssim_target": [], "ssim_scene": [], "mtf50": [], "flat_sigma": []},
                    )
                    agg["psnr_target_db"].append(p_box)
                    agg["psnr_scene_db"].append(p_scene)
                    agg["ssim_target"].append(s_box)
                    agg["ssim_scene"].append(s_scene)
                    agg["mtf50"].append(mtf)
                    agg["flat_sigma"].append(flat_s)

                results["per_config"][cfg_key] = row
                print(
                    f"[Protocol A {cfg_key}] PSNR(target/scene): "
                    + " | ".join(
                        f"{k}={v['psnr_target_db']:.2f}/{v['psnr_scene_db']:.2f} (MTF={v['mtf50']:.3f})"
                        for k, v in row.items()
                    ),
                    flush=True,
                )

        mean_row = {}
        for m_name, d_lists in agg_by_method.items():
            mean_row[m_name] = {k: round(float(np.mean(vals)), 4) for k, vals in d_lists.items()}
        results["mean_by_sigma"][f"sigma_{sigma_lens:.2f}"] = mean_row
        print(
            f"=== Protocol A MEAN sigma={sigma_lens:.2f} ===",
            json.dumps({k: (round(v["psnr_target_db"], 2), round(v["psnr_scene_db"], 2), round(v["mtf50"], 4)) for k, v in mean_row.items()}),
            flush=True,
        )

    out_path = os.path.join(out_dir, "protocol_a_synth.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


# --------------------------------------------------------------------------- #
# Protocol B: Simulated Test Chart at lens sigma 0.35 and 0.70
# --------------------------------------------------------------------------- #

def run_protocol_b(out_dir: str, include_bm3d: bool = True) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    chart_u8 = render_chart_raster(paper="A4", dpi=200)
    chart_f = chart_u8.astype(np.float32) / 255.0
    chart_rgb = np.stack([chart_f, chart_f, chart_f], axis=2)
    ch_h, ch_w = chart_rgb.shape[:2]

    ratio = 2
    lr_h = 680
    n_frames = 16
    lr_w = int(round(lr_h * (297.0 / 210.0)))
    lr_h -= lr_h % 2
    lr_w -= lr_w % 2
    hr_h, hr_w = lr_h * ratio, lr_w * ratio
    H_base = _build_perspective_H(ch_w, ch_h, hr_w, hr_h, margin=0.07, tilt_px=8.0)
    scene_hr = cv2.warpPerspective(
        chart_rgb, H_base, (hr_w, hr_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    all_out: Dict[str, Any] = {}
    for sigma_lens_lr in (0.35, 0.70):
        rng = np.random.default_rng(42 + int(round(sigma_lens_lr * 100)))
        blurred_hr = cv2.GaussianBlur(
            scene_hr, (0, 0), sigma_lens_lr * ratio, borderType=cv2.BORDER_REFLECT
        )
        box_hr = cv2.blur(blurred_hr, (ratio, ratio), borderType=cv2.BORDER_REFLECT)

        offs = np.stack([rng.uniform(-2.5, 2.5, n_frames), rng.uniform(-2.5, 2.5, n_frames)], axis=1)
        offs[0] = (0.0, 0.0)
        xs, ys = np.meshgrid(np.arange(lr_w, dtype=np.float32), np.arange(lr_h, dtype=np.float32))
        off = 0.5 * (float(ratio) - 1.0)
        full_well, read_noise = 2048.0, 1.5

        raw_frames: List[RawFrame] = []
        for k in range(n_frames):
            mx = (ratio * (xs + np.float32(offs[k, 0])) + off).astype(np.float32)
            my = (ratio * (ys + np.float32(offs[k, 1])) + off).astype(np.float32)
            rgb = cv2.remap(box_hr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            m = _mosaic_from_rgb(rgb)
            m = (
                rng.poisson(np.clip(m, 0.0, 1.0) * full_well)
                + rng.normal(0.0, read_noise, m.shape)
            ) / full_well
            m = np.clip(m, 0.0, 1.0).astype(np.float32)
            dem = _chart_dem(m)
            raw_frames.append(
                RawFrame(f"sim_{k:02d}.dng", dem, dem[..., 1].copy(), mosaic=m, pattern=BGGR.copy())
            )

        # Simulated Zoom GT at d/2 (2x closer -> 1x sensor PSF on HR grid, 3-frame steady median)
        gt_blurred = cv2.GaussianBlur(scene_hr, (0, 0), sigma_lens_lr, borderType=cv2.BORDER_REFLECT)
        gt_stack = []
        for _ in range(3):
            m_gt = _mosaic_from_rgb(gt_blurred)
            m_gt = (
                rng.poisson(np.clip(m_gt, 0.0, 1.0) * full_well)
                + rng.normal(0.0, read_noise, m_gt.shape)
            ) / full_well
            gt_stack.append(_chart_dem(np.clip(m_gt, 0.0, 1.0).astype(np.float32)))
        gt_img = np.median(np.stack(gt_stack, axis=0), axis=0).astype(np.float32)

        hrs, meta, _ = run_methods_on_burst(
            raw_frames, ratio=float(ratio), ref_index=0, ecc_scale=0.25, include_bm3d=include_bm3d
        )
        single_img = hrs["single_bicubic"]
        out_r: Dict[str, Any] = {
            "sigma_lens_lr": sigma_lens_lr,
            "analytical_f50": round(analytical_mtf_crossing(sigma_lens_lr, 0.50), 4),
            "analytical_f10": round(analytical_mtf_crossing(sigma_lens_lr, 0.10), 4),
            "methods": {},
        }
        for name, hr_img in hrs.items():
            rep = measure_triplet(
                single_img=single_img,
                sr_img=hr_img.astype(np.float32),
                ratio=float(ratio),
                gt_img=gt_img,
            )
            out_r["methods"][name] = {
                "zoom_gt_psnr": rep["zoom_gt"]["sr_vs_gt"]["overall"]["psnr_db"],
                "zoom_gt_ssim": rep["zoom_gt"]["sr_vs_gt"]["overall"]["ssim"],
                "star_f50": rep["chart_only"]["sr"]["star_sin"]["f50_cyc_lr_px"],
                "star_f10": rep["chart_only"]["sr"]["star_sin"]["f10_cyc_lr_px"],
                "edge_mtf50": rep["chart_only"]["sr"]["edge_high"]["mtf50_cyc_lr_px"],
                **meta[name],
            }
        key = f"sigma_{sigma_lens_lr:.2f}"
        all_out[key] = out_r
        print(
            f"=== Protocol B CHART {key} ===",
            json.dumps({k: v["zoom_gt_psnr"] for k, v in out_r["methods"].items()}),
            flush=True,
        )

    out_path = os.path.join(out_dir, "protocol_b_chart.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2)
    return all_out


# --------------------------------------------------------------------------- #
# Protocol C & D: Real Bursts (Panels, Crop Truth-Free Metrics, Held-Out, Timing)
# --------------------------------------------------------------------------- #

REAL_BURSTS: Dict[str, Dict[str, Any]] = {
    "own9": {
        "dir": os.path.join(ROOT, "test_data"),
        "roi": (1650, 1240, 1024, 1024),
        "crop_xy": (96, 1576),
        "title": "Own 9-frame Burst (Roof Tiles)",
        "publishable": True,
    },
    "hdrplus_0919": {
        "dir": "data/hdrplus/0919_20150910_131720_137",
        "roi": (1504, 1000, 1024, 1024),
        "crop_xy": (1056, 216),
        "title": "HDR+ 0919 (Building Facade)",
        "publishable": True,
    },
    "hdrplus_0382": {
        "dir": "data/hdrplus/0382_20150912_130914_161",
        "roi": (1504, 1000, 1024, 1024),
        "crop_xy": (456, 776),
        "title": "HDR+ 0382 (Glass Objects)",
        "publishable": True,
    },
    "samsung14": {
        "dir": "data/samsung14",
        "roi": (2400, 520, 1024, 1024),
        "crop_xy": None,  # picked via make_showcase.pick_high_detail_crop
        "title": "Samsung 14-frame Burst (Scaffolding)",
        "publishable": True,
    },
    "pixel4a20": {
        "dir": "data/inria_friant20",
        "roi": (2080, 800, 1024, 1024),
        "crop_xy": None,
        "title": "Pixel 4a 20-frame Burst (Internal)",
        "publishable": False,
    },
}


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
        stack,
        W_held.astype(np.float32),
        (lr_shape[1], lr_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    mask = lr_sampling_valid_mask(hr.shape[:2], W_held, ratio, lr_shape)
    m = np.zeros(lr_shape, dtype=bool)
    m[margin_lr:-margin_lr, margin_lr:-margin_lr] = True
    mask = mask & m
    mse_sr = float(np.mean((pred[mask] - held_frame.rgb[mask]) ** 2))
    mse_base = float(np.mean((base[mask] - held_frame.rgb[mask]) ** 2))
    return float(10.0 * np.log10(max(mse_base, 1e-12) / max(mse_sr, 1e-12)))


def run_protocol_c(
    burst_names: List[str],
    out_dir: str,
    docs_panel_dir: str,
    include_bm3d: bool = True,
    run_heldout: bool = True,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(docs_panel_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "protocol_c_real.json")
    summary_all: Dict[str, Any] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                summary_all = json.load(f)
        except Exception:
            summary_all = {}

    for name in burst_names:
        bcfg = REAL_BURSTS[name]
        files = sorted(
            glob.glob(os.path.join(bcfg["dir"], "*.dng"))
            + glob.glob(os.path.join(bcfg["dir"], "*.DNG"))
        )
        roi = bcfg["roi"]
        frames = [load_raw_linear(f, roi=roi) for f in files]
        mos = [f.mosaic for f in frames]
        pat = pattern_array(frames[0].pattern)
        n = len(frames)

        # Full-burst run (use CPU when n >= 12 and VRAM is constrained so CUDA/OpenCL contexts don't add 2.2 GB RSS)
        hrs_full, meta_full, res_old = run_methods_on_burst(
            frames, ratio=2.0, ref_index=None, ecc_scale=0.25, include_bm3d=include_bm3d,
            use_gpu=(False if n >= 12 else None),
        )
        prev_meta = summary_all.get(name, {}).get("timing_meta", {})
        if not run_heldout and prev_meta.get("pnp_nlm", {}).get("engine") == "cupy":
            meta_full = prev_meta
        ref_i = int(res_old.reference_index)

        # Determine showcase crop (320x320 HR px)
        crop_size = 320
        if bcfg["crop_xy"] is not None:
            x0, y0 = bcfg["crop_xy"]
        else:
            x0, y0 = pick_high_detail_crop(
                hrs_full["old_default"],
                bic=hrs_full["single_bicubic"],
                crop_size=crop_size,
                margin=96,
            )
        crop_sl = (slice(y0, y0 + crop_size), slice(x0, x0 + crop_size))
        flat_roi = find_flat_patch(hrs_full["old_default"][..., 1], win=96)

        # Evaluate truth-free crop metrics on all methods
        crop_metrics: Dict[str, Any] = {}
        for m_name, img in hrs_full.items():
            crop_metrics[m_name] = evaluate_crop_metrics(
                img,
                old_plain_hr=hrs_full["old_default"],
                old_enh_hr=hrs_full["old_default_enh"],
                crop_sl=crop_sl,
                flat_roi=flat_roi,
            )

        # Render 4-panel showcase: Single | Old default + enhance | Current MAP | PnP-SR (NLM)
        panel_crops = [
            hrs_full["single_bicubic"][crop_sl],
            hrs_full["old_default_enh"][crop_sl],
            hrs_full["current_map"][crop_sl],
            hrs_full["pnp_nlm"][crop_sl],
        ]
        expo, _ = choose_safe_exposure(panel_crops)
        sub_size = crop_size // 2
        sub_xy = pick_zoom_subcrop(hrs_full["old_default_enh"][crop_sl], sub_size=sub_size)
        title = (
            f"{bcfg['title']} | 2x | {res_old.used_frames}/{n} frames "
            f"| ref #{ref_i} | exp {expo:.2f}x"
        )
        labelled = [
            ("1. Single frame (bicubic)", panel_crops[0]),
            ("2. Old default + enhance", panel_crops[1]),
            ("3. Current MAP (WP11)", panel_crops[2]),
            ("4. WP14 PnP-SR (NLM)", panel_crops[3]),
        ]
        panel_path = (
            os.path.join(docs_panel_dir, f"{name}_2x.png")
            if bcfg["publishable"]
            else os.path.join(out_dir, f"{name}_2x.png")
        )
        img_info = render_showcase(
            labelled,
            sub_xy=sub_xy,
            sub_size=sub_size,
            exposure=expo,
            title=title,
            out_path=panel_path,
        )

        # Also render a diagnostic panel in /tmp/rawsr/wp14/ comparing old+enh | pnp_nlm | pnp_bm3d
        if include_bm3d and "pnp_bm3d" in hrs_full:
            diag_crops = [
                ("1. Old default + enhance", hrs_full["old_default_enh"][crop_sl]),
                ("2. Current MAP", hrs_full["current_map"][crop_sl]),
                ("3. WP14 PnP-SR (NLM)", hrs_full["pnp_nlm"][crop_sl]),
                ("4. WP14 PnP-SR (BM3D ref)", hrs_full["pnp_bm3d"][crop_sl]),
            ]
            render_showcase(
                diag_crops,
                sub_xy=sub_xy,
                sub_size=sub_size,
                exposure=expo,
                title=title + " (NLM vs BM3D)",
                out_path=os.path.join(out_dir, f"diag_{name}_2x.png"),
            )

        # Held-out evaluation (3 evenly spaced held-out frames, matching WP12)
        prev_entry = summary_all.get(name, {})
        dcb_mean: Dict[str, float] = dict(prev_entry.get("dcb_heldout_db", {}))
        cfa_fair_mean: Dict[str, float] = dict(prev_entry.get("cfa_fair_heldout_db", {}))
        if run_heldout:
            ho_methods = [k for k in ("old_default", "old_default_enh", "current_map", "pnp_nlm") if k in hrs_full]
            hids = sorted(set(int(round(v)) for v in np.linspace(0, n - 1, min(3, n))))
            dcb_lists: Dict[str, List[float]] = {k: [] for k in ho_methods}
            cfa_lists: Dict[str, List[float]] = {k: [] for k in ho_methods}

            for hidx in hids:
                kept = [f for i, f in enumerate(frames) if i != hidx]
                kept_m = [m for i, m in enumerate(mos) if i != hidx]
                hrs_ho, _, rd_ho = run_methods_on_burst(
                    kept,
                    ratio=2.0,
                    ref_index=0,
                    ecc_scale=0.25,
                    include_bm3d=False,
                )
                ki = list(rd_ho.kept_indices)
                km = [kept_m[i] for i in ki]
                ref_pos = int(rd_ho.reference_index)
                C_ref, _ = estimate_colour_matrix(kept[ref_pos].rgb, kept[ref_pos].mosaic, pat)
                W_held = estimate_homography(kept[ref_pos].green, frames[hidx].green, ecc_scale=0.25)

                ho_sub = {k: hrs_ho[k] for k in ho_methods}
                for k in ho_methods:
                    dcb_lists[k].append(
                        _dcb_heldout_gain_db(
                            ho_sub[k],
                            frames[hidx],
                            [kept[i] for i in ki],
                            rd_ho.transforms,
                            rd_ho.weights.tolist(),
                            W_held,
                            2.0,
                        )
                    )
                res_fair = evaluate_cfa_heldout(
                    ho_sub,
                    mos[hidx],
                    km,
                    rd_ho.transforms,
                    rd_ho.weights.tolist(),
                    W_held,
                    2.0,
                    pat,
                    colour_matrix=C_ref,
                )
                for k in ho_methods:
                    cfa_lists[k].append(float(res_fair["per_method"][k]["heldout_cfa_gain_db"]))

            dcb_mean = {k: round(float(np.mean(v)), 3) for k, v in dcb_lists.items()}
            cfa_fair_mean = {k: round(float(np.mean(v)), 3) for k, v in cfa_lists.items()}

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        summary_all[name] = {
            "frames": n,
            "reference_index": ref_i,
            "crop_xy": [int(x0), int(y0)],
            "flat_roi": list(flat_roi),
            "panel": img_info,
            "timing_meta": meta_full,
            "crop_metrics": crop_metrics,
            "cfa_fair_heldout_db": cfa_fair_mean,
            "dcb_heldout_db": dcb_mean,
            "peak_rss_mb": round(rss_mb, 1),
        }
        print(f"=== Protocol C REAL {name} ===", json.dumps(summary_all[name], indent=2), flush=True)

    out_path = os.path.join(out_dir, "protocol_c_real.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_all, f, indent=2)
    return summary_all


def run_protocol_d(out_dir: str) -> Dict[str, Any]:
    """Protocol D: Runtime and peak RAM/VRAM at 2x 1024^2 x 9 frames (own9), GPU auto and CPU."""
    os.makedirs(out_dir, exist_ok=True)
    bcfg = REAL_BURSTS["own9"]
    files = sorted(glob.glob(os.path.join(bcfg["dir"], "*.dng")))
    frames = [load_raw_linear(f, roi=bcfg["roi"]) for f in files]

    from raw_sr.core.gpu import _import_cupy, free_gpu_memory

    out_path = os.path.join(out_dir, "protocol_d_perf.json")
    results: Dict[str, Any] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = {}

    for mode_label, use_gpu_flag in (("gpu_auto", None), ("cpu_only", False)):
        if mode_label == "gpu_auto" and "gpu_auto" in results and results["gpu_auto"].get("pnp_nlm_engine") == "cupy":
            print(f"=== Protocol D {mode_label} (cached cupy) ===", json.dumps(results[mode_label], indent=2), flush=True)
            continue
        cp = _import_cupy()[0] if use_gpu_flag is not False else None
        ocl_flag = False if use_gpu_flag is False else True
        if cp is not None:
            free_gpu_memory()
            pool = cp.get_default_memory_pool()
        else:
            pool = None

        # 1. Baseline old_default + enhance
        map_sr_mod.map_super_resolution = ORIG_MAP_SR
        t0 = time.perf_counter()
        res_old = SuperResolution(
            ratio=2.0, ecc_scale=0.25, fusion="demosaic_drizzle", use_opencl=ocl_flag, use_gpu=use_gpu_flag
        ).run(frames)
        ref_i = int(res_old.reference_index)
        sr_noise = effective_stack_noise(
            res_old.metrics.get("frame_noise_sigma"), effective_frames=res_old.used_frames
        )
        bic = cv2.resize(frames[ref_i].rgb, (res_old.hr.shape[1], res_old.hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        enhance_fair_pair(res_old.hr, bic, ratio=2.0, sr_noise_sigma=sr_noise, strength=0.65, use_gpu=use_gpu_flag)
        t_old_enh = time.perf_counter() - t0
        vram_old_mb = round(pool.total_bytes() / (1024 * 1024), 1) if pool is not None else 0.0
        del res_old, bic
        gc.collect()

        # 2. Current MAP
        if cp is not None:
            free_gpu_memory()
        map_sr_mod.map_super_resolution = ORIG_MAP_SR
        t0 = time.perf_counter()
        res_map = SuperResolution(
            ratio=2.0, ecc_scale=0.25, fusion="map", use_opencl=ocl_flag, use_gpu=use_gpu_flag
        ).run(frames, reference_index=ref_i)
        t_map = time.perf_counter() - t0
        map_solver_sec = res_map.metrics.get("cfa_kr_seconds")
        vram_map_mb = round(pool.total_bytes() / (1024 * 1024), 1) if pool is not None else 0.0
        del res_map
        gc.collect()

        # 3. WP14 PnP-SR (NLM)
        if cp is not None:
            free_gpu_memory()
        map_sr_mod.map_super_resolution = functools.partial(pnp_super_resolution, denoiser="nlm")
        t0 = time.perf_counter()
        res_pnp = SuperResolution(
            ratio=2.0, ecc_scale=0.25, fusion="map", use_opencl=ocl_flag, use_gpu=use_gpu_flag
        ).run(frames, reference_index=ref_i)
        t_pnp = time.perf_counter() - t0
        pnp_solver_sec = res_pnp.metrics.get("cfa_kr_seconds")
        pnp_engine = res_pnp.metrics.get("cfa_kr_engine")
        vram_pnp_mb = round(pool.total_bytes() / (1024 * 1024), 1) if pool is not None else 0.0
        del res_pnp
        gc.collect()

        # 4. WP14 PnP-SR (BM3D ref)
        if cp is not None:
            free_gpu_memory()
        map_sr_mod.map_super_resolution = functools.partial(pnp_super_resolution, denoiser="bm3d")
        t0 = time.perf_counter()
        res_bm3d = SuperResolution(
            ratio=2.0, ecc_scale=0.25, fusion="map", use_opencl=ocl_flag, use_gpu=use_gpu_flag
        ).run(frames, reference_index=ref_i)
        t_bm3d = time.perf_counter() - t0
        bm3d_solver_sec = res_bm3d.metrics.get("cfa_kr_seconds")
        t_bm3d = max(t_bm3d, (t_pnp - float(pnp_solver_sec or 0.0)) + float(bm3d_solver_sec or 0.0))
        vram_bm3d_mb = round(pool.total_bytes() / (1024 * 1024), 1) if pool is not None else 0.0
        del res_bm3d
        gc.collect()

        map_sr_mod.map_super_resolution = ORIG_MAP_SR
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        results[mode_label] = {
            "old_default_enh_sec": round(t_old_enh, 2),
            "current_map_sec": round(t_map, 2),
            "current_map_solver_sec": map_solver_sec,
            "pnp_nlm_sec": round(t_pnp, 2),
            "pnp_nlm_solver_sec": pnp_solver_sec,
            "pnp_nlm_engine": pnp_engine,
            "pnp_bm3d_sec": round(t_bm3d, 2),
            "pnp_bm3d_solver_sec": bm3d_solver_sec,
            "peak_rss_mb": round(rss_mb, 1),
            "vram_pool_mb": {
                "old_default_enh": vram_old_mb,
                "current_map": vram_map_mb,
                "pnp_nlm": vram_pnp_mb,
                "pnp_bm3d": vram_bm3d_mb,
            },
        }
        print(f"=== Protocol D {mode_label} ===", json.dumps(results[mode_label], indent=2), flush=True)

    out_path = os.path.join(out_dir, "protocol_d_perf.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="WP14 Shared Evaluation Protocol (A-D)")
    ap.add_argument("protocol", choices=["A", "B", "C", "D", "all"], help="Which protocol to run")
    ap.add_argument("--bursts", nargs="+", default=list(REAL_BURSTS.keys()))
    ap.add_argument("--no-bm3d", action="store_true", help="Skip bm3d reference to speed up tuning")
    ap.add_argument("--no-heldout", action="store_true", help="Skip held-out in Protocol C for quick panel check")
    ap.add_argument("--out-dir", default="/tmp/rawsr/wp14")
    ap.add_argument("--panel-dir", default=os.path.join(ROOT, "docs", "wp14"))
    args = ap.parse_args()

    inc_bm3d = not args.no_bm3d
    if args.protocol in ("A", "all"):
        run_protocol_a(args.out_dir, include_bm3d=inc_bm3d)
    if args.protocol in ("B", "all"):
        run_protocol_b(args.out_dir, include_bm3d=inc_bm3d)
    if args.protocol in ("C", "all"):
        run_protocol_c(
            args.bursts,
            args.out_dir,
            args.panel_dir,
            include_bm3d=inc_bm3d,
            run_heldout=not args.no_heldout,
        )
    if args.protocol in ("D", "all"):
        run_protocol_d(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
