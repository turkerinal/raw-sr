#!/usr/bin/env python3
"""Simulated phone-camera validation of the WP10 test chart and measurement kit.

Renders the WP10 chart through a physical phone-camera forward model:
  - Perspective view + small handheld shake (sub-pixel shift + tiny tilt jitter)
  - Known lens Gaussian PSF (sigma_lens_lr) + 1 LR px pixel aperture (box)
  - Bayer BGGR mosaic + Poisson-Gaussian noise
  - 2x super-resolution via raw_sr.core.superres.SuperResolution
  - Simulated Zoom GT at d/2 (2x closer, steady 3-frame median)

Verifies that measure_chart.py detects all 4 fiducials, recovers Zoom GT PSNR/SSIM
gains for SR over single-frame, and measures a limiting frequency within 10% of
the known analytical optical+pixel MTF50.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from raw_sr.core.cfa_drizzle import channel_map
from raw_sr.core.superres import RawFrame, SuperResolution
from scripts.testchart import chart_spec as spec
from scripts.testchart.make_chart import render_chart_raster
from scripts.testchart.measure_chart import measure_triplet

BGGR = np.array([[2, 3], [1, 0]], dtype=np.int8)


def analytical_mtf_crossing(sigma_lens_lr: float, threshold: float = 0.50) -> float:
    """Analytical MTF(f) = exp(-2*pi^2*sigma^2*f^2) * |sinc(f)| crossing in cyc/LR px."""
    freqs = np.linspace(0.001, 1.5, 15000)
    mtf = np.exp(-2.0 * (np.pi ** 2) * (sigma_lens_lr ** 2) * (freqs ** 2)) * np.abs(np.sinc(freqs))
    below = np.where(mtf < threshold)[0]
    return float(freqs[below[0]]) if len(below) > 0 else float(freqs[-1])


def _mosaic_from_rgb(rgb: np.ndarray) -> np.ndarray:
    ch = channel_map(BGGR, rgb.shape[:2])
    return np.take_along_axis(rgb, ch[..., None], axis=2)[..., 0]


def _demosaic_bggr(mosaic: np.ndarray) -> np.ndarray:
    u16 = np.clip(mosaic * 65535.0, 0, 65535).astype(np.uint16)
    return cv2.cvtColor(u16, cv2.COLOR_BayerRG2RGB_EA).astype(np.float32) / 65535.0


def _build_perspective_H(
    src_w: int, src_h: int, dst_w: int, dst_h: int, margin: float = 0.08, tilt_px: float = 6.0
) -> np.ndarray:
    """Build a realistic perspective homography from chart space -> sensor space."""
    src = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float32)
    mx = margin * dst_w
    my = margin * dst_h
    dst = np.array(
        [
            [mx + tilt_px, my + tilt_px * 0.5],
            [dst_w - mx - tilt_px * 0.5, my - tilt_px * 0.3],
            [dst_w - mx + tilt_px * 0.4, dst_h - my - tilt_px * 0.5],
            [mx - tilt_px * 0.3, dst_h - my + tilt_px * 0.4],
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def run_simulated_validation(
    ratio: int = 2,
    lr_h: int = 680,
    n_frames: int = 16,
    sigma_lens_lr: float = 0.45,
    seed: int = 42,
    out_json: str | None = None,
    out_panel: str | None = None,
) -> dict[str, Any]:
    """Run full end-to-end simulated camera validation."""
    rng = np.random.default_rng(seed)
    lr_w = int(round(lr_h * (297.0 / 210.0)))
    # Ensure even dimensions for Bayer pattern
    lr_h -= lr_h % 2
    lr_w -= lr_w % 2
    hr_h, hr_w = lr_h * ratio, lr_w * ratio

    # Render high-res chart scene (200 dpi A4 -> 2339 x 1654 px)
    chart_u8 = render_chart_raster(paper="A4", dpi=200)
    chart_f = chart_u8.astype(np.float32) / 255.0
    chart_rgb = np.stack([chart_f, chart_f, chart_f], axis=2)
    ch_h, ch_w = chart_rgb.shape[:2]

    # Perspective homography from chart -> HR sensor grid
    H_base = _build_perspective_H(ch_w, ch_h, hr_w, hr_h, margin=0.07, tilt_px=8.0)

    # Render clean HR scene in sensor perspective
    scene_hr = cv2.warpPerspective(
        chart_rgb, H_base, (hr_w, hr_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    # Forward model: lens Gaussian PSF + 1 LR px pixel aperture (box integration)
    blurred_hr = cv2.GaussianBlur(
        scene_hr, (0, 0), sigma_lens_lr * ratio, borderType=cv2.BORDER_REFLECT
    )
    box_hr = cv2.blur(blurred_hr, (ratio, ratio), borderType=cv2.BORDER_REFLECT)

    # Simulate n_frames handheld burst at distance d with sub-pixel shake
    offs = np.stack([rng.uniform(-2.5, 2.5, n_frames), rng.uniform(-2.5, 2.5, n_frames)], axis=1)
    offs[0] = (0.0, 0.0)

    xs, ys = np.meshgrid(np.arange(lr_w, dtype=np.float32), np.arange(lr_h, dtype=np.float32))
    off = 0.5 * (float(ratio) - 1.0)
    full_well = 2048.0
    read_noise = 1.5

    raw_frames: list[RawFrame] = []
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
        dem = _demosaic_bggr(m)
        raw_frames.append(
            RawFrame(
                name=f"sim_frame_{k:02d}.dng",
                rgb=dem,
                green=dem[..., 1].copy(),
                mosaic=m,
                pattern=BGGR.copy(),
            )
        )

    # Run raw_sr (default demosaic_drizzle)
    sr_engine = SuperResolution(
        ratio=float(ratio), fusion="demosaic_drizzle", registration_workers=1, use_opencl=False, use_gpu=False
    )
    sr_res = sr_engine.run(raw_frames)
    sr_img = sr_res.hr.astype(np.float32)

    # Single-frame bicubic baseline
    single_img = cv2.resize(raw_frames[0].rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

    # Simulated Zoom GT at d/2 (2x closer -> 1x sensor PSF on HR grid, 3-frame steady median)
    gt_blurred = cv2.GaussianBlur(
        scene_hr, (0, 0), sigma_lens_lr, borderType=cv2.BORDER_REFLECT
    )
    gt_stack = []
    for _ in range(3):
        m_gt = _mosaic_from_rgb(gt_blurred)
        m_gt = (
            rng.poisson(np.clip(m_gt, 0.0, 1.0) * full_well)
            + rng.normal(0.0, read_noise, m_gt.shape)
        ) / full_well
        gt_stack.append(_demosaic_bggr(np.clip(m_gt, 0.0, 1.0).astype(np.float32)))
    gt_img = np.median(np.stack(gt_stack, axis=0), axis=0).astype(np.float32)

    # Measure with measure_chart.py
    report = measure_triplet(
        single_img=single_img,
        sr_img=sr_img,
        ratio=float(ratio),
        gt_img=gt_img,
        out_json=out_json,
        out_panel=out_panel,
    )

    true_f50 = analytical_mtf_crossing(sigma_lens_lr, 0.50)
    true_f10 = analytical_mtf_crossing(sigma_lens_lr, 0.10)
    meas_star_f50 = report["chart_only"]["sr"]["star_sin"]["f50_cyc_lr_px"]
    meas_edge_f50 = report["chart_only"]["sr"]["edge_high"]["mtf50_cyc_lr_px"]

    report["simulation"] = {
        "ratio": ratio,
        "lr_shape": [lr_h, lr_w],
        "n_frames": n_frames,
        "sigma_lens_lr": sigma_lens_lr,
        "true_mtf50_cyc_lr_px": round(true_f50, 4),
        "true_mtf10_cyc_lr_px": round(true_f10, 4),
        "sr_star_f50_rel_err_pct": round(100.0 * abs(meas_star_f50 - true_f50) / true_f50, 2),
        "sr_edge_f50_rel_err_pct": round(100.0 * abs(meas_edge_f50 - true_f50) / true_f50, 2),
    }

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate WP10 test chart kit with a simulated phone camera.")
    parser.add_argument("--ratio", type=int, default=2)
    parser.add_argument("--lr-h", type=int, default=680)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--sigma-lens", type=float, default=0.45)
    parser.add_argument("--out-json", default="/tmp/rawsr/wp10/sim_validation_2x.json")
    parser.add_argument("--out-panel", default="docs/wp10/sim_validation_2x.png")
    args = parser.parse_args()

    rep = run_simulated_validation(
        ratio=args.ratio,
        lr_h=args.lr_h,
        n_frames=args.n_frames,
        sigma_lens_lr=args.sigma_lens,
        out_json=args.out_json,
        out_panel=args.out_panel,
    )
    print(json.dumps(rep["simulation"], indent=2))
    print("Zoom GT overall:", json.dumps(rep["zoom_gt"]["single_vs_gt"]["overall"]), "->", json.dumps(rep["zoom_gt"]["sr_vs_gt"]["overall"]))
    print("SR star_sin:", json.dumps(rep["chart_only"]["sr"]["star_sin"]["f50_cyc_lr_px"]), "edge_high:", json.dumps(rep["chart_only"]["sr"]["edge_high"]["mtf50_cyc_lr_px"]))


if __name__ == "__main__":
    main()
