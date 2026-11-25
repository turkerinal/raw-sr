#!/usr/bin/env python3
"""WP6 Evaluation: Synthetic PSF sweep + Real 2x 4-scene benchmark & visual panels."""

from __future__ import annotations

import glob
import json
import math
import os
import sys

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
from raw_sr.core.superres import RawFrame, SuperResolution, load_raw_linear
from scripts.audit_c11_part2 import (
    _demosaic_bggr,
    compute_slanted_edge_mtf50,
    immerkaer_sigma,
    psnr,
    synthesize_target_burst,
)
from scripts.make_showcase import (
    choose_safe_exposure,
    find_best_edge_roi,
    find_flat_patch,
    measure_phase_shift,
    pick_high_detail_crop,
    pick_zoom_subcrop,
    render_showcase,
)


def measure_edge_overshoot_pct(img: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    """Measure edge overshoot (%) relative to step height on a slanted-edge ROI."""
    gray = img[..., 1].astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    x0, y0, rw, rh = roi
    patch = gray[y0:y0 + rh, x0:x0 + rw]
    p05 = float(np.percentile(patch, 5))
    p95 = float(np.percentile(patch, 95))
    step = max(p95 - p05, 1e-4)
    over_hi = max(0.0, float(patch.max()) - p95)
    over_lo = max(0.0, p05 - float(patch.min()))
    return round(100.0 * max(over_hi, over_lo) / step, 2)


def run_synthetic() -> list[dict]:
    print("=== SYNTHETIC 2x PSF SWEEP ===")
    rows = []
    for true_sigma in (0.30, 0.45, 0.60, 0.80):
        b = synthesize_target_burst(ratio=2, lr=256, n=9, seed=7, sigma_lens_lr=true_sigma)
        mos, mask = b["mosaics"], b["mask"]
        target_sensor, target_lens = b["target"], b["blurred"]
        dems = [_demosaic_bggr(m) for m in mos]
        frames = [RawFrame(str(k), d, d[..., 1].copy()) for k, d in enumerate(dems)]
        hr_h, hr_w = target_sensor.shape[:2]

        bic = cv2.resize(frames[0].rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
        sr = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, registration="homography", ecc_scale=0.25)
        res_sr = sr.run(frames, reference_index=0)
        sr_noise = effective_stack_noise(
            res_sr.metrics.get("frame_noise_sigma"), effective_frames=res_sr.used_frames
        )

        wp2_sr = enhance_image(res_sr.hr, ratio=2.0, noise_sigma=sr_noise, strength=DEFAULT_STRENGTH, mode="wp2")
        wp6_sr, wp6_bic = enhance_fair_pair(
            res_sr.hr, bic, ratio=2.0, sr_noise_sigma=sr_noise, strength=DEFAULT_STRENGTH, mode="adaptive"
        )

        edge_roi = (int(hr_w * 0.17), int(hr_h * 0.2), int(hr_w * 0.16), int(hr_h * 0.6))
        flat_sl = (slice(32, 128), slice(32, 96))

        def _metrics(im: np.ndarray) -> dict:
            return {
                "mtf50": round(compute_slanted_edge_mtf50(im, ratio=2.0, roi=edge_roi), 4),
                "psnr_sensor": round(psnr(im, target_sensor, mask), 3),
                "psnr_lens": round(psnr(im, target_lens, mask), 3),
                "overshoot_pct": measure_edge_overshoot_pct(im, edge_roi),
                "flat_noise": round(float(im[flat_sl][..., 1].std()), 6),
            }

        wp2_err = round(100.0 * abs(wp2_sr.optics_sigma_lr - true_sigma) / true_sigma, 1)
        wp6_err = round(100.0 * abs(wp6_sr.optics_sigma_lr - true_sigma) / true_sigma, 1)
        row = {
            "true_sigma_lr": true_sigma,
            "wp2_sigma_est": round(wp2_sr.optics_sigma_lr, 4),
            "wp2_sigma_err_pct": wp2_err,
            "wp6_sigma_est": round(wp6_sr.optics_sigma_lr, 4),
            "wp6_sigma_err_pct": wp6_err,
            "wp6_backoff": wp6_sr.details.get("safety_backoff_factor", 1.0),
            "single_bicubic": _metrics(bic),
            "single_wp6": _metrics(wp6_bic.image),
            "plain_sr": _metrics(res_sr.hr),
            "wp2_sr": _metrics(wp2_sr.image),
            "wp6_sr": _metrics(wp6_sr.image),
        }
        rows.append(row)
        print(
            f"sigma={true_sigma:.2f} | WP2 est={wp2_sr.optics_sigma_lr:.3f} ({wp2_err:4.1f}%) "
            f"MTF={row['wp2_sr']['mtf50']:.3f} PSNR_s={row['wp2_sr']['psnr_sensor']:.2f} | "
            f"WP6 est={wp6_sr.optics_sigma_lr:.3f} ({wp6_err:4.1f}%) "
            f"MTF={row['wp6_sr']['mtf50']:.3f} PSNR_s={row['wp6_sr']['psnr_sensor']:.2f} "
            f"PSNR_l={row['wp6_sr']['psnr_lens']:.2f} ov={row['wp6_sr']['overshoot_pct']:.1f}% "
            f"bo={row['wp6_backoff']}"
        )
    return rows


def run_real_scenes() -> list[dict]:
    print("\n=== REAL 2x 4-SCENE BENCHMARK ===")
    scenes = [
        ("hdrplus_0919", "HDR+ 0919", "data/hdrplus/0919_20150910_131720_137", None),
        ("hdrplus_0382", "HDR+ 0382", "data/hdrplus/0382_20150912_130914_161", None),
        ("own9", "Own 9-frame", os.path.join(ROOT, "test_data"), None),
        ("pixel4a_20", "Pixel4a 20", "data/inria_friant20", (2080, 800, 1024, 1024)),
    ]
    out_dir = os.path.join(ROOT, "docs", "wp6")
    os.makedirs(out_dir, exist_ok=True)
    results = []

    for slug, name, bdir, roi in scenes:
        files = sorted(glob.glob(os.path.join(bdir, "*.dng")))
        if not files:
            raise RuntimeError(f"No DNG files in {bdir}")
        if roi is None:
            import rawpy
            with rawpy.imread(files[0]) as raw:
                H, W = raw.sizes.height, raw.sizes.width
            sz = min(1024, H, W)
            roi = (((W - sz) // 2) & ~1, ((H - sz) // 2) & ~1, sz, sz)

        frames = [load_raw_linear(f, roi=roi) for f in files]
        sr = SuperResolution(ratio=2.0, registration="homography", ecc_scale=0.25, max_hr_pixels=20_000_000)
        res = sr.run(frames)
        ref_i = int(res.reference_index)
        hr_h, hr_w = res.hr.shape[:2]
        bic = cv2.resize(frames[ref_i].rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

        sr_noise = effective_stack_noise(
            res.metrics.get("frame_noise_sigma"), effective_frames=res.used_frames
        )
        wp2_sr = enhance_image(res.hr, ratio=2.0, noise_sigma=sr_noise, strength=DEFAULT_STRENGTH, mode="wp2")
        wp6_sr, wp6_bic = enhance_fair_pair(
            res.hr, bic, ratio=2.0, sr_noise_sigma=sr_noise, strength=DEFAULT_STRENGTH, mode="adaptive"
        )

        x0, y0 = pick_high_detail_crop(res.hr, bic=bic, crop_size=320, margin=96)
        sl = (slice(y0, y0 + 320), slice(x0, x0 + 320))
        dx, dy = measure_phase_shift(bic[sl], res.hr[sl])
        offset_norm = math.hypot(dx, dy)

        edge_roi = find_best_edge_roi(res.hr[..., 1], win=160)
        flat_roi = find_flat_patch(res.hr[..., 1], win=96)
        fx, fy, fw, fh = flat_roi

        imgs = {
            "single_bicubic": bic,
            "single_wp6": wp6_bic.image,
            "plain_sr": res.hr,
            "wp2_sr": wp2_sr.image,
            "wp6_sr": wp6_sr.image,
        }
        metrics = {}
        for k, im in imgs.items():
            metrics[k] = {
                "mtf50": round(compute_slanted_edge_mtf50(im, ratio=2.0, roi=edge_roi), 4),
                "flat_immerkaer": round(immerkaer_sigma(im[fy:fy + fh, fx:fx + fw, 1]), 6),
                "flat_std": round(float(im[fy:fy + fh, fx:fx + fw, 1].std()), 6),
                "overshoot_pct": measure_edge_overshoot_pct(im, edge_roi),
            }

        crops = [bic[sl], res.hr[sl], wp2_sr.image[sl], wp6_sr.image[sl]]
        expo, _ = choose_safe_exposure(crops)
        sub_xy = pick_zoom_subcrop(crops[1], sub_size=160)
        title = (
            f"{name} | 2x | {res.used_frames}/{len(frames)} frames | ref #{ref_i} "
            f"| shift {offset_norm:.2f} HR px | PSF {wp6_sr.optics_sigma_lr:.2f} LR px"
        )
        labelled = [
            ("1. Single (bicubic)", crops[0]),
            ("2. Plain SR", crops[1]),
            ("3. WP2 enhanced", crops[2]),
            ("4. WP6 enhanced", crops[3]),
        ]
        png_path = os.path.join(out_dir, f"{slug}_2x_wp6.png")
        img_info = render_showcase(labelled, sub_xy=sub_xy, sub_size=160, exposure=expo, title=title, out_path=png_path)

        scene_res = {
            "slug": slug,
            "name": name,
            "frames_used": int(res.used_frames),
            "frames_total": len(frames),
            "ref_idx": ref_i,
            "shift_hr_px": round(offset_norm, 3),
            "wp2_sigma_est": round(wp2_sr.optics_sigma_lr, 4),
            "wp6_sigma_est": round(wp6_sr.optics_sigma_lr, 4),
            "wp6_sigma_rgb": [
                round(wp6_sr.details.get("psf_sigma_r", wp6_sr.optics_sigma_lr), 4),
                round(wp6_sr.details.get("psf_sigma_g", wp6_sr.optics_sigma_lr), 4),
                round(wp6_sr.details.get("psf_sigma_b", wp6_sr.optics_sigma_lr), 4),
            ],
            "wp6_backoff": wp6_sr.details.get("safety_backoff_factor", 1.0),
            "wp6_stop_reason": wp6_sr.rl_stopped_reason,
            "wp6_iters": wp6_sr.rl_iterations_run,
            "png_kb": img_info["size_kb"],
            "metrics": metrics,
        }
        results.append(scene_res)
        print(
            f"[{name}] used={res.used_frames}/{len(frames)} PSF={wp6_sr.optics_sigma_lr:.3f} "
            f"bo={scene_res['wp6_backoff']} stop={wp6_sr.rl_stopped_reason}({wp6_sr.rl_iterations_run}) "
            f"png={img_info['size_kb']}KB\n"
            f"  single={metrics['single_bicubic']['mtf50']:.3f} "
            f"plain={metrics['plain_sr']['mtf50']:.3f} "
            f"wp2={metrics['wp2_sr']['mtf50']:.3f} "
            f"wp6={metrics['wp6_sr']['mtf50']:.3f} | "
            f"flat_std: plain={metrics['plain_sr']['flat_std']:.5f} "
            f"wp2={metrics['wp2_sr']['flat_std']:.5f} "
            f"wp6={metrics['wp6_sr']['flat_std']:.5f} | "
            f"ov%: plain={metrics['plain_sr']['overshoot_pct']} "
            f"wp2={metrics['wp2_sr']['overshoot_pct']} "
            f"wp6={metrics['wp6_sr']['overshoot_pct']}"
        )
    return results


def main() -> int:
    synth = run_synthetic()
    real = run_real_scenes()
    out = {"synthetic": synth, "real": real}
    with open("/tmp/rawsr/wp6_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nSaved /tmp/rawsr/wp6_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
