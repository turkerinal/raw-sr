"""Quick ablation of cfa_kr kernel parameters (k_detail, k_denoise, k_stretch,
chroma_scale) on synthetic 2x and real bursts (pixel4a20_first9, own9, 0919).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from audit_c11_part2 import synthesize_target_burst, _demosaic_bggr, BGGR, psnr
from raw_sr.core.superres import SuperResolution, load_raw_linear, estimate_homography
from raw_sr.core.cfa_drizzle import (
    cfa_drizzle, cfa_kernel_regression, load_cfa_frame,
    calibrate_cfa_levels, apply_level_gains, pattern_array,
)
from raw_sr.core.cfa_heldout import evaluate_cfa_heldout


def _gain_fit(img, target, mask):
    x = img[..., 1][mask].astype(np.float64)
    y = target[..., 1][mask].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], 1)
    (g, o), *_ = np.linalg.lstsq(A, y, rcond=None)
    return (img * g + o).astype(np.float32)


def main():
    configs = [
        ("old_kd18_dn55_85", dict(k_detail=0.18, k_denoise_min=0.55, k_denoise_max=0.85, k_stretch=2.8, chroma_scale=1.25)),
        ("dn30_45_kd18",     dict(k_detail=0.18, k_denoise_min=0.30, k_denoise_max=0.45, k_stretch=2.2, chroma_scale=1.15)),
        ("dn22_35_kd16",     dict(k_detail=0.16, k_denoise_min=0.22, k_denoise_max=0.35, k_stretch=2.0, chroma_scale=1.10)),
        ("dn18_28_kd15",     dict(k_detail=0.15, k_denoise_min=0.18, k_denoise_max=0.28, k_stretch=1.8, chroma_scale=1.05)),
        ("dn25_40_kd18_cs1", dict(k_detail=0.18, k_denoise_min=0.25, k_denoise_max=0.40, k_stretch=2.0, chroma_scale=1.00)),
    ]

    # 1. Synthetic 2x (exact & 0.1px reg noise)
    b = synthesize_target_burst(ratio=2, lr=256, n=9, seed=7)
    mos = b["mosaics"]
    dems = [_demosaic_bggr(m) for m in mos]
    ref_g = dems[0][..., 1].copy()
    target, mask = b["target"], b["mask"]
    oh, ow = target.shape[:2]
    fb_hr = cv2.resize(dems[0], (ow, oh), interpolation=cv2.INTER_CUBIC)
    base_offs = -np.asarray(b["offsets"], dtype=np.float64)
    al_exact = [(0.0, 0.0)] + [(float(dx), float(dy)) for dx, dy in base_offs[1:]]
    rng = np.random.default_rng(42)
    al_01 = [(0.0, 0.0)] + [(float(dx + rng.normal(0, 0.1)), float(dy + rng.normal(0, 0.1))) for dx, dy in base_offs[1:]]

    print("=== Synthetic 2x GT PSNR (dB) ===")
    p_drz0 = psnr(_gain_fit(cfa_drizzle(mos, al_exact, 2.0, BGGR, pixfrac=0.5).hr, target, mask), target, mask)
    p_drz1 = psnr(_gain_fit(cfa_drizzle(mos, al_01, 2.0, BGGR, pixfrac=0.5).hr, target, mask), target, mask)
    print(f"  {'cfa_drizzle':20s} | exact={p_drz0:.2f} | reg0.1={p_drz1:.2f}")
    for name, kw in configs:
        p0 = psnr(_gain_fit(cfa_kernel_regression(mos, al_exact, 2.0, BGGR, ref_green=ref_g, fallback_hr=fb_hr, **kw).hr, target, mask), target, mask)
        p1 = psnr(_gain_fit(cfa_kernel_regression(mos, al_01, 2.0, BGGR, ref_green=ref_g, fallback_hr=fb_hr, **kw).hr, target, mask), target, mask)
        print(f"  {name:20s} | exact={p0:.2f} | reg0.1={p1:.2f}")

    # 2. Real bursts (1 held-out frame each)
    bursts = [
        ("own9", os.path.join(ROOT, "test_data"), (1650, 1240)),
        ("hdrplus_0919", "data/hdrplus/0919_20150910_131720_137", (1504, 1000)),
        ("pixel4a20_first9", "data/inria_friant20_first9", (2080, 800)),
    ]
    for bname, bdir, roi_xy in bursts:
        files = sorted(glob.glob(os.path.join(bdir, "*.dng")))
        roi = (roi_xy[0], roi_xy[1], 1024, 1024)
        frames = [load_raw_linear(f, roi=roi) for f in files]
        cm = [load_cfa_frame(f, roi=roi) for f in files]
        for i, cf in enumerate(cm):
            g, _ = calibrate_cfa_levels(cf, frames[i].rgb)
            cm[i] = apply_level_gains(cf, g)
        mos_r = [c.mosaic for c in cm]
        pat = pattern_array(cm[0].pattern)
        hidx = len(frames) // 2
        kept = [f for i, f in enumerate(frames) if i != hidx]
        kept_m = [m for i, m in enumerate(mos_r) if i != hidx]
        res_def = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(kept)
        ki = list(res_def.kept_indices)
        km = [kept_m[i] for i in ki]
        ref_g_r = kept[res_def.reference_index].green
        oh_r, ow_r = res_def.hr.shape[:2]
        fb_r = cv2.resize(kept[res_def.reference_index].rgb, (ow_r, oh_r), interpolation=cv2.INTER_CUBIC)
        hrs = {
            "cfa_drizzle": cfa_drizzle(km, res_def.transforms, 2.0, pat, res_def.weights.tolist(), pixfrac=0.5).hr,
        }
        for name, kw in configs:
            hrs[name] = cfa_kernel_regression(
                km, res_def.transforms, 2.0, pat, ref_green=ref_g_r,
                weights=res_def.weights.tolist(), fallback_hr=fb_r, **kw,
            ).hr
        W_held = estimate_homography(kept[res_def.reference_index].green, frames[hidx].green, ecc_scale=0.25)
        ev = evaluate_cfa_heldout(hrs, mos_r[hidx], km, res_def.transforms, res_def.weights.tolist(), W_held, 2.0, pat)
        gains = {k: round(v["heldout_cfa_gain_db"], 3) for k, v in ev["per_method"].items()}
        print(f"\n=== Real {bname} (h={hidx}) CFA held-out gain (dB) ===")
        for k, v in gains.items():
            print(f"  {k:20s}: {v:+.3f} dB")


if __name__ == "__main__":
    main()
