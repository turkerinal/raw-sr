"""WP9 full evaluation:
1. Synthetic 2x GT PSNR & MTF50 (default, cfa_drizzle, cfa_kr old, cfa_kr new adaptive).
2. Real 2x on all 5 bursts (own9, HDR+ 0919, HDR+ 0382, Samsung 14, Pixel 4a 20 first9):
   - CFA-space held-out gain (mean over 3 held-out frames)
   - DCB-space held-out gain (mean over 3 held-out frames)
3. 4-panel visual comparisons (default | cfa_drizzle | cfa_kr old | cfa_kr new):
   - docs/wp9/<burst>_2x_wp9_4panel.png (< 1.2 MB) for 4 publishable bursts
   - /tmp/rawsr/wp9/pixel4a20_2x_wp9_4panel.png (internal only, never committed)
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from audit_c11_part2 import (
    synthesize_target_burst, _demosaic_bggr, BGGR,
    compute_slanted_edge_mtf50, psnr,
)
from raw_sr.core.superres import (
    RawFrame, SuperResolution, load_raw_linear, estimate_homography,
    predict_lr_from_hr, denoised_lr_stack, lr_sampling_valid_mask,
    SENSOR_PSF_SIGMA_PER_RATIO,
)
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


def _dcb_heldout_gain(hr, held_rgb, fused_frames, transforms, weights, W_held, ratio):
    fsig = SENSOR_PSF_SIGMA_PER_RATIO * ratio
    h, w = held_rgb.shape[:2]
    pred = predict_lr_from_hr(hr, W_held, ratio, (h, w), forward_sigma=fsig)
    base_ref, cov_ref = denoised_lr_stack(
        fused_frames, transforms, weights,
        coverage_normalised=True, return_coverage=True,
    )
    W_h = np.asarray(W_held, dtype=np.float64)
    if W_h.shape == (2, 3):
        W_h = np.vstack([W_h, [0.0, 0.0, 1.0]])
    base = cv2.warpPerspective(base_ref, W_h, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    cov = cv2.warpPerspective(cov_ref.astype(np.float32), W_h, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    inner = np.zeros((h, w), dtype=bool)
    inner[24:h - 24, 24:w - 24] = True
    supp = lr_sampling_valid_mask(hr.shape[:2], W_h, ratio, (h, w), forward_sigma=fsig)
    vmask = inner & supp & (cov >= 0.99 * float(np.sum(weights)))
    mse_p = float(np.mean((pred[vmask] - held_rgb[vmask]) ** 2))
    mse_b = float(np.mean((base[vmask] - held_rgb[vmask]) ** 2))
    psnr_p = 10.0 * np.log10(1.0 / max(mse_p, 1e-30))
    psnr_b = 10.0 * np.log10(1.0 / max(mse_b, 1e-30))
    return psnr_p - psnr_b


def _to_srgb_u8(img, scale=None):
    x = np.clip(img, 0.0, None)
    if scale is None:
        p99 = float(np.percentile(x, 99.5))
        scale = 1.0 / max(p99, 1e-4)
    x = np.clip(x * scale, 0.0, 1.0)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 1e-8), 1.0 / 2.4) - 0.055)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8), scale


def _save_4panel(hrs_dict, out_path, title):
    # Pick a 384x384 high-contrast center crop from the 2048x2048 HR
    keys = ["default", "cfa_drizzle", "cfa_kr_old", "cfa_kr_new"]
    labels = ["default (demosaic+IBP)", "cfa_drizzle (pf=0.5)", "cfa_kr old (IBP=1)", "cfa_kr new (adaptive, IBP=0)"]
    H, W = hrs_dict["default"].shape[:2]
    cy, cx = H // 2, W // 2
    cs = 384
    y0, x0 = max(0, cy - cs // 2), max(0, cx - cs // 2)
    crops = [hrs_dict[k][y0:y0 + cs, x0:x0 + cs] for k in keys]
    _, scale = _to_srgb_u8(crops[0])
    tiles = []
    for crop, lbl in zip(crops, labels):
        u8, _ = _to_srgb_u8(crop, scale=scale)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (cs, 28), (20, 20, 20), -1)
        cv2.putText(bgr, lbl, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(bgr)
    row1 = np.hstack([tiles[0], tiles[1]])
    row2 = np.hstack([tiles[2], tiles[3]])
    panel = np.vstack([row1, row2])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, panel, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    kb = os.path.getsize(out_path) / 1024.0
    print(f"  saved panel {out_path} ({kb:.1f} KB)")
    return kb


def eval_synthetic():
    b = synthesize_target_burst(ratio=2, lr=256, n=9, seed=7)
    mos = b["mosaics"]
    dems = [_demosaic_bggr(m) for m in mos]
    frames = [RawFrame(str(k), d, d[..., 1].copy(), mosaic=m, pattern=BGGR)
              for k, (d, m) in enumerate(zip(dems, mos))]
    target, mask = b["target"], b["mask"]
    oh, ow = target.shape[:2]
    fb_hr = cv2.resize(dems[0], (ow, oh), interpolation=cv2.INTER_CUBIC)

    sr_def = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, registration="homography", ecc_scale=0.25)
    r_def = sr_def.run(frames, reference_index=0)

    sr_kr_old = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, registration="homography", ecc_scale=0.25, fusion="cfa_kr")
    r_kr_old = sr_kr_old.run(frames, reference_index=0)

    r_drz = cfa_drizzle(mos, r_def.transforms, 2.0, BGGR, r_def.weights.tolist(), pixfrac=0.5)
    r_kr_new = cfa_kernel_regression(
        mos, r_def.transforms, 2.0, BGGR, ref_green=dems[0][..., 1],
        weights=r_def.weights.tolist(), fallback_hr=fb_hr,
        adaptive=True, gr_gb_balance=True,
    )

    out = {}
    for name, hr in [
        ("default", r_def.hr),
        ("cfa_drizzle", r_drz.hr),
        ("cfa_kr_old", r_kr_old.hr),
        ("cfa_kr_new", r_kr_new.hr),
    ]:
        fit = _gain_fit(hr, target, mask)
        out[name] = {
            "psnr_db": round(psnr(fit, target, mask), 2),
            "mtf50": round(compute_slanted_edge_mtf50(fit, 2.0), 4),
        }
    print("=== Synthetic 2x (estimated reg) ===")
    print(json.dumps(out, indent=2))
    return out


def eval_real_burst(name, burst_dir, roi_xy, panel_path):
    files = sorted(glob.glob(os.path.join(burst_dir, "*.dng")))
    roi = (roi_xy[0], roi_xy[1], 1024, 1024)
    frames = [load_raw_linear(f, roi=roi) for f in files]
    cm = [load_cfa_frame(f, roi=roi) for f in files]
    for i, cf in enumerate(cm):
        g, _ = calibrate_cfa_levels(cf, frames[i].rgb)
        cm[i] = apply_level_gains(cf, g)
    mos = [c.mosaic for c in cm]
    pat = pattern_array(cm[0].pattern)
    n = len(frames)
    hids = sorted(set([0, n // 2, n - 1]))

    cfa_gains = {k: [] for k in ("default", "cfa_drizzle", "cfa_kr_old", "cfa_kr_new")}
    dcb_gains = {k: [] for k in ("default", "cfa_drizzle", "cfa_kr_old", "cfa_kr_new")}

    for hidx in hids:
        kept = [f for i, f in enumerate(frames) if i != hidx]
        kept_m = [m for i, m in enumerate(mos) if i != hidx]
        res_def = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(kept)
        res_kr_old = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, fusion="cfa_kr").run(kept)

        ki = list(res_def.kept_indices)
        km = [kept_m[i] for i in ki]
        fused_frames = [kept[i] for i in ki]
        ref_g = kept[res_def.reference_index].green
        oh, ow = res_def.hr.shape[:2]
        fb_hr = cv2.resize(kept[res_def.reference_index].rgb, (ow, oh), interpolation=cv2.INTER_CUBIC)
        reg_sig = float(np.median([x for x in res_def.align_residuals if x > 0])) if any(x > 0 for x in res_def.align_residuals) else 0.0

        hr_drz = cfa_drizzle(km, res_def.transforms, 2.0, pat, res_def.weights.tolist(), pixfrac=0.5).hr
        hr_kr_new = cfa_kernel_regression(
            km, res_def.transforms, 2.0, pat, ref_green=ref_g,
            weights=res_def.weights.tolist(), fallback_hr=fb_hr,
            adaptive=True, gr_gb_balance=True, reg_sigma_lr=reg_sig,
        ).hr

        hrs = {
            "default": res_def.hr,
            "cfa_drizzle": hr_drz,
            "cfa_kr_old": res_kr_old.hr,
            "cfa_kr_new": hr_kr_new,
        }
        W_held = estimate_homography(kept[res_def.reference_index].green, frames[hidx].green, ecc_scale=0.25)
        ev_cfa = evaluate_cfa_heldout(hrs, mos[hidx], km, res_def.transforms, res_def.weights.tolist(), W_held, 2.0, pat)
        for k in hrs:
            cfa_gains[k].append(round(ev_cfa["per_method"][k]["heldout_cfa_gain_db"], 3))
            dcb_gains[k].append(round(_dcb_heldout_gain(hrs[k], frames[hidx].rgb, fused_frames, res_def.transforms, res_def.weights.tolist(), W_held, 2.0), 3))

        if hidx == n // 2 and panel_path:
            _save_4panel(hrs, panel_path, name)

    summ = {
        "n_frames": n,
        "heldout_indices": hids,
        "cfa_mean_db": {k: round(float(np.mean(v)), 3) for k, v in cfa_gains.items()},
        "dcb_mean_db": {k: round(float(np.mean(v)), 3) for k, v in dcb_gains.items()},
        "cfa_per_frame": cfa_gains,
        "dcb_per_frame": dcb_gains,
    }
    print(f"=== Real {name} (n={n}) ===")
    print("  CFA held-out mean (dB):", json.dumps(summ["cfa_mean_db"]))
    print("  DCB held-out mean (dB):", json.dumps(summ["dcb_mean_db"]))
    return summ


def main():
    res = {"synthetic": eval_synthetic(), "real": {}}
    bursts = [
        ("own9", os.path.join(ROOT, "test_data"), (1650, 1240), os.path.join(ROOT, "docs/wp9/own9_2x_wp9_4panel.png")),
        ("hdrplus_0919", "data/hdrplus/0919_20150910_131720_137", (1504, 1000), os.path.join(ROOT, "docs/wp9/hdrplus_0919_2x_wp9_4panel.png")),
        ("hdrplus_0382", "data/hdrplus/0382_20150912_130914_161", (1504, 1000), os.path.join(ROOT, "docs/wp9/hdrplus_0382_2x_wp9_4panel.png")),
        ("samsung14", "data/samsung14", (2400, 520), os.path.join(ROOT, "docs/wp9/samsung14_2x_wp9_4panel.png")),
        ("pixel4a20_first9", "data/inria_friant20_first9", (2080, 800), "/tmp/rawsr/wp9/pixel4a20_2x_wp9_4panel.png"),
    ]
    for name, bdir, roi_xy, ppath in bursts:
        res["real"][name] = eval_real_burst(name, bdir, roi_xy, ppath)

    out_path = "/tmp/rawsr/wp9/eval_full.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
