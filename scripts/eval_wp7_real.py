#!/usr/bin/env python3
"""WP7 real-burst evaluation: DCB + CFA held-out, timing/RSS, and pixel-aligned panels."""

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
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from raw_sr.core.cfa_drizzle import (
    apply_level_gains,
    calibrate_cfa_levels,
    cfa_drizzle,
    channel_map,
    load_cfa_frame,
    pattern_array,
)
from raw_sr.core.cfa_heldout import evaluate_cfa_heldout
from raw_sr.core.superres import (
    SuperResolution,
    estimate_homography,
    load_raw_linear,
    predict_lr_from_hr,
)
from sr_acceptance import heldout_baseline_lr, heldout_compare, heldout_valid_mask


BURSTS = [
    ("own9", os.path.join(ROOT, "test_data"), (1650, 1240), True),
    ("hdrplus_0919", "data/hdrplus/0919_20150910_131720_137", (1504, 1000), True),
    ("hdrplus_0382", "data/hdrplus/0382_20150912_130914_161", (1504, 1000), True),
    ("samsung14", "data/samsung14", (2400, 520), True),
    ("pixel4a20", "data/inria_friant20", (2080, 800), False),
]


def level_fit(hr: np.ndarray, ref_mosaic: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    h, w = ref_mosaic.shape
    lr = cv2.resize(hr, (w, h), interpolation=cv2.INTER_AREA)
    ch = channel_map(pattern, (h, w))
    out = hr.copy()
    m = np.zeros((h, w), dtype=bool)
    m[32:-32, 32:-32] = True
    for c in range(3):
        sel = (ch == c) & m
        x = lr[..., c][sel].astype(np.float64)
        y = ref_mosaic[sel].astype(np.float64)
        (g, o), *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
        out[..., c] = hr[..., c] * g + o
    return out.astype(np.float32)


def save_panel(name: str, frames, r_m, r_r, public: bool) -> dict:
    out_dir = os.path.join(ROOT, "docs", "wp7") if public else "/tmp/rawsr/wp7"
    os.makedirs(out_dir, exist_ok=True)
    ref_idx = r_m.reference_index
    single = cv2.resize(
        frames[ref_idx].rgb,
        (r_m.hr.shape[1], r_m.hr.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    diff10 = np.clip(np.abs(r_r.hr - r_m.hr) * 10.0, 0.0, 1.0)

    s = 256
    g = cv2.cvtColor(np.clip(r_m.hr, 0, 1), cv2.COLOR_RGB2GRAY)
    e = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F)), -1, (s, s))
    e[:s, :] = 0
    e[-s:, :] = 0
    e[:, :s] = 0
    e[:, -s:] = 0
    cy, cx = np.unravel_index(np.argmax(e), e.shape)
    x0, y0 = int(cx - s // 2), int(cy - s // 2)

    imgs = [
        ("1. Single (bicubic)", single),
        ("2. master (reg_refine=0)", r_m.hr),
        ("3. reg_refine=2", r_r.hr),
        ("4. |diff| x10", diff10),
    ]
    crop_m = r_m.hr[y0:y0 + s, x0:x0 + s]
    gain = 0.85 / max(float(np.percentile(crop_m, 99.5)), 1e-6)

    tiles = []
    for label, im in imgs:
        c = im[y0:y0 + s, x0:x0 + s]
        if "diff" in label:
            u = (np.clip(c, 0, 1) * 255).astype(np.uint8)
        else:
            u = (np.clip(c * gain, 0, 1) ** (1.0 / 2.2) * 255).astype(np.uint8)
        u = cv2.resize(u, (2 * s, 2 * s), interpolation=cv2.INTER_NEAREST)
        u = cv2.cvtColor(u, cv2.COLOR_RGB2BGR)
        cv2.rectangle(u, (0, 0), (2 * s, 30), (20, 20, 20), -1)
        cv2.putText(u, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(u)

    gap = np.full((2 * s, 6, 3), 30, np.uint8)
    top = np.hstack([tiles[0], gap, tiles[1]])
    bot = np.hstack([tiles[2], gap, tiles[3]])
    canvas = np.vstack([top, np.full((6, top.shape[1], 3), 30, np.uint8), bot])
    path = os.path.join(out_dir, f"{name}_2x.png")
    cv2.imwrite(path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    size_kb = os.path.getsize(path) / 1024.0
    return {"panel": path, "size_kb": round(size_kb, 1), "crop": [x0, y0]}


def eval_burst(name: str, burst_dir: str, roi_xy: tuple[int, int], public: bool,
               n_heldout: int = 5) -> dict:
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

    # Full-burst run for timing, RSS, and visual panel
    t0 = time.perf_counter()
    r_m_full = SuperResolution(
        ratio=2.0, blur_threshold_ratio=0.0, ecc_scale=0.25,
        reg_refine=0, use_gpu=False, use_opencl=False, use_gpu_scatter=False,
    ).run(frames)
    t_master = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_r_full = SuperResolution(
        ratio=2.0, blur_threshold_ratio=0.0, ecc_scale=0.25,
        reg_refine=2, use_gpu=False, use_opencl=False, use_gpu_scatter=False,
    ).run(frames)
    t_refine = time.perf_counter() - t0
    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    panel_info = save_panel(name, frames, r_m_full, r_r_full, public=public)

    # Held-out sweep (5 evenly spaced frames)
    hids = sorted(set(int(round(v)) for v in np.linspace(0, n - 1, min(n_heldout, n))))
    dcb = {"master": [], "reg_refine": []}
    cfa = {"master_def": [], "refine_def": [], "master_cfa": [], "refine_cfa": []}

    for hidx in hids:
        kept = [f for i, f in enumerate(frames) if i != hidx]
        kept_m = [m for i, m in enumerate(mos) if i != hidx]
        hw = frames[hidx]

        r_m = SuperResolution(
            ratio=2.0, blur_threshold_ratio=0.0, ecc_scale=0.25,
            reg_refine=0, use_gpu=False, use_opencl=False, use_gpu_scatter=False,
        ).run(kept)
        r_r = SuperResolution(
            ratio=2.0, blur_threshold_ratio=0.0, ecc_scale=0.25,
            reg_refine=2, use_gpu=False, use_opencl=False, use_gpu_scatter=False,
        ).run(kept)

        ref_green = kept[r_m.reference_index].green
        W_held = estimate_homography(ref_green, hw.green, ecc_scale=0.25)
        dst = (hw.rgb.shape[1], hw.rgb.shape[0])

        for tag, res in (("master", r_m), ("reg_refine", r_r)):
            pred_sr = predict_lr_from_hr(res.hr, W_held, 2.0, hw.rgb.shape[:2])
            base_lr, base_cov, _ = heldout_baseline_lr(kept, res)
            base_lr_held = cv2.warpPerspective(
                base_lr, W_held.astype(np.float32), dst,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            base_cov_held = cv2.warpPerspective(
                base_cov, W_held.astype(np.float32), dst,
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )
            mask = heldout_valid_mask(pred_sr, res.hr.shape[:2], W_held, 2.0, hw.rgb.shape[:2])
            mask &= base_cov_held >= float(np.sum(res.weights)) * 0.999
            comp = heldout_compare(pred_sr, base_lr_held, hw.rgb, mask)
            dcb[tag].append(comp["heldout_gain_db"])

        ki = list(r_m.kept_indices)
        km = [kept_m[i] for i in ki]
        ref_mos = kept_m[r_m.reference_index]
        cfa_m = cfa_drizzle(km, r_m.transforms, 2.0, pat, r_m.weights.tolist(), pixfrac=0.5).hr

        ki_r = list(r_r.kept_indices)
        km_r = [kept_m[i] for i in ki_r]
        cfa_r = cfa_drizzle(km_r, r_r.transforms, 2.0, pat, r_r.weights.tolist(), pixfrac=0.5).hr

        hrs = {
            "master_def": level_fit(r_m.hr, ref_mos, pat),
            "refine_def": level_fit(r_r.hr, ref_mos, pat),
            "master_cfa": level_fit(cfa_m, ref_mos, pat),
            "refine_cfa": level_fit(cfa_r, ref_mos, pat),
        }
        m_cfa = evaluate_cfa_heldout(
            hrs, mos[hidx], km, r_m.transforms, r_m.weights.tolist(), W_held, 2.0, pat
        )
        for k in cfa:
            cfa[k].append(m_cfa["per_method"][k]["heldout_cfa_gain_db"])

    out = {
        "name": name,
        "frames": n,
        "heldout_indices": hids,
        "time_master_s": round(t_master, 2),
        "time_refine_s": round(t_refine, 2),
        "reg_refine_seconds": r_r_full.metrics.get("reg_refine_seconds", 0.0),
        "reg_refine_iters": r_r_full.metrics.get("reg_refine_iters", 0),
        "reg_refine_max_delta_lr_px": r_r_full.metrics.get("reg_refine_max_delta_lr_px", 0.0),
        "rss_mib": round(rss_mib, 1),
        "dcb": {k: {"mean": round(float(np.mean(v)), 3), "min": round(float(min(v)), 3)}
                for k, v in dcb.items()},
        "cfa": {k: {"mean": round(float(np.mean(v)), 3), "min": round(float(min(v)), 3)}
                for k, v in cfa.items()},
        **panel_info,
    }
    print(f"=== {name} ({n} frames) ===", flush=True)
    print(
        f"  DCB: master={out['dcb']['master']['mean']:+.3f} ({out['dcb']['master']['min']:+.3f})  "
        f"refine={out['dcb']['reg_refine']['mean']:+.3f} ({out['dcb']['reg_refine']['min']:+.3f})",
        flush=True,
    )
    print(
        f"  CFA: m_def={out['cfa']['master_def']['mean']:+.3f}  r_def={out['cfa']['refine_def']['mean']:+.3f}  "
        f"m_cfa={out['cfa']['master_cfa']['mean']:+.3f}  r_cfa={out['cfa']['refine_cfa']['mean']:+.3f}",
        flush=True,
    )
    print(
        f"  time: master={t_master:.2f}s  refine={t_refine:.2f}s (rr={out['reg_refine_seconds']:.2f}s, "
        f"delta={out['reg_refine_max_delta_lr_px']:.4f} px)  panel={out['panel']} ({out['size_kb']} KB)",
        flush=True,
    )
    return out


def main():
    os.makedirs("/tmp/rawsr/wp7", exist_ok=True)
    results = {}
    for name, bdir, roi_xy, pub in BURSTS:
        results[name] = eval_burst(name, bdir, roi_xy, public=pub, n_heldout=5)
    out_path = "/tmp/rawsr/wp7/real_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
