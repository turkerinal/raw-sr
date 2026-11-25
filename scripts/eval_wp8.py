"""WP8 evaluation script: synthetic 2x/3x + real 2x (DCB & CFA held-out + panels)."""

from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from audit_c11_part2 import (
    BGGR,
    _demosaic_bggr,
    compute_slanted_edge_mtf50,
    immerkaer_sigma,
    psnr,
    synthesize_target_burst,
)
from raw_sr.core.cfa_drizzle import (
    apply_level_gains,
    calibrate_cfa_levels,
    cfa_drizzle,
    channel_map,
    load_cfa_frame,
    pattern_array,
)
from raw_sr.core.cfa_heldout import evaluate_cfa_heldout
from raw_sr.core.enhance import EnhanceConfig, enhance_image
from raw_sr.core.map_sr import map_super_resolution
from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    denoised_lr_stack,
    drizzle,
    estimate_homography,
    load_raw_linear,
    lr_sampling_valid_mask,
    predict_lr_from_hr,
    to_uint8_view,
)


def _overshoot_pct(img: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    """Measure 99th-percentile overshoot relative to GT dynamic range."""
    y = img[..., 1][mask].astype(np.float64)
    r = ref[..., 1][mask].astype(np.float64)
    lo, hi = float(np.percentile(r, 1)), float(np.percentile(r, 99))
    span = max(hi - lo, 1e-4)
    over = np.maximum(0.0, y - hi) + np.maximum(0.0, lo - y)
    return round(float(np.percentile(over, 99) / span * 100.0), 2)


def _syn_metrics(img: np.ndarray, b: dict, ratio: int) -> dict:
    return {
        "psnr_db": round(float(psnr(img, b["target"], b["mask"])), 2),
        "mtf50": round(float(compute_slanted_edge_mtf50(img, float(ratio))), 4),
        "overshoot_pct": _overshoot_pct(img, b["target"], b["mask"]),
        "flat_sigma": round(float(immerkaer_sigma(img[30:90, 30:90, 1])), 5),
    }


def run_synth(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for ratio in (2, 3):
        for n in (9, 20):
            b = synthesize_target_burst(ratio=ratio, lr=256, n=n, seed=7)
            mos = b["mosaics"]
            dems = [_demosaic_bggr(m) for m in mos]
            frames = [
                RawFrame(str(k), d, d[..., 1].copy(), mosaic=m, pattern=BGGR)
                for k, (d, m) in enumerate(zip(dems, mos))
            ]
            row = {}

            # 1. Estimated registration
            t0 = time.perf_counter()
            sr_def = SuperResolution(
                ratio=float(ratio), blur_threshold_ratio=0.0,
                registration="homography", ecc_scale=0.25, fusion="demosaic_drizzle",
            )
            r_def = sr_def.run(frames, reference_index=0)
            t_def = time.perf_counter() - t0
            row["est_default"] = {**_syn_metrics(r_def.hr, b, ratio), "sec": round(t_def, 2)}

            # default + wp2 enhance
            t0 = time.perf_counter()
            enh = enhance_image(r_def.hr, ratio=float(ratio), config=EnhanceConfig(mode="wp2")).image
            t_enh = time.perf_counter() - t0
            row["est_default_wp2"] = {**_syn_metrics(enh, b, ratio), "sec": round(t_def + t_enh, 2)}

            # cfa_drizzle (using estimated transforms)
            t0 = time.perf_counter()
            cfa_est = cfa_drizzle(
                mos, r_def.transforms, float(ratio), BGGR, r_def.weights.tolist(), pixfrac=0.5,
            ).hr
            t_cfa = time.perf_counter() - t0
            row["est_cfa_drizzle"] = {**_syn_metrics(cfa_est, b, ratio), "sec": round(t_cfa, 2)}

            # map (using SuperResolution(fusion="map"))
            t0 = time.perf_counter()
            sr_map = SuperResolution(
                ratio=float(ratio), blur_threshold_ratio=0.0,
                registration="homography", ecc_scale=0.25, fusion="map",
            )
            r_map = sr_map.run(frames, reference_index=0)
            t_map = time.perf_counter() - t0
            row["est_map"] = {
                **_syn_metrics(r_map.hr, b, ratio),
                "sec": round(t_map, 2),
                "iters": r_map.metrics.get("cfa_kr_iterations"),
                "engine": r_map.metrics.get("cfa_kr_engine"),
            }

            # 2. Exact registration (match sign of r_def.transforms[:2, 2])
            tru = np.asarray(b["offsets"], dtype=np.float64)
            est_t = np.array([[T[0, 2], T[1, 2]] for T in r_def.transforms], dtype=np.float64)
            sign = -1.0 if np.linalg.norm(est_t[1:] + tru[1:], axis=1).mean() < np.linalg.norm(est_t[1:] - tru[1:], axis=1).mean() else 1.0
            exact_T = [
                np.array([[1.0, 0.0, sign * dx], [0.0, 1.0, sign * dy], [0.0, 0.0, 1.0]], dtype=np.float64)
                for dx, dy in tru
            ]
            w_unif = [1.0] * n
            hr_init_ex, _ = drizzle(frames, exact_T, float(ratio), w_unif)
            row["exact_default"] = _syn_metrics(hr_init_ex, b, ratio)

            enh_ex = enhance_image(hr_init_ex, ratio=float(ratio), config=EnhanceConfig(mode="wp2")).image
            row["exact_default_wp2"] = _syn_metrics(enh_ex, b, ratio)

            cfa_ex = cfa_drizzle(mos, exact_T, float(ratio), BGGR, w_unif, pixfrac=0.5).hr
            row["exact_cfa_drizzle"] = _syn_metrics(cfa_ex, b, ratio)

            t0 = time.perf_counter()
            map_ex = map_super_resolution(hr_init_ex, mos, exact_T, float(ratio), BGGR, w_unif)
            t_mex = time.perf_counter() - t0
            row["exact_map"] = {
                **_syn_metrics(map_ex.hr, b, ratio),
                "sec": round(t_mex, 2),
                "iters": map_ex.metrics.get("iterations"),
                "engine": map_ex.metrics.get("engine"),
            }

            key = f"{ratio}x_n{n}"
            results[key] = row
            print(f"=== {key} ===", json.dumps(row, indent=1), flush=True)

    with open(os.path.join(out_dir, "synth_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def _level_fit(hr: np.ndarray, ref_mosaic: np.ndarray, pat: np.ndarray) -> np.ndarray:
    h, w = ref_mosaic.shape
    lr = cv2.resize(hr, (w, h), interpolation=cv2.INTER_AREA)
    ch = channel_map(pat, (h, w))
    out = hr.copy()
    m = np.zeros((h, w), bool)
    m[32:-32, 32:-32] = True
    for c in range(3):
        sel = (ch == c) & m
        x = lr[..., c][sel].astype(np.float64)
        y = ref_mosaic[sel].astype(np.float64)
        (g, o), *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
        out[..., c] = hr[..., c] * g + o
    return out.astype(np.float32)


def _dcb_heldout_gain_db(
    hr: np.ndarray,
    held_frame: RawFrame,
    kept_frames: List[RawFrame],
    transforms: list,
    weights: List[float],
    W_held: np.ndarray,
    ratio: float,
) -> float:
    lr_shape = held_frame.rgb.shape[:2]
    pred = predict_lr_from_hr(hr, W_held, ratio, lr_shape)
    stack = denoised_lr_stack(kept_frames, transforms, weights)
    base = cv2.warpPerspective(
        stack, W_held.astype(np.float32), (lr_shape[1], lr_shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    mask = lr_sampling_valid_mask(hr.shape[:2], W_held, ratio, lr_shape)
    m24 = np.zeros(lr_shape, dtype=bool)
    m24[24:-24, 24:-24] = True
    mask = mask & m24
    mse_sr = float(np.mean((pred[mask] - held_frame.rgb[mask]) ** 2))
    mse_base = float(np.mean((base[mask] - held_frame.rgb[mask]) ** 2))
    return float(10.0 * np.log10(max(mse_base, 1e-12) / max(mse_sr, 1e-12)))


BURSTS = {
    "own9": (os.path.join(ROOT, "test_data"), (1650, 1240), 5),
    "hdrplus_0919": ("data/hdrplus/0919_20150910_131720_137", (1504, 1000), 5),
    "hdrplus_0382": ("data/hdrplus/0382_20150912_130914_161", (1504, 1000), 5),
    "samsung14": ("data/samsung14", (2400, 520), 5),
    "pixel4a20": ("data/inria_friant20", (2080, 800), 5),
}


def _make_panel(
    name: str,
    single_hr: np.ndarray,
    def_hr: np.ndarray,
    enh_hr: np.ndarray,
    map_hr: np.ndarray,
    out_path: str,
) -> None:
    """Create pixel-aligned 2x2 panel (single | default | default+wp2 | map) < 1.2 MB."""
    H, W = def_hr.shape[:2]
    # Auto-pick highest-gradient 360x360 crop away from borders
    g = def_hr[..., 1]
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    cs = 360
    best_s, best_yx = -1.0, (H // 2 - cs // 2, W // 2 - cs // 2)
    for y in range(160, H - cs - 160, 80):
        for x in range(160, W - cs - 160, 80):
            s = float(np.mean(grad[y:y + cs, x:x + cs]))
            if s > best_s:
                best_s = s
                best_yx = (y, x)
    cy, cx = best_yx

    # Shared exposure from def_hr crop
    p99 = float(np.percentile(def_hr[cy:cy + cs, cx:cx + cs], 99.0))
    exp = 0.85 / max(p99, 1e-4)

    tiles = []
    for label, img in (
        ("Single (bicubic)", single_hr),
        ("Default (demosaic_drizzle)", def_hr),
        ("Default + WP2 enhance", enh_hr),
        ("WP8 MAP (joint CFA)", map_hr),
    ):
        crop = img[cy:cy + cs, cx:cx + cs]
        u8 = to_uint8_view(crop, gamma=2.2, exposure=exp)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        # 2x nearest zoom for crisp pixel inspection -> 720x720
        z = cv2.resize(bgr, (cs * 2, cs * 2), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(z, (0, 0), (420, 34), (0, 0, 0), -1)
        cv2.putText(z, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(z)

    top = np.hstack([tiles[0], tiles[1]])
    bot = np.hstack([tiles[2], tiles[3]])
    panel = np.vstack([top, bot])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, panel, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    # If > 1.15 MB, downscale slightly to stay strictly < 1.2 MB
    if os.path.getsize(out_path) > 1_150_000:
        panel = cv2.resize(panel, (1200, 1200), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_path, panel, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def run_real(burst_names: List[str], out_dir: str, docs_panel_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(docs_panel_dir, exist_ok=True)
    summary_all = {}

    for name in burst_names:
        bdir, (rx, ry), n_ho = BURSTS[name]
        files = sorted(glob.glob(os.path.join(bdir, "*.dng")))
        roi = (rx, ry, 1024, 1024)
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
        r_def_full = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(frames)
        t_def_full = time.perf_counter() - t0
        enh_full = enhance_image(r_def_full.hr, ratio=2.0, config=EnhanceConfig(mode="wp2")).image

        t0 = time.perf_counter()
        r_map_full = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, fusion="map").run(frames)
        t_map_full = time.perf_counter() - t0
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        ref_rgb = frames[r_def_full.reference_index].rgb
        single_hr = cv2.resize(ref_rgb, (r_def_full.hr.shape[1], r_def_full.hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        panel_path = (
            os.path.join(out_dir, f"{name}_2x.png")
            if name == "pixel4a20"
            else os.path.join(docs_panel_dir, f"{name}_2x.png")
        )
        _make_panel(name, single_hr, r_def_full.hr, enh_full, r_map_full.hr, panel_path)

        # Held-out sweep (DCB + CFA)
        hids = sorted(set(int(round(v)) for v in np.linspace(0, n - 1, min(n_ho, n))))
        dcb_def, dcb_map = [], []
        cfa_def, cfa_map = [], []
        for hidx in hids:
            kept = [f for i, f in enumerate(frames) if i != hidx]
            kept_m = [m for i, m in enumerate(mos) if i != hidx]
            rd = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(kept)
            rm = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, fusion="map").run(kept)
            ki = list(rd.kept_indices)
            km = [kept_m[i] for i in ki]
            ref_mos = kept_m[rd.reference_index]
            W_held = estimate_homography(kept[rd.reference_index].green, frames[hidx].green, ecc_scale=0.25)

            dcb_def.append(_dcb_heldout_gain_db(rd.hr, frames[hidx], [kept[i] for i in ki], rd.transforms, rd.weights.tolist(), W_held, 2.0))
            dcb_map.append(_dcb_heldout_gain_db(rm.hr, frames[hidx], [kept[i] for i in ki], rd.transforms, rd.weights.tolist(), W_held, 2.0))

            hrs = {
                "default": _level_fit(rd.hr, ref_mos, pat),
                "map": _level_fit(rm.hr, ref_mos, pat),
            }
            cfa_res = evaluate_cfa_heldout(hrs, mos[hidx], km, rd.transforms, rd.weights.tolist(), W_held, 2.0, pat)
            cfa_def.append(float(cfa_res["per_method"]["default"]["heldout_cfa_gain_db"]))
            cfa_map.append(float(cfa_res["per_method"]["map"]["heldout_cfa_gain_db"]))

        summary_all[name] = {
            "frames": n,
            "full_def_sec": round(t_def_full, 2),
            "full_map_sec": round(t_map_full, 2),
            "map_engine": r_map_full.metrics.get("cfa_kr_engine"),
            "map_iters": r_map_full.metrics.get("cfa_kr_iterations"),
            "peak_rss_mb": round(rss_mb, 1),
            "panel_kb": round(os.path.getsize(panel_path) / 1024.0, 1),
            "dcb_default_mean": round(float(np.mean(dcb_def)), 3),
            "dcb_default_min": round(float(np.min(dcb_def)), 3),
            "dcb_map_mean": round(float(np.mean(dcb_map)), 3),
            "dcb_map_min": round(float(np.min(dcb_map)), 3),
            "cfa_default_mean": round(float(np.mean(cfa_def)), 3),
            "cfa_default_min": round(float(np.min(cfa_def)), 3),
            "cfa_map_mean": round(float(np.mean(cfa_map)), 3),
            "cfa_map_min": round(float(np.min(cfa_map)), 3),
        }
        print(f"=== REAL {name} ===", json.dumps(summary_all[name], indent=1), flush=True)

    with open(os.path.join(out_dir, "real_results.json"), "w") as f:
        json.dump(summary_all, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synth", "real"])
    ap.add_argument("--bursts", nargs="+", default=list(BURSTS.keys()))
    ap.add_argument("--out-dir", default="/tmp/rawsr/wp8")
    ap.add_argument("--panel-dir", default=os.path.join(ROOT, "docs", "wp8"))
    args = ap.parse_args()
    if args.mode == "synth":
        run_synth(args.out_dir)
    else:
        run_real(args.bursts, args.out_dir, args.panel_dir)
