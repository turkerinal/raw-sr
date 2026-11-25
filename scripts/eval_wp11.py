"""WP11 evaluation script: synthetic 2x/3x, WP10 simulated chart, and real 2x (5 bursts)."""

from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import sys
import time
from typing import Dict, List

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
from raw_sr.core.cfa_drizzle import channel_map, pattern_array
from raw_sr.core.cfa_heldout import evaluate_cfa_heldout
from raw_sr.core.enhance import EnhanceConfig, enhance_image
from raw_sr.core.map_sr import estimate_colour_matrix, map_super_resolution
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

            t0 = time.perf_counter()
            r_def = SuperResolution(
                ratio=float(ratio), blur_threshold_ratio=0.0,
                registration="homography", ecc_scale=0.25, fusion="demosaic_drizzle",
            ).run(frames, reference_index=0)
            row["est_default"] = {**_syn_metrics(r_def.hr, b, ratio), "sec": round(time.perf_counter() - t0, 2)}

            enh = enhance_image(r_def.hr, ratio=float(ratio), config=EnhanceConfig(mode="wp2")).image
            row["est_default_wp2"] = _syn_metrics(enh, b, ratio)

            t0 = time.perf_counter()
            r_map = SuperResolution(
                ratio=float(ratio), blur_threshold_ratio=0.0,
                registration="homography", ecc_scale=0.25, fusion="map",
            ).run(frames, reference_index=0)
            row["est_map"] = {
                **_syn_metrics(r_map.hr, b, ratio),
                "sec": round(time.perf_counter() - t0, 2),
                "iters": r_map.metrics.get("cfa_kr_iterations"),
                "engine": r_map.metrics.get("cfa_kr_engine"),
            }

            # Exact registration
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
            map_ex = map_super_resolution(hr_init_ex, mos, exact_T, float(ratio), BGGR, w_unif)
            row["exact_map"] = _syn_metrics(map_ex.hr, b, ratio)

            key = f"{ratio}x_n{n}"
            results[key] = row
            print(f"=== SYNTH {key} ===", json.dumps(row), flush=True)

    with open(os.path.join(out_dir, "synth_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def run_chart(out_dir: str) -> None:
    """WP10 simulated phone-camera test chart: default vs map (WP11)."""
    from scripts.testchart.make_chart import render_chart_raster
    from scripts.testchart.measure_chart import measure_triplet
    from scripts.testchart.validate_sim import (
        _build_perspective_H,
        _demosaic_bggr as _chart_dem,
        _mosaic_from_rgb,
        analytical_mtf_crossing,
    )

    os.makedirs(out_dir, exist_ok=True)
    ratio = 2
    lr_h = 680
    n_frames = 16
    sigma_lens_lr = 0.45
    rng = np.random.default_rng(42)
    lr_w = int(round(lr_h * (297.0 / 210.0)))
    lr_h -= lr_h % 2
    lr_w -= lr_w % 2
    hr_h, hr_w = lr_h * ratio, lr_w * ratio

    chart_u8 = render_chart_raster(paper="A4", dpi=200)
    chart_f = chart_u8.astype(np.float32) / 255.0
    chart_rgb = np.stack([chart_f, chart_f, chart_f], axis=2)
    ch_h, ch_w = chart_rgb.shape[:2]

    H_base = _build_perspective_H(ch_w, ch_h, hr_w, hr_h, margin=0.07, tilt_px=8.0)
    scene_hr = cv2.warpPerspective(
        chart_rgb, H_base, (hr_w, hr_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    blurred_hr = cv2.GaussianBlur(scene_hr, (0, 0), sigma_lens_lr * ratio, borderType=cv2.BORDER_REFLECT)
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
        m = (rng.poisson(np.clip(m, 0.0, 1.0) * full_well) + rng.normal(0.0, read_noise, m.shape)) / full_well
        m = np.clip(m, 0.0, 1.0).astype(np.float32)
        dem = _chart_dem(m)
        raw_frames.append(RawFrame(f"sim_{k:02d}.dng", dem, dem[..., 1].copy(), mosaic=m, pattern=BGGR.copy()))

    single_img = cv2.resize(raw_frames[0].rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
    gt_blurred = cv2.GaussianBlur(scene_hr, (0, 0), sigma_lens_lr, borderType=cv2.BORDER_REFLECT)
    gt_stack = []
    for _ in range(3):
        m_gt = _mosaic_from_rgb(gt_blurred)
        m_gt = (rng.poisson(np.clip(m_gt, 0.0, 1.0) * full_well) + rng.normal(0.0, read_noise, m_gt.shape)) / full_well
        gt_stack.append(_chart_dem(np.clip(m_gt, 0.0, 1.0).astype(np.float32)))
    gt_img = np.median(np.stack(gt_stack, axis=0), axis=0).astype(np.float32)

    out = {"analytical_f50": round(analytical_mtf_crossing(sigma_lens_lr, 0.50), 4),
           "analytical_f10": round(analytical_mtf_crossing(sigma_lens_lr, 0.10), 4)}
    for mode in ("demosaic_drizzle", "map"):
        sr = SuperResolution(ratio=float(ratio), fusion=mode, registration_workers=1, use_opencl=False)
        res = sr.run(raw_frames)
        rep = measure_triplet(single_img=single_img, sr_img=res.hr.astype(np.float32), ratio=float(ratio), gt_img=gt_img)
        out[mode] = {
            "star_f50": rep["chart_only"]["sr"]["star_sin"]["f50_cyc_lr_px"],
            "star_f10": rep["chart_only"]["sr"]["star_sin"]["f10_cyc_lr_px"],
            "edge_mtf50": rep["chart_only"]["sr"]["edge_high"]["mtf50_cyc_lr_px"],
            "zoom_gt_psnr": rep["zoom_gt"]["sr_vs_gt"]["overall"]["psnr_db"],
            "zoom_gt_ssim": rep["zoom_gt"]["sr_vs_gt"]["overall"]["ssim"],
            "single_gt_psnr": rep["zoom_gt"]["single_vs_gt"]["overall"]["psnr_db"],
        }
    print("=== CHART SIM ===", json.dumps(out, indent=2), flush=True)
    with open(os.path.join(out_dir, "chart_results.json"), "w") as f:
        json.dump(out, f, indent=2)


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
    single_hr: np.ndarray,
    def_hr: np.ndarray,
    enh_hr: np.ndarray,
    map_hr: np.ndarray,
    out_path: str,
) -> None:
    """Pixel-aligned 2x2 panel (single | default | default+wp2 | map new), < 1.2 MB."""
    H, W = def_hr.shape[:2]
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

    p99 = float(np.percentile(def_hr[cy:cy + cs, cx:cx + cs], 99.0))
    exp = 0.85 / max(p99, 1e-4)

    tiles = []
    for label, img in (
        ("Single (bicubic)", single_hr),
        ("Default (demosaic_drizzle)", def_hr),
        ("Default + WP2 enhance", enh_hr),
        ("WP11 MAP (colour-fixed)", map_hr),
    ):
        crop = img[cy:cy + cs, cx:cx + cs]
        u8 = to_uint8_view(crop, gamma=2.2, exposure=exp)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        z = cv2.resize(bgr, (cs * 2, cs * 2), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(z, (0, 0), (440, 34), (0, 0, 0), -1)
        cv2.putText(z, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(z)

    panel = np.vstack([np.hstack([tiles[0], tiles[1]]), np.hstack([tiles[2], tiles[3]])])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, panel, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if os.path.getsize(out_path) > 1_150_000:
        panel = cv2.resize(panel, (1200, 1200), interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_path, panel, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def run_real(burst_names: List[str], out_dir: str, docs_panel_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(docs_panel_dir, exist_ok=True)
    summary_all = {}

    for name in burst_names:
        bdir, (rx, ry), n_ho = BURSTS[name]
        files = sorted(glob.glob(os.path.join(bdir, "*.dng")) + glob.glob(os.path.join(bdir, "*.DNG")))
        roi = (rx, ry, 1024, 1024)
        frames = [load_raw_linear(f, roi=roi) for f in files]
        mos = [f.mosaic for f in frames]
        pat = pattern_array(frames[0].pattern)
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
        _make_panel(single_hr, r_def_full.hr, enh_full, r_map_full.hr, panel_path)

        # Held-out sweep (DCB + CFA raw + CFA C-mapped)
        hids = sorted(set(int(round(v)) for v in np.linspace(0, n - 1, min(n_ho, n))))
        dcb: Dict[str, List[float]] = {k: [] for k in ("default", "default_wp2", "map", "map_wp6", "map_rr1")}
        cfa_raw: Dict[str, List[float]] = {k: [] for k in ("default", "default_wp2", "map")}
        cfa_cam: Dict[str, List[float]] = {k: [] for k in ("default", "default_wp2", "map", "map_wp6", "map_rr1")}

        for hidx in hids:
            kept = [f for i, f in enumerate(frames) if i != hidx]
            kept_m = [m for i, m in enumerate(mos) if i != hidx]
            rd = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(kept)
            rd_wp2 = enhance_image(rd.hr, ratio=2.0, config=EnhanceConfig(mode="wp2")).image
            rm = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, fusion="map").run(kept)
            rm_wp6 = enhance_image(rm.hr, ratio=2.0, config=EnhanceConfig(mode="adaptive")).image
            rm_rr1 = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, fusion="map", reg_refine=1).run(kept)

            ki = list(rd.kept_indices)
            km = [kept_m[i] for i in ki]
            ref_pos = rd.reference_index
            C_ref, _ = estimate_colour_matrix(kept[ref_pos].rgb, kept[ref_pos].mosaic, pat)
            C32T = C_ref.T.astype(np.float32)
            W_held = estimate_homography(kept[ref_pos].green, frames[hidx].green, ecc_scale=0.25)

            hrs_map = {
                "default": rd.hr,
                "default_wp2": rd_wp2,
                "map": rm.hr,
                "map_wp6": rm_wp6,
                "map_rr1": rm_rr1.hr,
            }
            for k, hr_img in hrs_map.items():
                dcb[k].append(_dcb_heldout_gain_db(hr_img, frames[hidx], [kept[i] for i in ki], rd.transforms, rd.weights.tolist(), W_held, 2.0))

            # Raw CFA held-out (no 3x3 C)
            res_raw = evaluate_cfa_heldout(
                {"default": rd.hr, "default_wp2": rd_wp2, "map": rm.hr},
                mos[hidx], km, rd.transforms, rd.weights.tolist(), W_held, 2.0, pat,
            )
            for k in ("default", "default_wp2", "map"):
                cfa_raw[k].append(float(res_raw["per_method"][k]["heldout_cfa_gain_db"]))

            # C-mapped CFA held-out (camera space)
            res_cam = evaluate_cfa_heldout(
                {k: np.matmul(hr_img, C32T) for k, hr_img in hrs_map.items()},
                mos[hidx], km, rd.transforms, rd.weights.tolist(), W_held, 2.0, pat,
            )
            for k in hrs_map:
                cfa_cam[k].append(float(res_cam["per_method"][k]["heldout_cfa_gain_db"]))

        summary_all[name] = {
            "frames": n,
            "full_def_sec": round(t_def_full, 2),
            "full_map_sec": round(t_map_full, 2),
            "map_engine": r_map_full.metrics.get("cfa_kr_engine"),
            "map_iters": r_map_full.metrics.get("cfa_kr_iterations"),
            "gr_gb_ratio": r_map_full.metrics.get("cfa_kr_gr_gb_ratio"),
            "colour_cond": r_map_full.metrics.get("cfa_kr_colour_cond"),
            "peak_rss_mb": round(rss_mb, 1),
            "panel_kb": round(os.path.getsize(panel_path) / 1024.0, 1),
            "dcb_mean": {k: round(float(np.mean(v)), 3) for k, v in dcb.items()},
            "dcb_min": {k: round(float(np.min(v)), 3) for k, v in dcb.items()},
            "cfa_raw_mean": {k: round(float(np.mean(v)), 3) for k, v in cfa_raw.items()},
            "cfa_cam_mean": {k: round(float(np.mean(v)), 3) for k, v in cfa_cam.items()},
            "cfa_cam_min": {k: round(float(np.min(v)), 3) for k, v in cfa_cam.items()},
        }
        print(f"=== REAL {name} ===", json.dumps(summary_all[name], indent=1), flush=True)

    with open(os.path.join(out_dir, "real_results.json"), "w") as f:
        json.dump(summary_all, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synth", "chart", "real"])
    ap.add_argument("--bursts", nargs="+", default=list(BURSTS.keys()))
    ap.add_argument("--out-dir", default="/tmp/rawsr/wp11")
    ap.add_argument("--panel-dir", default=os.path.join(ROOT, "docs", "wp11"))
    args = ap.parse_args()
    if args.mode == "synth":
        run_synth(args.out_dir)
    elif args.mode == "chart":
        run_chart(args.out_dir)
    else:
        run_real(args.bursts, args.out_dir, args.panel_dir)
