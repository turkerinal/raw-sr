"""WP12 final fair comparison: synthetic 2x/3x (+ metric bias check), chart 2x/3x, and real 2x (5 bursts)."""

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
from raw_sr.core.cfa_drizzle import pattern_array
from raw_sr.core.cfa_heldout import evaluate_cfa_heldout
from raw_sr.core.enhance import EnhanceConfig, enhance_image
from raw_sr.core.map_sr import estimate_colour_matrix
from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    denoised_lr_stack,
    estimate_homography,
    load_raw_linear,
    lr_sampling_valid_mask,
    predict_lr_from_hr,
    to_uint8_view,
)


def _syn_metrics(img: np.ndarray, b: dict, ratio: int) -> dict:
    return {
        "psnr_db": round(float(psnr(img, b["target"], b["mask"])), 2),
        "mtf50": round(float(compute_slanted_edge_mtf50(img, float(ratio))), 4),
        "flat_sigma": round(float(immerkaer_sigma(img[30:90, 30:90, 1])), 5),
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


def _run_all_methods(frames: List[RawFrame], ratio: float, ref_idx: int = 0, ecc_scale: float = 0.25):
    """Run all 7 WP12 candidate methods and return (hr_dict, meta_dict, rd_obj)."""
    hrs: Dict[str, np.ndarray] = {}
    meta: Dict[str, dict] = {}

    t0 = time.perf_counter()
    rd = SuperResolution(
        ratio=ratio, blur_threshold_ratio=0.0, registration="homography",
        ecc_scale=ecc_scale, fusion="demosaic_drizzle",
    ).run(frames, reference_index=ref_idx)
    t_def = time.perf_counter() - t0
    hrs["default"] = rd.hr
    meta["default"] = {"sec": round(t_def, 2)}

    t0 = time.perf_counter()
    hrs["default_wp2"] = enhance_image(rd.hr, ratio=ratio, config=EnhanceConfig(mode="wp2")).image
    meta["default_wp2"] = {"sec": round(t_def + (time.perf_counter() - t0), 2)}

    t0 = time.perf_counter()
    hrs["default_wp6"] = enhance_image(rd.hr, ratio=ratio, config=EnhanceConfig(mode="adaptive")).image
    meta["default_wp6"] = {"sec": round(t_def + (time.perf_counter() - t0), 2)}

    t0 = time.perf_counter()
    rcd = SuperResolution(
        ratio=ratio, blur_threshold_ratio=0.0, registration="homography",
        ecc_scale=ecc_scale, fusion="cfa_drizzle",
    ).run(frames, reference_index=ref_idx)
    hrs["cfa_drizzle"] = rcd.hr
    meta["cfa_drizzle"] = {"sec": round(time.perf_counter() - t0, 2)}

    t0 = time.perf_counter()
    rck = SuperResolution(
        ratio=ratio, blur_threshold_ratio=0.0, registration="homography",
        ecc_scale=ecc_scale, fusion="cfa_kr", cfa_adaptive=True,
    ).run(frames, reference_index=ref_idx)
    hrs["cfa_kr_adapt"] = rck.hr
    meta["cfa_kr_adapt"] = {"sec": round(time.perf_counter() - t0, 2)}

    t0 = time.perf_counter()
    rm = SuperResolution(
        ratio=ratio, blur_threshold_ratio=0.0, registration="homography",
        ecc_scale=ecc_scale, fusion="map",
    ).run(frames, reference_index=ref_idx)
    hrs["map"] = rm.hr
    meta["map"] = {
        "sec": round(time.perf_counter() - t0, 2),
        "engine": rm.metrics.get("cfa_kr_engine"),
        "iters": rm.metrics.get("cfa_kr_iterations"),
    }

    t0 = time.perf_counter()
    rmr = SuperResolution(
        ratio=ratio, blur_threshold_ratio=0.0, registration="homography",
        ecc_scale=ecc_scale, fusion="map", reg_refine=1,
    ).run(frames, reference_index=ref_idx)
    hrs["map_rr1"] = rmr.hr
    meta["map_rr1"] = {
        "sec": round(time.perf_counter() - t0, 2),
        "engine": rmr.metrics.get("cfa_kr_engine"),
    }

    return hrs, meta, rd


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
            hrs, meta, _ = _run_all_methods(frames, float(ratio), ref_idx=0, ecc_scale=0.25)
            row = {k: {**_syn_metrics(img, b, ratio), **meta[k]} for k, img in hrs.items()}
            key = f"{ratio}x_n{n}"
            results[key] = row
            print(f"=== SYNTH {key} ===", json.dumps({k: v["psnr_db"] for k, v in row.items()}), flush=True)

    # Synthetic metric-bias check (2x, n=10 -> fuse 9, hold out frame 9)
    b_bias = synthesize_target_burst(ratio=2, lr=256, n=10, seed=11)
    mos_b = b_bias["mosaics"]
    dems_b = [_demosaic_bggr(m) for m in mos_b]
    frames_b = [
        RawFrame(str(k), d, d[..., 1].copy(), mosaic=m, pattern=BGGR)
        for k, (d, m) in enumerate(zip(dems_b, mos_b))
    ]
    kept_b = frames_b[:9]
    held_b = frames_b[9]
    hrs_b, _, rd_b = _run_all_methods(kept_b, 2.0, ref_idx=0, ecc_scale=0.25)
    ki = list(rd_b.kept_indices)
    W_held = estimate_homography(kept_b[0].green, held_b.green, ecc_scale=0.25)
    C_syn, _ = estimate_colour_matrix(kept_b[0].rgb, kept_b[0].mosaic, BGGR)

    cfa_res = evaluate_cfa_heldout(
        hrs_b, held_b.mosaic, [kept_b[i].mosaic for i in ki],
        rd_b.transforms, rd_b.weights.tolist(), W_held, 2.0, BGGR,
        margin_lr=16, colour_matrix=C_syn,
    )
    bias_check = {}
    for k, img in hrs_b.items():
        bias_check[k] = {
            "gt_psnr_db": round(float(psnr(img, b_bias["target"], b_bias["mask"])), 2),
            "dcb_heldout_db": round(_dcb_heldout_gain_db(
                img, held_b, [kept_b[i] for i in ki], rd_b.transforms, rd_b.weights.tolist(), W_held, 2.0, margin_lr=16
            ), 3),
            "cfa_fair_heldout_db": round(float(cfa_res["per_method"][k]["heldout_cfa_gain_db"]), 3),
        }
    results["metric_bias_check_2x_n9"] = bias_check
    print("=== METRIC BIAS CHECK ===", json.dumps(bias_check, indent=1), flush=True)

    with open(os.path.join(out_dir, "synth_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def run_chart(out_dir: str) -> None:
    """WP10 simulated phone-camera test chart at 2x and 3x for all 7 methods."""
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

    all_out = {}
    for ratio in (2, 3):
        lr_h = 680 if ratio == 2 else 460
        n_frames = 16
        sigma_lens_lr = 0.45
        rng = np.random.default_rng(42 + ratio)
        lr_w = int(round(lr_h * (297.0 / 210.0)))
        lr_h -= lr_h % 2
        lr_w -= lr_w % 2
        hr_h, hr_w = lr_h * ratio, lr_w * ratio

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

        hrs, meta, _ = _run_all_methods(raw_frames, float(ratio), ref_idx=0, ecc_scale=0.25)
        out_r = {
            "analytical_f50": round(analytical_mtf_crossing(sigma_lens_lr, 0.50), 4),
            "analytical_f10": round(analytical_mtf_crossing(sigma_lens_lr, 0.10), 4),
        }
        for name, hr_img in hrs.items():
            rep = measure_triplet(single_img=single_img, sr_img=hr_img.astype(np.float32), ratio=float(ratio), gt_img=gt_img)
            out_r[name] = {
                "star_f50": rep["chart_only"]["sr"]["star_sin"]["f50_cyc_lr_px"],
                "star_f10": rep["chart_only"]["sr"]["star_sin"]["f10_cyc_lr_px"],
                "edge_mtf50": rep["chart_only"]["sr"]["edge_high"]["mtf50_cyc_lr_px"],
                "zoom_gt_psnr": rep["zoom_gt"]["sr_vs_gt"]["overall"]["psnr_db"],
                "zoom_gt_ssim": rep["zoom_gt"]["sr_vs_gt"]["overall"]["ssim"],
                "single_gt_psnr": rep["zoom_gt"]["single_vs_gt"]["overall"]["psnr_db"],
                "sec": meta[name]["sec"],
            }
        all_out[f"{ratio}x"] = out_r
        print(f"=== CHART {ratio}x ===", json.dumps({k: v["zoom_gt_psnr"] for k, v in out_r.items() if isinstance(v, dict)}), flush=True)

    with open(os.path.join(out_dir, "chart_results.json"), "w") as f:
        json.dump(all_out, f, indent=2)


BURSTS = {
    "own9": (os.path.join(ROOT, "test_data"), (1650, 1240), 3),
    "hdrplus_0919": ("data/hdrplus/0919_20150910_131720_137", (1504, 1000), 3),
    "hdrplus_0382": ("data/hdrplus/0382_20150912_130914_161", (1504, 1000), 3),
    "samsung14": ("data/samsung14", (2400, 520), 3),
    "pixel4a20": ("data/inria_friant20", (2080, 800), 3),
}


def _make_panel(
    items: List[tuple],
    ref_img: np.ndarray,
    out_path: str,
) -> None:
    """Pixel-aligned 2x2 panel (4 items), 2x nearest zoom on finest texture, < 1.2 MB."""
    H, W = ref_img.shape[:2]
    g = ref_img[..., 1]
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

    p99 = float(np.percentile(ref_img[cy:cy + cs, cx:cx + cs], 99.0))
    exp = 0.85 / max(p99, 1e-4)

    tiles = []
    for label, img in items:
        crop = img[cy:cy + cs, cx:cx + cs]
        u8 = to_uint8_view(crop, gamma=2.2, exposure=exp)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        z = cv2.resize(bgr, (cs * 2, cs * 2), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(z, (0, 0), (460, 34), (0, 0, 0), -1)
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
    methods = ("default", "default_wp2", "default_wp6", "cfa_drizzle", "cfa_kr_adapt", "map", "map_rr1")

    for name in burst_names:
        bdir, (rx, ry), n_ho = BURSTS[name]
        files = sorted(glob.glob(os.path.join(bdir, "*.dng")) + glob.glob(os.path.join(bdir, "*.DNG")))
        roi = (rx, ry, 1024, 1024)
        frames = [load_raw_linear(f, roi=roi) for f in files]
        mos = [f.mosaic for f in frames]
        pat = pattern_array(frames[0].pattern)
        n = len(frames)

        # Full-burst run for timing, RSS, and visual panel
        hrs_full, meta_full, rd_full = _run_all_methods(frames, 2.0, ref_idx=0, ecc_scale=0.25)
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

        ref_rgb = frames[rd_full.reference_index].rgb
        single_hr = cv2.resize(ref_rgb, (rd_full.hr.shape[1], rd_full.hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        panel_path = (
            os.path.join(out_dir, f"{name}_2x.png")
            if name == "pixel4a20"
            else os.path.join(docs_panel_dir, f"{name}_2x.png")
        )
        _make_panel(
            [
                ("Single (bicubic)", single_hr),
                ("Default (demosaic_drizzle)", hrs_full["default"]),
                ("CFA-KR (adaptive, colour-fixed)", hrs_full["cfa_kr_adapt"]),
                ("MAP (WP11, colour-fixed)", hrs_full["map"]),
            ],
            hrs_full["default"],
            panel_path,
        )

        # Held-out sweep: DCB + old CFA raw + new colour-fair CFA
        hids = sorted(set(int(round(v)) for v in np.linspace(0, n - 1, min(n_ho, n))))
        dcb: Dict[str, List[float]] = {k: [] for k in methods}
        cfa_raw: Dict[str, List[float]] = {k: [] for k in methods}
        cfa_fair: Dict[str, List[float]] = {k: [] for k in methods}

        for hidx in hids:
            kept = [f for i, f in enumerate(frames) if i != hidx]
            kept_m = [m for i, m in enumerate(mos) if i != hidx]
            hrs_ho, _, rd = _run_all_methods(kept, 2.0, ref_idx=0, ecc_scale=0.25)
            ki = list(rd.kept_indices)
            km = [kept_m[i] for i in ki]
            ref_pos = rd.reference_index
            C_ref, _ = estimate_colour_matrix(kept[ref_pos].rgb, kept[ref_pos].mosaic, pat)
            W_held = estimate_homography(kept[ref_pos].green, frames[hidx].green, ecc_scale=0.25)

            for k in methods:
                dcb[k].append(_dcb_heldout_gain_db(
                    hrs_ho[k], frames[hidx], [kept[i] for i in ki], rd.transforms, rd.weights.tolist(), W_held, 2.0
                ))

            res_raw = evaluate_cfa_heldout(
                hrs_ho, mos[hidx], km, rd.transforms, rd.weights.tolist(), W_held, 2.0, pat, colour_matrix=None,
            )
            res_fair = evaluate_cfa_heldout(
                hrs_ho, mos[hidx], km, rd.transforms, rd.weights.tolist(), W_held, 2.0, pat, colour_matrix=C_ref,
            )
            for k in methods:
                cfa_raw[k].append(float(res_raw["per_method"][k]["heldout_cfa_gain_db"]))
                cfa_fair[k].append(float(res_fair["per_method"][k]["heldout_cfa_gain_db"]))

        summary_all[name] = {
            "frames": n,
            "full_sec": {k: meta_full[k]["sec"] for k in methods},
            "map_engine": meta_full["map"].get("engine"),
            "peak_rss_mb": round(rss_mb, 1),
            "panel_kb": round(os.path.getsize(panel_path) / 1024.0, 1),
            "dcb_mean": {k: round(float(np.mean(v)), 3) for k, v in dcb.items()},
            "cfa_raw_mean": {k: round(float(np.mean(v)), 3) for k, v in cfa_raw.items()},
            "cfa_fair_mean": {k: round(float(np.mean(v)), 3) for k, v in cfa_fair.items()},
        }
        print(f"=== REAL {name} ===", json.dumps(summary_all[name], indent=1), flush=True)

    with open(os.path.join(out_dir, "real_results.json"), "w") as f:
        json.dump(summary_all, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synth", "chart", "real"])
    ap.add_argument("--bursts", nargs="+", default=list(BURSTS.keys()))
    ap.add_argument("--out-dir", default="/tmp/rawsr/wp12")
    ap.add_argument("--panel-dir", default=os.path.join(ROOT, "docs", "wp12"))
    args = ap.parse_args()
    if args.mode == "synth":
        run_synth(args.out_dir)
    elif args.mode == "chart":
        run_chart(args.out_dir)
    else:
        run_real(args.bursts, args.out_dir, args.panel_dir)
