"""WP15 evaluation script: robust single-frame+drizzle PSF, residual-adaptive
envelope overshoot/ringing control, and PSF-aware CFA held-out comparison.

Compares:
  1. Single (bicubic)
  2. Old default (demosaic_drizzle) + WP2 enhance
  3. WP13 MAP (drizzle-only PSF, no envelope constraint)
  4. WP15 MAP (single-frame + drizzle min PSF, residual-adaptive envelope)

Outputs:
  - /tmp/rawsr/wp15/synth_results.json
  - /tmp/rawsr/wp15/real_results.json
  - docs/wp15/<burst>_2x.png (4 publishable bursts)
  - /tmp/rawsr/wp15/pixel4a20_2x.png (internal only)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from raw_sr.core.cfa_heldout import evaluate_cfa_heldout
from raw_sr.core.enhance import effective_stack_noise, enhance_fair_pair
from raw_sr.core.map_sr import estimate_colour_matrix, immerkaer_sigma
from raw_sr.core.cfa_drizzle import pattern_array
from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    estimate_homography,
    load_raw_linear,
)
from scripts.eval_wp13 import (
    BGGR,
    BURSTS,
    _demosaic_bggr,
    choose_safe_exposure,
    compute_slanted_edge_mtf50,
    compute_ssim,
    find_flat_patch,
    measure_colour_drift_pct,
    measure_crop_overshoot,
    measure_gradient_energy,
    measure_nyquist_energy_frac,
    measure_phase_shift,
    pick_high_detail_crop,
    pick_zoom_subcrop,
    psnr,
    render_showcase,
    synthesize_target_burst,
)


def _run_wp15_methods(
    frames: List[RawFrame],
    ratio: float = 2.0,
    ref_idx: Optional[int] = 0,
    ecc_scale: float = 0.25,
    use_gpu: Optional[bool] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict], Any]:
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

    # WP13 baseline: drizzle-only PSF, no envelope constraint
    t0 = time.perf_counter()
    rm_13 = SuperResolution(
        ratio=ratio,
        blur_threshold_ratio=0.0,
        registration="homography",
        ecc_scale=ecc_scale,
        fusion="map",
        map_mode="wp13",
        map_kwargs={"ref_rgb": None, "env_tol": None},
        use_gpu=use_gpu,
    ).run(frames, reference_index=ref_used)
    hrs["map_wp13"] = rm_13.hr
    meta["map_wp13"] = {
        "sec": round(time.perf_counter() - t0, 3),
        "engine": rm_13.metrics.get("cfa_kr_engine"),
        "iters": rm_13.metrics.get("cfa_kr_iterations"),
        "stop": rm_13.metrics.get("cfa_kr_stop_reason"),
        "psf_rgb": rm_13.metrics.get("cfa_kr_psf_sigma_rgb"),
        "psf_source": rm_13.metrics.get("cfa_kr_psf_source"),
    }

    # WP15: robust single-frame+drizzle PSF + residual-adaptive envelope
    t0 = time.perf_counter()
    rm_15 = SuperResolution(
        ratio=ratio,
        blur_threshold_ratio=0.0,
        registration="homography",
        ecc_scale=ecc_scale,
        fusion="map",
        map_mode="wp13",
        use_gpu=use_gpu,
    ).run(frames, reference_index=ref_used)
    hrs["map_wp15"] = rm_15.hr
    meta["map_wp15"] = {
        "sec": round(time.perf_counter() - t0, 3),
        "engine": rm_15.metrics.get("cfa_kr_engine"),
        "iters": rm_15.metrics.get("cfa_kr_iterations"),
        "stop": rm_15.metrics.get("cfa_kr_stop_reason"),
        "psf_rgb": rm_15.metrics.get("cfa_kr_psf_sigma_rgb"),
        "psf_ref_rgb": rm_15.metrics.get("cfa_kr_psf_ref_rgb"),
        "psf_hr_rgb": rm_15.metrics.get("cfa_kr_psf_hr_rgb"),
        "psf_source": rm_15.metrics.get("cfa_kr_psf_source"),
    }

    return hrs, meta, rd


def run_synth(
    out_dir: str,
    sigmas: Tuple[float, ...] = (0.35, 0.6, 0.9),
    ns: Tuple[int, ...] = (9, 20),
    seeds: Tuple[int, ...] = (7, 23),
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
                hrs, meta, _ = _run_wp15_methods(frames, ratio=2.0, ref_idx=0, ecc_scale=0.25)
                row = {}
                for mname, img in hrs.items():
                    row[mname] = {
                        "psnr_scene": round(float(psnr(img, b["scene"], b["mask"])), 2),
                        "ssim_scene": round(compute_ssim(img, b["scene"], b["mask"]), 4),
                        "mtf50": round(float(compute_slanted_edge_mtf50(img, 2.0)), 4),
                        "flat_sigma": round(float(immerkaer_sigma(img[30:90, 30:90, 1])), 6),
                        **meta[mname],
                    }
                results[key] = row
                sstr = " | ".join(f"{k}:{v['psnr_scene']:.2f}" for k, v in row.items())
                print(f"[SYNTH {key}] {sstr}", flush=True)

    agg: Dict[str, Any] = {}
    for sig in sigmas:
        skey = f"s{sig:.2f}"
        matching = [v for k, v in results.items() if k.startswith(skey)]
        methods = list(matching[0].keys())
        agg[skey] = {
            m: {
                "psnr_scene": round(float(np.mean([r[m]["psnr_scene"] for r in matching])), 2),
                "ssim_scene": round(float(np.mean([r[m]["ssim_scene"] for r in matching])), 4),
                "mtf50": round(float(np.mean([r[m]["mtf50"] for r in matching])), 4),
                "flat_sigma": round(float(np.mean([r[m]["flat_sigma"] for r in matching])), 6),
            }
            for m in methods
        }
        print(f"=== SYNTH MEAN {skey} ===", flush=True)
        for m, v in agg[skey].items():
            print(f"  {m:15s} PSNR={v['psnr_scene']:.2f} SSIM={v['ssim_scene']:.4f} MTF50={v['mtf50']:.4f}", flush=True)

    payload = {"per_run": results, "mean_by_sigma": agg}
    with open(os.path.join(out_dir, "synth_results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def run_real(
    burst_names: List[str],
    out_dir: str,
    docs_panel_dir: str,
    n_ho: int = 3,
    skip_heldout: bool = False,
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

        hrs, meta, rd = _run_wp15_methods(frames, ratio=2.0, ref_idx=None, ecc_scale=0.25)
        ref_i = int(rd.reference_index)

        if fixed_cxy is not None:
            x0, y0 = fixed_cxy
        else:
            x0, y0 = pick_high_detail_crop(hrs["old_default"], bic=hrs["single_bicubic"], crop_size=cs, margin=96)
        sl = (slice(y0, y0 + cs), slice(x0, x0 + cs))

        fx, fy, fw, fh = find_flat_patch(hrs["old_default"][..., 1], win=96)
        flat_sl = (slice(fy, fy + fh), slice(fx, fx + fw))

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
                "nyquist_frac": round(measure_nyquist_energy_frac(crop), 5),
                "overshoot": round(measure_crop_overshoot(crop, ref_crop=ref_crop_old), 4),
                "colour_drift_vs_old_pct": round(drift_old, 3),
                "colour_drift_vs_enh_pct": round(drift_enh, 3),
                "mean_rgb": mean_rgb,
                **meta[mname],
            }

        # 4-panel showcase: single | old default + enhance | WP13 | WP15
        panel_keys = ["single_bicubic", "old_enh", "map_wp13", "map_wp15"]
        panel_labels = [
            "1. Single (bicubic)",
            "2. Old default + enhance",
            "3. WP13 MAP",
            "4. WP15 MAP",
        ]
        crops = [hrs[k][sl] for k in panel_keys]
        expo, _ = choose_safe_exposure(crops)
        sub_size = cs // 2
        sub_xy = pick_zoom_subcrop(hrs["old_enh"][sl], sub_size=sub_size)
        dx, dy = measure_phase_shift(hrs["single_bicubic"][sl], hrs["map_wp15"][sl])
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

        # Held-out CFA evaluation: both default (box-only) and per-burst PSF-aware
        cfa_box_mean: Dict[str, float] = {}
        cfa_psf_mean: Dict[str, float] = {}
        burst_psf_rgb = meta["map_wp15"].get("psf_rgb")
        if not skip_heldout:
            methods_ho = [k for k in hrs.keys() if k != "single_bicubic"]
            hids = sorted(set(int(round(v)) for v in np.linspace(0, n - 1, min(n_ho, n))))
            cfa_box: Dict[str, List[float]] = {k: [] for k in methods_ho}
            cfa_psf: Dict[str, List[float]] = {k: [] for k in methods_ho}
            for hidx in hids:
                kept = [f for i, f in enumerate(frames) if i != hidx]
                kept_m = [m for i, m in enumerate(mos) if i != hidx]
                hrs_ho, meta_ho, rd_ho = _run_wp15_methods(kept, ratio=2.0, ref_idx=0, ecc_scale=0.25)
                ki = list(rd_ho.kept_indices)
                km = [kept_m[i] for i in ki]
                ref_pos = rd_ho.reference_index
                C_ref, _ = estimate_colour_matrix(kept[ref_pos].rgb, kept[ref_pos].mosaic, pat)
                W_held = estimate_homography(kept[ref_pos].green, frames[hidx].green, ecc_scale=0.25)
                ho_sub = {k: hrs_ho[k] for k in methods_ho}
                psf_use = meta_ho["map_wp15"].get("psf_rgb") or burst_psf_rgb

                res_box = evaluate_cfa_heldout(
                    ho_sub, mos[hidx], km, rd_ho.transforms, rd_ho.weights.tolist(),
                    W_held, 2.0, pat, colour_matrix=C_ref,
                )
                res_psf = evaluate_cfa_heldout(
                    ho_sub, mos[hidx], km, rd_ho.transforms, rd_ho.weights.tolist(),
                    W_held, 2.0, pat, colour_matrix=C_ref, lens_sigma_lr=psf_use,
                )
                for k in methods_ho:
                    cfa_box[k].append(float(res_box["per_method"][k]["heldout_cfa_gain_db"]))
                    cfa_psf[k].append(float(res_psf["per_method"][k]["heldout_cfa_gain_db"]))
            cfa_box_mean = {k: round(float(np.mean(v)), 3) for k, v in cfa_box.items()}
            cfa_psf_mean = {k: round(float(np.mean(v)), 3) for k, v in cfa_psf.items()}

        summary_all[name] = {
            "frames": n,
            "ref_index": ref_i,
            "publishable": publishable,
            "crop_xy": [x0, y0],
            "flat_roi": [fx, fy, fw, fh],
            "panel": img_info,
            "crop_metrics": crop_metrics,
            "cfa_fair_box_db": cfa_box_mean,
            "cfa_fair_psf_db": cfa_psf_mean,
        }
        print(f"\n=== REAL {name} (ref #{ref_i}) ===", flush=True)
        m15 = meta["map_wp15"]
        print(
            f"  PSF: wp15={m15.get('psf_rgb')} src={m15.get('psf_source')} "
            f"(ref={m15.get('psf_ref_rgb')}, hr={m15.get('psf_hr_rgb')}) | wp13={meta['map_wp13'].get('psf_rgb')}",
            flush=True,
        )
        for mname, cm in crop_metrics.items():
            print(
                f"  {mname:15s} grad={cm['grad_energy']:.5f} flat_n={cm['flat_immerkaer']:.6f} "
                f"nyq={cm['nyquist_frac']:.5f} ov={cm['overshoot']:.3f} "
                f"drift={cm['colour_drift_vs_old_pct']:.2f}% sec={cm['sec']:.2f}",
                flush=True,
            )
        if cfa_box_mean:
            print("  CFA-fair (box): ", json.dumps(cfa_box_mean), flush=True)
            print("  CFA-fair (PSF): ", json.dumps(cfa_psf_mean), flush=True)

    with open(os.path.join(out_dir, "real_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_all, f, indent=2)
    return summary_all


def main() -> None:
    ap = argparse.ArgumentParser(description="WP15 evaluation protocol")
    ap.add_argument("--out-dir", default="/tmp/rawsr/wp15")
    ap.add_argument("--docs-panel-dir", default=os.path.join(ROOT, "docs", "wp15"))
    ap.add_argument("--only", nargs="+", choices=["synth", "real"], default=["synth", "real"])
    ap.add_argument("--bursts", nargs="+", default=list(BURSTS.keys()))
    ap.add_argument("--n-ho", type=int, default=3)
    ap.add_argument("--skip-heldout", action="store_true")
    args = ap.parse_args()

    if "synth" in args.only:
        run_synth(args.out_dir)
    if "real" in args.only:
        run_real(args.bursts, args.out_dir, args.docs_panel_dir, n_ho=args.n_ho, skip_heldout=args.skip_heldout)


if __name__ == "__main__":
    main()
