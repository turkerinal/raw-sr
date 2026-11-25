#!/usr/bin/env python3
"""WP5 evaluation: synthetic non-rigid + moving patch, real held-out (DCB & CFA),
timing/RSS benchmark, and pixel-aligned visual panels.
"""

from __future__ import annotations

import argparse
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
    RawFrame,
    SuperResolution,
    estimate_homography,
    load_raw_linear,
    predict_lr_from_hr,
)
from audit_c11_part2 import (
    BGGR,
    _demosaic_bggr,
    _mosaic_from_rgb,
    compute_slanted_edge_mtf50,
    make_resolution_target,
    synthesize_target_burst,
)
from sr_acceptance import heldout_baseline_lr, heldout_compare, heldout_valid_mask


# --------------------------------------------------------------------------- #
# 1. Synthetic evaluation (rigid + non-rigid parallax + moving patch)
# --------------------------------------------------------------------------- #
def synthesize_nonrigid_burst(
    ratio: int = 2,
    lr: int = 256,
    n: int = 9,
    seed: int = 7,
    warp_amp_lr: float = 0.75,
    sigma_lens_lr: float = 0.35,
    full_well: float = 2048.0,
    read_noise: float = 1.5,
):
    """Extend synthesize_target_burst with smooth non-rigid warp + moving patch."""
    rng = np.random.default_rng(seed)
    hr_size = lr * ratio
    scene = make_resolution_target(hr_size=hr_size, tilt_deg=5.7)
    blurred = cv2.GaussianBlur(
        scene, (0, 0), sigma_lens_lr * ratio, borderType=cv2.BORDER_REFLECT
    )
    box = cv2.blur(blurred, (ratio, ratio), borderType=cv2.BORDER_REFLECT)

    offs = np.stack([rng.uniform(-3.0, 3.0, n), rng.uniform(-3.0, 3.0, n)], axis=1)
    offs[0] = (0.0, 0.0)

    xs, ys = np.meshgrid(np.arange(lr, dtype=np.float32), np.arange(lr, dtype=np.float32))
    off = 0.5 * (float(ratio) - 1.0)

    # Moving patch region in LR coords (top-left flat/edge border area)
    patch_lr = (48, 32, 36, 36)  # (x, y, w, h)
    mosaics, rgb_frames = [], []
    for k in range(n):
        if k == 0:
            ndx = np.zeros((lr, lr), dtype=np.float32)
            ndy = np.zeros((lr, lr), dtype=np.float32)
        else:
            ax = float(rng.uniform(0.35, warp_amp_lr)) * rng.choice([-1.0, 1.0])
            ay = float(rng.uniform(0.35, warp_amp_lr)) * rng.choice([-1.0, 1.0])
            phx = float(rng.uniform(0.0, 2.0 * np.pi))
            phy = float(rng.uniform(0.0, 2.0 * np.pi))
            ndx = (ax * np.sin(2.0 * np.pi * xs / lr + phx) * np.cos(np.pi * ys / lr)).astype(np.float32)
            ndy = (ay * np.cos(2.0 * np.pi * ys / lr + phy) * np.sin(np.pi * xs / lr)).astype(np.float32)

        mx = (ratio * (xs + np.float32(offs[k, 0]) + ndx) + off).astype(np.float32)
        my = (ratio * (ys + np.float32(offs[k, 1]) + ndy) + off).astype(np.float32)
        rgb = cv2.remap(box, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Inject a moving bright patch in 3 non-reference frames
        if k in (2, 5, 7):
            px, py, pw, ph = patch_lr
            shift_p = (k - 2) * 3
            rgb[py:py + ph, px + shift_p:px + shift_p + pw] = 0.90

        rgb_frames.append(rgb)
        m = _mosaic_from_rgb(rgb)
        m = (
            rng.poisson(np.clip(m, 0, 1) * full_well)
            + rng.normal(0.0, read_noise, m.shape)
        ) / full_well
        m = np.clip(m, 0.0, 1.0).astype(np.float32)
        mosaics.append(m)

    margin = 4 * ratio + 8
    mask = np.zeros((hr_size, hr_size), dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    return {
        "target": box,
        "mosaics": mosaics,
        "rgb": rgb_frames,
        "offsets": offs,
        "mask": mask,
        "patch_lr": patch_lr,
    }


def _build_frames_from_mosaics(mosaics):
    frames = []
    for k, m in enumerate(mosaics):
        rgb = _demosaic_bggr(m)
        frames.append(RawFrame(name=f"s{k}", rgb=rgb, green=rgb[..., 1].copy(),
                               mosaic=m, pattern=BGGR))
    return frames


def _align_gt_to_ref(target, offs, ref_idx, ratio):
    hr_size = target.shape[0]
    dx = float(offs[ref_idx, 0]) * ratio
    dy = float(offs[ref_idx, 1]) * ratio
    xs = np.arange(hr_size, dtype=np.float32)[None, :] + dx
    ys = np.arange(hr_size, dtype=np.float32)[:, None] + dy
    xs, ys = np.broadcast_arrays(xs, ys)
    return cv2.remap(target, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _psnr(img, gt, mask):
    mse = float(np.mean(((img - gt) ** 2)[mask]))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def run_synth():
    results = {}
    for kind in ("rigid", "nonrigid_moving"):
        if kind == "rigid":
            data = synthesize_target_burst(ratio=2, lr=256, n=9, seed=7)
        else:
            data = synthesize_nonrigid_burst(ratio=2, lr=256, n=9, seed=7, warp_amp_lr=0.80)
        frames = _build_frames_from_mosaics(data["mosaics"])
        mask = data["mask"]

        out_k = {}
        for tag, kwargs in [
            ("master", dict(local_align=False)),
            ("local_align", dict(local_align=True, local_robust=True)),
            ("local_select_70", dict(local_align=True, local_robust=True, local_select=0.7)),
        ]:
            res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, **kwargs).run(frames)
            gt_ref = _align_gt_to_ref(data["target"], data["offsets"], res.reference_index, 2)
            p_all = _psnr(res.hr, gt_ref, mask)
            mtf = compute_slanted_edge_mtf50(res.hr, ratio=2.0)
            entry = {
                "gt_psnr_db": round(p_all, 3),
                "mtf50": round(float(mtf), 4),
                "ghost_energy": round(float(res.metrics.get("ghost_energy", 0.0)), 6),
            }
            if kind == "nonrigid_moving":
                px, py, pw, ph = data["patch_lr"]
                # Measure ghosting RMSE inside the moving patch region (HR coords)
                sl_y = slice(py * 2, (py + ph) * 2)
                sl_x = slice(px * 2, (px + pw + 20) * 2)
                patch_rmse = float(np.sqrt(np.mean((res.hr[sl_y, sl_x] - gt_ref[sl_y, sl_x]) ** 2)))
                entry["moving_patch_rmse"] = round(patch_rmse, 5)
            out_k[tag] = entry

            # Also CFA drizzle on same registration/local_shifts
            if tag in ("master", "local_align"):
                km = [frames[i].mosaic for i in res.kept_indices]
                cfa = cfa_drizzle(
                    km, res.transforms, 2.0, BGGR,
                    weights=res.weights.tolist(), pixfrac=0.5,
                    local_shifts=res.local_shifts,
                    local_weights=res.local_weights,
                )
                p_cfa = _psnr(cfa.hr, gt_ref, mask)
                mtf_cfa = compute_slanted_edge_mtf50(cfa.hr, ratio=2.0)
                c_entry = {
                    "gt_psnr_db": round(p_cfa, 3),
                    "mtf50": round(float(mtf_cfa), 4),
                }
                if kind == "nonrigid_moving":
                    px, py, pw, ph = data["patch_lr"]
                    sl_y = slice(py * 2, (py + ph) * 2)
                    sl_x = slice(px * 2, (px + pw + 20) * 2)
                    c_entry["moving_patch_rmse"] = round(
                        float(np.sqrt(np.mean((cfa.hr[sl_y, sl_x] - gt_ref[sl_y, sl_x]) ** 2))), 5
                    )
                out_k[f"{tag}_cfa_drizzle"] = c_entry

        results[kind] = out_k
    print("SYNTH:", json.dumps(results, indent=2), flush=True)
    return results


# --------------------------------------------------------------------------- #
# 2. Real bursts: DCB held-out + CFA held-out + timing/RSS
# --------------------------------------------------------------------------- #
def level_fit(hr, ref_mosaic, pattern):
    h, w = ref_mosaic.shape
    lr = cv2.resize(hr, (w, h), interpolation=cv2.INTER_AREA)
    ch = channel_map(pattern, (h, w))
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


def eval_real_burst(name: str, burst_dir: str, roi_xy: tuple[int, int],
                    n_heldout: int = 0, test_select: bool = False):
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

    hids = (list(range(n)) if n_heldout <= 0 or n_heldout >= n else
            sorted(set(int(round(v)) for v in np.linspace(0, n - 1, n_heldout))))

    configs = [
        ("master", dict(local_align=False)),
        ("local", dict(local_align=True, local_robust=True)),
    ]
    if test_select:
        configs.append(("local_sel70", dict(local_align=True, local_robust=True, local_select=0.7)))

    dcb_gains = {c[0]: [] for c in configs}
    cfa_gains = {
        "master_default": [],
        "local_default": [],
        "master_cfa": [],
        "local_cfa": [],
    }

    for hidx in hids:
        kept = [f for i, f in enumerate(frames) if i != hidx]
        kept_m = [m for i, m in enumerate(mos) if i != hidx]
        hw = frames[hidx]

        res_map = {}
        for cname, ckw in configs:
            sr = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                                 ecc_scale=0.25, **ckw)
            res = sr.run(kept)
            res_map[cname] = res

            ref_green = kept[res.reference_index].green
            W_held = estimate_homography(ref_green, hw.green, ecc_scale=0.25)
            pred_sr = predict_lr_from_hr(res.hr, W_held, 2.0, hw.rgb.shape[:2])
            base_lr, base_cov, _ = heldout_baseline_lr(kept, res)
            dst = (hw.rgb.shape[1], hw.rgb.shape[0])
            base_lr_held = cv2.warpPerspective(base_lr, W_held.astype(np.float32), dst,
                                               flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_REPLICATE)
            base_cov_held = cv2.warpPerspective(base_cov, W_held.astype(np.float32), dst,
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT)
            mask = heldout_valid_mask(pred_sr, res.hr.shape[:2], W_held, 2.0, hw.rgb.shape[:2])
            total_w = float(np.sum(res.weights))
            mask &= base_cov_held >= total_w * 0.999
            comp = heldout_compare(pred_sr, base_lr_held, hw.rgb, mask)
            dcb_gains[cname].append(comp["heldout_gain_db"])

        # CFA-space held-out (master vs local, default vs cfa_drizzle)
        r_m = res_map["master"]
        r_l = res_map["local"]
        ki = list(r_m.kept_indices)
        km = [kept_m[i] for i in ki]
        ref_mos = kept_m[r_m.reference_index]
        cfa_m = cfa_drizzle(km, r_m.transforms, 2.0, pat, r_m.weights.tolist(), pixfrac=0.5).hr

        ki_l = list(r_l.kept_indices)
        km_l = [kept_m[i] for i in ki_l]
        cfa_l = cfa_drizzle(
            km_l, r_l.transforms, 2.0, pat, r_l.weights.tolist(), pixfrac=0.5,
            local_shifts=r_l.local_shifts, local_weights=r_l.local_weights,
        ).hr

        hrs = {
            "master_default": level_fit(r_m.hr, ref_mos, pat),
            "local_default": level_fit(r_l.hr, ref_mos, pat),
            "master_cfa": level_fit(cfa_m, ref_mos, pat),
            "local_cfa": level_fit(cfa_l, ref_mos, pat),
        }
        W_held = estimate_homography(kept[r_m.reference_index].green, hw.green, ecc_scale=0.25)
        m_cfa = evaluate_cfa_heldout(
            hrs, mos[hidx], km, r_m.transforms, r_m.weights.tolist(), W_held, 2.0, pat
        )
        for k in cfa_gains:
            cfa_gains[k].append(m_cfa["per_method"][k]["heldout_cfa_gain_db"])

        print(f"  {name} h={hidx}: DCB master={dcb_gains['master'][-1]:+.3f} "
              f"local={dcb_gains['local'][-1]:+.3f} | CFA m_def={cfa_gains['master_default'][-1]:+.3f} "
              f"l_def={cfa_gains['local_default'][-1]:+.3f} m_cfa={cfa_gains['master_cfa'][-1]:+.3f} "
              f"l_cfa={cfa_gains['local_cfa'][-1]:+.3f}", flush=True)

    summary = {
        "name": name,
        "frames": n,
        "heldout_indices": hids,
        "dcb": {k: {"min": round(float(min(v)), 3), "mean": round(float(np.mean(v)), 3)}
                for k, v in dcb_gains.items()},
        "cfa": {k: {"min": round(float(min(v)), 3), "mean": round(float(np.mean(v)), 3)}
                for k, v in cfa_gains.items()},
    }
    print(f"SUMMARY {name}:", json.dumps(summary), flush=True)
    return summary


# --------------------------------------------------------------------------- #
# 3. Visual panels (single | master | local default | local + cfa_drizzle)
# --------------------------------------------------------------------------- #
def colour_fit(src, dst):
    m = np.zeros(src.shape[:2], bool)
    m[64:-64, 64:-64] = True
    X = np.concatenate([src[m], np.ones((m.sum(), 1), np.float32)], 1)
    M, *_ = np.linalg.lstsq(X, dst[m], rcond=None)
    return (np.concatenate([src, np.ones(src.shape[:2] + (1,), np.float32)], 2) @ M).astype(np.float32)


def phase_align(img, ref):
    (dx, dy), _ = cv2.phaseCorrelate(ref[..., 1].astype(np.float64), img[..., 1].astype(np.float64))
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), (dx, dy)


def make_panel(name: str, burst_dir: str, roi_xy: tuple[int, int], out_dir: str, size: int = 256):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(burst_dir, "*.dng")))
    frames = [load_raw_linear(f, roi=(roi_xy[0], roi_xy[1], 1024, 1024)) for f in files]

    t0 = time.perf_counter()
    res_m = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, ecc_scale=0.25, local_align=False).run(frames)
    t_m = time.perf_counter() - t0

    t0 = time.perf_counter()
    res_l = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, ecc_scale=0.25, local_align=True).run(frames)
    t_l = time.perf_counter() - t0

    ref = res_m.reference_index
    pat = pattern_array(frames[0].pattern)
    km = [frames[i].mosaic for i in res_l.kept_indices]
    cfa_l = cfa_drizzle(
        km, res_l.transforms, 2.0, pat, res_l.weights.tolist(), pixfrac=0.5,
        local_shifts=res_l.local_shifts, local_weights=res_l.local_weights,
    ).hr

    single = cv2.resize(frames[ref].rgb, (res_m.hr.shape[1], res_m.hr.shape[0]), interpolation=cv2.INTER_CUBIC)
    imgs = {
        "1. Single (bicubic)": single,
        "2. Master (global H)": res_m.hr,
        "3. Local align (default)": res_l.hr,
        "4. Local align + CFA drizzle": colour_fit(cfa_l, res_m.hr),
    }
    imgs["4. Local align + CFA drizzle"], off_cfa = phase_align(imgs["4. Local align + CFA drizzle"], res_m.hr)

    # Pick most textured window away from borders
    g = cv2.cvtColor(np.clip(res_m.hr, 0, 1), cv2.COLOR_RGB2GRAY)
    e = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F)), -1, (size, size))
    e[:size, :] = 0; e[-size:, :] = 0; e[:, :size] = 0; e[:, -size:] = 0
    cy, cx = np.unravel_index(np.argmax(e), e.shape)
    x0, y0 = int(cx - size // 2), int(cy - size // 2)

    crops = {k: v[y0:y0 + size, x0:x0 + size] for k, v in imgs.items()}
    gain = 0.85 / max(float(np.percentile(crops["2. Master (global H)"], 99.5)), 1e-6)
    tiles = []
    for k, c in crops.items():
        u = (np.clip(c * gain, 0, 1) ** (1.0 / 2.2) * 255.0).astype(np.uint8)
        u = cv2.resize(u, (2 * size, 2 * size), interpolation=cv2.INTER_NEAREST)
        u = cv2.cvtColor(u, cv2.COLOR_RGB2BGR)
        cv2.rectangle(u, (0, 0), (2 * size, 30), (20, 20, 20), -1)
        cv2.putText(u, k, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(u)
    gap = np.full((2 * size, 6, 3), 30, np.uint8)
    top = np.hstack([tiles[0], gap, tiles[1]])
    bot = np.hstack([tiles[2], gap, tiles[3]])
    canvas = np.vstack([top, np.full((6, top.shape[1], 3), 30, np.uint8), bot])
    out_path = os.path.join(out_dir, f"{name}_2x.png")
    cv2.imwrite(out_path, canvas)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    info = {
        "name": name,
        "path": out_path,
        "bytes": os.path.getsize(out_path),
        "crop": [x0, y0],
        "t_master_s": round(t_m, 2),
        "t_local_s": round(t_l, 2),
        "local_shift_rms_lr": round(res_l.metrics.get("local_shift_rms_lr", 0.0), 4),
        "local_shift_max_lr": round(res_l.metrics.get("local_shift_max_lr", 0.0), 4),
        "local_mean_weight": round(res_l.metrics.get("local_mean_weight", 1.0), 4),
        "peak_rss_mb": round(rss_mb, 1),
    }
    print("PANEL:", json.dumps(info), flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synth", "real", "panels", "bench20"])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-json", default="/tmp/rawsr/wp5_results.json")
    a = ap.parse_args()

    bursts = [
        ("hdrplus_0919", os.path.join(a.data_root, "hdrplus/0919_20150910_131720_137"), (1504, 1000), 0, False),
        ("own9", os.path.join(a.data_root, "own9"), (1650, 1240), 0, False),
        ("samsung14", os.path.join(a.data_root, "samsung14"), (2400, 520), 5, True),
        ("pixel4a20", os.path.join(a.data_root, "inria_friant20"), (2080, 800), 5, True),
        ("pixel4a9", os.path.join(a.data_root, "inria_friant20_first9"), (2080, 800), 5, False),
    ]

    if a.mode == "synth":
        res = run_synth()
        with open(a.out_json, "w") as fh:
            json.dump(res, fh, indent=2)
    elif a.mode == "real":
        out = {}
        for name, bdir, rxy, nho, tsel in bursts:
            out[name] = eval_real_burst(name, bdir, rxy, n_heldout=nho, test_select=tsel)
        with open(a.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
    elif a.mode == "panels":
        out_dir = os.path.join(ROOT, "docs", "wp5")
        out = {}
        for name, bdir, rxy, _, _ in bursts[:4]:
            out[name] = make_panel(name, bdir, rxy, out_dir)
        with open(a.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
    elif a.mode == "bench20":
        # Standalone peak-RSS + wall-time benchmark on 20-frame Pixel 4a
        bdir = os.path.join(a.data_root, "inria_friant20")
        files = sorted(glob.glob(os.path.join(bdir, "*.dng")))
        frames = [load_raw_linear(f, roi=(2080, 800, 1024, 1024)) for f in files]
        for label, la in [("master", False), ("local", True)]:
            t0 = time.perf_counter()
            res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, ecc_scale=0.25, local_align=la).run(frames)
            dt = time.perf_counter() - t0
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            print(f"BENCH20 {label}: {dt:.2f}s peak_rss={rss:.1f} MiB local_s={res.metrics.get('local_align_seconds', 0):.2f}s", flush=True)


if __name__ == "__main__":
    main()
