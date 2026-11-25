"""WP9 hypothesis-driven diagnosis: why cfa_kr wins on synthetic (+3.6 dB)
but loses on real bursts (-0.3 to -1.8 dB vs plain cfa_drizzle).

Tests hypotheses (a)-(f) in controlled experiments:
  (a) Kernel bandwidth/footprint sweep vs injected registration noise
  (b) Registration error sensitivity: default vs cfa_drizzle vs cfa_kr
  (c) Real sensor effects (Gr/Gb imbalance, hot pixels, black/white, DNG meta,
      load_raw_linear vs load_cfa_frame mosaic diff, synthetic injection)
  (d) Exposure normalisation interaction
  (e) IBP stage after cfa_kr (IBP=0 vs IBP=1 on real)
  (f) Edge handling / coverage holes (low-weight HR pixel fractions)
"""
from __future__ import annotations

import argparse
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
    compute_slanted_edge_mtf50, psnr, immerkaer_sigma,
)
from raw_sr.core.superres import (
    RawFrame, SuperResolution, load_raw_linear, estimate_homography,
)
from raw_sr.core.cfa_drizzle import (
    cfa_drizzle, cfa_kernel_regression, load_cfa_frame,
    calibrate_cfa_levels, apply_level_gains, pattern_array, channel_map,
    compute_structure_tensor_precision,
)
from raw_sr.core.cfa_heldout import evaluate_cfa_heldout


def _gain_fit(img, target, mask):
    x = img[..., 1][mask].astype(np.float64)
    y = target[..., 1][mask].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], 1)
    (g, o), *_ = np.linalg.lstsq(A, y, rcond=None)
    return (img * g + o).astype(np.float32)


def run_synthetic_diag():
    """Hypotheses (a), (b), (c-syn), (f-syn) on synthetic 2x burst."""
    b = synthesize_target_burst(ratio=2, lr=256, n=9, seed=7)
    mos = b["mosaics"]
    dems = [_demosaic_bggr(m) for m in mos]
    ref_g = dems[0][..., 1].copy()
    target = b["target"]
    mask = b["mask"]
    oh, ow = target.shape[:2]
    fb_hr = cv2.resize(dems[0], (ow, oh), interpolation=cv2.INTER_CUBIC)
    tru_offsets = np.asarray(b["offsets"], dtype=np.float64)

    # True ref->frame transforms: frame k samples at (x + offs[k]), so ref->frame is T(-offs[k])
    # Wait: verify sign from synthesize_target_burst!
    # Let's test both signs on cfa_drizzle to be 100% sure.
    def make_aligns(offs, noise_std, rng):
        out = []
        for k, (dx, dy) in enumerate(offs):
            if k == 0:
                out.append((0.0, 0.0))
            else:
                nx = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
                ny = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
                out.append((float(dx + nx), float(dy + ny)))
        return out

    # Check sign convention
    a_pos = make_aligns(tru_offsets, 0.0, np.random.default_rng(0))
    a_neg = make_aligns(-tru_offsets, 0.0, np.random.default_rng(0))
    p_pos = psnr(_gain_fit(cfa_drizzle(mos, a_pos, 2.0, BGGR, pixfrac=0.5).hr, target, mask), target, mask)
    p_neg = psnr(_gain_fit(cfa_drizzle(mos, a_neg, 2.0, BGGR, pixfrac=0.5).hr, target, mask), target, mask)
    base_offs = tru_offsets if p_pos >= p_neg else -tru_offsets
    print(f"[sign check] pos={p_pos:.2f} dB, neg={p_neg:.2f} dB -> using {'pos' if p_pos >= p_neg else 'neg'}")

    # (a) & (b): Registration noise sweep x methods / k_detail
    print("\n=== (a)+(b) Synthetic 2x GT PSNR (dB) vs Registration Error (LR px) ===")
    reg_noises = [0.0, 0.1, 0.2, 0.3]
    k_details = [0.18, 0.28, 0.40, 0.55]
    results_ab = {}
    header = f"{'reg_err':>7} | {'cfa_drz':>8} | " + " | ".join(f"kr_{kd:.2f}" for kd in k_details) + " | " + " | ".join(f"d_kr_{kd:.2f}" for kd in k_details)
    print(header)
    print("-" * len(header))
    for sig in reg_noises:
        rng = np.random.default_rng(42)
        al = make_aligns(base_offs, sig, rng)
        drz = cfa_drizzle(mos, al, 2.0, BGGR, pixfrac=0.5).hr
        p_drz = psnr(_gain_fit(drz, target, mask), target, mask)
        row = {"cfa_drizzle": round(p_drz, 3)}
        p_krs = []
        for kd in k_details:
            kr = cfa_kernel_regression(mos, al, 2.0, BGGR, ref_green=ref_g, fallback_hr=fb_hr, k_detail=kd).hr
            pk = psnr(_gain_fit(kr, target, mask), target, mask)
            row[f"kr_{kd:.2f}"] = round(pk, 3)
            p_krs.append(pk)
        results_ab[str(sig)] = row
        vals = " | ".join(f"{pk:7.2f}" for pk in p_krs)
        diffs = " | ".join(f"{pk - p_drz:+9.2f}" for pk in p_krs)
        print(f"{sig:7.2f} | {p_drz:8.2f} | {vals} | {diffs}")

    # (c-syn): Inject Gr/Gb imbalance (5%, 10%) and hot pixels (0.1%) into synthetic
    print("\n=== (c-syn) Synthetic 2x Sensor Effects (reg_err=0.1 LR px) ===")
    rng = np.random.default_rng(42)
    al_01 = make_aligns(base_offs, 0.1, rng)
    results_csyn = {}
    for label, mod_fn in [
        ("clean", lambda m: m.copy()),
        ("gr_gb_5pct", lambda m: _inject_gr_gb(m, BGGR, 1.05)),
        ("gr_gb_10pct", lambda m: _inject_gr_gb(m, BGGR, 1.10)),
        ("hot_px_0.2pct", lambda m: _inject_hot(m, 0.002)),
    ]:
        m_mod = [mod_fn(m) for m in mos]
        d0 = _demosaic_bggr(m_mod[0])
        rg0 = d0[..., 1].copy()
        fb0 = cv2.resize(d0, (ow, oh), interpolation=cv2.INTER_CUBIC)
        p_drz = psnr(_gain_fit(cfa_drizzle(m_mod, al_01, 2.0, BGGR, pixfrac=0.5).hr, target, mask), target, mask)
        p_kr18 = psnr(_gain_fit(cfa_kernel_regression(m_mod, al_01, 2.0, BGGR, ref_green=rg0, fallback_hr=fb0, k_detail=0.18).hr, target, mask), target, mask)
        p_kr40 = psnr(_gain_fit(cfa_kernel_regression(m_mod, al_01, 2.0, BGGR, ref_green=rg0, fallback_hr=fb0, k_detail=0.40).hr, target, mask), target, mask)
        results_csyn[label] = {"cfa_drizzle": round(p_drz, 3), "kr_0.18": round(p_kr18, 3), "kr_0.40": round(p_kr40, 3)}
        print(f"{label:15s} | drz={p_drz:.2f} | kr_0.18={p_kr18:.2f} ({p_kr18-p_drz:+.2f}) | kr_0.40={p_kr40:.2f} ({p_kr40-p_drz:+.2f})")

    return {"ab": results_ab, "c_syn": results_csyn}


def _inject_gr_gb(m, pat, factor):
    out = m.copy()
    # BGGR: (0,1) is G1, (1,0) is G2
    out[0::2, 1::2] *= factor
    return np.clip(out, 0.0, 1.0)


def _inject_hot(m, frac):
    out = m.copy()
    rng = np.random.default_rng(123)
    sel = rng.random(out.shape) < frac
    out[sel] = 1.0
    return out


def inspect_real_burst(name, burst_dir, roi_xy, max_frames=None):
    """Run (a-real), (c-real), (d), (e), (f-real) on one real burst (1 held-out frame)."""
    import rawpy
    files = sorted(glob.glob(os.path.join(burst_dir, "*.dng")))
    if max_frames:
        files = files[:max_frames]
    roi = (roi_xy[0], roi_xy[1], 1024, 1024)

    # (c-real) Sensor metadata & pixel properties on frame 0
    with rawpy.imread(files[0]) as raw:
        pat = np.asarray(raw.raw_pattern, dtype=np.int8).copy()
        bl = raw.black_level_per_channel
        wl = float(raw.white_level)
        wb = raw.camera_whitebalance[:4]
        vis = raw.raw_image_visible
        x0, y0 = roi[0] - (roi[0] % 2), roi[1] - (roi[1] % 2)
        crop = vis[y0:y0 + 1024, x0:x0 + 1024].astype(np.float64)
        # Find two green positions
        g_pos = [(r, c) for r in range(2) for c in range(2) if int(pat[r, c]) in (1, 3)]
        g1 = crop[g_pos[0][0]::2, g_pos[0][1]::2] - bl[int(pat[g_pos[0]])]
        g2 = crop[g_pos[1][0]::2, g_pos[1][1]::2] - bl[int(pat[g_pos[1]])]
        gr_gb_ratio = float(np.median(g1) / max(np.median(g2), 1e-6))
        # Hot pixel estimate on green1
        g1_med = cv2.medianBlur(g1.astype(np.float32), 3)
        diff = np.abs(g1 - g1_med)
        mad = float(np.median(diff)) * 1.4826
        hot_frac = float((diff > max(10.0 * mad, 50.0)).mean())

    frames = [load_raw_linear(f, roi=roi) for f in files]
    cm = [load_cfa_frame(f, roi=roi) for f in files]
    for i, cf in enumerate(cm):
        g, _ = calibrate_cfa_levels(cf, frames[i].rgb)
        cm[i] = apply_level_gains(cf, g)
    mos = [c.mosaic for c in cm]
    pat = pattern_array(cm[0].pattern)

    # Compare frames[0].mosaic vs mos[0]
    mos_diff = float(np.abs(frames[0].mosaic - mos[0]).max()) if frames[0].mosaic is not None else -1.0
    noise_sig = immerkaer_sigma(frames[0].green)

    # Hold out middle frame
    hidx = len(frames) // 2
    kept = [f for i, f in enumerate(frames) if i != hidx]
    kept_m = [m for i, m in enumerate(mos) if i != hidx]

    # Run default SR once to get transforms, weights, ref_index
    res_def = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(kept)
    ki = list(res_def.kept_indices)
    km = [kept_m[i] for i in ki]
    ref_pos = ki.index(res_def.reference_index)
    ref_g = kept[res_def.reference_index].green
    oh, ow = res_def.hr.shape[:2]
    fb_hr = cv2.resize(kept[res_def.reference_index].rgb, (ow, oh), interpolation=cv2.INTER_CUBIC)

    # (e) IBP=0 vs IBP=1 for cfa_kr
    res_kr_ibp1 = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, fusion="cfa_kr", ibp_iterations=1).run(kept)
    res_kr_ibp0 = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, fusion="cfa_kr", ibp_iterations=0).run(kept)

    # (d) Exposure normalisation: compare kept[i].mosaic (with SuperResolution exp_norm) vs raw km (no exp_norm)
    # Direct cfa_drizzle vs cfa_kr sweeps (no IBP) on identical km, transforms, weights
    hrs = {
        "default_ibp1": res_def.hr,
        "cfa_drizzle": cfa_drizzle(km, res_def.transforms, 2.0, pat, res_def.weights.tolist(), pixfrac=0.5).hr,
        "sr_cfa_kr_ibp1": res_kr_ibp1.hr,
        "sr_cfa_kr_ibp0": res_kr_ibp0.hr,
    }
    cov_stats = {}
    for kd in (0.18, 0.28, 0.40, 0.55, 0.70):
        r_kr = cfa_kernel_regression(
            km, res_def.transforms, 2.0, pat, ref_green=ref_g,
            weights=res_def.weights.tolist(), fallback_hr=fb_hr, k_detail=kd,
        )
        hrs[f"direct_kr_{kd:.2f}"] = r_kr.hr
        cov_stats[f"kr_{kd:.2f}"] = {
            "cov_frac": round(float(r_kr.metrics["coverage_fraction"]), 4),
            "mean_sig_p": round(float(r_kr.metrics["mean_sigma_perp"]), 3),
            "mean_sig_t": round(float(r_kr.metrics["mean_sigma_para"]), 3),
        }

    # Evaluate CFA-space held-out
    W_held = estimate_homography(kept[res_def.reference_index].green, frames[hidx].green, ecc_scale=0.25)
    ev = evaluate_cfa_heldout(hrs, mos[hidx], km, res_def.transforms, res_def.weights.tolist(), W_held, 2.0, pat)
    gains = {k: round(v["heldout_cfa_gain_db"], 3) for k, v in ev["per_method"].items()}
    per_ch = {k: {c: round(val, 2) if val is not None else None for c, val in v["heldout_cfa_psnr_db_per_channel"].items()}
              for k, v in ev["per_method"].items()}

    out = {
        "name": name,
        "n_frames": len(files),
        "heldout": hidx,
        "sensor": {
            "black": [float(x) for x in bl],
            "white": wl,
            "wb": [round(float(x), 3) for x in wb],
            "gr_gb_ratio": round(gr_gb_ratio, 4),
            "hot_frac": round(hot_frac, 6),
            "noise_sigma": round(noise_sig, 5),
            "mosaic_diff_max": round(mos_diff, 6),
            "exp_scales": res_kr_ibp0.metrics.get("exposure_scales"),
            "align_residuals": [round(float(x), 4) for x in res_def.align_residuals],
        },
        "cfa_heldout_gain_db": gains,
        "cov_stats": cov_stats,
    }
    print(f"\n=== Real Burst: {name} (n={len(files)}, h={hidx}) ===")
    print("  sensor:", json.dumps(out["sensor"]))
    print("  CFA held-out gain (dB):", json.dumps(gains))
    print("  cov_stats:", json.dumps(cov_stats))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["syn", "real", "all"], default="syn")
    ap.add_argument("--out", default="/tmp/rawsr/wp9/diag.json")
    args = ap.parse_args()

    res = {}
    if args.mode in ("syn", "all"):
        res["syn"] = run_synthetic_diag()
    if args.mode in ("real", "all"):
        bursts = [
            ("own9", os.path.join(ROOT, "test_data"), (1650, 1240)),
            ("hdrplus_0919", "data/hdrplus/0919_20150910_131720_137", (1504, 1000)),
            ("samsung14", "data/samsung14", (2400, 520)),
            ("pixel4a20_first9", "data/inria_friant20_first9", (2080, 800)),
        ]
        res["real"] = {}
        for name, bdir, roi_xy in bursts:
            res["real"][name] = inspect_real_burst(name, bdir, roi_xy)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
