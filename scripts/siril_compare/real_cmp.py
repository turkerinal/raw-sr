"""Real-burst comparison in Bayer measurement space (CFA held-out metric).

For each held-out frame h: reconstruct from the other frames with
  - raw_sr default (demosaic_drizzle), raw_sr cfa_kr, raw_sr cfa_drizzle,
  - Siril 1.4.5 Bayer drizzle (registration injected from raw_sr's run,
    because Siril's star registration fails on daytime scenes),
then predict h's own Bayer samples and score against the CFA-domain denoised
stack of the same frames (raw_sr.core.cfa_heldout.evaluate_cfa_heldout).
Every HR estimate gets the same per-channel affine level fit against the
*reference* frame's mosaic (a kept frame, never the held-out one).
"""
import argparse, glob, json, os, shutil, sys, time
import numpy as np, cv2

ROOT = "."
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raw_sr.core.superres import SuperResolution, load_raw_linear, estimate_homography
from raw_sr.core.cfa_drizzle import (apply_level_gains, calibrate_cfa_levels, load_cfa_frame,
                                     pattern_array, channel_map, cfa_drizzle)
from raw_sr.core.cfa_heldout import evaluate_cfa_heldout
from sirilio import write_cfa_fits, read_fits, run_siril, inject_registration

WORK = "output/siril/real"
LET = {0: "R", 1: "G", 2: "B", 3: "G"}


def fourier_shift(img, dx, dy):
    H, W = img.shape[:2]
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    ph = np.exp(-2j * np.pi * (fx * dx + fy * dy))
    return np.stack([np.real(np.fft.ifft2(np.fft.fft2(img[..., c]) * ph)) for c in range(3)], 2).astype(np.float32)


def level_fit(hr, ref_mosaic, pattern, ratio):
    """Per-channel affine fit of HR (area-downsampled) to the reference mosaic."""
    h, w = ref_mosaic.shape
    lr = cv2.resize(hr, (w, h), interpolation=cv2.INTER_AREA)
    ch = channel_map(pattern, (h, w))
    out = hr.copy(); fit = []
    m = np.zeros((h, w), bool); m[32:-32, 32:-32] = True
    for c in range(3):
        sel = (ch == c) & m
        x = lr[..., c][sel].astype(np.float64); y = ref_mosaic[sel].astype(np.float64)
        (g, o), *_ = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)
        out[..., c] = hr[..., c] * g + o; fit.append([round(float(g), 4), round(float(o), 5)])
    return out.astype(np.float32), fit


def siril_hr(mosaics, transforms, ref_pos, ratio, pattern, pixfrac, kernel, tag):
    wd = os.path.join(WORK, tag)
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    bp = "".join(LET[int(v)] for v in np.asarray(pattern).ravel())
    for k, m in enumerate(mosaics):
        write_cfa_fits(os.path.join(wd, f"s_{k + 1:05d}.fit"), np.clip(m, 0, 1) * 0.2, bp)
    h = mosaics[0].shape[0]
    F = np.array([[1, 0, 0], [0, -1, h - 1], [0, 0, 1]], float)
    Hs = [F @ np.linalg.inv(np.asarray(T, float)) @ F for T in transforms]
    n = len(mosaics)
    with open(os.path.join(wd, "s_.seq"), "w") as f:
        f.write(f"#Siril sequence file.\nS 's_' 1 {n} {n} 5 {ref_pos} 4 0 0 0\n")
    inject_registration(os.path.join(wd, "s_.seq"), Hs, ref=ref_pos)
    t0 = time.perf_counter()
    run_siril(wd, [f"seqapplyreg s_ -drizzle -scale={ratio} -pixfrac={pixfrac} -kernel={kernel}",
                   "stack r_s_ rej w 3 3 -nonorm -32b -out=res"])
    dt = time.perf_counter() - t0
    a, hdr = read_fits(os.path.join(wd, "res.fit"))
    shutil.rmtree(wd, ignore_errors=True)
    a = np.transpose(a, (1, 2, 0)).astype(np.float32)
    if hdr.get("ROWORDER", "BOTTOM-UP").upper().startswith("BOTTOM"):
        a = a[::-1]
    return a, dt, float((a == 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst"); ap.add_argument("--name")
    ap.add_argument("--roi", type=int, nargs=2); ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--n-heldout", type=int, default=0, help="0 = all frames")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.burst, "*.dng")))
    roi = (a.roi[0], a.roi[1], 1024, 1024)
    frames = [load_raw_linear(f, roi=roi) for f in files]
    cm = [load_cfa_frame(f, roi=roi) for f in files]
    for i, cf in enumerate(cm):
        g, _ = calibrate_cfa_levels(cf, frames[i].rgb); cm[i] = apply_level_gains(cf, g)
    mos = [c.mosaic for c in cm]; pat = pattern_array(cm[0].pattern)
    n = len(frames)
    hids = list(range(n)) if a.n_heldout <= 0 or a.n_heldout >= n else \
        sorted(set(int(round(v)) for v in np.linspace(0, n - 1, a.n_heldout)))
    rows = []
    for hidx in hids:
        kept = [f for i, f in enumerate(frames) if i != hidx]
        kept_m = [m for i, m in enumerate(mos) if i != hidx]
        t0 = time.perf_counter()
        res = SuperResolution(ratio=a.ratio, blur_threshold_ratio=0.0).run(kept)
        t_def = time.perf_counter() - t0
        ki = list(res.kept_indices); km = [kept_m[i] for i in ki]
        ref_pos = ki.index(res.reference_index)
        ref_mos = kept_m[res.reference_index]
        t0 = time.perf_counter()
        kr = SuperResolution(ratio=a.ratio, blur_threshold_ratio=0.0, fusion="cfa_kr").run(kept)
        t_kr = time.perf_counter() - t0
        cfa = cfa_drizzle(km, res.transforms, a.ratio, pat, res.weights.tolist(), pixfrac=0.5).hr
        hrs = {"rawsr_default": res.hr, "rawsr_cfa_kr": kr.hr, "rawsr_cfa_drizzle": cfa}
        info = {"seconds": {"rawsr_default": round(t_def, 1), "rawsr_cfa_kr": round(t_kr, 1)},
                "kr_same_ref": int(kr.reference_index) == int(res.reference_index)}
        siril_mos = [kept[i].mosaic for i in ki]
        for pf, ker in ((0.7, "gaussian"), (1.0, "square")):
            tag = f"siril_{ker}_pf{pf}"
            s, dt, z = siril_hr(siril_mos, res.transforms, ref_pos, int(a.ratio), kept[0].pattern,
                                pf, ker, f"{a.name}_h{hidx}_{tag}")
            s_fit, _ = level_fit(s, ref_mos, pat, a.ratio)
            (dx, dy), resp = cv2.phaseCorrelate(res.hr[..., 1].astype(np.float64), s_fit[..., 1].astype(np.float64))
            hrs[tag] = fourier_shift(s_fit, -dx, -dy)
            info.setdefault("siril", {})[tag] = {"seconds": round(dt, 1), "zero_frac": round(z, 4),
                                                 "offset_corrected_hr_px": [round(dx, 3), round(dy, 3)]}
        fits = {}
        for k in list(hrs):
            hrs[k], fits[k] = level_fit(hrs[k], ref_mos, pat, a.ratio)
        hrs["rawsr_cfa_kr_nofit"] = kr.hr
        W_held = estimate_homography(kept[res.reference_index].green, frames[hidx].green, ecc_scale=0.25)
        m = evaluate_cfa_heldout(hrs, mos[hidx], km, res.transforms, res.weights.tolist(),
                                 W_held, a.ratio, pat)
        g = {k: round(v["heldout_cfa_gain_db"], 3) for k, v in m["per_method"].items()}
        rows.append({"heldout": hidx, "gain_db": g, "level_fit": fits, **info})
        print(a.name, "h", hidx, json.dumps(g), json.dumps(info.get("siril")), flush=True)
    summ = {}
    for k in rows[0]["gain_db"]:
        v = [r["gain_db"][k] for r in rows]
        summ[k] = {"min": round(min(v), 3), "mean": round(float(np.mean(v)), 3)}
    out = {"name": a.name, "ratio": a.ratio, "roi": roi, "frames": n, "heldout": hids,
           "summary": summ, "rows": rows}
    json.dump(out, open(os.path.join(WORK, f"{a.name}_{int(a.ratio)}x.json"), "w"), indent=1)
    print("SUMMARY", a.name, json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
