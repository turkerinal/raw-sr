"""Synthetic GT comparison: Siril 1.4.5 Bayer drizzle vs raw_sr (same burst).

Siril cannot register these star-less scenes itself, so its registration is
injected (exact known shifts, and raw_sr's own estimated shifts). The Siril H
sign convention is calibrated on a separate probe seed, then frozen.
"""
import json, os, shutil, sys, time
import numpy as np, cv2

ROOT = "."
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_c11_part2 import (synthesize_target_burst, _demosaic_bggr, BGGR,
                             compute_slanted_edge_mtf50, psnr, immerkaer_sigma)
from raw_sr.core.superres import RawFrame, SuperResolution
from sirilio import write_cfa_fits, read_fits, run_siril, inject_registration

BASE = "output/siril/syn"


def fourier_shift(img, dx, dy):
    H, W = img.shape[:2]
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.fftfreq(W)[None, :]
    ph = np.exp(-2j * np.pi * (fx * dx + fy * dy))
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[..., c] = np.real(np.fft.ifft2(np.fft.fft2(img[..., c]) * ph))
    return out


def align_to(img, ref, mask):
    a = np.where(mask, img[..., 1], 0).astype(np.float64)
    b = np.where(mask, ref[..., 1], 0).astype(np.float64)
    (dx, dy), _ = cv2.phaseCorrelate(b, a)
    return fourier_shift(img, -dx, -dy), (round(dx, 3), round(dy, 3))


def siril_drizzle(mosaics, Hs, ratio, pixfrac, kernel, tag, rej=True):
    wd = os.path.join(BASE, tag)
    shutil.rmtree(wd, ignore_errors=True); os.makedirs(wd)
    for k, m in enumerate(mosaics):
        write_cfa_fits(os.path.join(wd, f"s_{k + 1:05d}.fit"), m * (0.2 if ratio == 2 else 0.06), "BGGR")
    n = len(mosaics)
    with open(os.path.join(wd, "s_.seq"), "w") as f:
        f.write(f"#Siril sequence file.\nS 's_' 1 {n} {n} 5 0 4 0 0 0\n")
    inject_registration(os.path.join(wd, "s_.seq"), Hs, ref=0)
    st = "rej w 3 3" if rej else "mean none"
    t0 = time.perf_counter()
    run_siril(wd, [f"seqapplyreg s_ -drizzle -scale={ratio} -pixfrac={pixfrac} -kernel={kernel}",
                   f"stack r_s_ {st} -nonorm -32b -out=res"])
    dt = time.perf_counter() - t0
    a, hdr = read_fits(os.path.join(wd, "res.fit"))
    shutil.rmtree(wd, ignore_errors=True)
    if a.ndim == 3:
        a = np.transpose(a, (1, 2, 0))
    else:
        a = np.stack([a] * 3, 2)
    if hdr.get("ROWORDER", "BOTTOM-UP").upper().startswith("BOTTOM"):
        a = a[::-1]
    if a.max() > 2.0:
        a = a / 65535.0
    if a.max() >= 0.999:
        print("WARNING clipped", tag)
    print(tag, "max", round(float(a.max()), 3), "zero%", round(float((a == 0).mean() * 100), 2), flush=True)
    return a.astype(np.float32), dt


def H_from(shifts, sx, sy):
    return [np.array([[1, 0, sx * dx], [0, 1, sy * dy], [0, 0, 1]], float) for dx, dy in shifts]


def gain_fit(img, b):
    m = b["mask"]; x = img[..., 1][m].astype(np.float64); y = b["target"][..., 1][m].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], 1); (g, o), *_ = np.linalg.lstsq(A, y, rcond=None)
    return (img * g + o).astype(np.float32), round(float(g), 4)


def metrics(img, b, ratio):
    img, g = gain_fit(img, b)
    return {"gain": g, "psnr_box_db": round(psnr(img, b["target"], b["mask"]), 2),
            "mtf50": round(compute_slanted_edge_mtf50(img, float(ratio)), 4),
            "flat_sigma": round(immerkaer_sigma(img[30:90, 30:90, 1]), 5)}


def main():
    os.makedirs(BASE, exist_ok=True)
    res = {}
    # 1. calibrate Siril H convention on probe seed 3
    pb = synthesize_target_burst(ratio=2, lr=256, n=9, seed=3)
    cal = {}
    for sx in (1, -1):
        for sy in (1, -1):
            img, _ = siril_drizzle(pb["mosaics"], H_from(pb["offsets"], sx, sy), 2, 0.5,
                                   "square", f"probe_{sx}_{sy}", rej=False)
            al, off = align_to(img, pb["target"], pb["mask"])
            al, g = gain_fit(np.where(al == 0, np.nan, al), pb) if False else gain_fit(al, pb)
            cal[f"{sx},{sy}"] = {"psnr": round(psnr(al, pb["target"], pb["mask"]), 2), "offset": off, "gain": g}
    best = max(cal, key=lambda k: cal[k]["psnr"])
    sx, sy = map(int, best.split(","))
    res["calibration"] = {"candidates": cal, "chosen": best}
    print("calibration", json.dumps(res["calibration"]), flush=True)

    for ratio in (2, 3):
        for n in (9, 20):
            b = synthesize_target_burst(ratio=ratio, lr=256, n=n, seed=7)
            mos = b["mosaics"]
            dems = [_demosaic_bggr(m) for m in mos]
            frames = [RawFrame(str(k), d, d[..., 1].copy(), mosaic=m, pattern=BGGR)
                      for k, (d, m) in enumerate(zip(dems, mos))]
            hr = b["target"].shape[:2]
            row = {}
            bic = cv2.resize(dems[0], (hr[1], hr[0]), interpolation=cv2.INTER_CUBIC)
            row["single_bicubic"] = metrics(bic, b, ratio)
            est = None
            for tag, fusion in (("raw_sr_default", "demosaic_drizzle"), ("raw_sr_cfa_kr", "cfa_kr")):
                sr = SuperResolution(ratio=float(ratio), blur_threshold_ratio=0.0,
                                     registration="homography", ecc_scale=0.25, fusion=fusion)
                t0 = time.perf_counter(); r = sr.run(frames, reference_index=0)
                m = metrics(r.hr, b, ratio); m["seconds"] = round(time.perf_counter() - t0, 2)
                row[tag] = m
                if est is None:
                    est = np.array(r.shifts, float)
            tru = b["offsets"]
            if np.linalg.norm(est[1:] + tru[1:], axis=1).mean() < np.linalg.norm(est[1:] - tru[1:], axis=1).mean():
                est = -est
            row["registration_err_lr_px"] = round(float(np.linalg.norm(est[1:] - tru[1:], axis=1).mean()), 4)
            for regname, sh in (("exact", tru), ("rawsr_est", est)):
                for pf, ker in ((0.5, "square"), (0.7, "square"), (1.0, "square"),
                                (0.7, "gaussian"), (1.0, "lanczos3")):
                    tag = f"siril_{regname}_pf{pf}_{ker}"
                    try:
                        img, dt = siril_drizzle(mos, H_from(sh, sx, sy), ratio, pf, ker,
                                                f"r{ratio}_n{n}_{tag}")
                    except Exception as e:
                        row[tag] = {"error": str(e).splitlines()[-3:]}; continue
                    al, off = align_to(img, b["target"], b["mask"])
                    m = metrics(al, b, ratio); m["offset_corrected"] = off; m["seconds"] = round(dt, 2)
                    row[tag] = m
            res[f"{ratio}x_n{n}"] = row
            print(f"== {ratio}x n={n}", json.dumps(row), flush=True)
    json.dump(res, open(os.path.join(BASE, "syn_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
