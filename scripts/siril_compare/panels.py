"""Pixel-aligned 2x panels: single | raw_sr default | raw_sr CFA drizzle | Siril Bayer drizzle."""
import argparse, glob, json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ".")
from real_cmp import siril_hr
from raw_sr.core.superres import SuperResolution, load_raw_linear
from raw_sr.core.cfa_drizzle import cfa_drizzle, pattern_array

OUT = "output/siril/panels"


def colour_fit(src, dst):
    """Global 3x3+offset colour map src->dst (display only, adds no detail)."""
    m = np.zeros(src.shape[:2], bool); m[64:-64, 64:-64] = True
    X = np.concatenate([src[m], np.ones((m.sum(), 1), np.float32)], 1)
    M, *_ = np.linalg.lstsq(X, dst[m], rcond=None)
    return (np.concatenate([src, np.ones(src.shape[:2] + (1,), np.float32)], 2) @ M).astype(np.float32)


def phase_align(img, ref):
    (dx, dy), _ = cv2.phaseCorrelate(ref[..., 1].astype(np.float64), img[..., 1].astype(np.float64))
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), (dx, dy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--burst"); ap.add_argument("--name"); ap.add_argument("--roi", type=int, nargs=2)
    ap.add_argument("--crop", type=int, nargs=2, default=None); ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.burst, "*.dng")))
    frames = [load_raw_linear(f, roi=(a.roi[0], a.roi[1], 1024, 1024)) for f in files]
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(frames)
    ki = list(res.kept_indices); ref = res.reference_index
    pat = pattern_array(frames[0].pattern)
    km = [frames[i].mosaic for i in ki]
    cfa = cfa_drizzle(km, res.transforms, 2.0, pat, res.weights.tolist(), pixfrac=0.5).hr
    sir, _, z = siril_hr(km, res.transforms, ki.index(ref), 2, frames[0].pattern, 0.7, "gaussian", f"{a.name}_panel")
    single = cv2.resize(frames[ref].rgb, (res.hr.shape[1], res.hr.shape[0]), interpolation=cv2.INTER_CUBIC)
    imgs = {"1. Tek kare (bikubik)": single, "2. raw_sr varsayilan": res.hr,
            "3. raw_sr Bayer drizzle": colour_fit(cfa, res.hr), "4. Siril Bayer drizzle": colour_fit(sir, res.hr)}
    offs = {}
    for k in list(imgs)[2:]:
        imgs[k], offs[k] = phase_align(imgs[k], res.hr)
    s = a.size
    if a.crop:
        x0, y0 = a.crop
    else:  # most textured window of the default output, away from borders
        g = cv2.cvtColor(np.clip(res.hr, 0, 1), cv2.COLOR_RGB2GRAY)
        e = cv2.boxFilter(np.abs(cv2.Laplacian(g, cv2.CV_32F)), -1, (s, s))
        e[:s, :] = 0; e[-s:, :] = 0; e[:, :s] = 0; e[:, -s:] = 0
        cy, cx = np.unravel_index(np.argmax(e), e.shape); x0, y0 = cx - s // 2, cy - s // 2
    crops = {k: v[y0:y0 + s, x0:x0 + s] for k, v in imgs.items()}
    gain = 0.85 / max(np.percentile(crops["2. raw_sr varsayilan"], 99.5), 1e-6)
    tiles = []
    for k, c in crops.items():
        u = (np.clip(c * gain, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        u = cv2.resize(u, (2 * s, 2 * s), interpolation=cv2.INTER_NEAREST)
        u = cv2.cvtColor(u, cv2.COLOR_RGB2BGR)
        cv2.rectangle(u, (0, 0), (2 * s, 30), (20, 20, 20), -1)
        cv2.putText(u, k, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(u)
    gap = np.full((2 * s, 6, 3), 30, np.uint8)
    top = np.hstack([tiles[0], gap, tiles[1]]); bot = np.hstack([tiles[2], gap, tiles[3]])
    canvas = np.vstack([top, np.full((6, top.shape[1], 3), 30, np.uint8), bot])
    p = os.path.join(OUT, f"{a.name}_2x.png"); cv2.imwrite(p, canvas)
    print(p, "crop", x0, y0, "frames", len(ki), "ref", ref, "siril_zero", round(z, 4),
          {k: (round(v[0], 3), round(v[1], 3)) for k, v in offs.items()})


if __name__ == "__main__":
    main()
