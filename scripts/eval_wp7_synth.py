"""WP7 synthetic evaluation: 2x & 3x, 9 & 20 frames (GT PSNR, MTF50, reg error, time/RSS)."""

import json
import os
import resource
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from audit_c11_part2 import (
    BGGR,
    _demosaic_bggr,
    compute_slanted_edge_mtf50,
    psnr,
    synthesize_target_burst,
)
from raw_sr.core.reg_refine import corner_displacement_lr_px
from raw_sr.core.superres import RawFrame, SuperResolution


def gain_fit(img: np.ndarray, b: dict) -> np.ndarray:
    m = b["mask"]
    x = img[..., 1][m].astype(np.float64)
    y = b["target"][..., 1][m].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    (g, o), *_ = np.linalg.lstsq(A, y, rcond=None)
    return (img * g + o).astype(np.float32)


def reg_error_vs_gt(transforms, true_offsets, lr_shape):
    """Mean and max corner displacement error (LR px) over non-ref frames.

    Handles sign convention of synthesize_target_burst offsets automatically.
    """
    errs_pos = []
    errs_neg = []
    for k in range(1, len(transforms)):
        dx, dy = float(true_offsets[k][0]), float(true_offsets[k][1])
        W_pos = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        W_neg = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        errs_pos.append(corner_displacement_lr_px(transforms[k], W_pos, lr_shape))
        errs_neg.append(corner_displacement_lr_px(transforms[k], W_neg, lr_shape))
    m_pos = float(np.mean([e[0] for e in errs_pos]))
    m_neg = float(np.mean([e[0] for e in errs_neg]))
    chosen = errs_pos if m_pos <= m_neg else errs_neg
    mean_err = float(np.mean([e[0] for e in chosen]))
    max_err = float(np.max([e[1] for e in chosen]))
    return mean_err, max_err


def main():
    out_path = "/tmp/rawsr/wp7/synth_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = {}

    for ratio in (2, 3):
        for n in (9, 20):
            key = f"{ratio}x_n{n}"
            b = synthesize_target_burst(ratio=ratio, lr=256, n=n, seed=7)
            mos = b["mosaics"]
            dems = [_demosaic_bggr(m) for m in mos]
            frames = [
                RawFrame(str(k), d, d[..., 1].copy(), mosaic=m, pattern=BGGR)
                for k, (d, m) in enumerate(zip(dems, mos))
            ]
            lr_shape = mos[0].shape[:2]
            row = {}
            for fusion in ("demosaic_drizzle", "cfa_kr"):
                for rr in (0, 2):
                    tag = f"{fusion}_rr{rr}"
                    t0 = time.perf_counter()
                    sr = SuperResolution(
                        ratio=float(ratio),
                        blur_threshold_ratio=0.0,
                        registration="homography",
                        ecc_scale=0.25,
                        fusion=fusion,
                        reg_refine=rr,
                        use_gpu=False,
                        use_opencl=False,
                        use_gpu_scatter=False,
                    )
                    res = sr.run(frames, reference_index=0)
                    dt = time.perf_counter() - t0
                    hr_fit = gain_fit(res.hr, b)
                    p_db = float(psnr(hr_fit, b["target"], b["mask"]))
                    mtf = float(compute_slanted_edge_mtf50(hr_fit, float(ratio)))
                    mean_e, max_e = reg_error_vs_gt(res.transforms, b["offsets"], lr_shape)
                    rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                    row[tag] = {
                        "psnr_db": round(p_db, 3),
                        "mtf50": round(mtf, 4),
                        "reg_err_mean_lr_px": round(mean_e, 4),
                        "reg_err_max_lr_px": round(max_e, 4),
                        "seconds": round(dt, 2),
                        "reg_refine_seconds": res.metrics.get("reg_refine_seconds", 0.0),
                        "reg_refine_iters": res.metrics.get("reg_refine_iters", 0),
                        "rss_mib": round(rss_mib, 1),
                    }
            results[key] = row
            print(f"=== {key} ===", flush=True)
            for tag, m in row.items():
                print(
                    f"  {tag:22s}: PSNR={m['psnr_db']:6.3f} dB  MTF50={m['mtf50']:.4f}  "
                    f"reg_err(mean/max)={m['reg_err_mean_lr_px']:.4f}/{m['reg_err_max_lr_px']:.4f} px  "
                    f"t={m['seconds']:.2f}s (rr={m['reg_refine_seconds']:.2f}s, it={m['reg_refine_iters']})",
                    flush=True,
                )

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
