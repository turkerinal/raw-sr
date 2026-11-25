#!/usr/bin/env python3
"""B1 demo: ghost gate on a synthetic burst whose subject dominates the frame.

The failure this targets: a subject covering a large share of the frame drags
the global homography, so the old gate's per-pixel tail statistic (normalized by
one global MAD) falls below threshold and the ghost is baked in.

Writes into output/:
  ghost_B1_truth.png   ground-truth HR scene
  ghost_B1_before.png  plain fusion (robust_fusion=False)
  ghost_B1_after.png   auto gate (robust_fusion=None)
  ghost_B1_diff.png    |before - after|, 6x amplified (what the gate changed)

and prints the ghost-trail energy before/after.

Usage: venv/bin/python scripts/ghost_gate_demo.py [--side 220] [--n 9]
"""

import argparse
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raw_sr.core.superres import (  # noqa: E402
    RawFrame, SuperResolution, to_uint8_view,
)

RATIO = 2.0


def scene(size, seed):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    for sig, amp in ((9.0, 1.0), (4.0, 0.55), (1.6, 0.30), (0.7, 0.12)):
        n = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32),
                             (0, 0), sig)
        n = (n - n.mean()) / (n.std() + 1e-9)
        img += amp * n
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    return (0.12 + 0.72 * img).astype(np.float32)


def subject_patch(sw, sh, seed):
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.random((sh, sw)).astype(np.float32), (0, 0), 1.6)
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-9)
    p = 0.30 + 0.62 * tex
    p[:2, :] = 0.95; p[-2:, :] = 0.15
    p[:, :2] = 0.95; p[:, -2:] = 0.15
    return p.astype(np.float32)


def burst(n=9, side=220, size=384, jitter=1.6, noise=0.0035, seed=7):
    """Band-limited handheld burst with a textured subject crossing the frame."""
    h = w = size
    lh, lw = h // 2, w // 2
    rng = np.random.default_rng(seed + 501)
    base = scene(size, seed)
    offs = jitter * (rng.random((n, 2)) - 0.5) * 2.0
    sweep = min(max(0.75 * side, 26.0), max(2.0, w * 0.5 - side * 0.75 - 6))
    cxs = np.linspace(w * 0.5 - sweep, w * 0.5 + sweep, n)
    patch = subject_patch(side, side, seed + 9)
    frames = []
    for i in range(n):
        gx, gy = offs[i]
        img = cv2.warpAffine(base, np.array([[1, 0, -gx], [0, 1, -gy]],
                                            np.float32), (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        x0 = int(np.clip(round(cxs[i] - side / 2), 0, w - side))
        y0 = int(np.clip(round(h * 0.5 - side / 2), 0, h - side))
        img = img.copy()
        img[y0:y0 + side, x0:x0 + side] = patch
        lr = cv2.resize(np.clip(img, 0, 1), (lw, lh),
                        interpolation=cv2.INTER_AREA)
        lr = np.clip(lr + rng.normal(0, noise, lr.shape).astype(np.float32), 0, 1)
        rgb = np.stack([lr] * 3, axis=2).astype(np.float32)
        frames.append(RawFrame(name=f"s{i}", rgb=rgb, green=rgb[:, :, 1].copy()))
    truth = base.copy()
    x0 = int(np.clip(round(w * 0.5 - side / 2), 0, w - side))
    y0 = int(np.clip(round(h * 0.5 - side / 2), 0, h - side))
    truth[y0:y0 + side, x0:x0 + side] = patch
    trail = np.zeros((h, w), bool)
    lo = int(w * 0.5 - sweep - side / 2)
    hi = int(w * 0.5 + sweep + side / 2)
    trail[:, max(0, lo):max(0, x0)] = True
    trail[:, x0 + side:min(w, hi)] = True
    return frames, np.stack([truth] * 3, axis=2).astype(np.float32), trail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=220)
    ap.add_argument("--n", type=int, default=9)
    args = ap.parse_args()

    frames, truth, trail = burst(n=args.n, side=args.side)
    n_img = int(round(frames[0].rgb.shape[0] * RATIO))

    expo = 1.6
    def view(x):
        return cv2.cvtColor(to_uint8_view(x, gamma=2.2, exposure=expo),
                            cv2.COLOR_RGB2BGR)

    plain = SuperResolution(ratio=RATIO, use_gpu_scatter=False,
                            robust_fusion=False).run(frames)
    auto = SuperResolution(ratio=RATIO, use_gpu_scatter=False).run(frames)

    # Ghost-trail energy: mean |SR - truth| over the swept band (borders out).
    interior = np.ones_like(trail)
    interior[:16] = interior[-16:] = False
    interior[:, :16] = interior[:, -16:] = False
    sel = trail & interior
    d_plain = np.abs(plain.hr - truth).mean(axis=2)
    d_auto = np.abs(auto.hr - truth).mean(axis=2)
    e_plain, e_auto = float(d_plain[sel].mean()), float(d_auto[sel].mean())
    vis_plain = float((d_plain[sel] > 0.08).mean())
    vis_auto = float((d_auto[sel] > 0.08).mean())

    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "output")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "ghost_B1_truth.png"), view(truth))
    cv2.imwrite(os.path.join(out_dir, "ghost_B1_before.png"), view(plain.hr))
    cv2.imwrite(os.path.join(out_dir, "ghost_B1_after.png"), view(auto.hr))
    diff = np.abs(np.clip(plain.hr, 0, 1) - np.clip(auto.hr, 0, 1)).mean(axis=2)
    cv2.imwrite(os.path.join(out_dir, "ghost_B1_diff.png"),
                cv2.applyColorMap(np.clip(diff * 6.0 * 255, 0, 255
                                          ).astype(np.uint8), cv2.COLORMAP_INFERNO))

    print(f"subject {args.side}px of {n_img}px "
          f"({(args.side / n_img) ** 2:.0%} of the frame), N={args.n}")
    print(f"auto gate : {auto.metrics['robust_active']} "
          f"[{auto.metrics['robust_reason']}]")
    print(f"trail energy  plain {e_plain:.5f} -> auto {e_auto:.5f} "
          f"({e_plain / max(e_auto, 1e-12):.2f}x)")
    print(f"visible-ghost share (>0.08)  plain {vis_plain:.3f} -> "
          f"auto {vis_auto:.3f}")
    print(f"hf_gain_db    plain {plain.metrics['hf_gain_db']:.3f} -> "
          f"auto {auto.metrics['hf_gain_db']:.3f}")
    print(f"ghost_energy  {auto.metrics['ghost_energy']:.5f} "
          f"(std {auto.metrics['interframe_std']:.5f})")
    print("wrote output/ghost_B1_{truth,before,after,diff}.png")


if __name__ == "__main__":
    main()
