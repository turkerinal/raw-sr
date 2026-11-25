#!/usr/bin/env python3
"""Generate honest, pixel-aligned 4-panel showcase images (+ 2x zoomed sub-crop)
for raw_sr with per-panel real-edge MTF50, flat-area noise sigma, and JSON metadata.

Panels (left -> right):
  1. Single frame (bicubic) — SR run's own reference_index on exact SR lattice
  2. Single frame + same enhancement (fairness baseline)
  3. raw_sr SR (default fusion)
  4. raw_sr SR + enhancement (RL-TV + starlet, non-neural)

Bottom row: 2x nearest-neighbour zoomed sub-crop of the highest-detail region
inside the main crop so differences are clearly visible on phone screens.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from raw_sr.core.enhance import (
    DEFAULT_STRENGTH,
    effective_stack_noise,
    enhance_fair_pair,
)
from raw_sr.core.superres import (
    SuperResolution,
    load_raw_linear,
)
from scripts.audit_c11_part2 import (
    compute_slanted_edge_mtf50,
    immerkaer_sigma,
)


def measure_phase_shift(a_rgb: np.ndarray, b_rgb: np.ndarray) -> tuple[float, float]:
    """Sub-pixel translation (dx, dy) in HR px from a_rgb to b_rgb via phase correlation."""
    ga = a_rgb[..., 1].astype(np.float32)
    gb = b_rgb[..., 1].astype(np.float32)
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ga * win, gb * win)
    return float(dx), float(dy)


def auto_expose_uint8(img: np.ndarray, exposure: float, gamma: float = 2.2) -> np.ndarray:
    v = np.clip(img.astype(np.float32) * exposure, 0.0, 1.0)
    v = np.power(v, 1.0 / gamma)
    return np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)


def choose_safe_exposure(
    panels: list[np.ndarray],
    target_p995: float = 0.90,
    max_clip_frac: float = 0.009,
) -> tuple[float, float]:
    """Choose a single linear exposure multiplier so max highlight clipping < 1% across all panels."""
    ref = panels[2]  # plain SR
    p995 = float(np.percentile(ref, 99.5))
    expo = float(target_p995 / p995) if p995 > 1e-6 else 1.0
    expo = float(np.clip(expo, 0.25, 12.0))
    for _ in range(12):
        clips = []
        for arr in panels:
            u8 = auto_expose_uint8(arr, exposure=expo)
            clips.append(float(np.mean(np.any(u8 >= 255, axis=2))))
        mc = max(clips)
        if mc <= max_clip_frac:
            return expo, mc
        expo *= 0.90
    return expo, mc


def pick_high_detail_crop(
    img: np.ndarray,
    bic: np.ndarray | None = None,
    crop_size: int = 320,
    margin: int = 96,
    max_shift_hr_px: float = 0.45,
) -> tuple[int, int]:
    """Find a high-texture static interior crop (x0, y0) avoiding borders, blown highlights,
    and parallax/motion (verified by phase-correlation shift < max_shift_hr_px)."""
    gray = img[..., 1].astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    h, w = gray.shape
    candidates: list[tuple[float, int, int]] = []
    step = max(32, crop_size // 8)
    for y in range(margin, max(margin + 1, h - margin - crop_size), step):
        for x in range(margin, max(margin + 1, w - margin - crop_size), step):
            patch = img[y:y + crop_size, x:x + crop_size]
            if float(np.percentile(patch, 95)) > 0.95:
                continue
            score = float(mag[y:y + crop_size, x:x + crop_size].mean())
            candidates.append((score, (x // 4) * 4, (y // 4) * 4))
    candidates.sort(key=lambda t: t[0], reverse=True)
    if bic is not None:
        for _, cx, cy in candidates:
            sl = (slice(cy, cy + crop_size), slice(cx, cx + crop_size))
            dx, dy = measure_phase_shift(bic[sl], img[sl])
            if math.hypot(dx, dy) < max_shift_hr_px:
                return cx, cy
    if candidates:
        return candidates[0][1], candidates[0][2]
    return max(0, (w // 2 - crop_size // 2) // 4 * 4), max(0, (h // 2 - crop_size // 2) // 4 * 4)


def pick_zoom_subcrop(crop_rgb: np.ndarray, sub_size: int) -> tuple[int, int]:
    """Find highest-gradient sub_size x sub_size sub-crop inside crop_rgb."""
    gray = crop_rgb[..., 1].astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    h, w = gray.shape
    best_score = -1.0
    best_xy = ((w - sub_size) // 2, (h - sub_size) // 2)
    step = 16
    for y in range(8, max(9, h - sub_size - 8), step):
        for x in range(8, max(9, w - sub_size - 8), step):
            score = float(mag[y:y + sub_size, x:x + sub_size].mean())
            if score > best_score:
                best_score = score
                best_xy = (x, y)
    return (best_xy[0] // 2) * 2, (best_xy[1] // 2) * 2


def find_best_edge_roi(gray: np.ndarray, win: int = 160) -> tuple[int, int, int, int]:
    """Find a high-contrast vertical-ish edge patch for slanted-edge MTF50."""
    H, W = gray.shape
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    score = np.maximum(gx - 0.5 * gy, 0.0)
    margin = min(win, H // 6, W // 6)
    best = (0.0, W // 2 - win // 2, H // 2 - win // 2)
    step = max(32, win // 4)
    for y in range(margin, max(margin + 1, H - margin - win), step):
        for x in range(margin, max(margin + 1, W - margin - win), step):
            s = float(score[y:y + win, x:x + win].mean())
            if s > best[0]:
                best = (s, x, y)
    return (best[1], best[2], win, win)


def find_flat_patch(gray: np.ndarray, win: int = 96) -> tuple[int, int, int, int]:
    """Find the quietest interior patch (lowest local gradient) for flat noise sigma."""
    H, W = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.hypot(gx, gy)
    margin = min(win, H // 6, W // 6)
    best = (float("inf"), W // 2 - win // 2, H // 2 - win // 2)
    step = max(32, win // 2)
    for y in range(margin, max(margin + 1, H - margin - win), step):
        for x in range(margin, max(margin + 1, W - margin - win), step):
            patch = gray[y:y + win, x:x + win]
            if float(patch.mean()) < 0.02 or float(patch.max()) > 0.95:
                continue
            s = float(gmag[y:y + win, x:x + win].mean())
            if s < best[0]:
                best = (s, x, y)
    return (best[1], best[2], win, win)


def render_showcase(
    panels: list[tuple[str, np.ndarray]],
    sub_xy: tuple[int, int],
    sub_size: int,
    exposure: float,
    title: str,
    out_path: str,
) -> dict:
    """Render 4-panel main crop + 2x nearest-neighbour zoomed sub-crop row."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sx, sy = sub_xy
    ph, pw = panels[0][1].shape[:2]
    hdr_h = 28
    sub_hdr_h = 22
    title_h = 34
    gap = 4

    cols = []
    clip_fracs = []
    for idx, (label, arr) in enumerate(panels):
        u8 = auto_expose_uint8(arr, exposure=exposure)
        clip_fracs.append(float(np.mean(np.any(u8 >= 255, axis=2))))
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)

        sub = bgr[sy:sy + sub_size, sx:sx + sub_size]
        sub_zoom = cv2.resize(sub, (pw, ph), interpolation=cv2.INTER_NEAREST)

        # Draw subtle zoom box on the main crop of panel 1
        bgr_box = bgr.copy()
        if idx == 0:
            cv2.rectangle(bgr_box, (sx, sy), (sx + sub_size, sy + sub_size), (220, 220, 60), 1)

        bar1 = np.full((hdr_h, pw, 3), 24, dtype=np.uint8)
        cv2.putText(
            bar1, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA
        )
        bar2 = np.full((sub_hdr_h, pw, 3), 32, dtype=np.uint8)
        cv2.putText(
            bar2, "2x zoom", (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA
        )
        cols.append(np.vstack([bar1, bgr_box, bar2, sub_zoom]))

    sep = np.full((cols[0].shape[0], gap, 3), 16, dtype=np.uint8)
    body = cols[0]
    for c in cols[1:]:
        body = np.hstack([body, sep, c])

    title_bar = np.full((title_h, body.shape[1], 3), 16, dtype=np.uint8)
    cv2.putText(
        title_bar, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA
    )
    canvas = np.vstack([title_bar, body])

    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    size_kb = os.path.getsize(out_path) / 1024.0
    return {
        "path": out_path,
        "shape_hxw": [int(canvas.shape[0]), int(canvas.shape[1])],
        "size_kb": round(size_kb, 1),
        "exposure": round(exposure, 3),
        "max_clip_pct": round(max(clip_fracs) * 100.0, 3),
    }


def parse_roi(s: str | None) -> tuple[int, int, int, int] | None:
    if not s:
        return None
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected x,y,w,h for --roi, got {s!r}")
    return (parts[0], parts[1], parts[2], parts[3])


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate honest pixel-aligned 4-panel showcase.")
    ap.add_argument("--burst-dir", required=True, help="Directory containing .dng frames")
    ap.add_argument("--name", required=True, help="Scene display name")
    ap.add_argument("--ratio", type=float, default=2.0, help="SR scale ratio (default: 2.0)")
    ap.add_argument("--roi", default=None, help="Optional input ROI x,y,w,h (default: centred 1024x1024)")
    ap.add_argument("--crop-size", type=int, default=320, help="Main crop size in HR px (default: 320)")
    ap.add_argument("--strength", type=float, default=DEFAULT_STRENGTH, help="WP2 enhance strength")
    ap.add_argument("--experimental", action="store_true", help="Mark ratio as experimental in title")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--crop-xy", default=None, help="Fixed main crop x,y in HR px (default: auto)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.burst_dir, "*.dng")))
    if not files:
        raise SystemExit(f"No .dng files found in {args.burst_dir}")

    roi = parse_roi(args.roi)
    if roi is None:
        import rawpy
        with rawpy.imread(files[0]) as raw:
            H, W = raw.sizes.height, raw.sizes.width
        sz = min(1024, H, W)
        rx = ((W - sz) // 2) & ~1
        ry = ((H - sz) // 2) & ~1
        roi = (rx, ry, sz, sz)

    frames = [load_raw_linear(f, roi=roi) for f in files]
    sr = SuperResolution(
        ratio=float(args.ratio),
        registration="homography",
        ecc_scale=0.25,
        max_hr_pixels=20_000_000,
    )
    res = sr.run(frames)  # fusion='auto': MAP at 2x with a CUDA GPU
    ref_i = int(res.reference_index)
    is_map = res.metrics.get("fusion") == "map"
    old = None
    if is_map:
        # Previous default (demosaic_drizzle) on the same reference frame and
        # HR lattice, shown with the WP2 enhancement it was published with.
        old = SuperResolution(
            ratio=float(args.ratio), registration="homography", ecc_scale=0.25,
            max_hr_pixels=20_000_000, fusion="demosaic_drizzle",
        ).run(frames, reference_index=ref_i)
    hr_h, hr_w = res.hr.shape[:2]

    # Single-frame baseline: SR run's OWN reference frame on exact SR lattice
    bic = cv2.resize(frames[ref_i].rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

    sr_noise = effective_stack_noise(
        res.metrics.get("frame_noise_sigma"), effective_frames=res.used_frames
    )
    enh_sr, enh_bic = enhance_fair_pair(
        (old.hr if is_map else res.hr), bic, ratio=float(args.ratio),
        sr_noise_sigma=sr_noise, strength=args.strength,
    )

    # Pick high-texture static crop (verified < 0.45 HR px shift)
    if args.crop_xy:
        x0, y0 = (int(v) for v in args.crop_xy.split(","))
    else:
        x0, y0 = pick_high_detail_crop(res.hr, bic=bic, crop_size=args.crop_size, margin=96)
    sl = (slice(y0, y0 + args.crop_size), slice(x0, x0 + args.crop_size))

    # Verify pixel alignment numerically (< 0.5 HR px). Measured on the whole
    # HR frame: inside a crop of fine periodic texture (e.g. louvres) the
    # single frame's aliasing/moire can mislead phase correlation.
    m = 32
    full = (slice(m, hr_h - m), slice(m, hr_w - m))
    dx, dy = measure_phase_shift(bic[full], res.hr[full])
    offset_norm = math.hypot(dx, dy)
    cdx, cdy = measure_phase_shift(bic[sl], res.hr[sl])
    print(
        f"[{args.name} {args.ratio:g}x] alignment_offset_hr_px=({dx:+.3f}, {dy:+.3f}), "
        f"norm={offset_norm:.3f} HR px (crop-only {math.hypot(cdx, cdy):.3f}; "
        f"ref_idx={ref_i}, used={res.used_frames}/{len(frames)})"
    )
    if offset_norm >= 0.5:
        raise RuntimeError(
            f"Alignment verification failed: offset {offset_norm:.3f} >= 0.5 HR px"
        )
    if is_map:
        odx, ody = measure_phase_shift(old.hr[sl], res.hr[sl])
        print(f"  old default -> MAP crop offset {math.hypot(odx, ody):.3f} HR px")
        if math.hypot(odx, ody) >= 0.5:
            raise RuntimeError("Alignment verification failed: old default vs MAP")

    # Per-panel real-edge MTF50 and flat-area noise sigma
    edge_win = 160 if args.ratio <= 2.5 else 192
    flat_win = 96 if args.ratio <= 2.5 else 128
    edge_roi = find_best_edge_roi(res.hr[..., 1], win=edge_win)
    flat_roi = find_flat_patch(res.hr[..., 1], win=flat_win)
    fx, fy, fw, fh = flat_roi

    if is_map:
        dxo, dyo = measure_phase_shift(old.hr[sl], res.hr[sl])
        print(f"  old-vs-map lattice offset=({dxo:+.3f}, {dyo:+.3f}) HR px")
        if math.hypot(dxo, dyo) >= 0.5:
            raise RuntimeError("old default and MAP are not pixel-aligned")
        panel_imgs = {
            "single_bicubic": bic,
            "single_enhanced": enh_bic.image,
            "old_default_enhanced": enh_sr.image,
            "map_default": res.hr,
        }
    else:
        panel_imgs = {
            "single_bicubic": bic,
            "single_enhanced": enh_bic.image,
            "raw_sr": res.hr,
            "raw_sr_enhanced": enh_sr.image,
        }
    panel_metrics = {}
    for key, img in panel_imgs.items():
        mtf = compute_slanted_edge_mtf50(img, ratio=float(args.ratio), roi=edge_roi)
        flat_im = immerkaer_sigma(img[fy:fy + fh, fx:fx + fw, 1])
        flat_std = float(img[fy:fy + fh, fx:fx + fw, 1].std())
        panel_metrics[key] = {
            "mtf50_cyc_per_lr_px": round(float(mtf), 4),
            "flat_immerkaer_sigma": round(float(flat_im), 6),
            "flat_std_sigma": round(flat_std, 6),
        }
        print(
            f"  {key:16s}: MTF50={mtf:.4f} cyc/LR px | "
            f"flat_immerkaer={flat_im:.6f} | flat_std={flat_std:.6f}"
        )

    # Crop all 4 panels & compute shared safe exposure (< 1% clip)
    crops = [img[sl] for img in panel_imgs.values()]
    expo, max_clip = choose_safe_exposure(crops)
    sub_size = args.crop_size // 2
    sub_xy = pick_zoom_subcrop(res.hr[sl], sub_size=sub_size)

    ratio_tag = f"{args.ratio:g}x" + (" (experimental)" if args.experimental else "")
    title = (
        f"{args.name} | {ratio_tag} | {res.used_frames}/{len(frames)} frames "
        f"| ref #{ref_i} | shift {offset_norm:.2f} HR px | exp {expo:.2f}x"
    )
    labels = (["1. Single frame (bicubic)", "2. Single + enhance",
               "3. Old default + enhance", "4. New default (MAP)"] if is_map else
              ["1. Single frame (bicubic)", "2. Single + same enhance",
               "3. raw_sr SR", "4. raw_sr SR enhanced"])
    labelled = list(zip(labels, crops))
    img_info = render_showcase(
        labelled,
        sub_xy=sub_xy,
        sub_size=sub_size,
        exposure=expo,
        title=title,
        out_path=args.out,
    )
    print(
        f"  saved {args.out} ({img_info['size_kb']} KB, {img_info['shape_hxw'][1]}x{img_info['shape_hxw'][0]}, "
        f"max_clip={img_info['max_clip_pct']}%)"
    )

    meta = {
        "name": args.name,
        "ratio": float(args.ratio),
        "experimental": bool(args.experimental),
        "frames_total": len(frames),
        "frames_used": int(res.used_frames),
        "reference_index": ref_i,
        "input_roi": list(roi),
        "hr_shape_hxw": [int(hr_h), int(hr_w)],
        "crop_xy_size": [int(x0), int(y0), int(args.crop_size)],
        "zoom_subcrop_xy_size": [int(sub_xy[0]), int(sub_xy[1]), int(sub_size)],
        "alignment_offset_hr_px": [round(dx, 4), round(dy, 4)],
        "alignment_offset_norm_hr_px": round(offset_norm, 4),
        "edge_roi": list(edge_roi),
        "flat_roi": list(flat_roi),
        "fusion": res.metrics.get("fusion"),
        "panel_layout": list(panel_imgs.keys()),
        "enhance_strength": float(args.strength),
        "optics_sigma_est_lr_px": round(float(enh_sr.optics_sigma_lr), 4),
        "image": img_info,
        "panels": panel_metrics,
    }
    json_path = os.path.splitext(args.out)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
