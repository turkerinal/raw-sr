#!/usr/bin/env python3
"""C11 Audit — Part 1: Audit published images & generate honest 4-panel comparisons.

1. Inspects the published comparison PNGs (dimensions, panel alignment offsets,
   highlight clipping, RMS differences).
2. Runs current raw_sr at 4x on:
   - own 9-frame burst (ROI 1650,1240 size 1024)
   - HDR+ burst 0919_20150910_131720_137 (centred 1024x1024 even-aligned ROI)
   and writes honest, pixel-aligned, identically tone-mapped 4-panel PNGs under
   docs/audit-c11/:
     (a) Raw CFA mosaic of the reference frame (nearest-neighbour 4x)
     (b) Single demosaiced reference frame (nearest-neighbour 4x)
     (c) Single demosaiced reference frame (bicubic 4x)
     (d) Current raw_sr 4x SR output
"""

from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from raw_sr.core.cfa_drizzle import (
    apply_level_gains,
    calibrate_cfa_levels,
    channel_map,
    load_cfa_frame,
    pattern_array,
)
from raw_sr.core.superres import (
    SuperResolution,
    load_raw_linear,
    to_uint8_view,
)


def measure_panel_shift(img_a: np.ndarray, img_b: np.ndarray) -> tuple[float, float]:
    """Sub-pixel translation (dx, dy) from img_a to img_b via phase correlation."""
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ga * win, gb * win)
    return float(dx), float(dy)


def audit_published_images() -> dict:
    """Measure alignment, clipping, and panel differences in published PNGs."""
    report = {}

    # 1. compare_4x_crop.png (3 panels of 256x256 separated by 4px white bars)
    cmp_path = "output/compare/compare_4x_crop.png"
    if os.path.exists(cmp_path):
        im = cv2.imread(cmp_path)
        h, w = im.shape[:2]
        p_left = im[:, 0:256]
        p_mid = im[:, 260:516]
        p_right = im[:, 520:776]
        dx_lm, dy_lm = measure_panel_shift(p_left, p_mid)
        dx_mr, dy_mr = measure_panel_shift(p_mid, p_right)
        rms_lm = float(np.sqrt(np.mean((p_left.astype(np.float32) - p_mid.astype(np.float32)) ** 2)))
        rms_mr = float(np.sqrt(np.mean((p_mid.astype(np.float32) - p_right.astype(np.float32)) ** 2)))
        report["compare_4x_crop.png"] = {
            "shape_hxw": [int(h), int(w)],
            "panel_size": [256, 256],
            "shift_left_to_mid_hr_px": [round(dx_lm, 2), round(dy_lm, 2)],
            "shift_mid_to_right_hr_px": [round(dx_mr, 2), round(dy_mr, 2)],
            "rms_diff_left_vs_mid_u8": round(rms_lm, 2),
            "rms_diff_mid_vs_right_u8": round(rms_mr, 2),
        }

    # 2. hdrplus_0382_4x_sbs.png and hdrplus_6G7M_4x_sbs.png
    for name in ("hdrplus_0382_4x_sbs.png", "hdrplus_6G7M_4x_sbs.png"):
        p = os.path.join(ROOT, "docs", "images", name)
        if not os.path.exists(p):
            continue
        im = cv2.imread(p)
        h, w = im.shape[:2]
        # Header bar is ~28-32 px; find split column (dark vertical separator near w//2)
        mid = w // 2
        left = im[32:, :mid - 2]
        right = im[32:, mid + 2:mid + 2 + left.shape[1]]
        dx, dy = measure_panel_shift(left, right)
        rms = float(np.sqrt(np.mean((left.astype(np.float32) - right.astype(np.float32)) ** 2)))
        clip_l = float(np.mean(np.all(left >= 254, axis=2)) * 100.0)
        clip_r = float(np.mean(np.all(right >= 254, axis=2)) * 100.0)
        report[name] = {
            "shape_hxw": [int(h), int(w)],
            "shift_left_to_right_hr_px": [round(dx, 2), round(dy, 2)],
            "rms_diff_u8": round(rms, 2),
            "clipped_white_pct_left": round(clip_l, 2),
            "clipped_white_pct_right": round(clip_r, 2),
        }

    return report


def cfa_mosaic_to_rgb(mosaic: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Render a 2D Bayer mosaic as an RGB image where each pixel carries only
    its measured colour channel (other two channels = 0)."""
    ch = channel_map(pattern, mosaic.shape)
    rgb = np.zeros((*mosaic.shape, 3), dtype=np.float32)
    for c in range(3):
        mask = ch == c
        rgb[mask, c] = mosaic[mask]
    return rgb


def choose_exposure(linear_img: np.ndarray, target_p98: float = 0.85) -> float:
    """Choose a single linear exposure multiplier so the 98th percentile maps
    to target_p98 without blowing out highlights."""
    p98 = float(np.percentile(linear_img, 98.0))
    if p98 <= 1e-6:
        return 4.0
    return float(np.clip(target_p98 / p98, 0.5, 8.0))


def pick_high_detail_crop(img: np.ndarray, crop_size: int = 320, margin: int = 128) -> tuple[int, int]:
    """Find a crop (x0, y0) with high gradient energy away from borders."""
    gray = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    h, w = gray.shape
    best_score = -1.0
    best_xy = (w // 2 - crop_size // 2, h // 2 - crop_size // 2)
    step = 64
    for y in range(margin, h - margin - crop_size, step):
        for x in range(margin, w - margin - crop_size, step):
            patch = img[y:y + crop_size, x:x + crop_size]
            # Avoid blown-out regions
            if np.percentile(patch, 95) > 0.95:
                continue
            score = float(mag[y:y + crop_size, x:x + crop_size].mean())
            if score > best_score:
                best_score = score
                best_xy = (x, y)
    return best_xy


def make_labelled_grid(
    panels: list[tuple[str, np.ndarray]],
    out_path: str,
    title: str,
) -> dict:
    """Assemble 4 RGB uint8 panels into a 2x2 grid with clean headers."""
    ph, pw = panels[0][1].shape[:2]
    hdr_h = 28
    title_h = 34
    gap = 6
    W = 2 * pw + gap
    H = title_h + 2 * (hdr_h + ph) + gap
    canvas = np.full((H, W, 3), 24, dtype=np.uint8)

    cv2.putText(
        canvas, title, (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA,
    )

    positions = [
        (0, title_h),
        (pw + gap, title_h),
        (0, title_h + hdr_h + ph + gap),
        (pw + gap, title_h + hdr_h + ph + gap),
    ]
    for (label, img), (x0, y0) in zip(panels, positions):
        cv2.rectangle(canvas, (x0, y0), (x0 + pw, y0 + hdr_h), (40, 40, 40), -1)
        cv2.putText(
            canvas, label, (x0 + 8, y0 + 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA,
        )
        canvas[y0 + hdr_h:y0 + hdr_h + ph, x0:x0 + pw] = img

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    size_kb = os.path.getsize(out_path) / 1024.0
    return {"path": out_path, "shape_hxw": [int(H), int(W)], "size_kb": round(size_kb, 1)}


def run_honest_comparison(
    burst_name: str,
    burst_dir: str,
    roi: tuple[int, int, int, int] | None,
    out_png: str,
    crop_size: int = 384,
    ratio: float = 4.0,
    fixed_crop: tuple[int, int] | None = None,
) -> dict:
    files = sorted(glob.glob(os.path.join(burst_dir, "*.dng")))
    if not files:
        raise RuntimeError(f"No DNGs in {burst_dir}")

    if roi is None:
        import rawpy
        with rawpy.imread(files[0]) as raw:
            H, W = raw.sizes.height, raw.sizes.width
        s = 1024
        rx = ((W - s) // 2) & ~1
        ry = ((H - s) // 2) & ~1
        roi = (rx, ry, s, s)

    frames = [load_raw_linear(f, roi=roi) for f in files]
    sr = SuperResolution(
        ratio=ratio, registration="homography", ecc_scale=0.25,
        memory_budget_bytes=4 * 1024**3,
    )
    res = sr.run(frames)
    ref_i = int(res.reference_index)

    # Load reference raw CFA mosaic & calibrate to LibRaw linear level scale
    cfa_ref = load_cfa_frame(files[ref_i], roi=roi)
    gains, _ = calibrate_cfa_levels(cfa_ref, frames[ref_i].rgb)
    cfa_ref = apply_level_gains(cfa_ref, gains)
    pat = pattern_array(cfa_ref.pattern)
    cfa_rgb_lr = cfa_mosaic_to_rgb(cfa_ref.mosaic, pat)

    hr_h, hr_w = res.hr.shape[:2]
    # All four panels on the exact same HR lattice (reference frame grid)
    # cv2.resize with INTER_NEAREST and INTER_CUBIC uses pixel-centre alignment,
    # matching raw_sr's to_drizzle_matrix pixel-centre HR grid.
    p_cfa_nn = cv2.resize(cfa_rgb_lr, (hr_w, hr_h), interpolation=cv2.INTER_NEAREST)
    p_dem_nn = cv2.resize(frames[ref_i].rgb, (hr_w, hr_h), interpolation=cv2.INTER_NEAREST)
    p_dem_bic = cv2.resize(frames[ref_i].rgb, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)
    p_sr = res.hr

    # Pick high-detail crop on the SR output (or use fixed_crop)
    if fixed_crop is not None:
        x0, y0 = fixed_crop
    else:
        x0, y0 = pick_high_detail_crop(p_sr, crop_size=crop_size, margin=128)
    # Snap crop to multiples of 4 so NN blocks are intact
    x0 = (x0 // 4) * 4
    y0 = (y0 // 4) * 4

    sl = (slice(y0, y0 + crop_size), slice(x0, x0 + crop_size))
    expo = choose_exposure(p_sr[sl], target_p98=0.82)

    # Same display transform for all 4 panels (gamma 2.2, identical exposure)
    # Note: for raw CFA mosaic, each pixel has 2 zero channels so it looks darker
    # at 1x exposure; we apply the EXACT same display transform to be 100% honest.
    v_cfa = to_uint8_view(p_cfa_nn[sl], gamma=2.2, exposure=expo)
    v_nn = to_uint8_view(p_dem_nn[sl], gamma=2.2, exposure=expo)
    v_bic = to_uint8_view(p_dem_bic[sl], gamma=2.2, exposure=expo)
    v_sr = to_uint8_view(p_sr[sl], gamma=2.2, exposure=expo)

    # Verify pixel alignment between bicubic and SR
    dx, dy = measure_panel_shift(
        cv2.cvtColor(v_bic, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(v_sr, cv2.COLOR_RGB2BGR),
    )

    panels = [
        ("(a) Raw CFA mosaic (ref frame, 4x NN)", v_cfa),
        ("(b) Single demosaiced frame (4x NN)", v_nn),
        ("(c) Single demosaiced frame (4x bicubic)", v_bic),
        ("(d) Current raw_sr 4x SR", v_sr),
    ]
    title = f"{burst_name} | 4x | ROI {roi[0]},{roi[1]} {roi[2]}x{roi[3]} | crop ({x0},{y0}) {crop_size}px | exp={expo:.2f}x"
    img_info = make_labelled_grid(panels, out_png, title)

    return {
        "burst": burst_name,
        "frames_total": len(files),
        "frames_used": int(res.used_frames),
        "reference_index": ref_i,
        "roi": list(roi),
        "crop_hr": [int(x0), int(y0), int(crop_size)],
        "exposure": round(expo, 3),
        "bicubic_to_sr_shift_hr_px": [round(dx, 3), round(dy, 3)],
        "rms_bicubic_vs_sr_u8": round(float(np.sqrt(np.mean((v_bic.astype(np.float32) - v_sr.astype(np.float32)) ** 2))), 2),
        "image": img_info,
    }


def main():
    out_dir = os.path.join(ROOT, "docs", "audit-c11")
    os.makedirs(out_dir, exist_ok=True)

    pub = audit_published_images()
    print("=== Published Images Audit ===")
    print(json.dumps(pub, indent=2))

    own9 = run_honest_comparison(
        burst_name="Own 9-frame burst",
        burst_dir="data/own9",
        roi=(1650, 1240, 1024, 1024),
        out_png=os.path.join(out_dir, "own9_4x_honest_4panel.png"),
        crop_size=384,
        ratio=4.0,
        fixed_crop=(464, 2672),
    )
    print("=== Own 9-frame Honest 4-Panel ===")
    print(json.dumps(own9, indent=2))

    hdr0919 = run_honest_comparison(
        burst_name="HDR+ burst 0919",
        burst_dir="data/hdrplus/0919_20150910_131720_137",
        roi=None,
        out_png=os.path.join(out_dir, "hdrplus_0919_4x_honest_4panel.png"),
        crop_size=384,
        ratio=4.0,
    )
    print("=== HDR+ 0919 Honest 4-Panel ===")
    print(json.dumps(hdr0919, indent=2))

    summary = {"published_audit": pub, "honest_comparisons": [own9, hdr0919]}
    with open("/tmp/rawsr/c11_part1.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("Wrote /tmp/rawsr/c11_part1.json")


if __name__ == "__main__":
    main()
