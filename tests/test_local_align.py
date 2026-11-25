"""Tests for tile-based local alignment and per-tile robustness / lucky selection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from raw_sr.core.cfa_drizzle import cfa_drizzle, channel_map
from raw_sr.core.local_align import (
    estimate_burst_local_alignment,
    estimate_local_displacement,
)
from raw_sr.core.superres import RawFrame, SuperResolution

BGGR = np.array([[2, 3], [1, 0]], dtype=np.int8)


def _make_textured_image(h: int = 256, w: int = 256, seed: int = 42) -> np.ndarray:
    """Multi-scale textured synthetic image (0.1..0.9 float32)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    img = (
        0.5 * cv2.GaussianBlur(base, (0, 0), 1.5)
        + 0.35 * cv2.GaussianBlur(base, (0, 0), 4.0)
        + 0.15 * cv2.GaussianBlur(base, (0, 0), 10.0)
    )
    img = (img - img.min()) / max(float(img.max() - img.min()), 1e-6)
    return (0.15 + 0.70 * img).astype(np.float32)


def test_local_displacement_recovers_known_nonrigid_warp():
    """Field recovers a smooth non-rigid warp to < 0.1 LR px on textured synthetic."""
    h, w = 256, 256
    ref = _make_textured_image(h, w, seed=11)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Smooth parallax-like non-rigid displacement (amplitude ~0.75 LR px)
    true_dx = (0.75 * np.sin(2.0 * np.pi * xx / w) * np.cos(np.pi * yy / h)).astype(np.float32)
    true_dy = (0.60 * np.cos(2.0 * np.pi * yy / h) * np.sin(np.pi * xx / w)).astype(np.float32)

    # Convention: aligned(x, y) = ref(x + true_dx, y + true_dy)
    mov = cv2.remap(
        ref, xx + true_dx, yy + true_dy,
        interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT,
    )

    dx_est, dy_est, info = estimate_local_displacement(
        ref, mov, np.eye(3, dtype=np.float64), tile_size=48, max_shift=2.0,
    )

    # Evaluate on interior (excluding 24 px border)
    m = 24
    err = np.hypot(dx_est[m:-m, m:-m] - true_dx[m:-m, m:-m],
                   dy_est[m:-m, m:-m] - true_dy[m:-m, m:-m])
    mean_err = float(err.mean())
    assert mean_err < 0.10, f"mean local shift error {mean_err:.4f} >= 0.10 LR px"
    assert info["valid_tile_frac"] > 0.85


def test_textureless_tiles_fall_back_to_global():
    """Flat / noise-only regions produce near-zero residual (fall back to global)."""
    h, w = 192, 192
    rng = np.random.default_rng(3)
    ref = np.full((h, w), 0.5, dtype=np.float32) + rng.normal(0.0, 0.002, (h, w)).astype(np.float32)
    mov = np.full((h, w), 0.5, dtype=np.float32) + rng.normal(0.0, 0.002, (h, w)).astype(np.float32)

    dx_est, dy_est, info = estimate_local_displacement(
        ref, mov, np.eye(3, dtype=np.float64), tile_size=48, max_shift=2.0,
    )
    assert info["valid_tile_frac"] == 0.0
    assert float(np.max(np.abs(dx_est))) < 1e-6
    assert float(np.max(np.abs(dy_est))) < 1e-6


def test_rigid_burst_no_regression():
    """Plain rigid synthetic burst must not regress (> -0.1 dB) with local_align=True."""
    rng = np.random.default_rng(7)
    lr = 160
    ratio = 2
    hr_size = lr * ratio
    gt = _make_textured_image(hr_size, hr_size, seed=21)
    gt_rgb = np.stack([gt, gt, gt], axis=2)
    box = cv2.blur(gt_rgb, (ratio, ratio), borderType=cv2.BORDER_REFLECT)

    n = 7
    offs = np.stack([rng.uniform(-2.0, 2.0, n), rng.uniform(-2.0, 2.0, n)], axis=1)
    offs[0] = (0.0, 0.0)
    xs, ys = np.meshgrid(np.arange(lr, dtype=np.float32), np.arange(lr, dtype=np.float32))
    off = 0.5 * (ratio - 1.0)
    frames = []
    for k in range(n):
        mx = (ratio * (xs + np.float32(offs[k, 0])) + off).astype(np.float32)
        my = (ratio * (ys + np.float32(offs[k, 1])) + off).astype(np.float32)
        rgb = cv2.remap(box, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        rgb = np.clip(rgb + rng.normal(0.0, 0.003, rgb.shape).astype(np.float32), 0.0, 1.0)
        frames.append(RawFrame(name=f"f{k}", rgb=rgb, green=rgb[..., 1].copy()))

    res_base = SuperResolution(ratio=ratio, blur_threshold_ratio=0.0, local_align=False).run(frames)
    res_loc = SuperResolution(ratio=ratio, blur_threshold_ratio=0.0, local_align=True).run(frames)

    assert res_loc.metrics["local_align_active"] is True
    # Compare PSNR vs GT (aligned to reference frame)
    ref_i = res_base.reference_index
    mx_gt = (np.arange(hr_size, dtype=np.float32)[None, :] + ratio * np.float32(offs[ref_i, 0]))
    my_gt = (np.arange(hr_size, dtype=np.float32)[:, None] + ratio * np.float32(offs[ref_i, 1]))
    mx_gt, my_gt = np.broadcast_arrays(mx_gt, my_gt)
    gt_ref = cv2.remap(box, mx_gt, my_gt, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    m = 20
    def _psnr(a, b):
        mse = float(np.mean((a[m:-m, m:-m] - b[m:-m, m:-m]) ** 2))
        return 10.0 * np.log10(1.0 / max(mse, 1e-12))

    p_base = _psnr(res_base.hr, gt_ref)
    p_loc = _psnr(res_loc.hr, gt_ref)
    assert p_loc >= p_base - 0.10, f"rigid regression: local {p_loc:.3f} vs base {p_base:.3f} dB"


def test_local_select_and_cfa_drizzle_integration():
    """local_select keeps reference and cfa_drizzle accepts local_shifts/local_weights."""
    h, w = 128, 128
    ref = _make_textured_image(h, w, seed=9)
    frames = []
    mosaics = []
    ch = channel_map(BGGR, (h, w))
    for k in range(5):
        rgb = np.stack([ref, ref, ref], axis=2).copy()
        if k == 4:
            # Moving bright patch in frame 4
            rgb[40:72, 40:72] = 0.95
        mos = np.take_along_axis(rgb, ch[..., None], axis=2)[..., 0]
        frames.append(RawFrame(name=f"f{k}", rgb=rgb, green=rgb[..., 1].copy(),
                               mosaic=mos, pattern=BGGR))
        mosaics.append(mos)

    transforms = [np.eye(3, dtype=np.float64) for _ in range(5)]
    loc = estimate_burst_local_alignment(
        frames, transforms, reference_pos=0, tile_size=32,
        local_robust=True, local_select=0.6,
    )
    # Reference frame keeps weight 1.0 everywhere
    assert float(loc.weights_mov[0].min()) == 1.0
    # Moving patch in frame 4 is strongly suppressed
    assert float(loc.weights_mov[4][48:64, 48:64].mean()) < 0.25

    cfa_res = cfa_drizzle(
        mosaics, transforms, 2.0, BGGR,
        local_shifts=(loc.dx_mov, loc.dy_mov),
        local_weights=loc.weights_mov,
    )
    assert cfa_res.hr.shape == (256, 256, 3)
    assert np.isfinite(cfa_res.hr).all()
