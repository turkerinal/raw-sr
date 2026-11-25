"""Unit tests for WP7 joint registration refinement (raw_sr/core/reg_refine.py)."""

import numpy as np
import cv2
import pytest

from raw_sr.core.reg_refine import (
    corner_displacement_lr_px,
    refine_burst_registration,
    refine_frame_registration,
    _blur_hr,
    _hr_gradients,
)
from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    predict_lr_from_hr,
)


def _make_textured_hr(h: int = 256, w: int = 256, seed: int = 42) -> np.ndarray:
    """Create a rich, smooth+textured RGB HR image in [0.1, 0.9]."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    base = (
        0.45
        + 0.15 * np.sin(2 * np.pi * (5 * xx + 3 * yy))
        + 0.12 * np.cos(2 * np.pi * (9 * xx - 7 * yy))
        + 0.10 * np.sin(2 * np.pi * (14 * xx + 11 * yy))
    )
    noise = cv2.GaussianBlur(rng.normal(0, 0.08, (h, w)).astype(np.float32), (5, 5), 1.2)
    r = np.clip(base + noise + 0.05 * np.sin(2 * np.pi * 6 * xx), 0.05, 0.95)
    g = np.clip(base + noise, 0.05, 0.95)
    b = np.clip(base + noise - 0.05 * np.cos(2 * np.pi * 6 * yy), 0.05, 0.95)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def _sample_mosaic(rgb: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    mos = np.empty((h, w), dtype=np.float32)
    for r in range(2):
        for c in range(2):
            ci = int(pattern[r, c])
            ch = 1 if ci in (1, 3) else (0 if ci == 0 else 2)
            mos[r::2, c::2] = rgb[r::2, c::2, ch]
    return mos


def test_recovers_known_homography_perturbation_and_exposure():
    """Refinement recovers a known small homography + gain/offset perturbation to < 0.05 LR px."""
    ratio = 2.0
    hr = _make_textured_hr(256, 256, seed=7)
    lr_shape = (128, 128)
    pattern = np.array([[2, 1], [1, 0]], dtype=np.int8)  # BGGR

    # True ref->mov homography with translation, small rotation/scale, and perspective
    W_true = np.array(
        [[1.004, -0.006, 1.35],
         [0.005, 0.997, -0.92],
         [1.2e-5, -1.0e-5, 1.0]],
        dtype=np.float64,
    )
    true_gain = 1.06
    true_offset = 0.015

    ref_rgb = predict_lr_from_hr(hr, np.eye(3), ratio, lr_shape)
    mov_rgb = predict_lr_from_hr(hr, W_true, ratio, lr_shape)
    mov_rgb_exp = np.clip(mov_rgb * true_gain + true_offset, 0.0, 1.0).astype(np.float32)

    frames = [
        RawFrame("ref", ref_rgb, ref_rgb[..., 1].copy(),
                 mosaic=_sample_mosaic(ref_rgb, pattern), pattern=pattern),
        RawFrame("mov", mov_rgb_exp, mov_rgb_exp[..., 1].copy(),
                 mosaic=_sample_mosaic(mov_rgb_exp, pattern), pattern=pattern),
    ]

    # Perturb W_true by ~0.45 LR px at corners
    W_pert = W_true.copy()
    W_pert[0, 2] += 0.35
    W_pert[1, 2] -= 0.28
    W_pert[0, 1] += 0.002
    W_pert[2, 0] -= 8e-6

    mean_before, max_before = corner_displacement_lr_px(W_pert, W_true, lr_shape)
    assert mean_before > 0.25

    res = refine_burst_registration(
        hr, frames, [np.eye(3), W_pert], reference_pos=0, ratio=ratio, use_cfa=True,
    )
    mean_after, max_after = corner_displacement_lr_px(res.transforms[1], W_true, lr_shape)
    assert max_after < 0.05, f"max corner error {max_after:.4f} >= 0.05 LR px"
    assert abs(res.gains[1] - true_gain) < 0.01
    assert abs(res.offsets[1] - true_offset) < 0.005


def test_noop_on_exact_registration():
    """When given exact registration and unit exposure, refinement is a near no-op."""
    ratio = 2.0
    hr = _make_textured_hr(256, 256, seed=11)
    lr_shape = (128, 128)
    pattern = np.array([[0, 1], [1, 2]], dtype=np.int8)  # RGGB
    W_true = np.array([[1.0, 0.0, 0.75], [0.0, 1.0, -0.50], [0.0, 0.0, 1.0]], dtype=np.float64)

    ref_rgb = predict_lr_from_hr(hr, np.eye(3), ratio, lr_shape)
    mov_rgb = predict_lr_from_hr(hr, W_true, ratio, lr_shape)
    frames = [
        RawFrame("ref", ref_rgb, ref_rgb[..., 1].copy(),
                 mosaic=_sample_mosaic(ref_rgb, pattern), pattern=pattern),
        RawFrame("mov", mov_rgb, mov_rgb[..., 1].copy(),
                 mosaic=_sample_mosaic(mov_rgb, pattern), pattern=pattern),
    ]

    res = refine_burst_registration(
        hr, frames, [np.eye(3), W_true], reference_pos=0, ratio=ratio, use_cfa=True,
    )
    _, max_err = corner_displacement_lr_px(res.transforms[1], W_true, lr_shape)
    assert max_err < 0.01
    assert abs(res.gains[1] - 1.0) < 0.005
    assert abs(res.offsets[1]) < 0.002


def test_default_path_bit_identical_when_reg_refine_zero():
    """SuperResolution(reg_refine=0) (default) is bit-identical to explicit default."""
    ratio = 2.0
    hr = _make_textured_hr(128, 128, seed=3)
    lr_shape = (64, 64)
    shifts = [(0.0, 0.0), (0.4, -0.3), (-0.5, 0.6), (0.2, 0.5)]
    frames = []
    for k, (dx, dy) in enumerate(shifts):
        W = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        rgb = predict_lr_from_hr(hr, W, ratio, lr_shape)
        frames.append(RawFrame(f"f{k}", rgb, rgb[..., 1].copy()))

    r_def = SuperResolution(ratio=ratio, blur_threshold_ratio=0.0,
                            use_gpu=False, use_opencl=False, use_gpu_scatter=False).run(frames, reference_index=0)
    r_zero = SuperResolution(ratio=ratio, blur_threshold_ratio=0.0,
                             use_gpu=False, use_opencl=False, use_gpu_scatter=False, reg_refine=0).run(frames, reference_index=0)
    assert np.array_equal(r_def.hr, r_zero.hr)
    assert r_def.metrics["reg_refine_iters"] == 0

    # Also verify reg_refine=2 runs cleanly end-to-end
    r_ref = SuperResolution(ratio=ratio, blur_threshold_ratio=0.0,
                            use_gpu=False, use_opencl=False, use_gpu_scatter=False, reg_refine=2).run(frames, reference_index=0)
    assert r_ref.hr.shape == r_def.hr.shape
    assert r_ref.metrics["reg_refine_iters"] >= 1
