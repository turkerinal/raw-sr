"""Unit tests for WP6 PSF-estimated, scene-adaptive classical enhancement."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from raw_sr.core.enhance import (
    DEFAULT_OPTICS_SIGMA_LR,
    EnhanceConfig,
    build_psf_1d,
    compute_scene_adaptive_mask,
    enhance_image,
    estimate_psf_sigma_edges,
    evaluate_enhance_safety,
)


def _make_slanted_edge_rgb(size: int, sigma_lr: float, ratio: float = 2.0) -> np.ndarray:
    """Create a synthetic slanted-edge RGB image blurred by the exact forward PSF."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    theta = np.deg2rad(5.7)
    dist = (xx - size * 0.5) * np.cos(theta) + (yy - size * 0.5) * np.sin(theta)
    edge = (0.15 + 0.65 * np.clip(0.5 + dist, 0.0, 1.0)).astype(np.float32)
    psf_1d = build_psf_1d(ratio=ratio, optics_sigma_lr=sigma_lr)
    blurred = cv2.sepFilter2D(edge, cv2.CV_32F, psf_1d, psf_1d, borderType=cv2.BORDER_REPLICATE)
    return np.stack([blurred, blurred, blurred], axis=2)


@pytest.mark.parametrize("true_sigma", [0.30, 0.45, 0.60, 0.80])
def test_psf_sigma_recovery_on_synthetic_edges(true_sigma: float):
    """Structure-tensor ESF estimator must recover known lens PSF within 15%."""
    img = _make_slanted_edge_rgb(size=256, sigma_lr=true_sigma, ratio=2.0)
    est = estimate_psf_sigma_edges(img, ratio=2.0)
    assert not est.used_fallback
    assert est.n_edges >= 2
    rel_err = abs(est.sigma_lr - true_sigma) / true_sigma
    assert rel_err < 0.15, f"true={true_sigma}, est={est.sigma_lr:.4f}, rel_err={rel_err:.3f}"


def test_psf_fallback_with_no_edges():
    """Flat / pure-noise images with no strong straight edges must fall back cleanly."""
    flat = np.full((128, 128, 3), 0.4, dtype=np.float32)
    est = estimate_psf_sigma_edges(flat, ratio=2.0)
    assert est.used_fallback
    assert est.sigma_lr == pytest.approx(DEFAULT_OPTICS_SIGMA_LR)
    assert est.n_edges == 0


def test_scene_adaptive_mask_protects_flat_and_highlights():
    """Mask must be ~0 in flat regions and near-clipped highlights, and high on midtone edges."""
    H = W = 128
    luma = np.full((H, W), 0.35, dtype=np.float32)
    # Midtone edge in left-center
    luma[32:96, 20:50] = 0.65
    # Near-clipped specular highlight on right
    luma[32:96, 80:110] = 0.98

    mask = compute_scene_adaptive_mask(luma, noise_sigma=0.002, highlight_clip_knee=0.80)
    # Flat corner
    assert float(mask[8:20, 8:20].mean()) < 0.05
    # Specular highlight interior & boundary
    assert float(mask[40:88, 84:106].mean()) < 0.05
    # Midtone edge boundary
    assert float(mask[40:88, 49:52].mean()) > 0.5


def test_safety_check_detects_noise_and_mtf_degradation():
    """Safety check must fail if flat noise increases > 5% or edge MTF50 drops."""
    rng = np.random.default_rng(11)
    base = _make_slanted_edge_rgb(size=256, sigma_lr=0.40, ratio=2.0)[..., 1]
    noisy_base = np.clip(base + rng.normal(0.0, 0.002, base.shape).astype(np.float32), 0.0, 1.0)

    # 1. Excessive flat noise increase -> safety must fail
    too_noisy = np.clip(base + rng.normal(0.0, 0.005, base.shape).astype(np.float32), 0.0, 1.0)
    ok_noise, info_n = evaluate_enhance_safety(noisy_base, too_noisy, ratio=2.0, max_noise_ratio=1.05)
    assert not ok_noise
    assert info_n["safety_noise_ratio"] > 1.05

    # 2. Blurred (MTF50 drop) -> safety must fail
    blurred = cv2.GaussianBlur(noisy_base, (0, 0), 2.0)
    ok_mtf, info_m = evaluate_enhance_safety(noisy_base, blurred, ratio=2.0, max_noise_ratio=1.05)
    assert not ok_mtf
    assert info_m["safety_mtf_enh"] < info_m["safety_mtf_orig"]


def test_adaptive_enhance_end_to_end():
    """Adaptive mode runs cleanly, reports per-channel PSF, and respects safety check."""
    img = _make_slanted_edge_rgb(size=256, sigma_lr=0.45, ratio=2.0)
    res = enhance_image(img, ratio=2.0, preset="balanced", mode="adaptive")
    assert res.image.shape == img.shape
    assert "psf_sigma_g" in res.details
    assert "safety_backoff_factor" in res.details
    assert abs(res.optics_sigma_lr - 0.45) / 0.45 < 0.15
