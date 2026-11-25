"""Unit tests for raw_sr.core.enhance (WP2)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from raw_sr.core.enhance import (
    DEFAULT_OPTICS_SIGMA_LR,
    ENHANCE_PRESETS,
    EnhanceConfig,
    build_psf,
    build_psf_1d,
    effective_stack_noise,
    enhance_fair_pair,
    enhance_image,
    estimate_noise_sigma,
    estimate_optics_sigma,
    richardson_lucy_tv,
    starlet_decompose,
    starlet_sharpen,
)


def test_psf_normalisation_and_symmetry():
    """PSF 1D and 2D must be non-negative, symmetric, odd-length, and sum to 1."""
    for ratio in (1.0, 2.0, 3.0, 4.0):
        for sigma in (0.0, 0.25, 0.35, 0.55):
            k1d = build_psf_1d(ratio, sigma)
            assert k1d.ndim == 1
            assert len(k1d) % 2 == 1
            assert np.all(k1d >= 0.0)
            assert np.isclose(k1d.sum(), 1.0, atol=1e-6)
            assert np.allclose(k1d, k1d[::-1], atol=1e-6)

            psf2d = build_psf(ratio, sigma)
            assert psf2d.shape == (len(k1d), len(k1d))
            assert np.isclose(psf2d.sum(), 1.0, atol=1e-6)
            assert np.all(psf2d >= 0.0)


def test_rl_convergence_on_known_blur():
    """RL-TV must reduce MSE against ground truth on a known-PSF blurred image."""
    rng = np.random.default_rng(42)
    # Sharp synthetic pattern (bars + circle)
    H = W = 128
    gt = np.full((H, W), 0.2, dtype=np.float32)
    gt[24:104, 32:60] = 0.8
    gt[40:88, 72:100] = 0.75
    cv2.circle(gt, (64, 64), 18, 0.85, -1)

    psf_1d = build_psf_1d(ratio=2.0, optics_sigma_lr=0.35)
    blurred = cv2.sepFilter2D(gt, cv2.CV_32F, psf_1d, psf_1d, borderType=cv2.BORDER_REPLICATE)
    noisy = np.clip(blurred + rng.normal(0.0, 0.002, gt.shape).astype(np.float32), 0.0, 1.0)

    deconv, iters, _ = richardson_lucy_tv(
        noisy, psf_1d=psf_1d, max_iters=10, tv_weight=0.001, noise_sigma=0.002
    )
    assert iters >= 2
    mse_before = float(np.mean((noisy - gt) ** 2))
    mse_after = float(np.mean((deconv - gt) ** 2))
    assert mse_after < mse_before * 0.65, f"RL did not converge: {mse_before:.6f} -> {mse_after:.6f}"


def test_deringing_bound_on_step_edge():
    """Overshoot at a high-contrast step edge must stay bounded by dering_tolerance."""
    H = W = 128
    lo, hi = 0.15, 0.85
    step = np.full((H, W), lo, dtype=np.float32)
    step[:, W // 2:] = hi
    psf_1d = build_psf_1d(ratio=2.0, optics_sigma_lr=0.35)
    blurred = cv2.sepFilter2D(step, cv2.CV_32F, psf_1d, psf_1d, borderType=cv2.BORDER_REPLICATE)

    tol = 0.04
    deconv, _, _ = richardson_lucy_tv(
        blurred,
        psf_1d=psf_1d,
        max_iters=15,
        tv_weight=0.001,
        noise_sigma=None,
        dering_tolerance=tol,
        dering_radius=3,
    )
    step_height = hi - lo
    overshoot = max(float(deconv.max()) - hi, lo - float(deconv.min()), 0.0)
    assert overshoot <= (tol + 1e-3) * step_height, (
        f"Overshoot {overshoot / step_height:.4f} exceeded tolerance {tol}"
    )


def test_starlet_perfect_reconstruction_at_gain_one():
    """Starlet decomposition and sharpening at gain=1 must reconstruct input exactly."""
    rng = np.random.default_rng(7)
    img = rng.uniform(0.1, 0.9, (96, 96)).astype(np.float32)
    details, residual = starlet_decompose(img, n_scales=3)
    recon = residual + sum(details)
    assert np.allclose(recon, img, atol=1e-5)

    out = starlet_sharpen(img, gains=(1.0, 1.0, 1.0), k_sigma=3.0)
    assert np.allclose(out, img, atol=1e-5)


def test_starlet_noise_threshold_suppresses_flat_noise():
    """High k_sigma threshold must leave pure flat noise nearly untouched."""
    rng = np.random.default_rng(123)
    sigma = 0.005
    flat_noisy = np.clip(0.5 + rng.normal(0.0, sigma, (128, 128)).astype(np.float32), 0.0, 1.0)

    out_gated = starlet_sharpen(
        flat_noisy, gains=(2.0, 2.0, 1.5), k_sigma=4.0, noise_sigma=sigma
    )
    out_ungated = starlet_sharpen(
        flat_noisy, gains=(2.0, 2.0, 1.5), k_sigma=0.0, noise_sigma=sigma
    )
    std_in = float(flat_noisy.std())
    std_gated = float(out_gated.std())
    std_ungated = float(out_ungated.std())

    assert std_gated < std_in * 1.05, f"Gated noise grew too much: {std_in:.5f} -> {std_gated:.5f}"
    assert std_ungated > std_gated * 1.3, "Ungated sharpening should amplify noise more than gated"


def test_estimate_optics_sigma_and_fallback():
    """Optics sigma estimator recovers reasonable blur and falls back on flat input."""
    flat = np.full((64, 64, 3), 0.5, dtype=np.float32)
    assert estimate_optics_sigma(flat, ratio=2.0) == DEFAULT_OPTICS_SIGMA_LR

    # Slanted edge blurred by known PSF
    H = W = 256
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    edge = (0.15 + 0.70 * np.clip(0.5 + (xx - 128.0) * 0.995 + (yy - 128.0) * 0.1, 0.0, 1.0)).astype(np.float32)
    psf_1d = build_psf_1d(ratio=2.0, optics_sigma_lr=0.40)
    blurred = cv2.sepFilter2D(edge, cv2.CV_32F, psf_1d, psf_1d, borderType=cv2.BORDER_REPLICATE)
    est = estimate_optics_sigma(blurred, ratio=2.0)
    assert 0.15 <= est <= 0.65


def test_fairness_helper_and_presets():
    """Fairness helper applies identical config to SR and single with own noise."""
    rng = np.random.default_rng(9)
    H = W = 128
    gt = np.full((H, W, 3), 0.3, dtype=np.float32)
    gt[32:96, 32:96] = 0.7
    psf_1d = build_psf_1d(ratio=2.0, optics_sigma_lr=0.35)
    base = np.stack([
        cv2.sepFilter2D(gt[..., c], cv2.CV_32F, psf_1d, psf_1d, borderType=cv2.BORDER_REPLICATE)
        for c in range(3)
    ], axis=2)

    sr = np.clip(base + rng.normal(0.0, 0.001, base.shape).astype(np.float32), 0.0, 1.0)
    single = np.clip(base + rng.normal(0.0, 0.006, base.shape).astype(np.float32), 0.0, 1.0)

    res_sr, res_single = enhance_fair_pair(sr, single, ratio=2.0, preset="balanced")
    assert res_sr.strength == ENHANCE_PRESETS["balanced"]
    assert res_single.strength == ENHANCE_PRESETS["balanced"]
    assert res_sr.noise_sigma < res_single.noise_sigma
    assert res_sr.image.shape == sr.shape
    assert res_single.image.shape == single.shape

    # Strength 0 / preset "off" is a no-op
    res_off = enhance_image(sr, ratio=2.0, preset="off")
    assert np.allclose(res_off.image, sr, atol=1e-6)
    assert res_off.rl_iterations_run == 0


def test_effective_stack_noise():
    assert effective_stack_noise([0.004, 0.004, 0.004, 0.004]) == pytest.approx(0.002)
    assert effective_stack_noise(None) is None
    assert effective_stack_noise([]) is None
    with pytest.raises(ValueError):
        EnhanceConfig.from_preset("nonexistent")


def test_exporter_enhanced_tiff_and_png(tmp_path):
    """Exporter.export_enhanced writes both 16-bit TIFF and 8-bit PNG without mutating master."""
    from raw_sr.core.export import Exporter

    master = np.full((64, 64, 3), 0.25, dtype=np.float32)
    master[16:48, 16:48] = 0.75
    master_copy = master.copy()

    exp = Exporter()
    tiff_path = str(tmp_path / "out_enhanced.tiff")
    png_path = str(tmp_path / "out_enhanced.png")
    res_t = exp.export_enhanced(master, tiff_path, ratio=2.0, preset="balanced")
    res_p = exp.export_enhanced(master, png_path, ratio=2.0, preset="balanced")

    assert np.array_equal(master, master_copy)
    assert res_t.image.shape == master.shape
    assert res_p.image.shape == master.shape
    assert (tmp_path / "out_enhanced.tiff").stat().st_size > 0
    assert (tmp_path / "out_enhanced.png").stat().st_size > 0

