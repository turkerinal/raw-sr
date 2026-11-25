"""Unit tests for WP13 forward-model MAP SR (per-channel PSF, L-BFGS, noise model, colour)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from raw_sr.core.map_sr import (
    MAPForwardModel,
    _edge_chroma_value_and_grad_cpu,
    _huber_tv_value_and_grad_cpu,
    _hyper_lap_value_and_grad_cpu,
    _nyquist_penalty_value_and_grad_cpu,
    blur_adjoint_rgb_cpu,
    blur_forward_rgb_cpu,
    build_psf_kernels_rgb,
    estimate_burst_psf_rgb,
    estimate_poisson_gaussian_noise,
    map_super_resolution,
)

_SAMPLE_C = np.array(
    [
        [0.7479, 0.2984, 0.0301],
        [0.1357, 0.7828, 0.0696],
        [0.0486, 0.2573, 0.5733],
    ],
    dtype=np.float64,
)


def _make_transforms(n: int, rng: np.random.Generator):
    ts = [np.eye(3, dtype=np.float64)]
    for _ in range(n - 1):
        dx, dy = rng.uniform(-1.2, 1.2, size=2)
        H = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        ts.append(H)
    return ts


@pytest.mark.parametrize("ratio", [2.0, 3.0])
def test_per_channel_psf_adjoint_dot_product(ratio):
    """Exact adjoint test <A x, y> == <x, A^T y> with per-channel RGB PSF + colour matrix C."""
    rng = np.random.default_rng(13)
    lr_shape = (28, 32)
    hr_shape = (int(lr_shape[0] * ratio), int(lr_shape[1] * ratio))
    transforms = _make_transforms(4, rng)
    psf_rgb = (0.56, 0.48, 0.40)

    fm = MAPForwardModel(
        transforms,
        ratio=ratio,
        lr_shape=lr_shape,
        hr_shape=hr_shape,
        pattern="BGGR",
        lens_sigma_lr=psf_rgb,
        colour_matrix=_SAMPLE_C,
    )
    x = rng.random((*hr_shape, 3), dtype=np.float64)
    ys = [rng.random(lr_shape, dtype=np.float64) for _ in transforms]

    lhs = sum(float(np.sum(a * y)) for a, y in zip(fm.forward(x), ys))
    rhs = float(np.sum(x * fm.adjoint(ys)))
    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
    assert rel_err < 1e-12, f"float64 per-channel adjoint mismatch: {rel_err:.3e}"

    x32 = x.astype(np.float32)
    ys32 = [y.astype(np.float32) for y in ys]
    lhs32 = sum(float(np.sum(a.astype(np.float64) * y.astype(np.float64))) for a, y in zip(fm.forward(x32), ys32))
    rhs32 = float(np.sum(x32.astype(np.float64) * fm.adjoint(ys32).astype(np.float64)))
    rel_err32 = abs(lhs32 - rhs32) / max(abs(lhs32), abs(rhs32), 1e-12)
    assert rel_err32 < 1e-6, f"float32 per-channel adjoint mismatch: {rel_err32:.3e}"


@pytest.mark.parametrize("true_sigma", [0.35, 0.55, 0.80])
def test_psf_estimator_synthetic_blurred_edge(true_sigma):
    """PSF estimator recovers known lens sigma on a synthetic blurred slanted edge within 10%."""
    H = W = 256
    ratio = 2.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    theta = np.deg2rad(6.0)
    dist = (xx - W * 0.5) * np.cos(theta) + (yy - H * 0.5) * np.sin(theta)
    edge = (0.15 + 0.65 * np.clip(0.5 + dist, 0.0, 1.0)).astype(np.float32)
    rgb = np.stack([edge, edge, edge], axis=2)

    # Apply exact forward PSF: box(ratio) * Gaussian(true_sigma * ratio)
    k1d_rgb = build_psf_kernels_rgb(ratio, lens_sigma_lr=true_sigma)
    blurred = blur_forward_rgb_cpu(rgb, k1d_rgb)

    sigma_rgb, meta = estimate_burst_psf_rgb(blurred, ratio=ratio, drizzle_bias_lr=0.0)
    assert meta["source"] == "wp13_edges_rgb"
    est_g = sigma_rgb[1]
    rel_err = abs(est_g - true_sigma) / true_sigma
    assert rel_err < 0.10, f"true={true_sigma}, est={est_g:.4f} (err={rel_err*100:.1f}%)"


def test_legacy_estimate_psf_bug_fixed():
    """Legacy map_super_resolution(estimate_psf=True) must not silently fail on PSFEstimate."""
    H = W = 128
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    edge = (0.15 + 0.65 * np.clip(0.5 + (xx - 64.0) * 0.995 + (yy - 64.0) * 0.1, 0.0, 1.0)).astype(np.float32)
    x0 = np.stack([edge, edge, edge], axis=2)
    k = build_psf_kernels_rgb(2.0, lens_sigma_lr=0.5)
    x0_b = blur_forward_rgb_cpu(x0, k)
    ts = [np.eye(3, dtype=np.float64)]
    fm = MAPForwardModel(ts, ratio=2.0, lr_shape=(64, 64), hr_shape=(128, 128), pattern="BGGR", lens_sigma_lr=0.5)
    mos = fm.forward(x0)

    res = map_super_resolution(
        x0_b, mos, ts, 2.0, "BGGR", estimate_psf=True, max_iter=3, use_gpu=False, mode="legacy"
    )
    assert res.metrics["psf_source"] == "wp6_edges"


def test_poisson_gaussian_noise_estimator():
    """estimate_poisson_gaussian_noise recovers realistic shot+read noise sigma."""
    rng = np.random.default_rng(7)
    h, w = 256, 256
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    clean = (0.1 + 0.6 * (xx / w)).astype(np.float32)
    full_well = 2048.0
    read_noise = 1.5
    noisy = (rng.poisson(clean * full_well) + rng.normal(0.0, read_noise, clean.shape)) / full_well
    noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)

    a, b, sig_med = estimate_poisson_gaussian_noise(noisy, "BGGR")
    true_med = float(np.sqrt((read_noise / full_well) ** 2 + (1.0 / full_well) * np.median(clean)))
    assert abs(sig_med - true_med) / true_med < 0.15, f"est={sig_med:.5f}, true={true_med:.5f}"
    assert b > 0.0


def test_prior_gradients_and_nyquist_properties():
    """Verify analytic gradients via finite differences and Nyquist nullspace properties."""
    rng = np.random.default_rng(99)
    Y = rng.uniform(0.2, 0.8, size=(12, 14)).astype(np.float32)
    eps = 1e-3

    # Check Huber-TV, Hyper-Laplacian, Nyquist penalty against central finite differences
    for fn in (
        lambda u: _huber_tv_value_and_grad_cpu(u, eps=1e-2),
        lambda u: _hyper_lap_value_and_grad_cpu(u, eps=1e-2, p=0.8),
        lambda u: _nyquist_penalty_value_and_grad_cpu(u),
    ):
        v0, g = fn(Y)
        for iy, ix in ((3, 4), (6, 7), (0, 0), (11, 13)):
            Yp = Y.copy()
            Ym = Y.copy()
            Yp[iy, ix] += eps
            Ym[iy, ix] -= eps
            fd = (fn(Yp)[0] - fn(Ym)[0]) / (2.0 * eps)
            assert abs(fd - float(g[iy, ix])) < 2e-3, f"grad mismatch at ({iy},{ix}): {fd} vs {g[iy, ix]}"

    # Nyquist penalty must be identically 0 on pure horizontal/vertical edges and > 0 on checkerboard
    h_edge = np.zeros((16, 16, 3), dtype=np.float32)
    h_edge[8:, :, :] = 1.0
    v_edge = np.zeros((16, 16, 3), dtype=np.float32)
    v_edge[:, 8:, :] = 1.0
    assert _nyquist_penalty_value_and_grad_cpu(h_edge)[0] == 0.0
    assert _nyquist_penalty_value_and_grad_cpu(v_edge)[0] == 0.0

    yy, xx = np.mgrid[0:16, 0:16]
    checker = (((yy + xx) % 2).astype(np.float32))[..., None] * np.ones((1, 1, 3), dtype=np.float32)
    assert _nyquist_penalty_value_and_grad_cpu(checker)[0] > 10.0


def test_wp13_solver_convergence_and_colour_roundtrip():
    """WP13 L-BFGS solver converges via tolerance, lowers objective, and preserves colour < 0.5%."""
    rng = np.random.default_rng(21)
    lr_shape = (32, 32)
    hr_shape = (64, 64)
    yy, xx = np.mgrid[0:hr_shape[0], 0:hr_shape[1]].astype(np.float32)
    # Textured multi-colour scene
    x_true = np.stack(
        [
            0.30 + 0.30 * (xx > 32).astype(np.float32),
            0.40 + 0.25 * (yy > 32).astype(np.float32),
            0.25 + 0.35 * ((xx + yy) > 64).astype(np.float32),
        ],
        axis=2,
    )
    ts = _make_transforms(6, rng)
    psf_rgb = (0.45, 0.40, 0.35)
    fm = MAPForwardModel(
        ts, ratio=2.0, lr_shape=lr_shape, hr_shape=hr_shape,
        pattern="BGGR", lens_sigma_lr=psf_rgb, colour_matrix=_SAMPLE_C,
    )
    mosaics = [np.clip(m + rng.normal(0.0, 0.003, m.shape).astype(np.float32), 0.0, 1.0) for m in fm.forward(x_true)]
    x0 = cv2.GaussianBlur(x_true, (5, 5), 1.2)

    res = map_super_resolution(
        x0, mosaics, ts, 2.0, "BGGR",
        colour_matrix=_SAMPLE_C, lens_sigma_lr=psf_rgb,
        mode="wp13", max_iter=60, tol=1e-4, use_gpu=False,
    )
    assert res.metrics["obj_final"] < res.metrics["obj_init"]
    assert res.metrics["stop_reason"] in ("rel_step_tol", "rel_obj_tol", "zero_proj_grad")
    # Mean colour preserved within 0.5%
    m_true = x_true[8:-8, 8:-8].mean(axis=(0, 1))
    m_out = res.hr[8:-8, 8:-8].mean(axis=(0, 1))
    drift_pct = float(np.max(np.abs(m_out - m_true) / m_true) * 100.0)
    assert drift_pct < 0.5, f"Colour drift {drift_pct:.3f}% too high"

    # Also verify CPU vs GPU agreement if GPU available
    from raw_sr.core.gpu import gpu_available
    if gpu_available():
        res_g = map_super_resolution(
            x0, mosaics, ts, 2.0, "BGGR",
            colour_matrix=_SAMPLE_C, lens_sigma_lr=psf_rgb,
            mode="wp13", max_iter=60, tol=1e-4, use_gpu=True,
        )
        assert res_g.metrics["engine"] == "cupy"
        max_diff = float(np.max(np.abs(res.hr - res_g.hr)))
        assert max_diff < 5e-3, f"WP13 CPU vs GPU mismatch: {max_diff:.4e}"
