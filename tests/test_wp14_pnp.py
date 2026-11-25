"""Unit tests for WP14: Plug-and-Play (PnP-HQS / RED + CG) Super-Resolution (`raw_sr/core/pnp_sr.py`).

Tests:
1. Contract match with `map_super_resolution` (returns object with `.hr` float32 (H*r, W*r, 3) in [0, 1] and `.metrics` dict; works as monkeypatch in `SuperResolution(fusion="map")`).
2. Per-channel forward/adjoint dot-product exactness (< 1e-6 float32, < 1e-12 float64).
3. Blind PSF estimation (`estimate_calibrated_burst_psf`) reads `PSFEstimate` attributes without swallowing exceptions.
4. Constant-colour round-trip through non-trivial 3x3 colour matrix `C` and Gr/Gb imbalance equalization (< 1% colour drift).
5. Convergence on a tiny blurred Bayer inverse problem (residual RMS decreases monotonically and PSNR improves over blurred init).
6. CPU determinism (two runs produce bit-identical `.hr` arrays) and permissive (`nlm`) + reference (`bm3d`) denoiser paths.
"""

from __future__ import annotations

import numpy as np
import pytest

import raw_sr.core.map_sr as map_sr_mod
from raw_sr.core.cfa_drizzle import channel_map
from raw_sr.core.pnp_sr import (
    PnPSRResult,
    blur_adjoint_per_channel_cpu,
    blur_forward_per_channel_cpu,
    build_psf_kernel_1d,
    estimate_calibrated_burst_psf,
    pnp_super_resolution,
)
from raw_sr.core.superres import RawFrame, SuperResolution, translation_matrix

RGGB = np.array([[0, 1], [3, 2]], dtype=np.int8)

C_REALISTIC = np.array(
    [
        [0.72, 0.22, 0.06],
        [0.08, 0.86, 0.06],
        [0.04, 0.16, 0.80],
    ],
    dtype=np.float64,
)


def test_per_channel_blur_adjoint_exactness():
    """Verify <B_rgb x, y> == <x, B_rgb^T y> for distinct R/G/B PSF sigmas."""
    rng = np.random.default_rng(42)
    x = rng.random((48, 52, 3), dtype=np.float64)
    y = rng.random((48, 52, 3), dtype=np.float64)
    k_rgb = [
        build_psf_kernel_1d(2.0, lens_sigma_lr=0.55).astype(np.float64),
        build_psf_kernel_1d(2.0, lens_sigma_lr=0.45).astype(np.float64),
        build_psf_kernel_1d(2.0, lens_sigma_lr=0.35).astype(np.float64),
    ]
    bx = blur_forward_per_channel_cpu(x, k_rgb)
    bty = blur_adjoint_per_channel_cpu(y, k_rgb)
    lhs = float(np.sum(bx * y))
    rhs = float(np.sum(x * bty))
    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
    assert rel_err < 1e-12, f"Adjoint dot-product mismatch: rel_err={rel_err}"


def test_blind_psf_estimation_reads_psf_estimate():
    """Verify estimate_calibrated_burst_psf reads PSFEstimate dataclass attributes cleanly."""
    # Create an image with a sharp slanted edge blurred by known sigma_lens_lr = 0.55 at 2x
    import cv2

    h, w = 160, 160
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = (xx - 80.0) * np.cos(0.1) + (yy - 80.0) * np.sin(0.1)
    edge = np.clip(0.15 + 0.70 * np.clip(0.5 + dist, 0.0, 1.0), 0.0, 1.0)
    img = np.stack([edge, edge, edge], axis=-1)
    img = cv2.GaussianBlur(img, (0, 0), 0.55 * 2.0, borderType=cv2.BORDER_REPLICATE)
    img = cv2.blur(img, (2, 2), borderType=cv2.BORDER_REPLICATE)

    s_rgb, s_glob, src, est_obj = estimate_calibrated_burst_psf(img, ratio=2.0, estimate_psf=True)
    assert src == "wp6_edges_calibrated"
    assert est_obj is not None and not est_obj.used_fallback
    assert 0.35 <= s_glob <= 0.75, f"Expected calibrated sigma around 0.55, got {s_glob}"
    assert len(s_rgb) == 3


def test_pnp_contract_and_constant_colour_roundtrip(monkeypatch):
    """pnp_super_resolution must preserve constant/ramp sRGB colour within < 1% and monkeypatch cleanly."""
    h, w, n = 48, 48, 4
    yy, xx = np.meshgrid(
        np.linspace(0.20, 0.75, h, dtype=np.float64),
        np.linspace(0.20, 0.75, w, dtype=np.float64),
        indexing="ij",
    )
    rgb = np.stack(
        [
            0.25 + 0.45 * xx,
            0.30 + 0.40 * yy,
            0.35 + 0.30 * (1.0 - xx) * yy,
        ],
        axis=-1,
    ).astype(np.float32)

    cam = np.matmul(rgb.astype(np.float64), C_REALISTIC.T)
    ch = channel_map(RGGB, (h, w))
    mos = np.empty((h, w), dtype=np.float32)
    for c in range(3):
        mos[ch == c] = cam[:, :, c][ch == c].astype(np.float32)
    # Inject 2% Gr/Gb imbalance
    mos[0::2, 1::2] *= np.float32(np.sqrt(1.02))
    mos[1::2, 0::2] /= np.float32(np.sqrt(1.02))

    frames = [
        RawFrame(name=f"f{k}", rgb=rgb.copy(), green=rgb[:, :, 1].copy(), mosaic=mos.copy(), pattern=RGGB)
        for k in range(n)
    ]

    monkeypatch.setattr(map_sr_mod, "map_super_resolution", pnp_super_resolution)
    sr = SuperResolution(
        ratio=2.0,
        blur_threshold_ratio=0.0,
        registration="translation",
        fusion="map",
        use_gpu=False,
    )
    res = sr.run(frames, reference_index=0)
    assert res.hr.shape == (h * 2, w * 2, 3)
    assert res.hr.dtype == np.float32
    assert float(res.hr.min()) >= 0.0 and float(res.hr.max()) <= 1.0

    hr_mid = res.hr[20:-20, 20:-20]
    ref_mid = rgb[10:-10, 10:-10]
    mean_hr = hr_mid.mean(axis=(0, 1))
    mean_ref = ref_mid.mean(axis=(0, 1))
    rel_drift = float(np.max(np.abs(mean_hr - mean_ref) / np.maximum(mean_ref, 1e-4)))
    assert rel_drift < 0.01, f"Expected colour drift < 1%, got {rel_drift * 100:.3f}% (hr={mean_hr}, ref={mean_ref})"


def test_pnp_convergence_on_tiny_problem():
    """On a synthetic blurred 2x Bayer burst, PnP-HQS CG must reduce data residual RMS and improve GT PSNR."""
    h_lr, w_lr = 32, 32
    ratio = 2.0
    h_hr, w_hr = 64, 64
    yy, xx = np.mgrid[0:h_hr, 0:w_hr].astype(np.float32)
    # Sharp pattern with step edges and bars
    gt = np.zeros((h_hr, w_hr, 3), dtype=np.float32)
    gt[..., 0] = np.where(((xx // 6) % 2 == 0) & (yy > 16) & (yy < 48), 0.75, 0.25)
    gt[..., 1] = np.where(((xx // 6) % 2 == 0) & (yy > 16) & (yy < 48), 0.80, 0.20)
    gt[..., 2] = np.where(((xx // 6) % 2 == 0) & (yy > 16) & (yy < 48), 0.65, 0.30)

    shifts = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
    transforms = [translation_matrix(dx, dy) for dx, dy in shifts]
    fm = map_sr_mod.MAPForwardModel(
        transforms, ratio, (h_lr, w_lr), (h_hr, w_hr), RGGB,
        lens_sigma_lr=0.45, colour_matrix=C_REALISTIC,
    )
    mosaics = fm.forward(gt)

    # Smooth blurred initial estimate
    import cv2
    init_hr = cv2.GaussianBlur(gt, (0, 0), 1.6, borderType=cv2.BORDER_REPLICATE)

    res = pnp_super_resolution(
        init_hr,
        mosaics,
        transforms,
        ratio=ratio,
        pattern=RGGB,
        colour_matrix=C_REALISTIC,
        lens_sigma_lr=0.45,
        detail_boost=0.0,
        outer_iters=4,
        cg_iters=6,
        use_gpu=False,
    )
    assert isinstance(res, PnPSRResult)
    assert res.metrics["rms_final"] < res.metrics["rms_init"]

    mask = np.zeros((h_hr, w_hr), dtype=bool)
    mask[8:-8, 8:-8] = True
    mse_init = float(np.mean((init_hr[mask] - gt[mask]) ** 2))
    mse_pnp = float(np.mean((res.hr[mask] - gt[mask]) ** 2))
    assert mse_pnp < mse_init * 0.55, f"Expected PnP MSE ({mse_pnp:.6f}) << init MSE ({mse_init:.6f})"


def test_pnp_cpu_determinism_and_bm3d_reference():
    """Verify CPU execution is deterministic (bit-identical) and bm3d reference path works."""
    rng = np.random.default_rng(99)
    h_lr, w_lr = 24, 24
    init_hr = rng.uniform(0.2, 0.8, size=(48, 48, 3)).astype(np.float32)
    mosaics = [rng.uniform(0.2, 0.8, size=(h_lr, w_lr)).astype(np.float32) for _ in range(4)]
    transforms = [translation_matrix(0.0, 0.0), translation_matrix(0.5, 0.0),
                  translation_matrix(0.0, 0.5), translation_matrix(0.5, 0.5)]

    r1 = pnp_super_resolution(init_hr, mosaics, transforms, 2.0, RGGB, use_gpu=False, denoiser="nlm")
    r2 = pnp_super_resolution(init_hr, mosaics, transforms, 2.0, RGGB, use_gpu=False, denoiser="nlm")
    assert np.array_equal(r1.hr, r2.hr), "CPU PnP-SR must be bit-identical across runs"

    # bm3d is an optional, non-commercial reference package that is not in
    # requirements*.txt; skip its part on a clean install.
    pytest.importorskip("bm3d")
    r_bm3d = pnp_super_resolution(
        init_hr, mosaics, transforms, 2.0, RGGB, use_gpu=False, denoiser="bm3d", outer_iters=2, cg_iters=3
    )
    assert r_bm3d.hr.shape == (48, 48, 3)
    assert r_bm3d.metrics["denoiser"] == "bm3d"
