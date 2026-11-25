"""Unit tests for WP8 full forward-model MAP super-resolution (raw_sr/core/map_sr.py)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from raw_sr.core.map_sr import (
    MAPForwardModel,
    blur_adjoint_cpu,
    blur_forward_cpu,
    build_psf_kernel_1d,
    map_super_resolution,
)
from raw_sr.core.superres import RawFrame, SuperResolution


def _make_transforms(n: int, rng: np.random.Generator):
    ts = [np.eye(3, dtype=np.float64)]
    for _ in range(n - 1):
        dx, dy = rng.uniform(-1.5, 1.5, size=2)
        ang = rng.uniform(-0.01, 0.01)
        c, s = np.cos(ang), np.sin(ang)
        H = np.array(
            [[c, -s, dx], [s, c, dy], [rng.uniform(-1e-5, 1e-5), rng.uniform(-1e-5, 1e-5), 1.0]],
            dtype=np.float64,
        )
        ts.append(H)
    return ts


@pytest.mark.parametrize("lens_sigma_lr", [None, 0.0, 0.35])
@pytest.mark.parametrize("ratio", [2.0, 3.0])
def test_adjoint_dot_product(ratio, lens_sigma_lr):
    """Verify <A x, y> == <x, A^T y> to < 1e-6 relative error."""
    rng = np.random.default_rng(42)
    lr_shape = (32, 36)
    hr_shape = (int(lr_shape[0] * ratio), int(lr_shape[1] * ratio))
    transforms = _make_transforms(4, rng)
    fm = MAPForwardModel(
        transforms, ratio=ratio, lr_shape=lr_shape, hr_shape=hr_shape,
        pattern="BGGR", lens_sigma_lr=lens_sigma_lr,
    )

    x = rng.random((*hr_shape, 3), dtype=np.float64)
    ys = [rng.random(lr_shape, dtype=np.float64) for _ in transforms]

    Ax = fm.forward(x)
    ATy = fm.adjoint(ys)

    lhs = sum(float(np.sum(a * y)) for a, y in zip(Ax, ys))
    rhs = float(np.sum(x * ATy))
    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
    assert rel_err < 1e-12, f"float64 adjoint mismatch: {rel_err:.3e}"

    # Also check float32
    x32 = x.astype(np.float32)
    ys32 = [y.astype(np.float32) for y in ys]
    Ax32 = fm.forward(x32)
    ATy32 = fm.adjoint(ys32)
    lhs32 = sum(float(np.sum(a.astype(np.float64) * y.astype(np.float64))) for a, y in zip(Ax32, ys32))
    rhs32 = float(np.sum(x32.astype(np.float64) * ATy32.astype(np.float64)))
    rel_err32 = abs(lhs32 - rhs32) / max(abs(lhs32), abs(rhs32), 1e-12)
    assert rel_err32 < 1e-6, f"float32 adjoint mismatch: {rel_err32:.3e}"


def test_blur_adjoint_exact():
    rng = np.random.default_rng(7)
    k = build_psf_kernel_1d(2.0, lens_sigma_lr=0.4).astype(np.float64)
    x = rng.random((24, 28, 3), dtype=np.float64)
    y = rng.random((24, 28, 3), dtype=np.float64)
    lhs = float(np.sum(blur_forward_cpu(x, k) * y))
    rhs = float(np.sum(x * blur_adjoint_cpu(y, k)))
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
    assert rel < 1e-14


def test_synthetic_convergence_and_not_worse_than_init():
    """MAP solver decreases objective and never worsens GT PSNR vs demosaic_drizzle init."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "scripts"))
    from audit_c11_part2 import BGGR, _demosaic_bggr, psnr, synthesize_target_burst

    b = synthesize_target_burst(ratio=2, lr=128, n=9, seed=7)
    mos = b["mosaics"]
    dems = [_demosaic_bggr(m) for m in mos]
    frames = [
        RawFrame(str(k), d, d[..., 1].copy(), mosaic=m, pattern=BGGR)
        for k, (d, m) in enumerate(zip(dems, mos))
    ]

    sr_def = SuperResolution(
        ratio=2.0, blur_threshold_ratio=0.0, registration="homography",
        ecc_scale=0.5, fusion="demosaic_drizzle", use_gpu=False,
    )
    r_def = sr_def.run(frames, reference_index=0)

    sr_map = SuperResolution(
        ratio=2.0, blur_threshold_ratio=0.0, registration="homography",
        ecc_scale=0.5, fusion="map", use_gpu=False,
    )
    r_map = sr_map.run(frames, reference_index=0)

    psnr_def = psnr(r_def.hr, b["target"], b["mask"])
    psnr_map = psnr(r_map.hr, b["target"], b["mask"])
    assert r_map.metrics["cfa_kr_obj_final"] <= r_map.metrics["cfa_kr_obj_init"]
    assert psnr_map >= psnr_def - 1e-3, f"MAP ({psnr_map:.2f} dB) worse than default ({psnr_def:.2f} dB)"


def test_cpu_gpu_consistency():
    """If CuPy GPU is available, GPU and CPU MAP solves agree closely."""
    from raw_sr.core.gpu import gpu_available
    if not gpu_available():
        pytest.skip("CuPy GPU not available")

    rng = np.random.default_rng(11)
    lr_shape = (32, 32)
    hr_shape = (64, 64)
    x0 = rng.uniform(0.1, 0.9, size=(*hr_shape, 3)).astype(np.float32)
    ts = _make_transforms(4, rng)
    fm = MAPForwardModel(ts, ratio=2.0, lr_shape=lr_shape, hr_shape=hr_shape, pattern="BGGR")
    mosaics = fm.forward(x0)

    r_cpu = map_super_resolution(x0, mosaics, ts, 2.0, "BGGR", max_iter=10, use_gpu=False)
    r_gpu = map_super_resolution(x0, mosaics, ts, 2.0, "BGGR", max_iter=10, use_gpu=True)
    assert r_gpu.metrics["engine"] == "cupy"
    diff = float(np.max(np.abs(r_cpu.hr - r_gpu.hr)))
    assert diff < 1e-3, f"CPU vs GPU max diff too large: {diff:.3e}"


def test_default_fusion_unchanged():
    """SuperResolution default fusion remains 'demosaic_drizzle'."""
    sr = SuperResolution(ratio=2.0)
    assert sr.fusion == "demosaic_drizzle"
    with pytest.raises(ValueError):
        SuperResolution(ratio=2.0, fusion="invalid_mode")
