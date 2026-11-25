"""Tests for the optional CuPy/CUDA GPU backend (raw_sr.core.gpu)."""

from __future__ import annotations

import numpy as np
import pytest

from raw_sr.core.gpu import (
    gpu_available,
    gpu_device_info,
    resolve_use_gpu,
    reset_gpu_calibration,
    richardson_lucy_tv_gpu,
    starlet_sharpen_gpu,
    iterative_back_projection_gpu,
    scatter_frames_gpu,
)
from raw_sr.core.enhance import (
    build_psf_1d,
    enhance_image,
    richardson_lucy_tv,
    starlet_sharpen,
)
from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    drizzle,
    iterative_back_projection,
)

requires_gpu = pytest.mark.skipif(
    not gpu_available(), reason="CuPy / CUDA GPU not available"
)


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-16:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _make_synthetic_burst(n: int = 5, h: int = 128, w: int = 128, seed: int = 42):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (
        0.5
        + 0.25 * np.sin(2 * np.pi * xx / 16.0) * np.cos(2 * np.pi * yy / 20.0)
        + 0.15 * ((xx > w * 0.45) & (xx < w * 0.75)).astype(np.float32)
    )
    base = np.clip(base, 0.05, 0.95)
    frames = []
    aligns = []
    for i in range(n):
        dx = 0.0 if i == 0 else float(rng.uniform(-1.5, 1.5))
        dy = 0.0 if i == 0 else float(rng.uniform(-1.5, 1.5))
        rgb = np.stack([base, base * 0.95, base * 0.9], axis=-1).astype(np.float32)
        rgb = np.clip(rgb + rng.normal(0, 0.005, rgb.shape).astype(np.float32), 0.0, 1.0)
        frames.append(RawFrame(name=f"f{i}", rgb=rgb, green=rgb[:, :, 1].copy()))
        aligns.append((dx, dy))
    return frames, aligns


def test_auto_mode_off_when_no_device(monkeypatch):
    """Auto mode must cleanly return False when no GPU device is available."""
    import raw_sr.core.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "gpu_available", lambda: False)
    assert resolve_use_gpu(None, "enhance", 2048 * 2048) is False
    assert resolve_use_gpu(True, "enhance", 2048 * 2048) is False
    assert resolve_use_gpu(False, "enhance", 2048 * 2048) is False


@requires_gpu
def test_gpu_device_info():
    info = gpu_device_info()
    assert info is not None
    assert "name" in info
    assert info["vram_total_mb"] > 0


@requires_gpu
def test_rl_tv_gpu_matches_cpu():
    rng = np.random.default_rng(7)
    luma = rng.uniform(0.05, 0.95, (256, 256)).astype(np.float32)
    psf = build_psf_1d(2.0, 0.35)

    out_cpu, it_c, rsn_c = richardson_lucy_tv(
        luma, psf, max_iters=6, tv_weight=0.003, noise_sigma=0.005, use_gpu=False
    )
    out_gpu, it_g, rsn_g = richardson_lucy_tv(
        luma, psf, max_iters=6, tv_weight=0.003, noise_sigma=0.005, use_gpu=True
    )
    assert it_c == it_g
    assert rsn_c == rsn_g
    assert _psnr(out_cpu, out_gpu) >= 80.0
    assert float(np.max(np.abs(out_cpu - out_gpu))) < 1e-4


@requires_gpu
def test_starlet_gpu_matches_cpu():
    rng = np.random.default_rng(8)
    luma = rng.uniform(0.05, 0.95, (256, 256)).astype(np.float32)
    gains = (1.25, 1.15, 1.08)

    out_cpu = starlet_sharpen(luma, gains=gains, noise_sigma=0.005, use_gpu=False)
    out_gpu = starlet_sharpen(luma, gains=gains, noise_sigma=0.005, use_gpu=True)
    assert _psnr(out_cpu, out_gpu) >= 80.0
    assert float(np.max(np.abs(out_cpu - out_gpu))) < 1e-4


@requires_gpu
def test_enhance_image_gpu_matches_cpu():
    frames, _ = _make_synthetic_burst(n=1, h=256, w=256)
    img = frames[0].rgb
    for mode in ("wp2", "adaptive"):
        r_cpu = enhance_image(img, ratio=2.0, preset="balanced", mode=mode, use_gpu=False)
        r_gpu = enhance_image(img, ratio=2.0, preset="balanced", mode=mode, use_gpu=True)
        assert _psnr(r_cpu.image, r_gpu.image) >= 70.0
        assert r_gpu.details.get("gpu_active") == 1.0
        assert r_cpu.details.get("gpu_active") == 0.0


@requires_gpu
def test_drizzle_scatter_gpu_matches_cpu():
    frames, aligns = _make_synthetic_burst(n=5, h=128, w=128)
    weights = [1.0 / len(frames)] * len(frames)

    hr_cpu, cov_cpu = drizzle(frames, aligns, 2.0, weights, use_cupy=False)
    hr_gpu, cov_gpu = drizzle(frames, aligns, 2.0, weights, use_cupy=True)
    assert _psnr(hr_cpu, hr_gpu) >= 80.0
    assert float(np.max(np.abs(hr_cpu - hr_gpu))) < 1e-4
    assert float(np.max(np.abs(cov_cpu - cov_gpu))) < 1e-4


@requires_gpu
def test_ibp_gpu_matches_cpu():
    frames, aligns = _make_synthetic_burst(n=5, h=128, w=128)
    weights = [1.0 / len(frames)] * len(frames)
    hr0, cov = drizzle(frames, aligns, 2.0, weights, use_cupy=False)

    eng_cpu = {}
    eng_gpu = {}
    ibp_cpu = iterative_back_projection(
        hr0, frames, aligns, 2.0, weights, iterations=1, coverage=cov,
        use_gpu=False, engine_out=eng_cpu,
    )
    ibp_gpu = iterative_back_projection(
        hr0, frames, aligns, 2.0, weights, iterations=1, coverage=cov,
        use_gpu=True, engine_out=eng_gpu,
    )
    assert eng_cpu["ibp"] == "cpu"
    assert eng_gpu["ibp"] == "cupy"
    assert _psnr(ibp_cpu, ibp_gpu) >= 60.0


@requires_gpu
def test_oom_fallback_never_crashes(monkeypatch):
    """When VRAM guard declines or GPU raises OutOfMemoryError, CPU fallback runs."""
    import raw_sr.core.gpu as gpu_mod

    frames, aligns = _make_synthetic_burst(n=3, h=128, w=128)
    weights = [1.0 / 3.0] * 3
    luma = frames[0].green
    psf = build_psf_1d(2.0, 0.35)

    # Simulate insufficient VRAM
    monkeypatch.setattr(gpu_mod, "gpu_fits_vram", lambda req, safety_fraction=0.78: False)
    assert richardson_lucy_tv_gpu(luma, psf) is None
    assert starlet_sharpen_gpu(luma, (1.2, 1.1)) is None
    assert scatter_frames_gpu(frames, aligns, 2.0, weights, 256, 256, 0.5, 2) is None

    # Public APIs must still succeed via CPU fallback
    out_rl, _, _ = richardson_lucy_tv(luma, psf, use_gpu=True)
    assert out_rl.shape == luma.shape
    sr = SuperResolution(ratio=2.0, use_gpu=True)
    res = sr.run(frames)
    assert res.hr.shape == (256, 256, 3)
    assert res.metrics["gpu_stages"] == []
