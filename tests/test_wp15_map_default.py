"""Unit tests for WP15: robust single-frame+drizzle PSF, residual-adaptive
local envelope, PSF-aware CFA held-out, and auto/legacy map_mode routing.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from raw_sr.core.cfa_heldout import evaluate_cfa_heldout, predict_cfa_from_hr
from raw_sr.core.map_sr import (
    MAPForwardModel,
    _compute_local_envelope,
    blur_forward_rgb_cpu,
    build_psf_kernels_rgb,
    estimate_burst_psf_rgb,
    map_super_resolution,
)
from raw_sr.core.superres import RawFrame, SuperResolution


def _make_grid_scene(h: int, w: int) -> np.ndarray:
    """Multi-tile sharp slanted-edge scene with plenty of clean edges."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # Periodic tilted stripes so every 4x4 tile has clean straight edges
    theta = np.deg2rad(8.0)
    u = xx * np.cos(theta) + yy * np.sin(theta)
    period = 48.0
    phase = np.mod(u, period)
    edge = np.where(phase < period * 0.5, 0.20, 0.75)
    # Smooth 1-px transition
    edge = cv2.GaussianBlur(edge.astype(np.float32), (3, 3), 0.35)
    return np.stack([edge, edge, edge], axis=2)


def test_psf_ref_hr_min_rejects_drizzle_smear():
    """When init_hr is smeared by parallax/drizzle, ref_rgb caps the PSF estimate."""
    scene_hr = _make_grid_scene(512, 512)
    # True sharp lens sigma = 0.35 LR px -> ref_rgb at 1x LR
    k_ref = build_psf_kernels_rgb(1.0, lens_sigma_lr=0.35)
    ref_lr = blur_forward_rgb_cpu(scene_hr[::2, ::2], k_ref)
    # Simulate parallax-smeared drizzle init at 2x HR (apparent sigma = 0.85 LR px)
    k_smear = build_psf_kernels_rgb(2.0, lens_sigma_lr=0.85)
    init_hr = blur_forward_rgb_cpu(scene_hr, k_smear)

    sig_hr_only, _ = estimate_burst_psf_rgb(init_hr, ratio=2.0, drizzle_bias_lr=0.10)
    sig_both, meta = estimate_burst_psf_rgb(
        init_hr, ratio=2.0, ref_rgb=ref_lr, drizzle_bias_lr=0.10,
    )
    assert sig_hr_only[1] > 0.70
    assert sig_both[1] < 0.55, f"expected single-frame cap < 0.55, got {sig_both} ({meta})"
    assert meta["source"] in ("wp15_ref_hr_min", "wp15_ref_edges_rgb")


def test_psf_low_edge_fallback():
    """Smooth / edge-starved bursts fall back cleanly to default_sigma_lr."""
    flat_hr = np.full((256, 256, 3), 0.5, dtype=np.float32)
    flat_lr = np.full((128, 128, 3), 0.5, dtype=np.float32)
    sig, meta = estimate_burst_psf_rgb(flat_hr, ratio=2.0, ref_rgb=flat_lr, default_sigma_lr=0.35)
    assert sig == pytest.approx((0.35, 0.35, 0.35))
    assert meta["source"] == "wp15_low_edge_fallback"


def test_residual_adaptive_envelope_targets_mismatch_only():
    """Adaptive envelope constrains mismatched/clipped pixels and leaves clean pixels open."""
    rng = np.random.default_rng(7)
    x0 = np.full((64, 64, 3), 0.35, dtype=np.float32)
    x0[:, 32:, :] = 0.55
    mismatch = np.full((64, 64), 0.9, dtype=np.float32)  # clean static region
    mismatch[20:40, 20:40] = 6.0  # local parallax mismatch

    env = _compute_local_envelope(x0, sigma_med=0.005, env_tol=0.08, env_radius=2, mismatch_hr=mismatch)
    assert env is not None
    x_lo, x_hi = env
    # Clean corner stays unconstrained [0, 1]
    assert float(x_lo[4, 4, 1]) == pytest.approx(0.0)
    assert float(x_hi[4, 4, 1]) == pytest.approx(1.0)
    # Mismatched centre is tightly bounded near x0
    assert float(x_lo[30, 26, 1]) > 0.25
    assert float(x_hi[30, 26, 1]) < 0.65
    # x0 is always strictly feasible
    assert np.all(x_lo <= x0 + 1e-6)
    assert np.all(x0 <= x_hi + 1e-6)

    # Also verify solver runs cleanly on CPU with adaptive envelope enabled
    ts = [np.eye(3, dtype=np.float64)] + [
        np.array([[1.0, 0.0, float(rng.uniform(-0.3, 0.3))],
                  [0.0, 1.0, float(rng.uniform(-0.3, 0.3))],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
        for _ in range(5)
    ]
    fm = MAPForwardModel(ts, ratio=2.0, lr_shape=(32, 32), hr_shape=(64, 64), pattern="BGGR", lens_sigma_lr=0.4)
    mosaics = [np.clip(m + rng.normal(0.0, 0.003, m.shape).astype(np.float32), 0.0, 1.0) for m in fm.forward(x0)]
    res = map_super_resolution(
        x0, mosaics, ts, 2.0, "BGGR",
        lens_sigma_lr=0.4, mode="wp13", max_iter=20, use_gpu=False,
    )
    assert res.hr.shape == (64, 64, 3)
    assert np.all(np.isfinite(res.hr))
    assert res.metrics["env_tol"] == pytest.approx(0.08)


def test_cfa_heldout_optional_lens_sigma_lr():
    """Optional lens_sigma_lr leaves default unchanged and scores sharp HR higher under lens PSF."""
    rng = np.random.default_rng(11)
    hr_shape = (128, 128)
    lr_shape = (64, 64)
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    sharp = np.stack([
        0.2 + 0.5 * (np.sin(xx * 0.9) > 0).astype(np.float32),
        0.25 + 0.5 * (np.sin(yy * 0.9) > 0).astype(np.float32),
        0.2 + 0.45 * (np.sin((xx + yy) * 0.7) > 0).astype(np.float32),
    ], axis=2)

    lens_sig = 0.65
    # Generate lens-blurred held-out mosaic & fused mosaics
    ts = [
        np.array([[1.0, 0.0, float(rng.uniform(-0.4, 0.4))],
                  [0.0, 1.0, float(rng.uniform(-0.4, 0.4))],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
        for _ in range(6)
    ]
    w_held = np.eye(3, dtype=np.float64)
    fm = MAPForwardModel(
        ts + [w_held], ratio=2.0, lr_shape=lr_shape, hr_shape=hr_shape,
        pattern="BGGR", lens_sigma_lr=lens_sig,
    )
    all_mos = fm.forward(sharp)
    mosaics_fused = all_mos[:-1]
    held_mosaic = all_mos[-1]
    weights = [1.0] * len(mosaics_fused)

    # Compare sharp (deconvolved) vs lens-blurred HR
    blurred = cv2.GaussianBlur(sharp, (9, 9), lens_sig * 2.0, borderType=cv2.BORDER_REPLICATE)

    # Default (lens_sigma_lr=None) must match explicit None identically
    res_def = evaluate_cfa_heldout(
        {"sharp": sharp, "blurred": blurred}, held_mosaic, mosaics_fused, ts, weights, w_held, 2.0, "BGGR",
    )
    res_none = evaluate_cfa_heldout(
        {"sharp": sharp, "blurred": blurred}, held_mosaic, mosaics_fused, ts, weights, w_held, 2.0, "BGGR",
        lens_sigma_lr=None,
    )
    assert res_def["config"]["lens_sigma_lr"] is None
    assert res_def["per_method"]["sharp"]["heldout_cfa_gain_db"] == pytest.approx(
        res_none["per_method"]["sharp"]["heldout_cfa_gain_db"], abs=1e-6
    )

    # With per-burst lens_sigma_lr (both scalar and 3-tuple), sharp HR beats blurred HR!
    res_psf_s = evaluate_cfa_heldout(
        {"sharp": sharp, "blurred": blurred}, held_mosaic, mosaics_fused, ts, weights, w_held, 2.0, "BGGR",
        lens_sigma_lr=lens_sig,
    )
    res_psf_rgb = evaluate_cfa_heldout(
        {"sharp": sharp, "blurred": blurred}, held_mosaic, mosaics_fused, ts, weights, w_held, 2.0, "BGGR",
        lens_sigma_lr=(lens_sig, lens_sig, lens_sig),
    )
    assert res_psf_s["config"]["lens_sigma_lr"] == pytest.approx(lens_sig)
    assert res_psf_rgb["config"]["lens_sigma_lr"] == [lens_sig, lens_sig, lens_sig]
    assert res_psf_s["per_method"]["sharp"]["heldout_cfa_gain_db"] > res_psf_s["per_method"]["blurred"]["heldout_cfa_gain_db"] + 1.0
    assert res_psf_rgb["per_method"]["sharp"]["heldout_cfa_gain_db"] == pytest.approx(
        res_psf_s["per_method"]["sharp"]["heldout_cfa_gain_db"], abs=1e-4
    )


def test_auto_cpu_stays_drizzle_and_legacy_selectable():
    """CPU stays on demosaic_drizzle in auto; map_mode='legacy' stays selectable."""
    img = np.full((32, 32, 3), 0.3, dtype=np.float32)
    mos = np.full((32, 32), 0.3, dtype=np.float32)
    frames = [RawFrame(f"f{i}", img, img[..., 1].copy(), mosaic=mos, pattern="BGGR") for i in range(3)]
    sr_cpu = SuperResolution(ratio=2.0, fusion="auto", use_gpu=False)
    assert sr_cpu.map_mode == "wp13"
    fusion, reason = sr_cpu._resolve_fusion(frames)
    assert fusion == "demosaic_drizzle"
    assert "GPU" in reason

    sr_leg = SuperResolution(ratio=2.0, fusion="map", map_mode="legacy", use_gpu=False)
    assert sr_leg.map_mode == "legacy"
