"""Unit tests for WP1: Direct Bayer CFA Anisotropic Kernel Regression (cfa_kr)
and exposure gating / normalisation."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from raw_sr.core.cfa_drizzle import (
    cfa_kernel_regression,
    channel_map,
    compute_structure_tensor_precision,
)
from raw_sr.core.superres import RawFrame, SuperResolution


BGGR = np.array([[2, 3], [1, 0]], dtype=np.int8)
RGGB = np.array([[0, 1], [3, 2]], dtype=np.int8)


def _mosaic_from_rgb(rgb: np.ndarray, pattern=BGGR) -> np.ndarray:
    ch = channel_map(pattern, rgb.shape[:2])
    return np.take_along_axis(rgb, ch[..., None], axis=2)[..., 0].astype(np.float32)


def _demosaic_bggr(mosaic: np.ndarray) -> np.ndarray:
    u16 = np.clip(mosaic * 65535.0, 0, 65535).astype(np.uint16)
    return cv2.cvtColor(u16, cv2.COLOR_BayerRG2RGB_EA).astype(np.float32) / 65535.0


def test_cfa_kr_constant_colour_preserves_rgb_identity():
    """A constant-colour (R=0.8, G=0.5, B=0.2) Bayer mosaic must reconstruct
    the exact same (R, G, B) without R/B swap for both BGGR and RGGB."""
    h, w = 32, 32
    color = np.array([0.8, 0.5, 0.2], dtype=np.float32)
    rgb = np.broadcast_to(color, (h, w, 3)).copy()
    for pat in (BGGR, RGGB):
        mos = _mosaic_from_rgb(rgb, pat)
        aligns = [np.eye(3, dtype=np.float64)] * 3
        res = cfa_kernel_regression(
            [mos, mos, mos], aligns, ratio=2.0, pattern=pat, ref_green=rgb[..., 1]
        )
        assert res.hr.shape == (h * 2, w * 2, 3)
        center = res.hr[8:-8, 8:-8]
        np.testing.assert_allclose(center[..., 0], 0.8, atol=1e-3)
        np.testing.assert_allclose(center[..., 1], 0.5, atol=1e-3)
        np.testing.assert_allclose(center[..., 2], 0.2, atol=1e-3)


def test_cfa_kr_shifted_synthetic_burst_sign_and_phase():
    """Shifted synthetic burst: ref->mov transform T(-offs[k]) must align
    all frames cleanly and beat single-frame bicubic on HR GT PSNR."""
    ratio = 2
    lr = 64
    hr_sz = lr * ratio
    yy, xx = np.mgrid[0:hr_sz, 0:hr_sz].astype(np.float32)
    # Smooth high-contrast coloured pattern
    r = 0.5 + 0.3 * np.sin(2 * np.pi * xx / 16.0) * np.cos(2 * np.pi * yy / 20.0)
    g = 0.5 + 0.3 * np.sin(2 * np.pi * (xx + yy) / 18.0)
    b = 0.5 + 0.3 * np.cos(2 * np.pi * yy / 14.0)
    scene = np.stack([r, g, b], axis=2).astype(np.float32)
    box = cv2.blur(scene, (ratio, ratio), borderType=cv2.BORDER_REFLECT)

    rng = np.random.default_rng(42)
    n = 6
    offs = rng.uniform(-1.5, 1.5, size=(n, 2))
    offs[0] = (0.0, 0.0)
    xs, ys = np.meshgrid(np.arange(lr, dtype=np.float32), np.arange(lr, dtype=np.float32))
    off = 0.5 * (ratio - 1.0)
    mosaics = []
    for k in range(n):
        mx = (ratio * (xs + offs[k, 0]) + off).astype(np.float32)
        my = (ratio * (ys + offs[k, 1]) + off).astype(np.float32)
        rgb_k = cv2.remap(box, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mosaics.append(_mosaic_from_rgb(rgb_k, BGGR))

    aligns = [
        np.array([[1.0, 0.0, -offs[k, 0]], [0.0, 1.0, -offs[k, 1]], [0.0, 0.0, 1.0]])
        for k in range(n)
    ]
    ref_dem = _demosaic_bggr(mosaics[0])
    res = cfa_kernel_regression(
        mosaics, aligns, ratio=float(ratio), pattern=BGGR, ref_green=ref_dem[..., 1]
    )
    # Check interior PSNR vs ground truth: beats bicubic AND vastly beats wrong-sign transforms
    m = 12
    mse_kr = float(np.mean((res.hr[m:-m, m:-m] - box[m:-m, m:-m]) ** 2))
    psnr_kr = 10.0 * np.log10(1.0 / max(mse_kr, 1e-12))
    bic = cv2.resize(ref_dem, (hr_sz, hr_sz), interpolation=cv2.INTER_CUBIC)
    mse_bic = float(np.mean((bic[m:-m, m:-m] - box[m:-m, m:-m]) ** 2))
    psnr_bic = 10.0 * np.log10(1.0 / max(mse_bic, 1e-12))
    assert psnr_kr > psnr_bic + 0.5, f"Expected cfa_kr ({psnr_kr:.2f}) > bicubic ({psnr_bic:.2f}) + 0.5 dB"

    # Wrong-sign transforms T(+offs[k]) must degrade PSNR severely
    wrong_aligns = [
        np.array([[1.0, 0.0, +offs[k, 0]], [0.0, 1.0, +offs[k, 1]], [0.0, 0.0, 1.0]])
        for k in range(n)
    ]
    res_wrong = cfa_kernel_regression(
        mosaics, wrong_aligns, ratio=float(ratio), pattern=BGGR, ref_green=ref_dem[..., 1]
    )
    mse_wrong = float(np.mean((res_wrong.hr[m:-m, m:-m] - box[m:-m, m:-m]) ** 2))
    psnr_wrong = 10.0 * np.log10(1.0 / max(mse_wrong, 1e-12))
    assert psnr_kr > psnr_wrong + 5.0, f"Sign check failed: right={psnr_kr:.2f} vs wrong={psnr_wrong:.2f}"


def test_structure_tensor_kernel_shape():
    """Kernel precision must be compact across a vertical edge, elongated along
    it, and isotropic + noise-adaptive in flat regions."""
    h, w = 64, 64
    # Vertical edge at x = w // 2
    img = np.zeros((h, w), dtype=np.float32)
    img[:, w // 2:] = 0.8
    a, b, c, info = compute_structure_tensor_precision(img, noise_sigma=0.001)
    # On the vertical edge (cx = w//4 in half-res), gradient is along x:
    # precision 'a' (across edge, x) must be >> 'c' (along edge, y)
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    assert a[cy, cx] > 3.0 * c[cy, cx], f"Expected a ({a[cy, cx]}) >> c ({c[cy, cx]}) on vertical edge"

    # In flat region (far from edge), kernel should be isotropic: a == c, b ~ 0
    assert abs(float(a[4, 4]) - float(c[4, 4])) < 1e-3
    assert abs(float(b[4, 4])) < 1e-3

    # Higher noise_sigma must widen flat-region kernel (lower precision)
    a_hi, _, _, info_hi = compute_structure_tensor_precision(img, noise_sigma=0.02)
    assert info_hi["k_denoise"] > info["k_denoise"]
    assert a_hi[4, 4] < a[4, 4]


def test_cfa_kr_hole_filling_no_nans_or_zeros():
    """Even with a single frame at 4x (sparse Bayer lattice), cfa_kr must
    produce a finite, hole-free HR image with no zero-weight gaps."""
    h, w = 32, 32
    rgb = np.full((h, w, 3), 0.4, dtype=np.float32)
    mos = _mosaic_from_rgb(rgb, BGGR)
    fb = np.full((h * 4, w * 4, 3), 0.4, dtype=np.float32)
    res = cfa_kernel_regression(
        [mos], [np.eye(3)], ratio=4.0, pattern=BGGR,
        ref_green=rgb[..., 1], fallback_hr=fb,
    )
    assert np.isfinite(res.hr).all()
    assert float(res.hr.min()) > 0.35
    assert float(res.hr.max()) < 0.45


def test_exposure_gating_rejects_outlier_before_reference_selection():
    """A 3.8x brighter outlier frame (like HDR+ N009) must be rejected BEFORE
    reference selection, never picked as reference, and remaining frames fused."""
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.08 + 0.04 * np.sin(2 * np.pi * xx / 10.0) * np.cos(2 * np.pi * yy / 10.0)
    base_rgb = np.stack([base, base, base], axis=2).astype(np.float32)

    frames = []
    for k in range(5):
        # Slight exposure jitter (1.0, 1.05, 0.96, 1.02) within 1.5x
        sc = [1.0, 1.05, 0.96, 1.02, 3.8][k]
        rgb_k = np.clip(base_rgb * sc, 0.0, 1.0)
        mos_k = _mosaic_from_rgb(rgb_k, BGGR)
        frames.append(RawFrame(f"f{k}", rgb_k, rgb_k[..., 1].copy(),
                               mosaic=mos_k, pattern=BGGR))

    sr = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                         registration="translation", fusion="cfa_kr",
                         max_exposure_ratio=1.5)
    res = sr.run(frames)
    assert res.used_frames == 4
    assert res.total_frames == 5
    assert res.reference_index != 4
    assert 4 not in res.kept_indices
    dropped = res.metrics["exposure_gate_dropped"]
    assert len(dropped) == 1 and dropped[0]["index"] == 4
