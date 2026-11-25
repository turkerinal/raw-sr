"""Unit tests for WP11 MAP colour model (C / C^T, colour round-trip, Gr/Gb equalization)."""

from __future__ import annotations

import numpy as np
import pytest

from raw_sr.core.map_sr import (
    MAPForwardModel,
    equalize_green_imbalance,
    estimate_colour_matrix,
    map_super_resolution,
)
from raw_sr.core.superres import SuperResolution


def _make_transforms(n: int, rng: np.random.Generator):
    ts = [np.eye(3, dtype=np.float64)]
    for _ in range(n - 1):
        dx, dy = rng.uniform(-1.2, 1.2, size=2)
        H = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)
        ts.append(H)
    return ts


_SAMPLE_C = np.array(
    [
        [0.7479, 0.2984, 0.0301],
        [0.1357, 0.7828, 0.0696],
        [0.0486, 0.2573, 0.5733],
    ],
    dtype=np.float64,
)


@pytest.mark.parametrize("ratio", [2.0, 3.0])
def test_adjoint_dot_product_with_colour_matrix(ratio):
    """Verify <A x, y> == <x, A^T y> with non-identity 3x3 colour matrix C."""
    rng = np.random.default_rng(123)
    lr_shape = (28, 32)
    hr_shape = (int(lr_shape[0] * ratio), int(lr_shape[1] * ratio))
    transforms = _make_transforms(4, rng)
    fm = MAPForwardModel(
        transforms, ratio=ratio, lr_shape=lr_shape, hr_shape=hr_shape,
        pattern="BGGR", colour_matrix=_SAMPLE_C,
    )

    x = rng.random((*hr_shape, 3), dtype=np.float64)
    ys = [rng.random(lr_shape, dtype=np.float64) for _ in transforms]
    lhs = sum(float(np.sum(a * y)) for a, y in zip(fm.forward(x), ys))
    rhs = float(np.sum(x * fm.adjoint(ys)))
    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
    assert rel_err < 1e-12, f"float64 adjoint mismatch with C: {rel_err:.3e}"

    x32 = x.astype(np.float32)
    ys32 = [y.astype(np.float32) for y in ys]
    lhs32 = sum(float(np.sum(a.astype(np.float64) * y.astype(np.float64))) for a, y in zip(fm.forward(x32), ys32))
    rhs32 = float(np.sum(x32.astype(np.float64) * fm.adjoint(ys32).astype(np.float64)))
    rel_err32 = abs(lhs32 - rhs32) / max(abs(lhs32), abs(rhs32), 1e-12)
    assert rel_err32 < 1e-6, f"float32 adjoint mismatch with C: {rel_err32:.3e}"


def test_constant_colour_roundtrip():
    """A constant-colour scene stays the exact same colour after MAP with non-identity C."""
    rng = np.random.default_rng(7)
    lr_shape = (32, 32)
    hr_shape = (64, 64)
    const_rgb = np.array([0.65, 0.25, 0.45], dtype=np.float32)
    x_true = np.broadcast_to(const_rgb, (*hr_shape, 3)).copy()
    transforms = _make_transforms(5, rng)

    fm = MAPForwardModel(
        transforms, ratio=2.0, lr_shape=lr_shape, hr_shape=hr_shape,
        pattern="BGGR", colour_matrix=_SAMPLE_C,
    )
    mosaics = fm.forward(x_true)

    # Perturb init slightly and verify MAP recovers the exact constant colour
    x0 = np.clip(x_true + rng.normal(0.0, 0.02, x_true.shape).astype(np.float32), 0.0, 1.0)
    res = map_super_resolution(
        x0, mosaics, transforms, 2.0, "BGGR",
        colour_matrix=_SAMPLE_C, max_iter=20, use_gpu=False,
    )
    interior = res.hr[8:-8, 8:-8]
    mean_rgb = interior.mean(axis=(0, 1))
    assert np.allclose(mean_rgb, const_rgb, atol=2e-3), (
        f"Colour drifted: expected {const_rgb}, got {mean_rgb}"
    )


def test_estimate_colour_matrix_recovery():
    """estimate_colour_matrix recovers exact 3x3 C on a colour-varied unclipped scene."""
    rng = np.random.default_rng(99)
    h, w = 64, 64
    rgb = rng.uniform(0.1, 0.8, size=(h, w, 3)).astype(np.float32)
    cam = np.matmul(rgb, _SAMPLE_C.T.astype(np.float32))
    pat = np.array([[2, 3], [1, 0]], dtype=np.int8)
    mos = np.zeros((h, w), dtype=np.float32)
    mos[0::2, 0::2] = cam[0::2, 0::2, 2]
    mos[0::2, 1::2] = cam[0::2, 1::2, 1]
    mos[1::2, 0::2] = cam[1::2, 0::2, 1]
    mos[1::2, 1::2] = cam[1::2, 1::2, 0]

    C_est, info = estimate_colour_matrix(rgb, mos, pat)
    assert not info["fallback"]
    assert np.allclose(C_est, _SAMPLE_C, atol=1e-4)


def test_equalize_green_imbalance():
    """equalize_green_imbalance removes a 1% Gr/Gb mismatch without mutating input."""
    pat = np.array([[2, 3], [1, 0]], dtype=np.int8)
    mos = np.full((64, 64), 0.5, dtype=np.float32)
    mos[0::2, 1::2] = 0.505  # Gr +1%
    mos[1::2, 0::2] = 0.500  # Gb
    orig = mos.copy()

    eq, ratio = equalize_green_imbalance([mos], pat)
    assert abs(ratio - 1.01) < 1e-4
    assert np.array_equal(mos, orig)  # input not mutated
    diff = abs(float(eq[0][0::2, 1::2].mean()) - float(eq[0][1::2, 0::2].mean()))
    assert diff < 1e-5


def test_default_path_unchanged():
    """Default SuperResolution fusion remains 'demosaic_drizzle'."""
    sr = SuperResolution(ratio=2.0)
    assert sr.fusion == "demosaic_drizzle"
