"""Unit tests for WP9 cfa_kr diagnosis and opt-in adaptive improvements."""
from __future__ import annotations

import numpy as np
import pytest

from raw_sr.core.cfa_drizzle import (
    balance_gr_gb,
    cfa_kernel_regression,
    pattern_array,
)


def _make_synthetic_mosaic(h=64, w=64, pattern="BGGR", gr_gb_ratio=1.0, seed=0):
    rng = np.random.default_rng(seed)
    pat = pattern_array(pattern)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 0.2 + 0.5 * (0.5 + 0.5 * np.sin(2.0 * np.pi * xx / 16.0) * np.cos(2.0 * np.pi * yy / 20.0))
    m = base + rng.normal(0.0, 0.005, size=(h, w)).astype(np.float32)
    m = np.clip(m, 0.01, 0.95)
    g_pos = [(r, c) for r in range(2) for c in range(2) if int(pat[r, c]) in (1, 3)]
    r2, c2 = g_pos[1]
    m[r2::2, c2::2] = np.clip(m[r2::2, c2::2] / float(gr_gb_ratio), 0.0, 1.0)
    return m, base


def test_balance_gr_gb_recovers_known_imbalance():
    m, _ = _make_synthetic_mosaic(64, 64, "BGGR", gr_gb_ratio=1.06, seed=1)
    bal, ratio = balance_gr_gb(m, "BGGR")
    assert abs(ratio - 1.06) < 0.01
    # Re-balancing should now report ~1.0
    _, ratio2 = balance_gr_gb(bal, "BGGR")
    assert abs(ratio2 - 1.0) < 0.005


def test_balance_gr_gb_clamps_extreme_ratio():
    m, _ = _make_synthetic_mosaic(32, 32, "RGGB", gr_gb_ratio=1.5, seed=2)
    _, ratio = balance_gr_gb(m, "RGGB", clip_ratio=(0.90, 1.10))
    assert ratio == pytest.approx(1.10, abs=1e-5)


def test_cfa_kr_legacy_default_unchanged():
    m, base = _make_synthetic_mosaic(48, 48, "BGGR", seed=3)
    mosaics = [m, m]
    aligns = [(0.0, 0.0), (0.25, -0.25)]
    r_default = cfa_kernel_regression(mosaics, aligns, 2.0, "BGGR", ref_green=base)
    r_explicit = cfa_kernel_regression(
        mosaics, aligns, 2.0, "BGGR", ref_green=base,
        adaptive=False, gr_gb_balance=False, reg_sigma_lr=None,
    )
    assert np.array_equal(r_default.hr, r_explicit.hr)
    assert r_default.metrics["method"] == "cfa_kr"
    assert r_default.metrics["adaptive"] is False


def test_cfa_kr_adaptive_tighter_bandwidth_and_valid_output():
    m, base = _make_synthetic_mosaic(48, 48, "BGGR", seed=4)
    mosaics = [m, m, m]
    aligns = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)]
    r_old = cfa_kernel_regression(mosaics, aligns, 2.0, "BGGR", ref_green=base, k_detail=0.18)
    r_new = cfa_kernel_regression(
        mosaics, aligns, 2.0, "BGGR", ref_green=base,
        adaptive=True, gr_gb_balance=True, reg_sigma_lr=0.05,
    )
    assert r_new.hr.shape == r_old.hr.shape == (96, 96, 3)
    assert np.all(np.isfinite(r_new.hr))
    assert r_new.metrics["method"] == "cfa_kr_adaptive"
    assert r_new.metrics["adaptive"] is True
    assert r_new.metrics["gr_gb_balance"] is True
    assert r_new.metrics["k_denoise"] < r_old.metrics["k_denoise"]
    assert r_new.metrics["k_detail"] > 0.18  # broadened by reg_sigma_lr=0.05


def test_cfa_kr_adaptive_improves_gr_gb_imbalanced_mosaic():
    m_clean, base = _make_synthetic_mosaic(64, 64, "BGGR", gr_gb_ratio=1.0, seed=5)
    m_imb, _ = _make_synthetic_mosaic(64, 64, "BGGR", gr_gb_ratio=1.08, seed=5)
    aligns = [(0.0, 0.0), (0.5, 0.5)]
    r_ref = cfa_kernel_regression([m_clean, m_clean], aligns, 2.0, "BGGR", ref_green=base, adaptive=True)
    r_no_bal = cfa_kernel_regression([m_imb, m_imb], aligns, 2.0, "BGGR", ref_green=base, adaptive=True, gr_gb_balance=False)
    r_bal = cfa_kernel_regression([m_imb, m_imb], aligns, 2.0, "BGGR", ref_green=base, adaptive=True, gr_gb_balance=True)
    err_no_bal = float(np.mean((r_no_bal.hr[8:-8, 8:-8, 1] - r_ref.hr[8:-8, 8:-8, 1]) ** 2))
    err_bal = float(np.mean((r_bal.hr[8:-8, 8:-8, 1] - r_ref.hr[8:-8, 8:-8, 1]) ** 2))
    assert err_bal < err_no_bal * 0.25
