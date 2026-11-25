"""Memory-budget and streaming-deposit tests for the SR core.

These pin the memory work (agent B3/mem3):

1. The CPU scatter deposit streams band tables straight into the run-level
   float32 accumulators instead of materialising a per-frame float64 table plus
   an ``a * weight`` temp — same numbers within the documented f32 tolerance.
2. The nanmedian / deviation helpers used by the auto gate, robust weights and
   ghost metric are exactly equal to the numpy expressions they replace while
   bounding their temporaries.
3. IBP recomputes the per-frame footprint instead of storing an ``(N,oh,ow)``
   list; its output equals the stored-footprint formulation.
4. The HR-size guard raises a clear ``MemoryError`` above its limit and warns
   near it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import raw_sr.core.superres as core  # noqa: E402
from raw_sr.core.superres import (  # noqa: E402
    RawFrame,
    SuperResolution,
    _masked_absdev_max_axis0,
    _masked_absdev_median_value,
    _masked_temporal_median,
    _nanmedian_axis0,
    _scatter_accumulate,
    _scatter_fast,
    _scatter_frames,
    drizzle,
    iterative_back_projection,
    to_drizzle_matrix,
    translation_matrix,
)


def _burst(n=3, h=17, w=19, seed=7):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        rgb = rng.random((h, w, 3), dtype=np.float32)
        frames.append(RawFrame(name=str(i), rgb=rgb, green=rgb[:, :, 1].copy()))
    return frames


# --------------------------------------------------------------------------- #
# 1. Streaming deposit
# --------------------------------------------------------------------------- #

def test_scatter_fast_streams_into_float32_tables():
    rng = np.random.default_rng(3)
    src = rng.random((13, 15, 3), dtype=np.float32)
    M = to_drizzle_matrix(translation_matrix(0.37, -0.21), 2.0)
    ref_a, ref_c = _scatter_fast(src, M, 26, 30, 0.5, 3)
    fw = 0.4
    acc = np.zeros((26, 30, 3), dtype=np.float32)
    cov = np.zeros((26, 30), dtype=np.float32)
    out_a, out_c = _scatter_fast(src, M, 26, 30, 0.5, 3, fweight=fw,
                                 out_acc=acc, out_wt=cov)
    assert out_a is acc and out_c is cov
    assert out_a.dtype == np.float32 and out_c.dtype == np.float32
    assert np.allclose(out_a, ref_a * fw, rtol=1e-5, atol=1e-6)
    assert np.allclose(out_c, ref_c * fw, rtol=1e-5, atol=1e-6)


def test_scatter_frames_cpu_matches_independent_fold():
    frames = _burst(3, 14, 16)
    aligns = [translation_matrix(0.0, 0.0),
              translation_matrix(0.4, -0.3),
              translation_matrix(-0.6, 0.5)]
    weights = [0.5, 0.3, 0.2]
    ratio, oh, ow = 2.0, 28, 32
    acc, cov, engine = _scatter_frames(
        frames, aligns, ratio, weights, oh, ow, 0.5, 3, "bilinear", 0.5,
        False, None, "cpu")
    assert engine == "cpu"
    assert acc.dtype == np.float32 and cov.dtype == np.float32
    ref_a = np.zeros((oh, ow, 3), dtype=np.float64)
    ref_c = np.zeros((oh, ow), dtype=np.float64)
    for fr, al, wt in zip(frames, aligns, weights):
        a, c = _scatter_accumulate(fr.rgb, to_drizzle_matrix(al, ratio),
                                   oh, ow, 0.5, 3)
        ref_a += a * wt
        ref_c += c * wt
    assert np.allclose(acc, ref_a, rtol=1e-5, atol=1e-5)
    assert np.allclose(cov, ref_c, rtol=1e-5, atol=1e-6)


def test_drizzle_scatter_still_matches_manual_reference():
    frames = _burst(2, 10, 12)
    aligns = [translation_matrix(0.0, 0.0), translation_matrix(0.25, -0.5)]
    hr, cov = drizzle(frames, aligns, 2.0, [0.6, 0.4])
    from raw_sr.core.superres import default_deposit_params
    pf, sub = default_deposit_params(2.0)
    acc = np.zeros((20, 24, 3))
    wt = np.zeros((20, 24))
    for fr, al, w in zip(frames, aligns, [0.6, 0.4]):
        a, c = _scatter_accumulate(fr.rgb, to_drizzle_matrix(al, 2.0),
                                   20, 24, pf, sub)
        acc += a * w
        wt += c * w
    expected = np.zeros_like(acc)
    sel = wt >= 1e-6
    expected[sel] = acc[sel] / wt[sel][:, None]
    assert np.abs(hr.astype(np.float64) - expected).max() <= 2e-5
    assert np.abs(cov.astype(np.float64) - wt).max() <= 1e-5


# --------------------------------------------------------------------------- #
# 2. Chunked median / deviation helpers == numpy
# --------------------------------------------------------------------------- #

def _masked_stack(seed=11, n=5, h=23, w=19, nan_frac=0.15):
    rng = np.random.default_rng(seed)
    S = rng.random((n, h, w), dtype=np.float32)
    V = rng.random((n, h, w)) > nan_frac
    return S, V


def test_nanmedian_chunked_equals_numpy():
    S, V = _masked_stack()
    got = _nanmedian_axis0(S, mask=V, target_bytes=2048)  # force many bands
    want = np.nanmedian(np.where(V, S, np.nan), axis=0)
    assert np.array_equal(got, want.astype(np.float32))


def test_absdev_helpers_equal_numpy():
    S, V = _masked_stack(seed=5)
    raw = _nanmedian_axis0(S, mask=V)
    ref = np.nan_to_num(raw, nan=0.0).astype(np.float32)
    dev = np.abs(S - ref[None])
    want_max = np.where(V, dev, 0.0).max(axis=0)
    got_max = _masked_absdev_max_axis0(S, V, ref, target_bytes=2048)
    assert np.array_equal(got_max, want_max.astype(np.float32))
    # the ghost gate's chunked masked median wrapper is the same function
    assert np.array_equal(_masked_temporal_median(S, V, chunk=3), raw,
                          equal_nan=True)
    scale = float(dev.max()) + 1e-6
    want_scaled = np.where(V, dev / scale, 0.0).max(axis=0)
    got_scaled = core._masked_scaled_absdev_max_axis0(
        S, V, ref, scale, target_bytes=2048)
    assert np.array_equal(got_scaled, want_scaled.astype(np.float32))


def test_absdev_median_value_equals_numpy():
    S, V = _masked_stack(seed=9)
    ref = np.nan_to_num(_nanmedian_axis0(S, mask=V), nan=0.0).astype(np.float32)
    devm = np.where(V, np.abs(S - ref[None]), np.nan)
    want = float(np.median(devm[np.isfinite(devm)]))
    got = _masked_absdev_median_value(S, V, ref)
    assert abs(got - want) <= 1e-7


def test_ghost_gate_and_metric_unchanged_on_synthetic_stack():
    """The whole auto-gate/metric path still returns finite, sane numbers."""
    rng = np.random.default_rng(2)
    S = rng.random((4, 40, 44), dtype=np.float32)
    V = np.ones_like(S, dtype=bool)
    moving, tail, compact = core._ghost_gate_dispersion(S, V)
    assert np.isfinite(tail) and 0.0 <= compact <= 1.0
    pop = core._ghost_gate_population(S, V)
    assert pop is not None and np.isfinite(pop["share"])
    spread = core._ghost_gate_spread(S, V)
    assert spread is not None and np.isfinite(spread["ratio"])


# --------------------------------------------------------------------------- #
# 3. IBP footprint recompute == stored-footprint formulation
# --------------------------------------------------------------------------- #

def _ibp_with_stored_footprints(hr, frames, aligns, ratio, weights,
                                iterations=1, lam=0.5, reg_sigma=0.5,
                                max_step=0.1, forward_sigma=None,
                                coverage=None):
    """The pre-change IBP loop (footprints computed once and kept).

    Kept in the test so the recompute path is checked against the exact
    formulation it replaced, not against itself.
    """
    import cv2
    hr = hr.astype(np.float32).copy()
    oh, ow = hr.shape[:2]
    h, w = frames[0].rgb.shape[:2]
    ones = np.ones((h, w), dtype=np.float32)
    if forward_sigma is None:
        forward_sigma = core.SENSOR_PSF_SIGMA_PER_RATIO * float(ratio)
    matrices = [to_drizzle_matrix(al, ratio).astype(np.float64) for al in aligns]
    matrices_inv = [np.linalg.inv(M).astype(np.float32) for M in matrices]
    footprints = [cv2.warpPerspective(ones, M.astype(np.float32), (ow, oh),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
                  for M in matrices]
    norm = np.zeros((oh, ow), dtype=np.float32)
    for m, wt in zip(footprints, weights):
        norm += m * float(wt)
    norm_safe = np.maximum(norm, 1e-6)
    sigma_it = float(forward_sigma)
    fs_it = int(np.ceil(3.0 * sigma_it)) | 1
    for _ in range(int(iterations)):
        hr_fwd = cv2.GaussianBlur(hr, (fs_it, fs_it), sigma_it,
                                  borderType=cv2.BORDER_REPLICATE)
        corr = np.zeros_like(hr)
        for fr, M, M_inv, m, wt in zip(frames, matrices, matrices_inv,
                                       footprints, weights):
            sim = cv2.warpPerspective(hr_fwd, M_inv, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
            residual = fr.rgb.astype(np.float32) - sim
            up = cv2.warpPerspective(residual, M.astype(np.float32), (ow, oh),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)
            corr += up * (float(wt) * m)[:, :, None]
        corr /= norm_safe[:, :, None]
        if fs_it:
            corr = cv2.GaussianBlur(corr, (fs_it, fs_it), sigma_it,
                                    borderType=cv2.BORDER_REPLICATE)
        if reg_sigma and reg_sigma > 0:
            k = int(np.ceil(3.0 * reg_sigma)) | 1
            corr = cv2.GaussianBlur(corr, (k, k), reg_sigma,
                                    borderType=cv2.BORDER_REPLICATE)
        if max_step:
            corr = np.clip(corr, -float(max_step), float(max_step))
        hr = hr + float(lam) * corr
        np.clip(hr, 0.0, 1.0, out=hr)
        if coverage is not None:
            hr[coverage < 1e-6] = 0.0
    return hr


def test_ibp_recomputed_footprint_equals_stored():
    frames = _burst(3, 12, 14, seed=21)
    aligns = [translation_matrix(0.0, 0.0),
              translation_matrix(0.3, -0.2),
              translation_matrix(-0.4, 0.35)]
    rng = np.random.default_rng(4)
    hr = rng.random((24, 28, 3), dtype=np.float32)
    cov = np.ones((24, 28), dtype=np.float32)
    got = iterative_back_projection(hr, frames, aligns, 2.0,
                                    [1 / 3] * 3, iterations=2, coverage=cov)
    want = _ibp_with_stored_footprints(hr, frames, aligns, 2.0,
                                       [1 / 3] * 3, iterations=2,
                                       coverage=cov)
    assert np.abs(got - want).max() <= 1e-6


# --------------------------------------------------------------------------- #
# 4. HR-size guard
# --------------------------------------------------------------------------- #

def test_hr_guard_raises_above_limit():
    frames = _burst(1, 8, 8)
    sr = SuperResolution(ratio=4.0, max_hr_pixels=100)
    with pytest.raises(MemoryError) as ei:
        sr.run(frames)
    msg = str(ei.value)
    assert "HR output" in msg and "max_hr_pixels" in msg


def test_hr_guard_passes_below_limit():
    frames = _burst(1, 8, 8)
    sr = SuperResolution(ratio=4.0, max_hr_pixels=10 ** 6)
    oh, ow = sr._check_hr_size(frames)
    assert (oh, ow) == (32, 32)


def test_hr_guard_warns_near_budget(monkeypatch):
    frames = _burst(1, 8, 8)
    pixels = 32 * 32
    est = core.estimate_hr_peak_bytes(pixels)
    # 1.25x the estimated need: below the limit, above the 75% warn line.
    monkeypatch.setattr(core, "_hr_guard_budget_bytes",
                        lambda: int(est / 0.8))
    sr = SuperResolution(ratio=4.0)
    with pytest.warns(RuntimeWarning, match="memory limit"):
        sr._check_hr_size(frames)


def test_hr_guard_rejects_bad_parameters():
    with pytest.raises(ValueError):
        SuperResolution(max_hr_pixels=0)
    with pytest.raises(ValueError):
        SuperResolution(memory_budget_bytes=0)


def test_estimate_hr_peak_bytes_monotonic():
    assert (core.estimate_hr_peak_bytes(2 * 10 ** 6)
            > core.estimate_hr_peak_bytes(10 ** 6) > 0)


def test_estimate_hr_peak_bytes_scales_with_frame_count():
    """Long bursts grow the per-frame gate stack; the estimate must follow.

    Calibrated on the real burst (val16, 2026-11-20): 4x ROI 1024 peak RSS was
    144 B/HR-px at N=9 and 232 B/HR-px at N=18, i.e. ~10 B/px per extra frame.
    The floor stays at 160 for N<=9 so shorter bursts keep the old estimate.
    """
    assert core._hr_guard_bytes_per_pixel(9) == core._HR_GUARD_BYTES_PER_PIXEL
    assert core._hr_guard_bytes_per_pixel(4) == core._HR_GUARD_BYTES_PER_PIXEL
    assert (core._hr_guard_bytes_per_pixel(18)
            == 160 + 9 * core._HR_GUARD_BYTES_PER_PIXEL_PER_FRAME)
    assert core._hr_guard_bytes_per_pixel(None) == core._HR_GUARD_BYTES_PER_PIXEL
    px = 1024
    assert core.estimate_hr_peak_bytes(px, n_frames=30) > (
        1.25 * core.estimate_hr_peak_bytes(px, n_frames=9))


def test_hr_guard_limit_shrinks_for_long_bursts():
    """A 30-frame burst is refused where 2 frames fit the same budget."""
    budget = 230_000
    short = _burst(2, 8, 8)
    long = _burst(30, 8, 8)
    # Old flat 160 B/px estimate (163,840 B) fits; the 30-frame one does not.
    assert SuperResolution(ratio=4.0, memory_budget_bytes=budget)._check_hr_size(
        short) == (32, 32)
    with pytest.raises(MemoryError, match="30 frames"):
        SuperResolution(ratio=4.0,
                        memory_budget_bytes=budget)._check_hr_size(long)


# --------------------------------------------------------------------------- #
# 5. CPU (float32 streaming) vs GPU deposit parity
# --------------------------------------------------------------------------- #

_HAS_GPU = core._gpu_scatter_env() is not None


@pytest.mark.skipif(not _HAS_GPU, reason="no OpenCL GPU device / pyopencl")
def test_cpu_gpu_deposit_parity():
    import cv2
    rng = np.random.default_rng(1)
    h, w = 64, 48
    frames = []
    for i in range(4):
        rgb = cv2.GaussianBlur(
            rng.random((h, w, 3), dtype=np.float32), (3, 3), 0.6)
        frames.append(RawFrame(name=str(i), rgb=rgb, green=rgb[:, :, 1]))
    aligns = [np.eye(3), translation_matrix(-2.3, 1.7),
              cv2.getRotationMatrix2D((24.0, 16.0), 3.0, 1.0),
              translation_matrix(0.4, 0.4)]
    ws = [0.4, 0.3, 0.2, 0.1]
    for ratio, shape in ((2.3, (110, 148)), (4.0, (256, 192))):
        a, ca = drizzle(frames, aligns, ratio, ws, out_shape=shape,
                        use_gpu_scatter=False)
        b, cb = drizzle(frames, aligns, ratio, ws, out_shape=shape,
                        use_gpu_scatter=True)
        assert np.abs(a - b).max() <= 1e-5
        assert np.abs(ca - cb).max() <= 1e-5
