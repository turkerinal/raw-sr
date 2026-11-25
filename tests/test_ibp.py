"""Tests for classical iterative back-projection (IBP) and the drizzle footprint.

IBP is a data-consistency refinement: it must (a) leave its input untouched when
asked to do nothing, (b) actually reduce the forward-model residual it targets,
(c) sharpen a synthetic burst without inventing detail, and (d) not blow up
noise. All assertions are on synthetic bursts with known subpixel shifts.
"""

import numpy as np
import cv2
import pytest

from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    drizzle,
    iterative_back_projection,
    to_drizzle_matrix,
    translation_matrix,
    high_frequency_gain_db,
    to_view_float,
    laplacian_sharpness,
    _vst_burst_noise_sigma,
    _vst_bilateral_prox,
)


def _random_truth(h=256, w=256, seed=7, sigma=0.6):
    rng = np.random.default_rng(seed)
    t = rng.random((h, w, 3)).astype(np.float32)
    return cv2.GaussianBlur(t, (0, 0), sigma)


def _bandpass_truth(h=256, w=256, lo=0.30, hi=0.50, seed=7):
    rng = np.random.default_rng(seed)
    g = rng.random((h, w)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g))
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    F[(r < lo) | (r > hi)] = 0
    g = np.real(np.fft.ifft2(np.fft.ifftshift(F))).astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    return np.stack([g, g, g], axis=2)


def _sample_lr(truth, dx, dy, lr_shape):
    ratio = truth.shape[0] // lr_shape[0]
    off = 0.5 * (ratio - 1.0)
    xs, ys = np.meshgrid(np.arange(lr_shape[1], dtype=np.float32),
                         np.arange(lr_shape[0], dtype=np.float32))
    return cv2.remap(truth, ratio * (xs + dx) + off,
                     ratio * (ys + dy) + off,
                     cv2.INTER_LINEAR)


SHIFTS = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5),
          (0.25, 0.75), (0.75, 0.25), (0.3, 0.3), (0.7, 0.7), (0.15, 0.6)]


def _aligns(shifts):
    """ref->mov transforms for ``_sample_lr`` sampling shifts (mov(x)=ref(x+d))."""
    return [translation_matrix(-dx, -dy) for dx, dy in shifts]


def _burst(shifts=SHIFTS, truth=None, noise=0.002, seed=1):
    truth = truth if truth is not None else _random_truth()
    h, w = truth.shape[0] // 2, truth.shape[1] // 2
    rng = np.random.default_rng(seed)
    frames = []
    for i, (dx, dy) in enumerate(shifts):
        lr = _sample_lr(truth, dx, dy, (h, w))
        lr = np.clip(lr + rng.normal(0, noise, lr.shape).astype(np.float32), 0, 1)
        frames.append(RawFrame(name=f"f{i}", rgb=lr, green=lr[:, :, 1].copy()))
    return frames, truth


def _forward_residual_energy(hr, frames, aligns, ratio, forward_sigma=None):
    """sum_k || obs_k - D U_k hr ||^2  (the quantity IBP minimises).

    Must use the same forward model as :func:`iterative_back_projection`:
    integrate the HR estimate over the LR pixel support (Gaussian, sigma
    ``0.289 * ratio`` HR px by default) before sampling at the LR grid.
    """
    if forward_sigma is None:
        forward_sigma = 0.289 * float(ratio)
    hr_fwd = hr.astype(np.float32)
    if forward_sigma and forward_sigma > 0:
        k = int(np.ceil(3.0 * forward_sigma)) | 1
        hr_fwd = cv2.GaussianBlur(hr_fwd, (k, k), forward_sigma,
                                  borderType=cv2.BORDER_REPLICATE)
    total = 0.0
    for fr, al in zip(frames, aligns):
        M = to_drizzle_matrix(al, ratio)
        Minv = np.linalg.inv(M).astype(np.float32)
        h, w = fr.rgb.shape[:2]
        sim = cv2.warpPerspective(hr_fwd, Minv, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
        total += float(np.mean((fr.rgb.astype(np.float32) - sim) ** 2))
    return total


# --------------------------------------------------------------------------- #
# IBP core properties
# --------------------------------------------------------------------------- #

def test_ibp_zero_iterations_is_a_noop():
    frames, _ = _burst()
    hr, _cov = drizzle(frames, _aligns(SHIFTS), ratio=2.0)
    out = iterative_back_projection(hr, frames, _aligns(SHIFTS), 2.0, iterations=0)
    assert np.array_equal(out, hr)
    assert out is not hr  # returns a copy, never aliases the caller's array


def test_ibp_reduces_forward_model_residual():
    # With exact subpixel shifts and no noise, one back-projection sweep must
    # make the HR estimate more consistent with every measured frame.
    frames, _ = _burst(noise=0.0)
    hr0, cov = drizzle(frames, _aligns(SHIFTS), ratio=2.0)
    before = _forward_residual_energy(hr0, frames, _aligns(SHIFTS), 2.0)
    hr1 = iterative_back_projection(hr0, frames, _aligns(SHIFTS), 2.0,
                                    iterations=1, lam=0.5, reg_sigma=0.0)
    after = _forward_residual_energy(hr1, frames, _aligns(SHIFTS), 2.0)
    assert after < before, f"residual did not drop: {before:.6f} -> {after:.6f}"


def test_ibp_iterations_keep_improving_consistency():
    frames, _ = _burst(noise=0.0)
    hr0, _ = drizzle(frames, _aligns(SHIFTS), ratio=2.0)
    e = [_forward_residual_energy(hr0, frames, _aligns(SHIFTS), 2.0)]
    hr = hr0
    for _ in range(3):
        hr = iterative_back_projection(hr, frames, _aligns(SHIFTS), 2.0,
                                       iterations=1, lam=0.5, reg_sigma=0.0)
        e.append(_forward_residual_energy(hr, frames, _aligns(SHIFTS), 2.0))
    assert e[-1] < e[0]
    assert e[1] <= e[0] + 1e-9
    assert e[2] <= e[1] + 1e-9


def test_ibp_sharpens_a_synthetic_burst_and_stays_bounded():
    frames, truth = _burst()
    res_off = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                              ibp_iterations=0).run(frames)
    res_on = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                             ibp_iterations=3).run(frames)
    # Sharper and more high frequency than the same pipeline without IBP.
    assert laplacian_sharpness(res_on.hr) > laplacian_sharpness(res_off.hr)
    assert res_on.metrics["sharpness_ratio"] > res_off.metrics["sharpness_ratio"]
    assert res_on.metrics["hf_gain_db"] > res_off.metrics["hf_gain_db"]
    # Values remain valid linear measurements.
    assert res_on.hr.min() >= 0.0 and res_on.hr.max() <= 1.0
    assert np.all(np.isfinite(res_on.hr))
    # Uncovered pixels stay at zero.
    assert np.all(res_on.hr[res_on.coverage < 1e-6] == 0.0)


def test_ibp_does_not_amplify_noise_badly():
    # A noisier burst must still beat the naive single-frame upscale: the
    # regularized update is not allowed to run away. The reference is pinned to
    # frame 0 so the HR grid coincides with the synthetic truth grid (the
    # automatic reference selection intentionally moves the output grid to a
    # central frame, which would make a direct MSE-vs-truth comparison read a
    # pure sub-pixel phase offset).
    frames, truth = _burst(noise=0.02)
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(
        frames, reference_index=0)
    naive = cv2.resize(frames[0].rgb, (res.hr.shape[1], res.hr.shape[0]),
                       interpolation=cv2.INTER_CUBIC)
    err_sr = float(np.mean((res.hr - truth) ** 2))
    err_naive = float(np.mean((naive - truth) ** 2))
    assert err_sr < err_naive, f"noisy SR MSE {err_sr:.5f} >= naive {err_naive:.5f}"


def test_ibp_with_homography_transforms():
    # IBP must accept the 3x3 homographies the orchestrator produces, not only
    # (dx, dy) translations.
    from raw_sr.core.superres import estimate_homography
    frames, _ = _burst()
    ref = frames[0].green
    aligns = []
    for f in frames:
        if f is frames[0]:
            aligns.append(np.eye(3, dtype=np.float64))
        else:
            aligns.append(estimate_homography(ref, f.green, ecc_scale=0.5))
    hr0, _ = drizzle(frames, aligns, ratio=2.0)
    out = iterative_back_projection(hr0, frames, aligns, 2.0, iterations=2)
    assert out.shape == hr0.shape
    assert np.all(np.isfinite(out))
    assert _forward_residual_energy(out, frames, aligns, 2.0) < \
        _forward_residual_energy(hr0, frames, aligns, 2.0)


# --------------------------------------------------------------------------- #
# Drizzle footprint
# --------------------------------------------------------------------------- #

def test_footprint_spreads_an_impulse_and_lowers_its_peak():
    # The pre-blur footprint is a property of the legacy gather path; the
    # scatter default has its own footprint (pixfrac/splat). Pin the gather.
    rgb = np.zeros((32, 32, 3), dtype=np.float32)
    rgb[16, 16] = 1.0
    frame = RawFrame(name="imp", rgb=rgb, green=rgb[:, :, 1].copy())
    hr_bare, _ = drizzle([frame], _aligns([(0.0, 0.0)]), ratio=2.0,
                         deposit="gather", interp=cv2.INTER_NEAREST,
                         post_sigma=0.0, footprint_sigma=0.0)
    hr_fp, _ = drizzle([frame], _aligns([(0.0, 0.0)]), ratio=2.0,
                       deposit="gather", interp=cv2.INTER_NEAREST,
                       post_sigma=0.0, footprint_sigma=0.6)
    assert hr_fp.max() < hr_bare.max(), "footprint did not soften the deposit"
    assert (hr_fp > 1e-6).sum() > (hr_bare > 1e-6).sum(), \
        "footprint did not widen the deposit support"


def test_footprint_preserves_more_high_frequency_than_a_large_blur():
    truth = _bandpass_truth()
    frames, _ = _burst(truth=truth, noise=0.0)
    hr_small, _ = drizzle(frames, _aligns(SHIFTS), ratio=2.0, deposit="gather",
                          interp=cv2.INTER_NEAREST, footprint_sigma=0.3)
    hr_large, _ = drizzle(frames, _aligns(SHIFTS), ratio=2.0, deposit="gather",
                          interp=cv2.INTER_NEAREST, footprint_sigma=1.5)
    gain = high_frequency_gain_db(to_view_float(hr_small, exposure=1.0),
                                  to_view_float(hr_large, exposure=1.0))
    assert gain > 0, f"compact footprint lost the band (gain {gain:.2f} dB)"


def test_default_orchestrator_uses_the_scatter_deposit():
    sr = SuperResolution()
    assert sr.drizzle_post_sigma <= 0.25
    assert sr.drizzle_footprint_sigma > 0.0  # legacy gather footprint, retained
    assert sr.ibp_iterations >= 1
    # The default deposition is a true-drizzle *scatter*, not an interpolated
    # gather (a gather low-passes the recovered >Nyquist band).
    assert sr.drizzle_deposit == "scatter"
    assert sr.drizzle_splat in ("bilinear", "area", "cubic", "gaussian")
    assert sr.drizzle_pixfrac > 0.0
    assert sr.drizzle_subsamples >= 1


# --------------------------------------------------------------------------- #
# B1 (2026-11-18): IBP strength, measured against held-out PSNR.
#
# The default used to be 3 sweeps. With the corrected adjoint the held-out PSNR
# gain is monotone in IBP strength on the real burst, and 3 sweeps measured
# 1.0-1.4 dB *lower* held-out gain at 2x/3x/4x for the same HF-band gain (the
# extra sweeps only added acutance). The default is one sweep and the
# back-projection is the adjoint of the forward model.
# --------------------------------------------------------------------------- #

def test_ibp_defaults_are_one_sweep_and_the_adjoint():
    sr = SuperResolution()
    assert sr.ibp_iterations == 1
    assert sr.ibp_adjoint is True
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(_burst()[0])
    assert res.metrics["ibp_iterations"] == 1
    assert res.metrics["ibp_adjoint"] is True


def test_ibp_adjoint_band_limits_the_correction():
    """The adjoint correction carries the forward PSF, the plain one does not.

    The forward model blurs the HR estimate by sigma = 0.289 * ratio HR px
    before sampling, so its adjoint blurs the back-projected residual by the
    same PSF. The resulting step therefore adds much less above-band energy
    than the legacy plain splat at the same lam.
    """
    frames, _ = _burst(noise=0.0)
    hr0, cov = drizzle(frames, _aligns(SHIFTS), ratio=2.0)
    kw = dict(iterations=1, lam=1.0, reg_sigma=0.0, max_step=None)
    adj = iterative_back_projection(hr0, frames, _aligns(SHIFTS), 2.0, coverage=cov,
                                    adjoint=True, **kw)
    plain = iterative_back_projection(hr0, frames, _aligns(SHIFTS), 2.0, coverage=cov,
                                      adjoint=False, **kw)
    d_adj = adj - hr0
    d_plain = plain - hr0
    hf_adj = high_frequency_gain_db(to_view_float(np.abs(d_adj), exposure=1.0),
                                    to_view_float(np.abs(d_plain), exposure=1.0))
    assert hf_adj < 0.0, f"adjoint correction was not band-limited ({hf_adj:+.2f} dB)"
    # Data consistency is still improved by the band-limited step.
    before = _forward_residual_energy(hr0, frames, _aligns(SHIFTS), 2.0)
    after = _forward_residual_energy(adj, frames, _aligns(SHIFTS), 2.0)
    assert after < before


# --------------------------------------------------------------------------- #
# B2 (2026-11-19, audit fixes d/e): the resolved IBP configuration is part of
# the result. ``ibp_iterations`` changed default 3 -> 1, so any two runs (UI,
# desktop, acceptance, archived JSON) must be comparable from their metrics
# instead of from an assumption about the defaults.
# --------------------------------------------------------------------------- #

def test_metrics_report_the_resolved_ibp_configuration():
    frames, _ = _burst()
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(frames)
    m = res.metrics
    assert m["ibp_iterations"] == 1
    assert m["ibp_lambda"] == pytest.approx(0.5)
    assert m["ibp_reg_sigma"] == pytest.approx(0.5)
    assert m["ibp_max_step"] == pytest.approx(0.1)
    assert m["ibp_forward_sigma_hr"] == pytest.approx(0.289 * 2.0)
    assert m["ibp_joint_refine"] is False
    assert m["ibp_adjoint"] is True
    assert "1 sweep" in m["ibp_summary"] and "adjoint" in m["ibp_summary"]

    off = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                          ibp_iterations=0).run(frames)
    assert off.metrics["ibp_iterations"] == 0
    assert off.metrics["ibp_summary"] == ""

    three = SuperResolution(ratio=3.0, blur_threshold_ratio=0.0,
                            ibp_iterations=3, ibp_lambda=0.3,
                            ibp_adjoint=False).run(frames)
    m3 = three.metrics
    assert m3["ibp_iterations"] == 3
    assert m3["ibp_lambda"] == pytest.approx(0.3)
    assert m3["ibp_adjoint"] is False
    assert m3["ibp_forward_sigma_hr"] == pytest.approx(0.289 * 3.0)
    assert "3 sweeps" in m3["ibp_summary"] and "plain splat" in m3["ibp_summary"]


def test_ibp_summary_distinguishes_configurations():
    """Two runs with different refinement must not look identical in a dump."""
    frames, _ = _burst()
    a = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(frames)
    b = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                        ibp_iterations=3, ibp_adjoint=False).run(frames)
    assert a.metrics["ibp_summary"] != b.metrics["ibp_summary"]


# --------------------------------------------------------------------------- #
# MAP-IBP variance-stabilised edge-preserving proximal step (fix/fwd20)
# --------------------------------------------------------------------------- #

def test_vst_burst_noise_sigma_stands_down_on_pure_high_freq_texture():
    """Pure high-frequency texture must not be mistaken for sensor noise."""
    from raw_sr.core.superres import _vst_burst_noise_sigma

    bp = _bandpass_truth(h=128, w=128, lo=0.26, hi=0.34)
    frames_bp, _ = _burst(truth=bp, noise=0.0)
    assert _vst_burst_noise_sigma(frames_bp, [1.0] * len(frames_bp)) == 0.0

    # A normal noisy scene (smooth structures, not white-noise truth) must
    # return a positive, bounded VST noise estimate.
    smooth = _random_truth(sigma=6.0)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9) * 0.8 + 0.1
    frames_noisy, _ = _burst(truth=smooth, noise=0.01)
    sig = _vst_burst_noise_sigma(frames_noisy, [1.0] * len(frames_noisy))
    assert 0.0 < sig < 0.05


def test_vst_bilateral_prox_is_monotone_and_reduces_flat_noise():
    """Convex bilateral weights cannot ring/overshoot and must denoise flats."""
    from raw_sr.core.superres import _vst_bilateral_prox

    # Clean step edge: zero overshoot / undershoot.
    edge = np.full((64, 64, 3), 0.2, dtype=np.float32)
    edge[:, 32:, :] = 0.8
    out_edge = _vst_bilateral_prox(edge, ratio=3.0, sigma_vst=0.01,
                                   color_scale=2.5, space_lr=0.75)
    assert float(out_edge.min()) >= 0.2 - 1e-6
    assert float(out_edge.max()) <= 0.8 + 1e-6

    # Flat noisy patch: variance drops strictly.
    rng = np.random.default_rng(13)
    flat = np.clip(0.3 + rng.normal(0.0, 0.01, (64, 64, 3)), 0.0, 1.0).astype(
        np.float32)
    out_flat = _vst_bilateral_prox(flat, ratio=3.0, sigma_vst=0.01,
                                   color_scale=2.5, space_lr=0.75)
    assert float(np.var(out_flat)) < float(np.var(flat))


def test_ibp_prox_respects_disable_knobs_and_reports_metrics():
    smooth = _random_truth(sigma=6.0)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-9) * 0.8 + 0.1
    frames, _ = _burst(truth=smooth, noise=0.01)
    hr0, cov = drizzle(frames, _aligns(SHIFTS), ratio=2.0)
    off_scale = iterative_back_projection(
        hr0, frames, _aligns(SHIFTS), 2.0, coverage=cov,
        iterations=1, prox_noise_scale=0.0)
    on_default = iterative_back_projection(
        hr0, frames, _aligns(SHIFTS), 2.0, coverage=cov,
        iterations=1, prox_noise_scale=2.5)
    assert not np.allclose(off_scale, on_default)
    assert np.all(on_default[cov < 1e-6] == 0.0)

    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(frames)
    assert res.metrics["ibp_prox_noise_scale"] == pytest.approx(2.5)
    assert res.metrics["ibp_prox_space_lr"] == pytest.approx(0.75)
