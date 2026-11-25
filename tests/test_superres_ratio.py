"""Ratio-aware drizzle deposit tests (2x / 3x / 4x).

Until this file existed every SR test ran at ratio 2.0, so the shipped
"2x/3x/4x" claim was unverified for two of the three modes. What is pinned
here is the *measured* outcome, not a plausible-looking rule:

1. ``pixfrac`` stays 0.5 LR px at every ratio. The PSF therefore does widen in
   HR terms (rms 0.667 / 0.817 / 0.943 HR px at 2x/3x/4x), and that is kept on
   purpose: with the fixed held-out metric, shrinking the footprint to a
   constant HR size is **prediction-neutral** (a few tenths of a dB, inside
   the burst's scatter) while it multiplies Laplacian sharpness several-fold —
   acutance, not recovered resolution.
2. ``subsamples`` is derived from the footprint in **HR** pixels
   (``ceil(2 * pixfrac * ratio)``, clamped to [2, 3]) so the lattice pitch
   stays <= 0.5 HR px: 2 / 3 / 3 at 2x / 3x / 4x. At 2x the old fixed 3 spent
   2.25x the ``np.bincount`` scatter work for a small, measured-neutral change
   (0.04 HR px PSF rms, 0.67 % of the deposited value with fractional shifts,
   held-out +7.614 vs +7.615 dB) — a change, not an identity.
3. Explicit ``drizzle_pixfrac`` / ``drizzle_subsamples`` overrides are
   honoured unchanged, and the CPU and GPU scatter engines still agree.
"""

import os
import sys

import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from raw_sr.core.superres import (  # noqa: E402
    RawFrame,
    SuperResolution,
    _scatter_fast,
    default_deposit_params,
    deposit_subsamples,
    drizzle,
    high_frequency_gain_db,
    iterative_back_projection,
    to_drizzle_matrix,
    translation_matrix,
)

RATIOS = (2.0, 3.0, 4.0)


def _deposit_psf(ratio, pixfrac, subsamples, shift=None):
    """rms radius (HR px) and central-cell share of one deposited LR impulse.

    Measured on the raw scatter accumulation (:func:`_scatter_fast`), i.e. the
    deposit kernel itself; ``drizzle()`` additionally divides by the per-cell
    coverage map, which would fold the sub-sample lattice into the number.

    By default the source impulse is nudged by ``(r-1)/(2r)`` LR px so its
    deposited centre lands on an HR cell centre: the HR grid is pixel-centre
    aligned, and the discrete second moment of the splat depends on that
    sub-cell phase. Measuring on-cell keeps the number comparable across
    ratios (it is a property of the footprint, not of the parity).
    """
    src = np.zeros((40, 40, 3), np.float32)
    src[20, 20] = 1.0
    oh = int(40 * ratio)
    if shift is None:
        s = 0.5 * (float(ratio) - 1.0) / float(ratio)
        shift = (s, s)
    acc, _cov = _scatter_fast(src, to_drizzle_matrix(shift, ratio), oh, oh,
                              pixfrac, subsamples)
    w = acc[:, :, 0].astype(np.float64)
    w /= max(w.sum(), 1e-12)
    ys, xs = np.mgrid[0:oh, 0:oh]
    cy = float((w * ys).sum())
    cx = float((w * xs).sum())
    rms = float(np.sqrt((w * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum()))
    return rms, float(w.max())


def _synthetic_burst(ratio, n=9, size=128, noise=0.004, seed=1):
    """HR truth (content above the LR Nyquist) -> noisy LR burst + truths."""
    rng = np.random.default_rng(seed)
    hc = wc = int(size * ratio)
    base = rng.normal(0, 1, (hc, wc)).astype(np.float32)
    tex = cv2.GaussianBlur(base, (0, 0), 0.7)
    yy, xx = np.mgrid[0:hc, 0:wc].astype(np.float32)
    smooth = 0.5 + 0.25 * np.sin(xx / 37.0) * np.cos(yy / 53.0)
    truth = np.clip(0.35 + 0.25 * smooth
                    + 0.55 * tex / (np.abs(tex).max() + 1e-9), 0, 1)
    truth = truth.astype(np.float32)

    shifts = [(0.0, 0.0), (0.5, 0.25), (0.25, 0.5), (0.75, 0.1), (0.1, 0.75),
              (0.4, 0.4), (0.62, 0.83), (0.9, 0.55), (0.3, 0.95)][:n]
    frames, aligns = [], []
    for dx, dy in shifts:
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
        off = 0.5 * (ratio - 1.0)
        lr = cv2.remap(truth, ratio * (xs + dx) + off,
                       ratio * (ys + dy) + off,
                       cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
        lr = lr + rng.normal(0, noise, lr.shape).astype(np.float32)
        rgb = np.repeat(lr[:, :, None], 3, axis=2)
        frames.append(RawFrame(name="f", rgb=rgb, green=lr))
        aligns.append(translation_matrix(-dx, -dy))
    return frames, aligns, truth


def _reconstruct(frames, aligns, ratio, pixfrac, subsamples, ibp=3):
    hr, _cov = drizzle(frames, aligns, ratio, pixfrac=pixfrac,
                       subsamples=subsamples, use_gpu_scatter=False)
    if ibp:
        hr = iterative_back_projection(hr, frames, aligns, ratio,
                                       iterations=ibp, use_opencl=False)
    return np.clip(hr, 0, 1)


def _psnr(hr, truth):
    mse = float(np.mean((hr[:, :, 1].astype(np.float64) - truth) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-12))


class TestDepositDefaults:
    @pytest.mark.parametrize("ratio", RATIOS)
    def test_footprint_stays_half_lr_px(self, ratio):
        # Measured optimum on the real burst: 0.5 LR px at EVERY ratio (2x is
        # therefore unchanged from the historical tuning).
        sr = SuperResolution(ratio=ratio)
        assert sr.drizzle_pixfrac == 0.5

    @pytest.mark.parametrize("ratio,expected", [(2.0, 2), (3.0, 3), (4.0, 3)])
    def test_subsamples_scale_with_hr_footprint(self, ratio, expected):
        pf, sub = default_deposit_params(ratio)
        assert sub == expected
        # Lattice pitch in HR px never exceeds half an HR cell, and the count
        # is the cheapest that achieves it (the clamp only binds beyond 3x).
        pitch = pf * ratio / sub
        assert pitch <= 0.5 + 1e-9 or sub == 3
        if sub > 2:
            assert pf * ratio / (sub - 1) > 0.5

    @pytest.mark.parametrize("ratio", RATIOS)
    def test_deposit_psf_matches_measured_value(self, ratio):
        # Exactly the numbers recorded in default_deposit_params' docstring,
        # which is why a fixed pixfrac is kept (see TestRatioQuality).
        pf, sub = default_deposit_params(ratio)
        rms, share = _deposit_psf(ratio, pf, sub)
        expected = {2.0: (0.7071, 0.5625),
                    3.0: (0.8165, 0.4444),
                    4.0: (0.9428, 0.3086)}[ratio]
        assert rms == pytest.approx(expected[0], abs=0.01)
        assert share == pytest.approx(expected[1], abs=0.01)

    def test_two_x_subsamples_are_cheap_to_drop(self):
        # The measured basis of the 2x speedup: dropping 3 -> 2 moves the
        # impulse PSF by 0.04 HR px rms (0.667 -> 0.707, both well inside one
        # HR cell) and the real-burst measurement is a neutral-to-positive
        # *change*, not an identity: held-out +7.614 vs +7.615 dB (ROI 512,
        # fixed metric), HF +0.24 dB, sharpness +7.7 %. The scatter cost falls
        # by 2.25x.
        for shift in ((0.0, 0.0), (0.37, 0.21)):
            a = _deposit_psf(2.0, 0.5, 2, shift=shift)
            b = _deposit_psf(2.0, 0.5, 3, shift=shift)
            assert abs(a[0] - b[0]) < 0.05
            assert abs(a[1] - b[1]) < 0.05

    def test_third_subsample_is_needed_at_three_x(self):
        # At 3x, 2 sub-samples leave a 0.75 HR px pitch -> a measurably
        # coarser deposit, so the derived count stays 3 there.
        a = _deposit_psf(3.0, 0.5, 2)
        b = _deposit_psf(3.0, 0.5, 3)
        assert a[0] - b[0] > 0.03

    @pytest.mark.parametrize("ratio", RATIOS)
    def test_deposit_is_shift_stable(self, ratio):
        pf, sub = default_deposit_params(ratio)
        rms = [_deposit_psf(ratio, pf, sub, shift=s)[0]
               for s in ((0.0, 0.0), (0.37, 0.21))]
        assert max(rms) - min(rms) < 0.06

    def test_explicit_overrides_are_honoured(self):
        sr = SuperResolution(ratio=4.0, drizzle_pixfrac=0.25,
                             drizzle_subsamples=2)
        assert sr.drizzle_pixfrac == 0.25
        assert sr.drizzle_subsamples == 2
        # An explicit wide footprint still gets enough sub-samples ...
        sr2 = SuperResolution(ratio=3.0, drizzle_pixfrac=0.7)
        assert sr2.drizzle_pixfrac == 0.7
        assert sr2.drizzle_subsamples == 3
        # ... and an explicit count is never second-guessed.
        assert SuperResolution(ratio=2.0, drizzle_subsamples=3) \
            .drizzle_subsamples == 3
        with pytest.raises(ValueError):
            SuperResolution(ratio=2.0, drizzle_pixfrac=0.0)
        with pytest.raises(ValueError):
            SuperResolution(ratio=2.0, drizzle_subsamples=0)

    def test_deposit_subsamples_helper_bounds(self):
        assert deposit_subsamples(0.5, 1.0) == 2
        assert deposit_subsamples(0.5, 2.0) == 2
        assert deposit_subsamples(0.5, 3.0) == 3
        assert deposit_subsamples(0.5, 4.0) == 3      # clamp: 4th not shipped
        assert deposit_subsamples(0.5, 8.0) == 3      # clamped
        assert deposit_subsamples(float("nan"), 2.0) == 2


class TestRatioQuality:
    @pytest.mark.parametrize("ratio", RATIOS)
    def test_default_beats_bicubic_at_every_ratio(self, ratio):
        # Real resolution recovery at 3x/4x, not just at the tested 2x.
        frames, aligns, truth = _synthetic_burst(ratio, n=5, size=96)
        pf, sub = default_deposit_params(ratio)
        hr = _reconstruct(frames, aligns, ratio, pf, sub, ibp=2)
        single = cv2.resize(frames[0].rgb, (hr.shape[1], hr.shape[0]),
                            interpolation=cv2.INTER_CUBIC)
        stack = np.mean([cv2.resize(f.rgb, (hr.shape[1], hr.shape[0]),
                                    interpolation=cv2.INTER_CUBIC)
                         for f in frames], axis=0)
        assert _psnr(hr, truth) > _psnr(single, truth) + 2.0
        assert high_frequency_gain_db(hr, stack) > 3.0

    @pytest.mark.parametrize("ratio", RATIOS)
    def test_orchestrator_reports_resolved_deposit(self, ratio):
        frames, _aligns, _truth = _synthetic_burst(ratio, n=3, size=48)
        sr = SuperResolution(ratio=ratio, blur_threshold_ratio=0.0,
                             ibp_iterations=1, use_opencl=False,
                             use_gpu_scatter=False)
        res = sr.run(frames)
        assert res.hr.shape[:2] == (int(round(48 * ratio)),
                                    int(round(48 * ratio)))
        assert res.metrics["drizzle_pixfrac"] == 0.5
        assert res.metrics["drizzle_subsamples"] == {2.0: 2, 3.0: 3,
                                                     4.0: 3}[ratio]

    @pytest.mark.parametrize("ratio", (3.0, 4.0))
    def test_cpu_gpu_scatter_parity(self, ratio):
        frames, _aligns, _truth = _synthetic_burst(ratio, n=3, size=48)
        kw = dict(ratio=ratio, blur_threshold_ratio=0.0, ibp_iterations=1,
                  use_opencl=False)
        res_cpu = SuperResolution(**kw, use_gpu_scatter=False).run(frames)
        res_gpu = SuperResolution(**kw, use_gpu_scatter=True).run(frames)
        assert res_cpu.metrics["opencl_scatter"] in ("cpu", "gpu")
        d = np.abs(res_cpu.hr.astype(np.float64)
                   - res_gpu.hr.astype(np.float64)).max()
        assert d < 1e-5


class TestLowCoverageRegularisation:
    """Normalised-convolution blend for sparsely covered HR cells."""

    def test_well_covered_cells_are_bit_identical(self):
        frames, aligns, _truth = _synthetic_burst(4.0, n=5, size=48)
        weights = [0.2] * 5
        hr_off, cov = drizzle(frames, aligns, 4.0, weights,
                              use_gpu_scatter=False, low_cov_thresh=0.0)
        hr_on, cov_on = drizzle(frames, aligns, 4.0, weights,
                                use_gpu_scatter=False,
                                low_cov_thresh=0.5, low_cov_sigma=0.8)
        np.testing.assert_array_equal(cov, cov_on)
        exp_cov = float(np.sum(weights)) / (4.0 ** 2)
        well = cov >= 0.5 * exp_cov
        assert np.any(well)
        np.testing.assert_array_equal(hr_off[well], hr_on[well])

    def test_sparse_cells_regularised_without_overshoot(self):
        frames, aligns, truth = _synthetic_burst(4.0, n=5, size=64, noise=0.01)
        weights = [0.2] * 5
        hr_off, cov = drizzle(frames, aligns, 4.0, weights,
                              use_gpu_scatter=False, low_cov_thresh=0.0)
        hr_on, _ = drizzle(frames, aligns, 4.0, weights,
                           use_gpu_scatter=False,
                           low_cov_thresh=0.5, low_cov_sigma=0.8)
        assert np.all(np.isfinite(hr_on))
        exp_cov = float(np.sum(weights)) / (4.0 ** 2)
        low = (cov > 1e-6) & (cov < 0.5 * exp_cov)
        assert np.any(low)
        assert not np.array_equal(hr_off[low], hr_on[low])
        assert _psnr(np.clip(hr_on, 0, 1), truth) >= _psnr(np.clip(hr_off, 0, 1), truth)

    @pytest.mark.parametrize("ratio", RATIOS)
    def test_low_cov_cpu_gpu_parity(self, ratio):
        frames, aligns, _truth = _synthetic_burst(ratio, n=5, size=48)
        weights = [0.2] * 5
        hr_cpu, _ = drizzle(frames, aligns, ratio, weights,
                            use_gpu_scatter=False,
                            low_cov_thresh=0.5, low_cov_sigma=0.8)
        hr_gpu, _ = drizzle(frames, aligns, ratio, weights,
                            use_gpu_scatter=True,
                            low_cov_thresh=0.5, low_cov_sigma=0.8)
        assert float(np.max(np.abs(hr_cpu - hr_gpu))) < 1e-5

