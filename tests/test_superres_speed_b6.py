"""B6 speed: exactness of the vectorised reductions behind the SR run.

The speed pass replaced three hot, numerically load-bearing kernels with
equivalent faster ones:

* the temporal median (``np.nanmedian``'s masked argsort -> in-band sort +
  rank gather),
* the HF-band power spectrum (``np.fft.fft2`` float64 -> ``cv2.dft`` float32),
* the luma/display transforms (three numpy passes -> one ``cv2`` SIMD pass).

These tests pin the equivalence, so a later "optimisation" cannot silently
move the gate decisions or the reported HF/held-out numbers.
"""

import numpy as np
import cv2

from raw_sr.core.superres import (
    _ghost_gate_median,
    _interframe_ghost_energy,
    _masked_temporal_median,
    _nanmedian_axis0,
    _warp_frames_to_hr,
    high_frequency_gain_db,
    to_view_float,
    RawFrame,
    translation_matrix,
)


def _fast_band_energy(img, cutoff=0.25):
    if img.ndim == 3:
        g = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1]
             + 0.114 * img[:, :, 2])
    else:
        g = img
    g = np.asarray(g, dtype=np.float32)
    g = g - g.mean()
    win = np.outer(np.hanning(g.shape[0]),
                   np.hanning(g.shape[1])).astype(np.float32)
    F = np.fft.fft2(g * win)
    P = np.abs(F)
    np.square(P, out=P)
    fy2 = np.fft.fftfreq(P.shape[0])[:, None] ** 2
    fx2 = np.fft.fftfreq(P.shape[1])[None, :] ** 2
    return float(P[(fy2 + fx2) >= cutoff * cutoff].sum())


def test_temporal_median_matches_nanmedian_with_and_without_mask():
    rng = np.random.default_rng(7)
    for n in (2, 3, 4, 9):
        S = rng.random((n, 19, 23), dtype=np.float32)
        V = rng.random((n, 19, 23)) > 0.2
        if n == 3:
            V[:, :, :2] = False  # some all-NaN output pixels
        with np.errstate(invalid="ignore"):
            ref = np.nanmedian(np.where(V, S, np.nan), axis=0)
        got = _nanmedian_axis0(
            S, V, target_bytes=19 * 23 * n * 4 * 2)
        assert np.array_equal(np.isnan(ref), np.isnan(got))
        assert np.allclose(ref[~np.isnan(ref)], got[~np.isnan(got)],
                           rtol=0, atol=0)

        ref_plain = np.nanmedian(S, axis=0)
        got_plain = _nanmedian_axis0(
            S, None, target_bytes=19 * 23 * n * 4 * 2)
        assert np.array_equal(ref_plain, got_plain)


def test_masked_temporal_median_agrees_with_nanmedian():
    rng = np.random.default_rng(11)
    S = rng.random((5, 17, 13), dtype=np.float32)
    V = rng.random((5, 17, 13)) > 0.1
    with np.errstate(invalid="ignore"):
        ref = np.nanmedian(np.where(V, S, np.nan), axis=0)
    got = _masked_temporal_median(S, V, chunk=4)
    assert np.array_equal(np.isnan(ref), np.isnan(got))
    assert np.allclose(ref[~np.isnan(ref)], got[~np.isnan(got)],
                       rtol=0, atol=0)


def test_hf_metric_matches_the_float64_fft_definition():
    rng = np.random.default_rng(3)
    a = rng.random((96, 80, 3), dtype=np.float32)
    b = cv2.GaussianBlur(a, (5, 5), 1.2)
    ref = 10.0 * np.log10(_fast_band_energy(a) / _fast_band_energy(b))
    got = high_frequency_gain_db(a, b)
    assert abs(got - ref) < 1e-4


def test_to_view_float_is_the_same_curve_as_np_power():
    rng = np.random.default_rng(5)
    img = rng.random((64, 48, 3), dtype=np.float32)
    ref = np.power(np.clip(img * 4.0, 0.0, 1.0), 1.0 / 2.2)
    got = to_view_float(img)
    assert float(np.abs(got - ref).max()) <= 1e-6


def test_ghost_energy_reuses_the_gate_median_exactly():
    rng = np.random.default_rng(13)
    frames = []
    for i in range(4):
        rgb = rng.random((32, 40, 3), dtype=np.float32)
        frames.append(RawFrame(name=str(i), rgb=rgb, green=rgb[:, :, 1]))
    aligns = [np.eye(3), translation_matrix(0.6, -0.4),
              translation_matrix(-0.3, 0.9), translation_matrix(0.2, 0.1)]
    S, V, _oh, _ow = _warp_frames_to_hr(frames, aligns, 2.0)
    median = _ghost_gate_median(S)
    fresh = _interframe_ghost_energy(frames, aligns, 2.0, stack=(S, V))
    reused = _interframe_ghost_energy(frames, aligns, 2.0, stack=(S, V),
                                      median=median)
    assert fresh == reused
