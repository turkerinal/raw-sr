"""Tests for the CFA-domain multi-frame drizzle prototype.

The module deposits measured Bayer sites per channel instead of demosaicing
first.  The tests pin the properties that make it a measurement (not an
invention): channel separation, physical placement, hole filling only where
there is no data, and a ground-truth win over the demosaic+drizzle pipeline on
a synthetic burst with known HR truth.
"""

import numpy as np
import cv2
import pytest

from raw_sr.core.cfa_drizzle import (
    CFAFrame,
    cfa_drizzle,
    channel_map,
    pattern_array,
)
from raw_sr.core.superres import (
    RawFrame,
    denoised_lr_stack,
    drizzle,
    iterative_back_projection,
)

BGGR = np.array([[2, 3], [1, 0]], dtype=np.int8)


def _translation(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def _mosaic_from_rgb(rgb, pattern=BGGR):
    h, w = rgb.shape[:2]
    ch = channel_map(pattern, (h, w))
    return np.take_along_axis(rgb, ch[..., None], axis=2)[..., 0]


def _demosaic(mosaic, pattern=BGGR):
    u16 = np.clip(mosaic * 65535.0, 0, 65535).astype(np.uint16)
    # OpenCV's Bayer names are offset by one pixel for a mosaic whose (0,0) is
    # the first letter: our BGGR mosaic is demosaiced as "BayerRG" here.  See
    # the note in scripts/exp_cfa_sr.py; pinned by a per-channel test.
    return cv2.cvtColor(u16, cv2.COLOR_BayerRG2RGB_EA).astype(
        np.float32) / 65535.0


def _psnr(a, b, mask=None):
    d = (a.astype(np.float64) - b.astype(np.float64)) ** 2
    mse = float(d[mask].mean()) if mask is not None else float(d.mean())
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def _shift_hr(img, dx, dy):
    h, w = img.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    ph = np.exp(-2j * np.pi * (fx * dx + fy * dy))
    return np.stack([np.real(np.fft.ifft2(np.fft.fft2(img[:, :, c]) * ph))
                     for c in range(img.shape[2])], axis=2).astype(np.float32)


# --------------------------------------------------------------------------- #
# Pattern / channel handling
# --------------------------------------------------------------------------- #
def test_pattern_array_accepts_rawpy_and_string_forms():
    assert np.array_equal(pattern_array("RGGB"), np.array([[0, 1], [1, 2]]))
    assert np.array_equal(pattern_array(BGGR), BGGR)
    with pytest.raises(ValueError):
        pattern_array([[0, 1], [0, 2]])          # only one green
    with pytest.raises(ValueError):
        pattern_array("RGBG")                    # one G in the string form
    with pytest.raises(ValueError):
        pattern_array([[0, 4], [1, 2]])


def test_demosaic_test_helper_preserves_channel_identity():
    """Regression pin: the harness demosaic must not swap red and blue.

    ``load_raw_linear`` returns RGB, so a B5/B6 synthetic comparison has to
    demosaic the same way.  OpenCV's Bayer names are offset by one pixel for a
    mosaic whose (0,0) is the first letter, so the BGGR helper mosaic needs
    ``COLOR_BayerRG2RGB_EA``; using the BGGR code (or a BGR variant plus a
    channel swap) silently puts B in the R slot and made the demosaic pipeline
    look ~10 dB worse than it is.
    """
    h = w = 32
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.8, 0.5, 0.2
    out = _demosaic(_mosaic_from_rgb(rgb))
    inner = out[4:-4, 4:-4].mean(axis=(0, 1))
    assert np.allclose(inner, [0.8, 0.5, 0.2], atol=2e-3), inner


def test_channel_map_marks_both_green_phases_as_channel_one():
    ch = channel_map(BGGR, (4, 4))
    assert ch[0, 0] == 2 and ch[0, 1] == 1
    assert ch[1, 0] == 1 and ch[1, 1] == 0
    # Two greens per 2x2 block, one R, one B.
    assert np.bincount(ch.ravel(), minlength=3).tolist() == [4, 8, 4]


# --------------------------------------------------------------------------- #
# Deposit properties
# --------------------------------------------------------------------------- #
def test_constant_field_is_reconstructed_as_a_constant():
    h, w, ratio = 64, 64, 2
    rgb = np.empty((h, w, 3), np.float32)
    rgb[..., 0] = 0.2
    rgb[..., 1] = 0.5
    rgb[..., 2] = 0.8
    mosaic = _mosaic_from_rgb(rgb)
    shifts = [(0, 0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.25, 0.75)]
    aligns = [_translation(*s) for s in shifts]
    res = cfa_drizzle([mosaic] * len(aligns), aligns, ratio, BGGR)
    inner = res.hr[8:-8, 8:-8]
    assert np.allclose(inner[..., 0], 0.2, atol=2e-2)
    assert np.allclose(inner[..., 1], 0.5, atol=2e-2)
    assert np.allclose(inner[..., 2], 0.8, atol=2e-2)


def test_measured_sample_lands_at_its_physical_hr_position():
    h, w, ratio = 48, 48, 2
    mosaic = np.zeros((h, w), np.float32)
    x, y = 20, 17                    # r=1,c=0 -> a green site (BGGR)
    mosaic[y, x] = 1.0
    res = cfa_drizzle([mosaic], [_translation(0, 0)], ratio, BGGR,
                      pixfrac=0.5)
    g = res.hr[..., 1]
    peak = np.unravel_index(int(np.argmax(g)), g.shape)
    # to_drizzle_matrix maps LR pixel (x, y) to pixel-centre-aligned HR
    # (ratio*x + (ratio-1)/2, ratio*y + (ratio-1)/2).
    off = (ratio - 1) / 2.0
    assert abs(peak[0] - (ratio * y + off)) <= 1
    assert abs(peak[1] - (ratio * x + off)) <= 1


def test_channels_do_not_leak():
    h, w, ratio = 48, 48, 2
    rgb = np.empty((h, w, 3), np.float32)
    rgb[..., 0] = 0.8
    rgb[..., 1] = 0.0
    rgb[..., 2] = 0.0
    mosaic = _mosaic_from_rgb(rgb)
    res = cfa_drizzle([mosaic], [_translation(0, 0)], ratio, BGGR)
    inner = (slice(8, -8), slice(8, -8))
    assert res.hr[inner][..., 0].mean() > 0.7
    assert np.abs(res.hr[inner][..., 1]).max() < 1e-6
    assert np.abs(res.hr[inner][..., 2]).max() < 1e-6


def test_frame_weights_are_honoured():
    h, w, ratio = 24, 24, 2
    rgb1 = np.full((h, w, 3), 0.2, np.float32)
    rgb2 = np.full((h, w, 3), 0.6, np.float32)
    # Same alignment: both frames cover the same sites, so the deposit is a
    # true weighted mean there (different shifts would leave each frame
    # covering a different subset and the test would measure hole filling).
    res = cfa_drizzle([_mosaic_from_rgb(rgb1), _mosaic_from_rgb(rgb2)],
                      [_translation(0, 0), _translation(0, 0)],
                      ratio, BGGR, weights=[1.0, 3.0])
    inner = res.hr[6:-6, 6:-6]
    assert np.allclose(inner, 0.5, atol=2e-2)


def test_ratio_four_single_frame_holes_are_filled_and_finite():
    h, w, ratio = 32, 32, 4
    rgb = np.full((h, w, 3), 0.4, np.float32)
    mosaic = _mosaic_from_rgb(rgb)
    res = cfa_drizzle([mosaic], [_translation(0, 0)], ratio, BGGR)
    assert res.hr.shape == (h * ratio, w * ratio, 3)
    assert np.isfinite(res.hr).all()
    assert res.coverage.min() >= 0.0
    # A single 4x frame cannot fill every HR pixel with R/B measurements.
    assert res.metrics["fill_fraction"]["R"] > 0.0
    assert res.metrics["fill_fraction"]["B"] > 0.0
    assert res.hr[16:-16, 16:-16].mean() == pytest.approx(0.4, abs=0.02)


# --------------------------------------------------------------------------- #
# Ground-truth synthetic burst
# --------------------------------------------------------------------------- #
def _synthetic_burst(ratio=2, lr=128, n=7, seed=5, sigma_lens=0.55,
                     full_well=1023.0, read_noise=1.5, quantise=True):
    """A detail-rich HR scene and a noisy, sub-pixel-shifted Bayer burst.

    Coloured hard edges are essential: a grey or smooth scene flatters any
    demosaic (there is no chroma to interpolate), while the real burst and a
    natural scene both contain them.
    """
    rng = np.random.default_rng(seed)
    H = lr * ratio
    F = np.fft.fftshift(np.fft.fft2(rng.standard_normal((H, H))))
    fy = np.fft.fftshift(np.fft.fftfreq(H))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(H))[None, :]
    F *= 1.0 / (np.sqrt(fx ** 2 + fy ** 2) + 0.015) ** 0.85
    base = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    base = (base - base.mean()) / (base.std() + 1e-9)
    luma = np.clip(0.42 + 0.10 * base, 0.02, 0.92)
    scene = np.stack([luma * 0.96, luma, luma * 1.04], axis=2
                     ).astype(np.float32)
    for _ in range(24):
        kind = rng.integers(0, 3)
        cx, cy = rng.integers(6, H - 10, 2)
        a, b = int(rng.integers(3, max(4, H // 8))), int(
            rng.integers(3, max(4, H // 8)))
        col = rng.uniform(0.05, 0.9, 3).astype(np.float32)
        if kind == 0:
            cv2.rectangle(scene, (cx, cy), (cx + a, cy + b), col.tolist(), -1)
        elif kind == 1:
            cv2.ellipse(scene, (cx, cy), (a // 2, b // 2),
                        float(rng.uniform(0, 180)), 0, 360, col.tolist(), -1)
        else:
            cv2.line(scene, (cx, cy), (cx + a, cy + b), col.tolist(),
                     int(rng.integers(1, 3)))
    yy, xx = np.mgrid[0:max(8, H // 5), 0:max(8, H // 5)]
    f = 0.15 + 0.30 * xx / max(xx.max(), 1)
    scene[H // 2:H // 2 + yy.shape[0], H // 2:H // 2 + yy.shape[1]] += (
        0.08 * np.sin(2 * np.pi * (f * xx + 0.19 * yy)))[..., None]
    scene = np.clip(scene, 0.0, 1.0)

    blurred = cv2.GaussianBlur(scene, (0, 0), sigma_lens * ratio,
                               borderType=cv2.BORDER_REFLECT)
    box = cv2.blur(blurred, (int(ratio), int(ratio)),
                   borderType=cv2.BORDER_REFLECT)
    D = 2.0
    offs = np.stack([rng.uniform(-D, D, n), rng.uniform(-D, D, n)], axis=1)
    offs[0] = (0.0, 0.0)
    xs, ys = np.meshgrid(np.arange(lr, dtype=np.float32),
                         np.arange(lr, dtype=np.float32))
    mosaics, rgb_frames = [], []
    off = 0.5 * (float(ratio) - 1.0)
    for k in range(n):
        mx = (ratio * (xs + np.float32(offs[k, 0])) + off).astype(np.float32)
        my = (ratio * (ys + np.float32(offs[k, 1])) + off).astype(np.float32)
        rgb = cv2.remap(box, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
        m = _mosaic_from_rgb(rgb)
        # Photon shot + read noise in electrons (10-bit sensor), the regime of
        # the real burst: sigma ~0.014 at a 0.19 green level.
        m = (rng.poisson(np.clip(m, 0, 1) * full_well)
             + rng.normal(0.0, read_noise, m.shape)) / full_well
        if quantise:
            m = np.round(np.clip(m, 0, 1) * full_well) / full_well
        mosaics.append(m.astype(np.float32))
        rgb_frames.append(rgb)
    # ref->frame is T(-offset): frame_k(x) = target(x + offs[k]) and the
    # deposit/sample convention is dst(x) = src(M^-1 x).
    aligns = [_translation(-(offs[k, 0] - offs[0, 0]),
                           -(offs[k, 1] - offs[0, 1])) for k in range(n)]
    target = _shift_hr(box, ratio * offs[0, 0], ratio * offs[0, 1])
    target = target[:lr * ratio, :lr * ratio]
    margin = int(np.ceil(D)) * int(ratio) + 4
    mask = np.zeros(target.shape[:2], dtype=bool)
    mask[margin:-margin, margin:-margin] = True
    return mosaics, rgb_frames, target, mask, aligns, offs


def test_cfa_drizzle_is_at_parity_with_the_demosaic_pipeline_on_ground_truth():
    """The CFA path must not be *worse* than the shipped pipeline, and must not
    be claimed better.

    History: an earlier version of this test asserted a +0.2 dB win, and its
    ``_demosaic`` helper demosaiced the BGGR mosaic with ``COLOR_BayerBG2BGR``
    (R/B swapped), which made the *baseline* ~10 dB worse.  With the correct
    code the two are at parity on this burst (measured +0.3/-0.6/-1.0 dB at
    2x/3x/4x for ``pixfrac=0.5``; see docs/CFA_DRIZZLE.md §3).  The module is
    therefore EXPERIMENTAL: this test pins honesty, not a performance claim.
    """
    ratio = 2
    mosaics, rgb_frames, target, mask, aligns, _ = _synthetic_burst()
    dems = [_demosaic(m) for m in mosaics]
    frames = [RawFrame(str(k), d, d[:, :, 1].copy())
              for k, d in enumerate(dems)]
    weights = [1.0 / len(frames)] * len(frames)
    hr, cov = drizzle(frames, aligns, ratio, weights)
    hr = iterative_back_projection(hr, frames, aligns, ratio, weights,
                                   iterations=1, lam=0.5, reg_sigma=0.5,
                                   max_step=0.1, coverage=cov)
    p_dem = _psnr(hr, target, mask)
    for pixfrac in (0.5, 1.0):
        cfa = cfa_drizzle(mosaics, aligns, ratio, BGGR, weights,
                          pixfrac=pixfrac)
        p_cfa = _psnr(cfa.hr, target, mask)
        # Parity band: the CFA path may lose up to 1.5 dB on total PSNR but
        # must not collapse, and must not be reported as a win.
        assert p_cfa > p_dem - 1.5, (pixfrac, p_cfa, p_dem)
        assert cfa.hr[mask].std() > 0.02


def test_level_calibration_recovers_a_known_per_channel_scale():
    """LibRaw's colour scale differs from camera_wb/green per channel.

    A comparison must remove that constant bias first; the helper recovers a
    known scale exactly and leaves a correctly scaled mosaic untouched.
    """
    from raw_sr.core.cfa_drizzle import (CFAFrame, apply_level_gains,
                                         calibrate_cfa_levels)
    rng = np.random.default_rng(3)
    h = w = 64
    rgb = rng.random((h, w, 3)).astype(np.float32) * 0.5 + 0.1
    mosaic = _mosaic_from_rgb(rgb)
    frame = CFAFrame(name="t", mosaic=mosaic, pattern=BGGR)
    # The "reference" is the same signal with a known per-channel offset.
    scale = np.array([0.965, 0.973, 1.057])
    ref = rgb * scale
    gains, stats = calibrate_cfa_levels(frame, ref)
    assert np.allclose(gains, scale, atol=1e-3), gains
    assert stats["B"]["gain"] == pytest.approx(1.057, abs=1e-3)
    fixed = apply_level_gains(frame, gains)
    assert np.allclose(fixed.mosaic, mosaic * gains[
        channel_map(BGGR, (h, w))], atol=1e-6)
    assert fixed.meta["level_gains"] == pytest.approx(gains.tolist())
    # Idempotent on a mosaic that is already on the reference scale.
    same = calibrate_cfa_levels(CFAFrame(name="t", mosaic=mosaic,
                                         pattern=BGGR), rgb)[0]
    assert np.allclose(same, 1.0, atol=1e-3)
