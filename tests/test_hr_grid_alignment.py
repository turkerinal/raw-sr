"""Regression tests for the pixel-centre HR grid (review B8 items #4/#5).

The HR grid is pixel-centre aligned: HR pixel ``i`` has its centre at the
reference LR coordinate ``(i - (r - 1)/2) / r``, i.e. the ``cv2.resize``
geometry. Before 2026-11-20 the grid was corner-aligned and the output sat
``(r - 1)/2`` HR px up-left of every centre-aligned upscale (0.5 / 1.0 / 1.5 px
at 2x / 3x / 4x), misaligning the before/after panels, the bicubic baseline and
the GT comparisons; at 3x a 1-4 px black rim also stayed uncovered at the
bottom/right.
"""

import numpy as np
import cv2
import pytest

from raw_sr.core.superres import (
    RawFrame,
    drizzle,
    iterative_back_projection,
    predict_lr_from_hr,
    to_drizzle_matrix,
    translation_matrix,
)


RATIOS = (2.0, 3.0, 4.0)


def _impulse_centroid(hr):
    g = hr[:, :, 1].astype(np.float64)
    ys, xs = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    tot = float(g.sum())
    return float((g * xs).sum() / tot), float((g * ys).sum() / tot)


def _bandlimited_scene(seed=1, size=400):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32),
                         (0, 0), 2.5)
    t = (t - t.mean()) / (t.std() + 1e-9)
    t = np.clip(0.35 + 0.3 * t, 0.0, 1.0)
    return np.dstack([t, t, t])


def _lr_from_hr(truth, ratio, h, w):
    """Sensor model: integrate the LR footprint, sample at the pixel centre."""
    src = cv2.GaussianBlur(truth, (0, 0), 0.289 * ratio)
    off = 0.5 * (ratio - 1.0)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(src, ratio * xs + off, ratio * ys + off,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _phase_shift(a, b, size=256):
    a = cv2.cvtColor(np.asarray(a, np.float32), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(np.asarray(b, np.float32), cv2.COLOR_RGB2GRAY)
    size = int(min(size, a.shape[0], a.shape[1], b.shape[0], b.shape[1]))
    a = a[:size, :size]
    b = b[:size, :size]
    a = a - a.mean()
    b = b - b.mean()
    win = cv2.createHanningWindow((size, size), cv2.CV_32F)
    (dx, dy), _resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy)


# --------------------------------------------------------------------------- #
# #4 pixel-centre alignment
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ratio", RATIOS)
def test_impulse_deposits_at_the_pixel_centre(ratio):
    lr = np.zeros((32, 32, 3), np.float32)
    lr[10, 15] = 1.0
    fr = RawFrame(name="imp", rgb=lr, green=lr[:, :, 1].copy())
    hr, _ = drizzle([fr], [np.eye(3)], ratio=ratio, deposit="scatter")
    cx, cy = _impulse_centroid(hr)
    off = (ratio - 1.0) / 2.0
    assert abs(cx - (ratio * 15 + off)) < 1e-3, f"ratio {ratio}: cx={cx}"
    assert abs(cy - (ratio * 10 + off)) < 1e-3, f"ratio {ratio}: cy={cy}"


@pytest.mark.parametrize("ratio", RATIOS)
def test_drizzle_matches_a_centre_aligned_upscale(ratio):
    """Drizzle of a centre-sampled LR frame vs the GT HR scene: zero shift."""
    truth = _bandlimited_scene()
    lr = _lr_from_hr(truth, ratio, 100, 100)
    fr = RawFrame(name="f", rgb=lr, green=lr[:, :, 1].copy())
    hr, _cov = drizzle([fr], [np.eye(3)], ratio=ratio, deposit="scatter")
    dx, dy = _phase_shift(hr, truth)
    assert abs(dx) < 0.1 and abs(dy) < 0.1, (
        f"ratio {ratio}: drizzle is ({dx:+.3f},{dy:+.3f}) px from a "
        "centre-aligned upscale")


@pytest.mark.parametrize("ratio", (2, 3, 4))
def test_drizzle_matches_the_bicubic_baseline(ratio):
    """The 'before' panel (cv2.resize) and the SR output share a grid.

    A Gaussian blob is used as the measurable: drizzle's compact splat leaves a
    small lattice ripple on a ramp (it deposits measurements, it is not an
    interpolator), but its centroid is exact, and the bicubic baseline's
    centroid is the same if both grids are pixel-centre aligned.
    """
    h = w = 96
    x0, y0, sigma = 41.3, 30.7, 2.2
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    blob = 0.1 + 0.8 * np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2)
                              / (2.0 * sigma ** 2))
    lr = np.repeat(blob[:, :, None], 3, axis=2).astype(np.float32)
    fr = RawFrame(name="f", rgb=lr, green=lr[:, :, 1].copy())
    hr, _ = drizzle([fr], [np.eye(3)], ratio=ratio, deposit="scatter")
    up = cv2.resize(lr, (hr.shape[1], hr.shape[0]),
                    interpolation=cv2.INTER_CUBIC)

    def centroid(img):
        g = np.asarray(img[..., 1], np.float64)[24:-24, 24:-24]
        g = g - g.min()
        yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
        tot = g.sum()
        return (xx * g).sum() / tot + 24, (yy * g).sum() / tot + 24

    off = (ratio - 1.0) / 2.0
    ex, ey = ratio * x0 + off, ratio * y0 + off
    for name, img in (("drizzle", hr), ("bicubic", up)):
        cx, cy = centroid(img)
        assert abs(cx - ex) < 0.1 and abs(cy - ey) < 0.1, (
            f"ratio {ratio}: {name} centroid ({cx:.3f},{cy:.3f}) != "
            f"pixel-centre ({ex:.3f},{ey:.3f})")


def test_to_drizzle_matrix_includes_the_centre_offset():
    for ratio in RATIOS:
        M = to_drizzle_matrix(np.eye(3), ratio)
        off = (ratio - 1.0) / 2.0
        assert np.isclose(M[0, 2], off) and np.isclose(M[1, 2], off)
        # An LR pixel centre maps to the centre of its ratio x ratio block.
        assert np.isclose(M[0, 0], ratio) and np.isclose(M[1, 1], ratio)


def test_heldout_prediction_and_ibp_keep_the_same_grid():
    """Forward model, IBP and deposit must not move pixels relative to truth."""
    ratio = 3.0
    truth = _bandlimited_scene(seed=3)
    lr = _lr_from_hr(truth, ratio, 64, 64)
    fr = RawFrame(name="f", rgb=lr, green=lr[:, :, 1].copy())
    hr, cov = drizzle([fr], [np.eye(3)], ratio=ratio, deposit="scatter")
    refined = iterative_back_projection(hr, [fr], [np.eye(3)], ratio,
                                        iterations=1, lam=0.3)
    pred = predict_lr_from_hr(refined, np.eye(3), ratio, lr.shape[:2],
                              forward_sigma=0.0)
    # The prediction returns to the frame's own LR grid; a half-pixel grid
    # mismatch would show up as a shift here.
    assert np.mean(np.abs(pred[4:-4, 4:-4] - lr[4:-4, 4:-4])) < 0.05
    dx, dy = _phase_shift(refined, truth, size=150)
    assert abs(dx) < 0.1 and abs(dy) < 0.1


def test_constant_image_stays_constant_on_the_centre_grid():
    lr = np.empty((40, 50, 3), np.float32)
    lr[..., 0] = 0.2
    lr[..., 1] = 0.5
    lr[..., 2] = 0.8
    frames = [RawFrame(name=f"f{i}", rgb=lr, green=lr[:, :, 1].copy())
              for i in range(3)]
    aligns = [translation_matrix(dx, dy)
              for dx, dy in ((0, 0), (0.4, -0.3), (-0.2, 0.5))]
    hr, cov = drizzle(frames, aligns, ratio=3.0, deposit="scatter")
    inner = hr[12:-12, 12:-12]
    assert np.allclose(inner[..., 0], 0.2, atol=2e-3)
    assert np.allclose(inner[..., 1], 0.5, atol=2e-3)
    assert np.allclose(inner[..., 2], 0.8, atol=2e-3)


# --------------------------------------------------------------------------- #
# #5 the bottom/right rim
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ratio", (3, 4))
def test_no_uncovered_rim_at_odd_dimensions(ratio):
    """At 3x/4x the old corner grid left 1-4 black rows/cols bottom/right."""
    h, w = 157, 203
    rng = np.random.default_rng(0)
    base = rng.random((h, w, 3)).astype(np.float32)
    frames = [RawFrame(name=f"f{i}", rgb=base,
                       green=base[:, :, 1].copy()) for i in range(2)]
    aligns = [translation_matrix(0.0, 0.0), translation_matrix(0.3, -0.2)]
    hr, cov = drizzle(frames, aligns, ratio=ratio, deposit="scatter")
    zeros = cov < 1e-6
    assert int(zeros.sum()) == 0, (
        f"ratio {ratio}: {int(zeros.sum())} uncovered HR pixels "
        f"(last row {int(zeros[-1].sum())}, last col {int(zeros[:, -1].sum())})")
