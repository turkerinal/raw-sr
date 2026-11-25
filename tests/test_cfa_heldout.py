"""Tests for the measurement-space (Bayer CFA) held-out metric.

The metric must be a *fair* comparison: the held-out frame may not leak into
either side, both sides must be scored on the same sites, the baseline must be
a real multi-frame CFA average (no dark rim, no demosaic), and the ranking it
produces must agree with ground-truth HR PSNR on a synthetic burst.
"""

import numpy as np
import cv2
import pytest

from raw_sr.core.cfa_drizzle import cfa_drizzle, channel_map
from raw_sr.core.cfa_heldout import (
    CFA_HELDOUT_MIN_COVERAGE_FRACTION,
    cfa_heldout_compare,
    cfa_heldout_masks,
    cfa_lr_stack_at_held,
    evaluate_cfa_heldout,
    frame_to_held_matrix,
    predict_cfa_from_hr,
    site_support,
)
from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    denoised_lr_stack,
    drizzle,
    estimate_homography,
    iterative_back_projection,
)

BGGR = np.array([[2, 3], [1, 0]], dtype=np.int8)


def _demosaic(mosaic):
    u16 = np.clip(mosaic * 65535.0, 0, 65535).astype(np.uint16)
    return cv2.cvtColor(u16, cv2.COLOR_BayerRG2RGB_EA).astype(
        np.float32) / 65535.0


def _mosaic_from_rgb(rgb, pattern=BGGR):
    h, w = rgb.shape[:2]
    ch = channel_map(pattern, (h, w))
    return np.take_along_axis(rgb, ch[..., None], axis=2)[..., 0]


def _translation(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def _psnr(a, b, mask):
    return 10 * np.log10(1.0 / max(float(((a - b) ** 2)[mask].mean()), 1e-12))


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def test_frame_to_held_matrix_maps_frame_coordinates():
    """p_held = p_frame + (d_held - d_frame) for pure translations."""
    dk, dh = (1.5, -2.0), (-0.5, 0.75)
    M = frame_to_held_matrix(_translation(*dk), _translation(*dh))
    p = np.array([10.0, 20.0, 1.0])
    q = M @ p
    assert q[0] == pytest.approx(10.0 + dh[0] - dk[0])
    assert q[1] == pytest.approx(20.0 + dh[1] - dk[1])


def test_site_support_shrinks_near_the_hr_border():
    ratio, h, w = 2, 64, 64
    sup = site_support(np.eye(3), ratio, (h, w), 0.289 * ratio, (h * ratio,
                                                                w * ratio))
    assert sup[32, 32]
    assert not sup[0, 0] and not sup[-1, -1]
    # A larger PSF support removes a wider rim. sigma=2.0 (pad 7) rather than
    # 1.0: with the pixel-centre grid the sample positions sit at 2x+0.5, so
    # pads 3 and 4 happen to keep the same integer lattice; 7 shifts it.
    wider = site_support(np.eye(3), ratio, (h, w), 2.0, (h * ratio, w * ratio))
    assert wider.sum() < sup.sum()


# --------------------------------------------------------------------------- #
# Baseline properties
# --------------------------------------------------------------------------- #
def test_baseline_is_a_coverage_normalised_mean_without_a_dark_rim():
    """Constant scene + real shifts: every interior site reads its channel."""
    h = w = 64
    rgb = np.empty((h, w, 3), np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.2, 0.5, 0.8
    mosaic = _mosaic_from_rgb(rgb)
    shifts = [_translation(*d) for d in
              [(0, 0), (2.5, 3.5), (-1.5, 4.0), (3.0, -2.0), (0.5, 0.5)]]
    stack, cov = cfa_lr_stack_at_held([mosaic] * len(shifts),
                                      shifts, np.eye(3), BGGR,
                                      [1.0] * len(shifts))
    inner = (slice(8, -8), slice(8, -8))
    for c, want in enumerate((0.2, 0.5, 0.8)):
        assert np.allclose(stack[c][inner], want, atol=1e-4), (c, stack[c].min())
    # The stack is coverage-normalised: no dark rim where only some frames
    # reached, because the accumulator is divided by the weight that did.
    r = stack[1]
    assert r[:8].min() > 0.45 and r[:, :8].min() > 0.45


def test_baseline_uses_only_the_fused_frames():
    """A wildly different held-out frame cannot move the baseline (no leak)."""
    h = w = 48
    fused = [np.full((h, w), 0.1, np.float32)]
    held = np.full((h, w), 0.9, np.float32)
    tr = [_translation(0, 0)]
    b1, _ = cfa_lr_stack_at_held(fused, tr, np.eye(3), BGGR, [1.0])
    # Same fused set, a second call: identical.  (The held-out mosaic is not an
    # argument of the baseline at all, which is the structural guarantee.)
    b2, _ = cfa_lr_stack_at_held(fused, tr, np.eye(3), BGGR, [1.0])
    for c in range(3):
        assert np.array_equal(b1[c], b2[c])
        assert b1[c][16:-16, 16:-16].mean() == pytest.approx(0.1, abs=1e-6)
    # The baseline PSNR is exactly what a 0.1 baseline gives against the 0.9
    # held-out truth (10*log10(1/0.8**2) = 1.94 dB).  Had the held-out frame
    # leaked into the baseline (a 50/50 mix of 0.1 and 0.9), it would read
    # ~9.5 dB instead.
    m = evaluate_cfa_heldout({"true": np.full((h * 2, w * 2, 3), 0.9,
                                              np.float32)},
                            held, fused, tr, [1.0], np.eye(3), 2, BGGR,
                            margin_lr=4)
    base_psnr = m["per_method"]["true"]["heldout_cfa_baseline_psnr_db"]
    assert base_psnr == pytest.approx(1.938, abs=0.2), base_psnr


# --------------------------------------------------------------------------- #
# Prediction + comparison
# --------------------------------------------------------------------------- #
def test_prediction_is_read_only_at_its_own_channel_sites():
    hr = np.zeros((32, 32, 3), np.float32)
    hr[..., 0], hr[..., 1], hr[..., 2] = 0.8, 0.5, 0.2
    planes, _sup = predict_cfa_from_hr(hr, np.eye(3), 2, BGGR, forward_sigma=0,
                                      lr_shape=(16, 16))
    ch = channel_map(BGGR, (16, 16))
    for c in range(3):
        m = ch == c
        assert np.allclose(planes[c][m], (0.8, 0.5, 0.2)[c], atol=1e-6)
        assert np.isnan(planes[c][~m]).all()


def test_mask_keeps_only_its_channel_inside_the_interior_margin():
    h, w, ratio = 64, 64, 2
    ch = channel_map(BGGR, (h, w))
    sup = np.ones((h, w), bool)
    cov = [np.full((h, w), 10.0) for _ in range(3)]
    masks = cfa_heldout_masks(BGGR, (h, w), sup, cov, total_weight=10.0,
                              margin_lr=8,
                              min_coverage_fraction=CFA_HELDOUT_MIN_COVERAGE_FRACTION)
    assert not masks[0][:8].any() and not masks[0][-8:].any()
    for c in range(3):
        assert masks[c].sum() == int(np.count_nonzero(ch[8:-8, 8:-8] == c))
    # A baseline that never reached the site is excluded everywhere.
    masks2 = cfa_heldout_masks(BGGR, (h, w), sup, [np.zeros((h, w))] * 3,
                               total_weight=10.0, margin_lr=8)
    assert sum(int(m.sum()) for m in masks2) == 0


def test_compare_scores_both_sides_on_identical_sites():
    h, w = 32, 32
    rng = np.random.default_rng(0)
    truth = rng.random((h, w)).astype(np.float32)
    ch = channel_map(BGGR, (h, w))
    close = [np.where(ch == c, truth, np.nan).astype(np.float32)
             for c in range(3)]
    far = [np.where(ch == c, truth + 0.1, np.nan).astype(np.float32)
           for c in range(3)]
    masks = [ch == c for c in range(3)]
    a = cfa_heldout_compare(close, far, truth, masks, label="close")
    b = cfa_heldout_compare(far, far, truth, masks, label="far")
    assert a["heldout_cfa_gain_db"] > 10.0
    assert b["heldout_cfa_gain_db"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Synthetic burst: ranking vs ground truth
# --------------------------------------------------------------------------- #
def _burst(ratio=2, lr=112, n=7, seed=5):
    rng = np.random.default_rng(seed)
    H = lr * ratio
    F = np.fft.fftshift(np.fft.fft2(rng.standard_normal((H, H))))
    fy = np.fft.fftshift(np.fft.fftfreq(H))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(H))[None, :]
    F *= 1.0 / (np.sqrt(fx ** 2 + fy ** 2) + 0.015) ** 0.85
    base = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    base = (base - base.mean()) / (base.std() + 1e-9)
    luma = np.clip(0.42 + 0.10 * base, 0.02, 0.92)
    scene = np.stack([luma * 0.96, luma, luma * 1.04], axis=2)
    for _ in range(18):
        x, y = rng.integers(4, H - 12, 2)
        a, b = int(rng.integers(3, max(4, H // 8))), int(
            rng.integers(3, max(4, H // 8)))
        col = rng.uniform(0.05, 0.9, 3)
        if rng.random() < 0.5:
            cv2.rectangle(scene, (x, y), (x + a, y + b), col.tolist(), -1)
        else:
            cv2.ellipse(scene, (x, y), (a // 2, b // 2),
                        float(rng.uniform(0, 180)), 0, 360, col.tolist(), -1)
    scene = np.clip(scene, 0.0, 1.0).astype(np.float32)
    blurred = cv2.GaussianBlur(scene, (0, 0), 0.55 * ratio,
                               borderType=cv2.BORDER_REFLECT)
    box = cv2.blur(blurred, (int(ratio), int(ratio)),
                   borderType=cv2.BORDER_REFLECT)
    offs = np.stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)], 1)
    offs[0] = (0.0, 0.0)
    xs, ys = np.meshgrid(np.arange(lr, dtype=np.float32),
                         np.arange(lr, dtype=np.float32))
    mos, rgbs = [], []
    off = 0.5 * (float(ratio) - 1.0)
    for k in range(n):
        mx = (ratio * (xs + np.float32(offs[k, 0])) + off).astype(np.float32)
        my = (ratio * (ys + np.float32(offs[k, 1])) + off).astype(np.float32)
        rgb = cv2.remap(box, mx, my, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT)
        m = _mosaic_from_rgb(rgb)
        m = (rng.poisson(np.clip(m, 0, 1) * 1023) +
             rng.normal(0, 1.5, m.shape)) / 1023.0
        mos.append(np.round(np.clip(m, 0, 1) * 1023) / 1023.0)
        rgbs.append(rgb)
    aligns = [_translation(-(offs[k, 0] - offs[0, 0]),
                           -(offs[k, 1] - offs[0, 1])) for k in range(n)]
    target = box
    margin = 2 * int(ratio) + 4
    mask = np.zeros(target.shape[:2], bool)
    mask[margin:-margin, margin:-margin] = True
    return mos, rgbs, target.astype(np.float32), mask, aligns


def test_metric_ranks_methods_like_ground_truth():
    ratio, holdout = 2, 3
    mos, rgbs, target, gt_mask, _aligns = _burst(ratio=ratio)
    kept_idx = [i for i in range(len(mos)) if i != holdout]
    kept_m = [mos[i] for i in kept_idx]
    dems = [_demosaic(m) for m in mos]
    frames = [RawFrame(str(k), d, d[:, :, 1].copy())
              for k, d in enumerate(dems)]
    kept = [frames[i] for i in kept_idx]
    res = SuperResolution(ratio=ratio, blur_threshold_ratio=0.0).run(
        kept, reference_index=0)
    w = res.weights.tolist()

    stack = denoised_lr_stack(kept, res.transforms, w,
                              coverage_normalised=True)
    stack_hr = cv2.resize(stack, (target.shape[1], target.shape[0]),
                          interpolation=cv2.INTER_CUBIC)
    orf = [RawFrame(str(k), rgbs[i], rgbs[i][:, :, 1].copy())
           for k, i in enumerate(kept_idx)]
    hr_or, oc = drizzle(orf, res.transforms, ratio, w)
    hr_or = iterative_back_projection(hr_or, orf, res.transforms, ratio, w,
                                      1, 0.5, 0.5, 0.1, coverage=oc)
    cfa = cfa_drizzle(kept_m, res.transforms, ratio, BGGR, w, pixfrac=1.0)
    est = {"stack": stack_hr, "pipeline": res.hr, "cfa": cfa.hr,
           "oracle": hr_or}
    gt = {k: _psnr(v, target, gt_mask) for k, v in est.items()}

    W = estimate_homography(kept[0].green, frames[holdout].green,
                            ecc_scale=0.25)
    m = evaluate_cfa_heldout(est, mos[holdout], kept_m, res.transforms, w, W,
                             ratio, BGGR)
    gain = {k: v["heldout_cfa_gain_db"] for k, v in m["per_method"].items()}

    # Ground truth: the LR stack is worst, the full-RGB oracle best, and the
    # two multi-frame reconstructions sit strictly between them.
    assert gt["stack"] < gt["oracle"]
    assert gt["oracle"] > gt["cfa"] > gt["stack"]
    # The metric must reproduce that ordering (it is the acceptance criterion
    # for the metric itself, not for the reconstructions).
    assert (sorted(gt, key=lambda k: gt[k])
            == sorted(gain, key=lambda k: gain[k])), (gt, gain)
    assert gain["stack"] < gain["pipeline"] < gain["cfa"] < gain["oracle"]
    # And the sites are a real share of the frame, per channel in proportion.
    sites = m["mask"]["sites_per_channel"]
    assert sites[1] > sites[0] > 0 and sites[1] > sites[2] > 0


def test_baseline_psf_diagnostic_is_a_small_correction():
    ratio, holdout = 2, 2
    mos, _rgbs, target, _gt_mask, _al = _burst(ratio=ratio, lr=96, n=6,
                                               seed=7)
    kept_idx = [i for i in range(len(mos)) if i != holdout]
    kept_m = [mos[i] for i in kept_idx]
    dems = [_demosaic(m) for m in mos]
    frames = [RawFrame(str(k), d, d[:, :, 1].copy())
              for k, d in enumerate(dems)]
    res = SuperResolution(ratio=ratio, blur_threshold_ratio=0.0).run(
        [frames[i] for i in kept_idx], reference_index=0)
    W = estimate_homography(frames[kept_idx[0]].green,
                            frames[holdout].green, ecc_scale=0.25)
    base = evaluate_cfa_heldout({"hr": res.hr}, mos[holdout], kept_m,
                                res.transforms, res.weights.tolist(), W,
                                ratio, BGGR)
    psf = evaluate_cfa_heldout({"hr": res.hr}, mos[holdout], kept_m,
                               res.transforms, res.weights.tolist(), W,
                               ratio, BGGR, baseline_psf=True)
    d = abs(base["per_method"]["hr"]["heldout_cfa_gain_db"]
            - psf["per_method"]["hr"]["heldout_cfa_gain_db"])
    assert d < 0.1, d


# --------------------------------------------------------------------------- #
# Acceptance-script wiring
# --------------------------------------------------------------------------- #
def test_acceptance_cli_exposes_the_cfa_heldout_flag_off_by_default():
    """The CFA-space metric is opt-in and must not move the default run."""
    import importlib.util
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "sr_acceptance_b6", os.path.join(root, "scripts", "sr_acceptance.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ap = mod.build_parser()
    a = ap.parse_args([])
    assert a.cfa_heldout is False
    assert a.cfa_heldout_pixfrac == 0.5
    assert a.cfa_heldout_margin == 24
    assert a.cfa_heldout_min_coverage == CFA_HELDOUT_MIN_COVERAGE_FRACTION
    b = ap.parse_args(["--cfa-heldout", "--cfa-heldout-pixfrac", "1.0",
                       "--cfa-heldout-margin", "16"])
    assert b.cfa_heldout is True and b.cfa_heldout_pixfrac == 1.0
    assert b.cfa_heldout_margin == 16
