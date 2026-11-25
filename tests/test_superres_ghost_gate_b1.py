"""B1: ghost gate robustness to motion the global homography absorbs.

``SuperResolution.robust_fusion=None`` (AUTO) used to decide on one statistic:
the per-pixel peak deviation from the across-frame median, normalized by a
single global MAD (:func:`_ghost_gate_dispersion`). That is blind exactly where
the review said it would be:

* a subject covering a large share of the frame (or a high-contrast one) drags
  the two-pass homography and inflates the MAD, so the *tail* statistic falls
  below the threshold;
* with two frames the "median" is the mean, which halves every deviation, so
  even a small subject is missed.

These tests pin the added detectors on the *same* warped stack (no extra
warp): the population/blob statistic (:func:`_ghost_gate_population`), the
contrast-normalized inter-frame spread (:func:`_ghost_gate_spread`) and, since
B8, the registration-residual branch D4 (a burst whose frames no single global
warp explains). They also pin the memory win: the gate's warped stack is
reused by the ghost-energy metric, so the burst is gathered to the HR grid once
per run, bit-identically.

The sensor model is band-limited on purpose: the HR scene is shifted in
continuous space and box-averaged 2x2 to LR, and the moving layer is painted in
HR space per frame. A *static* burst therefore really is fusion-consistent,
which is what makes "static stays OFF" a meaningful assertion.
"""


import numpy as np
import cv2
import pytest

import raw_sr.core.superres as SR
from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    _auto_robust_fusion,
    _ghost_gate_spread,
    _interframe_ghost_energy,
    _warp_frames_to_hr,
    load_raw_linear,
    _GHOST_GATE_POP_SHARE_MIN,
    _GHOST_GATE_POP_COMPACT,
    _GHOST_GATE_POP_EXTENT,
    _GHOST_GATE_FLOOD_RATIO,
    _GHOST_GATE_FLOOD_STRONG,
    _GHOST_GATE_FLOOD_COMPACT,
)

from tests.data_paths import REAL_DATA_HELP, dng_files, missing_reason

REAL_DATA = dng_files(9)

RATIO = 2.0


# --------------------------------------------------------------------------- #
# Synthetic bursts (band-limited sensor, HR-painted moving layer)
# --------------------------------------------------------------------------- #

def _scene(size, seed, contrast=1.0):
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.float32)
    for sig, amp in ((9.0, 1.0), (4.0, 0.55), (1.6, 0.30), (0.7, 0.12)):
        n = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32),
                             (0, 0), sig)
        n = (n - n.mean()) / (n.std() + 1e-9)
        img += amp * n
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    lo = 0.12 + 0.72 * (1.0 - contrast) / 2.0
    return (lo + 0.72 * contrast * img).astype(np.float32)


def _subject_patch(sw, sh, seed):
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.random((sh, sw)).astype(np.float32), (0, 0), 1.6)
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-9)
    p = 0.30 + 0.62 * tex
    p[:2, :] = 0.95; p[-2:, :] = 0.15
    p[:, :2] = 0.95; p[:, -2:] = 0.15
    return p.astype(np.float32)


def _burst(n=9, subject="small", seed=7, noise=0.0035, jitter=1.6,
           sweep=None, subject_side=34, size=384, layer_scale=None,
           allow_outside=False, contrast=1.0, exposure_drift=0.0,
           exposure_step=None):
    """Returns ``(frames, truth_rgb, trail)``.

    ``subject="none"`` is the static control; ``"small"`` paints a textured
    patch of ``subject_side`` HR px that translates between frames;
    ``layer_scale`` paints a closer depth layer magnified about the centre
    (parallax — it follows the camera but a different homography applies).
    """
    h = w = int(size)
    lh, lw = h // 2, w // 2
    rng = np.random.default_rng(seed + 501)
    base = _scene(size, seed, contrast=contrast)
    offs = jitter * (rng.random((n, 2)) - 0.5) * 2.0
    sw = sh = subject_side if subject != "none" else 0
    if sweep is None:
        sweep = max(26.0, 0.75 * sw)
    if not allow_outside:
        sweep = min(sweep, max(2.0, w * 0.5 - sw * 0.75 - 6))
    cxs = np.linspace(w * 0.5 - sweep, w * 0.5 + sweep, n)
    cy = h * 0.5
    patch = _subject_patch(sw, sh, seed + 9) if sw else None
    out = []
    for i in range(n):
        gx, gy = offs[i]
        M = np.array([[1, 0, -gx], [0, 1, -gy]], np.float32)
        img = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        if layer_scale is not None:
            M2 = np.array(
                [[layer_scale, 0, w * 0.5 * (1 - layer_scale) - gx * layer_scale],
                 [0, layer_scale, h * 0.5 * (1 - layer_scale) - gy * layer_scale]],
                np.float32)
            near = cv2.warpAffine(base, M2, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
            mk = np.zeros((h, w), np.float32)
            mk[int(h * 0.12):int(h * 0.88), int(w * 0.10):int(w * 0.90)] = 1.0
            img = img * (1 - mk) + near * mk
        if sw:
            if allow_outside:
                # A subject wider than the margin it travels: clip the paint to
                # the frame instead of pinning it inside (the flooded regime).
                x0 = int(round(cxs[i] - sw / 2))
                y0 = int(round(cy - sh / 2))
                xa, ya = max(0, x0), max(0, y0)
                xb, yb = min(w, x0 + sw), min(h, y0 + sh)
                if xb > xa and yb > ya:
                    img = img.copy()
                    img[ya:yb, xa:xb] = patch[ya - y0:yb - y0,
                                              xa - x0:xb - x0]
            else:
                x0 = int(np.clip(round(cxs[i] - sw / 2), 0, w - sw))
                y0 = int(np.clip(round(cy - sh / 2), 0, h - sh))
                img = img.copy()
                img[y0:y0 + sh, x0:x0 + sw] = patch
        if exposure_step is not None and i == int(exposure_step):
            img = img * 1.05
        if exposure_drift:
            img = img * (1.0 + exposure_drift * (2.0 * i / max(1, n - 1) - 1.0))
        lr = cv2.resize(np.clip(img, 0, 1), (lw, lh),
                        interpolation=cv2.INTER_AREA)
        lr = np.clip(lr + rng.normal(0, noise, lr.shape).astype(np.float32),
                     0, 1)
        rgb = np.stack([lr] * 3, axis=2).astype(np.float32)
        out.append(RawFrame(name=f"s{i}", rgb=rgb, green=rgb[:, :, 1].copy()))
    truth = base.copy()
    if sw:
        x0 = int(np.clip(round(w * 0.5 - sw / 2), 0, w - sw))
        y0 = int(np.clip(round(cy - sh / 2), 0, h - sh))
        truth[y0:y0 + sh, x0:x0 + sw] = patch
    trail = np.zeros((h, w), bool)
    if sw:
        lo = int(w * 0.5 - sweep - sw / 2)
        hi = int(w * 0.5 + sweep + sw / 2)
        x0 = int(np.clip(round(w * 0.5 - sw / 2), 0, w - sw))
        trail[:, max(0, lo):max(0, x0)] = True
        trail[:, x0 + sw:min(w, hi)] = True
    return out, np.stack([truth] * 3, axis=2).astype(np.float32), trail


def _run(frames, **kw):
    kw.setdefault("use_gpu_scatter", False)
    # These tests cover the demosaic_drizzle robust-fusion gate; pin it (the
    # default became fusion="auto" -> map on GPU, which is not bitwise deterministic).
    kw.setdefault("fusion", "demosaic_drizzle")
    return SuperResolution(ratio=RATIO, blur_threshold_ratio=0.0, **kw).run(frames)


def _trail_error(hr, truth, trail, margin=12):
    interior = np.ones_like(trail)
    interior[:margin] = interior[-margin:] = False
    interior[:, :margin] = interior[:, -margin:] = False
    sel = trail & interior
    if not sel.any():
        return float("nan")
    return float(np.abs(hr - truth).mean(axis=2)[sel].mean())


# --------------------------------------------------------------------------- #
# Frozen contract: a static burst is OFF and byte-identical to plain fusion
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [9, 3, 2])
def test_static_synthetic_burst_auto_off_and_bit_identical(n):
    frames, _truth, _trail = _burst(n=n, subject="none")
    auto = _run(frames, robust_fusion=None)
    off = _run(frames, robust_fusion=False)
    assert auto.metrics["robust_active"] is False
    assert auto.metrics["robust_reason"].startswith("static burst")
    assert np.array_equal(auto.hr, off.hr)
    assert auto.metrics["hf_gain_db"] == off.metrics["hf_gain_db"]
    assert auto.metrics["sharpness_ratio"] == off.metrics["sharpness_ratio"]
    assert auto.metrics["ghost_energy"] == off.metrics["ghost_energy"]


# --------------------------------------------------------------------------- #
# Issue #2: motion the homography absorbs is still detected
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [9, 3, 2])
def test_small_moving_subject_detected_at_every_burst_length(n):
    frames, truth, trail = _burst(n=n, subject="small")
    res = _run(frames, robust_fusion=None)
    assert res.metrics["robust_active"] is True, res.metrics["robust_reason"]
    assert "moving subject detected" in res.metrics["robust_reason"]
    if n == 2:  # two frames cannot be separated by a robust median: say so
        assert "2-frame" in res.metrics["robust_reason"]
    # The gate must never make the trail worse, and on N>=3 it must improve it.
    off = _run(frames, robust_fusion=False)
    e_on = _trail_error(res.hr, truth, trail)
    e_off = _trail_error(off.hr, truth, trail)
    assert e_on <= e_off * 1.001, (e_on, e_off)
    if n >= 3:
        assert e_on < e_off


def test_dominant_subject_detected_even_with_a_diffuse_residual():
    # ~33% of the frame area. Before B8 this was carried by D2's population
    # statistic; the corrected ECC seed (B8) makes the fit globally diffuse, so
    # D2's mask is empty (share 0.003) and the registration-residual branch D4
    # carries the burst instead. What must hold either way: the burst is
    # detected and robust fusion does not make the trail worse.
    frames, truth, trail = _burst(n=9, subject="dominant", subject_side=220)
    res = _run(frames, robust_fusion=None)
    assert res.metrics["robust_active"] is True, res.metrics["robust_reason"]
    assert "moving subject detected" in res.metrics["robust_reason"]
    assert res.metrics["align_residual_mean_px"] > SR._GHOST_GATE_RESIDUAL_MAX
    off = _run(frames, robust_fusion=False)
    assert _trail_error(res.hr, truth, trail) <= _trail_error(off.hr, truth, trail)


def test_flooded_subject_over_half_the_frame_detected():
    # ~74% of the area, traversing the frame: D1's tail is diluted and the
    # tile floor is lifted by the subject itself. With B8's corrected ECC seed
    # the mask is one large blob whose bbox fill sits below D2's extent cut, so
    # the contrast-normalized spread branch (D3) carries the decision
    # (measured ratio 0.127 of scene contrast, coherent 1.00) — exactly what
    # D3 was added for. (The N=2 flooded case is pinned in the B2 test file.)
    frames, truth, _trail = _burst(n=9, subject="dominant", subject_side=760,
                                   size=1024, sweep=60.0, allow_outside=True)
    res = _run(frames, robust_fusion=None)
    assert res.metrics["robust_active"] is True, res.metrics["robust_reason"]
    assert "moving subject detected" in res.metrics["robust_reason"]
    kept = [frames[i] for i in res.kept_indices]
    S, V, _oh, _ow = _warp_frames_to_hr(kept, res.transforms, RATIO)
    spread = _ghost_gate_spread(S, V)
    assert spread["ratio"] >= SR._GHOST_GATE_FLOOD_RATIO


@pytest.mark.parametrize("n", [9, 3])
def test_parallax_layer_detected_without_quality_regression(n):
    # A closer depth layer follows the camera magnified, so one homography
    # cannot align both layers — structure, not an independently moving object.
    frames, truth, _trail = _burst(n=n, subject="none", layer_scale=1.9,
                                  jitter=6.0)
    res = _run(frames, robust_fusion=None)
    assert res.metrics["robust_active"] is True, res.metrics["robust_reason"]
    off = _run(frames, robust_fusion=False)
    mse_on = float(np.mean((res.hr - truth) ** 2))
    mse_off = float(np.mean((off.hr - truth) ** 2))
    assert mse_on <= 1.01 * mse_off, (mse_on, mse_off)


def test_gate_thresholds_sit_in_the_measured_gap():
    # Measured populations (see the threshold notes in superres.py, B2 grid):
    # the worst static real crop reaches share 0.313 with compact 0.34-0.73 /
    # extent 0.20-0.68, and spread 0.048; a moving layer is one blob with
    # compact 0.99-1.00, and a parallax layer reaches spread 0.047-0.069 with a
    # broad (compact 0.39-0.55) mask. Thresholds must sit inside those gaps.
    assert 0.02 <= _GHOST_GATE_POP_SHARE_MIN < 0.25
    assert 0.73 < _GHOST_GATE_POP_COMPACT < 0.99
    assert 0.20 < _GHOST_GATE_POP_EXTENT < 0.65
    assert 0.040 <= _GHOST_GATE_FLOOD_RATIO < _GHOST_GATE_FLOOD_STRONG
    assert _GHOST_GATE_FLOOD_STRONG < 0.15
    assert 0.20 < _GHOST_GATE_FLOOD_COMPACT < 0.60


# --------------------------------------------------------------------------- #
# Issue #4: the gate's warped stack is reused instead of re-warping
# --------------------------------------------------------------------------- #

def test_auto_run_warps_the_burst_once(monkeypatch):
    calls = {"n": 0}
    real_warp = SR._warp_frames_to_hr

    def counting(*a, **kw):
        calls["n"] += 1
        return real_warp(*a, **kw)

    monkeypatch.setattr(SR, "_warp_frames_to_hr", counting)
    frames, _truth, _trail = _burst(n=3, subject="small")
    res = _run(frames, robust_fusion=None)
    # gate (1) + ghost-energy metric (reused) = a single gather for the run.
    assert calls["n"] == 1
    assert np.isfinite(res.metrics["ghost_energy"])


def test_ghost_energy_is_identical_with_the_reused_stack():
    frames, _truth, _trail = _burst(n=5, subject="small")
    res = _run(frames, robust_fusion=None)
    kept = [frames[i] for i in res.kept_indices]
    S, V, _oh, _ow = _warp_frames_to_hr(kept, res.transforms, RATIO)
    fresh = _interframe_ghost_energy(kept, res.transforms, RATIO)
    reused = _interframe_ghost_energy(kept, res.transforms, RATIO, stack=(S, V))
    assert fresh == reused
    assert res.metrics["ghost_energy"] == reused["ghost_energy"]
    assert res.metrics["interframe_std"] == reused["interframe_std"]


def test_two_frame_ghost_energy_is_structurally_zero():
    # Documented caveat: with N=2 the across-frame median is the mean, so the
    # reported ghost_energy is exactly 0 while interframe_std still carries the
    # disagreement. Anything reading ghost_energy must not treat N=2 as proof.
    frames, _truth, _trail = _burst(n=2, subject="small")
    res = _run(frames, robust_fusion=None)
    assert res.metrics["ghost_energy"] == 0.0
    assert res.metrics["interframe_std"] > 0.0


# --------------------------------------------------------------------------- #
# Real static burst: still OFF at the test ROI and at the acceptance ROI
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(len(REAL_DATA) < 9, reason=missing_reason(9))
@pytest.mark.parametrize("roi", [(1800, 1224, 512, 512),
                                 (1650, 1240, 1024, 1024)])
def test_real_static_burst_stays_off(roi):
    frames = [load_raw_linear(f, roi=roi) for f in REAL_DATA]
    auto = _run(frames, robust_fusion=None, ecc_scale=0.25)
    assert auto.metrics["robust_active"] is False, auto.metrics["robust_reason"]
    assert auto.metrics["robust_reason"] == "static burst \u2014 plain fusion"
    off = _run(frames, robust_fusion=False, ecc_scale=0.25)
    assert np.array_equal(auto.hr, off.hr)
    assert auto.metrics["hf_gain_db"] == off.metrics["hf_gain_db"]
