"""B8 — ground-truth tests for the registration / reference-selection sign bugs.

Three defects shared one root cause: :func:`estimate_translation` returns a
*mov->ref resampling* offset ``d`` (``mov(x) ~= ref(x + d)``), while every
alignment matrix in the core is a *ref->mov* warp ``W`` (``mov(W(p)) ~= ref(p)``
== ``translation(-d)``).

* #1 ``registration="translation"`` deposited the frame shifted by ``2 d``
  (probe: known-shift MSE 4.95e-3 vs 2.2e-4 for ECC).
* #2 ``_select_reference`` scored candidates with the flipped sign, so a frame
  displaced by more than ~2 px scored ~0.01-0.2 and was disqualified.
* (seed) ``estimate_homography`` started every ECC solve 2*shift away.

Every test here uses a KNOWN synthetic shift, so the expected answer is not the
implementation's own output: content shift, ``W`` sign, rendered MSE, and the
end-to-end HR MSE against the known truth.
"""

import numpy as np
import cv2
import pytest

from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    estimate_translation,
    estimate_homography,
    translation_matrix,
    _alignment_score,
    _register_frame,
    _select_reference,
)


# --------------------------------------------------------------------------- #
# Ground-truth burst: LR(x) = truth(r*(x + shift) + (r-1)/2)
# --------------------------------------------------------------------------- #
#
# The ``(r-1)/2`` is the pixel-centre convention of the HR grid (see
# :func:`superres.to_drizzle_matrix`): without it the fixture yields a truth
# 0.5 HR px away from the reconstruction and a direct MSE comparison reads the
# offset instead of quality (review round 2, N1 — the fix/grid15 change was
# never applied to THIS fixture, which left four of these tests red).

def _truth(h=192, w=192, seed=5, sigma=0.8):
    rng = np.random.default_rng(seed)
    t = rng.random((h, w, 3)).astype(np.float32)
    return cv2.GaussianBlur(t, (0, 0), sigma)


def _lr(truth, shift, ratio=2, noise=0.0, seed=1):
    h, w = truth.shape[0] // ratio, truth.shape[1] // ratio
    off = 0.5 * (ratio - 1.0)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    img = cv2.remap(truth,
                    ratio * (xs + shift[0]) + off,
                    ratio * (ys + shift[1]) + off,
                    cv2.INTER_LINEAR)
    if noise:
        rng = np.random.default_rng(seed)
        img = np.clip(img + rng.normal(0, noise, img.shape).astype(np.float32),
                      0, 1)
    return img


def _frame(truth, shift, **kw):
    img = _lr(truth, shift, **kw)
    return RawFrame(name="f", rgb=img, green=img[:, :, 1].copy())


def _burst(truth, shifts, **kw):
    return [_frame(truth, s, seed=i + 1, **kw) for i, s in enumerate(shifts)]


# Non-integer, both axes, both signs. Two are beyond the ~2 px point where the
# old reference score collapsed (3.1, 2.2) / (-2.2, 1.4).
KNOWN_SHIFTS = [(1.3, -0.7), (3.1, 2.2), (-2.2, 1.4), (0.35, 0.0), (0.0, -1.75)]

def _mode_shift_params(shifts):
    """(mode, shift) pairs: every shift for every registration mode.

    Both modes share one ground truth, so the same content shift must land at
    the same place whichever model produced the warp — that is what caught the
    translation-mode sign bug (B8 #1)."""
    return [(mode, shift) for shift in shifts
            for mode in ("translation", "homography")]


# --------------------------------------------------------------------------- #
# #1 — a deposited frame must land where the scene says it is
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode,shift", _mode_shift_params(KNOWN_SHIFTS))
def test_register_frame_deposits_with_the_true_shift(mode, shift):
    truth = _truth()
    ref = _frame(truth, (0.0, 0.0))
    mov = _frame(truth, shift)

    _d, _a, W, score, residual = _register_frame(
        ref.green, ref.green, mov.green, False, mode, 0.5, False)

    # The ref->mov warp of a content shift s is translation(-s): a feature at
    # ref p sits at p - s in mov.
    assert np.allclose([W[0, 2], W[1, 2]], [-shift[0], -shift[1]], atol=0.15), (
        f"{mode} {shift}: W translation {(W[0, 2], W[1, 2])} != {-shift}")
    # ... and rendering mov through W must reproduce ref (the old pipeline
    # rendered it at 2*s and measured MSE ~2e-2 here).
    Mi = np.linalg.inv(W).astype(np.float32)
    h, w = ref.green.shape
    aligned = cv2.warpPerspective(mov.green, Mi, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
    ones = cv2.warpPerspective(np.ones((h, w), np.float32), Mi, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
    sel = ones > 0.99
    mse = float(np.mean((aligned[sel] - ref.green[sel]) ** 2))
    assert mse < 5e-3, f"{mode} {shift}: rendered MSE {mse:.2e} (ghosted)"
    assert residual < 0.15, f"{mode} {shift}: residual {residual:.3f} px"
    # The synthetic truth is mildly aliased at 2x, so a perfectly registered
    # pair measures 0.80-0.94 here (not 1.0).
    assert score > 0.75, f"{mode} {shift}: alignment score {score:.3f}"


@pytest.mark.parametrize("mode,shift", _mode_shift_params(
    [(1.3, -0.7), (3.1, 2.2), (-2.2, 1.4)]))
def test_full_pipeline_mse_matches_truth_in_every_mode(mode, shift):
    """End-to-end: beat the flipped-sign deposit that the bug produced."""
    truth = _truth()
    frames = [_frame(truth, (0.0, 0.0)), _frame(truth, shift)]
    res = SuperResolution(ratio=2.0, registration=mode,
                          blur_threshold_ratio=0.0,
                          ibp_iterations=0).run(frames, reference_index=0)
    assert res.reference_index == 0
    assert res.used_frames == 2

    err = float(np.mean((res.hr - truth) ** 2))
    # Same deposit with the (old, wrong) unnegated translation.
    W_bad = translation_matrix(*estimate_translation(*[
        f.green for f in frames]))
    hr_bad, _ = _drizzle_bad(frames, W_bad)
    err_bad = float(np.mean((hr_bad - truth) ** 2))
    assert err < 0.5 * err_bad, (
        f"{mode} {shift}: SR MSE {err:.2e} not clearly better than the "
        f"flipped-sign deposit {err_bad:.2e}")
    assert err < 4e-3, f"{mode} {shift}: SR MSE {err:.2e}"


def _drizzle_bad(frames, W_bad):
    from raw_sr.core.superres import drizzle
    return drizzle(frames, [np.eye(3), W_bad], ratio=2.0)


# --------------------------------------------------------------------------- #
# #2 — reference selection must not disqualify a correctly registered frame
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shift", KNOWN_SHIFTS)
def test_alignment_score_needs_the_negated_shift(shift):
    truth = _truth()
    ref = _frame(truth, (0.0, 0.0))
    mov = _frame(truth, shift)
    dx, dy = estimate_translation(ref.green, mov.green)
    s_ok = _alignment_score(ref.green, mov.green,
                            translation_matrix(-dx, -dy))
    s_bad = _alignment_score(ref.green, mov.green, translation_matrix(dx, dy))
    assert s_ok > 0.75, f"{shift}: correct sign scored only {s_ok:.3f}"
    # The flipped matrix renders the frame at 2*d. For a sub-pixel shift that
    # is still close (0.7 px at shift 0.35), so the gap only has to be clear;
    # from ~1 px on (4 tests) it must be decisive.
    margin = 0.15 if max(abs(dx), abs(dy)) >= 1.0 else 0.05
    assert s_ok > s_bad + margin, (
        f"{shift}: correct {s_ok:.3f} vs flipped {s_bad:.3f} — the score does "
        f"not depend on the sign (regression)")


def test_select_reference_keeps_a_shifted_sharp_frame():
    """The sharpest candidate is 3.1/2.2 px off the mid frame; it must stay a
    candidate (the old sign scored it 0.027 and rejected it)."""
    truth = _truth(seed=11)
    # Frame 2 sits 3.1/2.2 px from frame 0, ~2.9 px from the mid frame: with the
    # flipped sign it scored -0.008 (measured) and was dropped as a candidate.
    frames = _burst(truth, [(0.0, 0.0), (0.2, 0.1), (3.1, 2.2)])
    weights = np.array([1.0, 1.0, 1.6])   # frame 2 is the sharpest
    stats: dict = {}
    idx = _select_reference(frames, weights, [0, 1, 2], stats=stats)
    assert stats["rejected"] == [], f"correctly shifted frame rejected: {stats}"
    assert stats["scores"][2] > 0.9, stats["scores"]
    assert idx in (0, 1, 2)


def test_select_reference_still_rejects_a_junk_candidate():
    truth = _truth(seed=12)
    frames = _burst(truth, [(0.0, 0.0), (0.5, 0.5), (0.0, 0.5)])
    rng = np.random.default_rng(3)
    junk = RawFrame(name="junk", rgb=rng.random(frames[0].rgb.shape,
                                               dtype=np.float32),
                    green=rng.random(frames[0].green.shape, dtype=np.float32))
    frames.append(junk)
    stats: dict = {}
    idx = _select_reference(frames, np.ones(4), [0, 1, 2, 3], stats=stats)
    assert stats["rejected"] == [3]
    assert idx != 3


def test_reference_selection_reports_its_consistency_scores():
    truth = _truth(seed=13)
    frames = _burst(truth, [(0.0, 0.0), (0.4, 0.0)])
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                          ibp_iterations=0).run(frames)
    assert len(res.metrics["reference_consistency_scores"]) == 2
    assert res.metrics["reference_consistency_rejected"] == []


# --------------------------------------------------------------------------- #
# ECC seed sign: with no iterations the estimator must return its seed, and
# that seed must be the ref->mov translation (negative of the estimate).
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shift", [(1.3, -0.7), (-2.2, 1.4)])
def test_estimate_homography_seed_is_the_negated_translation(shift):
    truth = _truth()
    ref = _frame(truth, (0.0, 0.0))
    mov = _frame(truth, shift)
    d = estimate_translation(ref.green, mov.green)
    W = estimate_homography(ref.green, mov.green, seed=d, iters=0)
    assert np.allclose([W[0, 2], W[1, 2]], [-d[0], -d[1]], atol=0.06), (
        f"seed {d} became W translation {(W[0, 2], W[1, 2])} — the ECC seed is "
        f"flipped, so every solve starts 2*shift away")
    assert _alignment_score(ref.green, mov.green, W) > 0.75


def test_estimate_homography_converges_for_a_large_shift():
    """A 12 px shift is far outside a 2*shift-off seed's capture range."""
    truth = _truth(seed=21)
    ref = _frame(truth, (0.0, 0.0))
    mov = _frame(truth, (12.0, -9.0))
    W = estimate_homography(ref.green, mov.green, ecc_scale=0.5)
    assert np.allclose([W[0, 2], W[1, 2]], [-12.0, 9.0], atol=0.3), (
        W[:2, 2])
    assert _alignment_score(ref.green, mov.green, W) > 0.9


# --------------------------------------------------------------------------- #
# Frame acceptance: the default gate keeps registered frames, drops noise, and
# reports both in the metrics. It must NOT be so high that a correctly
# registered above-Nyquist burst loses frames (the measured reason it is 0.05
# and not 0.3).
# --------------------------------------------------------------------------- #

def test_default_gate_keeps_registered_frames_and_drops_noise():
    # 192 LR px: the smallest frame on which the measured junk distribution
    # (max 0.032) and the aliased-content distribution do not overlap, so the
    # absolute gate is active rather than advisory.
    truth = _truth(h=384, w=384, seed=31)
    frames = _burst(truth, [(0.0, 0.0), (0.4, 0.2), (0.8, 0.6)],
                    noise=0.002)
    rng = np.random.default_rng(7)
    junk = RawFrame(name="junk", rgb=rng.random(frames[0].rgb.shape,
                                               dtype=np.float32),
                    green=rng.random(frames[0].green.shape, dtype=np.float32))
    frames.append(junk)

    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                          ibp_iterations=0).run(frames)
    assert 3 not in res.kept_indices, "white-noise frame was silently fused"
    assert res.kept_indices == [0, 1, 2]
    m = res.metrics
    assert m["min_align_score"] == 0.1
    assert m["align_gate_mode"] == "absolute"
    assert m["align_gate_threshold"] == 0.1
    assert [d["index"] for d in m["align_gate_dropped"]] == [3]
    assert m["align_gate_dropped"][0]["score"] < 0.1
    assert "dropped" in m["align_gate_reason"]


def test_gate_relaxes_to_the_burst_median_on_aliased_content():
    """A burst whose content is entirely above the LR Nyquist scores only
    ~0.11-0.15 even with perfect registration; a fixed 0.1 sits inside that
    population. The resolved threshold is min(0.1, 0.25*median), which keeps
    every correctly registered frame (N2)."""
    rng = np.random.default_rng(17)
    n_hr = 512                      # 256 LR px: gate is active, not advisory
    hr = rng.random((n_hr, n_hr)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(hr))
    f = np.fft.fftshift(np.fft.fftfreq(n_hr))
    r = np.sqrt(f[None, :] ** 2 + f[:, None] ** 2)
    F[(r < 0.26) | (r > 0.50)] = 0
    g = np.real(np.fft.ifft2(np.fft.ifftshift(F))).astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    truth = np.stack([g, g, g], axis=2)

    shifts = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5),
              (0.25, 0.75), (0.75, 0.25)]
    frames = _burst(truth, shifts, noise=3e-3)
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                          ibp_iterations=0).run(frames)
    m = res.metrics
    assert m["align_gate_mode"] in ("relative", "absolute"), m["align_gate_mode"]
    # Registered frames must not be decimated by the absolute 0.1.
    assert res.used_frames >= len(frames) - 1, (res.kept_indices,
                                                m["align_gate_dropped"])
    if m["align_gate_mode"] == "relative":
        assert m["align_gate_threshold"] < 0.1
        assert m["align_gate_threshold"] == pytest.approx(
            0.25 * float(np.median([
                s for s in res.align_scores if s < 1.0])), rel=0.35)


def test_gate_is_advisory_below_192_px():
    """Below 192 px the measured junk floor overlaps the aliased-content
    population at every threshold, so the gate drops nothing and says so."""
    truth = _truth(h=192, w=192, seed=33)          # 96 LR px
    frames = _burst(truth, [(0.0, 0.0), (0.4, 0.2)])
    rng = np.random.default_rng(9)
    junk = RawFrame(name="junk", rgb=rng.random(frames[0].rgb.shape,
                                               dtype=np.float32),
                    green=rng.random(frames[0].green.shape, dtype=np.float32))
    frames.append(junk)
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                          ibp_iterations=0).run(frames)
    m = res.metrics
    assert m["align_gate_mode"] == "advisory"
    assert res.used_frames == 3, m["align_gate_reason"]
    assert m["align_gate_dropped"] == []
    assert [d["index"] for d in m["align_gate_advisory"]] == [2]
    assert m["align_gate_reason"].startswith("advisory")


def test_default_gate_does_not_collapse_an_above_nyquist_burst():
    # A burst whose content lives above the LR Nyquist scores only ~0.09-0.16
    # even when perfectly registered, so a 0.3 default would drop good data.
    # At 192 LR px the gate is active (not advisory) and the relative floor
    # must nevertheless keep the whole correctly registered burst.
    rng = np.random.default_rng(7)
    hr = rng.random((384, 384)).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(hr))
    fy = np.fft.fftshift(np.fft.fftfreq(384))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(384))[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2)
    F[(r < 0.30) | (r > 0.50)] = 0
    g = np.real(np.fft.ifft2(np.fft.ifftshift(F))).astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    truth = np.stack([g, g, g], axis=2)

    frames = _burst(truth, [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)],
                    noise=0.001)
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                          ibp_iterations=0).run(frames)
    assert res.metrics["align_gate_mode"] in ("relative", "absolute")
    assert res.used_frames >= 3, res.metrics["align_gate_dropped"]


def test_min_align_score_zero_still_disables_the_gate():
    truth = _truth(seed=32)
    frames = _burst(truth, [(0.0, 0.0), (0.4, 0.2)])
    rng = np.random.default_rng(8)
    frames.append(RawFrame(name="junk",
                           rgb=rng.random(frames[0].rgb.shape,
                                          dtype=np.float32),
                           green=rng.random(frames[0].green.shape,
                                            dtype=np.float32)))
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0,
                          min_align_score=0.0, ibp_iterations=0).run(frames)
    assert res.used_frames == 3


# --------------------------------------------------------------------------- #
# Real burst (skipped without the DNGs): the default gate must keep all 9 real
# frames at ROI 512 — the small-ROI case where the old reference score collapsed
# on the largest shifts (up to 17 px) — and report, not silently fuse, junk.
# --------------------------------------------------------------------------- #

from tests.data_paths import dng_files, missing_reason  # noqa: E402

REAL_DATA = dng_files(9)


@pytest.mark.skipif(len(REAL_DATA) < 9, reason=missing_reason(9))
def test_real_burst_default_gate_keeps_every_frame():
    from raw_sr.core.superres import load_raw_linear
    roi = (1650, 1240, 512, 512)
    frames = [load_raw_linear(f, roi=roi) for f in REAL_DATA[:9]]
    res = SuperResolution(ratio=2.0, ibp_iterations=0).run(frames)
    assert res.used_frames == 9, (res.kept_indices,
                                  res.metrics["align_gate_dropped"])
    assert res.metrics["align_gate_dropped"] == []
    assert res.metrics["reference_consistency_rejected"] == [], (
        "a correctly registered real frame was rejected as reference candidate")
    assert min(res.align_scores) > 0.9, res.align_scores
    assert res.metrics["align_gate_reason"].startswith("no frame dropped")


# --------------------------------------------------------------------------- #
# N3 — D4 must use a robust statistic: one failed lock cannot force robust
# fusion ON for an otherwise static burst. The median requires that at least
# half of the registered frames disagree with a single global warp.
# --------------------------------------------------------------------------- #

def test_residual_branch_ignores_one_bad_lock():
    from raw_sr.core.superres import _ghost_gate_residual_moving
    # 8 frames, seven locked (0.04 px), one at 20 px: the MEAN would be 2.54 px
    # and force ON; the median stays 0.04 and the branch stays OFF.
    d = _ghost_gate_residual_moving([0.04] * 7 + [20.0])
    assert d["moving"] is False, d
    assert d["median"] == pytest.approx(0.04)
    assert d["mean"] > 2.0            # the statistic the old code used
    assert d["frac_above"] == pytest.approx(1 / 8)


def test_residual_branch_fires_when_motion_is_unanimous():
    from raw_sr.core.superres import _ghost_gate_residual_moving
    d = _ghost_gate_residual_moving([3.3] * 8)
    assert d["moving"] is True and d["median"] == pytest.approx(3.3)
    # The decision is on the MEDIAN value, so a strict majority of disagreeing
    # frames is needed; an even split averages to 1.55 px and stays OFF.
    assert _ghost_gate_residual_moving([0.1, 0.1, 3.0, 3.0])["moving"] is False
    assert _ghost_gate_residual_moving([0.1, 0.1, 3.0, 3.0, 3.0])["moving"] is True
    assert _ghost_gate_residual_moving([0.1, 0.1, 0.1, 3.0])["moving"] is False
    # Non-finite entries are ignored, not propagated.
    d = _ghost_gate_residual_moving([0.1, np.nan, np.inf, 0.2])
    assert d["moving"] is False and d["n"] == 2


def test_residual_branch_margin_on_the_real_burst():
    """The real burst must stay far below the D4 threshold at both ROIs."""
    if len(REAL_DATA) < 9:
        pytest.skip(missing_reason(9))
    from raw_sr.core.superres import (
        _ghost_gate_residual_moving, load_raw_linear)
    for roi in ((1650, 1240, 512, 512), (1650, 1240, 1024, 1024)):
        frames = [load_raw_linear(f, roi=roi) for f in REAL_DATA[:9]]
        res = SuperResolution(ratio=2.0, ibp_iterations=0).run(frames)
        r = [x for x, s in zip(res.align_residuals, res.align_self) if not s]
        d = _ghost_gate_residual_moving(r)
        assert d["moving"] is False, d
        assert d["median"] < 1.0, (roi, d)          # >=2x margin to 2.0 px
        assert res.metrics["robust_active"] is False
