"""The held-out PSNR proof must predict from the run's OWN reference frame.

The held-out proof predicts a frame that was not fused, starting from the SR
image. That only works if the prediction starts from the frame whose LR
coordinates the HR grid is aligned to — the automatically selected reference
(``SRResult.reference_index``), which is normally NOT ``frames[0]``. Predicting
from ``frames[0]`` throws the inter-frame registration offset away: on the real
9-frame burst it collapsed the reported gain from ~+7 dB to ~0 dB at 2x/3x/4x
(ROI 1024), which is the "held-out gain is zero" regression these tests pin.
"""

import importlib.util
import os

import numpy as np
import cv2
import pytest

from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    estimate_homography,
    predict_lr_from_hr,
    SENSOR_PSF_SIGMA_PER_RATIO,
    to_drizzle_matrix,
    denoised_lr_stack,
    high_frequency_gain_db,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Synthetic burst helpers (same construction as tests/test_superres.py)
# --------------------------------------------------------------------------- #

def _bandpass_truth(h=128, w=128, lo=0.28, hi=0.50, seed=11):
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


def _sample_lr(truth, dx, dy, lr_shape, ratio=2, sensor_sigma=None):
    """LR measurement: integrate over the LR pixel (sensor PSF), then sample.

    Sampling is at the pixel centre, ``r*(x+dx)+(r-1)/2`` — the same
    pixel-centre convention as :func:`to_drizzle_matrix`.
    """
    src = truth
    if sensor_sigma:
        src = cv2.GaussianBlur(truth, (0, 0), float(sensor_sigma),
                               borderType=cv2.BORDER_REPLICATE)
    off = 0.5 * (ratio - 1.0)
    xs, ys = np.meshgrid(np.arange(lr_shape[1], dtype=np.float32),
                         np.arange(lr_shape[0], dtype=np.float32))
    return cv2.remap(src, ratio * (xs + dx) + off,
                     ratio * (ys + dy) + off,
                     cv2.INTER_LINEAR)


def _smooth_truth(h=128, w=128, seed=5, sigma=1.0):
    rng = np.random.default_rng(seed)
    t = rng.random((h, w, 3)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), sigma)
    return np.clip(t / (t.max() + 1e-9), 0, 1)


def _burst(shifts, truth, noise=0.001, blur_first=0.0, seed=3, ratio=2):
    h, w = truth.shape[0] // ratio, truth.shape[1] // ratio
    rng = np.random.default_rng(seed)
    frames = []
    for i, (dx, dy) in enumerate(shifts):
        lr = _sample_lr(truth, dx, dy, (h, w), ratio,
                        sensor_sigma=SENSOR_PSF_SIGMA_PER_RATIO * ratio)
        if i == 0 and blur_first > 0:
            lr = cv2.GaussianBlur(lr, (0, 0), blur_first)
        lr = np.clip(lr + rng.normal(0, noise, lr.shape).astype(np.float32), 0, 1)
        frames.append(RawFrame(name=f"f{i}", rgb=lr, green=lr[:, :, 1].copy()))
    return frames


def _psnr(a, b):
    return 10 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12))


SHIFTS = [(0.0, 0.0), (0.5, 0.25), (0.25, 0.75), (0.75, 0.5), (0.4, 0.1)]


# --------------------------------------------------------------------------- #
# predict_lr_from_hr: the forward model of the held-out prediction
# --------------------------------------------------------------------------- #

def test_predict_lr_from_hr_inverts_the_deposition_geometry():
    """Identity ref->frame: prediction is hr sampled at r*(x, y)."""
    rng = np.random.default_rng(0)
    lr = rng.random((24, 32, 3)).astype(np.float32)
    hr = np.repeat(np.repeat(lr, 2, axis=0), 2, axis=1)  # hr(2x,2y) == lr(x,y)
    pred = predict_lr_from_hr(hr, np.eye(3), 2.0, lr.shape[:2],
                              forward_sigma=0.0)
    assert pred.shape == lr.shape
    assert np.max(np.abs(pred - lr)) < 1e-5


def test_predict_lr_from_hr_honours_a_subpixel_offset():
    """A ref->frame translation of +0.5 LR px shifts the prediction by 0.5 px."""
    w_hr, h_hr = 64, 64
    # Linear ramp in x on the HR grid: exact under bilinear sampling.
    hr = np.zeros((h_hr, w_hr, 3), dtype=np.float32)
    hr[:, :, :] = (np.arange(w_hr, dtype=np.float32) /
                   (w_hr - 1))[None, :, None]
    W = np.array([[1, 0, 0.5], [0, 1, 0.0], [0, 0, 1]], dtype=np.float64)
    pred = predict_lr_from_hr(hr, W, 2.0, (32, 32), forward_sigma=0.0)
    # mov(W(p)) == ref(p) with W(p) = p + 0.5, so frame LR x samples ref
    # LR (x - 0.5), i.e. HR coordinate 2x - 1. With the HR grid's pixel-centre
    # offset, that is 2x - 1 + (r-1)/2 = 2x - 0.5 (replicated at x = 0).
    xs = np.arange(32, dtype=np.float32)
    expect = np.clip(2.0 * xs - 0.5, 0.0, w_hr - 1) / (w_hr - 1)
    assert np.allclose(pred[:, :, 0], expect[None, :], atol=1e-5)


def test_predict_lr_from_hr_applies_the_sensor_psf_by_default():
    """An HR image compared against a *measured* LR sample must be band-limited.

    The default forward model blurs by the per-LR-pixel integration window
    (sigma = SENSOR_PSF_SIGMA_PER_RATIO * ratio). Without it a ratio-r HR
    estimate aliases when sampled, which measured ~2.5 dB worse PSNR at 4x.
    """
    truth = _bandpass_truth()
    lr = _sample_lr(truth, 0.0, 0.0, (64, 64))
    sharp = predict_lr_from_hr(truth, np.eye(3), 2.0, (64, 64),
                               forward_sigma=0.0)
    band = predict_lr_from_hr(truth, np.eye(3), 2.0, (64, 64))
    assert SENSOR_PSF_SIGMA_PER_RATIO == pytest.approx(0.289)
    # Blurring removes HF energy that aliasing would have folded back in.
    assert high_frequency_gain_db(band, sharp, cutoff=0.25) < 0.0
    # ... and both stay inside the measured range.
    for img in (sharp, band):
        assert img.min() >= -1e-6 and img.max() <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# reference_index: reported, and the frame the HR grid is aligned to
# --------------------------------------------------------------------------- #

def test_reference_index_is_reported_and_carries_the_identity_transform():
    truth = _smooth_truth()
    frames = _burst(SHIFTS, truth, blur_first=1.0)
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(frames)
    assert res.reference_index == res.metrics["reference_index"]
    assert res.reference_index in res.kept_indices
    pos = res.kept_indices.index(res.reference_index)
    assert np.allclose(np.asarray(res.transforms[pos]), np.eye(3), atol=1e-12)


def test_heldout_prediction_from_the_run_reference_beats_frames0():
    """The exact bug: predicting from kept[0] when the reference is not kept[0].

    Frame 0 (kept, blurred) is deliberately not the sharpest frame, so the
    automatic reference selection moves the HR grid to another frame. The
    held-out frame is the last one, which was never fused.
    """
    truth = _smooth_truth()
    frames = _burst(SHIFTS, truth, blur_first=1.0)
    hidx = len(frames) - 1
    kept = frames[:-1]
    hw = frames[hidx]
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(kept)
    assert res.reference_index != 0, "fixture must force a non-zero reference"

    W_ref = estimate_homography(kept[res.reference_index].green, hw.green,
                                ecc_scale=0.5)
    good = predict_lr_from_hr(res.hr, W_ref, 2.0, hw.rgb.shape[:2])
    W_bug = estimate_homography(kept[0].green, hw.green, ecc_scale=0.5)
    bad = predict_lr_from_hr(res.hr, W_bug, 2.0, hw.rgb.shape[:2])
    assert _psnr(good, hw.rgb) > _psnr(bad, hw.rgb) + 3.0


# --------------------------------------------------------------------------- #
# The acceptance script itself
# --------------------------------------------------------------------------- #

def _load_acceptance_module():
    path = os.path.join(ROOT, "scripts", "sr_acceptance.py")
    spec = importlib.util.spec_from_file_location("sr_acceptance_b1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_acceptance_cli_exposes_output_tagging_and_heldout_flags():
    """Parallel/scale acceptance runs must not overwrite each other."""
    mod = _load_acceptance_module()
    ap = mod.build_parser()
    a = ap.parse_args([])
    assert a.ratio == 2.0 and a.roi_size == 1024
    assert a.out_suffix == "" and a.out_dir is None
    assert a.heldout_sweep is False and a.heldout_no_psf is False
    b = ap.parse_args(["--ratio", "4", "--out-suffix", "_B1heldout",
                       "--out-dir", "output", "--heldout-sweep",
                       "--heldout-no-psf", "--ibp-iterations", "0"])
    assert b.ratio == 4.0
    assert b.out_suffix == "_B1heldout"
    assert b.out_dir == "output"
    assert b.heldout_sweep and b.heldout_no_psf
    assert b.ibp_iterations == 0


def test_acceptance_heldout_uses_the_run_reference_and_has_positive_gain():
    mod = _load_acceptance_module()
    truth = _smooth_truth()
    frames = _burst(SHIFTS, truth, blur_first=1.0)
    d = mod.evaluate_heldout(frames, 2.0, "homography", 0.5, len(frames) - 1)
    assert d["heldout_reference_index"] != 0
    assert d["heldout_psnr_db_sr"] > 30.0        # not the ~10-20 dB bug path
    assert d["heldout_gain_db"] > 1.0
    # The baseline must be the denoised stack of the same frames, never the
    # held-out frame itself.
    assert d["heldout_psnr_db_denoised_stack"] < 60.0


# --------------------------------------------------------------------------- #
# B2 (2026-11-19, audit fix a): the baseline must follow kept_indices.
#
# ``SRResult.transforms`` / ``weights`` are per *kept* frame, indexed by
# ``kept_indices``. Master zipped the full ``kept`` list against them, so when
# the run dropped a frame (blur threshold or alignment gate) every following
# frame was paired with the previous frame's transform: the baseline was
# denoised from the wrong geometry, silently (zip truncates, no error).
# --------------------------------------------------------------------------- #

def test_heldout_baseline_selects_frames_through_kept_indices():
    from types import SimpleNamespace

    mod = _load_acceptance_module()
    truth = _smooth_truth()
    frames = _burst(SHIFTS, truth)
    res = SimpleNamespace(
        kept_indices=[0, 2, 4],
        transforms=[np.eye(3), np.eye(3), np.eye(3)],
        weights=np.array([0.5, 0.3, 0.2]),
    )
    base, cov, fused = mod.heldout_baseline_lr(frames, res)
    assert [f.name for f in fused] == [frames[i].name for i in (0, 2, 4)]
    assert base.shape == frames[0].rgb.shape
    assert cov.shape == base.shape[:2]
    # Fully covered everywhere (identity warps) and normalised per pixel.
    assert np.allclose(cov, float(np.sum(res.weights)), atol=1e-5)
    # The baseline is a weighted mean of exactly those frames (identity warps).
    expect = sum(w * f.rgb for w, f in zip(res.weights, fused))
    assert np.allclose(base, expect, atol=1e-6)


def test_heldout_baseline_rejects_a_mispaired_run():
    """A run whose transform list does not match its kept frames must not pass."""
    from types import SimpleNamespace

    mod = _load_acceptance_module()
    frames = _burst(SHIFTS, _smooth_truth())
    res = SimpleNamespace(kept_indices=[0, 1, 2],
                          transforms=[np.eye(3), np.eye(3)],
                          weights=np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="fused frames"):
        mod.heldout_baseline_lr(frames, res)


def test_acceptance_heldout_pairs_the_baseline_when_a_frame_is_dropped(
        monkeypatch):
    """End-to-end: with the blur gate active, baseline frames == the fused ones.

    A deliberately blurred frame is dropped by ``blur_threshold_ratio``; the
    recorded set the baseline was built from must be exactly
    ``[kept[i] for i in result.kept_indices]`` and in that order.
    """
    mod = _load_acceptance_module()
    truth = _smooth_truth()
    frames = _burst(SHIFTS + [(0.6, 0.2), (0.15, 0.85)], truth)
    blurred = cv2.GaussianBlur(frames[2].rgb, (0, 0), 4.0)
    frames[2] = RawFrame(name="blurred", rgb=blurred,
                         green=blurred[:, :, 1].copy())

    seen = {}

    class RecordingSR(SuperResolution):
        last = None
        inputs: list = []

        def run(self, fs, **kw):
            res = super().run(fs, **kw)
            RecordingSR.last = res
            RecordingSR.inputs = list(fs)
            return res

    real_baseline = mod.denoised_lr_stack

    def recording_baseline(fs, aligns, weights=None, **kw):
        seen["names"] = [f.name for f in fs]
        seen["n_aligns"] = len(aligns)
        return real_baseline(fs, aligns, weights, **kw)

    monkeypatch.setattr(mod, "SuperResolution", RecordingSR)
    monkeypatch.setattr(mod, "denoised_lr_stack", recording_baseline)

    d = mod.evaluate_heldout(frames, 2.0, "homography", 0.5,
                             len(frames) - 1, blur_threshold_ratio=0.95)
    res = RecordingSR.last
    kept = [f for i, f in enumerate(frames) if i != len(frames) - 1]
    expect = [kept[i].name for i in res.kept_indices]
    assert len(res.kept_indices) < len(kept), "the blur gate must drop a frame"
    assert seen["names"] == expect
    assert seen["n_aligns"] == len(expect)
    assert d["heldout_frames_fused"] == len(expect)
    assert d["heldout_blur_threshold_ratio"] == 0.95
    assert np.isfinite(d["heldout_gain_db"])


def test_acceptance_records_the_heldout_gating_configuration():
    """The metric states which gating produced it (0.0 is not the shipped 0.6)."""
    mod = _load_acceptance_module()
    frames = _burst(SHIFTS, _smooth_truth(), blur_first=1.0)
    d = mod.evaluate_heldout(frames, 2.0, "homography", 0.5,
                             len(frames) - 1)
    assert d["heldout_blur_threshold_ratio"] == 0.0
    assert d["heldout_frames_fused"] == len(frames) - 1


# --------------------------------------------------------------------------- #
# B4 / fix/metric6 (2026-11-19): the baseline's unfilled rim.
#
# ``denoised_lr_stack`` used to divide by the total weight of *all* frames.
# With real registration shifts every frame ends somewhere inside the ROI, so
# the accumulator is ~0 in a rim whose width is the shift; dividing by the full
# weight leaves a dark band there. The held-out frame is complete, so the old
# metric scored SR against a baseline that was wrong by construction along all
# four borders — 40-80 % of the baseline's total MSE sat in a 24 px ring, and
# that alone produced the "+3.7 dB held-out gain".
# --------------------------------------------------------------------------- #

def test_denoised_lr_stack_coverage_normalisation_removes_the_dark_rim():
    """A shifted burst must not darken the stack's border."""
    from raw_sr.core.superres import denoised_lr_stack

    truth = _smooth_truth(h=96, w=96) + 0.3     # DC offset: a rim shows as dark
    frames = _burst([(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (3.0, 3.0)], truth,
                    noise=0.0)
    aligns = [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0), (3.0, 3.0)]
    legacy = denoised_lr_stack(frames, aligns)
    fixed, cov = denoised_lr_stack(frames, aligns, coverage_normalised=True,
                                   return_coverage=True)
    total = 4.0
    # The no-shift corner is covered by every frame in both variants.
    assert np.allclose(fixed[4:-4, 4:-4], legacy[4:-4, 4:-4], atol=1e-6)
    # The rim is the difference: legacy is dark, the per-pixel mean is not.
    rim = np.zeros(cov.shape, dtype=bool)
    rim[:3, :] = rim[-3:, :] = True
    rim[:, :3] = rim[:, -3:] = True
    interior = np.zeros(cov.shape, dtype=bool)
    interior[4:-4, 4:-4] = True
    assert legacy[rim].mean() < 0.75 * fixed[rim].mean()
    # The corrected rim is a full weighted mean: same brightness as the middle.
    assert abs(float(fixed[rim].mean()) - float(fixed[interior].mean())) < 0.05
    # Coverage reports exactly which pixels are unreliable.
    assert cov[rim].min() < total * 0.999
    assert np.allclose(cov[4:-4, 4:-4], total, atol=1e-4)


def test_heldout_gain_is_invariant_to_the_baseline_rim():
    """SR == baseline in the covered interior => |gain| < 0.05 dB, for any rim.

    The rim is produced the way the real burst produces it: shifted frames end
    somewhere inside the ROI, so the stack has an uncovered band whose width is
    the shift. Four rim widths (6 to 40 px) are checked. With the per-pixel
    normalisation the covered area is a true weighted mean, and the coverage
    map excludes exactly the band no frame reached, so the gain is 0 whatever
    the rim. The legacy whole-frame number still shows the artefact — that is
    the regression this test pins.
    """
    from types import SimpleNamespace

    mod = _load_acceptance_module()
    truth = _smooth_truth(h=96, w=96) + 0.35
    rng = np.random.default_rng(7)

    def shifted(shifts, ratio=2):
        """LR samples of ``truth`` at sub-pixel offsets, replicated borders.

        Replication (not the 0-filled default) keeps the fixture free of a
        second, unrelated black band; the only rim is the warp coverage one.
        """
        h, w = truth.shape[0] // ratio, truth.shape[1] // ratio
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        out = []
        for i, (dx, dy) in enumerate(shifts):
            off = 0.5 * (ratio - 1.0)
            lr = cv2.remap(truth, ratio * (xs + dx) + off,
                           ratio * (ys + dy) + off,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out.append(RawFrame(name=f"f{i}", rgb=lr,
                                green=lr[:, :, 1].copy()))
        return out

    for shift_px in (6.0, 12.0, 24.0, 40.0):
        shifts = [(0.0, 0.0), (shift_px, 0.0), (0.0, shift_px),
                  (shift_px, shift_px)]
        frames = shifted(shifts)
        res = SimpleNamespace(kept_indices=[0, 1, 2, 3], transforms=shifts,
                              weights=np.array([1.0, 1.0, 1.0, 1.0]))
        base, cov, _fused = mod.heldout_baseline_lr(frames, res)
        total = 4.0
        mask = cov >= total * 0.999
        assert mask.any(), "the covered interior must not be empty"
        assert np.mean(mask) < 0.99, "fixture must produce an uncovered rim"
        held = frames[0].rgb
        # Covered area is a weighted mean, not a darkened sum: brightness holds.
        assert abs(float(base[mask].mean()) - float(held[mask].mean())) < 0.05
        # SR == baseline in the interior: same image, independent equal noise.
        pred = base + rng.normal(0, 1e-3, base.shape).astype(np.float32)
        base_e = base + rng.normal(0, 1e-3, base.shape).astype(np.float32)
        d = mod.heldout_compare(pred, base_e, held, mask)
        assert abs(d["heldout_gain_db"]) < 0.05, (
            f"shift/rim {shift_px}px leaked into the headline gain: "
            f"{d['heldout_gain_db']:+.3f} dB")
        if shift_px >= 12.0:
            # The pre-fix baseline (global-weight normalisation) still shows the
            # artefact this branch removes; that is what the regression is about.
            legacy_stack = denoised_lr_stack(frames, shifts,
                                             coverage_normalised=False)
            d_old = mod.heldout_compare(pred, legacy_stack, held, mask)
            assert d_old["heldout_gain_db_legacy"] > 0.5, (
                "the pre-fix baseline must reproduce the dark-rim artefact")


def test_heldout_gain_is_positive_when_aliasing_is_really_recovered():
    """Genuine >Nyquist recovery must still be rewarded by the masked metric.

    Content just above the LR Nyquist (0.26-0.34 cyc/HR px) is the one band the
    SR image can recover and an LR stack physically cannot, so the honest gain
    has to stay clearly positive — the mask must not be so aggressive that it
    hides real resolution recovery.
    """
    mod = _load_acceptance_module()
    # A broadband high-frequency texture rather than a pure grating: the
    # held-out harness registers the burst itself, and a periodic scene has
    # many equivalent translations (ECC locks a whole period away, which
    # measures the registrator, not resolution recovery). Gaussian-blurred
    # noise at sigma 0.9 still carries energy above the LR Nyquist, so the SR
    # image can recover detail the LR stack cannot.
    rng = np.random.default_rng(3)
    truth = cv2.GaussianBlur(rng.random((192, 192, 3)).astype(np.float32),
                             (0, 0), 0.9)
    truth = np.clip(0.3 + 0.6 * (truth - truth.min())
                    / (truth.max() - truth.min()), 0, 1).astype(np.float32)
    diverse = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.25, 0.75),
               (0.75, 0.25), (0.3, 0.3), (0.7, 0.7), (0.15, 0.6)]
    frames = _burst(diverse, truth, noise=0.0)
    # THIS test measures the metric, so it runs with frame acceptance disabled
    # (min_align_score=0.0): content that lives above the LR Nyquist makes a
    # correctly registered pair score a low ZNCC, and the gate would drop
    # frames for a reason that has nothing to do with the metric. The gate's
    # effect on this same fixture is covered separately, below.
    d = mod.evaluate_heldout(frames, 2.0, "homography", 0.5, len(frames) - 1,
                             min_align_score=0.0)
    assert 0.2 < d["heldout_valid_fraction"] < 0.99
    # Threshold re-baselined with B8 (#2 reference-selection sign). The
    # automatic reference moved from frame 3 to frame 4 (the temporal middle,
    # the frame the selection is documented to prefer) and this fixture's gain
    # is strongly reference-phase dependent: +1.607 dB at reference 3 vs
    # +0.473 dB at reference 4, same frames, same code (measured). The reward
    # is still clearly positive; the phase sensitivity is a property of the
    # synthetic aliased-only scene and is reported in state.md.
    assert d["heldout_gain_db"] > 0.4, (
        f"real aliasing recovery scored {d['heldout_gain_db']:+.3f} dB")


def test_default_alignment_gate_drops_only_its_misregistered_frames():
    """On an aliased fixture at a gate-active size (192 LR px) the default
    gate must remove the frames whose registration failed, not the ones that
    merely score a low ZNCC, and the honest gain stays clearly positive.

    At 96 LR px (the fixture size above) the gate is advisory by design
    (review round 2, N2): there the measured junk floor (max 0.094) overlaps
    the aliased-content population (median ~0.11, minima -0.02), so no
    threshold separates them and nothing is dropped.
    """
    mod = _load_acceptance_module()
    truth = _bandpass_truth(h=384, w=384, lo=0.26, hi=0.34)
    diverse = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.25, 0.75),
               (0.75, 0.25), (0.3, 0.3), (0.7, 0.7), (0.15, 0.6)]
    frames = _burst(diverse, truth, noise=0.0)
    from raw_sr.core.superres import SuperResolution
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(frames)
    m = res.metrics
    assert m["align_gate_mode"] in ("relative", "absolute"), m["align_gate_mode"]
    assert res.used_frames < res.total_frames, "gate did not fire at all"
    assert m["align_gate_reason"].startswith(
        f"{len(m['align_gate_dropped'])} frame")
    # Every survivor is one of the low-residual frames: the relative floor
    # keeps the aliased-but-aligned frames whose ZNCC is below 0.1.
    assert min(res.align_scores) < 0.3, res.align_scores
    # Measured on this fixture: the 4 dropped frames are genuine ECC failures
    # (scores -0.34..-0.002, residuals 0.0-6.5 px, one wild W of (-204,-1932)),
    # so the gate removes garbage; the honest gain is +0.412 dB gated vs
    # +0.442 dB with the gate off (that garbage is near-invisible in MSE on a
    # noise-free aliased scene). What matters: the reward stays clearly
    # positive and no correctly registered frame is lost.
    d = mod.evaluate_heldout(frames, 2.0, "homography", 0.5, len(frames) - 1)
    assert d["heldout_gain_db"] > 0.35, d["heldout_gain_db"]


def test_acceptance_reports_the_masked_metric_and_its_legacy_twin():
    mod = _load_acceptance_module()
    frames = _burst(SHIFTS, _smooth_truth(), blur_first=1.0)
    d = mod.evaluate_heldout(frames, 2.0, "homography", 0.5, len(frames) - 1)
    assert d["heldout_interior_margin_lr"] == mod.HELDOUT_INTERIOR_MARGIN_LR
    h, w = frames[0].rgb.shape[:2]
    assert 0 < d["heldout_valid_pixels"] < h * w
    assert 0.0 < d["heldout_valid_fraction"] < 1.0
    assert abs(d["heldout_gain_db"]
               - (d["heldout_psnr_db_sr"]
                  - d["heldout_psnr_db_denoised_stack"])) < 1e-9
    for k in ("heldout_gain_db_legacy", "heldout_psnr_db_sr_legacy",
              "heldout_psnr_db_denoised_stack_legacy"):
        assert np.isfinite(d[k])


def test_heldout_run_follows_the_ibp_override():
    """``--ibp-iterations`` must reach the held-out run, not just the headline.

    Before this, the acceptance CLI applied the override to the displayed SR
    image while every held-out number kept the core default — an IBP sweep
    would have compared identical predictions under three names.
    """
    mod = _load_acceptance_module()
    frames = _burst(SHIFTS, _smooth_truth(), blur_first=1.0)
    d0 = mod.evaluate_heldout(frames, 2.0, "homography", 0.5,
                              len(frames) - 1, ibp_iterations=0)
    d3 = mod.evaluate_heldout(frames, 2.0, "homography", 0.5,
                              len(frames) - 1, ibp_iterations=3)
    assert d0["heldout_ibp_iterations"] == 0
    assert d3["heldout_ibp_iterations"] == 3
    assert abs(d0["heldout_gain_db"] - d3["heldout_gain_db"]) > 1e-6
