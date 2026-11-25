"""B2: the ghost gate must not fire on a *static* real scene, and when it does
fire for real motion it must cost neither the static background nor the memory
budget.

The B1 audit measured the auto gate declaring "moving subject" on plain static
crops of the same handheld DNG burst — wind-blown foliage, a near-depth facade
corner — at (1300,900) 1024 (share 0.31 slipped past the old ``share >= 0.25``
shortcut), (0,0) D3 0.053 and (2600,1500) within 6% of the D3 cut. It also
measured D3 firing on 24 of 72 static variants, including a ±5% exposure step,
and an ON decision costing 1.82 dB of HF.

These tests pin the B2 fixes behaviourally through ``run``:

* the real burst resolves AUTO to OFF over a grid of ROIs (512 and 1024,
  corners and the three audited ROIs);
* a static synthetic burst stays OFF under ±5% exposure drift/step and under
  low contrast + heavy sensor noise;
* moving subjects (small, dominant, flooded) are still ON, including at N=2,
  where the flagged region keeps only the reference frame (a pair has no median
  to vote on, so re-weighting cannot remove a ghost);
* a real static burst at N=2 resolves AUTO to OFF, byte-identical to plain;
* the gate refuses to decide when the frames barely overlap;
* the robust pass re-weights only the region the gate flagged, so a mostly
  static frame keeps its HF.
"""
import numpy as np
import cv2
import pytest

import raw_sr.core.superres as SR
from raw_sr.core.superres import (
    SuperResolution,
    _auto_robust_fusion,
    _ghost_gate_dispersion,
    _ghost_gate_pair,
    _ghost_gate_population,
    _ghost_gate_region,
    _ghost_gate_spread,
    _robust_residual_weights,
    _warp_frames_to_hr,
    load_raw_linear,
)

from tests.data_paths import dng_files, missing_reason
from tests.test_superres_ghost_gate_b1 import _burst, _run, _trail_error, RATIO

REAL_DATA = dng_files(9)

# The audited offenders first, then a spread of corners/edges at both sizes.
REAL_ROIS_1024 = [(1300, 900), (0, 0), (2600, 1500),
                  (1650, 1240), (1800, 1224), (3600, 2448)]
REAL_ROIS_512 = [(1300, 900), (0, 0), (2600, 1500),
                 (1650, 1240), (2056, 1480), (3600, 2448)]


# --------------------------------------------------------------------------- #
# Real static burst: OFF at a grid of ROIs, not just the author's two
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(len(REAL_DATA) < 9, reason=missing_reason(9))
@pytest.mark.parametrize("roi", [(x, y, 1024) for x, y in REAL_ROIS_1024]
                                + [(x, y, 512) for x, y in REAL_ROIS_512])
def test_real_static_burst_is_off_across_the_roi_grid(roi):
    x, y, size = roi
    frames = [load_raw_linear(f, roi=(x, y, size, size)) for f in REAL_DATA]
    auto = _run(frames, robust_fusion=None, ecc_scale=0.25)
    assert auto.metrics["robust_active"] is False, auto.metrics["robust_reason"]
    off = _run(frames, robust_fusion=False, ecc_scale=0.25)
    assert np.array_equal(auto.hr, off.hr)
    assert auto.metrics["hf_gain_db"] == off.metrics["hf_gain_db"]


REAL_ROIS_N2 = [(1300, 900, 1024), (0, 0, 1024), (2600, 1500, 1024),
                (1800, 1224, 1024), (1300, 900, 512), (0, 0, 512),
                (2600, 1500, 512)]


@pytest.mark.skipif(len(REAL_DATA) < 9, reason=missing_reason(9))
@pytest.mark.parametrize("roi", REAL_ROIS_N2)
def test_real_static_burst_at_n2_is_off_and_byte_identical(roi):
    # Two frames is the regime where the gate lost its median. The N=2 pair
    # detector (D5) must still resolve OFF on a real handheld static crop, and
    # OFF must be the plain deposit bit for bit -- no robustness tax.
    x, y, size = roi
    frames = [load_raw_linear(f, roi=(x, y, size, size))
              for f in REAL_DATA[:2]]
    auto = _run(frames, robust_fusion=None, ecc_scale=0.25)
    off = _run(frames, robust_fusion=False, ecc_scale=0.25)
    assert auto.metrics["robust_active"] is False, auto.metrics["robust_reason"]
    assert np.array_equal(auto.hr, off.hr)
    assert auto.metrics["hf_gain_db"] == off.metrics["hf_gain_db"]
    assert auto.metrics["sharpness_ratio"] == off.metrics["sharpness_ratio"]


@pytest.mark.skipif(len(REAL_DATA) < 9, reason=missing_reason(9))
def test_real_burst_still_reports_the_gate_evidence():
    # OFF must be a decision from measured statistics, not a dead code path:
    # the audited (1300,900) crop has a large but *scattered* population, i.e.
    # a high share with a low bbox fill — exactly what the blob test rejects.
    frames = [load_raw_linear(f, roi=(1300, 900, 1024, 1024))
              for f in REAL_DATA]
    res = _run(frames, robust_fusion=None, ecc_scale=0.25)
    kept = [frames[i] for i in res.kept_indices]
    S, V, _oh, _ow = _warp_frames_to_hr(kept, res.transforms, RATIO)
    pop = _ghost_gate_population(S, V)
    spread = _ghost_gate_spread(S, V)
    assert pop["share"] > 0.10          # the audit's 0.31 case is real
    assert pop["extent"] < 0.55         # ... but it is not one blob
    assert spread["ratio"] < 0.040      # and the spread is below the D3 cut


# --------------------------------------------------------------------------- #
# Static synthetics the audit found firing: exposure drift/step, low contrast
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("kw", [
    dict(n=9, subject="none", exposure_drift=0.05),
    dict(n=2, subject="none", exposure_drift=0.05),
    dict(n=9, subject="none", exposure_step=1),
    dict(n=2, subject="none", exposure_step=1),
], ids=["drift5_n9", "drift5_n2", "step5_n9", "step5_n2"])
def test_static_burst_with_exposure_change_is_off(kw):
    # An exposure change is a per-frame gain, not motion. The gate normalizes
    # each frame by its median level before comparing.
    frames, _truth, _trail = _burst(**kw)
    res = _run(frames, robust_fusion=None)
    assert res.metrics["robust_active"] is False, res.metrics["robust_reason"]


@pytest.mark.parametrize("kw", [
    dict(n=9, subject="none", contrast=0.10),
    dict(n=9, subject="none", contrast=0.06),
    dict(n=9, subject="none", contrast=0.02),
    dict(n=9, subject="none", noise=0.012),
    dict(n=9, subject="none", noise=0.020),
    dict(n=9, subject="none", contrast=0.06, noise=0.012),
    dict(n=2, subject="none", contrast=0.06, noise=0.012),
], ids=["c10", "c06", "c02", "n12", "n20", "c06n12", "c06n12_n2"])
def test_low_contrast_noisy_static_burst_is_off(kw):
    # D3 is an inverse-SNR proxy: low contrast / high noise inflate it. The
    # gate now requires spatial corroboration, and a static burst flags no
    # populated region at all.
    frames, _truth, _trail = _burst(**kw)
    res = _run(frames, robust_fusion=None)
    assert res.metrics["robust_active"] is False, res.metrics["robust_reason"]


# --------------------------------------------------------------------------- #
# Moving subjects, including N=2
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [2, 3, 9])
def test_small_moving_subject_is_on_at_every_frame_count(n):
    frames, _truth, _trail = _burst(n=n, subject="small")
    res = _run(frames, robust_fusion=None)
    assert res.metrics["robust_active"] is True, res.metrics["robust_reason"]


@pytest.mark.parametrize("n", [2, 3, 9])
def test_dominant_moving_subject_is_on_at_every_frame_count(n):
    frames, _truth, _trail = _burst(n=n, subject="dominant", subject_side=220)
    res = _run(frames, robust_fusion=None)
    assert res.metrics["robust_active"] is True, res.metrics["robust_reason"]


def test_flooded_subject_at_n2_is_on_and_falls_back_to_the_reference():
    # With two frames the across-frame median is the mean, so D1-D3 see half the
    # disagreement and the Tukey re-weighting is inert (both frames deviate
    # equally, so they get the same weight). D5 measures the pair difference
    # itself; the robust pass then keeps the reference frame alone where the
    # pair disagrees. Before this, the gate resolved OFF here (a strict xfail
    # pinned it until fix/xf22) and the fused frame showed the 760-px subject
    # twice.
    frames, truth, trail = _burst(n=2, subject="dominant", subject_side=760,
                                  size=1024, sweep=60.0, allow_outside=True)
    res = _run(frames, robust_fusion=None)
    assert res.metrics["robust_active"] is True, res.metrics["robust_reason"]
    assert "2-frame pair disagreement" in res.metrics["robust_reason"]
    assert "reference frame" in res.metrics["robust_reason"]
    kept = [frames[i] for i in res.kept_indices]
    S, V, _oh, _ow = _warp_frames_to_hr(kept, res.transforms, RATIO)
    pair = _ghost_gate_pair(S, V)
    assert pair["moving"] is True
    assert pair["ratio"] >= SR._GHOST_GATE_PAIR_RATIO
    assert pair["share"] >= SR._GHOST_GATE_PAIR_SHARE
    # The swept trail is where plain fusion shows *both* copies of the subject
    # (each frame contributes one, at half weight). The reference fallback must
    # take a real bite out of that ghost rather than merely being switched on.
    off = _run(frames, robust_fusion=False)
    e_on = _trail_error(res.hr, truth, trail)
    e_off = _trail_error(off.hr, truth, trail)
    assert e_on < 0.95 * e_off, (e_on, e_off)


def test_n2_fallback_keeps_the_reference_not_the_average():
    # Inside the flagged region the result must come from the reference frame
    # (single frame bicubic, no ghost), not from the two-frame average: in the
    # reference's own copy of the subject there is no second opinion to average
    # with. The final HR image is then refined by IBP against BOTH frames, so
    # the region is not the reference render bit for bit -- but it must stay far
    # closer to the reference than to the other frame's displaced copy.
    frames, _truth, _trail = _burst(n=2, subject="dominant",
                                    subject_side=760, size=1024, sweep=60.0,
                                    allow_outside=True)
    res = _run(frames, robust_fusion=None)
    kept = [frames[i] for i in res.kept_indices]
    S, V, oh, ow = _warp_frames_to_hr(kept, res.transforms, RATIO)
    alpha = _ghost_gate_region(S, V, dilate_tiles=1, soft=True)
    interior = alpha > 0.999
    assert interior.mean() > 0.10
    rp = res.kept_indices.index(res.reference_index)
    other = 1 - rp

    def _up(pos):
        M = SR.to_drizzle_matrix(res.transforms[pos], RATIO)
        return cv2.warpPerspective(
            np.asarray(kept[pos].rgb, np.float32), M, (ow, oh),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    err_ref = float(np.abs(res.hr - _up(rp)).mean(axis=2)[interior].mean())
    err_oth = float(np.abs(res.hr - _up(other)).mean(axis=2)[interior].mean())
    assert err_ref < 0.4 * err_oth, (err_ref, err_oth)


# --------------------------------------------------------------------------- #
# Coverage guard: the gate refuses to reason on border junk
# --------------------------------------------------------------------------- #

def test_pair_detector_only_answers_for_two_frames():
    # D5 exists because two frames are the degenerate case for a median; it must
    # stay silent (None) for any other burst, so nothing but N=2 changes.
    frames, _truth, _trail = _burst(n=3, subject="small")
    res = _run(frames, robust_fusion=None)
    kept = [frames[i] for i in res.kept_indices]
    S, V, _oh, _ow = _warp_frames_to_hr(kept, res.transforms, RATIO)
    assert _ghost_gate_pair(S, V) is None


def test_gate_refuses_when_frames_barely_overlap(monkeypatch):
    frames, _truth, _trail = _burst(n=3, subject="small")
    S = np.zeros((3, 64, 64), np.float32)
    V = np.zeros((3, 64, 64), bool)
    V[:, :, :20] = True                      # 31% common coverage
    monkeypatch.setattr(SR, "_warp_frames_to_hr",
                        lambda *a, **k: (S, V, 64, 64))
    active, reason, stack = _auto_robust_fusion(
        frames, [np.eye(3)] * 3, RATIO)
    assert active is False
    assert "covered by every frame" in reason
    assert _ghost_gate_dispersion(S, V)[0] is False
    assert _ghost_gate_population(S, V)["moving"] is False
    assert _ghost_gate_spread(S, V)["moving"] is False
    assert _ghost_gate_region(S, V) is None


# --------------------------------------------------------------------------- #
# Localized robust fusion: weight only where the gate found disagreement
# --------------------------------------------------------------------------- #

def test_region_is_local_and_weights_are_one_outside_it():
    frames, _truth, _trail = _burst(n=5, subject="small")
    res = _run(frames, robust_fusion=None)
    kept = [frames[i] for i in res.kept_indices]
    S, V, oh, ow = _warp_frames_to_hr(kept, res.transforms, RATIO)
    region = _ghost_gate_region(S, V)
    assert region is not None
    assert 0.0 < float(region.mean()) < 0.5     # local, not the whole frame
    assert len(res.transforms) == len(kept)
    wts = _robust_residual_weights(S, V, kept, res.transforms, RATIO, 4.0,
                                   0.0, region=region)
    # The mask is dilated + softened, so a few LR weights inside the ramp may
    # already be 1; outside it *every* weight must be exactly plain (1.0).
    h, w = kept[0].rgb.shape[:2]
    lr_region = (region[::int(RATIO), ::int(RATIO)][:h, :w] > 0.5)
    # The drizzle maps LR positions with the frame transform, so allow a margin
    # (the region is 2 tiles of dilation already) before demanding full weight.
    k = np.ones((13, 13), np.uint8)
    near = cv2.dilate(lr_region.astype(np.uint8), k) > 0
    outside = ~near
    assert outside.mean() > 0.2
    for wt in wts:
        assert np.all(np.asarray(wt)[outside] == 1.0)


@pytest.mark.skipif(len(REAL_DATA) < 9, reason=missing_reason(9))
def test_localized_reweighting_keeps_hf_on_real_content():
    # Measured (B2, ROI 1024 (600,2000) of the real burst: a near-depth facade
    # with genuine wind/cloud motion, 7% of the frame flagged): plain 12.94 dB,
    # forced ON whole-frame 11.71 dB, forced ON localized 12.88 dB. The local
    # region therefore keeps the ON decision from spending the static scene.
    frames = [load_raw_linear(f, roi=(600, 2000, 1024, 1024))
              for f in REAL_DATA]
    plain = _run(frames, robust_fusion=False, ecc_scale=0.25)
    local = SuperResolution(ratio=RATIO, ecc_scale=0.25,
                            use_gpu_scatter=False, robust_fusion=True, fusion="demosaic_drizzle",
                            robust_localize=True).run(frames)
    global_ = SuperResolution(ratio=RATIO, ecc_scale=0.25,
                              use_gpu_scatter=False, robust_fusion=True, fusion="demosaic_drizzle",
                              robust_localize=False).run(frames)
    hf_plain = plain.metrics["hf_gain_db"]
    assert local.metrics["robust_active"] is True
    assert global_.metrics["hf_gain_db"] < hf_plain - 0.5   # the audited harm
    assert local.metrics["hf_gain_db"] >= hf_plain - 0.25   # the B2 fix
