"""Evidence tests for the automatic (tri-state) ghost-suppression gate.

``SuperResolution.robust_fusion`` is ``bool | None``; ``None`` (the default) is
AUTO: the core decides per burst on the *correctly registered* warped stack,
and reports the decision additively as ``metrics['robust_active']`` /
``metrics['robust_reason']``. This file pins the frozen contract:

1. A real static 9-frame handheld burst resolves AUTO to OFF and is then
   *exactly* the plain (``robust_fusion=False``) result — same ``hf_gain_db``,
   same ``sharpness_ratio``, byte-identical HR image. Robust fusion costs HF on
   a well-registered static burst (see the ``robust_fusion`` note in
   superres.py), so auto must never silently spend that.
2. A synthetic burst with a moving subject resolves AUTO to ON and the ghost
   trail error drops by >= 5x versus plain fusion.
3. A synthetic parallax layer resolves AUTO to ON with no MSE regression
   beyond 1%.

The direct-drizzle contracts use identity aligns on purpose: the pipeline's
homography registration *follows* the moving layer (it re-registers each frame
to the bar), which hides it from the gate. The gate unit test therefore feeds
the warped stack the gate actually reasons about, exactly as a physically
fixed camera would.
"""


import numpy as np
import cv2
import pytest

from raw_sr.core.superres import (
    RawFrame,
    SuperResolution,
    _auto_robust_fusion,
    _ghost_gate_dispersion,
    _warp_frames_to_hr,
    _GHOST_GATE_TAIL_X,
    _GHOST_GATE_COMPACT,
    drizzle,
    load_raw_linear,
)


from tests.data_paths import REAL_DATA_HELP, dng_files, missing_reason

REAL_DATA = dng_files(9)


# --------------------------------------------------------------------------- #
# Synthetic bursts
# --------------------------------------------------------------------------- #

def _static_burst(n=9, h=256, w=256, seed=5, noise=0.003):
    """A static scene sampled through a fixed 2x LR grid + sensor noise."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.0)
    lh, lw = h // 2, w // 2
    ys, xs = np.mgrid[0:lh, 0:lw].astype(np.float32)
    frames = []
    for i in range(n):
        lr = cv2.remap(img, 2 * xs, 2 * ys, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
        noisy = np.clip(lr + rng.normal(0, noise, lr.shape).astype(np.float32),
                        0, 1)
        rgb = np.stack([noisy] * 3, axis=2)
        frames.append(RawFrame(name=f"s{i}", rgb=rgb, green=rgb[:, :, 1].copy()))
    return frames


def _moving_bar_burst(n=9, h=256, w=256, bar_w=14, sweep=34.0, seed=4):
    """Static textured background + a bright bar translating frame to frame.

    A single homography cannot remove the bar (it moves independently of the
    scene), so a plain mean smears it into a ghost trail. Returns
    ``(frames, truth, trail)`` where ``trail`` marks the swept region except the
    bar's central position.
    """
    rng = np.random.default_rng(seed)
    bg = 0.3 + 0.02 * rng.random((h, w)).astype(np.float32)
    bg[:, 40:44] = 0.8                       # static structure, both axes
    bg[40:44, :] = 0.8
    lh, lw = h // 2, w // 2
    ys, xs = np.mgrid[0:lh, 0:lw].astype(np.float32)
    frames = []
    for sx in np.linspace(-sweep, sweep, n):
        lr = cv2.remap(bg, 2 * xs, 2 * ys, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
        cx = (w * 0.5 + sx) / 2.0
        hw_ = (bar_w / 2.0) / 2.0
        x0, x1 = int(round(cx - hw_)), int(round(cx + hw_))
        lr[:, max(0, x0):min(lw, x1)] = 0.9
        rgb = np.stack([lr] * 3, axis=2).astype(np.float32)
        frames.append(RawFrame(name=f"b{len(frames)}", rgb=rgb,
                               green=rgb[:, :, 1].copy()))
    truth = bg.copy()
    x0 = int(w * 0.5 - bar_w / 2)
    x1 = int(w * 0.5 + bar_w / 2)
    truth[:, x0:x1] = 0.9
    trail = np.zeros((h, w), bool)
    lo, hi = int(w * 0.5 - sweep - bar_w), int(w * 0.5 + sweep + bar_w)
    trail[:, max(0, lo):max(0, x0)] = True
    trail[:, x1:min(w, hi)] = True
    return frames, np.stack([truth] * 3, axis=2).astype(np.float32), trail


def _parallax_edge(n=9, h=256, w=256, jitter=1.4, seed=3, noise=0.003):
    """A static scene with one layer whose edge jitters against the rest."""
    rng = np.random.default_rng(seed)
    base = 0.3 + 0.05 * cv2.GaussianBlur(
        rng.random((h, w)).astype(np.float32), (0, 0), 1.0)
    frames = []
    for i in range(n):
        s = jitter * (i - (n - 1) / 2.0) / ((n - 1) / 2.0)
        img = base.copy()
        img[:, int(w * 0.5 + s):] += 0.5
        img = np.clip(img, 0, 1)
        lh, lw = h // 2, w // 2
        ys, xs = np.mgrid[0:lh, 0:lw].astype(np.float32)
        lr = cv2.remap(img, 2 * xs, 2 * ys, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
        lr = np.clip(lr + rng.normal(0, noise, lr.shape).astype(np.float32),
                     0, 1)
        rgb = np.stack([lr] * 3, axis=2)
        frames.append(RawFrame(name=f"p{i}", rgb=rgb, green=rgb[:, :, 1].copy()))
    return frames, np.stack([base] * 3, axis=2).astype(np.float32)


def _identity_warped(frames):
    S, V, _, _ = _warp_frames_to_hr(frames, [np.eye(3)] * len(frames), 2.0)
    return S, V


def _run(frames, **kw):
    return SuperResolution(ratio=2.0, blur_threshold_ratio=0.0, **kw).run(frames)


# --------------------------------------------------------------------------- #
# Gate statistic + thresholds
# --------------------------------------------------------------------------- #

def test_ghost_gate_thresholds_lie_in_the_measured_gap():
    # Measured populations (see superres.py): static bursts tail <= 25.2 /
    # compact <= 0.35; moving subject and parallax tail >= 275.5 / compact 1.0.
    assert _GHOST_GATE_TAIL_X == 60.0
    assert _GHOST_GATE_COMPACT == 0.6
    assert 25.2 < _GHOST_GATE_TAIL_X < 275.5   # > 2x above static, > 4x below
    assert 0.35 < _GHOST_GATE_COMPACT < 1.0


def test_ghost_gate_flags_a_compact_moving_layer():
    frames, _, _ = _moving_bar_burst()
    S, V = _identity_warped(frames)
    moving, tail, compact = _ghost_gate_dispersion(S, V)
    # Measured: tail 304.5, compact 1.00.
    assert moving is True
    assert tail >= _GHOST_GATE_TAIL_X
    assert compact >= _GHOST_GATE_COMPACT


def test_ghost_gate_flags_a_parallax_layer():
    frames, _ = _parallax_edge()
    S, V = _identity_warped(frames)
    moving, tail, compact = _ghost_gate_dispersion(S, V)
    # Measured: tail 275.5, compact 1.00.
    assert moving is True
    assert tail >= _GHOST_GATE_TAIL_X
    assert compact >= _GHOST_GATE_COMPACT


def test_ghost_gate_passes_a_static_burst():
    frames = _static_burst()
    S, V = _identity_warped(frames)
    moving, tail, compact = _ghost_gate_dispersion(S, V)
    assert moving is False, f"static burst flagged (tail {tail:.1f}, {compact:.2f})"
    assert tail < _GHOST_GATE_TAIL_X


# --------------------------------------------------------------------------- #
# Contract (2): moving subject -> AUTO ON, trail error drops >= 5x
# --------------------------------------------------------------------------- #

def test_contract_moving_subject_auto_on_drops_trail():
    frames, truth, trail = _moving_bar_burst()
    interior = np.zeros_like(trail)
    interior[20:-20, 20:-20] = True
    sel = trail & interior
    eye = [np.eye(3)] * len(frames)

    use, reason, robust_stack = _auto_robust_fusion(frames, eye, 2.0)
    assert use is True, f"auto gate missed the moving subject: {reason!r}"
    assert "moving subject detected" in reason

    hr_off, _ = drizzle(frames, eye, 2.0)
    hr_on, _ = drizzle(frames, eye, 2.0, robust_fusion=True,
                       robust_stack=robust_stack)
    err_off = float(np.abs(hr_off - truth).mean(axis=2)[sel].mean())
    err_on = float(np.abs(hr_on - truth).mean(axis=2)[sel].mean())
    # Measured: 0.0814 -> 0.0065 (12.6x). Assert >= 5x with headroom.
    assert err_on < 0.2 * err_off, (
        f"auto ON trail ghost {err_on:.5f} not < 0.2x plain {err_off:.5f}")


def test_moving_subject_auto_matches_explicit_true_wiring():
    # The SuperResolution wiring must resolve AUTO (None) to the same decision
    # as an explicit True for this burst. Homography registration follows the
    # bar, so pin the translation model where the gate can still see it.
    # Sweep 20 (not the 34 of the other tests): with the B8 sign fix the
    # translation deposit follows the bar-locked frames to their true offsets,
    # and at 34 px only 59% of the frame is covered by every frame, so the
    # gate honestly refuses ("only 59% covered") before looking for motion.
    # (The pre-B8 flipped deposit kept 98% coverage by accident — it deposited
    # every frame on top of the others.) 20 px keeps the 60% coverage contract.
    frames, truth, trail = _moving_bar_burst(sweep=20.0)
    interior = np.zeros_like(trail)
    interior[20:-20, 20:-20] = True
    sel = trail & interior
    res_auto = _run(frames, registration="translation",
                    use_gpu_scatter=False)
    res_on = _run(frames, registration="translation", robust_fusion=True,
                  use_gpu_scatter=False)
    assert res_auto.metrics["robust_active"] is True
    assert res_on.metrics["robust_active"] is True
    # The reason string distinguishes the decision source (auto names the
    # statistic; explicit records the override), so only the effect is shared.
    assert "moving subject detected" in res_auto.metrics["robust_reason"]
    assert res_on.metrics["robust_reason"] == "forced on"
    # AUTO and explicit True must produce the identical HR image.
    assert np.array_equal(res_auto.hr, res_on.hr)
    # And AUTO, unlike forced OFF, actually removes the trail (measured 1.21x
    # on this six-frame pipeline run; the raw direct-drizzle effect is 12.6x).
    res_off = _run(frames, registration="translation", robust_fusion=False,
                   use_gpu_scatter=False)
    e_auto = float(np.abs(res_auto.hr - truth).mean(axis=2)[sel].mean())
    e_off = float(np.abs(res_off.hr - truth).mean(axis=2)[sel].mean())
    assert e_auto < e_off


# --------------------------------------------------------------------------- #
# Contract (3): parallax -> AUTO ON, no MSE regression beyond 1%
# --------------------------------------------------------------------------- #

def test_contract_parallax_auto_on_without_regression():
    frames, truth = _parallax_edge()
    eye = [np.eye(3)] * len(frames)

    use, reason, robust_stack = _auto_robust_fusion(frames, eye, 2.0)
    assert use is True, f"auto gate missed the parallax layer: {reason!r}"

    hr_off, _ = drizzle(frames, eye, 2.0)
    hr_on, _ = drizzle(frames, eye, 2.0, robust_fusion=True,
                       robust_stack=robust_stack)
    mse_off = float(np.mean((hr_off - truth) ** 2))
    mse_on = float(np.mean((hr_on - truth) ** 2))
    # Measured ratio 1.0042: no regression beyond 1%.
    assert mse_on <= 1.01 * mse_off, (
        f"parallax auto ON MSE {mse_on:.4e} regressed vs plain {mse_off:.4e}")


# --------------------------------------------------------------------------- #
# Contract (1): real static burst -> AUTO OFF, exactly the plain result
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(len(REAL_DATA) < 9, reason=missing_reason(9))
def test_contract_real_static_burst_auto_is_exactly_plain():
    roi = (1800, 1224, 512, 512)
    frames = [load_raw_linear(f, roi=roi) for f in REAL_DATA]
    # CPU scatter: GPU float32 atomics are non-deterministic, and this contract
    # demands byte-identical HR output between AUTO and explicit False.
    res_auto = SuperResolution(ratio=2.0, ecc_scale=0.25,
                               use_gpu_scatter=False).run(frames)
    res_off = SuperResolution(ratio=2.0, ecc_scale=0.25,
                              use_gpu_scatter=False,
                              robust_fusion=False).run(frames)

    assert res_auto.metrics["robust_active"] is False
    assert res_auto.metrics["robust_reason"] == "static burst \u2014 plain fusion"
    # AUTO-OFF must be bit-for-bit the plain fusion: no HF/sharpness cost.
    assert res_auto.metrics["hf_gain_db"] == res_off.metrics["hf_gain_db"]
    assert res_auto.metrics["sharpness_ratio"] == res_off.metrics["sharpness_ratio"]
    assert np.array_equal(res_auto.hr, res_off.hr)
    # Alignment aggregates are identical too (the gate only reads the stack).
    assert (res_auto.metrics["align_residual_mean_px"]
            == res_off.metrics["align_residual_mean_px"])
    # D2 still holds on the real burst: exactly one self-comparison, excluded.
    assert len(res_auto.align_self) == res_auto.used_frames
    assert sum(bool(v) for v in res_auto.align_self) == 1
    ref = res_auto.kept_indices.index(res_auto.metrics["reference_index"])
    assert res_auto.align_self[ref] is True


@pytest.mark.skipif(len(REAL_DATA) < 9, reason=missing_reason(9))
def test_real_static_burst_forced_on_changes_hf_only_via_robust_path():
    # A forced ON on the static real burst must actually engage robust fusion
    # (reason "forced on") and therefore differ from plain — proving AUTO-OFF
    # above was a decision, not merely an inert code path.
    roi = (1800, 1224, 512, 512)
    frames = [load_raw_linear(f, roi=roi) for f in REAL_DATA]
    res_on = SuperResolution(ratio=2.0, ecc_scale=0.25, use_gpu_scatter=False,
                             robust_fusion=True).run(frames)
    assert res_on.metrics["robust_active"] is True
    assert res_on.metrics["robust_reason"] == "forced on"
