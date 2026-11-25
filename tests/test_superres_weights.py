"""Fusion weights: sharpness per unit noise variance, not raw Laplacian variance.

The blur gate/reference choice keep using the Laplacian sharpness proxy, but
fusion must weight frames by information content: scene acutance (Laplacian of
a lightly denoised copy) divided by a robust high-pass-MAD noise variance,
with the noise correction clipped to the burst median. Raw Laplacian variance
rises with sensor noise, so a mixed-ISO burst used to over-weight the noisier
frames.
"""

import numpy as np
import cv2
import pytest

from raw_sr.core.superres import (
    FUSION_WEIGHT_CLIP,
    RawFrame,
    SuperResolution,
    _clip_noise_correction,
    _fusion_weight,
    _hf_sharpness,
    _noise_variance,
    _quality_weight,
)


def _frames(n=4, noise=0.0005, noisy=(), noisy_sigma=0.01, seed=7,
            h=96, w=96):
    """Textured synthetic burst; frames in ``noisy`` get extra read noise."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    truth = (0.5 + 0.05 * np.sin(2 * np.pi * xs / 37)
             * np.cos(2 * np.pi * ys / 29)).astype(np.float32)
    truth = np.stack([truth] * 3, axis=2)
    out = []
    for i in range(n):
        dx, dy = 0.35 * (i % 2), 0.25 * ((i // 2) % 2)
        lr = cv2.remap(truth, xs + dx, ys + dy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT_101)
        sigma = noisy_sigma if i in noisy else noise
        lr = np.clip(lr + rng.normal(0, sigma, lr.shape).astype(np.float32),
                     0, 1)
        out.append(RawFrame(name=f"f{i}", rgb=lr, green=lr[:, :, 1].copy()))
    return out


def test_noise_variance_tracks_added_noise():
    clean = _frames(n=1, noise=0.0005)[0].green
    noisy = _frames(n=1, noise=0.01)[0].green
    n_clean, n_noisy = _noise_variance(clean), _noise_variance(noisy)
    # Scene curvature sets a small floor; added noise must dominate it.
    assert np.sqrt(n_noisy) > 2.0 * np.sqrt(n_clean)
    assert np.sqrt(n_noisy) == pytest.approx(0.01, rel=0.3)
    # A flat frame has no high-pass energy at all.
    assert _noise_variance(np.full((64, 64), 0.4, np.float32)) == 0.0


def test_sharpness_is_measured_on_scene_not_noise():
    clean = _frames(n=1, noise=0.0005)[0].green
    noisy = _frames(n=1, noise=0.02)[0].green
    # Light denoising before the Laplacian suppresses much of the noise, so
    # the numerator inflates far less than the raw proxy's 5x+.
    raw_ratio = _quality_weight(noisy) / _quality_weight(clean)
    sharp_ratio = _hf_sharpness(noisy) / _hf_sharpness(clean)
    assert sharp_ratio < 0.25 * raw_ratio
    # The raw Laplacian proxy DOES rise with noise (the old weight's flaw),
    # so the final weight must still prefer the clean frame.
    assert _quality_weight(noisy) > 5.0 * _quality_weight(clean)
    assert _fusion_weight(noisy) < _fusion_weight(clean)
    # A defocused frame has genuinely less acutance and keeps less weight.
    blurred = cv2.GaussianBlur(clean, (0, 0), 2.0)
    assert _hf_sharpness(blurred) < 0.9 * _hf_sharpness(clean)


def test_noise_correction_is_floored_at_the_quiet_quartile():
    # A frame that *measures* quieter than the burst's clean quartile is not
    # rewarded (its estimate is scene-limited, not evidence of less noise).
    nv = np.array([1e-6, 1e-4, 1e-4, 1e-4])
    floor = float(np.percentile(nv, 25))
    c = _clip_noise_correction(nv)
    assert c[0] == pytest.approx(floor)
    assert c[1:].tolist() == nv[1:].tolist()
    # A much noisier frame keeps its full penalty: the ceiling is not capped.
    loud = np.array([1e-4, 1e-4, 1e-4, 1e-1])
    assert _clip_noise_correction(loud)[3] == pytest.approx(1e-1)
    assert np.allclose(_clip_noise_correction(np.array([])), np.array([]))


def test_run_weights_favour_clean_frames_of_a_mixed_noise_burst():
    frames = _frames(n=4, noisy=(1, 3), noisy_sigma=0.012)
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(frames)
    w = np.asarray(res.weights, dtype=np.float64)
    noisy_pos = [res.kept_indices.index(i) for i in (1, 3)
                 if i in res.kept_indices]
    clean_pos = [res.kept_indices.index(i) for i in (0, 2)
                 if i in res.kept_indices]
    assert noisy_pos and clean_pos
    assert w[clean_pos].min() > w[noisy_pos].max()
    # The clip bounds the weight range relative to the burst median.
    assert w.max() / w.min() <= FUSION_WEIGHT_CLIP ** 2 + 1e-9
    assert res.metrics["fusion_weight_spread"] == pytest.approx(
        float(w.max() / w.min()))
    assert len(res.metrics["frame_noise_sigma"]) == len(res.weights)


def test_uniform_noise_burst_keeps_weights_near_uniform():
    frames = _frames(n=4, noise=0.003)
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(frames)
    w = np.asarray(res.weights, dtype=np.float64)
    assert w.max() / w.min() < 1.5
    assert res.metrics["fusion_weight_spread"] < 1.5


def test_weight_clip_bounds_one_pathological_frame():
    frames = _frames(n=4, noise=0.0005, noisy=(1,), noisy_sigma=0.05)
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(frames)
    w = np.asarray(res.weights, dtype=np.float64)
    # Unclipped, the noisy frame's Laplacian is inflated ~1000x; the clip keeps
    # the deposit within a 16x range.
    assert w.max() / w.min() <= FUSION_WEIGHT_CLIP ** 2 + 1e-9
    noisy_pos = [res.kept_indices.index(i) for i in (1,)
                 if i in res.kept_indices]
    if noisy_pos:
        assert w[noisy_pos[0]] < w.max()


def test_run_downweights_a_defocused_frame():
    """A soft frame loses weight even though its own HF looks quiet."""
    frames = _frames(n=3, noise=0.002)
    soft = cv2.GaussianBlur(frames[1].rgb, (0, 0), 2.0)
    frames[1] = RawFrame(name="soft", rgb=soft,
                         green=soft[:, :, 1].copy())
    res = SuperResolution(ratio=2.0, blur_threshold_ratio=0.0).run(frames)
    w = dict(zip(res.kept_indices, np.asarray(res.weights, dtype=np.float64)))
    assert 1 in w and 0 in w
    assert w[1] < min(w[i] for i in (0, 2) if i in w)
